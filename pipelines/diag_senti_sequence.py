# -*- coding: utf-8 -*-
"""GRU 前置验证: 研报情感的【时间序列结构】比【单期快照】多带信息吗?

课题回到项目本意——用 GRU 建模每只股票的研报情感演化轨迹。但直接上 GRU 在
小样本弱信号上几乎必然过拟合(参照最初有监督UMAP: train漂亮/OOS归零)。

GRU 能成立的【必要条件】是: 情感的序列结构(变化/趋势/加速度/波动)携带的信息
> 单期快照。本脚本用手工序列特征检验这个条件:

  senti_now  : 当期(本周)平均情感        —— 单期快照(基线, =已证伪的"水平")
  senti_chg  : 本次 vs 该股上一次研报情感 —— 一阶变化(语气转好/转坏)
  senti_ma3  : 最近3次研报情感均值        —— 平滑水平
  senti_trend: 最近4次研报情感线性斜率    —— 趋势
  senti_accel: 一阶变化的变化            —— 加速度
  senti_surp : 本次情感 - 最近3次均值     —— 情感"意外"

全部只用截至当周可得的研报(无前视), 按该股【有研报的时点序列】构造, 再对齐回
周频截面, 测对未来5日收益(市值中性化残差)的横截面 IC。

判读:
  - 若 senti_chg/trend/surp 的 OOS IC 明显强于 senti_now -> 序列结构有信息,
    GRU 值得训练, 且知道该学"变化/趋势";
  - 若都不强于 senti_now -> 序列结构无增量, GRU 会过拟合, 不必训练。

输入: data/report_title_sentiment.parquet + data/platform/px_daily.pqt
用法: python pipelines/diag_senti_sequence.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
SENT = ROOT / "data" / "report_title_sentiment.parquet"
FWD = 5
SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}
SEQ_FEATS = ["senti_now", "senti_chg", "senti_ma3", "senti_trend",
             "senti_accel", "senti_surp"]


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / n
    return x.mean() / math.sqrt(max(var, 1e-18) / n)


def neutralize_xs(df, ycol, fcols, date_col="date"):
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(date_col):
        sub = g[[ycol] + fcols].dropna()
        if len(sub) < len(fcols) + 5:
            continue
        X = sub[fcols].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X = np.column_stack([np.ones(len(X)), X])
        y = sub[ycol].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[sub.index] = y - X @ beta
    return out


def cs_ic(df, sig, ycol):
    s = df[["date", sig, ycol]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby("date").apply(lambda g: g[sig].corr(g[ycol], method="spearman"),
                                   include_groups=False).dropna()


def build_event_sequence(sent: pd.DataFrame) -> pd.DataFrame:
    """按【每股有研报的事件序列】构造序列特征 (无前视: 只用截至当行的历史)。

    sent: (date, stkcd, sent_avg) —— 每个有研报的(snap交易日,股)一行, 已按日聚合。
    """
    sent = sent.sort_values(["stkcd", "date"]).reset_index(drop=True)
    g = sent.groupby("stkcd", group_keys=False)["sent_avg"]
    sent["senti_now"] = sent["sent_avg"]
    sent["prev1"] = g.shift(1)
    sent["senti_chg"] = sent["sent_avg"] - sent["prev1"]
    sent["senti_ma3"] = g.rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    sent["senti_surp"] = sent["sent_avg"] - sent["senti_ma3"]
    sent["chg_prev"] = g.diff().groupby(sent["stkcd"]).shift(1)
    sent["senti_accel"] = (sent["sent_avg"] - sent["prev1"]) - sent["chg_prev"]

    # 最近4次的线性斜率(对序号回归), 只用历史
    def _trend(s):
        out = np.full(len(s), np.nan)
        v = s.to_numpy(float)
        for i in range(len(v)):
            lo = max(0, i - 3)
            w = v[lo:i + 1]
            w = w[~np.isnan(w)]
            if len(w) >= 3:
                x = np.arange(len(w))
                out[i] = np.polyfit(x, w, 1)[0]
        return pd.Series(out, index=s.index)
    sent["senti_trend"] = g.apply(_trend)
    return sent


def main():
    # 价格 + 未来收益 + 市值
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    gp = px.groupby("stkcd", group_keys=False)
    px["fwd"] = gp["adj_close"].shift(-FWD) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    # 研报情感: 发布日 snap 到交易日, 按(交易日,股)聚合平均情感
    sent = pd.read_parquet(SENT)
    sent["date"] = pd.to_datetime(sent["date"])
    tds = pd.Series(np.sort(px["date"].unique()))
    sent["date"] = sent["date"].map(
        lambda d: tds.iloc[tds.searchsorted(d, "left")] if tds.searchsorted(d, "left") < len(tds) else pd.NaT)
    sent = sent.dropna(subset=["date"]).groupby(["date", "stkcd"], as_index=False).agg(
        sent_sum=("sent_sum", "sum"), n_co=("n_co", "sum"))
    sent["sent_avg"] = sent["sent_sum"] / sent["n_co"]

    # 事件序列特征
    seq = build_event_sequence(sent[["date", "stkcd", "sent_avg"]])

    # 对齐回周频截面: 每股每周取最后一个交易日; 序列特征前向填充到该周
    #   (研报特征在下一篇研报前一直有效, 这是"最近一次研报观点"的自然延续)
    df = px.merge(seq[["date", "stkcd"] + SEQ_FEATS], on=["date", "stkcd"], how="left")
    df = df.sort_values(["stkcd", "date"])
    gg = df.groupby("stkcd", group_keys=False)
    for c in SEQ_FEATS:
        # 前向填充, 但限制 60 交易日(~3个月)内有效, 避免陈旧研报永久carry
        df[c] = gg[c].ffill(limit=60)

    df["week"] = df["date"].dt.to_period("W")
    last = df.groupby(["stkcd", "week"])["date"].transform("max")
    wk = df[df["date"] == last].copy()
    wk["fwd_resid"] = neutralize_xs(wk, "fwd", ["lnmktcap"])

    cov = wk.dropna(subset=["senti_now"])
    print("[seq] 周频面板: 截面=%d  有研报特征(周,股)=%d  周均覆盖股=%.0f"
          % (wk["date"].nunique(), len(cov), cov.groupby("date").size().mean()))

    print("\n研报情感【序列特征】vs【单期快照】 -> 未来5日收益 RankIC (市值中性化, NW lag=4)")
    print("%-12s %12s %8s %12s %8s %12s %8s"
          % ("feature", "train IC", "t", "valid IC", "t", "test IC", "t"))
    print("-" * 78)
    rows = []
    for f in SEQ_FEATS:
        ic_all = cs_ic(wk, f, "fwd_resid")
        seg = {}
        for name, (s, e) in SEG.items():
            m = (ic_all.index >= pd.Timestamp(s)) & (ic_all.index <= pd.Timestamp(e))
            ic = ic_all[m]
            seg[name] = (ic.mean(), nw_t(ic.to_numpy(), 4)) if len(ic) else (np.nan, np.nan)
        wf_t = nw_t(ic_all.to_numpy(), 4)
        tag = "  <-基线" if f == "senti_now" else ""
        print("%-12s %+12.4f %8.2f %+12.4f %8.2f %+12.4f %8.2f%s"
              % (f, seg["train"][0], seg["train"][1], seg["valid"][0], seg["valid"][1],
                 seg["test"][0], seg["test"][1], tag))
        rows.append((f, seg["train"][1], seg["valid"][1], seg["test"][1], wf_t))

    print("\nwalk-forward 汇总 t (全期):")
    for f, _, _, _, wf in rows:
        print("  %-12s wf_t=%.2f" % (f, wf))

    pd.DataFrame(rows, columns=["feature", "t_train", "t_valid", "t_test", "t_wf"]
                 ).to_csv(ROOT / "artifacts" / "diag_senti_sequence.csv", index=False)
    print("\n[seq] wrote artifacts/diag_senti_sequence.csv")
    print("判读: 若 senti_chg/trend/surp 的 valid+test t 明显强于 senti_now(基线)")
    print("      -> 序列结构有信息, 值得训 GRU; 若都≈基线 -> GRU 会过拟合, 不必训。")


if __name__ == "__main__":
    main()
