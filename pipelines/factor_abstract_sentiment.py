# -*- coding: utf-8 -*-
"""扩展信号源: 从【摘要 abstract】抽研报情感方向, 对比标题版。

前序: 标题情感(factor_title_sentiment.py)方向对但弱 —— train/test t≈1.2-1.8,
      valid 塌陷, 多空 Sharpe 0.92。瓶颈是标题太薄。本脚本从摘要抽信号:
  - 摘要信息密度高, 含明确评级措辞("给予买入评级"/"首次覆盖")、盈利数据、超预期判断;
  - 词典扩展 + 新增【显式评级】高权重信号(买入/增持/推荐, 计 +2)。

因子 abs_sent_avg = 有研报时每篇摘要情感分的平均, 同一套验证:
  1. 分段 IC (train/valid/test, NW校正 t, frac_pos)
  2. walk-forward 汇总 t (全期周度截面合一)
  3. 多空回测 (test段, 周度, 年化 Sharpe)
均做市值中性化, 与标题版逐项对比。

输入: RDS reportdata.report_info + data/platform/px_daily.pqt
缓存: data/report_abstract_sentiment.parquet
用法: python pipelines/factor_abstract_sentiment.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import load_real_docs as L  # noqa: E402

PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
CACHE = ROOT / "data" / "report_abstract_sentiment.parquet"
START, END = "2019-12-01", "2025-08-01"
FWD, ANN = 5, 52
SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}

# 显式评级措辞 (研报最直接的观点, 高权重 +2 / -2)
RATING_POS = ["给予买入", "买入评级", "增持评级", "推荐评级", "首次覆盖", "维持买入",
              "维持增持", "上调评级", "上调至", "强烈推荐", "跑赢行业", "优于大市"]
RATING_NEG = ["下调评级", "下调至", "中性评级", "减持评级", "回避", "跑输行业",
              "弱于大市", "维持中性"]
# 一般情感措辞 (+1 / -1)
POS = ["超预期", "高增", "高速增长", "快速增长", "稳健增长", "强劲", "复苏", "改善",
       "修复", "提升", "向好", "扩张", "景气", "放量", "兑现", "突破", "新高",
       "韧性", "受益", "加速", "领先", "亮眼", "回升", "拐点", "反转", "增厚",
       "扭亏", "弹性", "高景气", "量价齐升", "订单充足", "盈利能力增强", "毛利率提升"]
NEG = ["低于预期", "不及预期", "下滑", "放缓", "承压", "压力", "下降", "亏损",
       "下修", "疲软", "拖累", "走弱", "萎缩", "减速", "下行", "降价", "退坡",
       "需求不足", "业绩下滑", "盈利下滑", "毛利率下降", "增速放缓", "拖累业绩"]


def _score(text: str) -> int:
    if not text:
        return 0
    s = 0
    for w in RATING_POS:
        if w in text:
            s += 2
    for w in RATING_NEG:
        if w in text:
            s -= 2
    for w in POS:
        if w in text:
            s += 1
    for w in NEG:
        if w in text:
            s -= 1
    return s


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
    return s.groupby("date").apply(lambda g: g[sig].corr(g[ycol]),
                                   include_groups=False).dropna()


def pull_abstract() -> pd.DataFrame:
    if CACHE.exists():
        print("[abs] using cache: %s" % CACHE)
        return pd.read_parquet(CACHE)
    print("[abs] building name->stkcd map ...")
    nm = L.build_name_to_stkcd()
    conn = L._connect("reportdata")
    edges = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="3MS").tolist()
    edges = sorted(set([pd.Timestamp(START)] + edges + [pd.Timestamp(END) + pd.Timedelta(days=1)]))
    rows = []
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            a_s, b_s = a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat()
            print("[abs] pulling %s ~ %s" % (a_s, b_s))
            sql = ("SELECT infopubldate, abstract, stocks FROM report_info "
                   "WHERE infopubldate BETWEEN %s AND %s "
                   "  AND stocks IS NOT NULL AND stocks <> '' AND stocks <> 'None' "
                   "  AND classification LIKE '%%公司研究%%' "
                   "  AND abstract IS NOT NULL AND abstract <> ''")
            df = pd.read_sql(sql, conn, params=(a_s, b_s))
            if df.empty:
                continue
            df["sent"] = df["abstract"].astype(str).str.slice(0, 1500).map(_score)
            df["names"] = df["stocks"].map(L._parse_stocks_field)
            df = df[df["names"].map(len) > 0].explode("names", ignore_index=True)
            df["stkcd"] = df["names"].map(nm.name2code)
            df = df.dropna(subset=["stkcd"])
            df["date"] = pd.to_datetime(df["infopubldate"]).dt.normalize()
            rows.append(df[["date", "stkcd", "sent"]])
    finally:
        conn.close()
    raw = pd.concat(rows, ignore_index=True)
    agg = raw.groupby(["date", "stkcd"]).agg(
        sent_sum=("sent", "sum"), n_co=("sent", "size")).reset_index()
    agg.to_parquet(CACHE, index=False)
    print("[abs] cached %d rows -> %s" % (len(agg), CACHE))
    return agg


def main():
    sent = pull_abstract()
    sent["date"] = pd.to_datetime(sent["date"])
    print("[abs] rows=%d  净情感>0占比=%.1f%%  情感分布: %s"
          % (len(sent), 100 * (sent["sent_sum"] > 0).mean(),
             sent["sent_sum"].describe()[["mean", "std", "min", "max"]].round(2).to_dict()))

    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    px["fwd"] = g["adj_close"].shift(-FWD) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    tds = pd.Series(np.sort(px["date"].unique()))
    sent["date"] = sent["date"].map(
        lambda d: tds.iloc[tds.searchsorted(d, "left")] if tds.searchsorted(d, "left") < len(tds) else pd.NaT)
    sent = sent.dropna(subset=["date"]).groupby(["date", "stkcd"], as_index=False).agg(
        sent_sum=("sent_sum", "sum"), n_co=("n_co", "sum"))

    df = px.merge(sent, on=["date", "stkcd"], how="left")
    df["n_co"] = df["n_co"].fillna(0.0)
    df["abs_sent_avg"] = np.where(df["n_co"] > 0, df["sent_sum"] / df["n_co"], np.nan)

    df["week"] = df["date"].dt.to_period("W")
    last = df.groupby(["stkcd", "week"])["date"].transform("max")
    wk = df[df["date"] == last].copy()
    wk["fwd_resid"] = neutralize_xs(wk, "fwd", ["lnmktcap"])

    has = wk.dropna(subset=["abs_sent_avg"])
    print("[abs] 周频: 总截面=%d  有因子(周,股)=%d  周均覆盖股=%.0f"
          % (wk["date"].nunique(), len(has), has.groupby("date").size().mean()))

    ic_all = cs_ic(wk, "abs_sent_avg", "fwd_resid")
    print("\n【1】分段 IC (因子=abs_sent_avg, 标签=未来5日市值中性化残差, NW lag=4)")
    print("%-8s %14s %10s %8s %8s" % ("段", "IC_mean", "t(NW)", "frac>0", "截面数"))
    print("-" * 56)
    for name, (s, e) in SEG.items():
        m = (ic_all.index >= pd.Timestamp(s)) & (ic_all.index <= pd.Timestamp(e))
        ic = ic_all[m]
        if len(ic):
            print("%-8s %+14.4f %10.2f %8.0f%% %8d"
                  % (name, ic.mean(), nw_t(ic.to_numpy(), 4), 100 * (ic > 0).mean(), len(ic)))

    print("\n【2】walk-forward 汇总")
    print("  IC=%+.4f  汇总 t(NW)=%.2f  frac>0=%.0f%%  n=%d"
          % (ic_all.mean(), nw_t(ic_all.to_numpy(), 4), 100 * (ic_all > 0).mean(), len(ic_all)))

    ts, te = pd.Timestamp(SEG["test"][0]), pd.Timestamp(SEG["test"][1])
    sub = has[(has["date"] >= ts) & (has["date"] <= te)].copy()

    def _ls(gg):
        if gg["abs_sent_avg"].nunique() < 2 or len(gg) < 6:
            return np.nan
        q = gg["abs_sent_avg"].rank(method="first")
        n = len(gg)
        return gg.loc[q > n * 2 / 3, "fwd"].mean() - gg.loc[q <= n / 3, "fwd"].mean()
    ls = sub.groupby("date").apply(_ls, include_groups=False).dropna()
    sharpe = ls.mean() / (ls.std() + 1e-12) * math.sqrt(ANN)
    print("\n【3】多空回测 (test段, 3组 top-bottom, 周度)")
    print("  L/S 周均=%+.4f  胜率=%.0f%%  年化Sharpe(52)=%.2f  期数=%d"
          % (ls.mean(), 100 * (ls > 0).mean(), sharpe, len(ls)))

    print("\n--- 对比标题版(参考): test t=1.23, walk-fwd t=2.24, Sharpe=0.92 ---")
    has[["date", "stkcd", "abs_sent_avg", "fwd", "fwd_resid"]].to_csv(
        ROOT / "artifacts" / "factor_abstract_sentiment.csv", index=False)
    print("[abs] wrote artifacts/factor_abstract_sentiment.csv")


if __name__ == "__main__":
    main()
