# -*- coding: utf-8 -*-
"""选项3：研报【标题情感方向】因子 —— 测"研报观点"而非"研报覆盖度"。

前序结论:
  - "研报到达数量"市值中性化后归零 (diag_neutralize.py) -> 数量是规模代理;
  - report_info 无评级/目标价字段, 但标题含强方向措辞, classification 能筛公司研究。

本脚本两个改进:
  1. 只取 classification 含"公司研究"的研报 -> 剔除宏观/商品/策略噪声;
  2. 从标题抽【情感方向】= (利好词命中 - 利空词命中), 按(交易日,股)聚合,
     得到"研报观点"因子, 而非单纯计数。

然后测该因子在【市值中性化残差收益】上对未来 1/5/20 日的横截面 IC(NW校正),
直接对比之前"到达数量"(中性化后≈0)是否有质的提升。

输入: RDS reportdata.report_info + data/platform/px_daily.pqt
      复用 load_real_docs 的 name->stkcd 映射
用法: python pipelines/diag_title_sentiment.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import load_real_docs as L  # noqa: E402

PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
CACHE = ROOT / "data" / "report_title_sentiment.parquet"
START, END = "2019-12-01", "2025-08-01"
HORIZONS = [1, 5, 20]
SEG_TEST = ("2024-04-01", "2025-08-01")

# 标题情感词典 (A股研报常见措辞)
POS = ["修复", "增长", "高增", "强劲", "复苏", "改善", "提升", "超预期", "回暖",
       "向好", "扩张", "景气", "放量", "兑现", "突破", "新高", "稳健", "韧性",
       "受益", "加速", "领先", "优异", "亮眼", "回升", "拐点", "反转", "趋优",
       "增厚", "扩产", "放量", "订单饱满", "量价齐升", "持续增长"]
NEG = ["回落", "承压", "下滑", "放缓", "压力", "低于预期", "不及预期", "下降",
       "亏损", "下修", "疲软", "拖累", "走弱", "萎缩", "减速", "风险", "下行",
       "去库", "降价", "退坡", "承压下行", "需求不足", "盈利下滑", "业绩下滑"]


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / n
    return x.mean() / math.sqrt(max(var, 1e-18) / n)


def _score_title(t: str) -> int:
    if not t:
        return 0
    p = sum(1 for w in POS if w in t)
    n = sum(1 for w in NEG if w in t)
    return p - n


def pull_sentiment() -> pd.DataFrame:
    if CACHE.exists():
        print("[title] using cache: %s" % CACHE)
        return pd.read_parquet(CACHE)
    print("[title] building name->stkcd map ...")
    nm = L.build_name_to_stkcd()
    conn = L._connect("reportdata")
    edges = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="3MS").tolist()
    edges = sorted(set([pd.Timestamp(START)] + edges + [pd.Timestamp(END) + pd.Timedelta(days=1)]))
    rows = []
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            a_s, b_s = a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat()
            print("[title] pulling %s ~ %s" % (a_s, b_s))
            sql = ("SELECT infopubldate, infotitle, classification, stocks "
                   "FROM report_info WHERE infopubldate BETWEEN %s AND %s "
                   "  AND stocks IS NOT NULL AND stocks <> '' AND stocks <> 'None' "
                   "  AND classification LIKE '%%公司研究%%'")
            df = pd.read_sql(sql, conn, params=(a_s, b_s))
            if df.empty:
                continue
            df["sent"] = df["infotitle"].fillna("").map(_score_title)
            df["names"] = df["stocks"].map(L._parse_stocks_field)
            df = df[df["names"].map(len) > 0].explode("names", ignore_index=True)
            df["stkcd"] = df["names"].map(nm.name2code)
            df = df.dropna(subset=["stkcd"])
            df["date"] = pd.to_datetime(df["infopubldate"]).dt.normalize()
            rows.append(df[["date", "stkcd", "sent"]])
    finally:
        conn.close()
    raw = pd.concat(rows, ignore_index=True)
    # 按(发布日,股)聚合: 净情感和 + 研报数
    agg = raw.groupby(["date", "stkcd"]).agg(
        sent_sum=("sent", "sum"), n_co=("sent", "size")).reset_index()
    agg.to_parquet(CACHE, index=False)
    print("[title] cached %d rows -> %s" % (len(agg), CACHE))
    return agg


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


def main():
    sent = pull_sentiment()
    sent["date"] = pd.to_datetime(sent["date"])
    print("[title] sentiment rows=%d  date %s~%s  净情感>0占比=%.1f%%"
          % (len(sent), sent["date"].min().date(), sent["date"].max().date(),
             100 * (sent["sent_sum"] > 0).mean()))

    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    for h in HORIZONS:
        px["fwd_%d" % h] = g["adj_close"].shift(-h) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    # 发布日 snap 到当日或之后第一个交易日
    tds = pd.Series(np.sort(px["date"].unique()))
    def _snap(d):
        i = tds.searchsorted(d, side="left")
        return tds.iloc[i] if i < len(tds) else pd.NaT
    sent = sent.copy()
    sent["date"] = sent["date"].map(_snap)
    sent = sent.dropna(subset=["date"]).groupby(["date", "stkcd"], as_index=False).agg(
        sent_sum=("sent_sum", "sum"), n_co=("n_co", "sum"))

    df = px.merge(sent, on=["date", "stkcd"], how="left")
    df["sent_sum"] = df["sent_sum"].fillna(0.0)
    df["n_co"] = df["n_co"].fillna(0.0)
    # 情感方向(只在有研报的股上有意义), 同时给两种口径:
    #   sent_net : 净情感和 (无研报=0)
    #   sent_avg : 有研报时的平均情感 (=sent_sum/n_co), 无研报=NaN(不参与)
    df["sent_net"] = df["sent_sum"]
    df["sent_avg"] = np.where(df["n_co"] > 0, df["sent_sum"] / df["n_co"], np.nan)

    ts, te = pd.Timestamp(SEG_TEST[0]), pd.Timestamp(SEG_TEST[1])
    df = df[(df["date"] >= ts) & (df["date"] <= te)].copy()
    print("[title] test段 日均有公司研报股票数=%.1f"
          % df.groupby("date")["n_co"].apply(lambda s: (s > 0).sum()).mean())

    print("\n研报标题情感方向 -> 未来 N 日收益 IC (test段, 市值中性化, NW校正)")
    print("%-10s" % "signal" + "".join("   fwd_%-2d (t)  " % h for h in HORIZONS))
    print("-" * 70)
    rows = []
    for sig in ["sent_net", "sent_avg"]:
        cells = []
        for h in HORIZONS:
            yc = "fwd_%d" % h
            df["_resid"] = neutralize_xs(df, yc, ["lnmktcap"])
            ic = cs_ic(df, sig, "_resid")
            tv = nw_t(ic.to_numpy(), max(h - 1, 0))
            cells.append("%+.4f(%5.1f)" % (ic.mean(), tv))
            rows.append((sig, h, ic.mean(), tv, len(ic)))
        print("%-10s" % sig + "".join("  %s" % c for c in cells))

    pd.DataFrame(rows, columns=["signal", "horizon", "IC_mean", "t_nw", "n"]
                 ).to_csv(ROOT / "artifacts" / "diag_title_sentiment.csv", index=False)
    print("\n[title] wrote artifacts/diag_title_sentiment.csv")
    print("判读: 若情感方向因子 |t|>2 且优于纯到达数量(中性化后≈0) -> 研报观点有 alpha,")
    print("      值得做正经 NLP 方向抽取/重嵌入; 若仍≈0 -> 标题情感太弱, 需读正文评级。")


if __name__ == "__main__":
    main()
