# -*- coding: utf-8 -*-
"""把研报标题情感方向做成正式因子并全面验证。

承接 diag_title_sentiment.py 的突破: sent_avg 在 test段市值中性化残差上
fwd_5 t=4.0。本脚本确认它是不是稳健可用因子(而非 test 段偶然):

  1. 全样本 IC: train / valid / test 三段分别给 IC + NW校正 t + frac_pos
  2. walk-forward: 把全期所有周度截面 IC 拼成一个分布, 给单一汇总 t (最诚实)
  3. 多空回测: 按情感分位 top-bottom, 周度持仓, 年化夏普

口径(已定): 周频截面(周五), 标签=未来5日收益(market: vwap口径用收盘近似),
            因子=sent_avg(有研报时标题平均情感), 市值中性化。

输入: data/report_title_sentiment.parquet + data/platform/px_daily.pqt
用法: python pipelines/factor_title_sentiment.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
SENT = ROOT / "data" / "report_title_sentiment.parquet"

SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}
FWD = 5            # 未来5日 (情感因子最优持仓, 见衰减曲线)
ANN = 52           # 周频年化


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


def build_weekly():
    """构造周频面板: 因子 sent_avg + 未来5日收益 + 市值, 周五截面。"""
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    px["fwd"] = g["adj_close"].shift(-FWD) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    sent = pd.read_parquet(SENT)
    sent["date"] = pd.to_datetime(sent["date"])
    # 发布日 snap 到当日或之后首个交易日
    tds = pd.Series(np.sort(px["date"].unique()))
    sent["date"] = sent["date"].map(
        lambda d: tds.iloc[tds.searchsorted(d, "left")] if tds.searchsorted(d, "left") < len(tds) else pd.NaT)
    sent = sent.dropna(subset=["date"]).groupby(["date", "stkcd"], as_index=False).agg(
        sent_sum=("sent_sum", "sum"), n_co=("n_co", "sum"))

    df = px.merge(sent, on=["date", "stkcd"], how="left")
    df["n_co"] = df["n_co"].fillna(0.0)
    df["sent_avg"] = np.where(df["n_co"] > 0, df["sent_sum"] / df["n_co"], np.nan)

    # 周频采样: 每股每周最后一个交易日
    df["week"] = df["date"].dt.to_period("W")
    last = df.groupby(["stkcd", "week"])["date"].transform("max")
    wk = df[df["date"] == last].copy()
    # 市值中性化残差收益
    wk["fwd_resid"] = neutralize_xs(wk, "fwd", ["lnmktcap"])
    return wk


def main():
    wk = build_weekly()
    has = wk.dropna(subset=["sent_avg"])
    print("[factor] 周频面板: 总截面=%d  有情感因子的(周,股)行=%d  周均覆盖股=%.0f"
          % (wk["date"].nunique(), len(has),
             has.groupby("date").size().mean()))

    # ---- 1. 全样本分段 IC ----
    print("\n【1】分段 IC (因子=sent_avg, 标签=未来5日 市值中性化残差收益, NW lag=4)")
    print("%-8s %14s %10s %8s %8s" % ("段", "IC_mean", "t(NW)", "frac>0", "截面数"))
    print("-" * 56)
    ic_all = cs_ic(wk, "sent_avg", "fwd_resid")
    for name, (s, e) in SEG.items():
        m = (ic_all.index >= pd.Timestamp(s)) & (ic_all.index <= pd.Timestamp(e))
        ic = ic_all[m]
        if len(ic) == 0:
            continue
        print("%-8s %+14.4f %10.2f %8.0f%% %8d"
              % (name, ic.mean(), nw_t(ic.to_numpy(), 4),
                 100 * (ic > 0).mean(), len(ic)))

    # ---- 2. walk-forward 汇总 (全期所有截面 IC 合一) ----
    print("\n【2】walk-forward 汇总 (2020-01 ~ 2025-08 全部周度截面)")
    print("  IC_mean=%+.4f  IC_std=%.4f  汇总 t(NW)=%.2f  frac>0=%.0f%%  n=%d"
          % (ic_all.mean(), ic_all.std(), nw_t(ic_all.to_numpy(), 4),
             100 * (ic_all > 0).mean(), len(ic_all)))

    # ---- 3. 多空回测 (test段) ----
    print("\n【3】多空回测 (test段, 情感分3组 top-bottom, 周度持仓)")
    ts, te = pd.Timestamp(SEG["test"][0]), pd.Timestamp(SEG["test"][1])
    sub = has[(has["date"] >= ts) & (has["date"] <= te)].copy()

    def _ls(g):
        if g["sent_avg"].nunique() < 2 or len(g) < 6:
            return np.nan
        q = g["sent_avg"].rank(method="first")
        n = len(g)
        top = g.loc[q > n * 2 / 3, "fwd"].mean()
        bot = g.loc[q <= n / 3, "fwd"].mean()
        return top - bot
    ls = sub.groupby("date").apply(_ls, include_groups=False).dropna()
    # 周频收益 -> 年化夏普 (未来5日≈1周, 持仓不重叠近似)
    sharpe = ls.mean() / (ls.std() + 1e-12) * math.sqrt(ANN)
    print("  L/S 周均=%+.4f  胜率=%.0f%%  年化Sharpe(52)=%.2f  期数=%d"
          % (ls.mean(), 100 * (ls > 0).mean(), sharpe, len(ls)))

    out = wk[["date", "stkcd", "sent_avg", "fwd", "fwd_resid"]].dropna(subset=["sent_avg"])
    out.to_csv(ROOT / "artifacts" / "factor_title_sentiment.csv", index=False)
    print("\n[factor] wrote artifacts/factor_title_sentiment.csv")
    print("判读: train/valid/test 三段 t 同号且显著 + walk-forward 汇总 t>3 + 多空夏普>1")
    print("      -> 信号稳健, 可投产; 任一段崩 -> test 偶然, 需谨慎。")


if __name__ == "__main__":
    main()
