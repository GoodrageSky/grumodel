# -*- coding: utf-8 -*-
"""多因子合成: 标题情感(弱文本) + 量价因子(强), 周频, 市值中性化, walk-forward 验证。

前序结论:
  - 标题情感是唯一稳健的文本因子 (周频 walk-fwd t=2.24, 多空 Sharpe 0.92);
  - 摘要版扩词典过拟合 train, OOS 更差 (Sharpe 0.26) -> 文本单因子上限 Sharpe≈1;
  - 量价因子才是强 alpha 源 (日频 liq_amt20 t=-4.5, size t=+2.7)。
本脚本把它们合成: 各因子先逐截面 z-score, 等权(可调)合成 composite, 用同一套
市值中性化 + 分段 IC + walk-forward 汇总 + 多空回测验证, 并先看因子间相关性。

候选因子(全部周频、全市场、截面 z-score, 方向统一为"越大越看多"):
  senti     : 标题情感平均 (+)
  rev_1w    : 上周反转 (-上周收益)
  liq       : -log成交额 (低流动性溢价, 取负让"越大越看多")
  size      : -log流通市值 (小市值)
  mom       : 12-1 动量
  lowvol    : -60日波动率

标签: 未来5日收益, 市值中性化残差。
用法: python pipelines/factor_combo.py
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
FWD, ANN = 5, 52


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


def zscore_xs(df, cols, date_col="date"):
    """逐截面 winsorize(1%) + z-score。"""
    g = df.groupby(date_col)
    for c in cols:
        lo = g[c].transform(lambda s: s.quantile(0.01))
        hi = g[c].transform(lambda s: s.quantile(0.99))
        df[c] = df[c].clip(lo, hi)
    g = df.groupby(date_col)
    for c in cols:
        m = g[c].transform("mean"); s = g[c].transform("std")
        df[c + "_z"] = (df[c] - m) / (s + 1e-9)
    return df


def cs_ic(df, sig, ycol):
    s = df[["date", sig, ycol]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby("date").apply(lambda g: g[sig].corr(g[ycol]),
                                   include_groups=False).dropna()


def seg_ic_table(ic_all, label):
    print("\n%-s" % label)
    print("%-8s %12s %9s %7s %7s" % ("段", "IC", "t(NW)", "frac>0", "n"))
    print("-" * 50)
    for name, (s, e) in SEG.items():
        m = (ic_all.index >= pd.Timestamp(s)) & (ic_all.index <= pd.Timestamp(e))
        ic = ic_all[m]
        if len(ic):
            print("%-8s %+12.4f %9.2f %6.0f%% %7d"
                  % (name, ic.mean(), nw_t(ic.to_numpy(), 4), 100 * (ic > 0).mean(), len(ic)))
    print("walk-fwd %+12.4f %9.2f %6.0f%% %7d"
          % (ic_all.mean(), nw_t(ic_all.to_numpy(), 4), 100 * (ic_all > 0).mean(), len(ic_all)))


def long_short(df, sig, ycol="fwd"):
    ts, te = pd.Timestamp(SEG["test"][0]), pd.Timestamp(SEG["test"][1])
    sub = df[(df["date"] >= ts) & (df["date"] <= te)].dropna(subset=[sig, ycol])

    def _ls(g):
        if g[sig].nunique() < 2 or len(g) < 6:
            return np.nan
        q = g[sig].rank(method="first"); n = len(g)
        return g.loc[q > n * 2 / 3, ycol].mean() - g.loc[q <= n / 3, ycol].mean()
    ls = sub.groupby("date").apply(_ls, include_groups=False).dropna()
    sharpe = ls.mean() / (ls.std() + 1e-12) * math.sqrt(ANN)
    return ls.mean(), 100 * (ls > 0).mean(), sharpe, len(ls)


def build_panel():
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "amount",
                                            "turnover", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    px["ret1"] = g["adj_close"].pct_change(fill_method=None)
    px["fwd"] = g["adj_close"].shift(-FWD) / px["adj_close"] - 1.0
    px["c_5"] = g["adj_close"].shift(5)
    px["c_21"] = g["adj_close"].shift(21)
    px["c_252"] = g["adj_close"].shift(252)
    px["mom"] = px["c_21"] / px["c_252"] - 1.0
    px["rev_1w"] = -(px["adj_close"] / px["c_5"] - 1.0)
    px["vol60"] = g["ret1"].rolling(60, min_periods=30).std().reset_index(level=0, drop=True)
    px["lowvol"] = -px["vol60"]
    px["liq"] = -np.log(g["amount"].rolling(20, min_periods=10).mean()
                        .reset_index(level=0, drop=True) + 1.0)
    px["size"] = -np.log(px["float_mktcap"].astype(float) + 1.0)
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    sent = pd.read_parquet(SENT)
    sent["date"] = pd.to_datetime(sent["date"])
    tds = pd.Series(np.sort(px["date"].unique()))
    sent["date"] = sent["date"].map(
        lambda d: tds.iloc[tds.searchsorted(d, "left")] if tds.searchsorted(d, "left") < len(tds) else pd.NaT)
    sent = sent.dropna(subset=["date"]).groupby(["date", "stkcd"], as_index=False).agg(
        sent_sum=("sent_sum", "sum"), n_co=("n_co", "sum"))
    px = px.merge(sent, on=["date", "stkcd"], how="left")
    px["n_co"] = px["n_co"].fillna(0.0)
    px["senti"] = np.where(px["n_co"] > 0, px["sent_sum"] / px["n_co"], 0.0)  # 无研报=中性0

    px["week"] = px["date"].dt.to_period("W")
    last = px.groupby(["stkcd", "week"])["date"].transform("max")
    wk = px[px["date"] == last].copy()
    wk["fwd_resid"] = neutralize_xs(wk, "fwd", ["lnmktcap"])
    return wk


FACTORS = ["senti", "rev_1w", "liq", "size", "mom", "lowvol"]


def main():
    wk = build_panel()
    wk = zscore_xs(wk, FACTORS)
    zcols = [c + "_z" for c in FACTORS]
    print("[combo] 周频面板: 截面=%d  行=%d  周均股票=%.0f"
          % (wk["date"].nunique(), len(wk.dropna(subset=zcols, how="all")),
             wk.dropna(subset=["fwd_resid"]).groupby("date").size().mean()))

    # ---- 因子间相关性 (z-score 后, 全样本) ----
    print("\n【因子相关性矩阵 (截面z后)】")
    corr = wk[zcols].corr()
    corr.columns = FACTORS; corr.index = FACTORS
    print(corr.round(2).to_string())

    # ---- 单因子 OOS 速览 ----
    print("\n【单因子 test段 IC(t) / 多空Sharpe】")
    print("%-8s %14s %10s" % ("factor", "test IC(t)", "test Sharpe"))
    print("-" * 40)
    single = {}
    for c in FACTORS:
        ic = cs_ic(wk, c + "_z", "fwd_resid")
        ts, te = pd.Timestamp(SEG["test"][0]), pd.Timestamp(SEG["test"][1])
        ic_t = ic[(ic.index >= ts) & (ic.index <= te)]
        _, _, sh, _ = long_short(wk, c + "_z")
        single[c] = (ic_t.mean(), nw_t(ic_t.to_numpy(), 4), sh)
        print("%-8s %+.4f(%5.2f)   %6.2f" % (c, ic_t.mean(), nw_t(ic_t.to_numpy(), 4), sh))

    # ---- 合成: 等权 z-score 平均 ----
    wk["combo"] = wk[zcols].mean(axis=1)
    # 仅量价(不含文本)对照
    qcols = [c + "_z" for c in FACTORS if c != "senti"]
    wk["combo_qonly"] = wk[qcols].mean(axis=1)

    ic_combo = cs_ic(wk, "combo", "fwd_resid")
    seg_ic_table(ic_combo, "【合成因子 combo (等权6因子) 分段 IC】")
    m, wr, sh, n = long_short(wk, "combo")
    print("多空回测(test): L/S周均=%+.4f 胜率=%.0f%% 年化Sharpe=%.2f 期数=%d" % (m, wr, sh, n))

    ic_q = cs_ic(wk, "combo_qonly", "fwd_resid")
    seg_ic_table(ic_q, "【对照: 仅量价合成 combo_qonly (5因子,无文本)】")
    m2, wr2, sh2, n2 = long_short(wk, "combo_qonly")
    print("多空回测(test): L/S周均=%+.4f 胜率=%.0f%% 年化Sharpe=%.2f 期数=%d" % (m2, wr2, sh2, n2))

    print("\n【结论速读】")
    print("  文本单因子 senti test Sharpe=%.2f" % single["senti"][2])
    print("  纯量价合成 Sharpe=%.2f  ->  +文本合成 Sharpe=%.2f  (增量=%.2f)"
          % (sh2, sh, sh - sh2))

    wk[["date", "stkcd", "combo", "combo_qonly", "fwd", "fwd_resid"]].dropna(
        subset=["combo"]).to_csv(ROOT / "artifacts" / "factor_combo.csv", index=False)
    print("\n[combo] wrote artifacts/factor_combo.csv")


if __name__ == "__main__":
    main()
