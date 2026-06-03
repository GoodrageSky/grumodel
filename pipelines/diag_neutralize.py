# -*- coding: utf-8 -*-
"""中性化残差检验：剔除市值(规模)混杂后, 研报信号还剩什么?

承接 diag_report_event.py 的发现: "研报到达"对未来收益是【负向、长周期】的,
疑似只是"研报覆盖度 = 大盘/拥挤股"的风格代理。本脚本把未来收益对【市值】做
横截面回归取残差, 再测两类研报信号在残差收益上的 IC:

  (A) 研报到达 has_report / log_arr   —— 日频, 全市场 (现有 report_arrivals.parquet)
  (B) 现有月度文本嵌入 emb_0..9       —— 月频, 372股 (现有 panel.parquet)

判读:
  - 若中性化后到达事件的负向【消失】, 且嵌入在残差上冒出【正】IC
    -> 文本内容有被规模掩盖的真信号, 重嵌入值得;
  - 若中性化后到达仍负 / 嵌入仍 ≈0
    -> 研报信号主要就是规模代理, 文本路线需转向(当负向因子用 / 换数据)。

注: 此处只做【市值】中性化(平台 px 有 float_mktcap, 无行业)。行业中性化需
    另拉行业分类, 若市值中性化已给出明确结论则不必。

用法: python pipelines/diag_neutralize.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
ARRIVALS = ROOT / "data" / "report_arrivals.parquet"
PANEL = ROOT / "artifacts" / "panel.parquet"

SEG_TEST = ("2024-04-01", "2025-08-01")
HORIZONS = [1, 5, 20]


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / n
    return x.mean() / math.sqrt(max(var, 1e-18) / n)


def neutralize_xs(df, ycol, factor_cols, date_col="date"):
    """逐截面把 ycol 对 factor_cols 做 OLS, 返回残差。factor 先 z-score。"""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(date_col):
        sub = g[[ycol] + factor_cols].dropna()
        if len(sub) < len(factor_cols) + 5:
            continue
        X = sub[factor_cols].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X = np.column_stack([np.ones(len(X)), X])
        y = sub[ycol].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[sub.index] = y - X @ beta
    return out


def cs_ic_series(df, sig, ycol, date_col="date"):
    s = df[[date_col, sig, ycol]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby(date_col).apply(
        lambda g: g[sig].corr(g[ycol]), include_groups=False).dropna()


# ----------------- (A) 到达事件: 日频全市场 ----------------- #
def part_a():
    print("=" * 78)
    print("(A) 研报到达事件 —— 市值中性化前后对比 (test段, 日频全市场, NW校正)")
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    for h in HORIZONS:
        px["fwd_%d" % h] = g["adj_close"].shift(-h) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    arr = pd.read_parquet(ARRIVALS)
    arr["date"] = pd.to_datetime(arr["date"])
    df = px.merge(arr, on=["date", "stkcd"], how="left")
    df["arrival"] = df["arrival"].fillna(0.0)
    df["has_report"] = (df["arrival"] > 0).astype(float)
    df["log_arr"] = np.log1p(df["arrival"])

    ts, te = pd.Timestamp(SEG_TEST[0]), pd.Timestamp(SEG_TEST[1])
    df = df[(df["date"] >= ts) & (df["date"] <= te)].copy()

    print("%-12s %-8s %16s %16s" % ("signal", "label", "raw IC(t)", "市值中性化 IC(t)"))
    print("-" * 78)
    rows = []
    for sig in ["has_report", "log_arr"]:
        for h in HORIZONS:
            yc = "fwd_%d" % h
            ic_raw = cs_ic_series(df, sig, yc)
            t_raw = nw_t(ic_raw.to_numpy(), max(h - 1, 0))
            # 中性化: 残差收益
            df["_resid"] = neutralize_xs(df, yc, ["lnmktcap"])
            ic_neu = cs_ic_series(df, sig, "_resid")
            t_neu = nw_t(ic_neu.to_numpy(), max(h - 1, 0))
            print("%-12s %-8s %+.4f(%5.1f)   %+.4f(%5.1f)"
                  % (sig, yc, ic_raw.mean(), t_raw, ic_neu.mean(), t_neu))
            rows.append(("A", sig, h, ic_raw.mean(), t_raw, ic_neu.mean(), t_neu))
    return rows


# ----------------- (B) 现有月度嵌入: 月频 372股 ----------------- #
def part_b():
    print("\n" + "=" * 78)
    print("(B) 现有月度文本嵌入 emb_0..9 —— 市值中性化前后 (test段, 月频372股)")
    panel = pd.read_parquet(PANEL)
    if not isinstance(panel.index, pd.MultiIndex):
        panel = panel.set_index(["date", "stkcd"])
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])

    # 取市值: 月度面板日对齐平台 float_mktcap
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)
    df = panel.merge(px[["date", "stkcd", "lnmktcap"]], on=["date", "stkcd"], how="left")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    ts, te = pd.Timestamp(SEG_TEST[0]), pd.Timestamp(SEG_TEST[1])
    df = df[(df["date"] >= ts) & (df["date"] <= te)].copy()

    # 标签: vwap_rk(已是横截面分位收益代理)。中性化对象=vwap_rk 对 lnmktcap 取残差
    df["rk_resid"] = neutralize_xs(df, "vwap_rk", ["lnmktcap"])

    print("%-8s %16s %16s" % ("emb", "raw IC(t)", "市值中性化 IC(t)"))
    print("-" * 60)
    rows = []
    # 单嵌入维度的 IC + 多嵌入合成(简单等权 z-score 平均)看整体
    for c in emb_cols:
        ic_raw = cs_ic_series(df, c, "vwap_rk")
        ic_neu = cs_ic_series(df, c, "rk_resid")
        t_raw = nw_t(ic_raw.to_numpy(), 0)
        t_neu = nw_t(ic_neu.to_numpy(), 0)
        print("%-8s %+.4f(%5.1f)   %+.4f(%5.1f)"
              % (c, ic_raw.mean(), t_raw, ic_neu.mean(), t_neu))
        rows.append(("B", c, 0, ic_raw.mean(), t_raw, ic_neu.mean(), t_neu))
    # |IC| 最大维度的 t 作为整体"有没有信号"的快读
    best = max(rows, key=lambda r: abs(r[5]))
    print("-> 中性化后 |IC| 最大维度: %s  IC=%.4f t=%.1f" % (best[1], best[5], best[6]))
    return rows


def main():
    rows = part_a() + part_b()
    pd.DataFrame(rows, columns=["part", "signal", "horizon",
                                "IC_raw", "t_raw", "IC_neu", "t_neu"]
                 ).to_csv(ROOT / "artifacts" / "diag_neutralize.csv", index=False)
    print("\n[neutralize] wrote artifacts/diag_neutralize.csv")
    print("判读: (A)若中性化后到达负向 t 大幅缩小 -> 负向主要是规模代理;")
    print("      (B)若中性化后嵌入 |t| 仍<2 -> 现有文本嵌入无规模外增量信号。")


if __name__ == "__main__":
    main()
