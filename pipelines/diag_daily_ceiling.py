"""P1 日频天花板：把频率推到日频, 量价强因子的统计功效。

承接 diag_weekly_ceiling.py。日频截面数 ~1500(vs 周频 326), 功效进一步放大。
但注意: 日频下若用未来 5 日收益作标签, 相邻日截面标签重叠 -> 朴素 t 值虚高。
本脚本对重叠收益的 IC 序列用 Newey-West(lag=h-1) 校正 t 值, 给出诚实显著性。

同时给两个预测跨度做对比:
  fwd_1d : 未来 1 日收益 (非重叠, t 值干净; 但被反转/微观结构主导)
  fwd_5d : 未来 5 日收益 (重叠, NW 校正)

两个票池: full(全市场) / text372(研报覆盖股)。

输入: data/platform/px_daily.pqt, artifacts/panel.parquet
用法: python pipelines/diag_daily_ceiling.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
PANEL = ROOT / "artifacts" / "panel.parquet"

SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}
FEATS = ["mom_12_1", "rev_1d", "vol_60", "liq_amt20", "turn_20", "size"]
HORIZONS = {"fwd_1d": 1, "fwd_5d": 5}


def nw_tstat(x: pd.Series, lag: int) -> float:
    """Newey-West 调整后的均值 t 值 (检验 mean(x)=0)。lag=0 退化为普通 t。"""
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = math.sqrt(max(var, 1e-18) / n)
    return mu / se


def load_daily():
    px = pd.read_parquet(PLATFORM)
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    return px.sort_values(["stkcd", "date"])


def build_factors(px: pd.DataFrame) -> pd.DataFrame:
    g = px.groupby("stkcd", group_keys=False)
    px["ret1"] = g["adj_close"].pct_change(fill_method=None)
    px["c_1"] = g["adj_close"].shift(1)
    px["c_21"] = g["adj_close"].shift(21)
    px["c_252"] = g["adj_close"].shift(252)
    px["mom_12_1"] = px["c_21"] / px["c_252"] - 1.0
    px["rev_1d"] = -px["ret1"]
    px["vol_60"] = g["ret1"].rolling(60, min_periods=30).std().reset_index(level=0, drop=True)
    px["liq_amt20"] = np.log(
        g["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + 1.0)
    if "turnover" in px.columns:
        px["turn_20"] = g["turnover"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    if "float_mktcap" in px.columns:
        px["size"] = -np.log(px["float_mktcap"].astype(float) + 1.0)
    for col, h in HORIZONS.items():
        fwd = g["adj_close"].shift(-h)
        px[col] = fwd / px["adj_close"] - 1.0
    return px


def cs_ic(df: pd.DataFrame, fcol: str, ycol: str) -> pd.Series:
    sub = df[["date", fcol, ycol]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    ic = sub.groupby("date").apply(
        lambda gg: gg[fcol].corr(gg[ycol]), include_groups=False)
    return ic.dropna()


def run(df: pd.DataFrame, name: str, ycol: str, lag: int):
    n_sec = {k: int(((df["date"] >= pd.Timestamp(v[0])) &
                     (df["date"] <= pd.Timestamp(v[1]))).pipe(
                     lambda m: df.loc[m, "date"].nunique()))
             for k, v in SEG.items()}
    print("\n" + "=" * 80)
    print("[daily] universe=%s  label=%s  截面数=%s  (t值: NW lag=%d)"
          % (name, ycol, n_sec, lag))
    print("%-12s %18s %18s %18s" % ("factor", "train IC(t)", "valid IC(t)", "test IC(t)"))
    print("-" * 80)
    rows = []
    for f in [c for c in FEATS if c in df.columns]:
        cells = {}
        for split, seg in SEG.items():
            m = (df["date"] >= pd.Timestamp(seg[0])) & (df["date"] <= pd.Timestamp(seg[1]))
            ic = cs_ic(df[m], f, ycol)
            if ic.empty:
                cells[split] = "        NA        "
                continue
            tv = nw_tstat(ic, lag)
            cells[split] = "%+.4f (t=%6.2f)" % (ic.mean(), tv)
            rows.append((name, ycol, f, split, ic.mean(), tv, len(ic)))
        print("%-12s %18s %18s %18s" % (f, cells["train"], cells["valid"], cells["test"]))
    return rows


def main():
    px = build_factors(load_daily())
    panel = pd.read_parquet(PANEL)
    text_ids = set(panel.index.get_level_values(1).unique()
                   if isinstance(panel.index, pd.MultiIndex)
                   else panel["stkcd"].unique())
    px_text = px[px["stkcd"].isin(text_ids)].copy()

    all_rows = []
    for ycol, h in HORIZONS.items():
        lag = max(h - 1, 0)
        all_rows += run(px, "full", ycol, lag)
        all_rows += run(px_text, "text372", ycol, lag)

    out = pd.DataFrame(all_rows, columns=[
        "universe", "label", "factor", "split", "IC_mean", "t_nw", "n_periods"])
    out_path = ROOT / "artifacts" / "diag_daily_ceiling.csv"
    out.to_csv(out_path, index=False)
    print("\n[daily] wrote %s" % out_path)
    print("判读: 日频截面数~1500, 量价因子 |t| 应远超 2 -> 功效充足。")
    print("      注意 fwd_5d 的 t 已 NW 校正; 文本信号是否衰减需另做事件研究。")


if __name__ == "__main__":
    main()
