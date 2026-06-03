"""P1 验证：把频率从月频改成周频, 评估功效能否恢复?

月频天花板测试(diag_ceiling.py)结论是 (A): 连换手率/小市值这种强因子在
17 个月度截面里 |t| 都 <2 —— 评估窗口没统计功效。本脚本不动文本 pipeline,
纯用平台日频价格自建【周频】截面 + 未来1周收益标签 + 同样 6 个因子, 跑同一
train/valid/test 边界, 看 t 值能否因截面数增多(17 -> ~70+)而显著起来。

若周频下 turn_20/size 的 |t| 升到 >=2 -> 假设成立, 值得把整条 pipeline 改周频;
若仍 <2 -> 频率不是瓶颈, 得换思路(扩票池 / 信号本身)。

两个票池都测:
  full  : 平台全市场 (~5850 只)        —— 评估上限
  text  : panel 的 372 只研报覆盖股    —— 文本因子的实际作用域

输入: data/platform/px_daily.pqt, artifacts/panel.parquet
用法: python pipelines/diag_weekly_ceiling.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
PANEL = ROOT / "artifacts" / "panel.parquet"

# 与 config.SEGMENTS 一致的边界
SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}
FWD_WEEKS = 1          # 未来 1 周收益作标签 (非重叠)
PURGE_WEEKS = 1        # 段间按标签长度裁尾


def _t(mean, std, n):
    if not n or std in (0, None) or np.isnan(std) or std == 0:
        return float("nan")
    return mean / (std / math.sqrt(n))


def load_daily():
    px = pd.read_parquet(PLATFORM)
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.sort_values(["stkcd", "date"])
    return px


def build_daily_factors(px: pd.DataFrame) -> pd.DataFrame:
    """日频上算因子(用过去信息), 之后在周频采样日取值。无前视。"""
    g = px.groupby("stkcd", group_keys=False)
    px["ret1"] = g["adj_close"].pct_change(fill_method=None)
    px["c_5"] = g["adj_close"].shift(5)
    px["c_21"] = g["adj_close"].shift(21)
    px["c_252"] = g["adj_close"].shift(252)
    px["mom_12_1"] = px["c_21"] / px["c_252"] - 1.0
    px["rev_1w"] = -(px["adj_close"] / px["c_5"] - 1.0)
    px["vol_60"] = g["ret1"].rolling(60, min_periods=30).std().reset_index(level=0, drop=True)
    px["liq_amt20"] = np.log(
        g["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + 1.0)
    if "turnover" in px.columns:
        px["turn_20"] = g["turnover"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    if "float_mktcap" in px.columns:
        px["size"] = -np.log(px["float_mktcap"].astype(float) + 1.0)
    # 未来 1 周收益 (用 5 个交易日后的价格), 作标签
    px["fwd_close"] = g["adj_close"].shift(-5 * FWD_WEEKS)
    px["fwd_ret"] = px["fwd_close"] / px["adj_close"] - 1.0
    return px


def to_weekly(px: pd.DataFrame) -> pd.DataFrame:
    """每周取最后一个交易日作为截面日(周五口径)。"""
    px = px.copy()
    px["week"] = px["date"].dt.to_period("W")
    last = px.groupby(["stkcd", "week"])["date"].transform("max")
    wk = px[px["date"] == last].copy()
    return wk


FEATS = ["mom_12_1", "rev_1w", "vol_60", "liq_amt20", "turn_20", "size"]


def cs_ic(df: pd.DataFrame, fcol: str, ycol: str = "fwd_ret"):
    """逐截面 Pearson + Spearman IC。"""
    sub = df[["date", fcol, ycol]].dropna()
    if sub.empty:
        return None
    ic = sub.groupby("date").apply(lambda g: g[fcol].corr(g[ycol]), include_groups=False)
    ric = sub.groupby("date").apply(
        lambda g: g[fcol].corr(g[ycol], method="spearman"), include_groups=False)
    ic, ric = ic.dropna(), ric.dropna()
    return ic, ric


def split_mask(dates: pd.Series, seg, purge_days=5 * PURGE_WEEKS):
    s, e = pd.Timestamp(seg[0]), pd.Timestamp(seg[1])
    return (dates >= s) & (dates <= e)


def run_universe(wk: pd.DataFrame, name: str):
    print("\n" + "=" * 78)
    print("[weekly] universe=%s  截面数: %s"
          % (name, {k: int(((wk["date"] >= pd.Timestamp(v[0])) &
                            (wk["date"] <= pd.Timestamp(v[1]))).pipe(
                            lambda m: wk.loc[m, "date"].nunique()))
                    for k, v in SEG.items()}))
    print("%-12s %17s %17s %17s" % ("factor", "train IC(t)", "valid IC(t)", "test IC(t)"))
    print("-" * 78)
    rows = []
    for f in [c for c in FEATS if c in wk.columns]:
        cells = {}
        for split, seg in SEG.items():
            m = split_mask(wk["date"], seg)
            res = cs_ic(wk[m], f)
            if res is None:
                cells[split] = "       NA       "
                continue
            ic, _ = res
            tv = _t(ic.mean(), ic.std(), len(ic))
            cells[split] = "%+.4f (t=%5.2f)" % (ic.mean(), tv)
            rows.append((name, f, split, ic.mean(), tv, len(ic)))
        print("%-12s %17s %17s %17s" % (f, cells["train"], cells["valid"], cells["test"]))
    return rows


def main():
    px = load_daily()
    px = build_daily_factors(px)
    wk = to_weekly(px)

    # text 票池 = panel 的 372 只
    panel = pd.read_parquet(PANEL)
    text_ids = set(panel.index.get_level_values(1).unique()
                   if isinstance(panel.index, pd.MultiIndex)
                   else panel["stkcd"].unique())

    all_rows = []
    all_rows += run_universe(wk, "full")
    all_rows += run_universe(wk[wk["stkcd"].isin(text_ids)].copy(), "text372")

    out = pd.DataFrame(all_rows,
                       columns=["universe", "factor", "split", "IC_mean", "t", "n_periods"])
    out_path = ROOT / "artifacts" / "diag_weekly_ceiling.csv"
    out.to_csv(out_path, index=False)
    print("\n[weekly] wrote %s" % out_path)
    print("判读: 看各 universe 的 test 列。若 turn_20/size |t|>=2 -> 周频恢复功效,")
    print("      值得把 pipeline 改周频; 若仍 <2 -> 频率非瓶颈, 换思路。")


if __name__ == "__main__":
    main()
