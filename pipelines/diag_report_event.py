# -*- coding: utf-8 -*-
"""研报到达事件研究：研报发布本身有没有"漂移"信号? 衰减多快?

这是日频文本 pipeline 重建前的【零成本前置检验】。现有 artifacts 的研报发布日
已被 Stage1a snap 到月度网格(68个日期), 做不了事件研究; 但 RDS 里 infopubldate
是真实发布日。本脚本直连 RDS 拉【真实发布日】的每股每日研报计数, 对齐平台日频
价格, 测"研报到达"事件对未来 N 日收益的横截面 IC 衰减曲线。

判读:
  - 若 fwd_1d/3d/5d 的 IC 显著且随 N 衰减 -> 研报有漂移信号, 重嵌入值得,
    且衰减曲线直接给出最优持仓周期;
  - 若各 N 的 IC 都 ≈0 -> 连"有没有研报"都没信号, 内容嵌入更难有,
    省下重嵌入, 转向量价+文本融合 / 换数据源。

信号定义(每个交易日的横截面):
  arrival   : 该股当日研报发布数(0 = 无)
  log_arr   : log(1+研报数), 压极端值
  has_report: 0/1 是否有研报

输入: RDS reportdata.report_info + data/platform/px_daily.pqt
      复用 load_real_docs 的 name->stkcd 映射
用法: python pipelines/diag_report_event.py
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
ARRIVAL_CACHE = ROOT / "data" / "report_arrivals.parquet"

START, END = "2019-12-01", "2025-08-01"
HORIZONS = [1, 3, 5, 10, 20]
SEG = {
    "train": ("2020-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2024-03-31"),
    "test":  ("2024-04-01", "2025-08-01"),
}


def nw_tstat(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * (e[k:] @ e[:-k]) / n
    return x.mean() / math.sqrt(max(var, 1e-18) / n)


def pull_arrivals() -> pd.DataFrame:
    """从 RDS 拉真实发布日的每股每日研报计数。带本地缓存。"""
    if ARRIVAL_CACHE.exists():
        print("[event] using cached arrivals: %s" % ARRIVAL_CACHE)
        return pd.read_parquet(ARRIVAL_CACHE)

    print("[event] building name->stkcd map ...")
    nm = L.build_name_to_stkcd()
    print("  mapped %d names" % len(nm.name2code))

    conn = L._connect("reportdata")
    edges = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="3MS").tolist()
    edges = [pd.Timestamp(START)] + edges + [pd.Timestamp(END) + pd.Timedelta(days=1)]
    edges = sorted(set(edges))
    rows = []
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            a_s, b_s = a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat()
            print("[event] pulling %s ~ %s" % (a_s, b_s))
            sql = ("SELECT infopubldate, stocks FROM report_info "
                   "WHERE infopubldate BETWEEN %s AND %s "
                   "  AND stocks IS NOT NULL AND stocks <> '' AND stocks <> 'None'")
            df = pd.read_sql(sql, conn, params=(a_s, b_s))
            if df.empty:
                continue
            df["names"] = df["stocks"].map(L._parse_stocks_field)
            df = df[df["names"].map(len) > 0].explode("names", ignore_index=True)
            df["stkcd"] = df["names"].map(nm.name2code)
            df = df.dropna(subset=["stkcd"])
            df["date"] = pd.to_datetime(df["infopubldate"]).dt.normalize()
            rows.append(df[["date", "stkcd"]])
    finally:
        conn.close()

    raw = pd.concat(rows, ignore_index=True)
    arr = (raw.groupby(["date", "stkcd"]).size()
              .rename("arrival").reset_index())
    arr.to_parquet(ARRIVAL_CACHE, index=False)
    print("[event] cached %d (date,stk) arrival rows -> %s" % (len(arr), ARRIVAL_CACHE))
    return arr


def main():
    arr = pull_arrivals()
    print("[event] arrival dates: %s ~ %s  stocks=%d"
          % (arr["date"].min().date(), arr["date"].max().date(), arr["stkcd"].nunique()))

    # 平台日频价格 -> 交易日历 + 未来 N 日收益
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    for h in HORIZONS:
        px["fwd_%d" % h] = g["adj_close"].shift(-h) / px["adj_close"] - 1.0

    # 研报到达对齐到"发布日当日或之后第一个交易日"(发布日可能是非交易日)
    trade_dates = np.sort(px["date"].unique())
    td = pd.Series(trade_dates)
    def _snap(d):
        i = td.searchsorted(d, side="left")
        return td.iloc[i] if i < len(td) else pd.NaT
    arr = arr.copy()
    arr["date"] = arr["date"].map(_snap)
    arr = arr.dropna(subset=["date"])
    arr = arr.groupby(["date", "stkcd"], as_index=False)["arrival"].sum()

    # 全市场每个(交易日,股) 都有一行: arrival 缺失=0
    df = px.merge(arr, on=["date", "stkcd"], how="left")
    df["arrival"] = df["arrival"].fillna(0.0)
    df["log_arr"] = np.log1p(df["arrival"])
    df["has_report"] = (df["arrival"] > 0).astype(float)

    print("[event] panel rows=%d  日均有研报股票数=%.1f"
          % (len(df), df.groupby("date")["has_report"].sum().mean()))

    signals = ["has_report", "log_arr"]
    print("\n研报到达事件 -> 未来 N 日收益的横截面 IC (test 段, NW 校正 t)")
    print("%-12s" % "signal" + "".join("  fwd_%-2d (t)   " % h for h in HORIZONS))
    print("-" * 78)
    ts, te = pd.Timestamp(SEG["test"][0]), pd.Timestamp(SEG["test"][1])
    sub_all = df[(df["date"] >= ts) & (df["date"] <= te)]
    out_rows = []
    for sig in signals:
        cells = []
        for h in HORIZONS:
            yc = "fwd_%d" % h
            s = sub_all[["date", sig, yc]].dropna()
            ic = s.groupby("date").apply(lambda gg: gg[sig].corr(gg[yc]),
                                         include_groups=False).dropna()
            tv = nw_tstat(ic.to_numpy(), lag=max(h - 1, 0))
            cells.append("%+.4f(%5.1f)" % (ic.mean(), tv))
            out_rows.append(("test", sig, h, ic.mean(), tv, len(ic)))
        print("%-12s" % sig + "".join("  %s" % c for c in cells))

    pd.DataFrame(out_rows, columns=["split", "signal", "horizon", "IC_mean", "t_nw", "n"]
                 ).to_csv(ROOT / "artifacts" / "diag_report_event.csv", index=False)
    print("\n[event] wrote artifacts/diag_report_event.csv")
    print("判读: 若 IC 随 N 单调衰减且短周期显著 -> 有漂移, 重嵌入值得, 持仓周期看曲线;")
    print("      若各 N 都 ≈0 -> 研报到达无信号, 慎重投入重嵌入。")


if __name__ == "__main__":
    main()
