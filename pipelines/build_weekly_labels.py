"""构建【周频】标签源, 替代月频 data/dt_lgbm.parquet。

口径(已与用户确认 + 天花板测试验证):
  - 截面频率: 周频, 每周最后一个交易日(周五口径)
  - 标签收益: 未来 1 周 vwap 收益 = vwap[t+5交易日] / vwap[t] - 1  (非重叠)
  - 标签分位: 每个截面内对 vwap_ret_1w 做横截面 rank -> [-1, 1], 命名 vwap_rk
              (与月频 dt_lgbm 的 vwap_rk 口径一致, 下游 stage2/3 无需改列名)

输出: data/weekly_labels.parquet  (date, stkcd, vwap_ret_1w, vwap_rk)
      列名 vwap_ret_1w 同时另存一份别名 vwap_ret20 以兼容 STAGE3.return_col,
      避免改 config(回测用的是原始收益, 周频下它就是周收益)。

用法: python pipelines/build_weekly_labels.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
OUT = ROOT / "data" / "weekly_labels.parquet"

FWD_DAYS = 5          # 未来 1 周 ~ 5 个交易日


def main():
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "vwap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["vwap"]).sort_values(["stkcd", "date"])

    # 未来 1 周 vwap 收益 (按交易日 shift, 不跨股票)
    g = px.groupby("stkcd", group_keys=False)
    px["fwd_vwap"] = g["vwap"].shift(-FWD_DAYS)
    px["vwap_ret_1w"] = px["fwd_vwap"] / px["vwap"] - 1.0

    # 周频采样: 每股每周取最后一个交易日
    px["week"] = px["date"].dt.to_period("W")
    last = px.groupby(["stkcd", "week"])["date"].transform("max")
    wk = px[px["date"] == last].copy()
    wk = wk.dropna(subset=["vwap_ret_1w"])

    # 横截面 rank -> [-1, 1] (与月频 vwap_rk 同口径)
    def _rank(s):
        r = s.rank(method="average")
        n = r.notna().sum()
        if n <= 1:
            return s * 0.0
        return (r - 1) / (n - 1) * 2 - 1
    wk["vwap_rk"] = wk.groupby("date")["vwap_ret_1w"].transform(_rank)

    out = (wk[["date", "stkcd", "vwap_ret_1w", "vwap_rk"]]
           .rename(columns={})
           .reset_index(drop=True))
    # 兼容别名: STAGE3.return_col 默认 vwap_ret20, 周频下指向周收益
    out["vwap_ret20"] = out["vwap_ret_1w"]

    out = out.sort_values(["date", "stkcd"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    n_dates = out["date"].nunique()
    print("[weekly-labels] wrote %s" % OUT)
    print("[weekly-labels] rows=%d  weekly_sections=%d  stocks=%d"
          % (len(out), n_dates, out["stkcd"].nunique()))
    print("[weekly-labels] date range: %s ~ %s"
          % (out["date"].min().date(), out["date"].max().date()))
    print("[weekly-labels] vwap_rk range: [%.3f, %.3f] mean=%.4f"
          % (out["vwap_rk"].min(), out["vwap_rk"].max(), out["vwap_rk"].mean()))


if __name__ == "__main__":
    main()
