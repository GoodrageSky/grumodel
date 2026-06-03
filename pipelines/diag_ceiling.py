"""P0 天花板测试：量价 / Barra 风格因子在当前评估窗口有没有 OOS 信号？

回答岔路口问题：文本因子 OOS 是噪声——到底是
  (A) 评估窗口本身没统计功效（17 个月度截面太少）, 还是
  (B) 文本在月频上真没 alpha。

做法：把平台导出的量价/Barra 因子对齐到现有 panel 的 (date, stkcd) 截面，
用【和真实流水线完全相同】的 purged 切分 + compute_ic 算 IC，附 t 值。
若量价/风格因子在这里 |t| 也 <2 → (A) 评估没功效，去修实验设计；
若它们 IC 显著 → (B) 文本相对量价弱，去改文本。

输入：
  artifacts/panel.parquet                 (vwap_rk 标签 + 日期/股票)
  data/platform/px_daily.pqt              (export_platform_factors.py 产出)
  data/platform/barra_exposure.pqt        (同上, 可选)
用法：
  python pipelines/diag_ceiling.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gru_lgbm_factor as t3
from config import PATHS, SEGMENTS, STAGE3

PLATFORM_DIR = ROOT / "data" / "platform"
LABEL_COL = STAGE3["label_col"]            # vwap_rk


def _t(ic_mean, ic_std, n):
    if not n or ic_std in (0, None) or np.isnan(ic_std):
        return float("nan")
    return ic_mean / (ic_std / math.sqrt(n))


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PATHS["panel"])
    if not isinstance(p.index, pd.MultiIndex):
        p["date"] = pd.to_datetime(p["date"])
        p = p.set_index(["date", "stkcd"]).sort_index()
    return p


def build_price_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """从日频量价构造经典因子, 对齐到 panel 的 (date, stkcd) 截面。

    因子(全部为"截至该面板日可得"的过去信息, 无前视):
      mom_12_1  : 过去 ~252d 到 ~21d 的累计收益 (跳过最近一月) —— 动量
      rev_1m    : 最近 ~21d 收益取负 —— 短期反转
      vol_60    : 过去 60d 日收益波动率
      liq_amt20 : 过去 20d 日均成交金额对数 —— 流动性
      turn_20   : 过去 20d 日均换手率 —— 换手/流动性
      size      : 流通市值对数取负 (小市值方向) —— 规模
    """
    px = pd.read_parquet(PLATFORM_DIR / "px_daily.pqt")
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.sort_values(["stkcd", "date"])
    # 停牌/非交易日价格不参与: 交易状态非正常时 adj_close 置 NaN
    if "trade_status" in px.columns:
        st = px["trade_status"].astype(str)
        not_trading = ~st.isin(["1", "1.0", u"交易", u"正常交易", "True"])
        # 仅当该列确有可识别取值时才屏蔽, 避免把全部行误杀
        if not_trading.mean() < 0.9:
            px.loc[not_trading, "adj_close"] = np.nan

    g = px.groupby("stkcd", group_keys=False)
    px["ret1"] = g["adj_close"].pct_change()
    px["c_21"] = g["adj_close"].shift(21)
    px["c_252"] = g["adj_close"].shift(252)
    px["mom_12_1"] = px["c_21"] / px["c_252"] - 1.0
    px["rev_1m"] = -(px["adj_close"] / px["c_21"] - 1.0)
    px["vol_60"] = g["ret1"].rolling(60, min_periods=30).std().reset_index(level=0, drop=True)
    px["liq_amt20"] = np.log(
        g["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True) + 1.0)
    if "turnover" in px.columns:
        px["turn_20"] = g["turnover"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    if "float_mktcap" in px.columns:
        px["size"] = -np.log(px["float_mktcap"].astype(float) + 1.0)

    feat_cols = [c for c in ["mom_12_1", "rev_1m", "vol_60", "liq_amt20", "turn_20", "size"]
                 if c in px.columns]
    px = px[["date", "stkcd"] + feat_cols].dropna(how="all", subset=feat_cols)

    # 对齐到 panel: 每个 (panel_date, stk) 取 <= panel_date 的最近一条日频因子
    panel_keys = panel.index.to_frame(index=False)[["date", "stkcd"]]
    out = []
    for stk, grp in px.groupby("stkcd"):
        pk = panel_keys[panel_keys["stkcd"] == stk]
        if pk.empty:
            continue
        merged = pd.merge_asof(
            pk.sort_values("date"),
            grp.sort_values("date")[["date"] + feat_cols],
            on="date", direction="backward")
        out.append(merged)
    aligned = pd.concat(out, ignore_index=True)
    aligned = aligned.set_index(["date", "stkcd"]).sort_index()
    return aligned.reindex(panel.index)


def load_barra(panel: pd.DataFrame) -> pd.DataFrame | None:
    fp = PLATFORM_DIR / "barra_exposure.pqt"
    if not fp.exists():
        return None
    b = pd.read_parquet(fp)
    b["date"] = pd.to_datetime(b["date"].astype(str), format="%Y%m%d")
    b = b.set_index(["date", "stkcd"]).sort_index()
    # Barra 月末 -> panel 月初: 按 stk 用上一月末暴露 asof 贴到 panel 日
    feat_cols = [c for c in b.columns]
    panel_keys = panel.index.to_frame(index=False)[["date", "stkcd"]]
    out = []
    for stk, grp in b.reset_index().groupby("stkcd"):
        pk = panel_keys[panel_keys["stkcd"] == stk]
        if pk.empty:
            continue
        merged = pd.merge_asof(
            pk.sort_values("date"),
            grp.sort_values("date")[["date"] + feat_cols],
            on="date", direction="backward")
        out.append(merged)
    if not out:
        return None
    aligned = pd.concat(out, ignore_index=True).set_index(["date", "stkcd"]).sort_index()
    return aligned.reindex(panel.index)


def single_factor_ic(panel: pd.DataFrame, feat: pd.Series, masks: dict) -> dict:
    """单因子截面 IC: 直接 corr(factor, vwap_rk), 无需拟合 —— 最干净的信号检验。"""
    res = {}
    df = panel.copy()
    df["_f"] = feat.reindex(df.index)
    for name, m in masks.items():
        idx = df.index[m]
        pred = df.loc[idx, "_f"].dropna()
        if pred.empty:
            res[name] = None
            continue
        ic = t3.compute_ic(pred, df, LABEL_COL)
        res[name] = ic
    return res


def main():
    panel = load_panel()
    cfg = t3.FactorConfig(**STAGE3)
    meta = panel.reset_index()[["date", "stkcd"]]
    masks = t3.purged_masks(meta, SEGMENTS, cfg)
    print("[ceiling] panel rows=%d  splits: %s"
          % (len(panel), {k: int(v.sum()) for k, v in masks.items()}))

    candidates: dict[str, pd.Series] = {}

    if (PLATFORM_DIR / "px_daily.pqt").exists():
        print("[ceiling] building price/volume factors ...")
        pf = build_price_factors(panel)
        for c in pf.columns:
            candidates[c] = pf[c]
    else:
        print("[ceiling] WARN: %s not found — skip price factors"
              % (PLATFORM_DIR / "px_daily.pqt"))

    barra = load_barra(panel)
    if barra is not None:
        print("[ceiling] barra factors: %s" % list(barra.columns))
        for c in barra.columns:
            candidates["barra_" + str(c)] = barra[c]

    if not candidates:
        print("[ceiling] no candidate factors — did you copy *.pqt to data/platform/ ?")
        return

    print("\n%-22s %18s %18s %18s" % ("factor", "train IC(t)", "valid IC(t)", "test IC(t)"))
    print("-" * 80)
    rows = []
    for fname, fser in candidates.items():
        ics = single_factor_ic(panel, fser, masks)
        cells = {}
        for split in ("train", "valid", "test"):
            ic = ics.get(split)
            if ic is None:
                cells[split] = "      NA       "
                continue
            tval = _t(ic["IC_mean"], ic["IC_std"], ic["n_periods"])
            cells[split] = "%+.4f (t=%5.2f)" % (ic["IC_mean"], tval)
            rows.append((fname, split, ic["IC_mean"], tval, ic["n_periods"]))
        print("%-22s %18s %18s %18s"
              % (fname, cells["train"], cells["valid"], cells["test"]))

    out = pd.DataFrame(rows, columns=["factor", "split", "IC_mean", "t", "n_periods"])
    out_path = ROOT / "artifacts" / "diag_ceiling.csv"
    out.to_csv(out_path, index=False)
    print("\n[ceiling] wrote %s" % out_path)
    print("\n判读: 看 test 列。任何量价/Barra 因子 |t|>=2 -> 评估有功效, 问题在文本(B);")
    print("      若全部 |t|<2 -> 评估窗口没功效, 先修实验设计(频率/股票池/walk-forward)(A)。")


if __name__ == "__main__":
    main()
