"""Diagnostic: is there ANY out-of-sample signal, and where is it lost?

Runs a battery of controlled experiments on the SAME purged splits and the
SAME cross-sectional IC computation as the real Stage-3 pipeline, so results
are directly comparable to artifacts/stage3/direct_lgbm/metrics.json.

Experiments
-----------
  pos_control : feature = label + small noise        -> OOS IC must be HIGH
                (if not, pred<->label alignment / IC code is broken)
  neg_control : feature = pure noise                 -> OOS IC must be ~0
  raw768_ridge: 768-d FinBERT embeddings + ridge     -> is there signal in text?
  pca10_ridge : unsupervised PCA(10) + ridge         -> isolate supervised-UMAP effect
  umap10_ridge: existing supervised emb_* + ridge    -> linear read on current 10-d
  umap10_mean : existing emb_* simple average        -> model-free read

For every experiment we print, per split (valid/test):
  IC_mean, IC_std, ICIR, t = ICIR*sqrt(n), frac_pos, n_periods
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gru_lgbm_factor as t3
from config import PATHS, SEGMENTS, STAGE3

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

RNG = np.random.default_rng(42)


def load_aligned():
    panel = pd.read_parquet(PATHS["panel"]).sort_index()
    panel = panel[~panel.index.duplicated(keep="first")]
    daily = pd.read_parquet(PATHS["daily"])
    emb = np.load(ROOT / "artifacts/embeddings/summary_bert-base-chinese.npy")
    assert emb.shape[0] == daily.shape[0], (emb.shape, daily.shape)
    emb_df = pd.DataFrame(
        emb, index=daily.index,
        columns=[f"raw_{i}" for i in range(emb.shape[1])],
    )
    emb_df = emb_df.reindex(panel.index)
    n_nan = int(emb_df.isna().any(axis=1).sum())
    print(f"[align] panel={len(panel)} emb={len(emb_df)} "
          f"rows_unmatched_after_reindex={n_nan}")
    if n_nan:
        print("[align] WARNING: dropping unmatched rows (alignment imperfect)")
        keep = ~emb_df.isna().any(axis=1)
        panel, emb_df = panel[keep], emb_df[keep]
    return panel, emb_df


def make_masks(panel, cfg):
    meta = panel.reset_index()[[cfg.date_col, cfg.stock_col]]
    return t3.purged_masks(meta, SEGMENTS, cfg)


def ic_for_mask(pred, df_feat, cfg, split, masks):
    mask = np.asarray(masks[split])
    idx = df_feat.index[mask]
    sub = df_feat.loc[idx]
    p = pd.Series(np.asarray(pd.Series(pred).reindex(idx)), index=idx)
    res = t3.compute_ic(p, sub, cfg.label_col)
    n = res["n_periods"]
    icir = res["ICIR"]
    t = icir * math.sqrt(n) if n else float("nan")
    ic_series = res["IC_series"]
    frac_pos = float((ic_series > 0).mean())
    return dict(IC_mean=res["IC_mean"], IC_std=res["IC_std"], ICIR=icir,
                t=t, frac_pos=frac_pos, n=n)


def fit_ridge(X, y, masks, alpha=10.0):
    tr = masks["train"]
    model = Ridge(alpha=alpha)
    model.fit(X[tr], y[tr])
    return pd.Series(model.predict(X), index=y.index)


def report(name, pred, df_feat, cfg, masks):
    print(f"\n=== {name} ===")
    for split in ("valid", "test"):
        m = ic_for_mask(pred, df_feat, cfg, split, masks)
        print(f"  {split:5s} IC={m['IC_mean']:+.4f} std={m['IC_std']:.4f} "
              f"ICIR={m['ICIR']:+.3f} t={m['t']:+.2f} "
              f"frac_pos={m['frac_pos']:.2f} n={m['n']}")


def main():
    cfg = t3.FactorConfig(**STAGE3)
    panel, emb_df = load_aligned()

    # cross-sectional z-score the 10-d supervised features exactly like the pipeline
    feat10 = [c for c in panel.columns if c.startswith(cfg.feature_prefix)]
    df_feat = t3.cross_sectional_zscore(panel, feat10, cfg.winsorize_q)
    masks = make_masks(df_feat, cfg)
    for nm, mk in masks.items():
        print(f"[mask] {nm}: {int(mk.sum())} rows")

    y = df_feat[cfg.label_col]

    # ---- pos control: label + noise (alignment / IC sanity) ----
    noise = pd.Series(RNG.normal(0, 0.1, len(y)), index=y.index)
    report("pos_control (label + 0.1*noise)", y + noise, df_feat, cfg, masks)

    # ---- neg control: pure noise ----
    rnd = pd.Series(RNG.normal(0, 1, len(y)), index=y.index)
    report("neg_control (pure noise)", rnd, df_feat, cfg, masks)

    # ---- umap10 simple mean (model-free) ----
    mean10 = df_feat[feat10].mean(axis=1)
    report("umap10_mean (avg of emb_*)", mean10, df_feat, cfg, masks)

    # ---- umap10 + ridge ----
    X10 = df_feat[feat10].to_numpy()
    report("umap10_ridge", fit_ridge(X10, y, masks), df_feat, cfg, masks)

    # ---- raw 768 + ridge (cross-sectional z-score raw dims first) ----
    raw_cols = list(emb_df.columns)
    raw_panel = emb_df.join(df_feat[[cfg.label_col]])
    raw_z = t3.cross_sectional_zscore(raw_panel, raw_cols, cfg.winsorize_q)
    Xr = raw_z[raw_cols].to_numpy()
    for a in (1.0, 10.0, 100.0):
        report(f"raw768_ridge alpha={a}", fit_ridge(Xr, y, masks, alpha=a),
               df_feat, cfg, masks)

    # ---- unsupervised PCA(10) on raw 768 (fit on TRAIN only) ----
    tr = masks["train"]
    pca = PCA(n_components=10, random_state=42)
    pca.fit(Xr[tr])
    Xp = pca.transform(Xr)
    report("pca10_ridge (PCA fit on train)", fit_ridge(Xp, y, masks),
           df_feat, cfg, masks)

    print("\n[done] interpretation:")
    print("  pos_control t huge  -> alignment & IC code OK")
    print("  raw768 / pca10 OOS t ~0 and ~umap10 -> text has no OOS alpha (not a tuning issue)")


if __name__ == "__main__":
    main()
