"""
gru_lgbm_factor.py
==================
Stage 3 of the panel-frequency text-factor pipeline: turn a panel of
(date, stkcd) text-embedding features into a tradable factor via a
GRU sequence encoder followed by LightGBM.

This is a refined version of the original `gru.ipynb` baseline. The main
improvements over the baseline:

1.  PANEL-FREQUENCY AWARE sequence builder. Windows are only built when the
    `seq_len` steps fall within `max_gap_days` calendar days, so a window never
    silently spans a long trading suspension. Fully vectorised (no per-sample
    Python loop) -> ~100x faster than the baseline.

2.  LEAKAGE CONTROL. The label is a forward return (horizon = `label_horizon`
    trading days), so the last `label_horizon` days of every segment overlap the
    next segment. We *purge* that overlap between train / valid / test. The
    baseline did not, which inflates the validation metric.

3.  STACKING LEAKAGE is made explicit and optionally fixed. The baseline trains
    the GRU on train labels and then feeds GRU features (including on the train
    rows the GRU was fit to) into LightGBM. The test factor is still clean
    (the GRU never saw test labels), but the train/valid LGBM metrics are
    optimistic. `oof_gru_features=True` extracts GRU features out-of-fold.

4.  FACTOR-NATIVE EVALUATION. MSE on a rank label is nearly meaningless for a
    quant factor, so we report cross-sectional IC, RankIC, ICIR and a
    top/bottom quantile long-short curve.

The module depends only on numpy / pandas / torch / lightgbm — it does NOT
require the original `lgbm_model.py` wrapper (a self-contained fit is provided),
but the data contract (MultiIndex [date, stkcd], `emb_*` features, a forward
label column) is identical, so you can drop it into the existing workflow.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class FactorConfig:
    # columns / index
    date_col: str = "date"
    stock_col: str = "stkcd"
    label_col: str = "vwap_rk"            # cross-sectional rank of forward return
    return_col: Optional[str] = "vwap_ret20"  # raw forward return for the backtest (optional)
    feature_prefix: str = "emb_"          # which columns feed the GRU

    # sequence construction (in STEPS = unique dates of the panel; the panel can
    # be daily or monthly, the code treats one row per date as one step)
    seq_len: int = 20                     # history window length, in steps
    label_horizon: int = 20               # forward-return horizon, in steps -> purge size between segments
    max_gap_days: Optional[int] = 10      # reject a window if any consecutive-step gap exceeds this many CALENDAR days; None disables
    annualization_periods: int = 252      # periods per year for Sharpe annualization (12 for monthly panels)

    # cross-sectional preprocessing
    winsorize_q: float = 0.01             # clip features to [q, 1-q] per day; set None to disable

    # GRU
    proj_dim: int = 256
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    oof_gru_features: bool = False        # True -> extract train GRU features out-of-fold

    # LightGBM
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "regression", "metric": "l2", "learning_rate": 0.05,
        "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "min_data_in_leaf": 50, "lambda_l2": 1.0,
        "verbosity": -1, "seed": 42,
    })
    lgbm_num_boost_round: int = 1000
    lgbm_early_stopping: int = 50

    # misc
    seed: int = 42
    device: str = "auto"                  # "auto" | "cuda" | "cpu"
    model_dir: str = "./artifacts_gru"


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# Cross-sectional preprocessing
# --------------------------------------------------------------------------- #
def cross_sectional_zscore(df: pd.DataFrame, cols, winsorize_q: Optional[float] = 0.01) -> pd.DataFrame:
    """Per-date (cross-sectional) winsorize + z-score. No look-ahead: each day is
    standardised using only that day's cross-section."""
    df = df.copy()
    g = df.groupby(level=0)  # level 0 = date

    if winsorize_q:
        lo = g[cols].transform(lambda s: s.quantile(winsorize_q))
        hi = g[cols].transform(lambda s: s.quantile(1 - winsorize_q))
        df[cols] = df[cols].clip(lower=lo, upper=hi)
        g = df.groupby(level=0)

    mean = g[cols].transform("mean")
    std = g[cols].transform("std").replace(0, np.nan)
    df[cols] = (df[cols] - mean) / (std + 1e-8)
    df[cols] = df[cols].fillna(0.0)
    return df


# --------------------------------------------------------------------------- #
# Vectorised, gap-aware sequence builder
# --------------------------------------------------------------------------- #
def build_sequence_samples(
    df: pd.DataFrame,
    feature_cols,
    cfg: FactorConfig,
):
    """
    Input : df with MultiIndex [date, stkcd], sorted.
    Output: X (N, seq_len, D) float32, y (N,) float32, meta DataFrame[date, stkcd]
            (meta date = the date of the LAST step in the window = prediction date)

    A window is kept only if:
      * the label at the end step is not NaN,
      * no feature in the window is NaN,
      * the largest calendar gap between consecutive steps <= cfg.max_gap_days.
    """
    seq_len = cfg.seq_len
    df_reset = df.reset_index().sort_values([cfg.stock_col, cfg.date_col])

    X_parts, y_parts, date_parts, stk_parts = [], [], [], []

    for stk, g in df_reset.groupby(cfg.stock_col, sort=False):
        if len(g) < seq_len:
            continue
        feats = g[feature_cols].to_numpy(np.float32)              # (L, D)
        labels = g[cfg.label_col].to_numpy(np.float32)            # (L,)
        dates = g[cfg.date_col].to_numpy("datetime64[ns]")        # (L,)
        L, D = feats.shape

        # sliding windows over the time axis -> (n_win, seq_len, D)
        win = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)  # (n_win, D, seq_len)
        win = np.ascontiguousarray(win.transpose(0, 2, 1))                      # (n_win, seq_len, D)

        end_labels = labels[seq_len - 1:]
        end_dates = dates[seq_len - 1:]

        # max calendar gap inside each window
        day_diff = np.diff(dates).astype("timedelta64[D]").astype(np.int64)     # (L-1,)
        if seq_len > 1:
            gap_win = np.lib.stride_tricks.sliding_window_view(day_diff, seq_len - 1)  # (n_win, seq_len-1)
            max_gap = gap_win.max(axis=1)
        else:
            max_gap = np.zeros(len(win), dtype=np.int64)

        valid = (
            ~np.isnan(end_labels)
            & ~np.isnan(win).any(axis=(1, 2))
        )
        if cfg.max_gap_days is not None:
            valid &= (max_gap <= cfg.max_gap_days)
        if not valid.any():
            continue

        X_parts.append(win[valid])
        y_parts.append(end_labels[valid])
        date_parts.append(end_dates[valid])
        stk_parts.append(np.full(valid.sum(), stk))

    if not X_parts:
        raise ValueError("No valid sequences built — check seq_len / max_gap_days / NaNs.")

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    meta = pd.DataFrame({
        cfg.date_col: pd.to_datetime(np.concatenate(date_parts)),
        cfg.stock_col: np.concatenate(stk_parts),
    })
    return X, y, meta


# --------------------------------------------------------------------------- #
# Purged segment masks
# --------------------------------------------------------------------------- #
def purged_masks(meta: pd.DataFrame, segments: dict, cfg: FactorConfig) -> dict:
    """Boolean masks for each segment, with the tail of every non-final segment
    trimmed by `label_horizon` trading days so its forward-looking labels do not
    overlap the following segment. Returns {name: np.ndarray[bool]}."""
    d = pd.to_datetime(meta[cfg.date_col])
    # trading-day calendar from the data itself
    all_days = np.sort(d.unique())
    day_rank = pd.Series(np.arange(len(all_days)), index=all_days)

    ordered = sorted(segments.items(), key=lambda kv: pd.Timestamp(kv[1].start))
    masks = {}
    for i, (name, sl) in enumerate(ordered):
        start = pd.Timestamp(sl.start)
        end = pd.Timestamp(sl.stop)
        m = (d >= start) & (d <= end)
        # purge the tail of every segment except the last
        if i < len(ordered) - 1 and cfg.label_horizon > 0:
            end_idx = int(day_rank.asof(end))
            cut_idx = max(end_idx - cfg.label_horizon, -1)
            cut_day = all_days[cut_idx] if cut_idx >= 0 else all_days[0] - np.timedelta64(1, "D")
            m = m & (d <= cut_day)
        masks[name] = m.to_numpy()
    return masks


# --------------------------------------------------------------------------- #
# GRU encoder (kept architecturally close to the baseline)
# --------------------------------------------------------------------------- #
def _build_gru(input_dim: int, cfg: FactorConfig):
    import torch.nn as nn

    class GRUFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(input_dim, cfg.proj_dim)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(cfg.proj_dim)
            self.gru = nn.GRU(
                input_size=cfg.proj_dim, hidden_size=cfg.gru_hidden,
                num_layers=cfg.gru_layers, batch_first=True,
                dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
            )
            self.pred_head = nn.Linear(cfg.gru_hidden, 1)

        def forward(self, x, extract_features=False):
            x = self.norm(self.act(self.projection(x)))
            _, h_n = self.gru(x)
            feat = h_n[-1]
            if extract_features:
                return feat
            return self.pred_head(feat).squeeze(-1)

    return GRUFeatureExtractor()


def _make_loader(X, y, batch_size, shuffle):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class _DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i] if self.y is None else (self.X[i], self.y[i])

    return DataLoader(_DS(X, y), batch_size=batch_size, shuffle=shuffle, pin_memory=False)


def _run_epoch(model, loader, device, optimizer=None):
    import torch
    import torch.nn as nn
    train = optimizer is not None
    model.train() if train else model.eval()
    crit = nn.MSELoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = crit(model(x), y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_gru(X_tr, y_tr, X_va, y_va, cfg: FactorConfig, ckpt_path: str):
    import torch
    device = resolve_device(cfg.device)
    model = _build_gru(X_tr.shape[-1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = _make_loader(X_tr, y_tr, cfg.batch_size, shuffle=True)
    va = _make_loader(X_va, y_va, cfg.batch_size, shuffle=False)

    best, bad, hist = float("inf"), 0, []
    for ep in range(1, cfg.epochs + 1):
        trl = _run_epoch(model, tr, device, opt)
        val = _run_epoch(model, va, device, None)
        hist.append((ep, trl, val))
        print(f"Epoch {ep:02d} | train={trl:.6f} | valid={val:.6f}")
        if val < best:
            best, bad = val, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping.")
                break
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, pd.DataFrame(hist, columns=["epoch", "train_loss", "valid_loss"])


def extract_gru_features(model, X, cfg: FactorConfig):
    import torch
    device = resolve_device(cfg.device)
    model.eval()
    loader = _make_loader(X, None, cfg.batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for x in loader:
            out.append(model(x.to(device), extract_features=True).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Self-contained LightGBM fit over named segments
# --------------------------------------------------------------------------- #
def fit_lgbm_segments(df_feat: pd.DataFrame, feature_cols, label_col, masks: dict, cfg: FactorConfig):
    import lightgbm as lgb
    Xtr, ytr = df_feat.loc[masks["train"], feature_cols], df_feat.loc[masks["train"], label_col]
    Xva, yva = df_feat.loc[masks["valid"], feature_cols], df_feat.loc[masks["valid"], label_col]

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    booster = lgb.train(
        cfg.lgbm_params, dtr,
        num_boost_round=cfg.lgbm_num_boost_round,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping), lgb.log_evaluation(50)],
    )
    preds = {}
    for name, m in masks.items():
        idx = df_feat.index[m]
        preds[name] = pd.Series(booster.predict(df_feat.loc[m, feature_cols]), index=idx, name="pred")
    return booster, preds


# --------------------------------------------------------------------------- #
# Factor evaluation: IC / RankIC / ICIR + quantile long-short
# --------------------------------------------------------------------------- #
def compute_ic(pred: pd.Series, df: pd.DataFrame, label_col: str, date_col: str = "date"):
    """Cross-sectional IC (Pearson) and RankIC (Spearman) per day, summarised."""
    tmp = pd.DataFrame({"pred": pred.values, "y": df.loc[pred.index, label_col].values},
                       index=pred.index)
    lvl = tmp.index.get_level_values(0)
    ic = tmp.groupby(lvl).apply(lambda g: g["pred"].corr(g["y"]))
    ric = tmp.groupby(lvl).apply(lambda g: g["pred"].corr(g["y"], method="spearman"))
    ic, ric = ic.dropna(), ric.dropna()
    return {
        "IC_mean": ic.mean(), "IC_std": ic.std(),
        "ICIR": ic.mean() / (ic.std() + 1e-12),
        "RankIC_mean": ric.mean(), "RankIC_std": ric.std(),
        "RankICIR": ric.mean() / (ric.std() + 1e-12),
        "n_periods": len(ic), "n_days": len(ic), "IC_series": ic, "RankIC_series": ric,
    }


def quantile_backtest(pred: pd.Series, df: pd.DataFrame, ret_col: str, n_q: int = 5,
                      annualization_periods: int = 252):
    """Top-minus-bottom quantile per-period long-short return curve. Needs a raw
    forward-return column (e.g. vwap_ret20)."""
    tmp = pd.DataFrame({"pred": pred.values, "ret": df.loc[pred.index, ret_col].values},
                       index=pred.index).dropna()
    lvl = tmp.index.get_level_values(0)

    def _ls(g):
        if len(g) < n_q:
            return np.nan
        try:
            q = pd.qcut(g["pred"].rank(method="first"), n_q, labels=False, duplicates="drop")
        except ValueError:
            return np.nan
        top, bot = q.max(), q.min()
        if top == bot:
            return np.nan
        return g.loc[q == top, "ret"].mean() - g.loc[q == bot, "ret"].mean()

    period_ls = tmp.groupby(lvl).apply(_ls).dropna()
    sharpe = period_ls.mean() / (period_ls.std() + 1e-12) * math.sqrt(annualization_periods)
    return {"ls_period": period_ls, "ls_daily": period_ls, "ls_mean": period_ls.mean(),
            "ls_sharpe": sharpe, "annualization_periods": annualization_periods,
            "ls_cum": (1 + period_ls).cumprod()}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame, segments: dict, cfg: FactorConfig):
    """End-to-end Stage-3 run. `df` is either a MultiIndex [date, stkcd] panel,
    or a flat DataFrame whose columns include cfg.date_col and cfg.stock_col
    (the function will set the MultiIndex automatically). It must contain
    feature columns (prefix cfg.feature_prefix), cfg.label_col, and optionally
    cfg.return_col. Returns a results dict."""
    set_seed(cfg.seed)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    # accept either MultiIndex or flat DataFrame
    if not isinstance(df.index, pd.MultiIndex):
        if not {cfg.date_col, cfg.stock_col}.issubset(df.columns):
            raise ValueError(
                f"df must be MultiIndex[{cfg.date_col},{cfg.stock_col}] or contain those as columns"
            )
        df = df.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df = df.set_index([cfg.date_col, cfg.stock_col]).sort_index()

    # de-duplicate (date, stkcd) rows -- keep the first occurrence. The
    # downstream label/return lookup requires a unique MultiIndex.
    if df.index.duplicated().any():
        n_dup = int(df.index.duplicated().sum())
        print(f"[warn] {n_dup} duplicate (date, stkcd) rows in input -- keeping first occurrence")
        df = df[~df.index.duplicated(keep="first")]

    feature_cols = [c for c in df.columns if c.startswith(cfg.feature_prefix)]
    if not feature_cols:
        raise ValueError(f"no feature columns with prefix '{cfg.feature_prefix}' found")
    print(f"#features={len(feature_cols)}  device={resolve_device(cfg.device)}")

    df = cross_sectional_zscore(df, feature_cols, cfg.winsorize_q)
    X, y, meta = build_sequence_samples(df, feature_cols, cfg)
    print(f"sequences: X={X.shape}  y={y.shape}")

    masks = purged_masks(meta, segments, cfg)
    for k, m in masks.items():
        print(f"  {k}: {int(m.sum())} samples")

    # train GRU on the purged train split, validate on purged valid split
    model, hist = train_gru(X[masks["train"]], y[masks["train"]],
                            X[masks["valid"]], y[masks["valid"]],
                            cfg, ckpt_path=str(Path(cfg.model_dir) / "best_gru.pt"))

    gru_feats = extract_gru_features(model, X, cfg)

    if cfg.oof_gru_features:
        # refit a second GRU on valid+test? No — instead extract train features
        # from a model that did NOT see them: train on valid, predict train.
        # Simple, conservative OOF for the train block only.
        m2, _ = train_gru(X[masks["valid"]], y[masks["valid"]],
                          X[masks["valid"]], y[masks["valid"]],
                          cfg, ckpt_path=str(Path(cfg.model_dir) / "oof_gru.pt"))
        gru_feats[masks["train"]] = extract_gru_features(m2, X[masks["train"]], cfg)

    gru_cols = [f"gru_{i}" for i in range(gru_feats.shape[1])]
    df_gru = pd.concat([meta.reset_index(drop=True),
                        pd.DataFrame(gru_feats, columns=gru_cols)], axis=1)
    df_gru[cfg.label_col] = y
    df_gru = df_gru.set_index([cfg.date_col, cfg.stock_col]).sort_index()
    if cfg.return_col and cfg.return_col in df.columns:
        df_gru[cfg.return_col] = df[cfg.return_col].reindex(df_gru.index).to_numpy()

    # re-derive masks aligned to df_gru row order
    masks_g = purged_masks(df_gru.reset_index()[[cfg.date_col, cfg.stock_col]], segments, cfg)

    booster, preds = fit_lgbm_segments(df_gru, gru_cols, cfg.label_col, masks_g, cfg)

    metrics = {}
    for name, p in preds.items():
        m = compute_ic(p, df_gru, cfg.label_col)
        metrics[name] = {k: v for k, v in m.items() if not k.endswith("series")}
        print(f"[{name}] IC={m['IC_mean']:.4f} RankIC={m['RankIC_mean']:.4f} "
              f"ICIR={m['ICIR']:.3f} (n_periods={m['n_periods']})")

    bt = None
    if cfg.return_col and cfg.return_col in df_gru.columns:
        bt = quantile_backtest(
            preds["test"], df_gru, cfg.return_col,
            annualization_periods=cfg.annualization_periods,
        )
        print(f"[test] L/S mean={bt['ls_mean']:.4f} "
              f"ann.Sharpe({cfg.annualization_periods})={bt['ls_sharpe']:.2f}")

    return {"gru_model": model, "history": hist, "booster": booster,
            "preds": preds, "metrics": metrics, "backtest": bt,
            "df_gru": df_gru, "feature_cols": gru_cols}
