"""Shared configuration for stage1a / stage1b / stage2 / stage3 + compare script.

All scripts `from config import PATHS, SEGMENTS, STAGE1A, ...`.
Edit values here; do NOT scatter parameters across stage scripts.

Paths are anchored to the project root (parent of src/), so scripts work
regardless of the caller's CWD.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root (parent of src/)
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
CACHE = ROOT / "cache"

# Train / valid / test boundaries. All stage scripts read from here — never
# re-define these slices locally.
SEGMENTS = {
    "train": slice("2020-01-01", "2022-12-31"),
    "valid": slice("2023-01-01", "2024-03-31"),
    "test":  slice("2024-04-01", "2025-08-01"),
}

PATHS = dict(
    raw_docs      = ARTIFACTS / "raw_docs.parquet",
    daily         = ARTIFACTS / "daily.parquet",
    panel         = ARTIFACTS / "panel.parquet",
    label_src     = DATA / "dt_lgbm.parquet",           # truth source for vwap_rk / vwap_ret20
    qc_dir        = ARTIFACTS / "qc",
    stage3_dir    = ARTIFACTS / "stage3",
    summary_cache = CACHE / "summaries",                # LLM/offline summary cache (hashed)
    synth_cache   = CACHE / "synth_docs",               # synthetic-research-report cache
)

STAGE1A = dict(
    source="real",                                  # real / synth / file
    start="2020-01-01", end="2025-08-01",
    batch_months=3,
    use_content_fallback=True,
    max_text_chars=4000, min_text_chars=30,
    snap_to_panel=True, max_forward_days=40,        # align report date -> nearest panel date
)

STAGE1B = dict(
    max_chars=4000,
    target_min=300, target_max=500,
    use_llm=False,                                  # False: offline truncate. True: Anthropic.
    llm_model="deepseek-chat",
    qc_low_threshold=0.80,
    qc_embed_model="bert-base-chinese",
)

STAGE2 = dict(
    text_col="summary",                             # summary / full_text
    embed_model="bert-base-chinese",
    n_components=10,
    n_bins=10,
    use_bertopic=True,
    random_state=42,
)

STAGE3 = dict(
    seq_len=6,
    label_horizon=1,
    max_gap_days=None,                              # monthly panel: do NOT check calendar gap
    annualization_periods=12,                       # monthly panel -> annualized Sharpe uses sqrt(12)
    label_col="vwap_rk",
    return_col="vwap_ret20",
    feature_prefix="emb_",
    proj_dim=128,
    gru_hidden=64, gru_layers=2,
    epochs=10, batch_size=128, patience=3,
    lgbm_num_boost_round=500, lgbm_early_stopping=30,
    oof_gru_features=False,
    device="auto",
    model_dir=str(ARTIFACTS / "gru_models"),        # override: keep checkpoints under artifacts/
)
