"""Smoke test Stage 3 on the existing dt_lgbm.parquet (~monthly, emb_4096)."""
import pandas as pd
import gru_lgbm_factor as t3

df = pd.read_parquet("dt_lgbm.parquet")
print("loaded:", df.shape, "| dates:", df.date.nunique(), "| stocks:", df.stkcd.nunique())

SEGMENTS = {
    "train": slice("2017-02-01", "2022-12-31"),
    "valid": slice("2023-01-01", "2024-03-31"),
    "test":  slice("2024-04-01", "2025-08-01"),
}

cfg = t3.FactorConfig(
    seq_len=6,                 # 6 monthly steps ≈ half-year history
    label_horizon=1,           # 1 month purge between segments (label = 20-trading-day fwd return)
    max_gap_days=None,         # disable gap filter for non-daily data
    label_col="vwap_rk",
    return_col="vwap_ret20",
    feature_prefix="emb_",
    proj_dim=128, gru_hidden=64, gru_layers=2,
    epochs=5, batch_size=128, patience=3,
    lgbm_num_boost_round=200, lgbm_early_stopping=20,
    device="cpu",              # smoke test on CPU
)
res = t3.run_pipeline(df, SEGMENTS, cfg)
print()
print(pd.DataFrame(res["metrics"]).T[["IC_mean", "RankIC_mean", "ICIR", "n_days"]])
if res["backtest"] is not None:
    bt = res["backtest"]
    print(f"test L/S mean={bt['ls_mean']:.4f}  ann.Sharpe={bt['ls_sharpe']:.2f}")
