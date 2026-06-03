"""Smoke test load_real_docs on a tiny date window."""
import warnings, sys, io
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import load_real_docs as L

cfg = L.LoadConfig(start="2024-01-02", end="2024-01-03",  # 2 days only
                   batch_months=1, verbose=True)
raw = L.load_real_docs(cfg)
print("\n--- head ---")
print(raw.head(5).to_string(max_colwidth=80))
print("\n--- shape:", raw.shape)
print("--- distinct stocks (top 10):", raw.stkcd.value_counts().head(10).to_dict())

# also test snap_to_panel_dates against dt_lgbm.parquet
panel = pd.read_parquet("dt_lgbm.parquet")[["date", "stkcd"]]
panel["date"] = pd.to_datetime(panel["date"])
print(f"\n--- snap test against {panel.date.nunique()} panel dates ---")
snapped = L.snap_to_panel_dates(raw, panel, max_forward_days=40)
print("snapped shape:", snapped.shape)
if not snapped.empty:
    print(snapped[["date", "stkcd", "title"]].head(5).to_string())
