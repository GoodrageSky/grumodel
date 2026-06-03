"""按季度逐批跑 --- 大年不会因为单季超时挂掉。"""
import sys, time, traceback
from pathlib import Path
import pandas as pd
from build_news_docs import BuildConfig, build_raw_docs

sys.stdout.reconfigure(encoding='utf-8')
OUT_DIR = Path(__file__).resolve().parent  # 脚本所在目录

# 策略: 2020-2024 按季度, 其余按年
JOBS = [
    ("2026-Q1", "2026-01-01", "2026-03-31"),
    ("2026-Q2", "2026-04-01", "2026-06-03"),
    ("2025-Q1", "2025-01-01", "2025-03-31"),
    ("2025-Q2", "2025-04-01", "2025-06-30"),
    ("2025-Q3", "2025-07-01", "2025-09-30"),
    ("2025-Q4", "2025-10-01", "2025-12-31"),
    ("2024-Q1", "2024-01-01", "2024-03-31"),
    ("2024-Q2", "2024-04-01", "2024-06-30"),
    ("2024-Q3", "2024-07-01", "2024-09-30"),
    ("2024-Q4", "2024-10-01", "2024-12-31"),
    ("2023-Q1", "2023-01-01", "2023-03-31"),
    ("2023-Q2", "2023-04-01", "2023-06-30"),
    ("2023-Q3", "2023-07-01", "2023-09-30"),
    ("2023-Q4", "2023-10-01", "2023-12-31"),
    ("2022-Q1", "2022-01-01", "2022-03-31"),
    ("2022-Q2", "2022-04-01", "2022-06-30"),
    ("2022-Q3", "2022-07-01", "2022-09-30"),
    ("2022-Q4", "2022-10-01", "2022-12-31"),
    ("2021-Q1", "2021-01-01", "2021-03-31"),
    ("2021-Q2", "2021-04-01", "2021-06-30"),
    ("2021-Q3", "2021-07-01", "2021-09-30"),
    ("2021-Q4", "2021-10-01", "2021-12-31"),
    ("2020-Q1", "2020-01-01", "2020-03-31"),
    ("2020-Q2", "2020-04-01", "2020-06-30"),
    ("2020-Q3", "2020-07-01", "2020-09-30"),
    ("2020-Q4", "2020-10-01", "2020-12-31"),
]

for label, start, end in JOBS:
    out_file = f"raw_docs_news_{label}.parquet"
    out_path = OUT_DIR / out_file

    if out_path.exists():
        df_exist = pd.read_parquet(out_path)
        print(f"[{label}] 跳过（已有 {len(df_exist):,} 条）", flush=True)
        continue

    print(f"\n{'='*40}", flush=True)
    print(f"处理 {label} ({start} ~ {end}) ...", flush=True)
    t0 = time.time()

    try:
        cfg = BuildConfig(
            start_date=start,
            end_date=end,
            output_file=str(out_path),
            batch_size=10000,
            verbose=True,
        )
        df = build_raw_docs(cfg)
        elapsed = time.time() - t0
        print(f"[{label}] OK! {len(df):,} 条 | {elapsed/60:.1f} 分钟", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{label}] FAIL! {e}", flush=True)
        traceback.print_exc()
