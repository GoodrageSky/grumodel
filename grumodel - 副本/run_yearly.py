"""按年逐批跑 build_news_docs，每完成一年立刻落盘，挂了只需重跑当年。"""
import sys, time, traceback
from pathlib import Path
import pandas as pd
from build_news_docs import BuildConfig, build_raw_docs

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

OUT_DIR = Path("C:/Users/PA/WorkBuddy/2026-06-03-10-24-56")
YEARS = list(range(2018, 2027))  # 2018 ~ 2026

for yr in YEARS:
    out_file = f"raw_docs_news_{yr}.parquet"
    out_path = OUT_DIR / out_file

    # 已跑过的跳过
    if out_path.exists():
        df_exist = pd.read_parquet(out_path)
        print(f"[{yr}] 跳过（已有 {len(df_exist):,} 条）", flush=True)
        continue

    print(f"\n{'='*50}", flush=True)
    print(f"开始处理 {yr} 年 ...", flush=True)
    print(f"{'='*50}", flush=True)
    t0 = time.time()

    try:
        cfg = BuildConfig(
            start_date=f"{yr}-01-01",
            end_date=f"{yr}-12-31",
            output_file=str(out_path),
            batch_size=10000,
            verbose=True,
        )
        df = build_raw_docs(cfg)
        elapsed = time.time() - t0
        print(f"[{yr}] OK! {len(df):,} 条 | 耗时 {elapsed/60:.1f} 分钟 | -> {out_path.name}", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{yr}] FAIL! ({elapsed/60:.1f}分钟后) {e}", flush=True)
        traceback.print_exc()
        # 清理残文件
        if out_path.exists():
            out_path.unlink()
        print(f"[{yr}] 残文件已清理，稍后可重跑", flush=True)

# 汇总
print(f"\n{'='*50}", flush=True)
print("汇总所有年份:", flush=True)
total = 0
for yr in YEARS:
    f = OUT_DIR / f"raw_docs_news_{yr}.parquet"
    if f.exists():
        n = len(pd.read_parquet(f))
        print(f"  {yr}: {n:,} 条", flush=True)
        total += n
    else:
        print(f"  {yr}: ❌ 缺失", flush=True)
print(f"总计: {total:,} 条", flush=True)
