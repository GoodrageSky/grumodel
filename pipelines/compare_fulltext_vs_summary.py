"""Compare full_text vs summary as the input text column.

Runs Stage 2 + Stage 3 twice (text_col='full_text' then 'summary') and
writes a side-by-side test-set IC table. The point of this script is to
prove the 300-500-char summary does NOT lose alpha vs raw full_text.

Input : artifacts/daily.parquet  (run stage1b first)
Output: artifacts/compare_text_summary.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from config import PATHS
import stage2_embed_reduce as s2
import stage3_factor as s3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="compare",
                    help="(reserved for future use; currently unused)")
    _ = ap.parse_args()

    rows: dict[str, dict] = {}
    for col in ("full_text", "summary"):
        print(f"\n=== running stage2+stage3 with text_col={col!r} ===")
        print(f"[compare] stage2 start text_col={col!r}", flush=True)
        panel = s2.run(text_col=col)
        print(f"[compare] stage3 start text_col={col!r}", flush=True)
        rows[col] = s3.run(panel)["metrics"]["test"]

    df = pd.DataFrame(rows).T[["IC_mean", "RankIC_mean", "ICIR"]]
    out = PATHS["raw_docs"].parent / "compare_text_summary.csv"   # artifacts/
    print(f"[compare] writing comparison csv: {out}", flush=True)
    df.to_csv(out)
    print("\n", df)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
