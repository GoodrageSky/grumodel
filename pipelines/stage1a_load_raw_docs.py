"""Stage 1a: Load raw research-report documents.

Sources:
  real  - pull from aliyun reportdata.report_info via load_real_docs
  synth - generate offline via synth_research_reports
  file  - read an existing parquet (--file PATH required)

Output: artifacts/raw_docs.parquet (+ .meta.json sidecar)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import load_real_docs as L
import synth_research_reports as sr
from config import PATHS, STAGE1A
from meta import write_meta


def run(source: str, file_path: Path | None = None) -> pd.DataFrame:
    cfg = STAGE1A
    print(f"[stage1a] source={source}", flush=True)
    if source == "real":
        print("[stage1a] loading real research reports ...", flush=True)
        load_cfg = L.LoadConfig(
            start=cfg["start"], end=cfg["end"],
            batch_months=cfg["batch_months"],
            use_content_fallback=cfg["use_content_fallback"],
            max_text_chars=cfg["max_text_chars"],
            min_text_chars=cfg["min_text_chars"],
        )
        raw = L.load_real_docs(load_cfg)
        if cfg.get("snap_to_panel"):
            print(f"[stage1a] snapping report dates to panel dates: {PATHS['label_src']}", flush=True)
            panel_keys = pd.read_parquet(PATHS["label_src"])[["date", "stkcd"]]
            raw = L.snap_to_panel_dates(raw, panel_keys,
                                        max_forward_days=cfg["max_forward_days"])
    elif source == "synth":
        print("[stage1a] generating synthetic reports ...", flush=True)
        all_keys = pd.read_parquet(PATHS["label_src"])[["date", "stkcd"]].drop_duplicates()
        stk_pool = all_keys["stkcd"].drop_duplicates().sample(8, random_state=42).tolist()
        keys = all_keys[all_keys["stkcd"].isin(stk_pool)].reset_index(drop=True)
        gen = sr.make_offline_generator()
        raw = sr.generate_panel(keys, gen,
                                cfg=sr.SynthConfig(docs_per_day=2,
                                                   sources=("research", "news"),
                                                   cache_dir=str(PATHS["synth_cache"])))
    elif source == "file":
        if file_path is None:
            raise ValueError("--file PATH is required when --source=file")
        print(f"[stage1a] reading file: {file_path}", flush=True)
        raw = pd.read_parquet(file_path)
    else:
        raise ValueError(f"unknown source: {source!r}")
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("real", "synth", "file"),
                    default=STAGE1A["source"])
    ap.add_argument("--file", type=Path, default=None,
                    help="parquet path (required when --source=file)")
    args = ap.parse_args()

    PATHS["raw_docs"].parent.mkdir(parents=True, exist_ok=True)
    raw = run(args.source, args.file)
    print(f"[stage1a] writing raw docs: {PATHS['raw_docs']}", flush=True)
    raw.to_parquet(PATHS["raw_docs"])

    inputs = ([str(PATHS["label_src"])] if args.source != "file"
              else [str(args.file)])
    write_meta(PATHS["raw_docs"], stage="stage1a",
               config={"source": args.source, **STAGE1A}, inputs=inputs)

    print(f"raw_docs: rows={len(raw)} | dates={raw['date'].nunique()} | "
          f"stocks={raw['stkcd'].nunique()} -> {PATHS['raw_docs']}")


if __name__ == "__main__":
    main()
