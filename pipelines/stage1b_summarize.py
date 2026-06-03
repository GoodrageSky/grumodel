"""Stage 1b: Aggregate raw_docs into daily panel, LLM-summarize, run QC.

Input : artifacts/raw_docs.parquet
Output: artifacts/daily.parquet
        artifacts/qc/summary_quality.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import text_daily_summary as t1
from config import PATHS, STAGE1B
from meta import write_meta


def _offline_summarize(text: str) -> str:
    """Tiny truncate-only fallback used when --use-llm is not passed."""
    t = " ".join(text.split())
    cap = STAGE1B["target_max"]
    return t[:cap] if len(t) > cap else t


def run(use_llm: bool, run_qc: bool = True
        ) -> tuple[pd.DataFrame, dict | None]:
    raw = pd.read_parquet(PATHS["raw_docs"])
    daily = t1.aggregate_daily(raw, t1.AggregateConfig(max_chars=STAGE1B["max_chars"]))

    if use_llm:
        summarize_fn = t1.make_deepseek_summarizer(model=STAGE1B["llm_model"])
    else:
        summarize_fn = _offline_summarize

    sum_cfg = t1.SummaryConfig(
        target_min=STAGE1B["target_min"],
        target_max=STAGE1B["target_max"],
        cache_dir=str(PATHS["summary_cache"]),
    )
    daily = t1.summarize_panel(daily, summarize_fn, cfg=sum_cfg, text_col="full_text")

    qc_serializable: dict | None = None
    if run_qc:
        # local import: only need FinBERT when QC is on
        from embed_finbert_bertopic import FinBertEmbedder, EmbedConfig
        embedder = FinBertEmbedder(EmbedConfig(model_name=STAGE1B["qc_embed_model"]))
        qc = t1.summary_similarity(daily, embedder.encode,
                                   low_threshold=STAGE1B["qc_low_threshold"])
        # `summary_similarity` returns DataFrames for `flagged` and `per_row`;
        # strip those for JSON, keep counts.
        qc_serializable = {k: v for k, v in qc.items()
                           if not isinstance(v, pd.DataFrame)}
        qc_serializable["n_flagged"] = int(len(qc.get("flagged", [])))

    return daily, qc_serializable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-llm", action="store_true",
                    help="use real Anthropic LLM (default: offline truncate)")
    ap.add_argument("--no-qc", action="store_true", help="skip QC similarity check")
    args = ap.parse_args()

    PATHS["daily"].parent.mkdir(parents=True, exist_ok=True)
    PATHS["qc_dir"].mkdir(parents=True, exist_ok=True)

    daily, qc = run(use_llm=args.use_llm, run_qc=not args.no_qc)
    daily.to_parquet(PATHS["daily"])
    if qc is not None:
        (PATHS["qc_dir"] / "summary_quality.json").write_text(
            json.dumps(qc, indent=2, default=float), encoding="utf-8")
    write_meta(PATHS["daily"], stage="stage1b",
               config={"use_llm": args.use_llm, "run_qc": not args.no_qc, **STAGE1B},
               inputs=[str(PATHS["raw_docs"])])

    msg = (f"daily: rows={len(daily)} | "
           f"summary_chars_median={daily['summary_chars'].median():.0f} "
           f"-> {PATHS['daily']}")
    if qc is not None:
        msg += (f" | cos_median={qc['cos_median']:.3f} "
                f"| pct_below={qc['pct_below_threshold']:.2%}")
    print(msg)


if __name__ == "__main__":
    main()
