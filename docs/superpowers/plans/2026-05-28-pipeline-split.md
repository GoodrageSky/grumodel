# Pipeline Split (ipynb → 4 scripts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert [strategy_pipeline.ipynb](../../../strategy_pipeline.ipynb) into 4 standalone Python scripts (+ a config, a meta helper, and a compare utility) so each pipeline stage can be run, debugged, and re-run independently with intermediate artifacts persisted under `./artifacts/`.

**Architecture:** Each stage is a thin CLI wrapper around the existing modules (`text_daily_summary.py`, `embed_finbert_bertopic.py`, `gru_lgbm_factor.py`, `load_real_docs.py`, `synth_research_reports.py`) — **no changes** to those modules. A single `config.py` holds paths and per-stage hyperparameters; all scripts `from config import ...`. A tiny `_meta.py` helper writes `<output>.meta.json` sidecars for reproducibility.

**Tech Stack:** Python 3.13 (already on this machine), pandas, the existing `transformers / torch / lightgbm / umap-learn / bertopic / anthropic / pymysql` stack already used by the notebook.

**Note (no git):** The working directory is not a git repository (`Is a git repository: false`). Each task ends with a smoke-verification step instead of a `git commit`. If you `git init` first, append a commit step manually.

**Spec:** [docs/superpowers/specs/2026-05-28-pipeline-split-design.md](../specs/2026-05-28-pipeline-split-design.md)

---

## File Structure

**Create (7 files):**

| Path | Responsibility |
|---|---|
| `config.py` | Single source of truth: `PATHS`, `SEGMENTS`, `STAGE1A/1B/2/3` dicts |
| `_meta.py` | `write_meta(output_path, ...)` → JSON sidecar with config snapshot, input mtimes, optional git hash |
| `stage1a_load_raw_docs.py` | Pull/synth/load raw docs → `artifacts/raw_docs.parquet` |
| `stage1b_summarize.py` | Aggregate + summarize + QC → `artifacts/daily.parquet` + `artifacts/qc/summary_quality.json` |
| `stage2_embed_reduce.py` | FinBERT embed + supervised reduce → `artifacts/panel.parquet`. Exposes `run(text_col)` for import. |
| `stage3_factor.py` | GRU + LGBM factor → `artifacts/stage3/<tag>/`. Accepts `--input`, exposes `run(panel)` for import. |
| `compare_fulltext_vs_summary.py` | Calls `stage2.run()` + `stage3.run()` twice (one per text_col), outputs CSV |

**Do not touch:** `text_daily_summary.py`, `embed_finbert_bertopic.py`, `gru_lgbm_factor.py`, `load_real_docs.py`, `synth_research_reports.py`, `dt_lgbm.parquet`.

---

## Task 1: `config.py`

**Files:**
- Create: `config.py`

- [ ] **Step 1: Create `config.py` with the full config**

```python
"""Shared configuration for stage1a / stage1b / stage2 / stage3 + compare script.

All scripts `from config import PATHS, SEGMENTS, STAGE1A, ...`.
Edit values here; do NOT scatter parameters across stage scripts.
"""
from pathlib import Path

ARTIFACTS = Path("artifacts")

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
    label_src     = Path("dt_lgbm.parquet"),       # truth source for vwap_rk / vwap_ret20
    qc_dir        = ARTIFACTS / "qc",
    stage3_dir    = ARTIFACTS / "stage3",
    summary_cache = Path("cache_summaries"),       # reused from the notebook
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
    llm_model="claude-sonnet-4-5",
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
    label_col="vwap_rk",
    return_col="vwap_ret20",
    feature_prefix="emb_",
    proj_dim=128,
    gru_hidden=64, gru_layers=2,
    epochs=10, batch_size=128, patience=3,
    lgbm_num_boost_round=500, lgbm_early_stopping=30,
    oof_gru_features=False,
    device="auto",
)
```

- [ ] **Step 2: Verify config imports cleanly**

Run: `python -c "from config import PATHS, SEGMENTS, STAGE1A, STAGE1B, STAGE2, STAGE3; print(list(PATHS), list(SEGMENTS))"`

Expected stdout:
```
['raw_docs', 'daily', 'panel', 'label_src', 'qc_dir', 'stage3_dir', 'summary_cache'] ['train', 'valid', 'test']
```

---

## Task 2: `_meta.py` (sidecar writer)

**Files:**
- Create: `_meta.py`

- [ ] **Step 1: Create `_meta.py`**

```python
"""Write <output>.meta.json sidecar files for stage outputs.

Captures: stage name, timestamp, output path, input file mtimes/sizes,
config snapshot, and (if available) git HEAD. Used by every stage script.
"""
from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _input_info(paths: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if path.exists():
            st = path.stat()
            out.append({"path": str(path), "mtime": st.st_mtime, "size": st.st_size})
        else:
            out.append({"path": str(path), "mtime": None, "size": None})
    return out


def write_meta(output_path: Path | str, *, stage: str,
               config: dict, inputs: list[str]) -> Path:
    """Write `<output_path>.meta.json` next to `output_path`. Returns the meta path."""
    output_path = Path(output_path)
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    meta = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "input_paths": _input_info(inputs),
        "config": config,
        "git_hash": _git_hash(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta_path


if __name__ == "__main__":
    # smoke test — `python _meta.py`
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.parquet"
        out.write_bytes(b"hi")
        m = write_meta(out, stage="test", config={"a": 1, "b": "ok"}, inputs=[str(out)])
        loaded = json.loads(m.read_text(encoding="utf-8"))
        assert loaded["stage"] == "test"
        assert loaded["config"] == {"a": 1, "b": "ok"}
        assert loaded["input_paths"][0]["size"] == 2
        print("[_meta.py] smoke ok ->", m)
```

- [ ] **Step 2: Run the smoke test**

Run: `python _meta.py`

Expected stdout (last line):
```
[_meta.py] smoke ok -> <tempdir>/x.parquet.meta.json
```

---

## Task 3: `stage1a_load_raw_docs.py`

**Files:**
- Create: `stage1a_load_raw_docs.py`

- [ ] **Step 1: Create `stage1a_load_raw_docs.py`**

```python
"""Stage 1a: Load raw research-report documents.

Sources:
  real  - pull from aliyun reportdata.report_info via load_real_docs
  synth - generate offline via synth_research_reports
  file  - read an existing parquet (--file PATH required)

Output: artifacts/raw_docs.parquet (+ .meta.json sidecar)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

import load_real_docs as L
import synth_research_reports as sr
from config import PATHS, STAGE1A
from _meta import write_meta


def run(source: str, file_path: Path | None = None) -> pd.DataFrame:
    cfg = STAGE1A
    if source == "real":
        load_cfg = L.LoadConfig(
            start=cfg["start"], end=cfg["end"],
            batch_months=cfg["batch_months"],
            use_content_fallback=cfg["use_content_fallback"],
            max_text_chars=cfg["max_text_chars"],
            min_text_chars=cfg["min_text_chars"],
        )
        raw = L.load_real_docs(load_cfg)
        if cfg.get("snap_to_panel"):
            panel_keys = pd.read_parquet(PATHS["label_src"])[["date", "stkcd"]]
            raw = L.snap_to_panel_dates(raw, panel_keys,
                                        max_forward_days=cfg["max_forward_days"])
    elif source == "synth":
        all_keys = pd.read_parquet(PATHS["label_src"])[["date", "stkcd"]].drop_duplicates()
        stk_pool = all_keys["stkcd"].drop_duplicates().sample(8, random_state=42).tolist()
        keys = all_keys[all_keys["stkcd"].isin(stk_pool)].reset_index(drop=True)
        gen = sr.make_offline_generator()
        raw = sr.generate_panel(keys, gen,
                                cfg=sr.SynthConfig(docs_per_day=2,
                                                   sources=("research", "news")))
    elif source == "file":
        if file_path is None:
            raise ValueError("--file PATH is required when --source=file")
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
    raw.to_parquet(PATHS["raw_docs"])

    inputs = ([str(PATHS["label_src"])] if args.source != "file"
              else [str(args.file)])
    write_meta(PATHS["raw_docs"], stage="stage1a",
               config={"source": args.source, **STAGE1A}, inputs=inputs)

    print(f"raw_docs: rows={len(raw)} | dates={raw['date'].nunique()} | "
          f"stocks={raw['stkcd'].nunique()} -> {PATHS['raw_docs']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with synth source (no DB needed)**

Run: `python stage1a_load_raw_docs.py --source synth`

Expected: stdout line `raw_docs: rows=<N> | dates=<D> | stocks=8 -> artifacts\raw_docs.parquet`
and files exist:
```
artifacts/raw_docs.parquet
artifacts/raw_docs.parquet.meta.json
```

Verify with: `python -c "import pandas as pd; df=pd.read_parquet('artifacts/raw_docs.parquet'); print(df.shape, list(df.columns))"`

Expected columns include `date`, `stkcd`, plus the text/source columns produced by `synth_research_reports.generate_panel` (per [synth_research_reports.py](../../../synth_research_reports.py)).

---

## Task 4: `stage1b_summarize.py`

**Files:**
- Create: `stage1b_summarize.py`

- [ ] **Step 1: Create `stage1b_summarize.py`**

```python
"""Stage 1b: Aggregate raw_docs into daily panel, LLM-summarize, run QC.

Input : artifacts/raw_docs.parquet
Output: artifacts/daily.parquet
        artifacts/qc/summary_quality.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

import text_daily_summary as t1
from config import PATHS, STAGE1B
from _meta import write_meta


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
        summarize_fn = t1.make_anthropic_summarizer(model=STAGE1B["llm_model"])
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
        # drop the heavy "flagged" DataFrame for JSON; keep its row-count
        qc_serializable = {k: v for k, v in qc.items() if k != "flagged"}
        qc_serializable["n_flagged"] = int(len(qc["flagged"]))

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
```

- [ ] **Step 2: Smoke-run (offline summarizer, skip QC to keep it fast)**

Run: `python stage1b_summarize.py --no-qc`

Expected: stdout line `daily: rows=<N> | summary_chars_median=<M> -> artifacts\daily.parquet`
and file exists: `artifacts/daily.parquet`

Verify with: `python -c "import pandas as pd; d=pd.read_parquet('artifacts/daily.parquet'); print(d.shape, list(d.columns))"`

Expected columns: `full_text, n_docs, n_chars, summary, summary_chars`

- [ ] **Step 3: Re-run WITH QC (FinBERT will download once on first run)**

Run: `python stage1b_summarize.py`

Expected stdout includes `cos_median=<X> | pct_below=<Y>` and file exists:
`artifacts/qc/summary_quality.json`

---

## Task 5: `stage2_embed_reduce.py`

**Files:**
- Create: `stage2_embed_reduce.py`

- [ ] **Step 1: Create `stage2_embed_reduce.py`**

```python
"""Stage 2: FinBERT embed + supervised UMAP/BERTopic reduce.

Input : artifacts/daily.parquet + dt_lgbm.parquet (labels)
Output: artifacts/panel.parquet  (emb_0..emb_{k-1} + vwap_rk + vwap_ret20)

Leakage red line: SupervisedReducer.fit() is called ONLY on rows that fall
in the SEGMENTS['train'] window. Transform happens on the full sample.
The script prints `[stage2] reducer.fit on N_train / transform N_total`
as a visible audit trail.
"""
from __future__ import annotations
import argparse
import pandas as pd

import embed_finbert_bertopic as t2
from config import PATHS, SEGMENTS, STAGE2
from _meta import write_meta


def run(text_col: str = "summary") -> pd.DataFrame:
    daily = pd.read_parquet(PATHS["daily"])

    embedder = t2.FinBertEmbedder(t2.EmbedConfig(model_name=STAGE2["embed_model"]))
    emb_all = t2.embed_panel(daily, embedder, text_col=text_col)
    seg_masks = t2.split_by_segments(daily, SEGMENTS)

    src = (pd.read_parquet(PATHS["label_src"])
             .drop_duplicates(["date", "stkcd"])
             .set_index(["date", "stkcd"]))
    label = src["vwap_rk"]
    ret20 = src["vwap_ret20"]
    y_all = label.reindex(daily.index).to_numpy()

    red_cfg = t2.ReduceConfig(
        n_components=STAGE2["n_components"],
        n_bins=STAGE2["n_bins"],
        use_bertopic=STAGE2["use_bertopic"],
        random_state=STAGE2["random_state"],
    )
    reducer = t2.SupervisedReducer(red_cfg)
    n_train = int(seg_masks["train"].sum())
    n_total = len(daily)
    print(f"[stage2] reducer.fit on {n_train} train rows / "
          f"transform {n_total} total rows")
    reducer.fit(
        embeddings_train=emb_all[seg_masks["train"]],
        y_train=y_all[seg_masks["train"]],
        docs_train=daily[text_col][seg_masks["train"]].tolist(),
    )
    reduced = reducer.transform_embeddings(emb_all)
    panel = t2.build_feature_panel(
        daily, reduced,
        label_series=label.reindex(daily.index).rename("vwap_rk"),
        return_series=ret20.reindex(daily.index).rename("vwap_ret20"),
    )
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-col", choices=("summary", "full_text"),
                    default=STAGE2["text_col"])
    args = ap.parse_args()

    PATHS["panel"].parent.mkdir(parents=True, exist_ok=True)
    panel = run(text_col=args.text_col)
    panel.to_parquet(PATHS["panel"])
    write_meta(PATHS["panel"], stage="stage2",
               config={"text_col": args.text_col, **STAGE2},
               inputs=[str(PATHS["daily"]), str(PATHS["label_src"])])
    nan_ratio = float(panel.isna().mean().mean())
    print(f"panel: shape={panel.shape} | NaN ratio={nan_ratio:.4f} "
          f"-> {PATHS['panel']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with the synth-produced `daily.parquet` from Task 4**

Run: `python stage2_embed_reduce.py`

Expected stdout (first non-warning line):
```
[stage2] reducer.fit on <N_train> train rows / transform <N_total> total rows
```
Final line:
```
panel: shape=(<N>, 12) | NaN ratio=<X> -> artifacts\panel.parquet
```

Verify columns are right with: `python -c "import pandas as pd; p=pd.read_parquet('artifacts/panel.parquet'); print(p.shape, list(p.columns))"`

Expected columns: `emb_0, emb_1, ..., emb_9, vwap_rk, vwap_ret20` (12 cols total).

---

## Task 6: `stage3_factor.py`

**Files:**
- Create: `stage3_factor.py`

- [ ] **Step 1: Create `stage3_factor.py`**

```python
"""Stage 3: GRU + LGBM factor pipeline.

Input : artifacts/panel.parquet  (or any panel parquet via --input)
Output: artifacts/stage3/<tag>/factor_test.csv
        artifacts/stage3/<tag>/metrics.json
        artifacts/stage3/<tag>/metrics.json.meta.json

Quick path (skip Stage 1 & 2):
  python stage3_factor.py --input dt_lgbm.parquet --tag fast
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

import gru_lgbm_factor as t3
from config import PATHS, SEGMENTS, STAGE3
from _meta import write_meta


def run(panel: pd.DataFrame) -> dict:
    cfg = t3.FactorConfig(**STAGE3)
    return t3.run_pipeline(panel, SEGMENTS, cfg)


def _metrics_to_json(res: dict) -> dict:
    """Coerce numpy scalars to plain floats so json.dump won't choke."""
    out: dict = {}
    for split, m in res["metrics"].items():
        out[split] = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, str)
                          else v)
                      for k, v in m.items()}
    bt = res.get("backtest")
    if bt is not None:
        out["test_backtest"] = {
            "ls_mean": float(bt["ls_mean"]),
            "ls_sharpe": float(bt["ls_sharpe"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=PATHS["panel"],
                    help="panel parquet path")
    ap.add_argument("--tag", default="default",
                    help="output sub-directory under artifacts/stage3/")
    args = ap.parse_args()

    out_dir = PATHS["stage3_dir"] / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.input)
    res = run(panel)

    # 1) factor (test slice) -> CSV
    test_factor = res["preds"]["test"].rename("gru_lgbm_factor").reset_index()
    test_factor.to_csv(out_dir / "factor_test.csv", index=False)

    # 2) metrics -> JSON
    metrics = _metrics_to_json(res)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")

    # 3) meta sidecar (pin which panel + config produced these numbers)
    write_meta(metrics_path, stage="stage3",
               config={"tag": args.tag, "input": str(args.input), **STAGE3},
               inputs=[str(args.input)])

    # 4) human-readable summary
    print(pd.DataFrame(res["metrics"]).T[["IC_mean", "RankIC_mean", "ICIR", "n_days"]])
    if res.get("backtest") is not None:
        print(f"test L/S mean={res['backtest']['ls_mean']:.4f}  "
              f"ann.Sharpe={res['backtest']['ls_sharpe']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run via the "quick path" against the existing `dt_lgbm.parquet`**

Run: `python stage3_factor.py --input dt_lgbm.parquet --tag fast`

Expected: training/validation/test logs from `gru_lgbm_factor.run_pipeline` followed by an IC table and an L/S line, plus files:
```
artifacts/stage3/fast/factor_test.csv
artifacts/stage3/fast/metrics.json
artifacts/stage3/fast/metrics.json.meta.json
```

Verify metrics shape: `python -c "import json; print(list(json.load(open('artifacts/stage3/fast/metrics.json'))))"`

Expected keys include `train, valid, test` and optionally `test_backtest`.

- [ ] **Step 3: Smoke-run against the Stage 2 output too**

Run: `python stage3_factor.py --tag from_synth`

Expected: same shape of output under `artifacts/stage3/from_synth/`.

---

## Task 7: `compare_fulltext_vs_summary.py`

**Files:**
- Create: `compare_fulltext_vs_summary.py`

- [ ] **Step 1: Create `compare_fulltext_vs_summary.py`**

```python
"""Compare full_text vs summary as the input text column.

Runs Stage 2 + Stage 3 twice (text_col='full_text' then 'summary') and
writes a side-by-side test-set IC table. The point of this script is to
prove the 300-500-char summary does NOT lose alpha vs raw full_text.

Input : artifacts/daily.parquet  (run stage1b first)
Output: artifacts/compare_text_summary.csv
"""
from __future__ import annotations
import argparse
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
        panel = s2.run(text_col=col)
        rows[col] = s3.run(panel)["metrics"]["test"]

    df = pd.DataFrame(rows).T[["IC_mean", "RankIC_mean", "ICIR"]]
    out = PATHS["raw_docs"].parent / "compare_text_summary.csv"   # artifacts/
    df.to_csv(out)
    print("\n", df)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (requires `artifacts/daily.parquet` from Task 4)**

Run: `python compare_fulltext_vs_summary.py`

Expected: two `[stage2] reducer.fit ...` blocks, two GRU/LGBM training logs,
then a 2-row DataFrame:
```
            IC_mean  RankIC_mean      ICIR
full_text   ...      ...              ...
summary     ...      ...              ...
```
and file: `artifacts/compare_text_summary.csv`

---

## Task 8: End-to-end smoke + cleanup

- [ ] **Step 1: Wipe artifacts and re-run the full chain from scratch with `synth` source**

Run (PowerShell):
```powershell
Remove-Item -Recurse -Force artifacts -ErrorAction SilentlyContinue
python stage1a_load_raw_docs.py --source synth
python stage1b_summarize.py --no-qc
python stage2_embed_reduce.py
python stage3_factor.py --tag e2e
```

Expected: all four scripts exit 0; final directory:
```
artifacts/
  raw_docs.parquet (+ .meta.json)
  daily.parquet    (+ .meta.json)
  panel.parquet    (+ .meta.json)
  stage3/e2e/
    factor_test.csv
    metrics.json (+ .meta.json)
```

- [ ] **Step 2: Sanity-check meta.json contents**

Run: `python -c "import json; m=json.load(open('artifacts/panel.parquet.meta.json')); print(m['stage'], m['config']['text_col'], len(m['input_paths']))"`

Expected:
```
stage2 summary 2
```

- [ ] **Step 3: Verify the "quick path" still works independently**

Run: `python stage3_factor.py --input dt_lgbm.parquet --tag fast`

Expected: succeeds and writes `artifacts/stage3/fast/`.

---

## Self-Review (run after writing all tasks)

**Spec coverage check** — every spec section maps to at least one task:

| Spec section | Covered by |
|---|---|
| §2 Directory structure | Task 1 (config paths), Tasks 3–7 (the files themselves), Task 8 (verifies layout) |
| §3 `config.py` contract | Task 1 |
| §4 stage1a contract | Task 3 |
| §4 stage1b contract | Task 4 |
| §4 stage2 contract (incl. `run()` for import) | Task 5 |
| §4 stage3 contract (incl. `--input`, `--tag`, `run()`) | Task 6 |
| §5 compare script | Task 7 |
| §6 Leakage assertion | Task 5 Step 1 (explicit print line + fit only on train mask) |
| §6 Reproducibility / meta.json | Task 2 (helper), used by Tasks 3–6; Task 8 Step 2 verifies |
| §6 SEGMENTS consistency | Task 1 (single source); Tasks 5, 6 import from config |
| §7 Typical run commands (full / fast / compare / synth) | Tasks 3 (synth), 6 (fast), 7 (compare), 8 (full chain) |

**Placeholder scan:** all code blocks are concrete. No "TBD", no "implement later", no "add appropriate error handling".

**Type/name consistency check:**
- `run(text_col=...)` in Task 5 matches `s2.run(text_col=col)` call in Task 7. ✓
- `run(panel)` in Task 6 matches `s3.run(panel)` call in Task 7. ✓
- `write_meta(output_path, *, stage, config, inputs)` signature in Task 2 matches every call site in Tasks 3, 4, 5, 6. ✓
- `PATHS` keys used (`raw_docs`, `daily`, `panel`, `label_src`, `qc_dir`, `stage3_dir`, `summary_cache`) all defined in Task 1. ✓
- `STAGE3` dict keys map 1:1 to `t3.FactorConfig` constructor (per existing notebook usage of `FactorConfig(seq_len=..., label_horizon=..., etc.)`). ✓
- `STAGE2` keys map to `ReduceConfig` and `EmbedConfig` constructors as used in the notebook. ✓
