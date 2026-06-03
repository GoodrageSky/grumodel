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
import json
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import embed_finbert_bertopic as t2
from config import PATHS, SEGMENTS, STAGE2
from meta import write_meta


def _embedding_cache_paths(text_col: str) -> tuple[Path, Path]:
    model = STAGE2["embed_model"].replace("/", "_").replace("\\", "_")
    cache_dir = PATHS["panel"].parent / "embeddings"
    stem = f"{text_col}_{model}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _daily_snapshot() -> dict:
    st = PATHS["daily"].stat()
    return {"daily_mtime": st.st_mtime, "daily_size": st.st_size}


def _load_embedding_cache(daily: pd.DataFrame, text_col: str) -> np.ndarray | None:
    npy_path, meta_path = _embedding_cache_paths(text_col)
    if not npy_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    snap = _daily_snapshot()
    if (
        meta.get("rows") != len(daily)
        or meta.get("text_col") != text_col
        or meta.get("embed_model") != STAGE2["embed_model"]
        or meta.get("index_start") != str(daily.index[0])
        or meta.get("index_end") != str(daily.index[-1])
        or meta.get("daily_mtime") != snap["daily_mtime"]
        or meta.get("daily_size") != snap["daily_size"]
    ):
        print("[stage2] embedding cache stale; recomputing ...", flush=True)
        return None
    print(f"[stage2] loading embedding cache: {npy_path}", flush=True)
    return np.load(npy_path)


def _write_embedding_cache(emb_all: np.ndarray, daily: pd.DataFrame, text_col: str) -> None:
    npy_path, meta_path = _embedding_cache_paths(text_col)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[stage2] writing embedding cache: {npy_path}", flush=True)
    np.save(npy_path, emb_all)
    meta = {
        "rows": len(daily),
        "text_col": text_col,
        "embed_model": STAGE2["embed_model"],
        "index_start": str(daily.index[0]),
        "index_end": str(daily.index[-1]),
        "shape": list(emb_all.shape),
        **_daily_snapshot(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(text_col: str = "summary", use_bertopic: bool | None = None) -> pd.DataFrame:
    print(f"[stage2] loading daily panel: {PATHS['daily']}", flush=True)
    daily = pd.read_parquet(PATHS["daily"])
    print(
        f"[stage2] daily rows={len(daily)} text_col={text_col!r}",
        flush=True,
    )

    emb_all = _load_embedding_cache(daily, text_col)
    if emb_all is None:
        print(f"[stage2] embedding text with {STAGE2['embed_model']} ...", flush=True)
        embedder = t2.FinBertEmbedder(t2.EmbedConfig(model_name=STAGE2["embed_model"]))
        emb_all = t2.embed_panel(daily, embedder, text_col=text_col)
        _write_embedding_cache(emb_all, daily, text_col)
    print(f"[stage2] embeddings shape={emb_all.shape}", flush=True)
    seg_masks = t2.split_by_segments(daily, SEGMENTS)

    print(f"[stage2] loading labels: {PATHS['label_src']}", flush=True)
    src = (pd.read_parquet(PATHS["label_src"])
             .drop_duplicates(["date", "stkcd"])
             .set_index(["date", "stkcd"]))
    label = src["vwap_rk"]
    ret20 = src["vwap_ret20"]
    y_all = label.reindex(daily.index).to_numpy()

    red_cfg = t2.ReduceConfig(
        n_components=STAGE2["n_components"],
        n_bins=STAGE2["n_bins"],
        use_bertopic=STAGE2["use_bertopic"] if use_bertopic is None else use_bertopic,
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
    print("[stage2] reducer transform on full panel ...", flush=True)
    reduced = reducer.transform_embeddings(emb_all)
    print(f"[stage2] reduced shape={reduced.shape}", flush=True)
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
    ap.add_argument("--output", type=Path, default=PATHS["panel"],
                    help="panel parquet output path")
    ap.add_argument("--no-bertopic", action="store_true",
                    help="disable BERTopic; keep supervised UMAP only")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    use_bertopic = False if args.no_bertopic else STAGE2["use_bertopic"]
    panel = run(text_col=args.text_col, use_bertopic=use_bertopic)
    print(f"[stage2] writing panel parquet: {args.output}", flush=True)
    panel.to_parquet(args.output)
    write_meta(args.output, stage="stage2",
               config={"text_col": args.text_col, **STAGE2,
                       "use_bertopic": use_bertopic},
               inputs=[str(PATHS["daily"]), str(PATHS["label_src"])])
    nan_ratio = float(panel.isna().mean().mean())
    print(f"panel: shape={panel.shape} | NaN ratio={nan_ratio:.4f} "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
