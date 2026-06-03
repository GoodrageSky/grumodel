"""Stage 1b with DeepSeek summaries.

Input : artifacts/raw_docs.parquet
Output: artifacts/daily.parquet
        artifacts/qc/summary_quality.json

Run from repo root:
  python pipelines/stage1b_deepseek.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import text_daily_summary as t1
from config import PATHS, STAGE1B
from meta import write_meta


def _make_deepseek_summarizer(model: str, timeout: float):
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1",
                    timeout=timeout)

    def _fn(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    return _fn


def _summarize_panel_with_progress(df_daily: pd.DataFrame, summarize_fn,
                                   cfg: t1.SummaryConfig,
                                   text_col: str = "full_text") -> pd.DataFrame:
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    texts = df_daily[text_col].fillna("").tolist()
    keys = [t1._cache_key(t, cfg) if t.strip() else None for t in texts]
    results: list[str | None] = [None] * len(texts)
    todo: list[int] = []

    print(f"[summary] scanning cache: {cache}", flush=True)
    for i, (t, k) in enumerate(zip(texts, keys)):
        if not t.strip():
            results[i] = ""
            continue
        p = cache / f"{k}.json"
        if p.exists():
            results[i] = json.loads(p.read_text(encoding="utf-8"))["summary"]
        else:
            todo.append(i)

    cached = sum(r is not None for r in results)
    print(
        f"[summary] total={len(texts)} cached_or_empty={cached} "
        f"todo_api={len(todo)} max_workers={cfg.max_workers}",
        flush=True,
    )

    def _work(i: int):
        for attempt in range(cfg.max_retries):
            try:
                s = t1._summarize_one(texts[i], summarize_fn, cfg)
                (cache / f"{keys[i]}.json").write_text(
                    json.dumps({"summary": s}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return i, s, None
            except Exception as e:  # noqa: BLE001
                if attempt == cfg.max_retries - 1:
                    return i, f"[SUMMARY_FAILED] {e}", repr(e)
                wait = cfg.retry_backoff ** attempt
                time.sleep(wait)

    if todo:
        print("[summary] starting API requests ...", flush=True)
        started = time.time()
        failures = 0
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(_work, i) for i in todo]
            done = 0
            for f in as_completed(futs):
                i, s, err = f.result()
                results[i] = s
                done += 1
                if err is not None:
                    failures += 1
                    print(f"[summary] failed row={i}: {err}", flush=True)
                if done == 1 or done == len(todo) or done % 25 == 0:
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-9)
                    remaining = (len(todo) - done) / max(rate, 1e-9)
                    print(
                        f"[summary] completed {done}/{len(todo)} "
                        f"({done / len(todo):.1%}) failures={failures} "
                        f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                        flush=True,
                    )

    out = df_daily.copy()
    out["summary"] = results
    out["summary_chars"] = [t1._char_len_cn(s or "") for s in results]
    return out


def run(run_qc: bool = True, max_workers: int = 8,
        timeout: float = 60.0) -> tuple[pd.DataFrame, dict | None]:
    print(f"[stage1b_deepseek] loading raw docs: {PATHS['raw_docs']}", flush=True)
    raw = pd.read_parquet(PATHS["raw_docs"])
    print(f"[stage1b_deepseek] raw rows={len(raw)}", flush=True)

    print("[stage1b_deepseek] aggregating by (date, stkcd) ...", flush=True)
    daily = t1.aggregate_daily(raw, t1.AggregateConfig(max_chars=STAGE1B["max_chars"]))
    print(
        f"[stage1b_deepseek] aggregated rows={len(daily)} "
        f"dates={daily.index.get_level_values(0).nunique()} "
        f"stocks={daily.index.get_level_values(1).nunique()}",
        flush=True,
    )

    model = STAGE1B.get("deepseek_model", "deepseek-chat")
    print(
        f"[stage1b_deepseek] initializing DeepSeek model={model} "
        f"timeout={timeout}s",
        flush=True,
    )
    summarize_fn = _make_deepseek_summarizer(model=model, timeout=timeout)
    sum_cfg = t1.SummaryConfig(
        target_min=STAGE1B["target_min"],
        target_max=STAGE1B["target_max"],
        cache_dir=str(PATHS["summary_cache"]),
        model_name_for_cache=f"deepseek:{model}",
        max_workers=max_workers,
    )
    daily = _summarize_panel_with_progress(
        daily, summarize_fn, cfg=sum_cfg, text_col="full_text")
    failed = int(daily["summary"].fillna("").str.startswith("[SUMMARY_FAILED]").sum())
    print(f"[stage1b_deepseek] summary done; failed={failed}", flush=True)

    qc_serializable: dict | None = None
    if run_qc:
        print("[stage1b_deepseek] running full_text vs summary QC ...", flush=True)
        from embed_finbert_bertopic import EmbedConfig, FinBertEmbedder

        embedder = FinBertEmbedder(EmbedConfig(model_name=STAGE1B["qc_embed_model"]))
        qc = t1.summary_similarity(
            daily, embedder.encode,
            low_threshold=STAGE1B["qc_low_threshold"],
        )
        qc_serializable = {k: v for k, v in qc.items()
                           if not isinstance(v, pd.DataFrame)}
        qc_serializable["n_flagged"] = int(len(qc.get("flagged", [])))
        print(
            f"[stage1b_deepseek] QC done; cos_median="
            f"{qc_serializable.get('cos_median', float('nan')):.3f} "
            f"n_flagged={qc_serializable['n_flagged']}",
            flush=True,
        )

    return daily, qc_serializable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-qc", action="store_true", help="skip QC similarity check")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="concurrent DeepSeek requests")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in seconds")
    args = ap.parse_args()
    
    print(111)

    PATHS["daily"].parent.mkdir(parents=True, exist_ok=True)
    PATHS["qc_dir"].mkdir(parents=True, exist_ok=True)

    daily, qc = run(run_qc=not args.no_qc, max_workers=args.max_workers,
                    timeout=args.timeout)
    print(f"[stage1b_deepseek] writing daily parquet: {PATHS['daily']}", flush=True)
    daily.to_parquet(PATHS["daily"])
    if qc is not None:
        print(
            f"[stage1b_deepseek] writing QC json: "
            f"{PATHS['qc_dir'] / 'summary_quality.json'}",
            flush=True,
        )
        (PATHS["qc_dir"] / "summary_quality.json").write_text(
            json.dumps(qc, indent=2, default=float), encoding="utf-8")
    
    print(222)

    model = STAGE1B.get("deepseek_model", "deepseek-chat")
    write_meta(
        PATHS["daily"],
        stage="stage1b_deepseek",
        config={"provider": "deepseek", "deepseek_model": model,
                "run_qc": not args.no_qc, **STAGE1B},
        inputs=[str(PATHS["raw_docs"])],
    )

    print(333)

    msg = (f"daily: rows={len(daily)} | "
           f"summary_chars_median={daily['summary_chars'].median():.0f} "
           f"-> {PATHS['daily']} | provider=deepseek | model={model}")
    if qc is not None:
        msg += (f" | cos_median={qc['cos_median']:.3f} "
                f"| pct_below={qc['pct_below_threshold']:.2%}")
    print(msg)


if __name__ == "__main__":
    main()
