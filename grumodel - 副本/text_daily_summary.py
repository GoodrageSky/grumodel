"""
text_daily_summary.py
=====================
Stage 1 of the daily-frequency text-factor pipeline.

Goal (your idea #1): raise the text frequency to DAILY, fuse news + research
reports per (trading-day, stock), and produce a 300-500 character LLM summary,
then verify that the summary preserves the signal by comparing it against the
full text.

Pipeline:
    raw_docs[date, stkcd, text, source]
        -> aggregate_daily(...)        # one fused document per (date, stkcd)
        -> summarize_panel(...)        # 300-500 字 LLM summary, cached + concurrent
        -> summary_similarity(...)     # full-text vs summary QC

The LLM caller is pluggable (`summarize_fn`). An Anthropic example is provided,
but any callable `str -> str` works (OpenAI, a local model, etc.). Calls are
cached on disk by content hash so re-runs are cheap and resumable, and run
concurrently with retry. Length is validated against the 300-500 字 window with
one automatic re-ask if the model overshoots/undershoots.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Daily aggregation
# --------------------------------------------------------------------------- #
# Research reports tend to carry more forward-looking signal than newswire, so
# we order sources by informativeness when concatenating and budgeting tokens.
DEFAULT_SOURCE_PRIORITY = {"research": 0, "report": 0, "研报": 0,
                           "announcement": 1, "公告": 1,
                           "news": 2, "新闻": 2}


@dataclass
class AggregateConfig:
    date_col: str = "date"
    stock_col: str = "stkcd"
    text_col: str = "text"
    source_col: Optional[str] = "source"     # set None if you have no source labels
    title_col: Optional[str] = "title"       # used for dedup if present
    max_chars: int = 6000                     # char budget for the fused doc (truncate news first)
    source_priority: dict = field(default_factory=lambda: dict(DEFAULT_SOURCE_PRIORITY))


def _dedup_docs(g: pd.DataFrame, cfg: AggregateConfig) -> pd.DataFrame:
    key = cfg.title_col if (cfg.title_col and cfg.title_col in g.columns) else cfg.text_col
    return g.drop_duplicates(subset=[key])


def aggregate_daily(df_docs: pd.DataFrame, cfg: AggregateConfig = AggregateConfig()) -> pd.DataFrame:
    """Fuse all documents for a (date, stkcd) into a single ordered, deduped,
    length-budgeted text. Returns DataFrame[date, stkcd, full_text, n_docs, n_chars]."""
    df = df_docs.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col]).dt.normalize()
    if cfg.source_col and cfg.source_col in df.columns:
        df["_prio"] = df[cfg.source_col].map(
            lambda s: cfg.source_priority.get(str(s).lower(), 99)).fillna(99)
    else:
        df["_prio"] = 0

    out = []
    for (d, stk), g in df.groupby([cfg.date_col, cfg.stock_col], sort=True):
        g = _dedup_docs(g, cfg).sort_values("_prio")
        sep = "\n\n"
        pieces, used = [], 0
        for _, r in g.iterrows():
            t = str(r[cfg.text_col]).strip()
            if not t:
                continue
            tag = ""
            if cfg.source_col and cfg.source_col in g.columns:
                tag = f"【{r[cfg.source_col]}】"
            piece = f"{tag}{t}"
            sep_cost = len(sep) if pieces else 0
            budget_left = cfg.max_chars - used - sep_cost
            if budget_left <= 0:
                break
            if len(piece) > budget_left:
                piece = piece[:budget_left]
            pieces.append(piece)
            used += len(piece) + sep_cost
            if used >= cfg.max_chars:
                break
        full = "\n\n".join(pieces)
        out.append((d, stk, full, len(g), len(full)))

    res = pd.DataFrame(out, columns=[cfg.date_col, cfg.stock_col, "full_text", "n_docs", "n_chars"])
    return res.set_index([cfg.date_col, cfg.stock_col]).sort_index()


# --------------------------------------------------------------------------- #
# LLM summarization
# --------------------------------------------------------------------------- #
SUMMARY_PROMPT = """你是一名严谨的卖方金融分析师助理。下面是关于某只股票在某一交易日的全部新闻与研报原文。

请用中文撰写一段 {tmin}-{tmax} 字的客观摘要，要求：
1. 只提炼与该公司基本面、经营、行业、事件、市场情绪相关的关键信息；
2. 严格忠于原文，不得编造、不得加入原文没有的数字或结论；
3. 不得包含任何对未来股价或收益的预测性断言；
4. 直接输出摘要正文，不要标题、不要分点、不要前后缀说明。

原文：
\"\"\"
{text}
\"\"\""""


@dataclass
class SummaryConfig:
    target_min: int = 300            # 字 (Chinese characters)
    target_max: int = 500
    cache_dir: str = "./cache_summaries"
    max_workers: int = 8
    max_retries: int = 3
    retry_backoff: float = 2.0
    relength_if_out_of_range: bool = True
    model_name_for_cache: str = "summarizer-v1"   # bump to invalidate cache


def _char_len_cn(s: str) -> int:
    """Count characters the way '字' is usually meant: CJK + visible non-space."""
    return len(re.sub(r"\s", "", s))


def _cache_key(text: str, cfg: SummaryConfig) -> str:
    h = hashlib.sha1()
    h.update(cfg.model_name_for_cache.encode())
    h.update(f"{cfg.target_min}-{cfg.target_max}".encode())
    h.update(text.encode("utf-8", "ignore"))
    return h.hexdigest()


def make_anthropic_summarizer(model: str = "claude-sonnet-4-6", max_tokens: int = 800) -> Callable[[str], str]:
    """Return a `str -> str` summarizer backed by the Anthropic API.
    Requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment.
    Swap this out for OpenAI / a local model by writing your own callable."""
    import anthropic
    client = anthropic.Anthropic()

    def _fn(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    return _fn


def make_deepseek_summarizer(model: str = "deepseek-chat",
                             max_tokens: int = 800,
                             api_key: Optional[str] = None,
                             base_url: str = "https://api.deepseek.com/v1") -> Callable[[str], str]:
    """Return a `str -> str` summarizer backed by DeepSeek (OpenAI-compatible API).

    `pip install openai`. Reads DEEPSEEK_API_KEY from env if `api_key` not given.
    Models:
      - "deepseek-chat"     (V3, fast + cheap; default — right choice for summaries)
      - "deepseek-reasoner" (R1, slow + pricier; overkill for fixed-length summaries)
    """
    import os
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
                    base_url=base_url)

    def _fn(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    return _fn


def _summarize_one(text: str, summarize_fn: Callable[[str], str], cfg: SummaryConfig) -> str:
    prompt = SUMMARY_PROMPT.format(tmin=cfg.target_min, tmax=cfg.target_max, text=text)
    summary = summarize_fn(prompt)
    if cfg.relength_if_out_of_range:
        n = _char_len_cn(summary)
        if n < cfg.target_min or n > cfg.target_max:
            fix = (f"上面的摘要为 {n} 字，请改写为 {cfg.target_min}-{cfg.target_max} 字之间，"
                   f"内容要求不变，仅调整长度：\n\n{summary}")
            try:
                summary = summarize_fn(fix)
            except Exception:
                pass
    return summary.strip()


def summarize_panel(df_daily: pd.DataFrame,
                    summarize_fn: Callable[[str], str],
                    cfg: SummaryConfig = SummaryConfig(),
                    text_col: str = "full_text") -> pd.DataFrame:
    """Add a `summary` column to the daily panel. Disk-cached by content hash
    (resumable), concurrent, with retry. Empty texts get an empty summary."""
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    texts = df_daily[text_col].fillna("").tolist()
    keys = [_cache_key(t, cfg) if t.strip() else None for t in texts]
    results: list[Optional[str]] = [None] * len(texts)

    # load cache
    todo = []
    for i, (t, k) in enumerate(zip(texts, keys)):
        if not t.strip():
            results[i] = ""
            continue
        p = cache / f"{k}.json"
        if p.exists():
            results[i] = json.loads(p.read_text(encoding="utf-8"))["summary"]
        else:
            todo.append(i)

    def _work(i):
        for attempt in range(cfg.max_retries):
            try:
                s = _summarize_one(texts[i], summarize_fn, cfg)
                (cache / f"{keys[i]}.json").write_text(
                    json.dumps({"summary": s}, ensure_ascii=False), encoding="utf-8")
                return i, s
            except Exception as e:  # noqa: BLE001
                if attempt == cfg.max_retries - 1:
                    return i, f"[SUMMARY_FAILED] {e}"
                time.sleep(cfg.retry_backoff ** attempt)

    if todo:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(_work, i) for i in todo]
            for f in as_completed(futs):
                i, s = f.result()
                results[i] = s

    out = df_daily.copy()
    out["summary"] = results
    out["summary_chars"] = [ _char_len_cn(s or "") for s in results ]
    return out


# --------------------------------------------------------------------------- #
# Full-text vs summary similarity QC (your idea #1, last bullet)
# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def summary_similarity(df: pd.DataFrame,
                       embed_fn: Callable[[list], np.ndarray],
                       full_col: str = "full_text",
                       summary_col: str = "summary",
                       low_threshold: float = 0.80) -> dict:
    """Embed full text and summary with the SAME embedder and compare.
    `embed_fn(list[str]) -> np.ndarray (N, dim)`. Returns per-row cosine
    similarity, compression ratio, and a flagged subset where the summary may
    have dropped signal (cos < low_threshold)."""
    mask = (df[full_col].fillna("").str.strip() != "") & \
           (df[summary_col].fillna("").str.strip() != "")
    sub = df[mask]
    if sub.empty:
        return {"n": 0}

    emb_full = embed_fn(sub[full_col].tolist())
    emb_sum = embed_fn(sub[summary_col].tolist())
    cos = _cosine(np.asarray(emb_full), np.asarray(emb_sum))

    comp = sub[summary_col].str.len().to_numpy() / np.maximum(sub[full_col].str.len().to_numpy(), 1)
    res = sub.copy()
    res["cos_full_summary"] = cos
    res["compression"] = comp
    flagged = res[res["cos_full_summary"] < low_threshold]

    return {
        "n": int(mask.sum()),
        "cos_mean": float(np.mean(cos)),
        "cos_median": float(np.median(cos)),
        "cos_p10": float(np.percentile(cos, 10)),
        "compression_mean": float(np.mean(comp)),
        "pct_below_threshold": float((cos < low_threshold).mean()),
        "per_row": res[["cos_full_summary", "compression"]],
        "flagged": flagged,
    }
