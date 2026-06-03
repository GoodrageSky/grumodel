"""
synth_research_reports.py
=========================
Bridge utility: generate SYNTHETIC research reports / news with an LLM, so
Stage 1 (`text_daily_summary`) can be exercised before real broker text is
plugged in.

The output schema matches what `text_daily_summary.aggregate_daily` consumes:
    columns = [date, stkcd, text, source, title]
so you can swap in real raw_docs later by producing the same columns.

Usage
-----
    import synth_research_reports as sr
    gen = sr.make_anthropic_generator(model="claude-sonnet-4-6")
    raw_docs = sr.generate_panel(
        panel=pd.read_parquet("dt_lgbm.parquet")[["date","stkcd"]],
        generate_fn=gen,
        docs_per_day=2,          # how many synthetic docs per (date, stkcd)
    )
    raw_docs.to_parquet("raw_docs_synth.parquet")

Notes
-----
* Synthetic text is NOT real alpha. Use it only to verify the pipeline runs.
* Calls are disk-cached by (stkcd, date, source, idx) hash so reruns are cheap.
* Generation is concurrent with retry, same pattern as `summarize_panel`.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


SOURCES = ["research", "news"]   # 研报 + 新闻 (announcement 可自行加)

# Prompt is deliberately generic: produces plausible-looking Chinese text in the
# style of a sell-side note / wire story without inventing share-price targets.
PROMPT = """你是一名中文财经写作助手。请为股票代码 {stkcd}（{trade_date} 当日）撰写一条「{src_label}」体裁的{kind}，长度约 {nchars} 字。

要求：
1. 内容围绕公司近期经营、行业景气、政策、订单、产能、技术、管理层变动等任一主题展开（自行选定一个合理主题）；
2. 行文风格与该体裁一致（研报：理性、有数据感；新闻：客观、第三人称叙述）；
3. **不得包含**对未来股价/收益的预测、目标价、评级；不得出现"建议买入/卖出"之类字样；
4. 直接输出正文，不要标题、不要前后缀说明、不要 markdown 标记。"""


@dataclass
class SynthConfig:
    docs_per_day: int = 2                  # number of docs per (date, stkcd)
    sources: tuple = ("research", "news")  # which sources to sample from
    char_range: tuple = (400, 900)         # per-doc target length, 字
    cache_dir: str = "./cache_synth_docs"
    max_workers: int = 8
    max_retries: int = 3
    retry_backoff: float = 2.0
    model_name_for_cache: str = "synth-v1" # bump to invalidate cache
    seed: int = 42


def make_anthropic_generator(model: str = "claude-sonnet-4-6",
                             max_tokens: int = 1200) -> Callable[[str], str]:
    """Same shape as `text_daily_summary.make_anthropic_summarizer`."""
    import anthropic
    client = anthropic.Anthropic()

    def _fn(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    return _fn


def _cache_key(stkcd: str, date: pd.Timestamp, source: str, idx: int,
               cfg: SynthConfig) -> str:
    h = hashlib.sha1()
    h.update(cfg.model_name_for_cache.encode())
    h.update(f"{stkcd}|{date.date().isoformat()}|{source}|{idx}".encode())
    return h.hexdigest()


def _build_prompt(stkcd: str, date: pd.Timestamp, source: str, nchars: int) -> str:
    src_label = {"research": "研报", "news": "新闻", "announcement": "公告"}.get(source, source)
    kind = "研究简评" if source == "research" else "新闻报道"
    return PROMPT.format(stkcd=stkcd, trade_date=date.date().isoformat(),
                         src_label=src_label, kind=kind, nchars=nchars)


def generate_panel(panel: pd.DataFrame,
                   generate_fn: Callable[[str], str],
                   cfg: SynthConfig = SynthConfig(),
                   date_col: str = "date",
                   stock_col: str = "stkcd") -> pd.DataFrame:
    """For every (date, stkcd) in `panel`, synthesize `cfg.docs_per_day` docs.
    Returns DataFrame[date, stkcd, text, source, title]. Disk-cached + concurrent."""
    rng = random.Random(cfg.seed)
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    base = panel[[date_col, stock_col]].drop_duplicates().copy()
    base[date_col] = pd.to_datetime(base[date_col])

    # plan all (row, idx) tasks
    plan = []
    for _, r in base.iterrows():
        for i in range(cfg.docs_per_day):
            src = cfg.sources[i % len(cfg.sources)]
            nchars = rng.randint(*cfg.char_range)
            plan.append((r[date_col], r[stock_col], src, i, nchars))

    results: list[Optional[tuple]] = [None] * len(plan)

    def _work(k):
        date, stk, src, idx, nchars = plan[k]
        key = _cache_key(stk, date, src, idx, cfg)
        cf = cache / f"{key}.json"
        if cf.exists():
            text = json.loads(cf.read_text(encoding="utf-8"))["text"]
        else:
            prompt = _build_prompt(stk, date, src, nchars)
            text = None
            for attempt in range(cfg.max_retries):
                try:
                    text = generate_fn(prompt)
                    cf.write_text(json.dumps({"text": text}, ensure_ascii=False),
                                  encoding="utf-8")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == cfg.max_retries - 1:
                        text = f"[SYNTH_FAILED] {e}"
                    else:
                        time.sleep(cfg.retry_backoff ** attempt)
        title = f"{stk} {date.date().isoformat()} {src} #{idx}"
        return k, (date, stk, text, src, title)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = [ex.submit(_work, k) for k in range(len(plan))]
        for f in as_completed(futs):
            k, row = f.result()
            results[k] = row

    out = pd.DataFrame(results, columns=[date_col, stock_col, "text", "source", "title"])
    return out.sort_values([date_col, stock_col]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tiny offline generator (no API needed) -- useful for CI / a dry run.
# --------------------------------------------------------------------------- #
def make_offline_generator(seed: int = 0) -> Callable[[str], str]:
    """Deterministic, dependency-free generator that returns plausible Chinese
    boilerplate of roughly the requested length. Pure stand-in for dev/CI."""
    rng = random.Random(seed)
    snippets = [
        "公司主营业务保持平稳增长，下游需求结构性分化。",
        "近期行业景气度温和回升，库存周期处于被动去化阶段。",
        "渠道反馈终端动销环比改善，但同比仍有一定压力。",
        "公司持续推进研发投入，新产品线进入小批量送样阶段。",
        "管理层在近期交流中表示将加强成本管控、优化产品结构。",
        "上游原材料价格波动加大，对短期毛利率存在一定扰动。",
        "海外业务收入占比有所提升，汇率因素对报表有阶段性影响。",
        "公司在手订单饱满，新产能爬坡进度符合此前规划。",
        "行业政策维持稳定，重点关注后续配套实施细则落地节奏。",
        "公司治理层面引入新激励方案，覆盖中高层核心骨干。",
    ]

    def _fn(prompt: str) -> str:
        # crude length parse
        import re
        m = re.search(r"约\s*(\d+)\s*字", prompt)
        target = int(m.group(1)) if m else 500
        buf = []
        while len("".join(buf)) < target:
            buf.append(rng.choice(snippets))
        return "".join(buf)[: target + 30]

    return _fn
