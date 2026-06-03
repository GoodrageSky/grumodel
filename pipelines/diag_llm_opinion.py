# -*- coding: utf-8 -*-
"""方向②小样本验证: LLM 从研报【正文】抽结构化观点, 看"观点/评级变化"有无 alpha。

前序已证伪的文本利用: 无监督嵌入、情感水平、情感序列结构 —— 均无对量价的增量。
唯一未穷尽、理论更强的: 研报正文里的【结构化观点变化】(评级上调/下调、目标价
隐含涨幅、盈利预测调整方向)。这是与"情感水平"不同的信息维度。

本脚本【小样本、可控成本】先判路:
  1. 从 RDS 拉一段【公司研究】研报正文 (限定股票池 + 时间窗, 控制调用量);
  2. DeepSeek 抽 JSON: {rating(1-5), rating_change(-1/0/1), implied_upside, earnings_revision};
     带磁盘缓存(按 content 哈希), 可断点续跑;
  3. 构造因子: op_rating(评级水平) / op_change(评级变化) / op_upside(目标价涨幅);
  4. 对齐平台日频价格, 测对未来5日收益(市值中性化残差)的横截面 IC + 多空。

判读: 若 op_change/op_upside 的 OOS IC 显著且优于已证伪的情感水平(test t≈2.4)
      -> LLM 观点抽取这条路成立, 值得全量; 若≈0 -> 正文观点也无增量, 文本到顶。

成本控制: MAX_DOCS 上限 + 只抽 content 前 2500 字 + 缓存。默认抽 2023-2025、
          有平台价格的股, 约数千篇, 估算成本在脚本启动时打印, 需确认再跑。

输入: RDS reportdata.report_info + data/platform/px_daily.pqt + DEEPSEEK_API_KEY
缓存: cache/llm_opinion/<hash>.json
用法: python pipelines/diag_llm_opinion.py [--max-docs N] [--go]
      不带 --go 只做取数+成本估算(dry-run), 带 --go 才真正调 LLM。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import load_real_docs as L  # noqa: E402

PLATFORM = ROOT / "data" / "platform" / "px_daily.pqt"
CACHE_DIR = ROOT / "cache" / "llm_opinion"
DOCS_CACHE = ROOT / "data" / "llm_opinion_docs.parquet"      # 抽取结果(date,stkcd,字段)
RAW_CACHE = ROOT / "data" / "_llm_raw_docs.parquet"          # 原始正文(避免重复拉RDS)

# 小样本窗口: 聚焦 OOS 段附近, 控制调用量
START, END = "2023-01-01", "2025-08-01"
MODEL = "deepseek-chat"
MAX_CHARS = 2500
FWD = 5
SEG = {
    "train": ("2023-01-01", "2023-12-31"),   # 此处 train/test 仅用于看稳定性
    "test":  ("2024-04-01", "2025-08-01"),
}

PROMPT = u"""你是卖方研报分析助手。下面是一篇A股公司研报的正文(可能被截断)。
只依据文本, 输出一个JSON(不要任何多余文字), 字段:
- rating: 该研报对公司的评级, 整数1-5 (1=卖出/回避, 2=减持/中性偏空, 3=中性/持有, 4=增持/看好, 5=买入/强烈推荐); 无法判断填3
- rating_change: 相比该机构此前观点是否变化, -1(下调)/0(未提及或维持)/1(上调)
- implied_upside: 正文给出的目标价相对现价的隐含涨幅(小数, 如0.20表示+20%); 无则填null
- earnings_revision: 对盈利预测的调整方向, -1(下调)/0(未提)/1(上调)
正文如下:
---
__CONTENT__
---
只输出JSON。"""


def _hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / n
    return x.mean() / math.sqrt(max(var, 1e-18) / n)


def neutralize_xs(df, ycol, fcols, date_col="date"):
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(date_col):
        sub = g[[ycol] + fcols].dropna()
        if len(sub) < len(fcols) + 5:
            continue
        X = sub[fcols].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X = np.column_stack([np.ones(len(X)), X])
        y = sub[ycol].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[sub.index] = y - X @ beta
    return out


def cs_ic(df, sig, ycol):
    s = df[["date", sig, ycol]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby("date").apply(lambda g: g[sig].corr(g[ycol], method="spearman"),
                                   include_groups=False).dropna()


def _stratified_sample(df: pd.DataFrame, max_docs: int) -> pd.DataFrame:
    """按月分层抽样, 让样本覆盖整个窗口(含test段), 而非只取最早的文档。"""
    if not max_docs or len(df) <= max_docs:
        return df
    df = df.copy()
    df["_m"] = pd.to_datetime(df["date"]).dt.to_period("M")
    frac = max_docs / len(df)
    out = (df.groupby("_m", group_keys=False)
             .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))),
                                       random_state=42), include_groups=True))
    return out.drop(columns="_m").reset_index(drop=True)


def pull_raw_docs(max_docs: int) -> pd.DataFrame:
    """拉公司研报正文 (date, stkcd, content)。限定有平台价格的股, 控制量。"""
    if RAW_CACHE.exists():
        df = pd.read_parquet(RAW_CACHE)
        print("[llm] raw docs cache: %d" % len(df))
        return _stratified_sample(df, max_docs)

    px_ids = set(pd.read_parquet(PLATFORM, columns=["stkcd"])["stkcd"].unique())
    nm = L.build_name_to_stkcd()
    conn = L._connect("reportdata")
    edges = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq="3MS").tolist()
    edges = sorted(set([pd.Timestamp(START)] + edges + [pd.Timestamp(END) + pd.Timedelta(days=1)]))
    rows = []
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            a_s, b_s = a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat()
            print("[llm] pulling docs %s ~ %s" % (a_s, b_s))
            sql = ("SELECT infopubldate, content, stocks FROM report_info "
                   "WHERE infopubldate BETWEEN %s AND %s "
                   "  AND classification LIKE '%%公司研究%%' "
                   "  AND content IS NOT NULL AND content <> '' "
                   "  AND stocks IS NOT NULL AND stocks <> '' AND stocks <> 'None'")
            df = pd.read_sql(sql, conn, params=(a_s, b_s))
            if df.empty:
                continue
            df["names"] = df["stocks"].map(L._parse_stocks_field)
            # 只保留单票研报(多票研报观点不明确), 降噪 + 省调用
            df = df[df["names"].map(len) == 1].copy()
            df["stkcd"] = df["names"].map(lambda x: x[0]).map(nm.name2code)
            df = df.dropna(subset=["stkcd"])
            df = df[df["stkcd"].isin(px_ids)]
            df["date"] = pd.to_datetime(df["infopubldate"]).dt.normalize()
            df["content"] = df["content"].astype(str).str.slice(0, MAX_CHARS)
            rows.append(df[["date", "stkcd", "content"]])
    finally:
        conn.close()
    raw = pd.concat(rows, ignore_index=True).drop_duplicates(["date", "stkcd", "content"])
    raw.to_parquet(RAW_CACHE, index=False)
    print("[llm] cached raw docs: %d -> %s" % (len(raw), RAW_CACHE))
    return _stratified_sample(raw, max_docs)


def _make_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1", timeout=60)


def _parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```")[1].lstrip("json").strip() if "```" in s else s
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return {}
    return {}


def extract_opinions(raw: pd.DataFrame, go: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = raw.reset_index(drop=True)
    raw["h"] = raw["content"].map(_hash)
    results = {}
    todo = []
    for i, r in raw.iterrows():
        p = CACHE_DIR / ("%s.json" % r["h"])
        if p.exists():
            results[i] = json.loads(p.read_text(encoding="utf-8"))
        else:
            todo.append(i)
    print("[llm] docs=%d  cached=%d  todo_api=%d" % (len(raw), len(results), len(todo)))
    # 成本估算: deepseek-chat 约 ¥1/百万input token, 中文~1.5字/token, 2500字≈1700token
    est_in = len(todo) * 1700
    print("[llm] 预计 input token ≈ %.1f万 (约 RMB %.2f 元, 加输出更高); MAX_CHARS=%d"
          % (est_in / 1e4, est_in / 1e6 * 1.0, MAX_CHARS))
    if not go:
        print("[llm] DRY-RUN: 未加 --go, 不调用 LLM。确认成本后加 --go 重跑。")
        return pd.DataFrame()

    if todo:
        client = _make_client()

        def _work(i):
            prompt = PROMPT.replace("__CONTENT__", raw.at[i, "content"])
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=MODEL, max_tokens=200,
                        messages=[{"role": "user", "content": prompt}])
                    d = _parse_json(r.choices[0].message.content)
                    (CACHE_DIR / ("%s.json" % raw.at[i, "h"])).write_text(
                        json.dumps(d, ensure_ascii=False), encoding="utf-8")
                    return i, d
                except Exception as e:
                    if attempt == 2:
                        return i, {"_err": repr(e)}
                    time.sleep(2 ** attempt)
        print("[llm] calling DeepSeek (max_workers=8) ...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_work, i) for i in todo]
            done = 0
            for f in as_completed(futs):
                i, d = f.result()
                results[i] = d
                done += 1
                if done % 200 == 0:
                    print("  %d/%d  (%.0fs)" % (done, len(todo), time.time() - t0), flush=True)

    raw["op"] = raw.index.map(lambda i: results.get(i, {}))
    def _g(d, k):
        v = d.get(k)
        try:
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan
    out = pd.DataFrame({
        "date": raw["date"], "stkcd": raw["stkcd"],
        "op_rating": raw["op"].map(lambda d: _g(d, "rating")),
        "op_change": raw["op"].map(lambda d: _g(d, "rating_change")),
        "op_upside": raw["op"].map(lambda d: _g(d, "implied_upside")),
        "op_ern": raw["op"].map(lambda d: _g(d, "earnings_revision")),
    })
    out.to_parquet(DOCS_CACHE, index=False)
    print("[llm] wrote extractions: %s  (valid rating=%d)"
          % (DOCS_CACHE, out["op_rating"].notna().sum()))
    return out


def evaluate(op: pd.DataFrame):
    px = pd.read_parquet(PLATFORM, columns=["date", "stkcd", "adj_close", "float_mktcap"])
    px["date"] = pd.to_datetime(px["date"].astype(str), format="%Y%m%d")
    px = px.dropna(subset=["adj_close"]).sort_values(["stkcd", "date"])
    g = px.groupby("stkcd", group_keys=False)
    px["fwd"] = g["adj_close"].shift(-FWD) / px["adj_close"] - 1.0
    px["lnmktcap"] = np.log(px["float_mktcap"].astype(float) + 1.0)

    # 发布日 snap 到交易日, 按(交易日,股)聚合(同日多篇取均值)
    tds = pd.Series(np.sort(px["date"].unique()))
    op = op.copy()
    op["date"] = op["date"].map(
        lambda d: tds.iloc[tds.searchsorted(d, "left")] if tds.searchsorted(d, "left") < len(tds) else pd.NaT)
    op = op.dropna(subset=["date"])
    agg = op.groupby(["date", "stkcd"], as_index=False)[
        ["op_rating", "op_change", "op_upside", "op_ern"]].mean()

    df = px.merge(agg, on=["date", "stkcd"], how="inner")   # 只在有研报的(日,股)上评
    df["fwd_resid"] = neutralize_xs(df, "fwd", ["lnmktcap"])

    print("\nLLM 正文观点因子 -> 未来5日收益 RankIC (市值中性化, 仅研报覆盖日, NW lag=4)")
    print("%-12s %12s %8s %12s %8s %10s"
          % ("factor", "train IC", "t", "test IC", "t", "wf_t"))
    print("-" * 70)
    for f in ["op_rating", "op_change", "op_upside", "op_ern"]:
        ic_all = cs_ic(df, f, "fwd_resid")
        seg = {}
        for name, (s, e) in SEG.items():
            m = (ic_all.index >= pd.Timestamp(s)) & (ic_all.index <= pd.Timestamp(e))
            ic = ic_all[m]
            seg[name] = (ic.mean(), nw_t(ic.to_numpy(), 4)) if len(ic) else (np.nan, np.nan)
        print("%-12s %+12.4f %8.2f %+12.4f %8.2f %10.2f"
              % (f, seg["train"][0], seg["train"][1], seg["test"][0], seg["test"][1],
                 nw_t(ic_all.to_numpy(), 4)))
    df.to_csv(ROOT / "artifacts" / "diag_llm_opinion.csv", index=False)
    print("\n[llm] wrote artifacts/diag_llm_opinion.csv")
    print("判读: 若 op_change/op_upside 的 test t>2 且 wf_t>2 -> 正文观点变化有 alpha,")
    print("      值得全量 LLM 抽取; 若≈0 -> 文本增量已到顶, 收口转量价组合。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=4000)
    ap.add_argument("--go", action="store_true", help="真正调用 LLM (否则 dry-run)")
    args = ap.parse_args()

    raw = pull_raw_docs(args.max_docs)
    print("[llm] 样本: %d 篇单票公司研报, %s~%s, 覆盖股=%d"
          % (len(raw), raw["date"].min().date(), raw["date"].max().date(),
             raw["stkcd"].nunique()))
    op = extract_opinions(raw, go=args.go)
    if op.empty:
        return
    evaluate(op)


if __name__ == "__main__":
    main()
