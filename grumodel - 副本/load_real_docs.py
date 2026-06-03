"""
load_real_docs.py
=================
Pull real research reports from the Aliyun RDS (reportdata.report_info) and
shape them into the schema that `text_daily_summary.aggregate_daily` consumes:

    columns = [date, stkcd, text, source, title]

Drop-in replacement for synth_research_reports.generate_panel in
`strategy_pipeline.ipynb`.

Key design notes
----------------
1. `report_info.stocks` is a **comma-separated list of Chinese names**, not
   stkcds. We resolve names -> 6-digit-windcode (`600519.SH`) via
   financedata.asharedescription (current names) + asharepreviousname
   (historical names).
2. News table (`clsnews.news_info`) has no stock linkage column and is skipped
   in this loader, per project decision (research-only path).
3. Reports published on non-trading days (weekends/holidays) keep their
   `infopubldate` as-is here. Downstream you typically want to *snap forward*
   to the nearest panel date for join with labels -- use `snap_to_panel_dates`.
4. Heavy fields (`content`) can balloon to GB-scale for multi-year pulls; by
   default we use the shorter `abstract` and fall back to `content` only when
   abstract is empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pymysql


# --------------------------------------------------------------------------- #
# DB connection helpers
# --------------------------------------------------------------------------- #
DB_HOST = "quantstudio.mysql.rds.aliyuncs.com"
DB_USER = "gjzq"
DB_PASS = "Sinolink600109!"
DB_PORT = 3306


def _connect(database: str) -> pymysql.connections.Connection:
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS,
                           database=database, port=DB_PORT, charset="gbk")


# --------------------------------------------------------------------------- #
# Chinese name -> stkcd mapping
# --------------------------------------------------------------------------- #
@dataclass
class NameMap:
    name2code: dict        # cleaned-name -> windcode (e.g. "贵州茅台" -> "600519.SH")
    ambiguous: set         # names that map to >1 code (kept only as a flag set)


def _normalize_name(s: str) -> str:
    """Strip whitespace and remove common decorations so old/new names match."""
    if s is None:
        return ""
    s = str(s).strip()
    # remove common ST / suspension markers etc. that pollute exact match
    for tag in ("*ST", "ST", "（", "）", "(", ")"):
        s = s.replace(tag, "")
    return s.strip()


def build_name_to_stkcd() -> NameMap:
    """Read financedata.asharedescription + asharepreviousname, return name->code map.

    Conflict policy: current name wins over previous name. Within previous names,
    later windcode wins (rare). Conflicting current names land in `ambiguous`.
    """
    conn = _connect("financedata")
    try:
        # current names (one row per stock)
        cur_df = pd.read_sql(
            "SELECT S_INFO_WINDCODE AS stkcd, S_INFO_NAME AS name "
            "FROM asharedescription "
            "WHERE S_INFO_WINDCODE LIKE '%.SH' OR S_INFO_WINDCODE LIKE '%.SZ' "
            "   OR S_INFO_WINDCODE LIKE '%.BJ'",
            conn,
        )
        # historical names (multiple rows per stock; older naming variants)
        prev_df = pd.read_sql(
            "SELECT S_INFO_WINDCODE AS stkcd, S_INFO_NAME AS name "
            "FROM asharepreviousname",
            conn,
        )
    finally:
        conn.close()

    cur_df["name_n"] = cur_df["name"].map(_normalize_name)
    prev_df["name_n"] = prev_df["name"].map(_normalize_name)

    # detect ambiguous current names (e.g. same name held by two listings)
    counts = cur_df.groupby("name_n")["stkcd"].nunique()
    ambiguous = set(counts[counts > 1].index)

    name2code: dict[str, str] = {}
    # previous names first (so current overwrites)
    for _, r in prev_df.iterrows():
        n = r["name_n"]
        if not n or n in ambiguous:
            continue
        name2code[n] = r["stkcd"]
    for _, r in cur_df.iterrows():
        n = r["name_n"]
        if not n or n in ambiguous:
            continue
        name2code[n] = r["stkcd"]
    return NameMap(name2code=name2code, ambiguous=ambiguous)


# --------------------------------------------------------------------------- #
# Stocks-field parsing
# --------------------------------------------------------------------------- #
def _parse_stocks_field(s: object) -> list[str]:
    """Split 'A,B,C' / 'A、B' into a list of normalized names. Drop sentinel 'None'."""
    if s is None:
        return []
    txt = str(s).strip()
    if not txt or txt.lower() == "none":
        return []
    # support both Chinese and ASCII separators
    for sep in ("，", "、", ";", "；", "|"):
        txt = txt.replace(sep, ",")
    out = []
    seen = set()
    for piece in txt.split(","):
        n = _normalize_name(piece)
        if n and n.lower() != "none" and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #
@dataclass
class LoadConfig:
    start: str                          # 'YYYY-MM-DD' inclusive
    end: str                            # 'YYYY-MM-DD' inclusive
    use_content_fallback: bool = True   # use content when abstract is empty
    max_text_chars: int = 4000          # truncate per-doc to keep memory bounded
    min_text_chars: int = 30            # drop docs shorter than this
    batch_months: int = 3               # pull DB in N-month chunks to bound memory
    verbose: bool = True


def load_real_docs(cfg: LoadConfig,
                   name_map: Optional[NameMap] = None) -> pd.DataFrame:
    """Returns DataFrame[date, stkcd, text, source, title] for downstream Stage 1.

    Rows where every listed Chinese name fails to map to a stkcd are dropped.
    Rows whose `stocks` field is empty / 'None' are dropped (no anchor stock).
    """
    if name_map is None:
        if cfg.verbose:
            print("[load_real_docs] building name->stkcd map ...", flush=True)
        name_map = build_name_to_stkcd()
        if cfg.verbose:
            print(f"  mapped {len(name_map.name2code):,} unique names, "
                  f"{len(name_map.ambiguous)} ambiguous", flush=True)

    # Build batch boundaries (server-side date filter is cheap; client memory isn't).
    start_ts = pd.Timestamp(cfg.start).normalize()
    end_ts = pd.Timestamp(cfg.end).normalize()
    edges = pd.date_range(start_ts, end_ts, freq=f"{cfg.batch_months}MS").tolist()
    if not edges or edges[0] > start_ts:
        edges = [start_ts] + edges
    if edges[-1] < end_ts:
        edges.append(end_ts + pd.Timedelta(days=1))

    out_frames = []
    conn = _connect("reportdata")
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            a_s, b_s = a.date().isoformat(), (b - pd.Timedelta(days=1)).date().isoformat()
            if cfg.verbose:
                print(f"[load_real_docs] pulling {a_s} ~ {b_s} ...", flush=True)

            sql = (
                "SELECT id, infopubldate, infotitle, abstract, "
                + ("content, " if cfg.use_content_fallback else "")
                + "stocks "
                "FROM report_info "
                "WHERE infopubldate BETWEEN %s AND %s "
                "  AND stocks IS NOT NULL AND stocks <> '' AND stocks <> 'None'"
            )
            df = pd.read_sql(sql, conn, params=(a_s, b_s))
            if df.empty:
                continue

            # pick best text per row
            def _pick_text(r):
                ab = r.get("abstract")
                if ab and str(ab).strip() and str(ab).strip().lower() != "none":
                    return str(ab).strip()
                if cfg.use_content_fallback:
                    ct = r.get("content")
                    if ct and str(ct).strip() and str(ct).strip().lower() != "none":
                        return str(ct).strip()
                return ""

            df["text"] = df.apply(_pick_text, axis=1)
            df["text"] = df["text"].str.slice(0, cfg.max_text_chars)
            df = df[df["text"].str.len() >= cfg.min_text_chars]
            if df.empty:
                continue

            # explode stocks -> long form, then map name->stkcd
            df["names"] = df["stocks"].map(_parse_stocks_field)
            df = df[df["names"].map(len) > 0]
            if df.empty:
                continue
            df = df.explode("names", ignore_index=True)
            df["stkcd"] = df["names"].map(name_map.name2code)

            n_before = len(df)
            df = df.dropna(subset=["stkcd"])
            if cfg.verbose:
                kept = len(df)
                print(f"  rows: {n_before:,} -> {kept:,} after name resolution"
                      f"  ({kept / max(n_before, 1):.1%} mapped)", flush=True)
            if df.empty:
                continue

            out = pd.DataFrame({
                "date": pd.to_datetime(df["infopubldate"]),
                "stkcd": df["stkcd"].astype(str),
                "text": df["text"],
                "source": "research",
                "title": df["infotitle"].fillna("").astype(str),
                "_doc_id": df["id"].astype(str),       # for downstream dedup
            })
            out_frames.append(out)
    finally:
        conn.close()

    if not out_frames:
        return pd.DataFrame(columns=["date", "stkcd", "text", "source", "title"])

    raw = pd.concat(out_frames, ignore_index=True)
    # one report can list a stock once; if a (date, stkcd, doc_id) appears twice
    # (shouldn't, but be defensive), keep first
    raw = raw.drop_duplicates(subset=["_doc_id", "stkcd"]).drop(columns="_doc_id")
    raw = raw.sort_values(["date", "stkcd"]).reset_index(drop=True)
    if cfg.verbose:
        print(f"[load_real_docs] DONE  rows={len(raw):,}  dates={raw.date.nunique()}  "
              f"stocks={raw.stkcd.nunique()}", flush=True)
    return raw


# --------------------------------------------------------------------------- #
# Date alignment helper
# --------------------------------------------------------------------------- #
def snap_to_panel_dates(raw_docs: pd.DataFrame,
                        panel: pd.DataFrame,
                        date_col: str = "date",
                        stock_col: str = "stkcd",
                        max_forward_days: int = 35) -> pd.DataFrame:
    """For each report, snap its date forward to the next panel date that the
    same stock has in `panel` (within `max_forward_days`).

    Useful when `panel` is monthly: a report dated 2024-03-10 with the next
    panel date 2024-03-31 gets re-stamped to 2024-03-31, so downstream label
    join lines up.

    Rows whose stock has no upcoming panel date within the window are dropped.
    """
    panel_dates = (panel[[date_col, stock_col]]
                   .drop_duplicates()
                   .sort_values([stock_col, date_col]))
    panel_dates[date_col] = pd.to_datetime(panel_dates[date_col])
    raw = raw_docs.copy()
    raw[date_col] = pd.to_datetime(raw[date_col])

    out_chunks = []
    for stk, g_panel in panel_dates.groupby(stock_col, sort=False):
        g_raw = raw[raw[stock_col] == stk]
        if g_raw.empty:
            continue
        merged = pd.merge_asof(
            g_raw.sort_values(date_col),
            g_panel[[date_col]].rename(columns={date_col: "_snap_date"})
                                 .sort_values("_snap_date"),
            left_on=date_col, right_on="_snap_date",
            direction="forward",
            tolerance=pd.Timedelta(days=max_forward_days),
        )
        merged = merged.dropna(subset=["_snap_date"])
        if merged.empty:
            continue
        merged[date_col] = merged["_snap_date"]
        out_chunks.append(merged.drop(columns="_snap_date"))

    if not out_chunks:
        return raw_docs.iloc[0:0].copy()
    out = pd.concat(out_chunks, ignore_index=True)
    return out.sort_values([date_col, stock_col]).reset_index(drop=True)
