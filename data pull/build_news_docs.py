"""
build_news_docs.py
==================
从 clsnews.news_info 提取新闻数据，通过两层策略映射到 (date, stkcd):

  Layer 1: 正则提取显式股票代码（精准，~100% 准确率）
    - sh600xxx / sz000xxx / bjxxxxxx（小写无点）
    - 600519.SH / 002202.SZ（大写带点）
    - 代码：600500 / 股票代码：600519（显式代码前缀）
    - [[$*ST柳化(sh600423)$]]（特殊标记）

  Layer 2: 公司名→代码模糊匹配（高召回，需去噪）
    - 利用 financedata 的 asharedescription + asharepreviousname 构建对照表
    - 反去噪规则：排除泛词、排除非A股、要求上下文有金融关键词

输出: raw_docs[date, stkcd, text, source, title]  → 直接喂 aggregate_daily()

时间预估:
  Layer 1: ~5s (MySQL REGEXP COUNT) + ~30s (拉取5.6万条) = ~1 分钟
  Layer 2: ~5s (MySQL REGEXP COUNT) + ~5-15 分钟 (拉取含公司名的新闻) = ~5-15 分钟
  总计: ~10-20 分钟
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pymysql

# ---------------------------------------------------------------------------
# DB 连接配置（与 load_real_docs.py 一致，但 charset 已修正为 gbk）
# ---------------------------------------------------------------------------
DB_HOST = "quantstudio.mysql.rds.aliyuncs.com"
DB_USER = "gjzq"
DB_PASS = "Sinolink600109!"
DB_PORT = 3306


def _connect(database: str) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS,
        database=database, port=DB_PORT, charset="gbk",
    )


# ---------------------------------------------------------------------------
# 公司名 → 代码 对照表（复用 load_real_docs.py 逻辑）
# ---------------------------------------------------------------------------
@dataclass
class NameMap:
    name2code: dict   # 公司名 -> windcode (e.g. "贵州茅台" -> "600519.SH")
    ambiguous: set    # 一个名对应多个代码的（极少，保留标记）


def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for tag in ("*ST", "ST", "（", "）", "(", ")"):
        s = s.replace(tag, "")
    return s.strip()


def build_name_to_stkcd() -> NameMap:
    """读取 financedata.asharedescription + asharepreviousname，构建名字→代码映射。"""
    conn = _connect("financedata")
    try:
        cur_df = pd.read_sql(
            "SELECT S_INFO_WINDCODE AS stkcd, S_INFO_NAME AS name "
            "FROM asharedescription "
            "WHERE S_INFO_WINDCODE LIKE '%.SH' OR S_INFO_WINDCODE LIKE '%.SZ' "
            "   OR S_INFO_WINDCODE LIKE '%.BJ'",
            conn,
        )
        prev_df = pd.read_sql(
            "SELECT S_INFO_WINDCODE AS stkcd, S_INFO_NAME AS name "
            "FROM asharepreviousname",
            conn,
        )
    finally:
        conn.close()

    cur_df["name_n"] = cur_df["name"].map(_normalize_name)
    prev_df["name_n"] = prev_df["name"].map(_normalize_name)

    counts = cur_df.groupby("name_n")["stkcd"].nunique()
    ambiguous = set(counts[counts > 1].index)

    name2code: dict[str, str] = {}
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

    # 同时加入去括号/去ST的版本，扩大匹配面
    extra = {}
    for name, code in name2code.items():
        clean = _normalize_name(name)
        if clean and clean != name and clean not in name2code:
            extra[clean] = code
    name2code.update(extra)

    return NameMap(name2code=name2code, ambiguous=ambiguous)


# ---------------------------------------------------------------------------
# Layer 1: 正则提取显式股票代码
# ---------------------------------------------------------------------------
# 综合正则：覆盖 sh600xxx / 600519.SH / 代码：600500 / [[$...$]] 等
RE_STOCK_CODE = re.compile(
    r"(?:sh|sz|bj)\d{6}"             # sh600519
    r"|\d{6}\.(?:SH|SZ|BJ)"          # 600519.SH
    r"|代码[：:\s]*\d{6}"             # 代码：600519
    r"|\[\[\$[^]]*?\(\s*(?:sh|sz)\d{6}\s*\)\s*\$\]\]",  # [[$*ST柳化(sh600423)$]]
    re.IGNORECASE,
)

# 从匹配文本中提取纯 stkcd
RE_EXTRACT_CODE = re.compile(r"(\d{6})", re.IGNORECASE)
RE_EXTRACT_EXCHANGE = re.compile(r"(?:sh|sz|bj|\.SH|\.SZ|\.BJ)", re.IGNORECASE)


def _format_stkcd(code_digits: str, exchange_hint: str = "") -> str:
    """把 6 位数字 + 交易所提示 统一成 600519.SH 格式。"""
    code_digits = code_digits.zfill(6)
    hint = exchange_hint.upper().replace(".", "")
    if hint in ("SH", "SZ", "BJ"):
        return f"{code_digits}.{hint}"
    # 根据 A 股规则推断交易所
    if code_digits.startswith(("6", "9")):
        return f"{code_digits}.SH"
    elif code_digits.startswith(("0", "2", "3")):
        return f"{code_digits}.SZ"
    elif code_digits.startswith(("4", "8")):
        return f"{code_digits}.BJ"
    return f"{code_digits}.SH"  # fallback


def extract_codes_layer1(text: str) -> list[str]:
    """从文本中正则提取股票代码，返回统一格式 ['600519.SH', '000858.SZ']。"""
    if not text:
        return []
    matches = RE_STOCK_CODE.findall(text)
    codes = set()
    for m in matches:
        # m 可能是字符串（简单匹配）或元组（复杂捕获组）
        raw = m if isinstance(m, str) else "".join(x for x in m if x)
        digits = RE_EXTRACT_CODE.findall(raw)
        exch = RE_EXTRACT_EXCHANGE.findall(raw)
        for d in digits:
            codes.add(_format_stkcd(d, exch[0] if exch else ""))
    return sorted(codes)


# ---------------------------------------------------------------------------
# Layer 2: 公司名 → 代码 模糊匹配
# ---------------------------------------------------------------------------
# 去噪黑名单：这些公司名太泛/太短/容易误匹配，禁止通过 Layer 2 进入
AMBIGUOUS_NAME_BLACKLIST = {
    # 单字/两字泛名（极易误伤）
    "东方", "中国", "南方", "北方", "西部", "中部", "东部",
    "新华", "招商", "中信", "光大", "平安", "华夏", "国泰",
    "银河", "海通", "广发", "申万", "国信", "兴业", "华泰",
    "长江", "东北", "西南", "太平洋",
    # 泛词（和日常用语重叠）
    "机器人", "新世界", "大智慧", "同花顺", "好想你", "全新好",
    "太阳能", "风能", "水能", "核能",
    "老百姓", "人民网", "新华网",
    # 极短且极易与上下文混淆
    "银行", "保险", "证券", "信托", "地产", "医药",
    # 外文译名/人名
    "伊戈尔", "安德利", "威尔科技",
}

# Layer 2 匹配时要求 name 前后 N 个字符内有至少一个金融上下文词
FINANCIAL_CONTEXT_WORDS = {
    "公司", "股份", "集团", "有限", "控股", "上市",
    "公告", "年报", "季报", "业绩", "盈利", "亏损",
    "股价", "涨停", "跌停", "收盘", "开盘", "涨跌",
    "A股", "股票", "证券", "基金", "市值", "估值",
    "董事会", "股东", "分红", "回购", "增持", "减持",
    "重组", "并购", "停牌", "复牌", "ST", "*ST",
    "财联社", "科创板", "创业板", "北交所",
}

CONTEXT_WINDOW = 15  # 名字前后各 15 个字符


def _has_financial_context(text: str, name: str) -> bool:
    """检查 name 在 text 中第一次出现的位置前后是否有金融上下文词。"""
    idx = text.find(name)
    if idx == -1:
        return False
    start = max(0, idx - CONTEXT_WINDOW)
    end = min(len(text), idx + len(name) + CONTEXT_WINDOW)
    window = text[start:end]
    return any(w in window for w in FINANCIAL_CONTEXT_WORDS)


def extract_codes_layer2(text: str, name_map: NameMap) -> list[str]:
    """通过公司名→代码对照表，从文本中识别 A 股标的。

    反去噪规则：
    1. 黑名单中的泛名直接跳过
    2. 要求匹配到的名字前后有金融上下文（公司、公告、股价 等）
    3. 只有 .SH / .SZ / .BJ 结尾的 A 股代码
    """
    if not text:
        return []

    found = []
    # 按长度从长到短匹配，避免短名截胡长名
    for name in sorted(name_map.name2code, key=len, reverse=True):
        if len(name) < 3:
            continue
        if name in AMBIGUOUS_NAME_BLACKLIST:
            continue
        if name in text:
            # 去噪检查：必须有金融上下文
            if _has_financial_context(text, name):
                code = name_map.name2code[name]
                # 只保留 A 股
                if code.endswith((".SH", ".SZ", ".BJ")):
                    found.append(code)
                    # 一旦匹配，把这段文本里的这个名字"遮掉"，避免重复
                    text = text.replace(name, "▇" * len(name), 1)

    # 按代码去重
    seen = set()
    result = []
    for code in found:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass
class BuildConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2026-06-03"
    output_dir: str = "."
    output_file: str = "raw_docs_news.parquet"
    batch_size: int = 50000          # 每批从 DB 拉取的行数
    min_text_chars: int = 30         # 丢弃太短的 content
    max_text_chars: int = 4000       # 截断太长的 content
    verbose: bool = True


def build_raw_docs(cfg: BuildConfig = BuildConfig()) -> pd.DataFrame:
    """主函数：从新闻库构建 raw_docs 格式数据。"""
    t_total = time.time()

    # ---- 0. 构建公司名对照表 ----
    if cfg.verbose:
        print("[0/3] 构建公司名→代码对照表 ...", flush=True)
    t0 = time.time()
    name_map = build_name_to_stkcd()
    if cfg.verbose:
        print(f"      对照表 {len(name_map.name2code):,} 条 | 歧义 {len(name_map.ambiguous)} 个 "
              f"| 耗时 {time.time()-t0:.1f}s", flush=True)

    # ---- 1. Layer 1: 显式代码提取 ----
    if cfg.verbose:
        print("[1/3] Layer 1: 正则提取显式股票代码 ...", flush=True)
    t0 = time.time()

    # 先用 MySQL REGEXP 过滤（大幅减少需要拉取的数据量）
    layer1_regex = r"(sh|sz|bj)[0-9]{6}|[0-9]{6}\.(SH|SZ|BJ)|代码[：: ]{0,2}[0-9]{6}"
    conn = _connect("clsnews")
    try:
        # 统计命中量
        count_sql = (
            f"SELECT COUNT(*) FROM news_info "
            f"WHERE ctime>='{cfg.start_date}' AND ctime<='{cfg.end_date} 23:59:59' "
            f"AND content REGEXP '{layer1_regex}'"
        )
        l1_total = pd.read_sql(count_sql, conn).iloc[0, 0]
        if cfg.verbose:
            print(f"      MySQL REGEXP 命中: {l1_total:,} 条", flush=True)

        # 分批拉取
        l1_frames = []
        offset = 0
        while offset < l1_total:
            sql = (
                f"SELECT id, ctime, content FROM news_info "
                f"WHERE ctime>='{cfg.start_date}' AND ctime<='{cfg.end_date} 23:59:59' "
                f"AND content REGEXP '{layer1_regex}' "
                f"ORDER BY id LIMIT {cfg.batch_size} OFFSET {offset}"
            )
            batch = pd.read_sql(sql, conn)
            if batch.empty:
                break
            # 正则提取代码
            rows = []
            for _, r in batch.iterrows():
                content = str(r["content"] or "")
                if len(content) < cfg.min_text_chars:
                    continue
                if len(content) > cfg.max_text_chars:
                    content = content[:cfg.max_text_chars]
                codes = extract_codes_layer1(content)
                for code in codes:
                    rows.append({
                        "date": pd.to_datetime(r["ctime"]),
                        "stkcd": code,
                        "text": content,
                        "source": "news",
                        "title": "",
                        "_method": "layer1_regex",
                    })
            if rows:
                l1_frames.append(pd.DataFrame(rows))
            offset += cfg.batch_size
            if cfg.verbose and offset % (cfg.batch_size * 2) == 0:
                print(f"      Layer 1 进度: {offset:,}/{l1_total:,}", flush=True)
    finally:
        conn.close()

    df_l1 = pd.concat(l1_frames, ignore_index=True) if l1_frames else pd.DataFrame(
        columns=["date", "stkcd", "text", "source", "title", "_method"]
    )
    if cfg.verbose:
        print(f"      Layer 1 产出: {len(df_l1):,} 条 | 耗时 {time.time()-t0:.1f}s", flush=True)

    # ---- 2. Layer 2: 公司名匹配 ----
    if cfg.verbose:
        print("[2/3] Layer 2: 公司名→代码匹配 ...", flush=True)
    t0 = time.time()

    # 用 MySQL REGEXP 预筛选：内容含"公司/股份/集团/上市/A股"等金融标志词
    # 这样可以跳过纯宏观/国际新闻，大幅减少处理量
    l2_filter = "公司|股份|集团|上市|A股|股票|证券"
    conn = _connect("clsnews")
    try:
        count_sql = (
            f"SELECT COUNT(*) FROM news_info "
            f"WHERE ctime>='{cfg.start_date}' AND ctime<='{cfg.end_date} 23:59:59' "
            f"AND content REGEXP '{l2_filter}'"
        )
        l2_candidate_total = pd.read_sql(count_sql, conn).iloc[0, 0]
        if cfg.verbose:
            print(f"      MySQL 金融词预筛选候选: {l2_candidate_total:,} 条", flush=True)

        # 收集 Layer 1 已经覆盖的 (date, id) 组合，Layer 2 跳过它们
        l1_ids = set()
        if not df_l1.empty:
            # df_l1 没有存原始 id，所以我们换个策略：
            # Layer 2 处理全部金融词匹配的行，但过滤掉已有显式代码的行（在提取阶段去重）
            pass

        l2_frames = []
        offset = 0
        while offset < l2_candidate_total:
            sql = (
                f"SELECT id, ctime, content FROM news_info "
                f"WHERE ctime>='{cfg.start_date}' AND ctime<='{cfg.end_date} 23:59:59' "
                f"AND content REGEXP '{l2_filter}' "
                f"ORDER BY id LIMIT {cfg.batch_size} OFFSET {offset}"
            )
            # 每批重新建连接，防止长连接超时断线
            try:
                batch = pd.read_sql(sql, conn)
            except Exception:
                conn.close()
                conn = _connect("clsnews")
                batch = pd.read_sql(sql, conn)
            if batch.empty:
                break

            rows = []
            for _, r in batch.iterrows():
                content = str(r["content"] or "")
                if len(content) < cfg.min_text_chars:
                    continue
                if len(content) > cfg.max_text_chars:
                    content = content[:cfg.max_text_chars]

                # 先 Layer 1 检查（有些代码 L1 正则没覆盖到的边缘情况）
                codes_l1 = set(extract_codes_layer1(content))
                # Layer 2 匹配
                codes_l2 = set(extract_codes_layer2(content, name_map))
                # 合并去重
                all_codes = codes_l1 | codes_l2
                for code in all_codes:
                    method = "layer1_regex" if code in codes_l1 else "layer2_name"
                    rows.append({
                        "date": pd.to_datetime(r["ctime"]),
                        "stkcd": code,
                        "text": content,
                        "source": "news",
                        "title": "",
                        "_method": method,
                    })
            if rows:
                l2_frames.append(pd.DataFrame(rows))
            offset += cfg.batch_size
            if cfg.verbose and offset % (cfg.batch_size * 4) == 0:
                print(f"      Layer 2 进度: {offset:,}/{l2_candidate_total:,}", flush=True)
    finally:
        conn.close()

    df_l2 = pd.concat(l2_frames, ignore_index=True) if l2_frames else pd.DataFrame(
        columns=["date", "stkcd", "text", "source", "title", "_method"]
    )
    if cfg.verbose:
        print(f"      Layer 2 产出: {len(df_l2):,} 条 | 耗时 {time.time()-t0:.1f}s", flush=True)

    # ---- 3. 合并 + 去重 + 输出 ----
    if cfg.verbose:
        print("[3/3] 合并去重 + 输出 ...", flush=True)

    df_all = pd.concat([df_l1, df_l2], ignore_index=True)

    # 同一 (date, stkcd) 的多条新闻取 text 最长的保留
    df_all = df_all.sort_values(["date", "stkcd", "text"], key=lambda x: x.str.len()
                                if x.name == "text" else x, ascending=[True, True, False])
    df_all = df_all.drop_duplicates(subset=["date", "stkcd"], keep="first")

    # 只保留 A 股 stkcd
    df_all = df_all[df_all["stkcd"].str.match(r"^\d{6}\.(SH|SZ|BJ)$")]

    # 去 date NaN
    df_all = df_all.dropna(subset=["date"])
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.normalize()

    df_all = df_all.sort_values(["date", "stkcd"]).reset_index(drop=True)

    # 输出列（与 pipeline 对齐）
    output_cols = ["date", "stkcd", "text", "source", "title"]
    df_out = df_all[output_cols]

    out_path = Path(cfg.output_dir) / cfg.output_file
    df_out.to_parquet(out_path, index=False)

    if cfg.verbose:
        l1_count = (df_all["_method"] == "layer1_regex").sum()
        l2_count = (df_all["_method"] == "layer2_name").sum()
        print(f"\n{'='*60}")
        print(f"完成! 总耗时: {time.time()-t_total:.0f}s")
        print(f"总产出: {len(df_out):,} 条")
        print(f"  Layer 1 (正则): {l1_count:,} 条 ({l1_count/max(len(df_out),1):.1%})")
        print(f"  Layer 2 (公司名): {l2_count:,} 条 ({l2_count/max(len(df_out),1):.1%})")
        print(f"  覆盖日期: {df_out['date'].nunique():,} 天")
        print(f"  覆盖股票: {df_out['stkcd'].nunique():,} 只")
        print(f"输出文件: {out_path.resolve()}")
        print(f"{'='*60}")

    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 clsnews.news_info 构建 raw_docs")
    parser.add_argument("--start", default="2018-01-01", help="起始日期")
    parser.add_argument("--end", default="2026-06-03", help="结束日期")
    parser.add_argument("--output", default="./raw_docs_news.parquet", help="输出文件")
    parser.add_argument("--batch", type=int, default=50000, help="批大小")
    args = parser.parse_args()

    cfg = BuildConfig(
        start_date=args.start,
        end_date=args.end,
        output_file=args.output,
        batch_size=args.batch,
    )
    build_raw_docs(cfg)
