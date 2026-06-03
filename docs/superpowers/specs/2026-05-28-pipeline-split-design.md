# strategy_pipeline.ipynb → 多脚本拆分 · 设计稿

**日期**：2026-05-28
**目标**：把 `strategy_pipeline.ipynb` 的三阶段流水线拆成 4 个可独立运行的 Python 脚本，
中间产物落盘到统一目录，每阶段可单独调试/重跑而不污染其它阶段。

## 0. 背景

现状 `strategy_pipeline.ipynb` 串起三个模块：

1. **Stage 1**：`text_daily_summary.py` —— 拉/聚合文本，LLM 摘要，QC
2. **Stage 2**：`embed_finbert_bertopic.py` —— FinBERT 嵌入 + 有监督 UMAP/BERTopic 降维
3. **Stage 3**：`gru_lgbm_factor.py` —— GRU + LGBM 因子，IC / 多空回测

加上 `load_real_docs.py`（阿里云 RDS 拉真实研报）和 `synth_research_reports.py`（fallback 合成器）。

**痛点**：所有逻辑塞在一个 ipynb 里；重跑一段经常意味着重跑整段；DB I/O 慢的 stage 1a
和 LLM 慢的 stage 1b 绑在一起；切换 `text_col=summary/full_text` 时 cell 顺序容易乱。

## 1. 目标 & 非目标

**目标**

- 4 个 stage 脚本，每个 `python stageX_*.py` 直接能跑
- 中间产物统一落到 `./artifacts/`，路径写在 `config.py` 里
- 不动 4 个底层模块（`text_daily_summary.py` 等）的内部实现，只做封装调用
- 保留 ipynb 里 3 个可选场景：快速通道、全文 vs 摘要对比、synth fallback
- 防泄漏红线（reducer 只 fit train、stage 3 purge）一字不改

**非目标**

- 不引入 Airflow / Prefect / Make 之类编排框架
- 不做 GPU / 分布式改造
- 不为 reducer/embedder 做模型持久化（线上推理另立 spec）
- 不改 4 个底层 `.py` 模块的 API

## 2. 目录结构

```
grumodel/
├── config.py                          # 唯一配置：PATHS / SEGMENTS / 各 stage 超参
├── stage1a_load_raw_docs.py           # DB/synth/file → artifacts/raw_docs.parquet
├── stage1b_summarize.py               # raw_docs → daily.parquet (+ QC)
├── stage2_embed_reduce.py             # daily → artifacts/panel.parquet
├── stage3_factor.py                   # panel → 因子 + IC + 多空 (支持 --input)
├── compare_fulltext_vs_summary.py     # 复用 stage2+stage3 跑两遍出对比表
├── artifacts/
│   ├── raw_docs.parquet
│   ├── daily.parquet
│   ├── panel.parquet
│   ├── qc/summary_quality.json
│   └── stage3/<tag>/
│       ├── factor_test.csv
│       └── metrics.json
└── cache_summaries/                   # 沿用已有的 LLM 摘要缓存
```

底层模块（`text_daily_summary.py`、`embed_finbert_bertopic.py`、`gru_lgbm_factor.py`、
`load_real_docs.py`、`synth_research_reports.py`）保持原位、原 API 不动。

## 3. config.py 契约

集中所有可调项，所有 stage 脚本 `from config import PATHS, SEGMENTS, STAGE1A, ...`：

```python
# config.py
from pathlib import Path

ARTIFACTS = Path("artifacts")

SEGMENTS = {
    "train": slice("2020-01-01", "2022-12-31"),
    "valid": slice("2023-01-01", "2024-03-31"),
    "test":  slice("2024-04-01", "2025-08-01"),
}

PATHS = dict(
    raw_docs      = ARTIFACTS / "raw_docs.parquet",
    daily         = ARTIFACTS / "daily.parquet",
    panel         = ARTIFACTS / "panel.parquet",
    label_src     = Path("dt_lgbm.parquet"),         # 标签/收益的真源（vwap_rk, vwap_ret20）
    qc_dir        = ARTIFACTS / "qc",
    stage3_dir    = ARTIFACTS / "stage3",
    summary_cache = Path("cache_summaries"),
)

STAGE1A = dict(
    source="real",                                    # real / synth / file
    start="2020-01-01", end="2025-08-01",
    batch_months=3,
    use_content_fallback=True,
    max_text_chars=4000, min_text_chars=30,
    snap_to_panel=True, max_forward_days=40,         # 对齐到 dt_lgbm.parquet 的面板日期
)

STAGE1B = dict(
    max_chars=4000,
    target_min=300, target_max=500,
    use_llm=False,                                    # False 用 offline fallback；True 用 Anthropic
    llm_model="claude-sonnet-4-5",
    qc_low_threshold=0.80,
    qc_embed_model="bert-base-chinese",
)

STAGE2 = dict(
    text_col="summary",                               # summary / full_text
    embed_model="bert-base-chinese",
    n_components=10,
    n_bins=10,
    use_bertopic=True,
    random_state=42,
)

STAGE3 = dict(
    seq_len=6,
    label_horizon=1,
    max_gap_days=None,                                # 月频不查日历缺口
    label_col="vwap_rk",
    return_col="vwap_ret20",
    feature_prefix="emb_",
    gru_hidden=64, gru_layers=2, proj_dim=128,
    epochs=10, batch_size=128, patience=3,
    lgbm_num_boost_round=500, lgbm_early_stopping=30,
    oof_gru_features=False,
    device="auto",
)
```

CLI 用 argparse 暴露**少数高频开关**（`--source`、`--input`、`--text-col`、`--tag`、`--use-llm`），
其余从 `config.py` 拿默认值。

## 4. 4 个 stage 脚本契约

每个脚本遵循同一个契约：
**①** 从 `PATHS` 读固定输入 → **②** 跑核心逻辑 → **③** 写固定输出 + 一行 stdout summary + 一份 `<output>.meta.json`

### stage1a_load_raw_docs.py

- **CLI**：`--source {real,synth,file}` (默认 `real`)
- **输入**：DB（real）/ 合成器（synth）/ 已有 parquet（file 模式，路径由 `--file PATH` 提供）
- **输出**：`artifacts/raw_docs.parquet`
- **逻辑**：
  - `real`：调用 `load_real_docs.load_real_docs(LoadConfig(...))`，然后 `snap_to_panel_dates`
    把研报日期对齐到 `dt_lgbm.parquet` 的面板日期
  - `synth`：从 `dt_lgbm.parquet` 抽 8 只股票的 (date, stkcd) 网格，
    调 `synth_research_reports.generate_panel`
  - `file`：直接 `pd.read_parquet(args.file)`
- **stdout**：`raw_docs: rows=N | dates=D | stocks=S`

### stage1b_summarize.py

- **CLI**：`--use-llm`（默认 False，用 offline fallback）、`--no-qc`（默认跑 QC）
- **输入**：`artifacts/raw_docs.parquet`
- **输出**：
  - `artifacts/daily.parquet`（列：`full_text, n_docs, n_chars, summary, summary_chars`）
  - `artifacts/qc/summary_quality.json`（中位余弦相似度、压缩率、低于阈值比例）
- **逻辑**：
  - `t1.aggregate_daily(raw, AggregateConfig(max_chars=...))`
  - `summarize_fn` = offline 兜底 / `t1.make_anthropic_summarizer(...)`
  - `t1.summarize_panel(daily, summarize_fn, SummaryConfig(cache_dir=...))`
  - QC：`t1.summary_similarity(daily, embedder.encode, low_threshold=...)`
- **stdout**：`daily: rows=N | summary chars median=M | cos_median=X | pct_below=Y`

### stage2_embed_reduce.py

- **CLI**：`--text-col {summary,full_text}` (默认 `summary`)
- **输入**：`artifacts/daily.parquet` + `dt_lgbm.parquet`（取 `vwap_rk` / `vwap_ret20` 标签）
- **输出**：`artifacts/panel.parquet`（列：`emb_0..emb_{k-1}, vwap_rk, vwap_ret20`，索引 `(date, stkcd)`）
- **逻辑**：
  - `t2.FinBertEmbedder(EmbedConfig(model_name=...)).encode` → `emb_all`
  - `t2.split_by_segments(daily, SEGMENTS)` → seg masks
  - 从 `dt_lgbm.parquet` reindex 出 `y_all`、`ret20`
  - `SupervisedReducer.fit(emb_all[mask_train], y_all[mask_train], docs_train)` 然后 transform 全样本
  - `t2.build_feature_panel(...)` 拼面板写出
- **防泄漏断言**：fit 时显式打印 `reducer.fit on N_train rows | transform N_total rows`，
  禁止把非 train mask 喂给 fit
- **stdout**：`panel: shape=(N, K) | NaN ratio=...`

### stage3_factor.py

- **CLI**：`--input PATH`（默认 `artifacts/panel.parquet`）、`--tag NAME`（默认 `default`）
- **输入**：任意 panel parquet（列含 `emb_*`、`vwap_rk`、`vwap_ret20`）
- **输出**：
  - `artifacts/stage3/<tag>/factor_test.csv`
  - `artifacts/stage3/<tag>/metrics.json`（含 train/valid/test 的 IC/RankIC/ICIR/n_days + 多空 Sharpe）
- **逻辑**：调用 `t3.run_pipeline(panel, SEGMENTS, FactorConfig(**STAGE3))`
- **快速通道用法**：`python stage3_factor.py --input dt_lgbm.parquet --tag fast`
- **stdout**：表格化的三段 metrics + L/S Sharpe

## 5. compare_fulltext_vs_summary.py

- **CLI**：`--tag NAME`（默认 `compare`）
- **输入**：`artifacts/daily.parquet`
- **输出**：`artifacts/compare_text_summary.csv`（行 `full_text / summary`，列 `IC_mean / RankIC_mean / ICIR`）
- **逻辑**：
  ```python
  for col in ("full_text", "summary"):
      panel = run_stage2(text_col=col)                 # 复用 stage2 的核心函数
      metrics = run_stage3(panel)["metrics"]["test"]   # 复用 stage3
      rows.append((col, metrics))
  ```
  实现上把 stage2 / stage3 的"核心逻辑"抽成可 import 的函数（`run()`），
  CLI `main()` 只负责解析参数 + 路径管理；compare 脚本直接 import `run()`。

## 6. 防泄漏 & 复现性

- **Stage 2**：`SupervisedReducer.fit` **只** 接受 train mask；调用处打印 `[stage2] reducer.fit on
  {N_train} rows / transform {N_total} rows` 作为可视化的红线证据。
- **Stage 3**：`label_horizon` 参数继续控制 purge（已在 `gru_lgbm_factor.py` 里实现）。
- **跨脚本一致性**：所有脚本只从 `config.SEGMENTS` 读切分；不允许在脚本里再造 slice。
- **复现性**：每个 stage 写完产物后落一份 `<output>.meta.json`：
  ```json
  {
    "stage": "stage2",
    "timestamp": "2026-05-28T10:23:11",
    "config": { ... STAGE2 字典 ... },
    "input_paths": ["artifacts/daily.parquet", "dt_lgbm.parquet"],
    "output_path": "artifacts/panel.parquet",
    "git_hash": "<HEAD sha if available, else null>"
  }
  ```

## 7. 典型跑法

**完整跑一遍**：

```bash
python stage1a_load_raw_docs.py                          # → artifacts/raw_docs.parquet
python stage1b_summarize.py                              # → artifacts/daily.parquet
python stage2_embed_reduce.py                            # → artifacts/panel.parquet
python stage3_factor.py --tag main                       # → artifacts/stage3/main/
```

**快速通道（跳过 1+2）**：

```bash
python stage3_factor.py --input dt_lgbm.parquet --tag fast
```

**全文 vs 摘要对比**：

```bash
python stage1a_load_raw_docs.py
python stage1b_summarize.py
python compare_fulltext_vs_summary.py
```

**用合成研报**：

```bash
python stage1a_load_raw_docs.py --source synth
```

## 8. 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| `dt_lgbm.parquet` 被改动，stage2 标签错位 | stage2 写 meta.json 记录 `label_src` 的 mtime + 行数 |
| 用户忘了先跑前一阶段，下一阶段读到空/旧文件 | 每个 stage 开头检查输入文件存在 + mtime；缺则 raise 带提示 |
| `--use-llm` 跑一半 API 限流挂掉 | 沿用现有 `cache_summaries/`，幂等续跑 |
| Stage 2 BERTopic 在小训练集上不稳定 | 沿用 ipynb 的 `random_state=42`；meta.json 记录 |
| 改了 config.py 忘了重跑下游 | meta.json 包含 config 快照；调试时人工 diff |

## 9. 不在本 spec 范围

- LLM 真实调用（`STAGE1B["use_llm"]=True`）的细节本 spec 不展开，沿用 `t1.make_anthropic_summarizer` 已有实现
- 模型持久化（UMAP / BERTopic 落盘以便后续 transform 新数据）——线上推理另立 spec
- 任何 `text_daily_summary.py / embed_finbert_bertopic.py / gru_lgbm_factor.py` 内部的 bug fix / 重构
