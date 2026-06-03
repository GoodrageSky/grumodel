# GruModel · 日频文本因子流水线

把 A 股研报文本 →（FinBERT 嵌入 + 有监督降维）→ GRU + LGBM 跨截面因子的端到端流程，拆成 4 个可独立运行的脚本。

## 现网结果（2020-01 ~ 2025-08，372 只股票，68 个月度截面）

```
              IC_mean  RankIC_mean   ICIR
train          0.323        0.297   2.51
valid         +0.021       +0.021   0.30
test          +0.019       +0.022   0.20    →  L/S Sharpe = +1.70
```

跟把 `dt_lgbm.parquet`（原始 4096 维嵌入）直接喂 Stage 3 的「快速通道」相比：

|  | 本流水线 (文本 10 维) | 快速通道 (raw 4096 维) |
|---|---|---|
| train IC | 0.323 | 0.322 |
| **test IC** | **+0.019** | -0.023 |
| **test ICIR** | **+0.20** | -0.21 |
| **test L/S Sharpe** | **+1.70** | -3.83 |

样本外 IC **由负转正**、Sharpe 从 -3.83 翻到 +1.70。结论：有监督降维 + 研报文本压出来的 10 维特征，在小面板上的泛化性显著好过裸 4096 维。

---

## 数据流

```
研报库 (RDS reportdata.report_info)
   │  ① stage1a_load_raw_docs
   ▼
(date, stkcd, text, source, title)  raw_docs.parquet
   │  ② stage1b_summarize    (聚合 → 300-500 字摘要 → 全文/摘要 QC)
   ▼
(date, stkcd, full_text, summary)   daily.parquet
   │  ③ stage2_embed_reduce  (FinBERT → 仅 train fit 的 UMAP+BERTopic)
   ▼
(date, stkcd, emb_0..emb_9 + 标签)  panel.parquet
   │  ④ stage3_factor        (GRU 序列特征 → LGBM → 因子)
   ▼
factor_test.csv + metrics.json (IC / RankIC / ICIR / 多空)
```

## 目录结构

```
grumodel/
├── pipelines/                          ← 入口脚本（python ... 这些）
│   ├── stage1a_load_raw_docs.py            # DB / synth / file → raw_docs.parquet
│   ├── stage1b_summarize.py                # raw → daily.parquet + QC
│   ├── stage2_embed_reduce.py              # daily → panel.parquet
│   ├── stage3_factor.py                    # panel → 因子 + 指标
│   └── compare_fulltext_vs_summary.py      # 闭环验证：摘要是否丢 alpha
│
├── src/                                ← 库代码
│   ├── config.py                           # 唯一配置：PATHS / SEGMENTS / 各 stage 超参
│   ├── meta.py                             # 写 <output>.meta.json 复现快照
│   ├── text_daily_summary.py               # ② 聚合 + LLM 摘要 + 相似度 QC
│   ├── embed_finbert_bertopic.py           # ③ FinBERT + 有监督 UMAP/BERTopic
│   ├── gru_lgbm_factor.py                  # ④ GRU + LGBM + 因子评估
│   ├── load_real_docs.py                   # ① 阿里云 RDS 真实研报 loader
│   └── synth_research_reports.py           # ① 离线合成器（DB 不通时 fallback）
│
├── data/dt_lgbm.parquet                ← 标签真源 (vwap_rk, vwap_ret20)
├── artifacts/                          ← 跑出来的产物（每个文件都带 .meta.json 快照）
├── cache/                              ← 摘要 / 合成研报缓存
└── docs/                               ← PROJECT_NOTES + 设计 spec + 实现 plan
```

## 跑法

```powershell
# 完整跑一遍（从 RDS 拉研报开始）
python pipelines/stage1a_load_raw_docs.py                          # 默认 --source real
python pipelines/stage1b_summarize.py                              # 加 --use-llm 走 Anthropic
python pipelines/stage2_embed_reduce.py                            # 或 --text-col full_text
python pipelines/stage3_factor.py --tag main

# 快速通道：跳过 Stage 1+2，直接验 Stage 3 通不通
python pipelines/stage3_factor.py --input data/dt_lgbm.parquet --tag fast

# DB 不通时改用离线合成研报
python pipelines/stage1a_load_raw_docs.py --source synth
```

`config.py` 是唯一改参的地方（`SEGMENTS`、模型名、各 stage 超参全在这里）。

---

## 三条贯穿全程的防泄漏红线

1. **有监督降维只 fit train**。`SupervisedReducer` 把 768 维压到 10 维时用了未来收益分位作监督；如果在全样本上 fit，未来标签会顺着降维器渗进 train 特征，回测必然虚高。Stage 2 启动时显式打印 `[stage2] reducer.fit on N_train rows / transform N_total rows` 作为审计行。
2. **段间 purge**。标签是未来 `label_horizon` 步收益，train/valid/test 之间按该长度裁尾以消除重叠。
3. **摘要 prompt 硬约束**。日频文本里禁止出现对未来价格/收益的预测性断言，避免把"答案"喂进特征。

## 三个阶段的关键设计

### ① 文本日频化 + 摘要 (`text_daily_summary.py`)

- **聚合**：(交易日, 股票) 为键，去重后融合一篇；优先级 `研报 > 公告 > 新闻`，`max_chars` 预算用尽时先截新闻。
- **时点对齐**：只能用截至当日收盘可得的文本。研报落在非交易日时由 `snap_to_panel_dates` 前向贴到下一个面板日期。
- **300-500 字**：按中文字（去空白）计数，超界自动让模型重写一次。
- **工程**：磁盘缓存（按内容哈希，可断点续跑）、并发、重试。日频 × 数千股 × 数年 = 百万级调用，缓存是省钱和可复现的前提；改 prompt 时 bump `model_name_for_cache` 即可让缓存失效。

**全文 vs 摘要的两层验证**：
- 表层：同一嵌入器编码两份，看余弦分布、压缩率、低相似度样本——本次跑出 `cos_median=0.987 / pct_below=0%`。
- 本质：相似度高 ≠ 不丢 alpha。真正的判据是下游因子 IC，跑 `python pipelines/compare_fulltext_vs_summary.py` 比较两种 text_col 的 test IC。

### ② FinBERT 嵌入 + 有监督降维 (`embed_finbert_bertopic.py`)

- **嵌入器**：默认 `bert-base-chinese`（FinBERT 已下架公共权重）；MLM 模型用 **mean-pooling（带 attention mask）** 而非裸 CLS。要换 `BAAI/bge-large-zh-v1.5` 之类只改 `STAGE2["embed_model"]`。
- **有监督 UMAP**：未来收益排名 `vwap_rk` 离散成分位 bin 作监督目标，把 768 维压到 10 维；低维流形朝「能区分未来收益」的方向收敛。可选叠加 BERTopic 复用同一 UMAP 出主题特征。
- 输出列固定为 `emb_0..emb_{k-1}`，是 Stage 3 现成的输入格式。

### ③ GRU + LGBM 因子 (`gru_lgbm_factor.py`)

历史 ipynb 版本有几处会让 OOS 偏乐观，本模块已修：

| 问题 | 旧做法 | 现做法 |
|---|---|---|
| **频率** | `SEQ_LEN=6` 含糊 | 交易日窗口 + `max_gap_days` 过滤停牌缺口，窗口不再静默跨越长停牌 |
| **标签重叠泄漏** | 按日期硬切 | 段间按 `label_horizon` purge，裁尾去重叠 |
| **GRU→LGBM 堆叠泄漏** | GRU 在 train 训完用全样本抽特征 | 默认保留（test 仍干净）+ 提供 `oof_gru_features=True` 对 train 做 OOF |
| **评估** | 只看 MSE | 截面 IC / RankIC / ICIR + 顶底分位多空净值 + 年化 Sharpe |
| **序列构建** | Python 循环 | `sliding_window_view` 向量化，~100× 提速 + NaN/缺口掩码 |
| **截面预处理** | 仅 z-score | 先 winsorize 再 z-score，抗异常值 |

---

## 复现性

每个 stage 写完产物后落一份 `<output>.meta.json`，含：stage 名、时间戳、输入路径的 mtime/size、完整 config 快照、git hash（若有）。出问题时能直接定位是哪份输入/哪组参数。

## 运行环境

```bash
pip install torch lightgbm transformers bertopic umap-learn anthropic pymysql
# Anthropic 摘要要 ANTHROPIC_API_KEY；FinBERT/BERTopic 首次运行需联网下载权重
```

## 还能继续做的方向

- **摘要稳健性**：同一文本多温度采样取 embedding 均值，降 LLM 随机性。
- **降维监督目标**：分位 bin 之外，可试行业中性化后的残差收益，减少行业暴露。
- **序列编码替换**：GRU 换 Temporal Fusion / 因果卷积，与 GRU 做特征拼接交给 LGBM。
- **组合层**：当前到「因子」为止，下一步接行业/风格中性化 + 换手约束的组合优化。
# 频率口径说明

当前 `data/dt_lgbm.parquet` 是月度/近月度截面数据，Stage 1a 会把研报日期 forward-snap 到同一组面板日期。因此 `artifacts/raw_docs.parquet`、`artifacts/daily.parquet`、`artifacts/panel.parquet` 以及 Stage 3 的 `factor_test.csv` 当前都按月频截面计算。

Stage 3 配置中 `annualization_periods=12`，多空 Sharpe 使用 `sqrt(12)` 年化；`seq_len=6` 表示 6 个面板截面，约 6 个月，而不是 6 个交易日。
