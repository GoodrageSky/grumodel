# -*- coding: utf-8 -*-
"""生成项目进展报告 PDF (中文)。

按时间顺序梳理: 模型构建 -> 各阶段实验 -> 当前问题 -> 下一步方案。
用 reportlab + 系统 simhei 字体渲染中文。

用法: python pipelines/make_report.py
输出: artifacts/项目进展报告.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "项目进展报告.pdf"

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("CN", str(FONT_DIR / "simhei.ttf")))
pdfmetrics.registerFont(TTFont("CNsong", str(FONT_DIR / "simkai.ttf")))

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="CN",
                    fontSize=16, leading=22, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1a3c6e"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="CN",
                    fontSize=12.5, leading=18, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#244"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName="CN",
                      fontSize=10, leading=16, spaceAfter=4, alignment=TA_LEFT)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, leading=12,
                       textColor=colors.HexColor("#666"))
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontName="CN",
                       fontSize=22, leading=28, textColor=colors.HexColor("#1a3c6e"))
SUB = ParagraphStyle("SUB", parent=BODY, fontSize=11, leading=16,
                     textColor=colors.HexColor("#555"))


def P(t, s=BODY):
    return Paragraph(t, s)


def hr():
    return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#bbb"),
                      spaceBefore=4, spaceAfter=8)


def make_table(data, col_widths, header_bg="#1a3c6e", font_size=9):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "CN"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f5fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t


def build():
    story = []

    # ---- 封面 ----
    story.append(Spacer(1, 30 * mm))
    story.append(P("研报文本因子流水线", TITLE))
    story.append(Spacer(1, 4 * mm))
    story.append(P("模型构建与实验进展报告", H1))
    story.append(Spacer(1, 6 * mm))
    story.append(P("—— 模型构建 · 实验过程 · 当前问题 · 下一步方案", SUB))
    story.append(Spacer(1, 10 * mm))
    story.append(P("日期：2026-06-01", SUB))
    story.append(Spacer(1, 2 * mm))
    story.append(P("项目代号：grumodel（A股研报文本 → GRU+LGBM 跨截面因子）", SMALL))
    story.append(Spacer(1, 12 * mm))
    story.append(hr())
    story.append(P("一句话摘要：端到端流水线已跑通，但样本外 IC 始终在零附近波动。"
                   "经受控诊断定位，根因是<b>月频评估窗口样本量过小、统计功效不足</b>，"
                   "而非模型或符号错误。已用平台日频数据验证<b>改周频可恢复功效</b>，"
                   "下一步据此重建评估频率。", BODY))

    # ---- 1. 项目目标与模型构建 ----
    story.append(P("1. 项目目标与模型架构", H1))
    story.append(hr())
    story.append(P("目标：把 A股研报文本转化为可用于选股的跨截面因子，"
                   "并以严格的样本外（OOS）口径评估其有效性（IC / RankIC / ICIR / 多空夏普）。", BODY))
    story.append(P("四阶段流水线：", H2))
    arch = [
        ["阶段", "模块", "职责"],
        ["Stage 1a/1b", "文本日频化 + 摘要", "研报库 → (交易日,股票) 去重聚合 → 300-500字摘要"],
        ["Stage 2", "嵌入 + 有监督降维", "FinBERT(bert-base-chinese) 768维 → 仅train拟合的有监督UMAP → 10维"],
        ["Stage 3", "GRU + LGBM 因子", "GRU序列特征 → LGBM → 截面因子 + IC/多空回测"],
        ["对照", "全文 vs 摘要", "闭环验证摘要是否丢失 alpha"],
    ]
    story.append(make_table(arch, [22 * mm, 35 * mm, 100 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(P("三条防泄漏红线：①有监督降维只在 train 上拟合；"
                   "②段间按标签长度 purge 去重叠；③摘要 prompt 禁止未来信息。", SMALL))
    story.append(P("数据口径（关键）：标签源 dt_lgbm.parquet 为<b>月频</b>，"
                   "面板共 68 个月度截面、372 只研报覆盖股，每截面平均仅约 116 只有文本。", SMALL))

    # ---- 2. 实验过程（时间顺序）----
    story.append(P("2. 实验过程（按时间顺序）", H1))
    story.append(hr())

    story.append(P("2.1 基线流水线跑通", H2))
    story.append(P("完成 Stage1→3 全链路。文本压成 10 维特征后喂入 LGBM，"
                   "并与“裸 4096 维嵌入直喂 Stage3”的快速通道对比。", BODY))

    story.append(P("2.2 直接 LGBM 基线（direct_lgbm）实测结果", H2))
    t1 = [
        ["数据段", "IC_mean", "RankIC_mean", "ICIR", "截面数", "t 值"],
        ["train", "+0.1956", "+0.1906", "+1.598", "35", "+9.45"],
        ["valid", "-0.0154", "-0.0288", "-0.192", "14", "-0.72"],
        ["test", "+0.0161", "+0.0191", "+0.213", "17", "+0.88"],
    ]
    story.append(make_table(t1, [26 * mm, 24 * mm, 28 * mm, 22 * mm, 20 * mm, 22 * mm]))
    story.append(P("解读：train 强正（t=9.45）但 valid/test 的 |t|<2，与 0 无差异。"
                   "train 的高 IC 是“有监督降维用未来收益分位拟合 train 标签”的产物，"
                   "不代表泛化能力。LGBM 在第 1 轮即早停 —— 在 10 维上学不到可泛化结构。", BODY))

    story.append(P("2.3 滚动窗口（cons_roll）与全文/摘要对照", H2))
    t2 = [
        ["实验", "test IC", "test ICIR", "多空夏普"],
        ["cons_roll_2023_04", "-0.0226", "-0.220", "+0.14"],
        ["cons_roll_2023_10", "-0.0273", "-0.264", "-0.46"],
        ["cons_roll_2024_04", "+0.0066", "+0.089", "+1.75"],
        ["full_text vs summary", "-0.017 / -0.015", "-0.166 / -0.141", "—"],
    ]
    story.append(make_table(t2, [48 * mm, 32 * mm, 30 * mm, 28 * mm]))
    story.append(P("解读：test IC 符号在窗口间随机翻转；full_text 与 summary 两种文本"
                   "口径 OOS 都 ≈ -0.015。这排除了“符号学反”（train 强正），"
                   "指向 OOS 根本无可辨识信号。", BODY))

    story.append(P("2.4 受控诊断：信号到底在哪一层丢失（diag_signal）", H2))
    story.append(P("用与真实流水线相同的 purged 切分 + IC 计算代码，逐层消融：", BODY))
    t3 = [
        ["实验", "valid IC (t)", "test IC (t)", "判定"],
        ["正控制（标签+噪声）", "+0.995 (t=2760)", "+0.994 (t=1812)", "评估/对齐代码正确"],
        ["负控制（纯噪声）", "-0.018 (-0.63)", "+0.026 (+0.89)", "噪声基准"],
        ["raw 768维 + 岭回归", "+0.004 (0.11)", "-0.001 (-0.02)", "零信号"],
        ["无监督 PCA 10维", "-0.032 (-1.06)", "+0.048 (+1.09)", "零信号"],
        ["有监督 UMAP 10维", "-0.005 (-0.13)", "+0.060 (+2.00)", "≈噪声"],
    ]
    story.append(make_table(t3, [44 * mm, 34 * mm, 34 * mm, 32 * mm], font_size=8.5))
    story.append(P("关键结论：①正控制 t≈千级 → 评估/对齐/purge 代码无误，负 IC 是真实数据而非 bug；"
                   "②任何文本表示（原始768/PCA/有监督UMAP）OOS |t| 均 <2，与纯噪声无法区分 → "
                   "不是降维/调参问题；③有监督 UMAP 的 OOS 并不优于原始，印证 train IC 是泄漏式拟合产物。", BODY))

    story.append(P("2.5 天花板测试：评估窗口有没有统计功效（diag_ceiling）", H2))
    story.append(P("引入平台量价数据，用公认强因子检验当前月频窗口能否测出任何因子。"
                   "（test 段 17 个月度截面，372 股）", BODY))
    t4 = [
        ["因子", "test IC (t)", "因子", "test IC (t)"],
        ["动量 mom_12_1", "-0.0155 (t=-0.32)", "流动性 liq_amt20", "-0.0310 (t=-1.02)"],
        ["反转 rev_1m", "+0.0153 (t=0.30)", "换手率 turn_20", "-0.0387 (t=-1.07)"],
        ["波动率 vol_60", "-0.0278 (t=-0.45)", "小市值 size", "+0.0158 (t=0.36)"],
    ]
    story.append(make_table(t4, [38 * mm, 36 * mm, 38 * mm, 36 * mm], font_size=8.5))
    story.append(P("<b>决定性结论</b>：连换手率、小市值、波动率这些 A股最强因子，"
                   "在该月频窗口里 |t| 全部 <2 —— 评估窗口本身没有分辨力。"
                   "之前文本因子的“负 IC”只是这个无功效窗口里的噪声。", BODY))

    story.append(P("2.6 周频验证：改频率能否恢复功效（diag_weekly_ceiling）", H2))
    story.append(P("用平台日频价格自建周频截面（test 段 70 个截面）+ 未来1周收益标签，"
                   "重跑同样因子：", BODY))
    t5 = [
        ["因子", "月频 test t", "周频 test t（全市场）", "是否显著"],
        ["流动性 liq_amt20", "-1.02", "-3.09", "✓ 显著"],
        ["小市值 size", "+0.36", "+2.29", "✓ 显著"],
        ["反转 rev_1w", "—", "+1.65", "接近"],
        ["换手率 turn_20", "-1.07", "-1.20", "改善"],
    ]
    story.append(make_table(t5, [40 * mm, 30 * mm, 42 * mm, 28 * mm], font_size=8.5))
    story.append(P("<b>验证成立</b>：截面数 17→70 后，多个强因子 |t| 升破 2，"
                   "符号在 train/valid/test 三段一致。证明瓶颈是评估频率太粗，"
                   "t 值随 √截面数 放大的规律完全兑现。", BODY))
    story.append(P("已据此构建周频标签源 weekly_labels.parquet（326 个周度截面，"
                   "未来1周 vwap 收益 + 横截面分位）。", SMALL))

    # ---- 3. 当前问题 ----
    story.append(P("3. 当前遇到的核心问题", H1))
    story.append(hr())
    probs = [
        ["#", "问题", "证据 / 影响"],
        ["1", "月频评估无统计功效", "强因子 |t| 全<2；test 仅 17 截面，结构上测不出常规因子"],
        ["2", "train IC 失真，无诊断价值", "有监督降维用未来分位拟合 train 标签，train IC 0.196 是泄漏产物"],
        ["3", "文本特征 OOS 无可测信号", "原始768/PCA/UMAP 的 OOS |t| 均<2，与噪声无异"],
        ["4", "研报覆盖稀疏", "68 截面、每截面仅约 116 股有文本，改周频后文本新鲜度被稀释"],
        ["5", "文本票池压制因子上限", "372 只研报覆盖股多为大盘低换手，因子区分度天然弱于全市场"],
    ]
    story.append(make_table(probs, [10 * mm, 52 * mm, 95 * mm], font_size=8.8))

    # ---- 4. 下一步方案 ----
    story.append(P("4. 下一步方案", H1))
    story.append(hr())
    story.append(P("总原则：先把“评估”修到有功效，再回头判断文本因子真伪 —— "
                   "在无功效的窗口里调模型等于在噪声里调参。", BODY))
    steps = [
        ["优先级", "动作", "目的"],
        ["P0 已完成", "天花板测试 + 周频验证", "确认根因=评估频率，而非模型/符号"],
        ["P1 进行中", "评估频率改周频（标签源已建好）", "截面数 17→70，恢复统计功效"],
        ["P1", "标签换未来1周 vwap 收益（非重叠）", "与频率匹配，消除 20 日重叠自相关"],
        ["P1", "天花板/回测在全市场票池跑", "提升每截面股票数，降低 IC 估计方差"],
        ["P1", "评估口径改 walk-forward 汇总 t 值", "多窗口截面合并，单一 split 无统计意义"],
        ["P2", "（待验证后）去掉有监督 UMAP", "实测其 OOS 不优于原始/PCA，纯增过拟合"],
        ["P2", "换更强中文/金融嵌入器 bge/Qwen", "替代 bert-base-chinese mean-pooling"],
        ["P2", "抽事件/情绪/评级变动等结构化信号", "替代摘要截断+平均池化"],
    ]
    story.append(make_table(steps, [22 * mm, 70 * mm, 65 * mm], font_size=8.8))
    story.append(Spacer(1, 3 * mm))
    story.append(P("立即决策点：是否启动昂贵的文本 pipeline 全量重建（Stage1+2 重跑嵌入，"
                   "耗时数小时）。建议先用现有月度嵌入 forward-snap 到周频网格 + 周频标签跑一遍 Stage3，"
                   "用最小代价回答“文本在周频下是否冒出信号”，再决定是否全量重建。", BODY))

    story.append(Spacer(1, 6 * mm))
    story.append(hr())
    story.append(P("附：本报告涉及脚本 —— diag_signal.py（逐层消融）、"
                   "diag_ceiling.py（月频天花板）、diag_weekly_ceiling.py（周频验证）、"
                   "build_weekly_labels.py（周频标签源）、export_platform_factors.py（平台取数）。", SMALL))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="grumodel 项目进展报告")
    doc.build(story)
    print("[report] wrote %s" % OUT)


if __name__ == "__main__":
    build()
