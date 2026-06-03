# -*- coding: utf-8 -*-
"""在【量化平台机】(QuantStudio / QSEnv, Python 3.6) 上运行的取数脚本。

目的：为 grumodel 的"天花板测试"(P0) 导出量价原始数据 + Barra 风格暴露，
判断当前评估窗口到底有没有统计功效，以及文本因子相对风格因子有无增量。

只负责【取数 + 存 parquet】，不做对齐/算因子——那些在 grumodel 机上做。

用法（平台机）：
    1. 把本文件拷到平台机任意目录
    2. python export_platform_factors.py
    3. 把生成的 3 个 .pqt 拷回 grumodel/data/platform/

输出（当前目录）：
    px_daily.pqt        日频复权收盘价/均价/成交金额/换手率/流通市值/交易状态 (long 表)
    barra_exposure.pqt  月末 Barra 风格因子暴露 (long 表)
    barra_meta.pqt      Barra 因子名 / 日期清单 (排查用)

口径：
    - 日期范围 2019-06-01 ~ 2025-08-01：起点比 panel(2020-01) 提前 ~7 个月，
      留出动量/反转/波动率的 lookback。
    - 全市场股票（不在平台端按 panel 股票池过滤，回到 grumodel 再 inner-join，
      避免代码格式/退市股差异导致漏数）。
"""
from __future__ import print_function
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from QSEnvironment import QSEnv

START_DATE = "20190601"          # 提前于 panel(20200102) 以留 lookback
END_DATE = "20250801"

PRICE_TABLE = "ElementaryFactor"  # 全 A 量价表（factor.ipynb 里 ETF 版叫 ElementaryFactor_ETF）
# 名字按全 A 表 getFactorName('ElementaryFactor') 的真实命名:
#   成交额 -> 成交金额; 无"是否停牌", 改用"交易状态"; 另加换手率/流通市值做更多候选因子
PRICE_FACTORS = {
    "adj_close":    u"复权收盘价",
    "vwap":         u"复权均价",
    "amount":       u"成交金额",
    "turnover":     u"换手率",
    "float_mktcap": u"流通市值",
    "trade_status": u"交易状态",
}
BARRA_TABLE = "BarraRiskData"


def _wide_to_long(wide, value_name):
    """平台返回宽表 (index=日期str, columns=股票代码) -> long 表。"""
    wide = wide.copy()
    wide.index.name = "date"
    long = (wide.stack(dropna=False)
                .rename(value_name)
                .reset_index())
    long.columns = ["date", "stkcd", value_name]
    return long


def export_prices(env):
    print("[export] loading price/volume from %s ..." % PRICE_TABLE)
    # 先列出该表真实因子名, 名字对不上时给清晰报错(而不是炸在 .copy())
    available = env.FactorDB.getFactorName(PRICE_TABLE)
    print("[export] %s 现有因子: %s" % (PRICE_TABLE, available))
    frames = {}
    for key, cn_name in PRICE_FACTORS.items():
        print("  - %s (%s)" % (key, cn_name))
        if cn_name not in available:
            raise KeyError(
                u"因子名 %r 不在表 %s 中。可选: %s —— 请按真实名字改 PRICE_FACTORS。"
                % (cn_name, PRICE_TABLE, available))
        wide = env.FactorDB.loadFactor(
            table_name=PRICE_TABLE,
            factor_name=cn_name,
            start_date=START_DATE,
            end_date=END_DATE,
            dates=None, ids=None)
        if wide is None:
            raise ValueError(
                u"loadFactor(%s, %s) 返回 None —— 日期范围内可能无数据或参数不符。"
                % (PRICE_TABLE, cn_name))
        frames[key] = _wide_to_long(wide, key)

    out = frames["adj_close"]
    for key in ("vwap", "amount", "turnover", "float_mktcap", "trade_status"):
        out = out.merge(frames[key], on=["date", "stkcd"], how="outer")
    print("[export] price long shape:", out.shape)
    pq.write_table(pa.Table.from_pandas(out), "px_daily.pqt")
    print("[export] wrote px_daily.pqt")


def export_barra(env):
    print("[export] loading Barra exposures from %s ..." % BARRA_TABLE)
    # Barra 协方差/暴露为月频(月末)。getTableDate 返回每月最后交易日。
    all_dates = env.FRDB.getTableDate(BARRA_TABLE)
    dates = [d for d in all_dates if START_DATE <= str(d) <= END_DATE]
    print("[export] barra month-end dates: %d (%s ~ %s)"
          % (len(dates), dates[0] if dates else "NA", dates[-1] if dates else "NA"))

    rows = []
    for dt in dates:
        # loadFactorData: 个股 x Barra风格因子 暴露矩阵
        expo = env.FRDB.loadFactorData(BARRA_TABLE, dates=str(dt))
        expo = expo.copy()
        expo.index.name = "stkcd"
        expo = expo.reset_index()
        expo.insert(0, "date", str(dt))
        rows.append(expo)
    barra = pd.concat(rows, ignore_index=True)
    print("[export] barra long shape:", barra.shape)
    print("[export] barra factor cols:", [c for c in barra.columns if c not in ("date", "stkcd")])
    pq.write_table(pa.Table.from_pandas(barra), "barra_exposure.pqt")
    print("[export] wrote barra_exposure.pqt")

    meta = pd.DataFrame({"barra_factor": [c for c in barra.columns
                                          if c not in ("date", "stkcd")]})
    pq.write_table(pa.Table.from_pandas(meta), "barra_meta.pqt")
    print("[export] wrote barra_meta.pqt")


def main():
    env = QSEnv()
    env.startQS()
    export_prices(env)
    try:
        export_barra(env)
    except Exception as e:
        # Barra 是加分项，取不到也不挡量价天花板测试
        print("[export] WARN: barra export failed: %r" % e, file=sys.stderr)
    print("[export] done. copy *.pqt -> grumodel/data/platform/")


if __name__ == "__main__":
    main()
