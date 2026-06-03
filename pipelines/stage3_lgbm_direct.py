"""Stage 3 baseline: direct LightGBM on panel features, no GRU.

Input : artifacts/panel.parquet
Output: artifacts/stage3/<tag>/factor_test.csv
        artifacts/stage3/<tag>/metrics.json
        artifacts/stage3/<tag>/metrics.json.meta.json

Run from repo root:
  python pipelines/stage3_lgbm_direct.py --tag direct_lgbm
"""
from __future__ import annotations

import argparse
import json
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import gru_lgbm_factor as t3
from config import PATHS, SEGMENTS, STAGE3
from meta import write_meta


def _segments_from_args(args: argparse.Namespace) -> dict:
    custom = [
        args.train_start, args.train_end,
        args.valid_start, args.valid_end,
        args.test_start, args.test_end,
    ]
    if not any(custom):
        return SEGMENTS
    missing = [name for name, value in [
        ("--train-start", args.train_start),
        ("--train-end", args.train_end),
        ("--valid-start", args.valid_start),
        ("--valid-end", args.valid_end),
        ("--test-start", args.test_start),
        ("--test-end", args.test_end),
    ] if value is None]
    if missing:
        raise ValueError(f"custom segments require all boundaries; missing {missing}")
    return {
        "train": slice(args.train_start, args.train_end),
        "valid": slice(args.valid_start, args.valid_end),
        "test": slice(args.test_start, args.test_end),
    }


def _metrics_to_json(metrics: dict, backtest: dict | None) -> dict:
    out = {}
    for split, m in metrics.items():
        out[split] = {
            k: (float(v) if hasattr(v, "__float__") and not isinstance(v, str) else v)
            for k, v in m.items()
            if not k.endswith("series")
        }
    if backtest is not None:
        out["test_backtest"] = {
            "ls_mean": float(backtest["ls_mean"]),
            "ls_sharpe": float(backtest["ls_sharpe"]),
            "annualization_periods": int(backtest["annualization_periods"]),
        }
    return out


def write_period_metrics(out_dir: Path, res: dict) -> Path:
    """Write per-period IC / RankIC / L/S series for diagnostics."""
    rows = []
    for split, m in res["metrics"].items():
        ic = m.get("IC_series")
        ric = m.get("RankIC_series")
        if ic is None or ric is None:
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(ic.index),
            "split": split,
            "IC": ic.to_numpy(),
            "RankIC": ric.reindex(ic.index).to_numpy(),
        })
        rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    bt = res.get("backtest")
    if bt is not None and "ls_period" in bt and not out.empty:
        ls = bt["ls_period"].rename("long_short").reset_index()
        ls.columns = ["date", "long_short"]
        ls["date"] = pd.to_datetime(ls["date"])
        out = out.merge(ls, on="date", how="left")
    out_path = out_dir / "period_metrics.csv"
    print(f"[direct_lgbm] writing period metrics csv: {out_path}", flush=True)
    out.to_csv(out_path, index=False)
    return out_path


def run(panel: pd.DataFrame, segments: dict | None = None,
        config_overrides: dict | None = None) -> dict:
    cfg_dict = dict(STAGE3)
    if config_overrides:
        cfg_dict.update(config_overrides)
    cfg = t3.FactorConfig(**cfg_dict)
    segments = SEGMENTS if segments is None else segments

    if not isinstance(panel.index, pd.MultiIndex):
        panel = panel.copy()
        panel[cfg.date_col] = pd.to_datetime(panel[cfg.date_col])
        panel = panel.set_index([cfg.date_col, cfg.stock_col]).sort_index()
    if panel.index.duplicated().any():
        n_dup = int(panel.index.duplicated().sum())
        print(f"[direct_lgbm] warning: dropping {n_dup} duplicate index rows", flush=True)
        panel = panel[~panel.index.duplicated(keep="first")]

    feature_cols = [c for c in panel.columns if c.startswith(cfg.feature_prefix)]
    if not feature_cols:
        raise ValueError(f"no feature columns with prefix {cfg.feature_prefix!r}")

    print(
        f"[direct_lgbm] rows={len(panel)} features={len(feature_cols)} "
        f"annualization_periods={cfg.annualization_periods}",
        flush=True,
    )
    df_feat = t3.cross_sectional_zscore(panel, feature_cols, cfg.winsorize_q)
    meta = df_feat.reset_index()[[cfg.date_col, cfg.stock_col]]
    masks = t3.purged_masks(meta, segments, cfg)
    for name, m in masks.items():
        print(f"[direct_lgbm] {name}: {int(m.sum())} rows", flush=True)

    booster, preds = t3.fit_lgbm_segments(
        df_feat, feature_cols, cfg.label_col, masks, cfg)

    metrics = {}
    for name, p in preds.items():
        m = t3.compute_ic(p, df_feat, cfg.label_col)
        metrics[name] = m
        print(
            f"[{name}] IC={m['IC_mean']:.4f} RankIC={m['RankIC_mean']:.4f} "
            f"ICIR={m['ICIR']:.3f} (n_periods={m['n_periods']})",
            flush=True,
        )

    backtest = None
    if cfg.return_col and cfg.return_col in df_feat.columns:
        backtest = t3.quantile_backtest(
            preds["test"], df_feat, cfg.return_col,
            annualization_periods=cfg.annualization_periods,
        )
        print(
            f"[test] L/S mean={backtest['ls_mean']:.4f} "
            f"ann.Sharpe({cfg.annualization_periods})={backtest['ls_sharpe']:.2f}",
            flush=True,
        )

    return {
        "booster": booster,
        "preds": preds,
        "metrics": metrics,
        "backtest": backtest,
        "df_feat": df_feat,
        "feature_cols": feature_cols,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=PATHS["panel"])
    ap.add_argument("--tag", default="direct_lgbm")
    ap.add_argument("--annualization-periods", type=int, default=None)
    ap.add_argument("--lgbm-param", action="append", default=[],
                    help="override LightGBM param, e.g. --lgbm-param num_leaves=15")
    ap.add_argument("--train-start", default=None)
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--valid-start", default=None)
    ap.add_argument("--valid-end", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    args = ap.parse_args()

    out_dir = PATHS["stage3_dir"] / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[direct_lgbm] loading panel: {args.input}", flush=True)
    panel = pd.read_parquet(args.input)
    print(f"[direct_lgbm] loaded panel shape={panel.shape}", flush=True)

    overrides = {}
    if args.annualization_periods is not None:
        overrides["annualization_periods"] = args.annualization_periods
    if args.lgbm_param:
        params = dict(STAGE3.get("lgbm_params", t3.FactorConfig().lgbm_params))
        for item in args.lgbm_param:
            if "=" not in item:
                raise ValueError(f"--lgbm-param must be key=value, got {item!r}")
            k, v = item.split("=", 1)
            try:
                params[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                params[k] = v
        overrides["lgbm_params"] = params
    segments = _segments_from_args(args)
    print(f"[direct_lgbm] segments={segments}", flush=True)
    res = run(panel, segments=segments, config_overrides=overrides)

    factor_path = out_dir / "factor_test.csv"
    metrics_path = out_dir / "metrics.json"
    print(f"[direct_lgbm] writing factor csv: {factor_path}", flush=True)
    res["preds"]["test"].rename("direct_lgbm_factor").reset_index().to_csv(
        factor_path, index=False)

    metrics = _metrics_to_json(res["metrics"], res["backtest"])
    print(f"[direct_lgbm] writing metrics json: {metrics_path}", flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")
    write_period_metrics(out_dir, res)

    write_meta(
        metrics_path,
        stage="stage3_lgbm_direct",
        config={
            "tag": args.tag,
            "input": str(args.input),
            "segments": {k: [v.start, v.stop] for k, v in segments.items()},
            **STAGE3,
            **overrides,
        },
        inputs=[str(args.input)],
    )

    print(pd.DataFrame(res["metrics"]).T[["IC_mean", "RankIC_mean", "ICIR", "n_periods"]])
    if res["backtest"] is not None:
        print(
            f"test L/S mean={res['backtest']['ls_mean']:.4f}  "
            f"ann.Sharpe({res['backtest']['annualization_periods']})="
            f"{res['backtest']['ls_sharpe']:.2f}",
        )


if __name__ == "__main__":
    main()
