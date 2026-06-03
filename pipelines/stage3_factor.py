"""Stage 3: GRU + LGBM factor pipeline.

Input : artifacts/panel.parquet  (or any panel parquet via --input)
Output: artifacts/stage3/<tag>/factor_test.csv
        artifacts/stage3/<tag>/metrics.json
        artifacts/stage3/<tag>/metrics.json.meta.json

Quick path (skip Stage 1 & 2), run from repo root:
  python pipelines/stage3_factor.py --input data/dt_lgbm.parquet --tag fast
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import gru_lgbm_factor as t3
from config import PATHS, SEGMENTS, STAGE3
from meta import write_meta


def run(panel: pd.DataFrame, segments: dict | None = None,
        config_overrides: dict | None = None) -> dict:
    print(f"[stage3] starting GRU+LGBM rows={len(panel)}", flush=True)
    cfg_dict = dict(STAGE3)
    if config_overrides:
        cfg_dict.update(config_overrides)
    cfg = t3.FactorConfig(**cfg_dict)
    return t3.run_pipeline(panel, SEGMENTS if segments is None else segments, cfg)


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


def _metrics_to_json(res: dict) -> dict:
    """Coerce numpy scalars to plain floats so json.dump won't choke."""
    out: dict = {}
    for split, m in res["metrics"].items():
        out[split] = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, str)
                          else v)
                      for k, v in m.items()}
    bt = res.get("backtest")
    if bt is not None:
        out["test_backtest"] = {
            "ls_mean": float(bt["ls_mean"]),
            "ls_sharpe": float(bt["ls_sharpe"]),
            "annualization_periods": int(bt["annualization_periods"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=PATHS["panel"],
                    help="panel parquet path")
    ap.add_argument("--tag", default="default",
                    help="output sub-directory under artifacts/stage3/")
    ap.add_argument("--oof-gru-features", action="store_true",
                    help="extract train GRU features out-of-fold before LGBM")
    ap.add_argument("--annualization-periods", type=int, default=None,
                    help="periods per year for Sharpe annualization")
    ap.add_argument("--train-start", default=None)
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--valid-start", default=None)
    ap.add_argument("--valid-end", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    args = ap.parse_args()

    out_dir = PATHS["stage3_dir"] / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage3] loading panel: {args.input}", flush=True)
    panel = pd.read_parquet(args.input)
    print(f"[stage3] loaded panel shape={panel.shape}", flush=True)
    segments = _segments_from_args(args)
    overrides = {}
    if args.oof_gru_features:
        overrides["oof_gru_features"] = True
    if args.annualization_periods is not None:
        overrides["annualization_periods"] = args.annualization_periods
    print(f"[stage3] segments={segments}", flush=True)
    if overrides:
        print(f"[stage3] config overrides={overrides}", flush=True)
    res = run(panel, segments=segments, config_overrides=overrides)

    # 1) factor (test slice) -> CSV
    test_factor = res["preds"]["test"].rename("gru_lgbm_factor").reset_index()
    print(f"[stage3] writing factor csv: {out_dir / 'factor_test.csv'}", flush=True)
    test_factor.to_csv(out_dir / "factor_test.csv", index=False)

    # 2) metrics -> JSON
    metrics = _metrics_to_json(res)
    metrics_path = out_dir / "metrics.json"
    print(f"[stage3] writing metrics json: {metrics_path}", flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")

    # 3) meta sidecar (pin which panel + config produced these numbers)
    write_meta(metrics_path, stage="stage3",
               config={"tag": args.tag, "input": str(args.input),
                       "segments": {k: [v.start, v.stop] for k, v in segments.items()},
                       **STAGE3, **overrides},
               inputs=[str(args.input)])

    # 4) human-readable summary
    print(pd.DataFrame(res["metrics"]).T[["IC_mean", "RankIC_mean", "ICIR", "n_periods"]])
    if res.get("backtest") is not None:
        print(f"test L/S mean={res['backtest']['ls_mean']:.4f}  "
              f"ann.Sharpe({res['backtest']['annualization_periods']})="
              f"{res['backtest']['ls_sharpe']:.2f}")


if __name__ == "__main__":
    main()
