"""Run robustness experiments for the text GRU+LGBM factor.

Examples from repo root:
  python pipelines/run_experiments.py --experiments no_bertopic oof rolling
  python pipelines/run_experiments.py --experiments all

Outputs:
  artifacts/experiments/summary.csv
  artifacts/stage3/<tag>/metrics.json
  artifacts/stage3/<tag>/factor_test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import stage2_embed_reduce as s2
import stage3_factor as s3
import stage3_lgbm_direct as direct
from config import PATHS, SEGMENTS
from meta import write_meta


ROLLING_SEGMENTS = {
    "roll_2023_04": {
        "train": slice("2020-01-01", "2022-03-31"),
        "valid": slice("2022-04-01", "2023-03-31"),
        "test": slice("2023-04-01", "2024-03-31"),
    },
    "roll_2023_10": {
        "train": slice("2020-01-01", "2022-09-30"),
        "valid": slice("2022-10-01", "2023-09-30"),
        "test": slice("2023-10-01", "2024-09-30"),
    },
    "roll_2024_04": SEGMENTS,
}

DIRECT_LGBM_GRIDS = {
    "direct_lgbm_leaf7_min100": {
        "num_leaves": 7,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "lambda_l2": 2.0,
    },
    "direct_lgbm_leaf15_min100": {
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "lambda_l2": 2.0,
    },
    "direct_lgbm_leaf7_min200": {
        "num_leaves": 7,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "lambda_l2": 5.0,
    },
}

DIRECT_LGBM_CONSERVATIVE = {
    "num_leaves": 7,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "lambda_l2": 5.0,
}


def _write_stage3_outputs(tag: str, panel_path: Path, res: dict,
                          segments: dict, config_overrides: dict | None = None) -> dict:
    out_dir = PATHS["stage3_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_path = out_dir / "factor_test.csv"
    metrics_path = out_dir / "metrics.json"
    print(f"[experiments] writing {factor_path}", flush=True)
    res["preds"]["test"].rename("gru_lgbm_factor").reset_index().to_csv(
        factor_path, index=False)

    metrics = s3._metrics_to_json(res)
    print(f"[experiments] writing {metrics_path}", flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")

    write_meta(
        metrics_path,
        stage="stage3_experiment",
        config={
            "tag": tag,
            "input": str(panel_path),
            "segments": {k: [v.start, v.stop] for k, v in segments.items()},
            **(config_overrides or {}),
        },
        inputs=[str(panel_path)],
    )
    return metrics


def _summary_row(tag: str, metrics: dict) -> dict:
    test = metrics.get("test", {})
    bt = metrics.get("test_backtest", {})
    return {
        "tag": tag,
        "test_IC_mean": test.get("IC_mean"),
        "test_RankIC_mean": test.get("RankIC_mean"),
        "test_ICIR": test.get("ICIR"),
        "test_n_periods": test.get("n_periods", test.get("n_days")),
        "test_ls_mean": bt.get("ls_mean"),
        "test_ls_sharpe": bt.get("ls_sharpe"),
        "annualization_periods": bt.get("annualization_periods"),
    }


def run_stage3_experiment(tag: str, panel_path: Path, panel: pd.DataFrame,
                          segments: dict, overrides: dict | None = None) -> dict:
    overrides = dict(overrides or {})
    overrides.setdefault("model_dir", str(PATHS["stage3_dir"] / tag / "gru_models"))
    print(f"\n[experiments] === stage3 {tag} ===", flush=True)
    print(f"[experiments] panel={panel_path}", flush=True)
    print(f"[experiments] overrides={overrides}", flush=True)
    res = s3.run(panel, segments=segments, config_overrides=overrides)
    return _write_stage3_outputs(tag, panel_path, res, segments, overrides)


def run_no_bertopic() -> tuple[str, dict]:
    tag = "no_bertopic"
    panel_path = PATHS["panel"].parent / "panel_no_bertopic.parquet"
    print("\n[experiments] === stage2 no_bertopic ===", flush=True)
    panel = s2.run(text_col="summary", use_bertopic=False)
    print(f"[experiments] writing {panel_path}", flush=True)
    panel.to_parquet(panel_path)
    write_meta(
        panel_path,
        stage="stage2_experiment",
        config={"text_col": "summary", "use_bertopic": False},
        inputs=[str(PATHS["daily"]), str(PATHS["label_src"])],
    )
    metrics = run_stage3_experiment(tag, panel_path, panel, SEGMENTS)
    return tag, metrics


def run_oof() -> tuple[str, dict]:
    tag = "main_oof"
    print("\n[experiments] loading main panel for OOF run", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    metrics = run_stage3_experiment(
        tag, PATHS["panel"], panel, SEGMENTS,
        overrides={"oof_gru_features": True},
    )
    return tag, metrics


def run_rolling() -> list[tuple[str, dict]]:
    print("\n[experiments] loading main panel for rolling runs", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    rows = []
    for tag, segments in ROLLING_SEGMENTS.items():
        metrics = run_stage3_experiment(tag, PATHS["panel"], panel, segments)
        rows.append((tag, metrics))
    return rows


def run_direct_lgbm() -> tuple[str, dict]:
    tag = "direct_lgbm"
    print("\n[experiments] loading main panel for direct LGBM", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    res = direct.run(
        panel,
        segments=SEGMENTS,
        config_overrides={"model_dir": str(PATHS["stage3_dir"] / tag / "gru_models")},
    )
    metrics = direct._metrics_to_json(res["metrics"], res["backtest"])

    out_dir = PATHS["stage3_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_path = out_dir / "factor_test.csv"
    metrics_path = out_dir / "metrics.json"
    print(f"[experiments] writing {factor_path}", flush=True)
    res["preds"]["test"].rename("direct_lgbm_factor").reset_index().to_csv(
        factor_path, index=False)
    print(f"[experiments] writing {metrics_path}", flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")
    write_meta(
        metrics_path,
        stage="stage3_lgbm_direct_experiment",
        config={"tag": tag, "input": str(PATHS["panel"])},
        inputs=[str(PATHS["panel"])],
    )
    return tag, metrics


def _write_direct_outputs(tag: str, res: dict, metrics: dict, config: dict) -> None:
    out_dir = PATHS["stage3_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_path = out_dir / "factor_test.csv"
    metrics_path = out_dir / "metrics.json"
    print(f"[experiments] writing {factor_path}", flush=True)
    res["preds"]["test"].rename("direct_lgbm_factor").reset_index().to_csv(
        factor_path, index=False)
    print(f"[experiments] writing {metrics_path}", flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float),
                            encoding="utf-8")
    direct.write_period_metrics(out_dir, res)
    write_meta(
        metrics_path,
        stage="stage3_lgbm_direct_experiment",
        config={"tag": tag, "input": str(PATHS["panel"]), **config},
        inputs=[str(PATHS["panel"])],
    )


def run_direct_lgbm_rolling() -> list[tuple[str, dict]]:
    print("\n[experiments] loading main panel for direct LGBM rolling", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    rows = []
    for base_tag, segments in ROLLING_SEGMENTS.items():
        tag = f"direct_lgbm_{base_tag}"
        print(f"\n[experiments] === {tag} ===", flush=True)
        res = direct.run(panel, segments=segments)
        metrics = direct._metrics_to_json(res["metrics"], res["backtest"])
        _write_direct_outputs(
            tag, res, metrics,
            {"segments": {k: [v.start, v.stop] for k, v in segments.items()}},
        )
        rows.append((tag, metrics))
    return rows


def run_direct_lgbm_grid() -> list[tuple[str, dict]]:
    print("\n[experiments] loading main panel for direct LGBM grid", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    rows = []
    base_params = direct.t3.FactorConfig().lgbm_params
    for tag, param_update in DIRECT_LGBM_GRIDS.items():
        params = dict(base_params)
        params.update(param_update)
        print(f"\n[experiments] === {tag} params={param_update} ===", flush=True)
        res = direct.run(
            panel,
            segments=SEGMENTS,
            config_overrides={"lgbm_params": params},
        )
        metrics = direct._metrics_to_json(res["metrics"], res["backtest"])
        _write_direct_outputs(tag, res, metrics, {"lgbm_params": params})
        rows.append((tag, metrics))
    return rows


def run_direct_lgbm_conservative_rolling() -> list[tuple[str, dict]]:
    print("\n[experiments] loading main panel for conservative direct LGBM rolling", flush=True)
    panel = pd.read_parquet(PATHS["panel"])
    rows = []
    base_params = direct.t3.FactorConfig().lgbm_params
    params = dict(base_params)
    params.update(DIRECT_LGBM_CONSERVATIVE)
    for base_tag, segments in ROLLING_SEGMENTS.items():
        tag = f"direct_lgbm_cons_{base_tag}"
        print(f"\n[experiments] === {tag} params={DIRECT_LGBM_CONSERVATIVE} ===", flush=True)
        res = direct.run(
            panel,
            segments=segments,
            config_overrides={"lgbm_params": params},
        )
        metrics = direct._metrics_to_json(res["metrics"], res["backtest"])
        _write_direct_outputs(
            tag, res, metrics,
            {
                "segments": {k: [v.start, v.stop] for k, v in segments.items()},
                "lgbm_params": params,
            },
        )
        rows.append((tag, metrics))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiments", nargs="+", default=["all"],
        choices=(
            "all", "no_bertopic", "oof", "rolling", "direct_lgbm",
            "direct_lgbm_rolling", "direct_lgbm_grid",
            "direct_lgbm_conservative_rolling",
        ),
        help="which experiments to run",
    )
    args = ap.parse_args()

    requested = set(args.experiments)
    if "all" in requested:
        requested = {
            "no_bertopic", "oof", "rolling", "direct_lgbm",
            "direct_lgbm_rolling", "direct_lgbm_grid",
            "direct_lgbm_conservative_rolling",
        }

    rows: list[dict] = []
    if "no_bertopic" in requested:
        tag, metrics = run_no_bertopic()
        rows.append(_summary_row(tag, metrics))
    if "oof" in requested:
        tag, metrics = run_oof()
        rows.append(_summary_row(tag, metrics))
    if "rolling" in requested:
        for tag, metrics in run_rolling():
            rows.append(_summary_row(tag, metrics))
    if "direct_lgbm" in requested:
        tag, metrics = run_direct_lgbm()
        rows.append(_summary_row(tag, metrics))
    if "direct_lgbm_rolling" in requested:
        for tag, metrics in run_direct_lgbm_rolling():
            rows.append(_summary_row(tag, metrics))
    if "direct_lgbm_grid" in requested:
        for tag, metrics in run_direct_lgbm_grid():
            rows.append(_summary_row(tag, metrics))
    if "direct_lgbm_conservative_rolling" in requested:
        for tag, metrics in run_direct_lgbm_conservative_rolling():
            rows.append(_summary_row(tag, metrics))

    out_dir = PATHS["stage3_dir"].parent / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary.csv"
    summary = pd.DataFrame(rows)
    print(f"\n[experiments] writing summary: {out}", flush=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
