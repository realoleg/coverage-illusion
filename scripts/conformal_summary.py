from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.conformal import (
    METHODS,
    build_eval_frame,
    build_mix_calibration,
    calibration_frame,
    group_summaries,
    indicator_frame,
    predict_sets,
    reweight,
    worst_groups,
)
from src.utils import load_yaml_config

OVERALL_METRICS = [
    "n", "coverage", "set_size", "singleton", "empty", "full", "flag", "dismiss",
    "workload", "uncertainty_referral", "sensitivity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conformal summary of every run on the fixed calibration/test split.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def summarize_run(
        predictions: pd.DataFrame,
        metadata: pd.DataFrame,
        mix_assignment: pd.DataFrame,
        conditions: list[str],
        alphas: list[float],
        prevalence: float,
) -> list[dict[str, Any]]:

    """
    One row per (test condition, method, alpha): overall metrics, class coverage with bands,
    worst groups with bands, and metrics at the main prevalence. No per-group rows.
    """

    frames = {
        (split_name, condition): build_eval_frame(group, metadata)
        for (split_name, condition), group in predictions.groupby(["split", "condition"])
        if split_name in ("calibration", "test")
    }
    calibration_by_condition = {condition: frames[("calibration", condition)] for condition in conditions}
    mix_calibration = build_mix_calibration(calibration_by_condition, mix_assignment)

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        test = frames[("test", condition)]
        labels = test["label"].to_numpy()

        for method, spec in METHODS.items():
            calibration = calibration_frame(method, calibration_by_condition, mix_calibration, condition)
            for alpha in alphas:
                sets, n_infinite = predict_sets(calibration, test, method, alpha)
                groups = group_summaries(indicator_frame(sets, labels), test, calibration, alpha)

                overall = groups[groups["group_type"] == "all"].iloc[0]
                by_class = groups[groups["group_type"] == "class"].set_index("group")
                at_prevalence = reweight(overall.to_dict(), prevalence)

                row: dict[str, Any] = {
                    "condition": condition,
                    "method": method,
                    "calibration": spec["calibration"],
                    "alpha": alpha,
                    "n_infinite_thresholds": n_infinite,
                    **{metric: overall[metric] for metric in OVERALL_METRICS},
                    "coverage_band_low": overall["band_low"],
                    "coverage_band_high": overall["band_high"],
                }
                for label in ["0", "1"]:
                    row[f"c{label}"] = by_class.loc[label, "coverage"]
                    row[f"c{label}_n_test"] = int(by_class.loc[label, "n"])
                    row[f"c{label}_n_calibration"] = int(by_class.loc[label, "n_calibration"])
                    row[f"c{label}_band_low"] = by_class.loc[label, "band_low"]
                    row[f"c{label}_band_high"] = by_class.loc[label, "band_high"]
                row.update(worst_groups(groups))
                row["prevalence"] = prevalence
                row["coverage_at_prevalence"] = at_prevalence["coverage"]
                row["workload_per_1000_at_prevalence"] = 1000 * at_prevalence["workload"]
                row["uncertainty_referrals_per_1000_at_prevalence"] = 1000 * at_prevalence["uncertainty_referral"]
                rows.append(row)

    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conditions = config["conditions"]
    alphas = [float(alpha) for alpha in config["conformal"]["alpha_values"]]
    prevalence = float(config["conformal"]["main_prevalence"])

    metadata = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["calibration", "test"]])
    mix_assignment = pd.read_csv(data_dir / f"{dataset}_mix_assignment.csv")

    rows: list[dict[str, Any]] = []
    for predictions_path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        predictions = pd.read_csv(predictions_path)
        if "prob_stress" not in predictions.columns:
            continue  # LinearSVC has no probabilities and is not part of conformal prediction.

        model = predictions_path.parent.parent.name
        seed = int(predictions_path.parent.name.removeprefix("seed_"))
        for row in summarize_run(predictions, metadata, mix_assignment, conditions, alphas, prevalence):
            rows.append({"dataset": dataset, "model": model, "seed": seed, **row})
        print(f"Summarized {model} seed {seed}")

    summary = pd.DataFrame(rows)
    output_dir = Path("results/conformal")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "step5_summary.csv", index=False)
    print(f"Saved summary to: {(output_dir / 'step5_summary.csv').resolve()} ({len(summary)} rows)")

    main_alpha = alphas[0]
    view = summary[(summary["alpha"] == main_alpha) & summary["condition"].isin(["clean", "head_25"])]
    print(
        view.groupby(["model", "condition", "method"])[
            ["coverage", "c0", "c1", "workload", "workload_per_1000_at_prevalence"]
        ].median().round(3).to_string()
    )
    assert np.isfinite(summary["coverage"]).all()


if __name__ == "__main__":
    main()
