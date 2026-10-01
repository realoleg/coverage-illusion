from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.conformal import (
    build_conformal_prediction_frame,
    fit_lac_conformal,
    get_split_predictions,
    load_transformer_predictions,
    summarize_conformal_predictions,
)
from src.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run split-conformal post-processing on transformer predictions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    method_name = config["conformal"].get("method", "lac")
    alpha_values = [float(alpha) for alpha in config["conformal"]["alpha_values"]]

    if method_name != "lac":
        raise ValueError(
            f"Only method='lac' is implemented in this project, got '{method_name}'."
        )

    transformer_predictions_path = Path("results/predictions/transformer_predictions.csv")
    transformer_predictions = load_transformer_predictions(transformer_predictions_path)

    calibration_df = get_split_predictions(transformer_predictions, "calibration")
    test_df = get_split_predictions(transformer_predictions, "test")

    metrics_rows: list[dict] = []
    conformal_frames: list[pd.DataFrame] = []

    split_map = {
        "calibration": calibration_df,
        "test": test_df,
    }

    for alpha in alpha_values:
        q_hat, probability_threshold = fit_lac_conformal(
            calibration_df=calibration_df,
            alpha=alpha,
        )

        for split_name, split_df in split_map.items():
            conformal_df = build_conformal_prediction_frame(
                df=split_df,
                alpha=alpha,
                q_hat=q_hat,
                probability_threshold=probability_threshold,
            )
            conformal_frames.append(conformal_df)

            metrics_rows.append(
                summarize_conformal_predictions(
                    prediction_df=conformal_df,
                    split_name=split_name,
                    alpha=alpha,
                    method_name=method_name,
                )
            )

    metrics_out = pd.DataFrame(metrics_rows)
    predictions_out = pd.concat(conformal_frames, ignore_index=True)

    tables_dir = Path("results/tables")
    predictions_dir = Path("results/predictions")
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "conformal_metrics.csv"
    predictions_path = predictions_dir / "conformal_predictions.csv"

    metrics_out.to_csv(metrics_path, index=False)
    predictions_out.to_csv(predictions_path, index=False)

    print("Saved conformal metrics to:", metrics_path.resolve())
    print("Saved conformal predictions to:", predictions_path.resolve())

    print("\nConformal test summary:")
    test_summary = metrics_out[metrics_out["split"] == "test"].copy()
    test_summary = test_summary.sort_values("alpha", ascending=False).reset_index(drop=True)
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()