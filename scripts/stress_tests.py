from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.stress_tests import (
    build_conformal_metrics_and_predictions,
    build_inference_trainer,
    build_stress_test_splits,
    load_clean_calibration_predictions,
    load_pickled_model,
    load_saved_transformer,
    predict_with_baseline,
    predict_with_transformer,
    select_best_baseline_model_name,
)
from src.utils import load_processed_split, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run degraded-input stress tests for baseline, transformer, and conformal outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    seed = int(config["seed"])
    data_dir = Path(config["data"]["output_dir"])

    model_name = str(config["transformer"]["model_name"])
    max_length = int(config["transformer"]["max_length"])
    batch_size = int(config["transformer"]["batch_size"])

    alpha_values = [float(alpha) for alpha in config["conformal"]["alpha_values"]]
    truncate_fracs = [float(x) for x in config["stress_tests"]["truncate_fracs"]]
    deletion_probs = [float(x) for x in config["stress_tests"]["deletion_probs"]]

    test_df = load_processed_split(data_dir / "test.csv")
    stress_variants = build_stress_test_splits(
        test_df=test_df,
        truncate_fracs=truncate_fracs,
        deletion_probs=deletion_probs,
        seed=seed,
    )

    baseline_metrics_path = Path("results/tables/baseline_metrics.csv")
    best_baseline_name = select_best_baseline_model_name(baseline_metrics_path)
    best_baseline_path = Path("results/checkpoints/baselines") / f"{best_baseline_name}.pkl"
    baseline_model = load_pickled_model(best_baseline_path)

    transformer_checkpoint_dir = Path("results/checkpoints/distilbert")
    transformer_model, transformer_tokenizer = load_saved_transformer(
        checkpoint_dir=transformer_checkpoint_dir,
        fallback_model_name=model_name,
    )
    inference_trainer = build_inference_trainer(
        model=transformer_model,
        tokenizer=transformer_tokenizer,
        batch_size=batch_size,
        output_dir=Path("results/checkpoints/distilbert_inference"),
    )

    calibration_predictions_df = load_clean_calibration_predictions(
        Path("results/predictions/transformer_predictions.csv")
    )

    classification_metrics_rows: list[dict] = []
    classification_prediction_frames: list[pd.DataFrame] = []

    conformal_metrics_rows: list[dict] = []
    conformal_prediction_frames: list[pd.DataFrame] = []

    for stress_test_name, variant_df in stress_variants.items():
        baseline_metrics_row, baseline_predictions_df = predict_with_baseline(
            model=baseline_model,
            df=variant_df,
            model_name=best_baseline_name,
            stress_test_name=stress_test_name,
        )
        classification_metrics_rows.append(baseline_metrics_row)
        classification_prediction_frames.append(baseline_predictions_df)

        transformer_metrics_row, transformer_predictions_df = predict_with_transformer(
            trainer=inference_trainer,
            tokenizer=transformer_tokenizer,
            df=variant_df,
            model_name="distilbert",
            stress_test_name=stress_test_name,
            max_length=max_length,
        )
        classification_metrics_rows.append(transformer_metrics_row)
        classification_prediction_frames.append(transformer_predictions_df)

        conformal_rows, conformal_frames = build_conformal_metrics_and_predictions(
            transformer_prediction_df=transformer_predictions_df,
            calibration_predictions_df=calibration_predictions_df,
            alpha_values=alpha_values,
            stress_test_name=stress_test_name,
            method_name=config["conformal"].get("method", "lac"),
        )
        conformal_metrics_rows.extend(conformal_rows)
        conformal_prediction_frames.extend(conformal_frames)

    classification_metrics_out = pd.DataFrame(classification_metrics_rows)
    classification_predictions_out = pd.concat(
        classification_prediction_frames,
        ignore_index=True,
    )

    conformal_metrics_out = pd.DataFrame(conformal_metrics_rows)
    conformal_predictions_out = pd.concat(
        conformal_prediction_frames,
        ignore_index=True,
    )

    tables_dir = Path("results/tables")
    predictions_dir = Path("results/predictions")
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    classification_metrics_path = tables_dir / "stress_test_classification_metrics.csv"
    conformal_metrics_path = tables_dir / "stress_test_conformal_metrics.csv"
    classification_predictions_path = predictions_dir / "stress_test_predictions.csv"
    conformal_predictions_path = predictions_dir / "stress_test_conformal_predictions.csv"

    classification_metrics_out.to_csv(classification_metrics_path, index=False)
    classification_predictions_out.to_csv(classification_predictions_path, index=False)
    conformal_metrics_out.to_csv(conformal_metrics_path, index=False)
    conformal_predictions_out.to_csv(conformal_predictions_path, index=False)

    print("Best baseline selected from validation:", best_baseline_name)
    print("Saved classification metrics to:", classification_metrics_path.resolve())
    print("Saved classification predictions to:", classification_predictions_path.resolve())
    print("Saved conformal metrics to:", conformal_metrics_path.resolve())
    print("Saved conformal predictions to:", conformal_predictions_path.resolve())

    print("\nClassification test summary:")
    print(
        classification_metrics_out.sort_values(
            ["stress_test", "model_name"],
            ascending=[True, True],
        ).to_string(index=False)
    )

    print("\nConformal test summary:")
    print(
        conformal_metrics_out.sort_values(
            ["stress_test", "alpha"],
            ascending=[True, False],
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()

