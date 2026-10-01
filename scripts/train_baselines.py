from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.baselines import (
    build_baseline_models,
    evaluate_models_on_split,
    fit_models,
    save_models,
)
from src.utils import load_processed_split, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline models.")
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

    data_dir = Path(config["data"]["output_dir"])
    max_features = int(config["baselines"]["max_features"])
    ngram_range = tuple(config["baselines"]["ngram_range"])

    train_df = load_processed_split(data_dir / "train.csv")
    validation_df = load_processed_split(data_dir / "validation.csv")
    test_df = load_processed_split(data_dir / "test.csv")

    models = build_baseline_models(
        max_features=max_features,
        ngram_range=ngram_range,
        seed=int(config["seed"]),
    )
    models = fit_models(models=models, train_df=train_df)

    split_to_df = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for split_name, df in split_to_df.items():
        metrics_df, predictions_df = evaluate_models_on_split(
            models=models,
            df=df,
            split_name=split_name,
        )
        all_metrics.append(metrics_df)
        all_predictions.append(predictions_df)
    
    metrics_out = pd.concat(all_metrics, ignore_index=True)
    predictions_out = pd.concat(all_predictions, ignore_index=True)

    tables_dir = Path("results/tables")
    predictions_dir = Path("results/predictions")
    checkpoints_dir = Path("results/checkpoints/baselines")

    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metrics_out.to_csv(tables_dir / "baseline_metrics.csv", index=False)
    predictions_out.to_csv(predictions_dir / "baseline_predictions.csv", index=False)
    save_models(models=models, output_dir=checkpoints_dir)

    print("Saved metrics to: ", (tables_dir / "baseline_metrics.csv").resolve())
    print("Saved predictions to: ", (predictions_dir / "baseline_predictions.csv").resolve())
    print("Saved fitted baseline models to: ", checkpoints_dir.resolve())

    print("\nTest-set summary:")
    test_summary = (
        metrics_out[metrics_out["split"] == "test"]
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    print(test_summary.to_string(index=False))



if __name__ == "__main__":
    main()

