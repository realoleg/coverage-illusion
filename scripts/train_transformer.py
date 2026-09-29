from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.transformer import (
    build_tokenized_splits,
    build_trainer,
    initialise_seed,
    load_transformer_splits,
    predict_on_split,
    save_training_metrics,
)
from src.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilBERT on Dreaddit.")
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

    model_name = config["transformer"]["model_name"]
    max_length = int(config["transformer"]["max_length"])
    batch_size = int(config["transformer"]["batch_size"])
    learning_rate = float(config["transformer"]["learning_rate"])
    num_train_epochs = int(config["transformer"]["num_train_epochs"])
    weight_decay = float(config["transformer"]["weight_decay"])

    initialise_seed(seed)

    split_to_df = load_transformer_splits(data_dir=data_dir)
    tokenizer, tokenized_splits = build_tokenized_splits(
        split_to_df=split_to_df,
        model_name=model_name,
        max_length=max_length,
    )

    checkpoint_dir = Path("results/checkpoints/distilbert")
    tables_dir = Path("results/tables")
    predictions_dir = Path("results/predictions")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(
        model_name=model_name,
        tokenizer=tokenizer,
        output_dir=checkpoint_dir,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    # Final model is saved in checkpoint_dir, intermediate epoch checkpoints are not needed.
    for epoch_checkpoint in checkpoint_dir.glob("checkpoint-*"):
        shutil.rmtree(epoch_checkpoint)

    save_training_metrics(
        metrics=train_result.metrics,
        output_path=checkpoint_dir / "train_metrics.json",
    )

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_name in ["validation", "calibration", "test"]:
        metrics_row, predictions_df = predict_on_split(
            trainer=trainer,
            dataset=tokenized_splits[split_name],
            original_df=split_to_df[split_name],
            split_name=split_name,
            model_name="distilbert",
        )
        metrics_rows.append(metrics_row)
        prediction_frames.append(predictions_df)
    
    metrics_out = pd.DataFrame(metrics_rows)
    predictions_out = pd.concat(prediction_frames, ignore_index=True)

    metrics_path = tables_dir / "transformer_metrics.csv"
    predictions_path = predictions_dir / "transformer_predictions.csv"

    metrics_out.to_csv(metrics_path, index=False)
    predictions_out.to_csv(predictions_path, index=False)

    print("Saved checkpoint to: ", checkpoint_dir.resolve())
    print("Saved training metrics to: ", (checkpoint_dir / "train_metrics.json").resolve())
    print("Saved evaluation metrics to: ", metrics_path.resolve())
    print("Saved predictions to: ", predictions_path.resolve())

    print("\nTransformer summary: ")
    print(metrics_out.sort_values("macro_f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

