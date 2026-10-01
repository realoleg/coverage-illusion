from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.degradations import apply_condition
from src.evaluate import compute_classification_metrics, stable_softmax
from src.transformer import (
    EpochTimer,
    build_tokenized_splits,
    build_trainer,
    get_training_device,
    initialise_seed,
    load_transformer_splits,
    predict_logits,
    sentence_occlusion,
)
from src.utils import (
    get_commit_hash,
    get_library_versions,
    load_yaml_config,
    save_json,
    word_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one encoder (model, seed) and save its run folder.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model key from encoders.models.")
    parser.add_argument("--seed", type=int, required=True, help="Training seed.")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    started_at = now()
    start_time = time.perf_counter()

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    encoder_config = config["encoders"]
    conditions = config["conditions"]

    model_key = args.model
    model_name = encoder_config["models"][model_key]
    seed = args.seed
    max_length = int(encoder_config["max_length"])

    run_dir = Path(config["paths"]["runs_dir"]) / dataset / model_key / f"seed_{seed}"
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"]) / dataset / model_key / f"seed_{seed}"

    device = get_training_device()
    initialise_seed(seed)

    split_to_df = load_transformer_splits(data_dir=data_dir)
    tokenizer, tokenized_splits = build_tokenized_splits(
        split_to_df={name: split_to_df[name][["text", "label"]] for name in ["train", "validation"]},
        model_name=model_name,
        max_length=max_length,
    )

    epoch_timer = EpochTimer()
    trainer = build_trainer(
        model_name=model_name,
        tokenizer=tokenizer,
        output_dir=checkpoint_dir,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        batch_size=int(encoder_config["batch_size"]),
        gradient_accumulation_steps=int(encoder_config["gradient_accumulation_steps"]),
        learning_rate=float(encoder_config["learning_rate"]),
        num_train_epochs=int(encoder_config["num_train_epochs"]),
        weight_decay=float(encoder_config["weight_decay"]),
        warmup_ratio=float(encoder_config["warmup_ratio"]),
        seed=seed,
        callbacks=[epoch_timer],
    )
    assert trainer.args.device.type == device, (trainer.args.device, device)

    train_start = time.perf_counter()
    trainer.train()
    train_seconds = round(time.perf_counter() - train_start, 1)

    # Predictions with the best-epoch weights: validation clean only; calibration and test in every condition.
    prediction_frames: list[pd.DataFrame] = []
    clean_metrics: dict[str, dict[str, float]] = {}

    for split_name in ["validation", "calibration", "test"]:
        df = split_to_df[split_name]
        split_conditions = ["clean"] if split_name == "validation" else conditions

        for condition in split_conditions:
            texts = apply_condition(df, condition)
            logits = predict_logits(
                trainer, tokenizer, texts, max_length, desc=f"Tokenizing {split_name}/{condition}"
            )
            probabilities = stable_softmax(logits)

            prediction_frames.append(pd.DataFrame({
                "id": df["id"].to_numpy(),
                "split": split_name,
                "condition": condition,
                "label": df["label"].to_numpy(),
                "n_words": word_count(texts).to_numpy(),
                "prob_not_stress": probabilities[:, 0],
                "prob_stress": probabilities[:, 1],
                "logit_not_stress": logits[:, 0],
                "logit_stress": logits[:, 1],
            }))

            if condition == "clean":
                clean_metrics[split_name] = compute_classification_metrics(
                    df["label"].to_numpy(), np.argmax(logits, axis=-1)
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    with_occlusion = dataset == "dreaddit" and model_key in encoder_config["occlusion_models"]
    if with_occlusion:
        occlusion = sentence_occlusion(trainer, tokenizer, split_to_df["test"], max_length)
        occlusion.to_csv(run_dir / "occlusion.csv", index=False)

    log_history = trainer.state.log_history
    best_metric = trainer.state.best_metric
    best_epoch = next(
        entry["epoch"] for entry in log_history if entry.get("eval_macro_f1") == best_metric
    )
    save_json(
        {
            "epoch_seconds": epoch_timer.epoch_seconds,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_metric,
            "log_history": log_history,
        },
        run_dir / "train_log.json",
    )

    # The run keeps only predictions: the checkpoint is not stored.
    shutil.rmtree(checkpoint_dir)

    save_json(
        {
            "dataset": dataset,
            "model": model_key,
            "model_name": model_name,
            "seed": seed,
            "commit": get_commit_hash(),
            "device": str(trainer.args.device),
            "batch_size": int(encoder_config["batch_size"]),
            "gradient_accumulation_steps": int(encoder_config["gradient_accumulation_steps"]),
            "effective_batch_size": int(encoder_config["batch_size"])
            * int(encoder_config["gradient_accumulation_steps"]),
            # Batch 8 x accumulation 2 replaces batch 16 after an MPS out-of-memory error.
            "oom_fallback": int(encoder_config["gradient_accumulation_steps"]) > 1,
            "conditions": conditions,
            "occlusion": with_occlusion,
            "n_rows": {name: len(df) for name, df in split_to_df.items()},
            "clean_metrics": clean_metrics,
            "train_seconds": train_seconds,
            "epoch_seconds": epoch_timer.epoch_seconds,
            "total_seconds": round(time.perf_counter() - start_time, 1),
            "started_at": started_at,
            "finished_at": now(),
            "versions": get_library_versions(),
            "config": {
                "data": config["data"],
                "encoders": encoder_config,
                "conditions": conditions,
            },
        },
        run_dir / "run_meta.json",
    )

    print("Saved run folder to: ", run_dir.resolve())
    print(f"Best epoch: {best_epoch}, epoch seconds: {epoch_timer.epoch_seconds}, train seconds: {train_seconds}")
    print(pd.DataFrame(clean_metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()
