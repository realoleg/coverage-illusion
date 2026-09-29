from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.conformal import (
    build_conformal_prediction_frame,
    fit_lac_conformal,
    get_split_predictions,
    load_transformer_predictions,
    summarize_conformal_predictions,
)
from src.evaluate import build_metrics_row, build_prediction_frame
from src.transformer import predict_on_split, tokenize_dataframe
from src.utils import resolve_metric_column, add_text_length_features


def select_best_baseline_model_name(metrics_path: str | Path) -> str:
   
    """
    Select the best baseline model using validation performance.
    """

    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    validation_df = metrics_df[metrics_df["split"] == "validation"].copy()

    if validation_df.empty:
        raise ValueError("No validation rows found in baseline metrics file.")

    metric_column = resolve_metric_column(validation_df, preferred="macro_f1")
    best_row = validation_df.sort_values(metric_column, ascending=False).iloc[0]

    return str(best_row["model_name"])


def load_pickled_model(path: str | Path):
    
    """
    Load a fitted sklearn model from pickle.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickled model not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def truncate_text(text: str, fraction: float) -> str:
   
    """
    Keep only the first given fraction of whitespace-tokenized words.
    Always keeps at least one word if the input is non-empty.
    """

    words = text.split()
    if not words:
        return text

    keep_n = max(1, int(np.ceil(len(words) * fraction)))
    return " ".join(words[:keep_n])


def random_word_deletion(text: str, deletion_prob: float, seed: int) -> str:
    
    """
    Randomly delete words with a fixed probability.
    Uses a deterministic seed so the perturbation is reproducible.
    Always keeps at least one word if the input is non-empty.
    """

    words = text.split()
    if len(words) <= 1:
        return text

    rng = np.random.default_rng(seed)
    keep_mask = rng.random(len(words)) > deletion_prob
    kept_words = [word for word, keep in zip(words, keep_mask) if keep]

    if not kept_words:
        random_idx = int(rng.integers(low=0, high=len(words)))
        kept_words = [words[random_idx]]

    return " ".join(kept_words)


def build_stress_test_splits(
    test_df: pd.DataFrame,
    truncate_fracs: list[float],
    deletion_probs: list[float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    
    """
    Build the clean and degraded variants of the test split.
    """

    variants: dict[str, pd.DataFrame] = {}

    clean_df = test_df.copy()
    clean_df["original_text"] = clean_df["text"]
    clean_df["stress_test"] = "clean"
    clean_df = add_text_length_features(clean_df)
    variants["clean"] = clean_df

    for frac in truncate_fracs:
        variant_name = f"truncate_{int(frac * 100)}"
        df = test_df.copy()
        df["original_text"] = df["text"]
        df["text"] = df["text"].astype(str).apply(lambda x: truncate_text(x, fraction=frac))
        df["stress_test"] = variant_name
        df = add_text_length_features(df)
        variants[variant_name] = df

    for deletion_prob in deletion_probs:
        variant_name = f"delete_{int(deletion_prob * 100)}"
        df = test_df.copy()
        df["original_text"] = df["text"]

        df["text"] = df.apply(
            lambda row: random_word_deletion(
                text=str(row["text"]),
                deletion_prob=deletion_prob,
                seed=seed + int(row["example_id"]) + int(deletion_prob * 1000),
            ),
            axis=1,
        )
        df["stress_test"] = variant_name
        df = add_text_length_features(df)
        variants[variant_name] = df

    return variants


def predict_with_baseline(
    model,
    df: pd.DataFrame,
    model_name: str,
    stress_test_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    
    """
    Run one fitted baseline model on one stress-test split.
    """

    x = df["text"].astype(str).tolist()
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = model.predict(x).astype(int)

    metrics_row = build_metrics_row(
        model_name=model_name,
        split_name="test",
        y_true=y_true,
        y_pred=y_pred,
    )
    metrics_row["stress_test"] = stress_test_name

    predictions_df = build_prediction_frame(
        df=df,
        split_name="test",
        model_name=model_name,
        pred_labels=y_pred,
        stress_test_name=stress_test_name,
    )

    return metrics_row, predictions_df


def load_saved_transformer(
    checkpoint_dir: str | Path,
    fallback_model_name: str,
) -> tuple[Any, Any]:
    
    """
    Load the saved transformer checkpoint and tokenizer.
    Falls back to the base tokenizer name if tokenizer files are absent.
    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {checkpoint_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)

    return model, tokenizer


def build_inference_trainer(
    model,
    tokenizer,
    batch_size: int,
    output_dir: str | Path,
) -> Trainer:
    
    """
    Build a lightweight Trainer for batched transformer inference.
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )

    return Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
    )


def predict_with_transformer(
    trainer: Trainer,
    tokenizer,
    df: pd.DataFrame,
    model_name: str,
    stress_test_name: str,
    max_length: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    
    """
    Run the saved transformer checkpoint on one stress-test split.
    """

    dataset = tokenize_dataframe(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
        desc="Tokenizing stress-test split",
    )

    return predict_on_split(
        trainer=trainer,
        dataset=dataset,
        original_df=df,
        split_name="test",
        model_name=model_name,
        stress_test_name=stress_test_name,
    )


def build_conformal_metrics_and_predictions(
    transformer_prediction_df: pd.DataFrame,
    calibration_predictions_df: pd.DataFrame,
    alpha_values: list[float],
    stress_test_name: str,
    method_name: str = "lac",
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    
    """
    Apply split-conformal post-processing to one transformer prediction table.
    Calibration always comes from the clean held-out calibration split.
    """

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for alpha in alpha_values:
        q_hat, probability_threshold = fit_lac_conformal(
            calibration_df=calibration_predictions_df,
            alpha=alpha,
        )

        conformal_df = build_conformal_prediction_frame(
            df=transformer_prediction_df,
            alpha=alpha,
            q_hat=q_hat,
            probability_threshold=probability_threshold,
        )
        conformal_df["model_name"] = "distilbert_conformal"
        conformal_df["stress_test"] = stress_test_name

        metrics_row = summarize_conformal_predictions(
            prediction_df=conformal_df,
            split_name="test",
            alpha=alpha,
            method_name=method_name,
        )
        metrics_row["model_name"] = "distilbert_conformal"
        metrics_row["stress_test"] = stress_test_name

        prediction_frames.append(conformal_df)
        metrics_rows.append(metrics_row)

    return metrics_rows, prediction_frames


def load_clean_calibration_predictions(
    path: str | Path,
) -> pd.DataFrame:
    
    """
    Load the saved clean transformer predictions and return the calibration split.
    """

    transformer_predictions = load_transformer_predictions(path)
    return get_split_predictions(transformer_predictions, "calibration")

