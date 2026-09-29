from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.evaluate import (
    build_metrics_row,
    build_prediction_frame,
    compute_classification_metrics,
)
from src.utils import load_processed_split


def load_transformer_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:

    """
    Load all processed splits needed for transformer.
    """

    data_dir = Path(data_dir)

    return {
        "train": load_processed_split(data_dir / "train.csv"),
        "calibration": load_processed_split(data_dir / "calibration.csv"),
        "validation": load_processed_split(data_dir / "validation.csv"),
        "test": load_processed_split(data_dir / "test.csv"),
    }


def get_label_mappings() -> tuple[dict[str, int], dict[int, str]]:

    """
    Dreaddit is a binary stress classification task.
    """

    label2id = {
        "not_stress": 0,
        "stress": 1,
    }
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def tokenize_dataframe(
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        desc: str,
) -> Dataset:
    
    """
    Convert a DataFrame into a tokenized Hugging Face dataset.
    """

    dataset = Dataset.from_pandas(df, preserve_index=False)

    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")

    def tokenize_batch(batch: dict[str, list]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True, desc=desc)


def build_tokenized_splits(
        split_to_df: dict[str, pd.DataFrame],
        model_name: str,
        max_length: int,
) -> tuple[Any, dict[str, Dataset]]:
    
    """
    Builds a tokenizer and tokenize each split into a Hugging Face dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_splits = {
        split_name: tokenize_dataframe(
            df=df,
            tokenizer=tokenizer,
            max_length=max_length,
            desc=f"Tokenizing {split_name}",
        )
        for split_name, df in split_to_df.items()
    }

    return tokenizer, tokenized_splits


def compute_trainer_metrics(eval_pred) -> dict[str, float]:

    """
    Metric function used by Hugging Face trainer.
    """

    logits, labels = eval_pred
    pred_labels = np.argmax(logits, axis=-1)
    return compute_classification_metrics(labels, pred_labels)


def build_trainer(
        model_name: str,
        tokenizer,
        output_dir: str | Path,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int,
        learning_rate: float,
        num_train_epochs: int,
        weight_decay: float,
        seed: int,
) -> Trainer:
    
    """
    Create the model, tokenizer-related collation, training arguments, and trainer.
    """

    label2id, id2label = get_label_mappings()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_trainer_metrics,
    )

    return trainer


def predict_on_split(
        trainer: Trainer,
        dataset: Dataset,
        original_df: pd.DataFrame,
        split_name: str,
        model_name: str,
        stress_test_name: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    
    """
    Run prediction on one split and return: one metrics row, one predictions dataframe.
    """

    prediction_output = trainer.predict(dataset)
    logits = np.asarray(prediction_output.predictions)
    pred_labels = np.argmax(logits, axis=-1)

    metrics_row = build_metrics_row(
        model_name=model_name,
        split_name=split_name,
        y_true=original_df["label"].to_numpy(),
        y_pred=pred_labels,
    )
    if stress_test_name is not None:
        metrics_row["stress_test"] = stress_test_name

    predictions_df = build_prediction_frame(
        df=original_df,
        split_name=split_name,
        model_name=model_name,
        pred_labels=pred_labels,
        logits=logits,
        stress_test_name=stress_test_name,
    )

    return metrics_row, predictions_df


def save_training_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    
    """
    Save training metrics to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def initialise_seed(seed: int) -> None:

    """
    Set random seed for transformer / numpy / pytorch if needed.
    """

    set_seed(seed)

