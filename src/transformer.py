from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.evaluate import compute_classification_metrics, stable_softmax
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
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_train_epochs: int,
        weight_decay: float,
        warmup_ratio: float,
        torch_empty_cache_steps: int,
        seed: int,
        callbacks: list[TrainerCallback] | None = None,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        # transformers 5: a float < 1 is a fraction of total steps (warmup_ratio is deprecated).
        warmup_steps=warmup_ratio,
        # Releases the MPS allocator cache; it grows with dynamic padding until out of memory.
        torch_empty_cache_steps=torch_empty_cache_steps,
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
        callbacks=callbacks,
    )

    return trainer


class EpochTimer(TrainerCallback):

    """
    Wall-clock seconds of each training epoch (excludes the end-of-epoch evaluation and saving).
    """

    def __init__(self) -> None:
        self.epoch_seconds: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_seconds.append(round(time.perf_counter() - self._start, 1))


def get_training_device() -> str:

    """
    Encoders train on MPS; without MPS they fall back to CPU with a warning.
    """

    if torch.backends.mps.is_available():
        return "mps"

    print("WARNING: MPS is not available, training on CPU.")
    return "cpu"


def predict_logits(
        trainer: Trainer,
        tokenizer,
        texts: pd.Series,
        max_length: int,
        desc: str,
) -> np.ndarray:

    """
    Logits of shape (n_texts, 2), in the order of texts.
    Inputs are batched in order of token length (fewer batch shapes, less padding),
    then restored to the original order; the MPS cache is released after the pass.
    """

    dataset = tokenize_dataframe(
        df=pd.DataFrame({"text": texts.astype(str).to_numpy()}),
        tokenizer=tokenizer,
        max_length=max_length,
        desc=desc,
    )
    lengths = np.array([len(input_ids) for input_ids in dataset["input_ids"]])
    order = np.argsort(lengths, kind="stable")

    sorted_logits = np.asarray(trainer.predict(dataset.select(order)).predictions)
    logits = np.empty_like(sorted_logits)
    logits[order] = sorted_logits
    clear_device_cache()

    assert logits.shape == (len(texts), 2), logits.shape
    return logits


def clear_device_cache() -> None:

    """
    Release cached MPS memory that no tensor uses (no effect on results).
    """

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def split_sentences(text: str) -> list[str]:

    """
    Sentence split with nltk punkt (downloaded on first use).
    """

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    return nltk.sent_tokenize(text)


def sentence_occlusion(
        trainer: Trainer,
        tokenizer,
        df: pd.DataFrame,
        max_length: int,
) -> pd.DataFrame:

    """
    For every sentence of every text with >= 2 sentences: change in p(stress) when the sentence is removed.
    The reference is the text re-joined from all its sentences, so the delta reflects only the removal.
    Position: first / last / middle sentence.
    """

    rows: list[dict[str, Any]] = []
    for example_id, label, text in zip(df["id"], df["label"], df["text"].astype(str)):
        sentences = split_sentences(text)
        n_sentences = len(sentences)
        if n_sentences < 2:
            continue

        rows.append({
            "id": example_id, "label": label, "n_sentences": n_sentences,
            "sentence_idx": -1, "text": " ".join(sentences),
        })
        for idx in range(n_sentences):
            rows.append({
                "id": example_id, "label": label, "n_sentences": n_sentences,
                "sentence_idx": idx, "text": " ".join(sentences[:idx] + sentences[idx + 1:]),
            })

    variants = pd.DataFrame(rows)
    logits = predict_logits(trainer, tokenizer, variants["text"], max_length, desc="Tokenizing occlusion")
    variants["prob_stress"] = stable_softmax(logits)[:, 1]

    is_reference = variants["sentence_idx"] == -1
    reference = variants[is_reference].set_index("id")["prob_stress"]

    out = variants[~is_reference].drop(columns="text").reset_index(drop=True)
    out["prob_stress_full"] = out["id"].map(reference)
    out["delta_prob_stress"] = out["prob_stress"] - out["prob_stress_full"]
    out["position"] = np.select(
        [out["sentence_idx"] == 0, out["sentence_idx"] == out["n_sentences"] - 1],
        ["first", "last"],
        default="middle",
    )
    out = out.rename(columns={"prob_stress": "prob_stress_without"})

    return out[[
        "id", "label", "n_sentences", "sentence_idx", "position",
        "prob_stress_full", "prob_stress_without", "delta_prob_stress",
    ]]


def initialise_seed(seed: int) -> None:

    """
    Set random seed for transformer / numpy / pytorch if needed.
    """

    set_seed(seed)

