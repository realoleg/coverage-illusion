from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

OPTIONAL_PREDICTION_COLUMNS = [
    "subreddit",
]


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:

    """
    Compute standard binary classification metrics.
    Returns a dictionary with accuracy, precision, recall, binary F1 and macro F1.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_metrics_row(
        model_name: str,
        split_name: str,
        y_true,
        y_pred,
) -> dict[str, Any]:

    """
    Building one flat metrics row (in case of CSV export).
    """

    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "model_name": model_name,
        "split": split_name,
        "n_examples": int(len(y_true)),
        **metrics,
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:

    """
    Numerically stable softmax for turning logits into probabilities.
    """

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def build_prediction_frame(
        df: pd.DataFrame,
        split_name: str,
        model_name: str,
        pred_labels: np.ndarray,
        logits: np.ndarray | None = None,
        stress_test_name: str | None = None,
) -> pd.DataFrame:

    """
    Build a tidy prediction table for baseline or transformer predictions.
    Probabilities and logits are added only when logits are given.
    """

    columns = ["example_id", "text", "label"]
    columns += [col for col in OPTIONAL_PREDICTION_COLUMNS if col in df.columns]

    out = df[columns].copy()
    out["split"] = split_name
    if stress_test_name is not None:
        out["stress_test"] = stress_test_name
    out["model_name"] = model_name
    out["pred_label"] = pred_labels.astype(int)
    out["correct"] = (out["label"] == out["pred_label"]).astype(int)

    if logits is not None:
        probabilities = stable_softmax(logits)
        out["prob_not_stress"] = probabilities[:, 0]
        out["prob_stress"] = probabilities[:, 1]
        out["logit_not_stress"] = logits[:, 0]
        out["logit_stress"] = logits[:, 1]

    return out
