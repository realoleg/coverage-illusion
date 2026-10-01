from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import word_count


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


def stable_softmax(logits: np.ndarray) -> np.ndarray:

    """
    Numerically stable softmax for turning logits into probabilities.
    """

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def build_prediction_rows(
        ids: pd.Series,
        split_name: str,
        condition: str,
        labels: pd.Series,
        texts: pd.Series,
        logits: np.ndarray | None = None,
        decision_scores: np.ndarray | None = None,
) -> pd.DataFrame:

    """
    Prediction rows of one (split, condition) in the shared run schema. Texts are used only for n_words
    and are never stored. Probabilistic models give 2-class logits; LinearSVC gives decision scores.
    """

    out = pd.DataFrame({
        "id": ids.to_numpy(),
        "split": split_name,
        "condition": condition,
        "label": labels.to_numpy(),
        "n_words": word_count(texts).to_numpy(),
    })

    if logits is not None:
        assert logits.shape == (len(out), 2), logits.shape
        probabilities = stable_softmax(logits)
        out["prob_not_stress"] = probabilities[:, 0]
        out["prob_stress"] = probabilities[:, 1]
        out["logit_not_stress"] = logits[:, 0]
        out["logit_stress"] = logits[:, 1]

    if decision_scores is not None:
        assert decision_scores.shape == (len(out),), decision_scores.shape
        out["decision_score"] = decision_scores

    return out


def predicted_labels(predictions: pd.DataFrame) -> np.ndarray:

    """
    Hard labels: argmax of probabilities, or the sign of the decision score.
    """

    if "prob_stress" in predictions.columns:
        return (predictions["prob_stress"] > predictions["prob_not_stress"]).astype(int).to_numpy()
    return (predictions["decision_score"] > 0).astype(int).to_numpy()


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> None:

    """
    Save predictions as gzip CSV with a fixed gzip timestamp, so identical predictions give identical bytes.
    """

    assert "text" not in predictions.columns
    predictions.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
