from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "not_stress",
    1: "stress",
}

REQUIRED_COLUMNS = [
    "split",
    "label",
    "prob_not_stress",
    "prob_stress",
]


def load_transformer_predictions(path: str | Path) -> pd.DataFrame:
    
    """
    Load saved transformer predictions and validate the columns required for conformal post-processing.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transformer predictions file not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in transformer predictions file: {missing}"
        )

    out = df.copy()
    out["split"] = out["split"].astype(str)
    out["label"] = out["label"].astype(int)
    out["prob_not_stress"] = out["prob_not_stress"].astype(float)
    out["prob_stress"] = out["prob_stress"].astype(float)

    return out


def get_split_predictions(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    
    """
    Select one split from the saved transformer predictions.
    """

    out = df[df["split"] == split_name].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows found for split='{split_name}'.")
    return out


def extract_probability_matrix(df: pd.DataFrame) -> np.ndarray:
    
    """
    Extract the probability matrix of shape (n_samples, 2):
    column 0 -> not_stress
    column 1 -> stress
    """

    probs = df[["prob_not_stress", "prob_stress"]].to_numpy(dtype=float)

    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("Expected probability matrix with shape (n_samples, 2).")

    return probs


def compute_lac_scores(probabilities: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
   
    """
    LAC nonconformity scores:
        score_i = 1 - p_true_label(x_i)
    """

    row_idx = np.arange(len(true_labels))
    true_class_probs = probabilities[row_idx, true_labels]
    scores = 1.0 - true_class_probs
    return scores.astype(float)


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    
    """
    Split-conformal quantile with finite-sample correction:
        q_hat = Quantile(scores; ceil((n + 1) * (1 - alpha)) / n)

    Uses a conservative 'higher' rule.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile from an empty score array.")

    quantile_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    quantile_level = min(float(quantile_level), 1.0)

    q_hat = np.quantile(scores, quantile_level, method="higher")
    return float(q_hat)


def compute_probability_threshold(q_hat: float) -> float:
    
    """
    Under LAC, include class y if:
        p_y(x) >= 1 - q_hat
    """

    threshold = 1.0 - q_hat
    return float(np.clip(threshold, 0.0, 1.0))


def build_prediction_set_mask(
    probabilities: np.ndarray,
    probability_threshold: float,
) -> np.ndarray:
    
    """
    Build boolean prediction-set mask of shape (n_samples, 2).
    """

    return probabilities >= probability_threshold


def format_prediction_set(mask_row: np.ndarray) -> str:
   
    """
    Convert one boolean mask into a readable set string.
    """

    labels = [LABEL_NAMES[idx] for idx, include in enumerate(mask_row) if include]
    if not labels:
        return "{}"
    return "{" + ", ".join(labels) + "}"


def fit_lac_conformal(
    calibration_df: pd.DataFrame,
    alpha: float,
) -> tuple[float, float]:
    
    """
    Fit split-conformal LAC on the calibration split.

    Returns:
        q_hat, probability_threshold
    """

    probabilities = extract_probability_matrix(calibration_df)
    true_labels = calibration_df["label"].to_numpy(dtype=int)

    scores = compute_lac_scores(
        probabilities=probabilities,
        true_labels=true_labels,
    )
    q_hat = compute_conformal_quantile(scores=scores, alpha=alpha)
    probability_threshold = compute_probability_threshold(q_hat=q_hat)

    return q_hat, probability_threshold


def build_conformal_prediction_frame(
    df: pd.DataFrame,
    alpha: float,
    q_hat: float,
    probability_threshold: float,
) -> pd.DataFrame:
    
    """
    Add conformal set-valued outputs to a copy of the original prediction DataFrame.
    """

    out = df.copy()
    probabilities = extract_probability_matrix(out)
    prediction_set_mask = build_prediction_set_mask(
        probabilities=probabilities,
        probability_threshold=probability_threshold,
    )

    set_sizes = prediction_set_mask.sum(axis=1).astype(int)
    row_idx = np.arange(len(out))
    true_labels = out["label"].to_numpy(dtype=int)
    contains_true = prediction_set_mask[row_idx, true_labels].astype(int)

    singleton_pred_label_id = np.full(len(out), fill_value=-1, dtype=int)
    singleton_rows = set_sizes == 1
    if np.any(singleton_rows):
        singleton_pred_label_id[singleton_rows] = np.argmax(
            prediction_set_mask[singleton_rows],
            axis=1,
        )

    singleton_pred_label_name = [
        LABEL_NAMES[idx] if idx in LABEL_NAMES else None
        for idx in singleton_pred_label_id
    ]

    out["alpha"] = float(alpha)
    out["target_coverage"] = float(1.0 - alpha)
    out["q_hat"] = float(q_hat)
    out["probability_threshold"] = float(probability_threshold)

    out["prediction_set"] = [
        format_prediction_set(mask_row) for mask_row in prediction_set_mask
    ]
    out["set_size"] = set_sizes
    out["contains_true"] = contains_true
    out["is_singleton"] = (set_sizes == 1).astype(int)
    out["is_empty_set"] = (set_sizes == 0).astype(int)
    out["is_full_set"] = (set_sizes == 2).astype(int)

    out["singleton_pred_label_id"] = singleton_pred_label_id
    out["singleton_pred_label"] = singleton_pred_label_name

    return out


def summarize_conformal_predictions(
    prediction_df: pd.DataFrame,
    split_name: str,
    alpha: float,
    method_name: str = "lac",
) -> dict[str, Any]:
    
    """
    Compute summary conformal metrics for one split.
    """
    
    if prediction_df.empty:
        raise ValueError("prediction_df is empty.")

    return {
        "method": method_name,
        "split": split_name,
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "n_examples": int(len(prediction_df)),
        "q_hat": float(prediction_df["q_hat"].iloc[0]),
        "probability_threshold": float(prediction_df["probability_threshold"].iloc[0]),
        "empirical_coverage": float(prediction_df["contains_true"].mean()),
        "avg_set_size": float(prediction_df["set_size"].mean()),
        "singleton_rate": float(prediction_df["is_singleton"].mean()),
        "empty_rate": float(prediction_df["is_empty_set"].mean()),
        "full_set_rate": float(prediction_df["is_full_set"].mean()),
    }