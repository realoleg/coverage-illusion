from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_vectorizer(max_features: int, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )


def build_logreg(max_features: int, ngram_range: tuple[int, int], c_value: float, seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("tfidf", build_vectorizer(max_features, ngram_range)),
        ("clf", LogisticRegression(C=c_value, max_iter=2000, solver="liblinear", random_state=seed)),
    ])


def build_linear_svm(max_features: int, ngram_range: tuple[int, int], seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("tfidf", build_vectorizer(max_features, ngram_range)),
        ("clf", LinearSVC(max_iter=5000, random_state=seed)),
    ])


def fit_logreg_with_c_selection(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        c_grid: list[float],
        max_features: int,
        ngram_range: tuple[int, int],
        seed: int,
) -> tuple[Pipeline, float, dict[float, float]]:

    """
    Fit LogReg on train for every C; pick the C with the best validation macro-F1 (ties -> smaller C).
    Returns the model fitted on train with that C, the C, and the validation macro-F1 per C.
    """

    models: dict[float, Pipeline] = {}
    scores: dict[float, float] = {}
    for c_value in sorted(c_grid):
        model = build_logreg(max_features, ngram_range, c_value, seed)
        model.fit(train_df["text"].tolist(), train_df["label"].to_numpy())
        y_pred = model.predict(validation_df["text"].tolist())
        models[c_value] = model
        scores[c_value] = float(f1_score(validation_df["label"], y_pred, average="macro"))

    best_score = max(scores.values())
    best_c = min(c_value for c_value, score in scores.items() if score == best_score)

    return models[best_c], best_c, scores


def logreg_logits(model: Pipeline, texts: pd.Series) -> np.ndarray:

    """
    2-class logits (0, decision_function): their softmax equals predict_proba.
    """

    decision = model.decision_function(texts.tolist())
    logits = np.column_stack([np.zeros_like(decision), decision])

    probabilities = model.predict_proba(texts.tolist())
    assert np.allclose(1.0 / (1.0 + np.exp(-decision)), probabilities[:, 1])
    return logits
