from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import build_metrics_row, build_prediction_frame


def build_baseline_models(
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1,2),
        seed: int = 42,
) -> dict[str, Pipeline]:

    """
    Create two baseline pipelines (tfidf+logreg; tfidf+linearSVM).
    """

    vectorized_kwargs = {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "lowercase": True,
        "strip_accents": "unicode",
    }

    models = {
        "tfidf_logreg": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**vectorized_kwargs)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**vectorized_kwargs)),
                (
                    "clf",
                    LinearSVC(
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    return models


def fit_models(
        models: dict[str, Pipeline],
        train_df: pd.DataFrame,
) -> dict[str, Pipeline]:
    
    """
    Fit each baseline model on training DataFrame.
    """

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy()

    for model in models.values():
        model.fit(x_train, y_train)
    
    return models


def evaluate_models_on_split(
        models: dict[str, Pipeline],
        df: pd.DataFrame,
        split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Run predictions for all models on one split and return two df: metrics and predicitons.
    """

    x = df["text"].tolist()
    y = df["label"].to_numpy()

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in models.items():
        y_pred = model.predict(x).astype(int)

        metrics_rows.append(
            build_metrics_row(
                model_name=model_name,
                split_name=split_name,
                y_true=y,
                y_pred=y_pred,
            )
        )
        prediction_frames.append(
            build_prediction_frame(
                df=df,
                split_name=split_name,
                model_name=model_name,
                pred_labels=y_pred,
            )
        )
    
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return metrics_df, predictions_df


def save_models(
        models: dict[str, Pipeline],
        output_dir: str | Path,
) -> None:
    
    """
    Save fitted baseline pipelines for potential reuse.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        save_path = output_path / f"{model_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
