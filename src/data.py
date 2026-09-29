from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ["text", "label", "subreddit"]


def load_dreaddit_dataset(dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset = load_dataset(dataset_name)

    splits = {}
    for split_name in ["train", "validation", "test"]:
        splits[split_name] = dataset[split_name].to_pandas()

    return splits


def keep_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    out = df[REQUIRED_COLUMNS].copy()
    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    out["subreddit"] = out["subreddit"].astype(str)
    return out


def make_train_calibration_split(
        train_df: pd.DataFrame,
        calibration_size: float = 0.2,
        seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proper_train, calibration = train_test_split(
        train_df,
        test_size=calibration_size,
        random_state=seed,
        stratify=train_df["label"],
    )
    return proper_train.reset_index(drop=True), calibration.reset_index(drop=True)


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        df.to_csv(output_path / f"{split_name}.csv", index=False)


def summarize_split(df: pd.DataFrame, split_name: str) -> str:
    n_rows = len(df)
    class_balance = df["label"].value_counts(normalize=True).sort_index().to_dict()
    avg_words = float(df["text_length_words"].mean())

    return (
        f"{split_name}: n={n_rows}, "
        f"class_balance={class_balance}, "
        f"avg_words={avg_words:.2f}"
    )
