from __future__ import annotations

from itertools import combinations, count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_NAMES = ["train", "validation", "calibration", "test"]
POOL_SPLITS = ["train", "validation", "calibration"]

DOMAIN_TO_SUBREDDITS = {
    "abuse": ["domesticviolence", "survivorsofabuse"],
    "social": ["relationships"],
    "anxiety": ["anxiety", "stress"],
    "ptsd": ["ptsd"],
    "financial": ["almosthomeless", "assistance", "food_pantry", "homeless"],
}
SUBREDDIT_TO_DOMAIN = {
    subreddit: domain
    for domain, subreddits in DOMAIN_TO_SUBREDDITS.items()
    for subreddit in subreddits
}

BASE_COLUMNS = [
    "id",
    "post_id",
    "subreddit",
    "domain",
    "text",
    "label",
    "confidence",
    "social_timestamp",
]


def load_dreaddit_dataset(dataset_name: str) -> dict[str, pd.DataFrame]:
    dataset = load_dataset(dataset_name)

    splits = {}
    for split_name in ["train", "validation", "test"]:
        splits[split_name] = dataset[split_name].to_pandas()

    return splits


def select_columns(df: pd.DataFrame) -> pd.DataFrame:

    """
    Keep ids, metadata, text, label and all LIWC features; add the domain.
    confidence = 0 marks a missing agreement score and becomes NaN.
    """

    unknown = sorted(set(df["subreddit"]) - set(SUBREDDIT_TO_DOMAIN))
    if unknown:
        raise ValueError(f"Subreddits without a domain: {unknown}")

    out = df.copy()
    out["domain"] = out["subreddit"].map(SUBREDDIT_TO_DOMAIN)
    out["text"] = out["text"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    out["confidence"] = out["confidence"].astype(float).replace(0.0, np.nan)

    if (out["text"] == "").any():
        raise ValueError("Found empty texts.")

    liwc_columns = [col for col in df.columns if col.startswith("lex_liwc_")]
    return out[BASE_COLUMNS + liwc_columns].reset_index(drop=True)


def normalize_text_key(texts: pd.Series) -> pd.Series:

    """
    Duplicate key: strip and collapse internal whitespace runs; case is kept.
    """

    return texts.str.strip().str.replace(r"\s+", " ", regex=True)


def deduplicate_pool(
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:

    """
    Within the pool: identical labels -> keep the first copy, conflicting labels -> drop all copies.
    Pool rows that duplicate a test row are dropped; the test split is not changed.
    """

    key = normalize_text_key(pool_df["text"])
    group_size = key.map(key.value_counts())
    n_labels = pool_df.groupby(key)["label"].transform("nunique")

    is_conflicting = n_labels > 1
    is_extra_copy = key.duplicated(keep="first") & ~is_conflicting
    deduped = pool_df[~is_conflicting & ~is_extra_copy]

    in_test = normalize_text_key(deduped["text"]).isin(normalize_text_key(test_df["text"]))
    deduped = deduped[~in_test].reset_index(drop=True)

    test_key = normalize_text_key(test_df["text"])
    stats = {
        "duplicate_groups": int(key[group_size > 1].nunique()),
        "conflicting_label_groups": int(key[is_conflicting].nunique()),
        "rows_removed_extra_copies": int(is_extra_copy.sum()),
        "rows_removed_conflicting_labels": int(is_conflicting.sum()),
        "rows_removed_test_duplicates": int(in_test.sum()),
        "test_duplicate_groups": int(test_key[test_key.duplicated()].nunique()),
    }
    return deduped, stats


def assign_folds(df: pd.DataFrame, n_splits: int, random_state: int) -> np.ndarray:

    """
    Fold k = the k-th test fold yielded by StratifiedGroupKFold(...).split(),
    grouped by post_id and stratified by label.
    """

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.full(len(df), fill_value=-1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(df, df["label"], groups=df["post_id"])):
        folds[test_idx] = fold

    assert (folds >= 0).all()
    return folds


def split_pool(
        pool_df: pd.DataFrame,
        split_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], int]:

    """
    Split the pool by post into train / validation / calibration.
    Retries random_state + 1, + 2, ... until calibration meets the minimum size per split and per class.
    Returns the splits and the random_state used.
    """

    for random_state in count(split_config["random_state"]):
        folds = assign_folds(pool_df, split_config["n_splits"], random_state)

        splits = {}
        for split_name in POOL_SPLITS:
            first, last = split_config[f"{split_name}_folds"]
            in_split = (folds >= first) & (folds <= last)
            splits[split_name] = pool_df[in_split].reset_index(drop=True)

        assert sum(len(df) for df in splits.values()) == len(pool_df)

        calibration = splits["calibration"]
        class_counts = calibration["label"].value_counts()
        if (
            len(calibration) >= split_config["min_calibration_rows"]
            and class_counts.reindex([0, 1], fill_value=0).min() >= split_config["min_calibration_rows_per_class"]
        ):
            return splits, random_state


def check_overlaps(splits: dict[str, pd.DataFrame]) -> dict[tuple[str, str], dict[str, int]]:

    """
    Count shared posts and shared normalized texts for every pair of splits.
    """

    overlaps = {}
    for name_a, name_b in combinations(SPLIT_NAMES, 2):
        df_a, df_b = splits[name_a], splits[name_b]
        overlaps[(name_a, name_b)] = {
            "shared_posts": len(set(df_a["post_id"]) & set(df_b["post_id"])),
            "shared_texts": len(
                set(normalize_text_key(df_a["text"])) & set(normalize_text_key(df_b["text"]))
            ),
        }
    return overlaps


def build_split_report(
        dataset_name: str,
        raw_sizes: dict[str, int],
        dedup_stats: dict[str, int],
        split_config: dict[str, Any],
        random_state_used: int,
        splits: dict[str, pd.DataFrame],
        overlaps: dict[tuple[str, str], dict[str, int]],
) -> pd.DataFrame:

    """
    Long-format report with columns: section, split, key, value.
    """

    rows: list[tuple[str, str, str, Any]] = []

    rows.append(("source", "all", "dataset_name", dataset_name))
    for split_name, n_rows in raw_sizes.items():
        rows.append(("source", split_name, "n_rows", n_rows))

    for key, value in dedup_stats.items():
        split_name = "test" if key.startswith("test_") else "pool"
        rows.append(("duplicates", split_name, key.removeprefix("test_"), value))

    rows.append(("split_params", "all", "n_splits", split_config["n_splits"]))
    rows.append(("split_params", "all", "random_state_initial", split_config["random_state"]))
    rows.append(("split_params", "all", "random_state_used", random_state_used))
    for split_name in POOL_SPLITS:
        first, last = split_config[f"{split_name}_folds"]
        rows.append(("split_params", split_name, "folds", f"{first}-{last}"))

    for split_name in SPLIT_NAMES:
        df = splits[split_name]
        rows.append(("size", split_name, "n_rows", len(df)))
        rows.append(("size", split_name, "n_posts", df["post_id"].nunique()))
        for label in [0, 1]:
            rows.append(("size", split_name, f"n_label_{label}", int((df["label"] == label).sum())))
        rows.append(("size", split_name, "share_label_1", round(float(df["label"].mean()), 4)))
        rows.append(("size", split_name, "n_confidence_missing", int(df["confidence"].isna().sum())))

        for domain in DOMAIN_TO_SUBREDDITS:
            domain_df = df[df["domain"] == domain]
            rows.append(("domain", split_name, f"{domain}.n_rows", len(domain_df)))
            rows.append(("domain", split_name, f"{domain}.n_label_1", int(domain_df["label"].sum())))

    for (name_a, name_b), counts in overlaps.items():
        for key, value in counts.items():
            rows.append(("overlap", f"{name_a}|{name_b}", key, value))

    return pd.DataFrame(rows, columns=["section", "split", "key", "value"])


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        df.to_csv(output_path / f"{split_name}.csv", index=False)


def summarize_split(df: pd.DataFrame, split_name: str) -> str:
    n_rows = len(df)
    class_balance = df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()

    return (
        f"{split_name}: n={n_rows}, "
        f"posts={df['post_id'].nunique()}, "
        f"class_balance={class_balance}"
    )
