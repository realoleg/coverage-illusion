from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import word_count

# Families with a random stream; levels of one family share the stream, so they are nested.
RANDOM_FAMILIES = ["del", "typo"]
MIX_FAMILY = "mix"

QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

# Upper bounds (in words) of the length buckets after degradation: <=25, 26-50, 51-80, >80.
LENGTH_BUCKET_BINS = [0, 25, 50, 80, np.inf]
LENGTH_BUCKET_LABELS = ["<=25", "26-50", "51-80", ">80"]


def build_qwerty_neighbors() -> dict[str, list[str]]:

    """
    Letter neighbors on a staggered QWERTY layout: same row left/right,
    row above at the same and next position, row below at the previous and same position.
    """

    neighbors = {}
    for row_idx, row in enumerate(QWERTY_ROWS):
        for col_idx, key in enumerate(row):
            candidates = [
                (row_idx, col_idx - 1), (row_idx, col_idx + 1),
                (row_idx - 1, col_idx), (row_idx - 1, col_idx + 1),
                (row_idx + 1, col_idx - 1), (row_idx + 1, col_idx),
            ]
            neighbors[key] = [
                QWERTY_ROWS[r][c]
                for r, c in candidates
                if 0 <= r < len(QWERTY_ROWS) and 0 <= c < len(QWERTY_ROWS[r])
            ]
    return neighbors


QWERTY_NEIGHBORS = build_qwerty_neighbors()


def stream_seed(base_seed: int, example_id: int, family: str) -> int:

    """
    Seed of one random stream = hash of (base seed, example id, family); no seed arithmetic.
    """

    digest = hashlib.sha256(f"{base_seed}|{example_id}|{family}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_stream(base_seed: int, example_id: int, family: str) -> np.random.Generator:
    return np.random.default_rng(stream_seed(base_seed, example_id, family))


def parse_condition(condition: str) -> tuple[str, float]:

    """
    'head_25' -> ('head', 0.25); 'typo_05' -> ('typo', 0.05).
    """

    family, level = condition.split("_")
    return family, int(level) / 100


def keep_head(text: str, fraction: float) -> str:

    """
    Keep the first ceil(n * fraction) words, at least one.
    """

    words = text.split()
    keep_n = max(1, math.ceil(len(words) * fraction))
    return " ".join(words[:keep_n])


def keep_tail(text: str, fraction: float) -> str:

    """
    Keep the last ceil(n * fraction) words, at least one.
    """

    words = text.split()
    keep_n = max(1, math.ceil(len(words) * fraction))
    return " ".join(words[-keep_n:])


def delete_words(text: str, prob: float, rng: np.random.Generator) -> str:

    """
    One U(0,1) per word; the word is deleted if U < prob.
    If every word is deleted, the word with the largest U is kept (this keeps lower levels nested).
    """

    words = text.split()
    draws = rng.random(len(words))
    keep = draws >= prob
    if not keep.any():
        keep[np.argmax(draws)] = True

    return " ".join(word for word, kept in zip(words, keep) if kept)


def add_typos(text: str, prob: float, rng: np.random.Generator) -> str:

    """
    Per character, one U(0,1) and one neighbor choice are drawn up front;
    an ASCII letter is replaced by its QWERTY neighbor (case kept) if U < prob.
    """

    draws = rng.random(len(text))
    choices = rng.random(len(text))

    chars = list(text)
    for idx, char in enumerate(chars):
        neighbors = QWERTY_NEIGHBORS.get(char.lower())
        if neighbors is None or draws[idx] >= prob:
            continue
        replacement = neighbors[int(choices[idx] * len(neighbors))]
        chars[idx] = replacement.upper() if char.isupper() else replacement

    return "".join(chars)


def apply_condition(df: pd.DataFrame, condition: str, base_seed: int) -> pd.Series:

    """
    Texts of df under the given condition, aligned with df rows.
    """

    texts = df["text"].astype(str)
    if condition == "clean":
        return texts

    family, level = parse_condition(condition)
    if family == "head":
        out = [keep_head(text, level) for text in texts]
    elif family == "tail":
        out = [keep_tail(text, level) for text in texts]
    elif family == "del":
        out = [
            delete_words(text, level, make_stream(base_seed, example_id, family))
            for example_id, text in zip(df["id"], texts)
        ]
    elif family == "typo":
        out = [
            add_typos(text, level, make_stream(base_seed, example_id, family))
            for example_id, text in zip(df["id"], texts)
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return pd.Series(out, index=df.index, dtype=str)


def length_bucket(n_words: pd.Series) -> pd.Series:
    return pd.cut(n_words, bins=LENGTH_BUCKET_BINS, labels=LENGTH_BUCKET_LABELS).astype(str)


def build_condition_texts(
        splits: dict[str, pd.DataFrame],
        conditions: list[str],
        base_seed: int,
) -> pd.DataFrame:

    """
    Long table of texts: one row per (split, condition, example).
    """

    frames = []
    for split_name, df in splits.items():
        for condition in conditions:
            texts = apply_condition(df, condition, base_seed)
            n_words = word_count(texts)
            frames.append(pd.DataFrame({
                "id": df["id"].to_numpy(),
                "split": split_name,
                "condition": condition,
                "text": texts.to_numpy(),
                "n_words": n_words.to_numpy(),
                "length_bucket": length_bucket(n_words).to_numpy(),
            }))

    return pd.concat(frames, ignore_index=True)


def assign_mix(ids: pd.Series, conditions: list[str], base_seed: int) -> pd.Series:

    """
    M3-mix: one condition per example, drawn uniformly from the given conditions (in the given order)
    with the stream seeded by hash(base seed, id, 'mix').
    """

    return pd.Series(
        [conditions[make_stream(base_seed, example_id, MIX_FAMILY).integers(len(conditions))] for example_id in ids],
        index=ids.index,
        dtype=str,
    )


def load_condition_texts(path: str | Path) -> pd.DataFrame:

    """
    Load the saved condition texts; texts are read verbatim (no NA parsing).
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Condition texts not found: {path}. Run scripts.make_conditions first.")

    return pd.read_csv(path, keep_default_na=False, dtype={"text": str, "split": str, "condition": str})


def select_condition_texts(
        condition_texts: pd.DataFrame,
        split_name: str,
        condition: str,
        ids: pd.Series,
) -> pd.Series:

    """
    Texts of one (split, condition), in the order of ids.
    """

    selected = condition_texts[
        (condition_texts["split"] == split_name) & (condition_texts["condition"] == condition)
    ].set_index("id")["text"]

    return selected.loc[ids.to_numpy()].reset_index(drop=True)


def iter_prediction_inputs(
        split_to_df: dict[str, pd.DataFrame],
        condition_texts: pd.DataFrame,
        conditions: list[str],
) -> Iterator[tuple[str, str, pd.DataFrame, pd.Series]]:

    """
    Yield (split, condition, split df, texts) for every prediction a run makes:
    validation clean only; calibration and test in every condition (texts from the saved file).
    """

    yield "validation", "clean", split_to_df["validation"], split_to_df["validation"]["text"].astype(str)

    for split_name in ["calibration", "test"]:
        df = split_to_df[split_name]
        for condition in conditions:
            yield split_name, condition, df, select_condition_texts(condition_texts, split_name, condition, df["id"])
