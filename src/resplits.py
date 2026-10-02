from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.conformal import build_eval_frame
from src.degradations import make_stream

PROTOCOLS = ["resplit", "official_test"]


def load_pool(data_dir: Path) -> pd.DataFrame:

    """
    The resplit pool: calibration rows, then test rows, with the columns the evaluation needs.
    """

    return pd.concat(
        [pd.read_csv(data_dir / f"{name}.csv").assign(split=name) for name in ["calibration", "test"]],
        ignore_index=True,
    )[["id", "post_id", "label", "domain", "confidence", "split"]]


def make_replicates(
        pool: pd.DataFrame,
        n_replicates: int,
        n_folds: int,
        base_seed: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:

    """
    (calibration positions, test positions) in the pool (calibration rows, then test rows) for both protocols.
    resplit r: StratifiedGroupKFold(n_folds, shuffle=True, random_state=r) by post, fold 0 -> calibration, fold 1 -> test.
    official_test b: calibration posts drawn with replacement (all their rows), test = the official test.
    """

    labels = pool["label"].to_numpy()
    posts = pool["post_id"].to_numpy()
    replicates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {protocol: [] for protocol in PROTOCOLS}

    for r in range(n_replicates):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=r)
        folds = [test_index for _, test_index in splitter.split(pool, labels, groups=posts)]
        replicates["resplit"].append((np.sort(folds[0]), np.sort(folds[1])))

    calibration_positions = np.flatnonzero(pool["split"].to_numpy() == "calibration")
    test_positions = np.flatnonzero(pool["split"].to_numpy() == "test")
    positions_by_post = pd.Series(calibration_positions).groupby(posts[calibration_positions]).apply(np.asarray)
    calibration_posts = positions_by_post.index.to_numpy()

    for b in range(n_replicates):
        rng = make_stream(base_seed, b, "bootstrap")
        drawn = rng.choice(calibration_posts, size=len(calibration_posts), replace=True)
        calibration = np.concatenate(positions_by_post.loc[drawn].to_numpy())
        replicates["official_test"].append((calibration, test_positions))

    return replicates


def pool_frames(
        predictions: pd.DataFrame,
        pool: pd.DataFrame,
        mix_assignment: pd.DataFrame,
        conditions: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

    """
    Evaluation frames of the pool (calibration rows, then test rows) for every condition,
    and the M3-mix frame (each example in its assigned condition).
    """

    metadata = pool[["id", "domain", "confidence"]]
    frames = {}
    for condition in conditions:
        rows = predictions[predictions["condition"] == condition]
        frame = pd.concat(
            [build_eval_frame(rows[rows["split"] == name], metadata) for name in ["calibration", "test"]],
            ignore_index=True,
        )
        assert np.array_equal(frame["id"].to_numpy(), pool["id"].to_numpy())
        frames[condition] = frame

    assert np.array_equal(mix_assignment["id"].to_numpy(), pool["id"].to_numpy())
    assigned = mix_assignment["condition"].to_numpy()
    mix = pd.concat(
        [frames[condition].assign(position=np.arange(len(pool)))[assigned == condition] for condition in conditions]
    ).sort_values("position").drop(columns="position").reset_index(drop=True)
    assert np.array_equal(mix["id"].to_numpy(), pool["id"].to_numpy())

    return frames, mix


def ensemble_predictions(paths: list[Path]) -> pd.DataFrame:

    """
    Ensemble of runs of one model: mean of the class probabilities per (split, condition, id).
    Members must share rows and order; logits are not defined for the mean and are dropped.
    """

    members = [pd.read_csv(path) for path in paths]
    keys = ["id", "split", "condition", "label", "n_words"]
    for member in members[1:]:
        assert member[keys].equals(members[0][keys])

    out = members[0][keys].copy()
    for column in ["prob_not_stress", "prob_stress"]:
        out[column] = np.mean([member[column].to_numpy(dtype=np.float64) for member in members], axis=0)
    return out
