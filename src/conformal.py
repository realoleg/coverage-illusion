from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from src.degradations import length_bucket

RISK_CLASS = 1

# Annotator-agreement buckets of Dreaddit `confidence` (missing = confidence 0 in the raw data).
AGREEMENT_BINS = [0.0, 0.60, 0.87, 1.0]
AGREEMENT_LABELS = ["0.43-0.60", "0.67-0.86", "1.00"]
AGREEMENT_MISSING = "missing"

# Method -> calibration source and threshold groups.
#   calibration: "clean" (clean calibration), "matched" (calibration in the test condition), "mix" (M3-mix assignment)
#   group_columns: extra grouping columns; by_class: one threshold per candidate class; risk_only: M5 screening flag.
METHODS: dict[str, dict[str, Any]] = {
    "M0": {"calibration": "clean", "group_columns": [], "by_class": False, "risk_only": False},
    "M1": {"calibration": "clean", "group_columns": [], "by_class": True, "risk_only": False},
    "M2": {"calibration": "clean", "group_columns": ["domain"], "by_class": True, "risk_only": False},
    "M3": {"calibration": "matched", "group_columns": [], "by_class": True, "risk_only": False},
    "M3-mix": {"calibration": "mix", "group_columns": ["length_bucket"], "by_class": True, "risk_only": False},
    "M5": {"calibration": "clean", "group_columns": [], "by_class": True, "risk_only": True},
}

# Per-example indicators; every metric below is a mean of one of them (so it can be reweighted by class).
INDICATORS = [
    "covered", "set_size", "singleton", "empty", "full",
    "flag", "dismiss", "workload", "uncertainty_referral",
]


def conformal_k(n: int, alpha: float) -> int:

    """
    Rank of the conformal threshold: k = ceil((n + 1)(1 - alpha)), computed exactly.
    """

    return math.ceil((n + 1) * (1 - Fraction(str(alpha))))


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:

    """
    k-th smallest calibration score; +inf if k > n (the group is too small, every class is included).
    """

    n = len(scores)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def reference_band(n_calibration: int, m_test: int, alpha: float, level: float = 0.95) -> tuple[float, float]:

    """
    Central `level` interval of test coverage when the guarantee holds:
    BetaBinomial(m, n + 1 - l, l) / m with l = floor((n + 1) alpha). If l = 0 the threshold is +inf and coverage is 1.
    """

    if n_calibration == 0 or m_test == 0:
        return math.nan, math.nan

    ell = math.floor((n_calibration + 1) * Fraction(str(alpha)))
    if ell == 0:
        return 1.0, 1.0

    distribution = betabinom(m_test, n_calibration + 1 - ell, ell)
    tail = (1 - level) / 2
    return float(distribution.ppf(tail) / m_test), float(distribution.ppf(1 - tail) / m_test)


def agreement_bucket(confidence: pd.Series) -> pd.Series:
    buckets = pd.cut(confidence, bins=AGREEMENT_BINS, labels=AGREEMENT_LABELS).astype(object)
    assert buckets[confidence.notna()].notna().all(), "confidence value outside the agreement buckets"
    return buckets.where(confidence.notna(), AGREEMENT_MISSING).astype(str)


def build_eval_frame(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:

    """
    Predictions of one (split, condition) with scores and grouping columns:
    score_0 / score_1 = 1 - p_y, domain, agreement bucket, length bucket of the (degraded) text.
    """

    out = predictions.merge(metadata[["id", "domain", "confidence"]], on="id", how="left", validate="one_to_one")
    assert len(out) == len(predictions) and out["domain"].notna().all()

    out["score_0"] = 1.0 - out["prob_not_stress"].astype(np.float64)
    out["score_1"] = 1.0 - out["prob_stress"].astype(np.float64)
    out["agreement"] = agreement_bucket(out["confidence"])
    out["length_bucket"] = length_bucket(out["n_words"])
    return out.reset_index(drop=True)


def true_class_scores(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame["label"].to_numpy() == 1, frame["score_1"].to_numpy(), frame["score_0"].to_numpy())


def jitter_seed(base_seed: int, resplit_id: int | str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{resplit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def group_keys(frame: pd.DataFrame, group_columns: list[str], candidate_class: int | None) -> list[tuple]:
    columns = [frame[column].tolist() for column in group_columns]
    if candidate_class is None:
        return [tuple(values) for values in zip(*columns)] if columns else [()] * len(frame)
    return [(*values, candidate_class) for values in zip(*columns)] if columns else [(candidate_class,)] * len(frame)


def predict_sets(
        calibration: pd.DataFrame,
        test: pd.DataFrame,
        method: str,
        alpha: float,
        jitter: np.random.Generator | None = None,
        jitter_scale: float = 1e-9,
) -> tuple[np.ndarray, int]:

    """
    Prediction sets (n_test x 2 boolean) of one method; class y is included if s(x, y) <= threshold of its group.
    Groups without calibration examples get +inf. Returns the sets and the number of +inf thresholds used.
    Optional jitter (appendix tie-breaking): U(0, jitter_scale) added to every calibration and test score.
    """

    spec = METHODS[method]
    group_columns = spec["group_columns"]

    calibration_scores = true_class_scores(calibration)
    test_scores = test[["score_0", "score_1"]].to_numpy(dtype=np.float64)
    if jitter is not None:
        calibration_scores = calibration_scores + jitter.uniform(0, jitter_scale, size=calibration_scores.shape)
        test_scores = test_scores + jitter.uniform(0, jitter_scale, size=test_scores.shape)

    calibration_labels = calibration["label"].to_numpy()
    if spec["by_class"]:
        calibration_keys = group_keys(calibration, group_columns, None)
        calibration_keys = [(*key, label) for key, label in zip(calibration_keys, calibration_labels)]
    else:
        calibration_keys = group_keys(calibration, group_columns, None)

    scores_by_key: dict[tuple, list[float]] = {}
    for key, score in zip(calibration_keys, calibration_scores):
        scores_by_key.setdefault(key, []).append(score)

    sets = np.zeros((len(test), 2), dtype=bool)
    infinite_keys: set[tuple] = set()
    candidate_classes = [RISK_CLASS] if spec["risk_only"] else [0, 1]

    for candidate in candidate_classes:
        keys = group_keys(test, group_columns, candidate if spec["by_class"] else None)
        thresholds = {
            key: conformal_threshold(np.asarray(scores_by_key.get(key, [])), alpha) for key in set(keys)
        }
        infinite_keys |= {key for key, threshold in thresholds.items() if math.isinf(threshold)}
        sets[:, candidate] = test_scores[:, candidate] <= np.array([thresholds[key] for key in keys])

    if spec["risk_only"]:
        # M5 screening: flag -> {risk}; otherwise -> {not risk}.
        sets[:, 1 - RISK_CLASS] = ~sets[:, RISK_CLASS]

    return sets, len(infinite_keys)


def indicator_frame(sets: np.ndarray, labels: np.ndarray) -> pd.DataFrame:

    """
    Per-example indicators. Triage: {not risk} -> dismiss, {risk} -> flag, full or empty -> refer to a human.
    workload = not dismissed (flag + refer); uncertainty_referral = full or empty set.
    """

    size = sets.sum(axis=1)
    flag = sets[:, RISK_CLASS] & ~sets[:, 1 - RISK_CLASS]
    dismiss = sets[:, 1 - RISK_CLASS] & ~sets[:, RISK_CLASS]
    return pd.DataFrame({
        "covered": sets[np.arange(len(labels)), labels],
        "set_size": size,
        "singleton": size == 1,
        "empty": size == 0,
        "full": size == 2,
        "flag": flag,
        "dismiss": dismiss,
        "workload": ~dismiss,
        "uncertainty_referral": (size == 0) | (size == 2),
    }).astype(float)


def summarize_indicators(indicators: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:

    """
    Overall means plus class-conditional means (suffix _0 / _1). c0, c1 = class coverage;
    sensitivity = share of risk cases not dismissed.
    """

    out: dict[str, float] = {"n": len(labels)}
    out.update(indicators.mean().to_dict())
    for label in [0, 1]:
        in_class = labels == label
        out[f"n_{label}"] = int(in_class.sum())
        for name, value in indicators[in_class].mean().items():
            out[f"{name}_{label}"] = float(value)

    out["coverage"] = out.pop("covered")
    out["c0"] = out.pop("covered_0")
    out["c1"] = out.pop("covered_1")
    out["sensitivity"] = out[f"workload_{RISK_CLASS}"]
    return out


def reweight(summary: dict[str, float], prevalence: float) -> dict[str, float]:

    """
    Metric at risk prevalence pi: pi * metric(risk class) + (1 - pi) * metric(non-risk class).
    """

    class_metrics = {"coverage": ("c0", "c1")} | {name: (f"{name}_0", f"{name}_1") for name in INDICATORS if name != "covered"}
    return {
        name: prevalence * summary[risk] + (1 - prevalence) * summary[non_risk]
        for name, (non_risk, risk) in class_metrics.items()
    }


def group_summaries(
        indicators: pd.DataFrame,
        test: pd.DataFrame,
        calibration: pd.DataFrame,
        alpha: float,
) -> pd.DataFrame:

    """
    Metrics per group (all, class, domain, domain x class, agreement bucket), each with n_calibration,
    n_test and the reference band for coverage at those sizes.
    """

    labels = test["label"].to_numpy()
    groupings = {
        "all": [],
        "class": ["label"],
        "domain": ["domain"],
        "domain_class": ["domain", "label"],
        "agreement": ["agreement"],
    }

    rows = []
    for group_type, columns in groupings.items():
        if columns:
            # A single column is passed as a scalar so group keys are scalars, not 1-tuples.
            by = columns[0] if len(columns) == 1 else columns
            test_groups = test.groupby(by, sort=True).indices
            calibration_sizes = calibration.groupby(by).size()
        else:
            test_groups = {"all": np.arange(len(test))}
            calibration_sizes = pd.Series({"all": len(calibration)})

        for group, index in test_groups.items():
            n_calibration = int(calibration_sizes.get(group, 0))
            band_low, band_high = reference_band(n_calibration, len(index), alpha)
            group_name = "|".join(str(part) for part in group) if isinstance(group, tuple) else str(group)
            rows.append({
                "group_type": group_type,
                "group": group_name,
                "n_calibration": n_calibration,
                "band_low": band_low,
                "band_high": band_high,
                **summarize_indicators(indicators.iloc[index], labels[index]),
            })

    return pd.DataFrame(rows)


def worst_groups(groups: pd.DataFrame) -> dict[str, Any]:

    """
    Worst domain and worst (domain, class) by coverage, with group sizes and reference bands.
    """

    out: dict[str, Any] = {}
    for group_type in ["domain", "domain_class"]:
        worst = groups[groups["group_type"] == group_type].sort_values(["coverage", "group"]).iloc[0]
        prefix = f"worst_{group_type}"
        out[prefix] = worst["group"]
        out[f"{prefix}_coverage"] = float(worst["coverage"])
        out[f"{prefix}_n_test"] = int(worst["n"])
        out[f"{prefix}_n_calibration"] = int(worst["n_calibration"])
        out[f"{prefix}_band_low"] = float(worst["band_low"])
        out[f"{prefix}_band_high"] = float(worst["band_high"])
    return out


def calibration_frame(
        method: str,
        calibration_by_condition: dict[str, pd.DataFrame],
        mix_calibration: pd.DataFrame,
        test_condition: str,
) -> pd.DataFrame:

    """
    Calibration rows a method uses for a test condition.
    """

    source = METHODS[method]["calibration"]
    if source == "clean":
        return calibration_by_condition["clean"]
    if source == "matched":
        return calibration_by_condition[test_condition]
    return mix_calibration


def build_mix_calibration(
        calibration_by_condition: dict[str, pd.DataFrame],
        mix_assignment: pd.DataFrame,
) -> pd.DataFrame:

    """
    M3-mix calibration: for each calibration example, its row in the assigned condition.
    """

    stacked = pd.concat(calibration_by_condition.values(), ignore_index=True)
    out = mix_assignment.merge(stacked, on=["id", "condition"], how="left", validate="one_to_one")
    assert out["score_1"].notna().all(), "mix assignment refers to a missing condition"
    return out
