from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from functools import lru_cache
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


@lru_cache(maxsize=None)
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


GROUP_TYPES = {
    "all": [],
    "class": ["label"],
    "domain": ["domain"],
    "domain_class": ["domain", "label"],
    "agreement": ["agreement"],
}


def group_codes(
        calibration: pd.DataFrame,
        test: pd.DataFrame,
        columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Joint integer codes of the group columns for calibration and test rows, and the sorted group names
    ("|"-joined values; "all" without columns).
    """

    if not columns:
        return np.zeros(len(calibration), dtype=int), np.zeros(len(test), dtype=int), np.array(["all"])

    def keys(frame: pd.DataFrame) -> np.ndarray:
        joined = frame[columns[0]].astype(str)
        for column in columns[1:]:
            joined = joined + "|" + frame[column].astype(str)
        return joined.to_numpy()

    codes, names = pd.factorize(np.concatenate([keys(calibration), keys(test)]), sort=True)
    return codes[:len(calibration)], codes[len(calibration):], np.asarray(names)


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

    calibration_scores = true_class_scores(calibration)
    test_scores = test[["score_0", "score_1"]].to_numpy(dtype=np.float64)
    if jitter is not None:
        calibration_scores = calibration_scores + jitter.uniform(0, jitter_scale, size=calibration_scores.shape)
        test_scores = test_scores + jitter.uniform(0, jitter_scale, size=test_scores.shape)

    calibration_groups, test_groups, names = group_codes(calibration, test, spec["group_columns"])
    n_classes = 2 if spec["by_class"] else 1
    calibration_keys = calibration_groups * n_classes
    if spec["by_class"]:
        calibration_keys = calibration_keys + calibration["label"].to_numpy()

    thresholds = np.array([
        conformal_threshold(calibration_scores[calibration_keys == key], alpha)
        for key in range(len(names) * n_classes)
    ])

    sets = np.zeros((len(test), 2), dtype=bool)
    infinite_keys: set[int] = set()
    candidate_classes = [RISK_CLASS] if spec["risk_only"] else [0, 1]

    for candidate in candidate_classes:
        test_keys = test_groups * n_classes + (candidate if spec["by_class"] else 0)
        used = np.unique(test_keys)
        infinite_keys |= set(used[np.isinf(thresholds[used])].tolist())
        sets[:, candidate] = test_scores[:, candidate] <= thresholds[test_keys]

    if spec["risk_only"]:
        # M5 screening: flag -> {risk}; otherwise -> {not risk}.
        sets[:, 1 - RISK_CLASS] = ~sets[:, RISK_CLASS]

    return sets, len(infinite_keys)


def indicator_matrix(sets: np.ndarray, labels: np.ndarray) -> np.ndarray:

    """
    Per-example indicators (columns in INDICATORS order). Triage: {not risk} -> dismiss, {risk} -> flag,
    full or empty -> refer to a human. workload = not dismissed (flag + refer); uncertainty_referral = full or empty set.
    """

    size = sets.sum(axis=1)
    flag = sets[:, RISK_CLASS] & ~sets[:, 1 - RISK_CLASS]
    dismiss = sets[:, 1 - RISK_CLASS] & ~sets[:, RISK_CLASS]
    return np.column_stack([
        sets[np.arange(len(labels)), labels],
        size,
        size == 1,
        size == 0,
        size == 2,
        flag,
        dismiss,
        ~dismiss,
        (size == 0) | (size == 2),
    ]).astype(np.float64)


def indicator_frame(sets: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(indicator_matrix(sets, labels), columns=INDICATORS)


def summarize_codes(
        matrix: np.ndarray,
        labels: np.ndarray,
        codes: np.ndarray,
        n_groups: int,
) -> list[dict[str, float]]:
    
    """
    Per group: overall means of the indicators plus class-conditional means (suffix _0 / _1).
    c0, c1 = class coverage; sensitivity = share of risk cases not dismissed. Empty classes give NaN.
    All indicators are 0/1 or set sizes, so the sums are exact and the means do not depend on summation order.
    """

    counts = np.bincount(codes, minlength=n_groups)
    sums = np.column_stack([np.bincount(codes, weights=matrix[:, j], minlength=n_groups) for j in range(matrix.shape[1])])
    class_codes = codes * 2 + labels
    class_counts = np.bincount(class_codes, minlength=2 * n_groups)
    class_sums = np.column_stack([
        np.bincount(class_codes, weights=matrix[:, j], minlength=2 * n_groups) for j in range(matrix.shape[1])
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]
        class_means = class_sums / class_counts[:, None]

    out = []
    for group in range(n_groups):
        summary: dict[str, float] = {"n": int(counts[group])}
        summary.update({name: float(means[group, j]) for j, name in enumerate(INDICATORS)})
        for label in [0, 1]:
            row = 2 * group + label
            summary[f"n_{label}"] = int(class_counts[row])
            summary.update({f"{name}_{label}": float(class_means[row, j]) for j, name in enumerate(INDICATORS)})

        summary["coverage"] = summary.pop("covered")
        summary["c0"] = summary.pop("covered_0")
        summary["c1"] = summary.pop("covered_1")
        summary["sensitivity"] = summary[f"workload_{RISK_CLASS}"]
        out.append(summary)

    return out


def summarize_indicators(indicators: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:

    """
    Overall and class-conditional means of all examples (see summarize_codes).
    """

    assert list(indicators.columns) == INDICATORS
    return summarize_codes(indicators.to_numpy(), labels, np.zeros(len(labels), dtype=int), 1)[0]


def reweight(summary: dict[str, float], prevalence: float) -> dict[str, float]:

    """
    Metric at risk prevalence pi: pi * metric(risk class) + (1 - pi) * metric(non-risk class).
    Works on a dict of scalars or on a DataFrame of runs (column-wise).
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
        group_types: list[str] | None = None,
) -> pd.DataFrame:
    
    """
    Metrics per group (all, class, domain, domain x class, agreement bucket) for the groups present in test,
    each with n_calibration, n_test and the reference band for coverage at those sizes.
    """

    assert list(indicators.columns) == INDICATORS
    matrix = indicators.to_numpy()
    labels = test["label"].to_numpy()

    rows = []
    for group_type in group_types or list(GROUP_TYPES):
        calibration_codes, test_codes, names = group_codes(calibration, test, GROUP_TYPES[group_type])
        calibration_sizes = np.bincount(calibration_codes, minlength=len(names))
        summaries = summarize_codes(matrix, labels, test_codes, len(names))

        for group, name in enumerate(names):
            if summaries[group]["n"] == 0:
                continue
            band_low, band_high = reference_band(int(calibration_sizes[group]), summaries[group]["n"], alpha)
            rows.append({
                "group_type": group_type,
                "group": str(name),
                "n_calibration": int(calibration_sizes[group]),
                "band_low": band_low,
                "band_high": band_high,
                **summaries[group],
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
