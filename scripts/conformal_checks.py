from __future__ import annotations

import argparse
import io
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.conformal import (
    build_eval_frame,
    conformal_k,
    conformal_threshold,
    indicator_frame,
    predict_sets,
    reference_band,
    reweight,
    summarize_indicators,
)
from src.utils import load_yaml_config

# Regression on the DistilBERT predictions of commit 569d312 (calibration 454 rows, test 715 rows), in two parts
# (author's decision, replacing "same numbers as the old code"):
#   (a) new M0 with the plan's rule (k-th smallest score, k = ceil((n + 1)(1 - alpha))) -> frozen new constants;
#   (b) the same set construction with the (k + 1)-th score reproduces the old code's output exactly.
# P29: the old code used np.quantile(scores, k / n, method="higher"), which returns the (k + 1)-th score
# whenever k < n (one rank too conservative); that rank is the only difference between old and new code.
REGRESSION_COMMIT = "569d312"
REGRESSION_NEW_M0 = {
    0.1: {
        "threshold": 0.93093027, "coverage": 0.8825174825174825, "set_size": 1.2503496503496503,
        "singleton": 0.7496503496503496, "empty": 0.0, "full": 0.25034965034965034,
    },
    0.05: {
        "threshold": 0.97843152, "coverage": 0.9482517482517483, "set_size": 1.5188811188811189,
        "singleton": 0.4811188811188811, "empty": 0.0, "full": 0.5188811188811189,
    },
}
# Old code output (results/tables/conformal_metrics.csv of 569d312), reproduced with the old code before it was deleted.
REGRESSION_OLD_CODE = {
    0.1: {
        "threshold": 0.945091568, "coverage": 0.8923076923076924, "set_size": 1.283916083916084,
        "singleton": 0.7160839160839161, "empty": 0.0, "full": 0.2839160839160839,
    },
    0.05: {
        "threshold": 0.978802262, "coverage": 0.9482517482517483, "set_size": 1.5216783216783216,
        "singleton": 0.4783216783216783, "empty": 0.0, "full": 0.5216783216783217,
    },
}
REGRESSION_TOLERANCE = 1e-12
FINDINGS = {
    "P29_old_quantile_off_by_one": (
        "Old conformal code used np.quantile(scores, k/n, method='higher'), i.e. the (k+1)-th smallest score "
        "instead of the k-th (k = ceil((n+1)(1-alpha))) whenever k < n: one rank too conservative. On 569d312 "
        "DistilBERT predictions at alpha=0.10: old coverage 0.8923, plan rule 0.8825."
    ),
    "plan_pilot_numbers": (
        "The plan's pilot numbers (e.g. 0.892; 0.829/0.951; 0.836-0.842) came from the old code and are slightly "
        "conservative; they are superseded by the new runs."
    ),
}

SYNTHETIC_REPEATS = 1000
SYNTHETIC_SIZES = (700, 715)
SYNTHETIC_BAND_HIT_RANGE = (0.93, 0.97)

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_TOLERANCE = 0.005
BOOTSTRAP_RUN = ("distilbert", 42)
REWEIGHT_METRICS = ["coverage", "workload", "uncertainty_referral", "set_size"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-5 checks of the conformal layer.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def score_frame(prob_stress: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "label": labels.astype(int),
        "score_0": prob_stress.astype(np.float64),
        "score_1": 1.0 - prob_stress.astype(np.float64),
    })


def check_synthetic(alpha: float, rng: np.random.Generator) -> list[tuple[str, Any]]:

    """
    Exchangeable synthetic data (p ~ Beta(2, 2), y ~ Bernoulli(p)), random calibration/test split.
    Share of repeats whose M0 coverage (and M1 c0, c1) falls inside the reference band.
    """

    n_calibration, m_test = SYNTHETIC_SIZES
    hits = {"M0_coverage": 0, "M1_c0": 0, "M1_c1": 0}

    for _ in range(SYNTHETIC_REPEATS):
        prob_stress = rng.beta(2, 2, size=n_calibration + m_test)
        labels = (rng.random(n_calibration + m_test) < prob_stress).astype(int)
        order = rng.permutation(n_calibration + m_test)
        calibration = score_frame(prob_stress[order[:n_calibration]], labels[order[:n_calibration]])
        test = score_frame(prob_stress[order[n_calibration:]], labels[order[n_calibration:]])
        test_labels = test["label"].to_numpy()

        sets, _ = predict_sets(calibration, test, "M0", alpha)
        coverage = sets[np.arange(m_test), test_labels].mean()
        low, high = reference_band(n_calibration, m_test, alpha)
        hits["M0_coverage"] += low <= coverage <= high

        sets, _ = predict_sets(calibration, test, "M1", alpha)
        for label in [0, 1]:
            in_class = test_labels == label
            class_coverage = sets[in_class, label].mean()
            low, high = reference_band(int((calibration["label"] == label).sum()), int(in_class.sum()), alpha)
            hits[f"M1_c{label}"] += low <= class_coverage <= high

    rows: list[tuple[str, Any]] = []
    lower, upper = SYNTHETIC_BAND_HIT_RANGE
    for name, count in hits.items():
        share = count / SYNTHETIC_REPEATS
        rows.append((f"{name}_share_in_band", round(share, 4)))
        rows.append((f"{name}_ok", lower <= share <= upper))
    return rows


def check_infinite_threshold(alpha: float) -> list[tuple[str, Any]]:

    """
    Boundary of the +inf rule (k > n) and its effect through a grouped method (M2-like grouping).
    """

    rows: list[tuple[str, Any]] = []

    # Smallest calibration size with a finite threshold: n >= 1/alpha - 1.
    n_finite = math.ceil(1 / Fraction(str(alpha)) - 1)
    for n, expect_infinite in [(0, True), (n_finite - 1, True), (n_finite, False)]:
        threshold = conformal_threshold(np.linspace(0.0, 0.5, n), alpha)
        rows.append((f"n={n}_infinite", math.isinf(threshold)))
        rows.append((f"n={n}_ok", math.isinf(threshold) == expect_infinite))

    rng = np.random.default_rng(0)
    small, large = n_finite - 1, 200
    calibration = pd.concat([
        score_frame(rng.random(small), np.repeat([0, 1], small // 2 + 1)[:small]).assign(domain="small"),
        score_frame(rng.random(2 * large), np.repeat([0, 1], large)).assign(domain="large"),
    ], ignore_index=True)
    test = pd.concat([
        score_frame(rng.random(20), np.repeat([0, 1], 10)).assign(domain="small"),
        score_frame(rng.random(20), np.repeat([0, 1], 10)).assign(domain="unseen"),
        score_frame(rng.random(20), np.repeat([0, 1], 10)).assign(domain="large"),
    ], ignore_index=True)

    sets, n_infinite = predict_sets(calibration, test, "M2", alpha)
    full_small_unseen = sets[:40].all()
    rows.append(("grouped_small_and_unseen_groups_full_sets", bool(full_small_unseen)))
    rows.append(("grouped_n_infinite_thresholds", n_infinite))
    rows.append(("grouped_ok", bool(full_small_unseen) and n_infinite == 4))
    return rows


def check_reweighting(
        calibration: pd.DataFrame,
        test: pd.DataFrame,
        alpha: float,
        prevalence_grid: list[float],
        rng: np.random.Generator,
) -> list[tuple[str, Any]]:

    """
    Analytic prevalence reweighting vs a stratified bootstrap of the test set at prevalence pi.
    """

    rows: list[tuple[str, Any]] = []
    labels = test["label"].to_numpy()
    index_by_class = {label: np.flatnonzero(labels == label) for label in [0, 1]}
    max_difference = 0.0

    for method in ["M0", "M1"]:
        sets, _ = predict_sets(calibration, test, method, alpha)
        indicators = indicator_frame(sets, labels).rename(columns={"covered": "coverage"})
        summary = summarize_indicators(indicator_frame(sets, labels), labels)

        for prevalence in prevalence_grid:
            analytic = reweight(summary, prevalence)
            n_risk = round(len(labels) * prevalence)
            sample = np.concatenate([
                rng.choice(index_by_class[1], size=(BOOTSTRAP_REPEATS, n_risk)),
                rng.choice(index_by_class[0], size=(BOOTSTRAP_REPEATS, len(labels) - n_risk)),
            ], axis=1)
            for metric in REWEIGHT_METRICS:
                bootstrap = indicators[metric].to_numpy()[sample].mean(axis=1).mean()
                difference = abs(bootstrap - analytic[metric])
                max_difference = max(max_difference, difference)
                rows.append((f"{method}.pi={prevalence}.{metric}.analytic", round(analytic[metric], 6)))
                rows.append((f"{method}.pi={prevalence}.{metric}.bootstrap", round(float(bootstrap), 6)))

    rows.append(("max_abs_difference", round(max_difference, 6)))
    rows.append(("ok", max_difference <= BOOTSTRAP_TOLERANCE))
    return rows


def check_regression() -> list[tuple[str, Any]]:

    """
    Two-part regression on the DistilBERT predictions of commit 569d312:
    (a) new M0 (k-th score) matches the frozen new constants and its threshold is the k-th score;
    (b) the (k + 1)-th score with the same set construction reproduces the old code's output.
    """

    old_csv = subprocess.run(
        ["git", "show", f"{REGRESSION_COMMIT}:results/predictions/transformer_predictions.csv"],
        capture_output=True, text=True, check=True,
    ).stdout
    old = pd.read_csv(io.StringIO(old_csv))

    def frame(split_name: str) -> pd.DataFrame:
        split = old[old["split"] == split_name]
        return pd.DataFrame({
            "label": split["label"].astype(int).to_numpy(),
            "score_0": 1.0 - split["prob_not_stress"].to_numpy(dtype=np.float64),
            "score_1": 1.0 - split["prob_stress"].to_numpy(dtype=np.float64),
        })

    calibration, test = frame("calibration"), frame("test")
    labels = test["label"].to_numpy()
    rows: list[tuple[str, Any]] = [("n_calibration", len(calibration)), ("n_test", len(test))]
    all_ok = len(calibration) == 454 and len(test) == 715

    scores = np.sort(np.where(calibration["label"] == 1, calibration["score_1"], calibration["score_0"]))
    test_scores = test[["score_0", "score_1"]].to_numpy()

    def observed(sets: np.ndarray, threshold: float) -> dict[str, float]:
        summary = summarize_indicators(indicator_frame(sets, labels), labels)
        return {"threshold": threshold, **{name: summary[name] for name in ["coverage", "set_size", "singleton", "empty", "full"]}}

    for alpha in REGRESSION_NEW_M0:
        k = conformal_k(len(scores), alpha)

        # (a) New M0 through the library path; its threshold must be the k-th score.
        sets, _ = predict_sets(calibration, test, "M0", alpha)
        threshold = conformal_threshold(scores, alpha)
        new_ok = threshold == scores[k - 1] and np.array_equal(sets, test_scores <= threshold)
        rows.append((f"a.alpha={alpha}.threshold_is_kth_score_ok", bool(new_ok)))
        all_ok &= new_ok

        # (b) Same set construction with the (k + 1)-th score, as the old code did.
        old_threshold = float(scores[k])
        parts = {
            "a": (REGRESSION_NEW_M0[alpha], observed(sets, threshold)),
            "b": (REGRESSION_OLD_CODE[alpha], observed(test_scores <= old_threshold, old_threshold)),
        }
        for part, (expected, values) in parts.items():
            for name, value in values.items():
                ok = abs(value - expected[name]) <= REGRESSION_TOLERANCE
                all_ok &= ok
                rows.append((f"{part}.alpha={alpha}.{name}", value))
                rows.append((f"{part}.alpha={alpha}.{name}_ok", ok))

    rows.append(("ok", all_ok))
    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conformal_config = config["conformal"]
    main_alpha = float(conformal_config["alpha_values"][0])
    rng = np.random.default_rng(int(config["seed"]))

    metadata = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["calibration", "test"]])
    model, seed = BOOTSTRAP_RUN
    predictions = pd.read_csv(runs_dir / model / f"seed_{seed}" / "predictions.csv.gz")
    clean = predictions[predictions["condition"] == "clean"]
    calibration = build_eval_frame(clean[clean["split"] == "calibration"], metadata)
    test = build_eval_frame(clean[clean["split"] == "test"], metadata)

    sections: dict[str, list[tuple[str, Any]]] = {}
    for alpha in conformal_config["alpha_values"]:
        sections[f"synthetic_alpha={alpha}"] = check_synthetic(float(alpha), rng)
        sections[f"infinite_threshold_alpha={alpha}"] = check_infinite_threshold(float(alpha))
    sections[f"reweighting_{model}_seed_{seed}_alpha={main_alpha}"] = check_reweighting(
        calibration, test, main_alpha, [float(p) for p in conformal_config["prevalence_grid"]], rng
    )
    sections[f"regression_{REGRESSION_COMMIT}"] = check_regression()
    sections["findings"] = list(FINDINGS.items())

    report = pd.DataFrame(
        [(section, key, value) for section, rows in sections.items() for key, value in rows],
        columns=["section", "key", "value"],
    )
    output_dir = Path("results/conformal")
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "step5_checks.csv", index=False)

    verdicts = report[report["key"].str.endswith("ok")]
    print(report[~report["key"].str.contains("analytic|bootstrap")].to_string(index=False))
    print(f"\nSaved report to: {(output_dir / 'step5_checks.csv').resolve()}")

    failed = verdicts[verdicts["value"].astype(str) != "True"]
    if not failed.empty:
        raise RuntimeError(f"Conformal checks failed:\n{failed.to_string(index=False)}")


if __name__ == "__main__":
    main()
