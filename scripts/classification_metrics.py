from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score

from src.conformal import (
    METHODS,
    b1_sets,
    build_eval_frame,
    calibration_frame,
    calibration_metrics,
    fit_temperature,
    logit_margin,
    predict_sets,
)
from src.evaluate import predicted_labels
from src.resplits import ensemble_predictions, load_pool, make_replicates, pool_frames
from src.utils import load_yaml_config

INVARIANCE_RESPLITS = 10
STEP7_METHODS = {"B1", "B2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 7: classification / calibration table and T-invariance check.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def classification_rows(predictions: pd.DataFrame, temperature: float | None) -> list[dict[str, Any]]:

    """
    Per (split, condition): macro-F1 and AUROC; for probabilistic runs also Brier, NLL and both ECEs,
    raw and (if a temperature is given) after temperature scaling of the stored logits.
    """

    rows = []
    for (split_name, condition), group in predictions.groupby(["split", "condition"], sort=True):
        labels = group["label"].to_numpy()
        base = {
            "split": split_name, "condition": condition, "n": len(group),
            "macro_f1": float(f1_score(labels, predicted_labels(group), average="macro")),
        }
        if "prob_stress" not in group.columns:
            rows.append({**base, "probabilities": "none",
                         "auroc": float(roc_auc_score(labels, group["decision_score"]))})
            continue

        versions = [("raw", np.nan, group["prob_stress"].to_numpy(dtype=np.float64))]
        if temperature is not None:
            versions.append(("temperature", temperature, expit(logit_margin(group) / temperature)))
        for name, value, prob_risk in versions:
            rows.append({
                **base, "probabilities": name, "temperature": value,
                "auroc": float(roc_auc_score(labels, prob_risk)),
                **calibration_metrics(prob_risk, labels),
            })
    return rows


def with_probabilities(predictions: pd.DataFrame, temperature: float) -> pd.DataFrame:

    """
    Probabilities recomputed in float64 from the stored logits: sigmoid(margin / temperature).
    """

    out = predictions.copy()
    out["prob_stress"] = expit(logit_margin(predictions) / temperature)
    out["prob_not_stress"] = 1.0 - out["prob_stress"]
    return out


def invariance_check(
        predictions: pd.DataFrame,
        temperature: float,
        pool: pd.DataFrame,
        validation_metadata: pd.DataFrame,
        mix_assignment: pd.DataFrame,
        conditions: list[str],
        alphas: list[float],
        replicates: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:

    """
    Sets of M0-M5 and B1 with float64 probabilities from logits, before and after temperature scaling.
    Counts changed set rows and saturated probabilities (exactly 0 or 1).
    """

    versions = {}
    for name, value in [("raw", 1.0), ("scaled", temperature)]:
        scaled = with_probabilities(predictions, value)
        frames, mix = pool_frames(scaled, pool, mix_assignment, conditions)
        validation = build_eval_frame(scaled[scaled["split"] == "validation"], validation_metadata)
        saturated = scaled["prob_stress"].isin([0.0, 1.0]).sum()
        versions[name] = (frames, mix, validation, int(saturated))

    n_compared = n_changed = 0
    for calibration_positions, test_positions in replicates:
        sets = {}
        for name, (frames, mix, validation, _) in versions.items():
            calibration_by_condition = {c: frame.iloc[calibration_positions] for c, frame in frames.items()}
            mix_calibration = mix.iloc[calibration_positions]
            for condition in conditions:
                test = frames[condition].iloc[test_positions]
                for alpha in alphas:
                    for method in METHODS:
                        calibration = calibration_frame(method, calibration_by_condition, mix_calibration, condition)
                        sets[(name, condition, alpha, method)] = predict_sets(calibration, test, method, alpha)[0]
                    sets[(name, condition, alpha, "B1")] = b1_sets(validation, test, alpha)

        for (name, *key), raw_sets in sets.items():
            if name != "raw":
                continue
            changed = (raw_sets != sets[("scaled", *key)]).any(axis=1)
            n_compared += len(changed)
            n_changed += int(changed.sum())

    return {
        "set_rows_compared": n_compared,
        "set_rows_changed": n_changed,
        "saturated_probabilities_raw": versions["raw"][3],
        "saturated_probabilities_scaled": versions["scaled"][3],
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conditions = config["conditions"]
    alphas = [float(alpha) for alpha in config["conformal"]["alpha_values"]]
    output_dir = Path(config["evaluation"]["output_dir"])

    pool = load_pool(data_dir)
    validation_metadata = pd.read_csv(data_dir / "validation.csv")[["id", "domain", "confidence"]]
    mix_assignment = pd.read_csv(data_dir / f"{dataset}_mix_assignment.csv")

    # Fixed split (original calibration / official test) plus the first resplits of the main protocol.
    split = pool["split"].to_numpy()
    replicates = [(np.flatnonzero(split == "calibration"), np.flatnonzero(split == "test"))]
    replicates += make_replicates(pool, INVARIANCE_RESPLITS, int(config["evaluation"]["resplit_folds"]), base_seed)["resplit"]

    table_rows: list[dict[str, Any]] = []
    check_rows: list[tuple[str, str, str, Any]] = []
    runs_by_model: dict[str, list[Path]] = {}
    for path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        runs_by_model.setdefault(path.parent.parent.name, []).append(path)

    for model, paths in runs_by_model.items():
        for path in paths:
            seed = path.parent.name.removeprefix("seed_")
            predictions = pd.read_csv(path)
            temperature = None
            if "logit_stress" in predictions.columns:
                validation = predictions[predictions["split"] == "validation"]
                temperature = fit_temperature(logit_margin(validation), validation["label"].to_numpy())
                name = f"{model}/seed={seed}"
                check_rows.append(("temperature", name, "validation_nll_temperature", round(temperature, 4)))
                result = invariance_check(
                    predictions, temperature, pool, validation_metadata, mix_assignment, conditions, alphas, replicates
                )
                check_rows += [("t_invariance", name, key, value) for key, value in result.items()]
            table_rows += [{"model": model, "seed": seed, **row} for row in classification_rows(predictions, temperature)]

        if len(paths) > 1 and "prob_stress" in pd.read_csv(paths[0], nrows=1).columns:
            table_rows += [
                {"model": f"{model}_ensemble", "seed": "ensemble", **row}
                for row in classification_rows(ensemble_predictions(paths), None)
            ]

    table = pd.DataFrame(table_rows)
    table.insert(0, "dataset", dataset)
    columns = ["dataset", "model", "seed", "split", "condition", "probabilities", "temperature", "n",
               "macro_f1", "auroc", "brier", "nll", "ece_width15", "ece_mass15"]
    table = table[columns].sort_values(["model", "seed", "split", "condition", "probabilities"])
    table.to_csv(output_dir / f"{dataset}_classification.csv", index=False)

    # Step-7 done-when: B1, B2 and the ensembles (M0, M1) are in the long tables; the classification table exists.
    for long_path in sorted((output_dir / "long").glob(f"{dataset}_*.parquet")):
        long_table = pd.read_parquet(long_path, columns=["model", "seed", "method"])
        counts = long_table["method"].astype(str).value_counts()
        name = long_path.stem.removeprefix(f"{dataset}_")
        expected = ["M0", "M1"] if name.endswith("seedensemble") else sorted(STEP7_METHODS)
        for method in expected:
            check_rows.append(("long_table_rows", name, method, int(counts.get(method, 0))))
    check_rows.append(("classification_table", dataset, "rows", len(table)))

    checks = pd.DataFrame(check_rows, columns=["section", "scope", "key", "value"])
    checks.to_csv(output_dir / "step7_checks.csv", index=False)
    print(checks.to_string(index=False))

    changed = checks[(checks["section"] == "t_invariance") & (checks["key"] == "set_rows_changed")]
    missing = checks[(checks["section"] == "long_table_rows") & (checks["value"].astype(int) == 0)]
    if changed["value"].astype(int).sum() or not missing.empty:
        raise RuntimeError("Step-7 checks failed (changed sets under temperature scaling or missing long-table rows).")


if __name__ == "__main__":
    main()
