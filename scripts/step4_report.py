from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score

from src.evaluate import predicted_labels
from src.utils import load_yaml_config

BASELINES = ["tfidf_logreg", "tfidf_linear_svm"]
ENCODER = "distilbert"
GAP_THRESHOLD = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-4 report: run completeness and DistilBERT validation-test gap.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def macro_f1(frame: pd.DataFrame) -> float:
    return float(f1_score(frame["label"], predicted_labels(frame), average="macro"))


def check_run(
        run_dir: Path,
        expected_keys: set[tuple[str, str]],
        split_ids: dict[str, np.ndarray],
        with_occlusion: bool,
) -> tuple[list[tuple[str, Any]], pd.DataFrame]:

    """
    Completeness checks of one run folder. Returns (key, value) rows and the predictions.
    """

    rows: list[tuple[str, Any]] = []
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")

    keys = set(zip(predictions["split"], predictions["condition"]))
    ids_ok = all(
        np.array_equal(group["id"].to_numpy(), split_ids[split_name])
        for (split_name, _), group in predictions.groupby(["split", "condition"], sort=False)
    )

    rows.append(("run_meta_exists", (run_dir / "run_meta.json").exists()))
    rows.append(("n_rows", len(predictions)))
    rows.append(("all_split_conditions_present", keys == expected_keys))
    rows.append(("ids_match_splits", ids_ok))
    rows.append(("no_text_column", "text" not in predictions.columns))
    rows.append(("no_missing_values", not predictions.isna().any().any()))
    if with_occlusion:
        occlusion = pd.read_csv(run_dir / "occlusion.csv")
        rows.append(("occlusion_rows", len(occlusion)))
        rows.append(("occlusion_no_text_column", "text" not in occlusion.columns))
        rows.append(("train_log_exists", (run_dir / "train_log.json").exists()))

    return rows, predictions


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conditions = config["conditions"]
    seed = int(config["seed"])
    encoder_seeds = [int(s) for s in config["encoders"]["seeds"]]

    splits = {name: pd.read_csv(data_dir / f"{name}.csv") for name in ["validation", "calibration", "test"]}
    split_ids = {name: df["id"].to_numpy() for name, df in splits.items()}
    metadata = pd.concat(splits.values())[["id", "domain", "confidence"]]

    expected_keys = {("validation", "clean")} | {
        (split_name, condition) for split_name in ["calibration", "test"] for condition in conditions
    }

    runs = [(model, seed) for model in BASELINES] + [(ENCODER, s) for s in encoder_seeds]
    rows: list[tuple[str, str, str, Any]] = []
    clean: dict[tuple[str, int], pd.DataFrame] = {}

    for model, run_seed in runs:
        run_name = f"{model}/seed_{run_seed}"
        check_rows, predictions = check_run(
            runs_dir / model / f"seed_{run_seed}",
            expected_keys,
            split_ids,
            with_occlusion=model in config["encoders"]["occlusion_models"],
        )
        rows += [("files", run_name, key, value) for key, value in check_rows]

        clean[(model, run_seed)] = predictions[predictions["condition"] == "clean"].merge(metadata, on="id")
        frame = clean[(model, run_seed)]
        validation_f1 = macro_f1(frame[frame["split"] == "validation"])
        test_f1 = macro_f1(frame[frame["split"] == "test"])
        rows.append(("metrics", run_name, "validation_macro_f1", round(validation_f1, 4)))
        rows.append(("metrics", run_name, "test_macro_f1", round(test_f1, 4)))
        rows.append(("metrics", run_name, "gap_validation_minus_test", round(validation_f1 - test_f1, 4)))

    encoder_frames = [clean[(ENCODER, s)] for s in encoder_seeds]
    gaps = [
        macro_f1(f[f["split"] == "validation"]) - macro_f1(f[f["split"] == "test"]) for f in encoder_frames
    ]
    median_gap = float(np.median(gaps))
    rows.append(("gap", ENCODER, "median_gap_over_seeds", round(median_gap, 4)))
    rows.append(("gap", ENCODER, "threshold", GAP_THRESHOLD))
    rows.append(("gap", ENCODER, "gap_below_threshold", median_gap < GAP_THRESHOLD))

    # Diagnostics (descriptive; they do not change the criterion).
    split_report = pd.read_csv(data_dir / f"{dataset}_split_report.csv")
    overlap = split_report[split_report["section"] == "overlap"]
    rows.append(("diagnostics", "data", "shared_posts_all_pairs", int(overlap.loc[overlap["key"] == "shared_posts", "value"].astype(int).sum())))
    rows.append(("diagnostics", "data", "shared_texts_all_pairs", int(overlap.loc[overlap["key"] == "shared_texts", "value"].astype(int).sum())))

    gaps_known_confidence = [
        macro_f1(f[f["split"] == "validation"]) - macro_f1(f[(f["split"] == "test") & f["confidence"].notna()])
        for f in encoder_frames
    ]
    rows.append((
        "diagnostics", ENCODER, "median_gap_test_without_missing_confidence",
        round(float(np.median(gaps_known_confidence)), 4),
    ))
    rows.append((
        "diagnostics", ENCODER, "median_test_macro_f1_missing_confidence_rows",
        round(float(np.median([macro_f1(f[(f["split"] == "test") & f["confidence"].isna()]) for f in encoder_frames])), 4),
    ))

    for domain in sorted(metadata["domain"].unique()):
        validation_f1 = np.median([macro_f1(f[(f["split"] == "validation") & (f["domain"] == domain)]) for f in encoder_frames])
        test_f1 = np.median([macro_f1(f[(f["split"] == "test") & (f["domain"] == domain)]) for f in encoder_frames])
        rows.append(("diagnostics", ENCODER, f"{domain}.median_validation_macro_f1", round(float(validation_f1), 4)))
        rows.append(("diagnostics", ENCODER, f"{domain}.median_test_macro_f1", round(float(test_f1), 4)))

    report = pd.DataFrame(rows, columns=["section", "model", "key", "value"])
    report_path = runs_dir / "step4_report.csv"
    report.to_csv(report_path, index=False)

    print("Saved report to: ", report_path.resolve())
    print(report[report["section"] != "files"].to_string(index=False))

    failed = report[(report["section"] == "files") & (report["value"].astype(str) == "False")]
    if not failed.empty:
        raise RuntimeError(f"Run folder checks failed:\n{failed.to_string(index=False)}")


if __name__ == "__main__":
    main()
