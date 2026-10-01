from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.baselines import build_linear_svm, fit_logreg_with_c_selection, logreg_logits
from src.degradations import iter_prediction_inputs, load_condition_texts
from src.evaluate import (
    build_prediction_rows,
    compute_classification_metrics,
    predicted_labels,
    save_predictions,
)
from src.utils import (
    get_commit_hash,
    get_library_versions,
    load_processed_split,
    load_yaml_config,
    now_iso,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines and save their run folders.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    started_at = now_iso()
    start_time = time.perf_counter()

    seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    conditions = config["conditions"]
    baseline_config = config["baselines"]
    max_features = int(baseline_config["max_features"])
    ngram_range = tuple(baseline_config["ngram_range"])

    split_to_df = {
        name: load_processed_split(data_dir / f"{name}.csv")
        for name in ["train", "validation", "calibration", "test"]
    }
    condition_texts = load_condition_texts(data_dir / f"{dataset}_conditions.csv")

    logreg, best_c, c_scores = fit_logreg_with_c_selection(
        train_df=split_to_df["train"],
        validation_df=split_to_df["validation"],
        c_grid=[float(c_value) for c_value in baseline_config["logreg_c_grid"]],
        max_features=max_features,
        ngram_range=ngram_range,
        seed=seed,
    )

    svm = build_linear_svm(max_features, ngram_range, seed)
    svm.fit(split_to_df["train"]["text"].tolist(), split_to_df["train"]["label"].to_numpy())

    model_details = {
        "tfidf_logreg": {
            "logreg_c": best_c,
            "validation_macro_f1_by_c": {str(c_value): score for c_value, score in c_scores.items()},
            "in_conformal": True,
        },
        "tfidf_linear_svm": {"svm_c": float(svm.named_steps["clf"].C), "in_conformal": False},
    }

    for model_key, model in {"tfidf_logreg": logreg, "tfidf_linear_svm": svm}.items():
        prediction_frames: list[pd.DataFrame] = []
        clean_metrics: dict[str, dict[str, float]] = {}

        for split_name, condition, df, texts in iter_prediction_inputs(split_to_df, condition_texts, conditions):
            if model_key == "tfidf_logreg":
                frame = build_prediction_rows(
                    df["id"], split_name, condition, df["label"], texts, logits=logreg_logits(model, texts)
                )
            else:
                frame = build_prediction_rows(
                    df["id"], split_name, condition, df["label"], texts,
                    decision_scores=model.decision_function(texts.tolist()),
                )
            prediction_frames.append(frame)

            if condition == "clean":
                clean_metrics[split_name] = compute_classification_metrics(frame["label"], predicted_labels(frame))

        run_dir = Path(config["paths"]["runs_dir"]) / dataset / model_key / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(pd.concat(prediction_frames, ignore_index=True), run_dir / "predictions.csv.gz")

        save_json(
            {
                "dataset": dataset,
                "model": model_key,
                "seed": seed,
                "commit": get_commit_hash(),
                **model_details[model_key],
                "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                "conditions": conditions,
                "n_rows": {name: len(df) for name, df in split_to_df.items()},
                "clean_metrics": clean_metrics,
                "total_seconds": round(time.perf_counter() - start_time, 1),
                "started_at": started_at,
                "finished_at": now_iso(),
                "versions": get_library_versions(),
                "config": {
                    "seed": seed,
                    "data": config["data"],
                    "baselines": baseline_config,
                    "conditions": conditions,
                },
            },
            run_dir / "run_meta.json",
        )

        print(f"Saved run folder to: {run_dir.resolve()}")
        print(pd.DataFrame(clean_metrics).T.round(4).to_string())

    print(f"\nLogReg C by validation macro-F1: {c_scores} -> C = {best_c}")


if __name__ == "__main__":
    main()
