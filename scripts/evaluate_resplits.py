from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedGroupKFold

from src.conformal import (
    INDICATORS,
    METHODS,
    build_eval_frame,
    calibration_frame,
    group_summaries,
    indicator_frame,
    predict_sets,
    reference_band,
    reweight,
    summarize_indicators,
    true_class_scores,
)
from src.degradations import make_stream
from src.utils import load_yaml_config, now_iso

PROTOCOLS = ["resplit", "official_test"]
LONG_GROUP_TYPES = ["domain", "domain_class", "agreement"]
CLASS_INDICATORS = [name for name in INDICATORS if name != "covered"]
LONG_METRICS = (
    ["n", "n_0", "n_1", "coverage", "c0", "c1", "sensitivity"]
    + CLASS_INDICATORS
    + [f"{name}_{label}" for name in CLASS_INDICATORS for label in [0, 1]]
)
GROUP_METRICS = [
    "n", "n_0", "n_1", "n_calibration", "band_low", "band_high",
    "coverage", "c0", "c1", "set_size", "full", "empty", "workload", "uncertainty_referral", "sensitivity",
]
SUMMARY_METRICS = [
    "coverage", "c0", "c1", "sensitivity", "set_size", "singleton", "empty", "full", "flag",
    "workload", "uncertainty_referral", "n_infinite_thresholds", "n_calibration", "n",
]
PER_SEED_METRICS = ["coverage", "c0", "c1", "sensitivity", "workload"]
KEYS = ["dataset", "model", "protocol", "condition", "method", "calibration", "alpha"]
DONE_WHEN_TOLERANCE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6: resplits / official-test bootstraps, long table, aggregates.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


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


def evaluate_run(task: dict[str, Any]) -> dict[str, Any]:

    """
    All replicates x conditions x methods x alphas of one run; writes its long and group tables (parquet).
    """

    start = time.perf_counter()
    model, seed = task["model"], task["seed"]
    predictions = pd.read_csv(task["predictions_path"])
    frames, mix = pool_frames(predictions, task["pool"], task["mix_assignment"], task["conditions"])

    long_rows: list[dict[str, Any]] = []
    group_tables: list[pd.DataFrame] = []

    for protocol, replicates in task["replicates"].items():
        for replicate, (calibration_positions, test_positions) in enumerate(replicates):
            with_groups = replicate < task["n_group_replicates"]
            calibration_by_condition = {
                condition: frame.iloc[calibration_positions] for condition, frame in frames.items()
            }
            mix_calibration = mix.iloc[calibration_positions]

            for condition in task["conditions"]:
                test = frames[condition].iloc[test_positions]
                labels = test["label"].to_numpy()

                for method, spec in METHODS.items():
                    calibration = calibration_frame(method, calibration_by_condition, mix_calibration, condition)
                    calibration_labels = calibration["label"].to_numpy()
                    keys = {
                        "dataset": task["dataset"], "model": model, "seed": seed, "protocol": protocol,
                        "replicate": replicate, "condition": condition, "method": method,
                        "calibration": spec["calibration"],
                    }

                    for alpha in task["alphas"]:
                        sets, n_infinite = predict_sets(calibration, test, method, alpha)
                        indicators = indicator_frame(sets, labels)
                        summary = summarize_indicators(indicators, labels)
                        long_rows.append({
                            **keys,
                            "alpha": alpha,
                            "n_calibration": len(calibration_labels),
                            "n_calibration_0": int((calibration_labels == 0).sum()),
                            "n_calibration_1": int((calibration_labels == 1).sum()),
                            "n_infinite_thresholds": n_infinite,
                            **{metric: summary[metric] for metric in LONG_METRICS},
                        })

                        if with_groups:
                            groups = group_summaries(indicators, test, calibration, alpha, LONG_GROUP_TYPES)
                            group_tables.append(groups[["group_type", "group", *GROUP_METRICS]].assign(**keys, alpha=alpha))

            if (replicate + 1) % 100 == 0:
                print(f"[{now_iso()}] {model} seed {seed}: {protocol} {replicate + 1}/{len(replicates)}", flush=True)

    long_table = pd.DataFrame(long_rows)
    group_table = pd.concat(group_tables, ignore_index=True)
    for table, path in [(long_table, task["long_path"]), (group_table, task["groups_path"])]:
        for column in table.select_dtypes(include="object").columns:
            table[column] = table[column].astype("category")
        table.to_parquet(path, index=False, compression="zstd")

    # Descriptive exchangeability diagnostic on the original split: true-class M0 scores, calibration vs test.
    clean = frames["clean"]
    is_calibration = task["pool"]["split"].to_numpy() == "calibration"
    scores = true_class_scores(clean)
    ks = ks_2samp(scores[is_calibration], scores[~is_calibration])

    return {
        "model": model, "seed": seed,
        "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        "n_calibration": int(is_calibration.sum()), "n_test": int((~is_calibration).sum()),
        "long_rows": len(long_table), "group_rows": len(group_table),
        "seconds": round(time.perf_counter() - start, 1),
    }


def add_band_flags(table: pd.DataFrame) -> pd.DataFrame:

    """
    Per replicate: is coverage (c0, c1) below / inside / above its reference band at that replicate's sizes.
    Adds the columns in place.
    """

    out = table
    for metric, n_calibration, n_test in [
        ("coverage", "n_calibration", "n"),
        ("c0", "n_calibration_0", "n_0"),
        ("c1", "n_calibration_1", "n_1"),
    ]:
        sizes = out[[n_calibration, n_test, "alpha"]].drop_duplicates()
        bands = {
            (a, b, c): reference_band(int(a), int(b), float(c))
            for a, b, c in sizes.itertuples(index=False, name=None)
        }
        low_high = np.array([bands[key] for key in zip(out[n_calibration], out[n_test], out["alpha"])])
        out[f"{metric}_below_band"] = (out[metric] < low_high[:, 0]).astype(float)
        out[f"{metric}_above_band"] = (out[metric] > low_high[:, 1]).astype(float)
        out[f"{metric}_in_band"] = 1.0 - out[f"{metric}_below_band"] - out[f"{metric}_above_band"]
    return out


def add_prevalence_metrics(table: pd.DataFrame, prevalence_grid: list[float]) -> pd.DataFrame:
    out = table
    for prevalence in prevalence_grid:
        at_prevalence = reweight(out, prevalence)
        out[f"coverage_pi={prevalence}"] = at_prevalence["coverage"]
        out[f"workload_per_1000_pi={prevalence}"] = 1000 * at_prevalence["workload"]
        out[f"uncertainty_referrals_per_1000_pi={prevalence}"] = 1000 * at_prevalence["uncertainty_referral"]
    return out


def describe(table: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:

    """
    Long summary: one row per (keys, metric) with the number of values, mean, median, 2.5% and 97.5% quantiles.
    Computed on the wide table (no melt), so memory stays proportional to the input.
    """

    grouped = table.groupby(keys, observed=True, sort=True)[metrics]
    quantiles = grouped.quantile([0.025, 0.5, 0.975])
    statistics = {
        "mean": grouped.mean(),
        "q025": quantiles.xs(0.025, level=-1),
        "median": quantiles.xs(0.5, level=-1),
        "q975": quantiles.xs(0.975, level=-1),
    }
    out = pd.concat(
        {name: frame.stack() for name, frame in statistics.items()}, axis=1
    ).rename_axis([*keys, "metric"]).reset_index()
    sizes = grouped.size().rename("n_values").reset_index()
    out = out.merge(sizes, on=keys, how="left")
    return out[[*keys, "metric", "n_values", "mean", "median", "q025", "q975"]]


def aggregate_model(long_paths: list[Path], group_paths: list[Path], prevalence_grid: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Aggregates of one model over its seeds: over all (seed, replicate) and per seed; groups over all.
    """

    long_table = pd.concat([pd.read_parquet(path) for path in long_paths], ignore_index=True)
    long_table = add_prevalence_metrics(add_band_flags(long_table), prevalence_grid)
    for column in KEYS:
        long_table[column] = long_table[column].astype(str) if column != "alpha" else long_table[column]

    band_metrics = [f"{m}_{flag}" for m in ["coverage", "c0", "c1"] for flag in ["below_band", "in_band", "above_band"]]
    prevalence_metrics = [
        f"{name}_pi={p}" for p in prevalence_grid
        for name in ["coverage", "workload_per_1000", "uncertainty_referrals_per_1000"]
    ]
    overall = describe(long_table, KEYS, SUMMARY_METRICS + band_metrics + prevalence_metrics).assign(seed="all")
    per_seed_metrics = PER_SEED_METRICS + ["workload_per_1000_pi=0.1", "coverage_below_band", "c1_below_band"]
    per_seed = describe(long_table.assign(seed=long_table["seed"].astype(str)), KEYS + ["seed"], per_seed_metrics)
    summary = pd.concat([overall, per_seed], ignore_index=True)

    groups = pd.concat(
        [pd.read_parquet(path, columns=[*KEYS, "group_type", "group", "n", "n_calibration", "band_low", "band_high",
                                        "coverage", "c1", "workload", "sensitivity"]) for path in group_paths],
        ignore_index=True,
    )
    for column in [*KEYS, "group_type", "group"]:
        groups[column] = groups[column].astype(str) if column != "alpha" else groups[column]
    groups["coverage_below_band"] = (groups["coverage"] < groups["band_low"]).astype(float)
    grouped = groups.groupby([*KEYS, "group_type", "group"], sort=True)
    group_summary = pd.concat([
        grouped.size().rename("n_values"),
        grouped["n"].median().rename("n_test_median"),
        grouped["n_calibration"].median().rename("n_calibration_median"),
        grouped["coverage"].median().rename("coverage_median"),
        grouped["coverage"].quantile(0.025).rename("coverage_q025"),
        grouped["coverage"].quantile(0.975).rename("coverage_q975"),
        grouped["coverage_below_band"].mean().rename("coverage_below_band_share"),
        grouped["c1"].median().rename("c1_median"),
        grouped["workload"].median().rename("workload_median"),
        grouped["sensitivity"].median().rename("sensitivity_median"),
    ], axis=1).reset_index()

    return summary, group_summary


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)
    start = time.perf_counter()

    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conditions = config["conditions"]
    alphas = [float(alpha) for alpha in config["conformal"]["alpha_values"]]
    prevalence_grid = [float(p) for p in config["conformal"]["prevalence_grid"]]
    evaluation = config["evaluation"]
    output_dir = Path(evaluation["output_dir"])
    (output_dir / "long").mkdir(parents=True, exist_ok=True)
    (output_dir / "groups").mkdir(parents=True, exist_ok=True)

    pool = pd.concat(
        [pd.read_csv(data_dir / f"{name}.csv").assign(split=name) for name in ["calibration", "test"]],
        ignore_index=True,
    )[["id", "post_id", "label", "domain", "confidence", "split"]]
    mix_assignment = pd.read_csv(data_dir / f"{dataset}_mix_assignment.csv")

    replicates = make_replicates(pool, int(evaluation["n_replicates"]), int(evaluation["resplit_folds"]), base_seed)
    print(f"[{now_iso()}] Built {len(replicates['resplit'])} resplits and {len(replicates['official_test'])} bootstraps", flush=True)

    tasks = []
    for predictions_path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        if "prob_stress" not in pd.read_csv(predictions_path, nrows=1).columns:
            continue  # LinearSVC has no probabilities and is not part of conformal prediction.
        model = predictions_path.parent.parent.name
        seed = int(predictions_path.parent.name.removeprefix("seed_"))
        tasks.append({
            "dataset": dataset, "model": model, "seed": seed, "predictions_path": predictions_path,
            "pool": pool, "mix_assignment": mix_assignment, "conditions": conditions, "alphas": alphas,
            "replicates": replicates, "n_group_replicates": int(evaluation["n_group_replicates"]),
            "long_path": output_dir / "long" / f"{dataset}_{model}_seed{seed}.parquet",
            "groups_path": output_dir / "groups" / f"{dataset}_{model}_seed{seed}.parquet",
        })

    with ProcessPoolExecutor(max_workers=int(evaluation["workers"])) as executor:
        run_results = list(executor.map(evaluate_run, tasks))
    print(f"[{now_iso()}] All runs evaluated", flush=True)

    summaries, group_summaries_all = [], []
    for model in sorted({task["model"] for task in tasks}):
        model_tasks = [task for task in tasks if task["model"] == model]
        summary, group_summary = aggregate_model(
            [task["long_path"] for task in model_tasks], [task["groups_path"] for task in model_tasks], prevalence_grid
        )
        summaries.append(summary)
        group_summaries_all.append(group_summary)

    summary = pd.concat(summaries, ignore_index=True)
    summary = summary[["seed", *[c for c in summary.columns if c != "seed"]]]
    summary.to_csv(output_dir / f"{dataset}_summary.csv", index=False)
    pd.concat(group_summaries_all, ignore_index=True).to_csv(output_dir / f"{dataset}_groups_summary.csv", index=False)
    pd.DataFrame(run_results).sort_values(["model", "seed"]).drop(columns=["seconds"]).to_csv(
        output_dir / f"{dataset}_ks.csv", index=False
    )

    # Done-when: clean M0 median coverage over resplits within 0.90 +/- 0.01 for every model (all seeds pooled).
    report_rows: list[tuple[str, str, str, Any]] = []
    main_alpha = alphas[0]
    m0_clean = summary[
        (summary["protocol"] == "resplit") & (summary["condition"] == "clean") & (summary["method"] == "M0")
        & (summary["alpha"] == main_alpha) & (summary["metric"] == "coverage")
    ]
    for _, row in m0_clean.iterrows():
        report_rows.append(("done_when", f"{row['model']}/seed={row['seed']}", "M0_clean_median_coverage", round(row["median"], 4)))
    pooled = m0_clean[m0_clean["seed"] == "all"]
    target = 1 - main_alpha
    for _, row in pooled.iterrows():
        report_rows.append((
            "done_when", f"{row['model']}/seed=all", "within_0.90_pm_0.01",
            bool(abs(row["median"] - target) <= DONE_WHEN_TOLERANCE),
        ))

    sizes = np.array([[len(c), len(t)] for c, t in replicates["resplit"]])
    report_rows += [
        ("replicates", "resplit", "n_calibration_min", int(sizes[:, 0].min())),
        ("replicates", "resplit", "n_calibration_max", int(sizes[:, 0].max())),
        ("replicates", "resplit", "n_test_min", int(sizes[:, 1].min())),
        ("replicates", "resplit", "n_test_max", int(sizes[:, 1].max())),
        ("replicates", "official_test", "n_calibration_min", int(min(len(c) for c, _ in replicates["official_test"]))),
        ("replicates", "official_test", "n_calibration_max", int(max(len(c) for c, _ in replicates["official_test"]))),
    ]
    for result in run_results:
        name = f"{result['model']}/seed={result['seed']}"
        report_rows += [
            ("ks_descriptive", name, "ks_statistic", round(result["ks_statistic"], 4)),
            ("ks_descriptive", name, "ks_pvalue", round(result["ks_pvalue"], 4)),
            ("runtime", name, "long_rows", result["long_rows"]),
            ("runtime", name, "group_rows", result["group_rows"]),
        ]
    report_rows.append(("runtime", "all", "total_minutes", round((time.perf_counter() - start) / 60, 1)))

    report = pd.DataFrame(report_rows, columns=["section", "scope", "key", "value"])
    report.to_csv(output_dir / "step6_report.csv", index=False)
    print(report[report["section"] != "runtime"].to_string(index=False))


if __name__ == "__main__":
    main()
