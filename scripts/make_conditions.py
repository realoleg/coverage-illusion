from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.degradations import (
    LENGTH_BUCKET_LABELS,
    MIX_FAMILY,
    QWERTY_NEIGHBORS,
    RANDOM_FAMILIES,
    assign_mix,
    build_condition_texts,
    load_condition_texts,
    make_stream,
    parse_condition,
    stream_seed,
)
from src.utils import load_yaml_config

SPLITS = ["calibration", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build degraded texts of calibration and test for all conditions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_subsequence(short: list[str], long: list[str]) -> bool:
    remaining = iter(long)
    return all(word in remaining for word in short)


def check_typos(clean: str, low: str, high: str) -> bool:

    """
    Same length; every change is an ASCII letter replaced by a QWERTY neighbor (case kept);
    every change of the lower level appears in the higher level with the same replacement.
    """

    if not len(clean) == len(low) == len(high):
        return False

    for original, char_low, char_high in zip(clean, low, high):
        for changed in (char_low, char_high):
            if changed != original and (
                original.lower() not in QWERTY_NEIGHBORS
                or changed.lower() not in QWERTY_NEIGHBORS[original.lower()]
                or changed.isupper() != original.isupper()
            ):
                return False
        if char_low != original and char_high != char_low:
            return False

    return True


def run_checks(
        texts: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        conditions: list[str],
        base_seed: int,
) -> list[tuple[str, str, Any]]:

    """
    Done-when checks of step 3, computed on the texts read back from disk. Returns (split, key, value) rows.
    """

    rows: list[tuple[str, str, Any]] = []
    families = {condition: parse_condition(condition)[0] for condition in conditions if condition != "clean"}

    for split_name, df in splits.items():
        by_condition = {
            condition: texts[(texts["split"] == split_name) & (texts["condition"] == condition)]
            .set_index("id")["text"].loc[df["id"].to_numpy()].tolist()
            for condition in conditions
        }
        originals = df["text"].astype(str).tolist()

        rows.append((split_name, "clean_equals_original", by_condition["clean"] == originals))

        for condition, family in families.items():
            if family in ("head", "tail"):
                fraction = parse_condition(condition)[1]
                ok = True
                for original, degraded in zip(originals, by_condition[condition]):
                    words = original.split()
                    keep_n = max(1, math.ceil(len(words) * fraction))
                    expected = words[:keep_n] if family == "head" else words[-keep_n:]
                    ok &= degraded.split() == expected
                rows.append((split_name, f"{condition}_exact_words", ok))

        for family in RANDOM_FAMILIES:
            levels = sorted(
                (condition for condition, fam in families.items() if fam == family),
                key=lambda condition: parse_condition(condition)[1],
            )
            for low, high in zip(levels, levels[1:]):
                if family == "del":
                    ok = all(
                        is_subsequence(t_high.split(), t_low.split()) and is_subsequence(t_low.split(), original.split())
                        for original, t_low, t_high in zip(originals, by_condition[low], by_condition[high])
                    )
                else:
                    ok = all(
                        check_typos(original, t_low, t_high)
                        for original, t_low, t_high in zip(originals, by_condition[low], by_condition[high])
                    )
                rows.append((split_name, f"{low}_nested_in_{high}", ok))

    # One stream per (example, family): seeds and the first draws must all differ.
    stream_families = RANDOM_FAMILIES + [MIX_FAMILY]
    ids = pd.concat([df["id"] for df in splits.values()]).tolist()
    seeds = [stream_seed(base_seed, example_id, family) for example_id in ids for family in stream_families]
    first_draws = [
        tuple(make_stream(base_seed, example_id, family).random(2))
        for example_id in ids for family in stream_families
    ]
    rows.append(("all", "stream_pairs", len(seeds)))
    rows.append(("all", "stream_unique_seeds", len(set(seeds))))
    rows.append(("all", "stream_unique_first_draws", len(set(first_draws))))
    rows.append(("all", "streams_unique", len(set(seeds)) == len(set(first_draws)) == len(seeds)))

    return rows


def build_report(
        texts: pd.DataFrame,
        mix: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        conditions: list[str],
        base_seed: int,
        texts_path: Path,
        check_rows: list[tuple[str, str, Any]],
) -> pd.DataFrame:

    """
    Long-format report with columns: section, split, key, value.
    """

    rows: list[tuple[str, str, str, Any]] = []

    rows.append(("source", "all", "base_seed", base_seed))
    rows.append(("source", "all", "conditions", " ".join(conditions)))
    rows.append(("file", "all", "conditions_file", texts_path.name))
    rows.append(("file", "all", "conditions_sha256", file_sha256(texts_path)))
    rows.append(("file", "all", "conditions_rows", len(texts)))

    for split_name in splits:
        split_texts = texts[texts["split"] == split_name]
        clean_words = split_texts.loc[split_texts["condition"] == "clean", "n_words"].sum()

        for condition in conditions:
            cond = split_texts[split_texts["condition"] == condition]
            rows.append(("length", split_name, f"{condition}.min_words", int(cond["n_words"].min())))
            rows.append(("length", split_name, f"{condition}.median_words", float(cond["n_words"].median())))
            counts = cond["length_bucket"].value_counts()
            for bucket in LENGTH_BUCKET_LABELS:
                rows.append(("length", split_name, f"{condition}.{bucket}", int(counts.get(bucket, 0))))

            family = condition.split("_")[0]
            if family == "del":
                rows.append((
                    "rate", split_name, f"{condition}.deleted_word_share",
                    round(1 - cond["n_words"].sum() / clean_words, 4),
                ))
            if family == "typo":
                originals = splits[split_name].set_index("id")["text"].loc[cond["id"].to_numpy()]
                changed = sum(a != b for o, d in zip(originals, cond["text"]) for a, b in zip(o, d))
                letters = sum(char.lower() in QWERTY_NEIGHBORS for text in originals for char in text)
                rows.append(("rate", split_name, f"{condition}.changed_letter_share", round(changed / letters, 4)))

    calibration = splits["calibration"][["id", "label"]].merge(mix, on="id")
    mixed = calibration.merge(
        texts[texts["split"] == "calibration"][["id", "condition", "length_bucket"]],
        on=["id", "condition"],
    )
    assert len(mixed) == len(splits["calibration"])

    for condition in conditions:
        rows.append(("mix", "calibration", f"n.{condition}", int((mixed["condition"] == condition).sum())))
    group_sizes = mixed.groupby(["length_bucket", "label"]).size()
    for bucket in LENGTH_BUCKET_LABELS:
        for label in [0, 1]:
            rows.append((
                "mix", "calibration", f"bucket_label.{bucket}.{label}", int(group_sizes.get((bucket, label), 0))
            ))

    for split_name, key, value in check_rows:
        rows.append(("checks", split_name, key, value))

    return pd.DataFrame(rows, columns=["section", "split", "key", "value"])


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_yaml_config(args.config)

    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["output_dir"])
    conditions = config["conditions"]

    splits = {name: pd.read_csv(data_dir / f"{name}.csv") for name in SPLITS}

    texts_path = data_dir / f"{dataset}_conditions.csv"
    mix_path = data_dir / f"{dataset}_mix_assignment.csv"
    report_path = data_dir / f"{dataset}_conditions_report.csv"

    build_condition_texts(splits, conditions, base_seed).to_csv(texts_path, index=False)

    calibration_ids = splits["calibration"]["id"]
    mix = pd.DataFrame({
        "id": calibration_ids.to_numpy(),
        "condition": assign_mix(calibration_ids, conditions, base_seed).to_numpy(),
    })
    mix.to_csv(mix_path, index=False)

    # Checks and report use the texts read back from disk, exactly as the models will see them.
    texts = load_condition_texts(texts_path)
    check_rows = run_checks(texts, splits, conditions, base_seed)
    report = build_report(texts, mix, splits, conditions, base_seed, texts_path, check_rows)
    report.to_csv(report_path, index=False)

    print("Saved condition texts to: ", texts_path.resolve())
    print("Saved M3-mix assignment to: ", mix_path.resolve())
    print("Saved report to: ", report_path.resolve())
    print(report[report["section"].isin(["file", "rate", "checks"])].to_string(index=False))

    failed = [
        (split_name, key) for split_name, key, value in check_rows
        if isinstance(value, bool) and not value
    ]
    if failed:
        raise RuntimeError(f"Condition checks failed: {failed}")


if __name__ == "__main__":
    main()
