from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data import (
    build_split_report,
    check_overlaps,
    deduplicate_pool,
    load_dreaddit_dataset,
    save_splits,
    select_columns,
    split_pool,
    summarize_split,
)
from src.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Dreaddit and build processed splits.")
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

    dataset_name = config["data"]["dataset_name"]
    split_config = config["data"]["split"]
    output_dir = Path(config["data"]["output_dir"])

    raw_splits = load_dreaddit_dataset(dataset_name)
    raw_sizes = {f"hf_{name}": len(df) for name, df in raw_splits.items()}

    pool_df = select_columns(
        pd.concat([raw_splits["train"], raw_splits["validation"]], ignore_index=True)
    )
    test_df = select_columns(raw_splits["test"])
    raw_sizes["pool"] = len(pool_df)

    pool_df, dedup_stats = deduplicate_pool(pool_df, test_df)
    pool_splits, random_state_used = split_pool(pool_df, split_config)

    processed_splits = {**pool_splits, "test": test_df}
    overlaps = check_overlaps(processed_splits)

    shared = {pair: counts for pair, counts in overlaps.items() if any(counts.values())}
    if shared:
        raise RuntimeError(f"Splits share posts or texts: {shared}")

    report_df = build_split_report(
        dataset_name=dataset_name,
        raw_sizes=raw_sizes,
        dedup_stats=dedup_stats,
        split_config=split_config,
        random_state_used=random_state_used,
        splits=processed_splits,
        overlaps=overlaps,
    )

    save_splits(processed_splits, output_dir=output_dir)
    report_df.to_csv(output_dir / "dreaddit_split_report.csv", index=False)

    print("Processed splits saved to... ", output_dir.resolve())
    print(f"random_state used: {random_state_used}")
    print(f"duplicates: {dedup_stats}")
    for split_name, df in processed_splits.items():
        print(summarize_split(df, split_name))


if __name__ == "__main__":
    main()
