from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.plots import (
    plot_clean_model_comparison,
    plot_conformal_metric_under_stress,
    plot_stress_classification_performance,
)


def load_required_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required results table not found: {path}")
    return pd.read_csv(path)


def build_clean_performance_table(
    baseline_metrics_df: pd.DataFrame,
    transformer_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the main clean test comparison table:
    both baselines + DistilBERT.
    """
    baseline_test = baseline_metrics_df[baseline_metrics_df["split"] == "test"].copy()
    transformer_test = transformer_metrics_df[transformer_metrics_df["split"] == "test"].copy()

    out = pd.concat([baseline_test, transformer_test], ignore_index=True)

    out = out.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    keep_cols = [
        "model_name",
        "split",
        "n_examples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
    ]
    return out[keep_cols].copy()


def build_stress_classification_table(
    stress_classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only the columns we need for the main stress-test classification table.
    """
    keep_cols = [
        "stress_test",
        "model_name",
        "n_examples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
    ]
    out = stress_classification_df[keep_cols].copy()
    return out.reset_index(drop=True)


def build_conformal_test_table(
    stress_conformal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only the most useful columns for conformal reporting.
    """
    keep_cols = [
        "stress_test",
        "alpha",
        "target_coverage",
        "empirical_coverage",
        "avg_set_size",
        "singleton_rate",
        "empty_rate",
        "full_set_rate",
    ]
    out = stress_conformal_df[keep_cols].copy()
    return out.reset_index(drop=True)


def main() -> None:
    load_dotenv()
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_df = load_required_table(tables_dir / "baseline_metrics.csv")
    transformer_metrics_df = load_required_table(tables_dir / "transformer_metrics.csv")
    stress_classification_df = load_required_table(
        tables_dir / "stress_test_classification_metrics.csv"
    )
    stress_conformal_df = load_required_table(
        tables_dir / "stress_test_conformal_metrics.csv"
    )

    clean_table = build_clean_performance_table(
        baseline_metrics_df=baseline_metrics_df,
        transformer_metrics_df=transformer_metrics_df,
    )
    stress_classification_table = build_stress_classification_table(
        stress_classification_df=stress_classification_df
    )
    stress_conformal_table = build_conformal_test_table(
        stress_conformal_df=stress_conformal_df
    )

    clean_table_path = tables_dir / "final_clean_performance_table.csv"
    stress_classification_table_path = tables_dir / "final_stress_classification_table.csv"
    stress_conformal_table_path = tables_dir / "final_stress_conformal_table.csv"

    clean_table.to_csv(clean_table_path, index=False)
    stress_classification_table.to_csv(stress_classification_table_path, index=False)
    stress_conformal_table.to_csv(stress_conformal_table_path, index=False)

    clean_fig_path = figures_dir / "figure_clean_model_comparison.png"
    stress_fig_path = figures_dir / "figure_stress_classification_performance.png"
    conformal_set_size_fig_path = figures_dir / "figure_conformal_set_size.png"
    conformal_singleton_fig_path = figures_dir / "figure_conformal_singleton_rate.png"

    plot_clean_model_comparison(
        clean_metrics_df=clean_table,
        output_path=clean_fig_path,
        metric_name="macro_f1",
    )

    plot_stress_classification_performance(
        stress_metrics_df=stress_classification_df,
        output_path=stress_fig_path,
        metric_name="macro_f1",
    )

    plot_conformal_metric_under_stress(
        conformal_metrics_df=stress_conformal_df,
        output_path=conformal_set_size_fig_path,
        metric_col="avg_set_size",
        ylabel="average set size",
        title="Conformal set size under degraded input",
    )

    plot_conformal_metric_under_stress(
        conformal_metrics_df=stress_conformal_df,
        output_path=conformal_singleton_fig_path,
        metric_col="singleton_rate",
        ylabel="singleton rate",
        title="Conformal singleton rate under degraded input",
        ylim=(0.0, 1.0),
    )

    print("Saved final clean performance table to:", clean_table_path.resolve())
    print(
        "Saved final stress classification table to:",
        stress_classification_table_path.resolve(),
    )
    print(
        "Saved final stress conformal table to:",
        stress_conformal_table_path.resolve(),
    )

    print("Saved clean comparison figure to:", clean_fig_path.resolve())
    print("Saved stress classification figure to:", stress_fig_path.resolve())
    print("Saved conformal set-size figure to:", conformal_set_size_fig_path.resolve())
    print(
        "Saved conformal singleton-rate figure to:",
        conformal_singleton_fig_path.resolve(),
    )


if __name__ == "__main__":
    main()
