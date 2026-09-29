from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STRESS_TEST_ORDER = [
    "clean",
    "truncate_75",
    "truncate_50",
    "truncate_25",
    "delete_15",
    "delete_30",
]

STRESS_TEST_LABELS = {
    "clean": "clean",
    "truncate_75": "truncate 75%",
    "truncate_50": "truncate 50%",
    "truncate_25": "truncate 25%",
    "delete_15": "delete 15%",
    "delete_30": "delete 30%",
}

MODEL_LABELS = {
    "tfidf_logreg": "TF-IDF + LogReg",
    "tfidf_linear_svm": "TF-IDF + LinearSVM",
    "distilbert": "DistilBERT",
    "distilbert_conformal": "DistilBERT + conformal",
}

PURPLE = "#9671BD"
TEAL = "#77B5B6"
GREY = "#7A7A7A"
LIGHT_GREY = "#D9D9D9"
DARK_GREY = "#4A4A4A"

MODEL_COLORS = {
    "tfidf_logreg": GREY,
    "tfidf_linear_svm": LIGHT_GREY,
    "distilbert": PURPLE,
    "distilbert_conformal": TEAL,
}


def ensure_parent_dir(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def prettify_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def prettify_stress_test_name(stress_test_name: str) -> str:
    return STRESS_TEST_LABELS.get(stress_test_name, stress_test_name)


def sort_by_stress_test_order(df: pd.DataFrame, column: str = "stress_test") -> pd.DataFrame:
    out = df.copy()
    order_map = {name: i for i, name in enumerate(STRESS_TEST_ORDER)}
    out["_stress_order"] = out[column].map(order_map)
    out = out.sort_values("_stress_order").drop(columns="_stress_order").reset_index(drop=True)
    return out


@contextmanager
def figure_style():
    original = plt.rcParams.copy()
    try:
        plt.rcParams.update(
            {
                "figure.figsize": (8.0, 5.2),
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "font.family": "DejaVu Serif",
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "axes.edgecolor": DARK_GREY,
                "axes.linewidth": 1.2,
                "axes.labelcolor": DARK_GREY,
                "axes.titlecolor": DARK_GREY,
                "xtick.color": DARK_GREY,
                "ytick.color": DARK_GREY,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 9,
                "legend.frameon": True,
                "legend.fancybox": False,
                "legend.borderpad": 0.4,
                "legend.labelspacing": 0.35,
                "legend.handlelength": 1.8,
                "legend.handletextpad": 0.6,
                "grid.color": LIGHT_GREY,
                "grid.linestyle": "--",
                "grid.linewidth": 0.8,
                "axes.grid": True,
                "axes.axisbelow": True,
            }
        )
        yield
    finally:
        plt.rcParams.update(original)


def save_figure(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    fig.savefig(
        output_path.with_suffix(".png"),
        bbox_inches="tight",
        facecolor="white",
    )


def apply_common_axis_format(ax) -> None:
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.9)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_GREY)
    ax.spines["bottom"].set_color(DARK_GREY)


def style_legend(ax) -> None:
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor(LIGHT_GREY)
        frame.set_linewidth(0.8)
        frame.set_alpha(0.95)


def add_bar_value_labels(ax, values: list[float]) -> None:
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK_GREY,
        )


def plot_clean_model_comparison(
    clean_metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_name: str = "macro_f1",
) -> None:
    plot_df = clean_metrics_df.copy()
    plot_df["model_label"] = plot_df["model_name"].map(prettify_model_name)
    plot_df["color"] = plot_df["model_name"].map(MODEL_COLORS).fillna(GREY)
    plot_df = plot_df.sort_values(metric_name, ascending=False).reset_index(drop=True)

    with figure_style():
        fig, ax = plt.subplots()

        values = plot_df[metric_name].tolist()
        colors = plot_df["color"].tolist()
        labels = plot_df["model_label"].tolist()

        ax.bar(labels, values, color=colors, edgecolor=DARK_GREY, linewidth=0.8)
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.set_title("Clean test performance by model", pad=10)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=18)
        apply_common_axis_format(ax)
        add_bar_value_labels(ax, values)

        fig.tight_layout()
        save_figure(fig, output_path)
        plt.close(fig)


def plot_stress_classification_performance(
    stress_metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_name: str = "macro_f1",
) -> None:
    plot_df = sort_by_stress_test_order(stress_metrics_df)

    with figure_style():
        fig, ax = plt.subplots()

        for model_name in plot_df["model_name"].unique():
            subset = plot_df[plot_df["model_name"] == model_name].copy()
            subset = sort_by_stress_test_order(subset)

            x = [prettify_stress_test_name(xi) for xi in subset["stress_test"]]
            y = subset[metric_name].tolist()

            ax.plot(
                x,
                y,
                marker="o",
                markersize=7,
                linewidth=2.4,
                label=prettify_model_name(model_name),
                color=MODEL_COLORS.get(model_name, GREY),
            )

        ax.set_ylabel(metric_name.replace("_", " "))
        ax.set_title("Classification performance under degraded input", pad=10)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=22)
        apply_common_axis_format(ax)
        style_legend(ax)

        fig.tight_layout()
        save_figure(fig, output_path)
        plt.close(fig)


def plot_conformal_metric_under_stress(
    conformal_metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_col: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    plot_df = conformal_metrics_df.copy()

    alpha_values = sorted(plot_df["alpha"].unique(), reverse=True)
    alpha_colors = {
        0.10: TEAL,
        0.05: PURPLE,
    }

    with figure_style():
        fig, ax = plt.subplots()

        for alpha_value in alpha_values:
            subset = plot_df[plot_df["alpha"] == alpha_value].copy()
            subset = sort_by_stress_test_order(subset)

            x = [prettify_stress_test_name(xi) for xi in subset["stress_test"]]
            y = subset[metric_col].tolist()

            ax.plot(
                x,
                y,
                marker="o",
                markersize=7,
                linewidth=2.4,
                label=f"alpha={alpha_value:.2f}",
                color=alpha_colors.get(alpha_value, GREY),
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=22)
        apply_common_axis_format(ax)
        style_legend(ax)

        fig.tight_layout()
        save_figure(fig, output_path)
        plt.close(fig)
