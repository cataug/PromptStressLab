#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


PROMPT_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6"]

TRANSITION_ORDER = [
    "P1->P2",
    "P2->P3",
    "P3->P4",
    "P4->P5",
    "P5->P6",
]

ABLATION_ORDER = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
]

DATASET_ORDER = [
    "SciERC",
    "EBM-NLP",
    "SciER",
]

MODEL_ORDER = [
    "qwen3_8b",
    "gemma_3_12b_it",
    "mistral_7b_instruct_v03",
]

MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "gemma_3_12b_it": "Gemma-3-12B",
    "mistral_7b_instruct_v03": "Mistral-7B",
}

MODEL_COLORS = {
    "qwen3_8b": "#67C5B5",
    "gemma_3_12b_it": "#B7A1E5",
    "mistral_7b_instruct_v03": "#F39A7A",
}

MODEL_MARKERS = {
    "qwen3_8b": "o",
    "gemma_3_12b_it": "s",
    "mistral_7b_instruct_v03": "^",
}

BACKGROUND = "#FCFBF8"
PANEL_BACKGROUND = "#F7F6F2"
GRID_COLOR = "#D8D6D0"
TEXT_COLOR = "#202528"
MUTED_COLOR = "#667075"

VOLATILITY_CMAP = LinearSegmentedColormap.from_list(
    "volatility",
    [
        "#F8F6EF",
        "#CBE8DF",
        "#8BCBC0",
        "#5A9D9A",
    ],
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "delta",
    [
        "#ECA8B1",
        "#F8E4E0",
        "#FFF9ED",
        "#D8EEE6",
        "#75BBAA",
    ],
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 23,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "legend.title_fontsize": 16,
            "figure.titlesize": 29,
            "axes.linewidth": 1.4,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.65,
            "grid.linewidth": 1.0,
            "lines.linewidth": 3.4,
            "lines.markersize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required statistics file not found: {path}"
        )

    frame = pd.read_csv(path)

    print(
        f"[LOAD] {path.name}: {len(frame)} rows",
        flush=True,
    )

    return frame


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(
        f"[SAVED] {png_path}",
        flush=True,
    )

    print(
        f"[SAVED] {pdf_path}",
        flush=True,
    )


def panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        -0.12,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="left",
        color=TEXT_COLOR,
    )


def clean_axes(
    ax: plt.Axes,
    grid_axis: str = "y",
) -> None:
    ax.grid(
        True,
        axis=grid_axis,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def model_handles() -> list[Line2D]:
    handles = []

    for model_id in MODEL_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model_id],
                marker=MODEL_MARKERS[model_id],
                linewidth=3.5,
                markersize=10,
                label=MODEL_LABELS[model_id],
            )
        )

    return handles


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    formatter: str = ".3f",
    stars: np.ndarray | None = None,
) -> None:
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        title,
        pad=16,
        fontweight="bold",
    )

    ax.set_xticks(
        np.arange(
            len(column_labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(row_labels)
        )
    )

    ax.set_xticklabels(
        column_labels,
        rotation=0,
        fontweight="bold",
    )

    ax.set_yticklabels(
        row_labels,
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index
            ]

            if not np.isfinite(value):
                text = "—"
            else:
                text = format(
                    value,
                    formatter,
                )

                if (
                    stars is not None
                    and stars[
                        row_index,
                        column_index
                    ]
                ):
                    text += "*"

            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=TEXT_COLOR,
            )

    ax.set_xticks(
        np.arange(
            -0.5,
            len(column_labels),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(row_labels),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color=BACKGROUND,
        linewidth=3,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def non_dominated_frontier(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        [
            x_column,
            y_column,
        ],
        ascending=[
            True,
            False,
        ],
    )

    rows = []
    best_quality = -np.inf

    for _, row in ordered.iterrows():
        quality = float(
            row[y_column]
        )

        if quality > best_quality:
            rows.append(row)
            best_quality = quality

    if not rows:
        return pd.DataFrame(
            columns=frame.columns
        )

    return pd.DataFrame(rows)


def figure_f1_trajectories(
    condition_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = condition_summary[
        condition_summary[
            "condition"
        ].isin(PROMPT_ORDER)
    ].copy()

    if "phase" in frame.columns:
        frame = frame[
            frame["phase"]
            == "main"
        ]

    metrics = [
        (
            "entity_exact_f1_mean",
            "Exact entity F1",
        ),
        (
            "entity_partial_f1_mean",
            "Partial entity F1",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            24,
            13,
        ),
        sharex=True,
    )

    for metric_index, (
        metric_column,
        metric_label,
    ) in enumerate(metrics):
        values = pd.to_numeric(
            frame[metric_column],
            errors="coerce",
        )

        metric_max = max(
            float(
                values.max()
            )
            * 1.14,
            0.12,
        )

        for dataset_index, dataset in enumerate(
            DATASET_ORDER
        ):
            ax = axes[
                metric_index,
                dataset_index
            ]

            subset = frame[
                frame["dataset"]
                == dataset
            ]

            for model_id in MODEL_ORDER:
                model_frame = subset[
                    subset["model_id"]
                    == model_id
                ].copy()

                model_frame[
                    "condition"
                ] = pd.Categorical(
                    model_frame[
                        "condition"
                    ],
                    PROMPT_ORDER,
                    ordered=True,
                )

                model_frame = (
                    model_frame
                    .sort_values(
                        "condition"
                    )
                )

                if model_frame.empty:
                    continue

                ax.plot(
                    model_frame[
                        "condition"
                    ].astype(str),
                    model_frame[
                        metric_column
                    ],
                    color=MODEL_COLORS[
                        model_id
                    ],
                    marker=MODEL_MARKERS[
                        model_id
                    ],
                    markeredgecolor=TEXT_COLOR,
                    markeredgewidth=0.8,
                    label=MODEL_LABELS[
                        model_id
                    ],
                    zorder=3,
                )

            ax.set_ylim(
                0,
                metric_max,
            )

            if metric_index == 0:
                ax.set_title(
                    dataset,
                    fontweight="bold",
                    pad=14,
                )

            if dataset_index == 0:
                ax.set_ylabel(
                    metric_label,
                    fontweight="bold",
                )

            if metric_index == 1:
                ax.set_xlabel(
                    "Prompt level",
                    fontweight="bold",
                )

            clean_axes(ax)

    panel_label(
        axes[0, 0],
        "A",
    )

    panel_label(
        axes[1, 0],
        "B",
    )

    fig.suptitle(
        "Extraction quality across prompt elaboration levels",
        fontweight="bold",
        y=1.01,
    )

    fig.legend(
        handles=model_handles(),
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.035,
        ),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig01_f1_prompt_trajectories",
    )


def figure_psi_trajectories(
    psi_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = psi_summary.copy()

    frame["transition"] = pd.Categorical(
        frame["transition"],
        TRANSITION_ORDER,
        ordered=True,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            24,
            7.5,
        ),
        sharey=True,
    )

    for dataset_index, dataset in enumerate(
        DATASET_ORDER
    ):
        ax = axes[
            dataset_index
        ]

        subset = frame[
            frame["dataset"]
            == dataset
        ]

        for model_id in MODEL_ORDER:
            model_frame = subset[
                subset["model_id"]
                == model_id
            ].sort_values(
                "transition"
            )

            if model_frame.empty:
                continue

            x = np.arange(
                len(model_frame)
            )

            mean = model_frame[
                "mean"
            ].to_numpy(
                dtype=float
            )

            low = model_frame[
                "bootstrap_ci_low"
            ].to_numpy(
                dtype=float
            )

            high = model_frame[
                "bootstrap_ci_high"
            ].to_numpy(
                dtype=float
            )

            ax.plot(
                x,
                mean,
                color=MODEL_COLORS[
                    model_id
                ],
                marker=MODEL_MARKERS[
                    model_id
                ],
                markeredgecolor=TEXT_COLOR,
                markeredgewidth=0.8,
                zorder=3,
            )

            ax.fill_between(
                x,
                low,
                high,
                color=MODEL_COLORS[
                    model_id
                ],
                alpha=0.18,
                zorder=2,
            )

        ax.set_xticks(
            np.arange(
                len(
                    TRANSITION_ORDER
                )
            )
        )

        ax.set_xticklabels(
            TRANSITION_ORDER,
            rotation=27,
            ha="right",
        )

        ax.set_title(
            dataset,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Adjacent prompt transition",
            fontweight="bold",
        )

        if dataset_index == 0:
            ax.set_ylabel(
                "Prompt Sensitivity Index",
                fontweight="bold",
            )

        ax.set_ylim(
            bottom=0,
        )

        clean_axes(ax)

    panel_label(
        axes[0],
        "A",
    )

    fig.suptitle(
        "Semantic drift across adjacent prompt levels",
        fontweight="bold",
        y=1.02,
    )

    fig.legend(
        handles=model_handles(),
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.08,
        ),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig02_psi_prompt_transitions",
    )


def figure_pareto_map(
    condition_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = condition_summary[
        condition_summary[
            "condition"
        ].isin(PROMPT_ORDER)
    ].copy()

    if "phase" in frame.columns:
        frame = frame[
            frame["phase"]
            == "main"
        ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            24,
            8,
        ),
    )

    for dataset_index, dataset in enumerate(
        DATASET_ORDER
    ):
        ax = axes[
            dataset_index
        ]

        subset = frame[
            frame["dataset"]
            == dataset
        ].copy()

        for model_id in MODEL_ORDER:
            model_frame = subset[
                subset["model_id"]
                == model_id
            ].copy()

            model_frame[
                "condition"
            ] = pd.Categorical(
                model_frame[
                    "condition"
                ],
                PROMPT_ORDER,
                ordered=True,
            )

            model_frame = (
                model_frame
                .sort_values(
                    "condition"
                )
            )

            if model_frame.empty:
                continue

            schema = (
                model_frame[
                    "schema_valid_mean"
                ].fillna(0)
                .clip(
                    0,
                    1,
                )
            )

            sizes = (
                170
                + 450 * schema
            )

            ax.plot(
                model_frame[
                    "prompt_tokens_mean"
                ],
                model_frame[
                    "entity_partial_f1_mean"
                ],
                color=MODEL_COLORS[
                    model_id
                ],
                alpha=0.65,
                linewidth=2.7,
                zorder=2,
            )

            ax.scatter(
                model_frame[
                    "prompt_tokens_mean"
                ],
                model_frame[
                    "entity_partial_f1_mean"
                ],
                s=sizes,
                color=MODEL_COLORS[
                    model_id
                ],
                edgecolor=TEXT_COLOR,
                linewidth=1.1,
                marker=MODEL_MARKERS[
                    model_id
                ],
                alpha=0.95,
                zorder=3,
            )

            for _, row in model_frame.iterrows():
                ax.annotate(
                    str(
                        row[
                            "condition"
                        ]
                    ),
                    (
                        row[
                            "prompt_tokens_mean"
                        ],
                        row[
                            "entity_partial_f1_mean"
                        ],
                    ),
                    xytext=(
                        7,
                        7,
                    ),
                    textcoords="offset points",
                    fontsize=13,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )

        frontier = non_dominated_frontier(
            subset.dropna(
                subset=[
                    "prompt_tokens_mean",
                    "entity_partial_f1_mean",
                ]
            ),
            "prompt_tokens_mean",
            "entity_partial_f1_mean",
        )

        if len(frontier) >= 2:
            ax.plot(
                frontier[
                    "prompt_tokens_mean"
                ],
                frontier[
                    "entity_partial_f1_mean"
                ],
                color=TEXT_COLOR,
                linestyle="--",
                linewidth=2.4,
                alpha=0.75,
                zorder=1,
            )

        ax.set_title(
            dataset,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Mean prompt tokens",
            fontweight="bold",
        )

        if dataset_index == 0:
            ax.set_ylabel(
                "Partial entity F1",
                fontweight="bold",
            )

        ax.set_ylim(
            bottom=0,
        )

        clean_axes(
            ax,
            grid_axis="both",
        )

    panel_label(
        axes[0],
        "A",
    )

    fig.suptitle(
        "Prompt cost–quality landscape",
        fontweight="bold",
        y=1.02,
    )

    fig.legend(
        handles=model_handles(),
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.05,
        ),
        ncol=3,
        frameon=False,
    )

    fig.text(
        0.5,
        -0.005,
        "Marker size represents schema validity; dashed line marks the empirical Pareto frontier.",
        ha="center",
        fontsize=15,
        color=MUTED_COLOR,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig03_cost_quality_pareto",
    )


def figure_field_volatility(
    field_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = field_summary.copy()

    global_max = float(
        frame["mean"].max()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            24,
            10,
        ),
    )

    images = []

    for dataset_index, dataset in enumerate(
        DATASET_ORDER
    ):
        ax = axes[
            dataset_index
        ]

        subset = frame[
            frame["dataset"]
            == dataset
        ].copy()

        fields = (
            subset.groupby(
                "field"
            )["mean"]
            .mean()
            .sort_values(
                ascending=False
            )
            .index.tolist()
        )

        matrix = np.full(
            (
                len(fields),
                len(MODEL_ORDER),
            ),
            np.nan,
            dtype=float,
        )

        for row_index, field in enumerate(
            fields
        ):
            for column_index, model_id in enumerate(
                MODEL_ORDER
            ):
                values = subset[
                    (
                        subset["field"]
                        == field
                    )
                    & (
                        subset["model_id"]
                        == model_id
                    )
                ]["mean"]

                if not values.empty:
                    matrix[
                        row_index,
                        column_index
                    ] = float(
                        values.iloc[0]
                    )

        pretty_fields = [
            field.replace(
                "__RELATIONS__",
                "Relations",
            )
            for field in fields
        ]

        image = draw_heatmap(
            ax,
            matrix,
            pretty_fields,
            [
                MODEL_LABELS[
                    model_id
                ]
                for model_id
                in MODEL_ORDER
            ],
            dataset,
            VOLATILITY_CMAP,
            0,
            global_max,
            formatter=".3f",
        )

        images.append(image)

        ax.set_xlabel(
            "Model",
            fontweight="bold",
        )

        if dataset_index == 0:
            ax.set_ylabel(
                "Schema field",
                fontweight="bold",
            )

    panel_label(
        axes[0],
        "A",
    )

    fig.suptitle(
        "Field-wise semantic volatility",
        fontweight="bold",
        y=1.01,
    )

    colorbar = fig.colorbar(
        images[-1],
        ax=axes,
        fraction=0.025,
        pad=0.025,
    )

    colorbar.set_label(
        "Mean semantic drift",
        fontsize=18,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        labelsize=14,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.91,
        top=0.88,
        bottom=0.11,
        wspace=0.36,
    )

    save_figure(
        fig,
        output_dir,
        "fig04_field_volatility_heatmaps",
    )


def binned_median(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    subset = frame[
        [
            x_column,
            y_column,
        ]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if (
        len(subset) < bins
        or subset[
            x_column
        ].nunique()
        < 3
    ):
        return pd.DataFrame(
            columns=[
                x_column,
                y_column,
            ]
        )

    try:
        subset["bin"] = pd.qcut(
            subset[
                x_column
            ],
            q=bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame(
            columns=[
                x_column,
                y_column,
            ]
        )

    return (
        subset.groupby(
            "bin",
            observed=True,
        )
        .agg(
            **{
                x_column: (
                    x_column,
                    "median",
                ),
                y_column: (
                    y_column,
                    "median",
                ),
            }
        )
        .reset_index(
            drop=True
        )
    )


def figure_psi_f1_relationship(
    psi_pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = psi_pairwise.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna(
        subset=[
            "psi",
            "abs_delta_entity_partial_f1",
        ]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            24,
            8,
        ),
        sharey=True,
    )

    for dataset_index, dataset in enumerate(
        DATASET_ORDER
    ):
        ax = axes[
            dataset_index
        ]

        subset = frame[
            frame["dataset"]
            == dataset
        ]

        for model_id in MODEL_ORDER:
            model_frame = subset[
                subset["model_id"]
                == model_id
            ]

            if model_frame.empty:
                continue

            ax.scatter(
                model_frame[
                    "psi"
                ],
                model_frame[
                    "abs_delta_entity_partial_f1"
                ],
                s=28,
                color=MODEL_COLORS[
                    model_id
                ],
                alpha=0.22,
                edgecolors="none",
                rasterized=True,
            )

            trend = binned_median(
                model_frame,
                "psi",
                "abs_delta_entity_partial_f1",
                bins=10,
            )

            if len(trend) >= 2:
                ax.plot(
                    trend["psi"],
                    trend[
                        "abs_delta_entity_partial_f1"
                    ],
                    color=MODEL_COLORS[
                        model_id
                    ],
                    marker=MODEL_MARKERS[
                        model_id
                    ],
                    markeredgecolor=TEXT_COLOR,
                    markeredgewidth=0.6,
                    linewidth=3.2,
                    zorder=4,
                )

            correlation = spearmanr(
                model_frame[
                    "psi"
                ],
                model_frame[
                    "abs_delta_entity_partial_f1"
                ],
            )

            ax.text(
                0.03,
                0.96
                - 0.085
                * MODEL_ORDER.index(
                    model_id
                ),
                (
                    f"{MODEL_LABELS[model_id]}: "
                    f"ρ={correlation.statistic:.2f}"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=14,
                fontweight="bold",
                color=MODEL_COLORS[
                    model_id
                ],
            )

        ax.set_title(
            dataset,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Prompt Sensitivity Index",
            fontweight="bold",
        )

        if dataset_index == 0:
            ax.set_ylabel(
                r"$|\Delta|$ partial entity F1",
                fontweight="bold",
            )

        ax.set_xlim(
            left=0,
        )

        ax.set_ylim(
            bottom=0,
        )

        clean_axes(
            ax,
            grid_axis="both",
        )

    panel_label(
        axes[0],
        "A",
    )

    fig.suptitle(
        "Semantic drift tracks changes in extraction quality",
        fontweight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        -0.01,
        "Points show document-level transitions; solid paths show decile medians.",
        ha="center",
        fontsize=15,
        color=MUTED_COLOR,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig05_psi_f1_relationship",
    )


def figure_ablation_heatmap(
    paired_tests: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = paired_tests[
        (
            paired_tests["family"]
            == "ablation_vs_A0"
        )
        & (
            paired_tests["metric"]
            == "entity_partial_f1"
        )
        & (
            paired_tests[
                "right_condition"
            ].isin(
                ABLATION_ORDER
            )
        )
    ].copy()

    if frame.empty:
        print(
            "[SKIP] No partial-F1 ablation rows",
            flush=True,
        )

        return

    limit = float(
        np.nanmax(
            np.abs(
                frame[
                    "mean_delta_right_minus_left"
                ]
            )
        )
    )

    limit = max(
        limit,
        0.01,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            24,
            7.5,
        ),
    )

    images = []

    for dataset_index, dataset in enumerate(
        DATASET_ORDER
    ):
        ax = axes[
            dataset_index
        ]

        subset = frame[
            frame["dataset"]
            == dataset
        ]

        matrix = np.full(
            (
                len(MODEL_ORDER),
                len(ABLATION_ORDER),
            ),
            np.nan,
            dtype=float,
        )

        stars = np.zeros(
            matrix.shape,
            dtype=bool,
        )

        for row_index, model_id in enumerate(
            MODEL_ORDER
        ):
            for column_index, condition in enumerate(
                ABLATION_ORDER
            ):
                row = subset[
                    (
                        subset["model_id"]
                        == model_id
                    )
                    & (
                        subset[
                            "right_condition"
                        ]
                        == condition
                    )
                ]

                if row.empty:
                    continue

                matrix[
                    row_index,
                    column_index
                ] = float(
                    row[
                        "mean_delta_right_minus_left"
                    ].iloc[0]
                )

                p_holm = row[
                    "p_holm"
                ].iloc[0]

                stars[
                    row_index,
                    column_index
                ] = (
                    pd.notna(
                        p_holm
                    )
                    and float(
                        p_holm
                    )
                    < 0.05
                )

        image = draw_heatmap(
            ax,
            matrix,
            [
                MODEL_LABELS[
                    model_id
                ]
                for model_id
                in MODEL_ORDER
            ],
            ABLATION_ORDER,
            dataset,
            DIVERGING_CMAP,
            -limit,
            limit,
            formatter="+.3f",
            stars=stars,
        )

        images.append(image)

        ax.set_xlabel(
            "Ablation relative to A0",
            fontweight="bold",
        )

        if dataset_index == 0:
            ax.set_ylabel(
                "Model",
                fontweight="bold",
            )

    panel_label(
        axes[0],
        "A",
    )

    fig.suptitle(
        "Contribution of prompt components",
        fontweight="bold",
        y=1.02,
    )

    colorbar = fig.colorbar(
        images[-1],
        ax=axes,
        fraction=0.025,
        pad=0.025,
    )

    colorbar.set_label(
        "Change in partial entity F1",
        fontsize=18,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        labelsize=14,
    )

    fig.text(
        0.5,
        0.01,
        "* Holm-corrected p < 0.05",
        ha="center",
        fontsize=15,
        color=MUTED_COLOR,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        top=0.84,
        bottom=0.18,
        wspace=0.30,
    )

    save_figure(
        fig,
        output_dir,
        "fig06_ablation_effects",
    )


def figure_speed_validity(
    progress: pd.DataFrame,
    output_dir: Path,
) -> None:
    frame = progress.copy()

    frame["schema_percent"] = (
        frame["schema_valid_rate"]
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            9,
        )
    )

    median_speed = float(
        frame[
            "mean_seconds_per_item"
        ].median()
    )

    median_validity = float(
        frame[
            "schema_percent"
        ].median()
    )

    ax.axvline(
        median_speed,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=2,
        zorder=1,
    )

    ax.axhline(
        median_validity,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=2,
        zorder=1,
    )

    for _, row in frame.iterrows():
        model_id = str(
            row["model_id"]
        )

        ax.scatter(
            row[
                "mean_seconds_per_item"
            ],
            row[
                "schema_percent"
            ],
            s=900,
            color=MODEL_COLORS[
                model_id
            ],
            marker=MODEL_MARKERS[
                model_id
            ],
            edgecolor=TEXT_COLOR,
            linewidth=1.5,
            zorder=3,
        )

        ax.annotate(
            MODEL_LABELS[
                model_id
            ],
            (
                row[
                    "mean_seconds_per_item"
                ],
                row[
                    "schema_percent"
                ],
            ),
            xytext=(
                14,
                12,
            ),
            textcoords="offset points",
            fontsize=17,
            fontweight="bold",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Mean generation time per item, seconds (log scale)",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Schema-valid outputs, %",
        fontweight="bold",
    )

    ax.set_title(
        "Model efficiency and structural reliability",
        fontweight="bold",
        pad=18,
    )

    ax.set_ylim(
        min(
            frame[
                "schema_percent"
            ].min()
            - 3,
            80,
        ),
        100,
    )

    clean_axes(
        ax,
        grid_axis="both",
    )

    panel_label(
        ax,
        "A",
    )

    save_figure(
        fig,
        output_dir,
        "fig07_speed_schema_validity",
    )


def figure_overview(
    condition_summary: pd.DataFrame,
    psi_overall: pd.DataFrame,
    field_summary: pd.DataFrame,
    progress: pd.DataFrame,
    output_dir: Path,
) -> None:
    main = condition_summary[
        condition_summary[
            "condition"
        ].isin(
            PROMPT_ORDER
        )
    ].copy()

    if "phase" in main.columns:
        main = main[
            main["phase"]
            == "main"
        ]

    aggregated_f1 = (
        main.groupby(
            [
                "model_id",
                "condition",
            ],
            as_index=False,
        )[
            "entity_partial_f1_mean"
        ]
        .mean()
    )

    aggregated_f1[
        "condition"
    ] = pd.Categorical(
        aggregated_f1[
            "condition"
        ],
        PROMPT_ORDER,
        ordered=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            20,
            16,
        ),
    )

    ax = axes[
        0,
        0
    ]

    for model_id in MODEL_ORDER:
        subset = aggregated_f1[
            aggregated_f1[
                "model_id"
            ]
            == model_id
        ].sort_values(
            "condition"
        )

        ax.plot(
            subset[
                "condition"
            ].astype(str),
            subset[
                "entity_partial_f1_mean"
            ],
            color=MODEL_COLORS[
                model_id
            ],
            marker=MODEL_MARKERS[
                model_id
            ],
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=0.8,
        )

    ax.set_title(
        "Mean partial F1 across datasets",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Prompt level",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Partial entity F1",
        fontweight="bold",
    )

    ax.set_ylim(
        bottom=0,
    )

    clean_axes(ax)
    panel_label(
        ax,
        "A",
    )

    ax = axes[
        0,
        1
    ]

    x = np.arange(
        len(
            DATASET_ORDER
        )
    )

    width = 0.24

    for model_index, model_id in enumerate(
        MODEL_ORDER
    ):
        values = []

        for dataset in DATASET_ORDER:
            row = psi_overall[
                (
                    psi_overall[
                        "model_id"
                    ]
                    == model_id
                )
                & (
                    psi_overall[
                        "dataset"
                    ]
                    == dataset
                )
            ]

            values.append(
                float(
                    row["mean"].iloc[0]
                )
                if not row.empty
                else np.nan
            )

        ax.bar(
            x
            + (
                model_index
                - 1
            )
            * width,
            values,
            width=width,
            color=MODEL_COLORS[
                model_id
            ],
            edgecolor=TEXT_COLOR,
            linewidth=0.8,
            label=MODEL_LABELS[
                model_id
            ],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        DATASET_ORDER
    )

    ax.set_title(
        "Mean Prompt Sensitivity Index",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Dataset",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Mean PSI",
        fontweight="bold",
    )

    ax.set_ylim(
        bottom=0,
    )

    clean_axes(ax)
    panel_label(
        ax,
        "B",
    )

    ax = axes[
        1,
        0
    ]

    progress_frame = (
        progress.copy()
    )

    progress_frame[
        "schema_percent"
    ] = (
        progress_frame[
            "schema_valid_rate"
        ]
        * 100
    )

    for _, row in progress_frame.iterrows():
        model_id = str(
            row["model_id"]
        )

        ax.scatter(
            row[
                "mean_seconds_per_item"
            ],
            row[
                "schema_percent"
            ],
            s=700,
            color=MODEL_COLORS[
                model_id
            ],
            marker=MODEL_MARKERS[
                model_id
            ],
            edgecolor=TEXT_COLOR,
            linewidth=1.2,
        )

        ax.annotate(
            MODEL_LABELS[
                model_id
            ],
            (
                row[
                    "mean_seconds_per_item"
                ],
                row[
                    "schema_percent"
                ],
            ),
            xytext=(
                11,
                9,
            ),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_title(
        "Speed versus schema validity",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Seconds per item (log scale)",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Schema validity, %",
        fontweight="bold",
    )

    ax.set_ylim(
        80,
        100,
    )

    clean_axes(
        ax,
        grid_axis="both",
    )

    panel_label(
        ax,
        "C",
    )

    ax = axes[
        1,
        1
    ]

    top_fields = (
        field_summary.sort_values(
            "mean",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    top_fields[
        "label"
    ] = (
        top_fields[
            "model_id"
        ].map(
            MODEL_LABELS
        )
        + " · "
        + top_fields[
            "dataset"
        ].astype(str)
        + " · "
        + top_fields[
            "field"
        ].astype(str)
        .str.replace(
            "__RELATIONS__",
            "Relations",
            regex=False,
        )
    )

    top_fields = top_fields.sort_values(
        "mean",
        ascending=True,
    )

    bar_colors = [
        MODEL_COLORS[
            model_id
        ]
        for model_id
        in top_fields[
            "model_id"
        ]
    ]

    ax.barh(
        top_fields[
            "label"
        ],
        top_fields[
            "mean"
        ],
        color=bar_colors,
        edgecolor=TEXT_COLOR,
        linewidth=0.7,
        zorder=3,
    )

    for index, value in enumerate(
        top_fields[
            "mean"
        ]
    ):
        ax.text(
            value + 0.005,
            index,
            f"{value:.3f}",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title(
        "Most volatile model–field pairs",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Mean semantic drift",
        fontweight="bold",
    )

    clean_axes(ax)
    panel_label(
        ax,
        "D",
    )

    fig.suptitle(
        "Prompt verbosity and semantic stability",
        fontweight="bold",
        y=0.995,
    )

    fig.legend(
        handles=model_handles(),
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.015,
        ),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(
        rect=[
            0,
            0.035,
            1,
            0.97,
        ]
    )

    save_figure(
        fig,
        output_dir,
        "fig08_overview_poster",
    )


def write_manifest(
    output_dir: Path,
) -> None:
    lines = [
        "PromptStressLab final figure set",
        "",
        "fig01_f1_prompt_trajectories",
        "  Exact and partial entity F1 across P1–P6.",
        "",
        "fig02_psi_prompt_transitions",
        "  PSI across adjacent prompt transitions with bootstrap confidence intervals.",
        "",
        "fig03_cost_quality_pareto",
        "  Prompt-token cost versus partial F1 with schema-validity marker sizes.",
        "",
        "fig04_field_volatility_heatmaps",
        "  Field-wise semantic volatility across models and datasets.",
        "",
        "fig05_psi_f1_relationship",
        "  Relationship between PSI and absolute partial-F1 changes.",
        "",
        "fig06_ablation_effects",
        "  Partial-F1 changes for A1–A6 relative to A0.",
        "",
        "fig07_speed_schema_validity",
        "  Model latency versus schema validity.",
        "",
        "fig08_overview_poster",
        "  Four-panel summary figure.",
        "",
        "Every figure is provided as 300-dpi PNG and vector PDF.",
    ]

    path = (
        output_dir
        / "figure_manifest.txt"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[SAVED] {path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            "/home/tahiti/"
            "PromptStressLab"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    statistics_dir = (
        root
        / "outputs"
        / "statistics"
    )

    output_dir = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else (
            root
            / "outputs"
            / "figures_final"
        )
    )

    configure_style()

    condition_summary = load_csv(
        statistics_dir
        / "condition_summary.csv"
    )

    psi_summary = load_csv(
        statistics_dir
        / "psi_summary.csv"
    )

    psi_overall = load_csv(
        statistics_dir
        / "psi_summary_overall.csv"
    )

    field_summary = load_csv(
        statistics_dir
        / "field_volatility_summary_overall.csv"
    )

    psi_pairwise = load_csv(
        statistics_dir
        / "psi_pairwise.csv"
    )

    paired_tests = load_csv(
        statistics_dir
        / "paired_wilcoxon_tests.csv"
    )

    progress = load_csv(
        statistics_dir
        / "progress_by_model.csv"
    )

    figure_f1_trajectories(
        condition_summary,
        output_dir,
    )

    figure_psi_trajectories(
        psi_summary,
        output_dir,
    )

    figure_pareto_map(
        condition_summary,
        output_dir,
    )

    figure_field_volatility(
        field_summary,
        output_dir,
    )

    figure_psi_f1_relationship(
        psi_pairwise,
        output_dir,
    )

    figure_ablation_heatmap(
        paired_tests,
        output_dir,
    )

    figure_speed_validity(
        progress,
        output_dir,
    )

    figure_overview(
        condition_summary,
        psi_overall,
        field_summary,
        progress,
        output_dir,
    )

    write_manifest(
        output_dir
    )

    print()
    print(
        "=== ALL FIGURES COMPLETE ==="
    )

    print(
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
