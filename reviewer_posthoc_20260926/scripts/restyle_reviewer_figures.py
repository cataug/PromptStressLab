from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba


ROOT = Path("/home/tahiti/PromptStressLab")
SRC = ROOT / "outputs" / "reviewer_posthoc"
OUT = ROOT / "FIGURES_NEW"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 23,

    "axes.linewidth": 1.25,
    "axes.edgecolor": "black",

    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,

    "axes.grid": False,

    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ENC_ORDER = [
    "clip_vit_b32",
    "mpnet",
    "specter",
    "set_jaccard",
    "token_jaccard",
]

SEM_ENC = [
    "clip_vit_b32",
    "mpnet",
    "specter",
]

ENC_LABEL = {
    "clip_vit_b32": "CLIP",
    "mpnet": "MPNet",
    "specter": "SPECTER",
    "set_jaccard": "Set Jaccard",
    "token_jaccard": "Token Jaccard",
}

MODEL_ORDER = [
    "Gemma-3-12B-IT",
    "Mistral-7B-Instruct-v0.3",
    "Qwen3-8B",
]

MODEL_SHORT = {
    "Gemma-3-12B-IT": "Gemma",
    "Mistral-7B-Instruct-v0.3": "Mistral",
    "Qwen3-8B": "Qwen",
}

DATASET_ORDER = [
    "EBM-NLP",
    "SciERC",
    "SciER",
]


def read(name):
    return pd.read_csv(SRC / name)


def read_optional(name):
    p = SRC / name
    return pd.read_csv(p) if p.exists() else None


# ============================================================
# Visual helpers
# ============================================================

def figure_gradient(fig, cmap="Greys", alpha=0.13):
    """
    Very light whole-figure diagonal gradient.
    """
    axbg = fig.add_axes([0, 0, 1, 1], zorder=-100)
    n = 512
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    g = (np.outer(y, np.ones(n)) + np.outer(np.ones(n), x)) / 2
    axbg.imshow(
        g,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        aspect="auto",
    )
    axbg.axis("off")


def panel_background(ax, alpha=0.68):
    """
    Soft translucent plotting panel with black border.
    """
    ax.patch.set_alpha(alpha)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("black")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.18,
        zorder=0,
    )


def gradient_bar(ax, x, heights, width=0.72, cmap="viridis",
                 edgecolor="black", linewidth=1.15,
                 alpha_low=0.45, alpha_high=0.90,
                 zorder=3, baseline=0.0):
    """
    Bars filled by vertical gradient, outlined in black.
    """
    heights = np.asarray(heights, dtype=float)
    finite = heights[np.isfinite(heights)]

    if len(finite):
        lo = np.nanmin(finite)
        hi = np.nanmax(finite)
    else:
        lo, hi = 0.0, 1.0

    cm = plt.get_cmap(cmap)

    patches = []

    for xi, h in zip(x, heights):
        if not np.isfinite(h):
            continue

        rect = Rectangle(
            (xi - width / 2, baseline),
            width,
            h - baseline,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder + 2,
        )
        ax.add_patch(rect)
        patches.append(rect)

        steps = 64
        vals = np.linspace(0, 1, steps).reshape(-1, 1)

        frac = (h - lo) / (hi - lo + 1e-12)
        base = 0.25 + 0.65 * frac

        rgba = cm(np.clip(vals * 0.45 + base * 0.55, 0, 1))
        rgba[:, :, 3] = np.linspace(alpha_low, alpha_high, steps).reshape(-1, 1)

        im = ax.imshow(
            rgba,
            extent=[
                xi - width/2,
                xi + width/2,
                min(baseline, h),
                max(baseline, h),
            ],
            origin="lower",
            aspect="auto",
            zorder=zorder,
        )

        im.set_clip_path(rect)

    return patches


def gradient_grouped_bars(
    ax,
    centers,
    value_matrix,
    labels,
    width=0.22,
    cmaps=("viridis", "plasma", "cividis"),
    baseline=0.0,
):
    """
    value_matrix shape = [n_series, n_groups]
    """
    n = len(labels)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    for j, label in enumerate(labels):
        vals = np.asarray(value_matrix[j], dtype=float)
        xs = centers + offsets[j]

        gradient_bar(
            ax,
            xs,
            vals,
            width=width * 0.90,
            cmap=cmaps[j % len(cmaps)],
            linewidth=1.1,
            baseline=baseline,
        )

        # legend proxy
        ax.bar(
            [],
            [],
            label=label,
            facecolor=plt.get_cmap(cmaps[j % len(cmaps)])(0.65),
            edgecolor="black",
            linewidth=1.0,
            alpha=0.72,
        )


def annotate_gradient_bars(ax, fmt=".3f", fs=10.5, pad=0.012):
    """
    Put a compact translucent bbox on every gradient Rectangle bar.
    Handles positive and negative bars and non-zero baselines.
    """
    ymin, ymax = ax.get_ylim()
    yrng = ymax - ymin

    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue

        w = patch.get_width()
        h = patch.get_height()

        # Skip zero-width legend proxies / irrelevant rectangles
        if not np.isfinite(w) or not np.isfinite(h):
            continue
        if abs(w) < 1e-10 or abs(h) < 1e-10:
            continue
        if w > 2.0:
            continue

        x = patch.get_x() + w / 2
        y0 = patch.get_y()
        y1 = y0 + h

        # Actual bar endpoint/value is y1.
        value = y1

        if h >= 0:
            yy = y1 + pad * yrng
            va = "bottom"
        else:
            yy = y1 - pad * yrng
            va = "top"

        ax.text(
            x,
            yy,
            format(value, fmt),
            ha="center",
            va=va,
            fontsize=fs,
            weight="bold",
            zorder=50,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.20,rounding_size=0.12",
                facecolor=(1, 1, 1, 0.78),
                edgecolor="black",
                linewidth=0.72,
            ),
        )


def value_bbox(ax, x, y, text, fs=12, dy=0.012):
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin

    ax.text(
        x,
        y + dy * rng,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        zorder=20,
        bbox=dict(
            boxstyle="round,pad=0.22,rounding_size=0.18",
            facecolor=(1, 1, 1, 0.68),
            edgecolor="black",
            linewidth=0.75,
        ),
    )


def title_box(ax, text, subtitle=None):
    title = text
    if subtitle:
        title += "\n" + subtitle

    ax.set_title(
        title,
        pad=16,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.38,rounding_size=0.16",
            facecolor=(1, 1, 1, 0.48),
            edgecolor="black",
            linewidth=0.9,
        ),
    )


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.pdf",
        transparent=False,
    )
    fig.savefig(
        OUT / f"{name}.png",
        transparent=False,
    )
    plt.close(fig)


def heatmap_cell_borders(ax, nrows, ncols):
    for i in range(nrows):
        for j in range(ncols):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.65,
                    zorder=5,
                )
            )


# ============================================================
# Load
# ============================================================

encoder_agreement = read("encoder_agreement.csv")
encoder_transition = read_optional("encoder_transition_rank_agreement.csv")
encoder_field = read_optional("encoder_field_rank_agreement.csv")

psi_f1 = read("psi_f1_correlations_multi_encoder.csv")
practical = read("psi_practical_utility.csv")

posix_summary = read("posix_summary.csv")
posix_all = read("posix_vs_semantic_and_prosa.csv")
posix_adj_psi = read("posix_adjacent_vs_semantic_psi.csv")
posix_adj_f1 = read("posix_adjacent_vs_f1_change.csv")
posix_adj = read("posix_adjacent_transitions.csv")

sens = read("sensitivity_adjacent_multi_encoder.csv")


# ============================================================
# Prep
# ============================================================

overall_psi = psi_f1[
    (psi_f1["level"] == "overall") &
    (psi_f1["target"] == "abs_delta_partial_f1")
].copy()

overall_psi["order"] = overall_psi["encoder"].map(
    {e:i for i,e in enumerate(ENC_ORDER)}
)
overall_psi = overall_psi.sort_values("order")


model_psi = psi_f1[
    (psi_f1["level"] == "model") &
    (psi_f1["target"] == "abs_delta_partial_f1") &
    (psi_f1["encoder"].isin(SEM_ENC))
].copy()


practical_20 = practical[
    (practical["threshold_abs_delta_f1"] == 0.20) &
    (practical["encoder"].isin(SEM_ENC))
].copy()


adj_overall = posix_adj_psi[
    posix_adj_psi["level"] == "overall"
].copy()

adj_overall["order"] = adj_overall["encoder"].map(
    {e:i for i,e in enumerate(ENC_ORDER)}
)
adj_overall = adj_overall.sort_values("order")


posix_f1_overall = posix_adj_f1[
    (posix_adj_f1["level"] == "overall") &
    (posix_adj_f1["target"] == "abs_delta_partial_f1")
]["spearman_rho"].iloc[0]


# ============================================================
# 01 Encoder agreement heatmap
# ============================================================

encs = SEM_ENC

mat = pd.DataFrame(
    np.eye(len(encs)),
    index=encs,
    columns=encs,
)

for _, r in encoder_agreement.iterrows():
    a = r["encoder_a"]
    b = r["encoder_b"]
    if a in encs and b in encs:
        mat.loc[a, b] = r["spearman_rho"]
        mat.loc[b, a] = r["spearman_rho"]


fig, ax = plt.subplots(figsize=(8.6, 7.6))
figure_gradient(fig, "Greys", 0.18)
panel_background(ax, 0.60)

im = ax.imshow(
    mat.values,
    vmin=0.90,
    vmax=1.00,
    cmap="viridis",
    alpha=0.82,
    aspect="equal",
)

heatmap_cell_borders(ax, len(encs), len(encs))

ax.set_xticks(range(len(encs)))
ax.set_yticks(range(len(encs)))

ax.set_xticklabels(
    [ENC_LABEL[e] for e in encs]
)
ax.set_yticklabels(
    [ENC_LABEL[e] for e in encs]
)

for i in range(len(encs)):
    for j in range(len(encs)):
        ax.text(
            j,
            i,
            f"{mat.iloc[i,j]:.3f}",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=(1,1,1,0.60),
                edgecolor="black",
                linewidth=0.7,
            ),
        )

cb = fig.colorbar(im, ax=ax)
cb.outline.set_edgecolor("black")
cb.outline.set_linewidth(1.0)
cb.set_label("Spearman rho")

title_box(
    ax,
    "PSI Is Highly Stable Across Semantic Encoders",
    "4,605 adjacent prompt transitions",
)

save(fig, "01_encoder_agreement_heatmap")


# ============================================================
# 02 Sensitivity vs |delta F1|
# ============================================================

fig, ax = plt.subplots(figsize=(11.2, 7.8))
figure_gradient(fig, "Greys", 0.13)
panel_background(ax, 0.64)

x = np.arange(len(overall_psi))
vals = overall_psi["spearman_rho"].to_numpy()

gradient_bar(
    ax,
    x,
    vals,
    width=0.68,
    cmap="viridis",
)

ax.set_xlim(-0.7, len(x)-0.3)
ax.set_ylim(0, 0.78)

ax.set_xticks(x)
ax.set_xticklabels(
    [ENC_LABEL[e] for e in overall_psi["encoder"]],
    rotation=16,
    ha="right",
)

ax.set_ylabel("Spearman rho")

for xi, yi in zip(x, vals):
    value_bbox(ax, xi, yi, f"{yi:.3f}")

title_box(
    ax,
    "Sensitivity Measures Track Extraction Change",
    "Association with |Δ partial F1| across 4,605 transitions",
)

save(fig, "02_sensitivity_vs_abs_delta_partial_f1")


# ============================================================
# 03 Semantic PSI by model
# ============================================================

fig, ax = plt.subplots(figsize=(12.3, 8.0))
figure_gradient(fig, "Greys", 0.13)
panel_background(ax, 0.64)

centers = np.arange(len(MODEL_ORDER))

matrix = []

for enc in SEM_ENC:
    vals = []
    for model in MODEL_ORDER:
        row = model_psi[
            (model_psi["encoder"] == enc) &
            (model_psi["model_id"] == model)
        ]
        vals.append(row["spearman_rho"].iloc[0])
    matrix.append(vals)

gradient_grouped_bars(
    ax,
    centers,
    matrix,
    [ENC_LABEL[e] for e in SEM_ENC],
    width=0.24,
)

ax.set_xlim(-0.6, len(centers)-0.4)
ax.set_ylim(0, 0.83)

ax.set_xticks(centers)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("Spearman rho")

annotate_gradient_bars(ax, fmt=".3f", fs=10.5, pad=0.010)

leg = ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.72,
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)

title_box(
    ax,
    "Semantic PSI–Quality Association Persists Across Models",
    "PSI vs |Δ partial F1|",
)

save(fig, "03_semantic_psi_vs_abs_delta_f1_by_model")


# ============================================================
# 04 Practical utility
# ============================================================

fig, ax = plt.subplots(figsize=(12.3, 8.0))
figure_gradient(fig, "Greys", 0.13)
panel_background(ax, 0.64)

centers = np.arange(len(MODEL_ORDER))

matrix = []

for enc in SEM_ENC:
    vals = []
    for model in MODEL_ORDER:
        row = practical_20[
            (practical_20["encoder"] == enc) &
            (practical_20["model_id"] == model)
        ]
        vals.append(row["auroc"].iloc[0])
    matrix.append(vals)

gradient_grouped_bars(
    ax,
    centers,
    matrix,
    [ENC_LABEL[e] for e in SEM_ENC],
    width=0.24,
    cmaps=("viridis", "plasma", "cividis"),
    baseline=0.5,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.4,
    color="black",
    alpha=0.72,
)

ax.text(
    len(centers)-0.45,
    0.512,
    "random",
    ha="right",
    va="bottom",
    fontsize=12,
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor=(1,1,1,0.62),
        edgecolor="black",
        linewidth=0.65,
    ),
)

ax.set_xlim(-0.6, len(centers)-0.4)
ax.set_ylim(0.45, 0.93)

ax.set_xticks(centers)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("AUROC")

annotate_gradient_bars(ax, fmt=".3f", fs=10.5, pad=0.010)

leg = ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.72,
)
leg.get_frame().set_edgecolor("black")

title_box(
    ax,
    "PSI as an Operational Risk Signal",
    "Detecting prompt transitions with |Δ partial F1| ≥ 0.20",
)

save(fig, "04_psi_practical_utility_auroc")


# ============================================================
# 05 Adjacent POSIX vs PSI
# ============================================================

fig, ax = plt.subplots(figsize=(11.2, 7.8))
figure_gradient(fig, "Greys", 0.13)
panel_background(ax, 0.64)

x = np.arange(len(adj_overall))
vals = adj_overall["spearman_rho"].to_numpy()

gradient_bar(
    ax,
    x,
    vals,
    width=0.68,
    cmap="plasma",
)

ax.set_xlim(-0.7, len(x)-0.3)
ax.set_ylim(0, 0.64)

ax.set_xticks(x)
ax.set_xticklabels(
    [ENC_LABEL[e] for e in adj_overall["encoder"]],
    rotation=16,
    ha="right",
)

ax.set_ylabel("Spearman rho")

for xi, yi in zip(x, vals):
    value_bbox(ax, xi, yi, f"{yi:.3f}")

title_box(
    ax,
    "Likelihood Sensitivity and Output Drift Are Related",
    "Adjacent POSIX vs PSI on matched Pk→Pk+1 transitions",
)

save(fig, "05_adjacent_posix_vs_psi")


# ============================================================
# 06 Semantic PSI vs POSIX quality association
# ============================================================

rows = []

semantic_overall = overall_psi[
    overall_psi["encoder"].isin(SEM_ENC)
]

for _, r in semantic_overall.iterrows():
    rows.append({
        "measure": f"{ENC_LABEL[r['encoder']]} PSI",
        "rho": r["spearman_rho"],
    })

rows.append({
    "measure": "Adjacent POSIX",
    "rho": posix_f1_overall,
})

cmp = pd.DataFrame(rows).sort_values("rho")

fig, ax = plt.subplots(figsize=(10.6, 7.4))
figure_gradient(fig, "Greys", 0.16)
panel_background(ax, 0.60)

y = np.arange(len(cmp))

cmap = plt.get_cmap("viridis")

for idx, (_, r) in enumerate(cmp.iterrows()):
    val = r["rho"]

    ax.plot(
        [0, val],
        [idx, idx],
        linewidth=6,
        alpha=0.28,
        solid_capstyle="round",
        color=cmap(0.30 + 0.55 * idx / max(len(cmp)-1, 1)),
        zorder=2,
    )

    ax.scatter(
        val,
        idx,
        s=180,
        facecolor=cmap(0.30 + 0.55 * idx / max(len(cmp)-1, 1)),
        edgecolor="black",
        linewidth=1.3,
        alpha=0.88,
        zorder=4,
    )

    ax.text(
        val + 0.014,
        idx,
        f"{val:.3f}",
        va="center",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor=(1,1,1,0.64),
            edgecolor="black",
            linewidth=0.65,
        ),
    )

ax.set_yticks(y)
ax.set_yticklabels(cmp["measure"])

ax.set_xlim(0, 0.76)

ax.set_xlabel(
    "Spearman rho with |Δ partial F1|"
)

title_box(
    ax,
    "Realized Semantic Drift Tracks Extraction Change",
    "Comparison with adjacent likelihood sensitivity",
)

save(fig, "06_psi_vs_posix_quality_association")


# ============================================================
# 07 POSIX by model and dataset
# ============================================================

fig, ax = plt.subplots(figsize=(12.3, 8.0))
figure_gradient(fig, "Greys", 0.14)
panel_background(ax, 0.64)

centers = np.arange(len(DATASET_ORDER))

matrix = []

for model in MODEL_ORDER:
    vals = []
    for dataset in DATASET_ORDER:
        row = posix_summary[
            (posix_summary["model_id"] == model) &
            (posix_summary["dataset"] == dataset)
        ]
        vals.append(row["mean_posix"].iloc[0])
    matrix.append(vals)

gradient_grouped_bars(
    ax,
    centers,
    matrix,
    [MODEL_SHORT[m] for m in MODEL_ORDER],
    width=0.24,
)

ax.set_xlim(-0.6, len(centers)-0.4)

ax.set_xticks(centers)
ax.set_xticklabels(DATASET_ORDER)

ax.set_ylabel("Mean canonical POSIX")

# Headroom for numeric bboxes.
_flat_posix = np.asarray(matrix, dtype=float).ravel()
ax.set_ylim(0, np.nanmax(_flat_posix) * 1.24)

annotate_gradient_bars(ax, fmt=".3f", fs=10.5, pad=0.010)

leg = ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.72,
)
leg.get_frame().set_edgecolor("black")

title_box(
    ax,
    "Prompt-Likelihood Sensitivity Is Strongly Task Dependent",
    "Canonical all-pairs POSIX",
)

save(fig, "07_posix_by_model_dataset")


# ============================================================
# 08 Model × encoder heatmap
# ============================================================

tmp = posix_adj_psi[
    (posix_adj_psi["level"] == "model") &
    (posix_adj_psi["encoder"].isin(SEM_ENC))
]

matrix = pd.DataFrame(
    index=MODEL_ORDER,
    columns=SEM_ENC,
    dtype=float,
)

for _, r in tmp.iterrows():
    matrix.loc[
        r["model_id"],
        r["encoder"],
    ] = r["spearman_rho"]


fig, ax = plt.subplots(figsize=(9.4, 7.4))
figure_gradient(fig, "Greys", 0.16)
panel_background(ax, 0.58)

im = ax.imshow(
    matrix.values,
    vmin=0.35,
    vmax=0.62,
    cmap="plasma",
    alpha=0.84,
    aspect="auto",
)

heatmap_cell_borders(
    ax,
    len(MODEL_ORDER),
    len(SEM_ENC),
)

ax.set_xticks(range(len(SEM_ENC)))
ax.set_xticklabels(
    [ENC_LABEL[e] for e in SEM_ENC]
)

ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

for i in range(len(MODEL_ORDER)):
    for j in range(len(SEM_ENC)):
        val = matrix.iloc[i,j]

        ax.text(
            j,
            i,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=17,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=(1,1,1,0.58),
                edgecolor="black",
                linewidth=0.7,
            ),
        )

cb = fig.colorbar(im, ax=ax)
cb.outline.set_edgecolor("black")
cb.set_label("Spearman rho")

title_box(
    ax,
    "Adjacent POSIX–PSI Association",
    "Consistent across models and semantic encoders",
)

save(fig, "08_adjacent_posix_psi_model_heatmap")


# ============================================================
# 09 All-pairs POSIX vs semantic PSI
# ============================================================

semantic_all = posix_all[
    (posix_all["comparison"] == "POSIX_vs_semantic_psi") &
    (posix_all["encoder"].isin(SEM_ENC))
]

fig, ax = plt.subplots(figsize=(12.3, 8.0))
figure_gradient(fig, "Greys", 0.14)
panel_background(ax, 0.64)

centers = np.arange(len(MODEL_ORDER))

matrix = []

for enc in SEM_ENC:
    vals = []
    for model in MODEL_ORDER:
        row = semantic_all[
            (semantic_all["encoder"] == enc) &
            (semantic_all["model_id"] == model)
        ]
        vals.append(row["spearman_rho"].iloc[0])
    matrix.append(vals)

gradient_grouped_bars(
    ax,
    centers,
    matrix,
    [ENC_LABEL[e] for e in SEM_ENC],
    width=0.24,
)

ax.axhline(
    0,
    color="black",
    linewidth=1.35,
)

ax.set_xticks(centers)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("Spearman rho")

# Symmetric vertical room for negative and positive grouped bars.
_flat_allpairs = np.asarray(matrix, dtype=float).ravel()
_lim = max(0.36, np.nanmax(np.abs(_flat_allpairs)) * 1.27)
ax.set_ylim(-_lim, _lim)

annotate_gradient_bars(ax, fmt=".3f", fs=10.2, pad=0.012)

leg = ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.72,
)
leg.get_frame().set_edgecolor("black")

title_box(
    ax,
    "Global POSIX and Local Semantic Drift Capture Different Scales",
    "Canonical all-pairs POSIX vs adjacent-output PSI",
)

save(fig, "09_allpairs_posix_vs_semantic_psi")


# ============================================================
# 10 Hexbin POSIX vs CLIP PSI
# ============================================================

clip = sens[
    sens["encoder"] == "clip_vit_b32"
].copy()

keys = [
    "model_id",
    "dataset",
    "record_id",
    "from_condition",
    "to_condition",
]

sc = clip.merge(
    posix_adj[
        keys + ["adjacent_posix"]
    ],
    on=keys,
    how="inner",
    validate="one_to_one",
)

fig, ax = plt.subplots(figsize=(9.6, 8.0))
figure_gradient(fig, "Greys", 0.12)
panel_background(ax, 0.56)

hb = ax.hexbin(
    sc["adjacent_posix"],
    sc["psi"],
    gridsize=46,
    mincnt=1,
    cmap="viridis",
    alpha=0.83,
    linewidths=0.28,
    edgecolors="black",
)

cb = fig.colorbar(hb, ax=ax)
cb.outline.set_edgecolor("black")
cb.outline.set_linewidth(1.0)
cb.set_label("Transition count")

ax.set_xlabel("Adjacent POSIX")
ax.set_ylabel("CLIP PSI")

title_box(
    ax,
    "Likelihood Sensitivity vs Realized Semantic Drift",
    "4,605 matched adjacent transitions",
)

save(fig, "10_adjacent_posix_vs_clip_psi_hexbin")


# ============================================================
# 11 Hexbin PSI vs F1
# ============================================================

fig, ax = plt.subplots(figsize=(9.6, 8.0))
figure_gradient(fig, "Greys", 0.12)
panel_background(ax, 0.56)

hb = ax.hexbin(
    clip["psi"],
    clip["abs_delta_partial_f1"],
    gridsize=46,
    mincnt=1,
    cmap="plasma",
    alpha=0.82,
    linewidths=0.28,
    edgecolors="black",
)

cb = fig.colorbar(hb, ax=ax)
cb.outline.set_edgecolor("black")
cb.outline.set_linewidth(1.0)
cb.set_label("Transition count")

ax.set_xlabel("CLIP PSI")
ax.set_ylabel("|Δ partial F1|")

title_box(
    ax,
    "Semantic Drift Predicts the Magnitude of Quality Change",
    "4,605 adjacent transitions",
)

save(fig, "11_clip_psi_vs_abs_delta_partial_f1_hexbin")


# ============================================================
# 12 POSIX vs F1 by model
# ============================================================

pf = posix_adj_f1[
    (posix_adj_f1["level"] == "model") &
    (posix_adj_f1["target"] == "abs_delta_partial_f1")
].copy()

pf["order"] = pf["model_id"].map(
    {m:i for i,m in enumerate(MODEL_ORDER)}
)
pf = pf.sort_values("order")

fig, ax = plt.subplots(figsize=(9.7, 7.4))
figure_gradient(fig, "Greys", 0.14)
panel_background(ax, 0.64)

x = np.arange(len(pf))
vals = pf["spearman_rho"].to_numpy()

gradient_bar(
    ax,
    x,
    vals,
    width=0.66,
    cmap="cividis",
)

ax.set_xlim(-0.7, len(x)-0.3)
ax.set_ylim(0, 0.54)

ax.set_xticks(x)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in pf["model_id"]]
)

ax.set_ylabel("Spearman rho")

for xi, yi in zip(x, vals):
    value_bbox(ax, xi, yi, f"{yi:.3f}")

title_box(
    ax,
    "Adjacent POSIX Also Tracks Extraction Change",
    "but more weakly than semantic PSI",
)

save(fig, "12_adjacent_posix_vs_abs_delta_f1_by_model")


# ============================================================
# 13 Transition robustness
# ============================================================

if encoder_transition is not None:

    tr = encoder_transition[
        encoder_transition["encoder_a"].isin(SEM_ENC) &
        encoder_transition["encoder_b"].isin(SEM_ENC)
    ].copy()

    tr["pair"] = (
        tr["encoder_a"].map(ENC_LABEL)
        + " vs "
        + tr["encoder_b"].map(ENC_LABEL)
    )

    tr_sum = (
        tr.groupby("pair", as_index=False)
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
        )
        .sort_values("mean_spearman", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    figure_gradient(fig, "Greys", 0.14)
    panel_background(ax, 0.64)

    x = np.arange(len(tr_sum))
    vals = tr_sum["mean_spearman"].to_numpy()

    gradient_bar(
        ax,
        x,
        vals,
        width=0.64,
        cmap="viridis",
    )

    ax.set_xlim(-0.7, len(x)-0.3)
    ax.set_ylim(0, 1.06)

    ax.set_xticks(x)
    ax.set_xticklabels(
        tr_sum["pair"],
        rotation=16,
        ha="right",
    )

    ax.set_ylabel("Mean Spearman rho")

    for xi, yi in zip(x, vals):
        value_bbox(ax, xi, yi, f"{yi:.3f}")

    title_box(
        ax,
        "Prompt-Transition Rankings Are Robust Across Encoders"
    )

    save(fig, "13_transition_rank_encoder_robustness")


# ============================================================
# 14 Field robustness
# ============================================================

if encoder_field is not None:

    fr = encoder_field[
        encoder_field["encoder_a"].isin(SEM_ENC) &
        encoder_field["encoder_b"].isin(SEM_ENC)
    ].copy()

    fr["pair"] = (
        fr["encoder_a"].map(ENC_LABEL)
        + " vs "
        + fr["encoder_b"].map(ENC_LABEL)
    )

    fr_sum = (
        fr.groupby("pair", as_index=False)
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
        )
        .sort_values("mean_spearman", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    figure_gradient(fig, "Greys", 0.14)
    panel_background(ax, 0.64)

    x = np.arange(len(fr_sum))
    vals = fr_sum["mean_spearman"].to_numpy()

    gradient_bar(
        ax,
        x,
        vals,
        width=0.64,
        cmap="plasma",
    )

    ax.set_xlim(-0.7, len(x)-0.3)
    ax.set_ylim(0, 1.06)

    ax.set_xticks(x)
    ax.set_xticklabels(
        fr_sum["pair"],
        rotation=16,
        ha="right",
    )

    ax.set_ylabel("Mean Spearman rho")

    for xi, yi in zip(x, vals):
        value_bbox(ax, xi, yi, f"{yi:.3f}")

    title_box(
        ax,
        "Field Rankings Are More Encoder Dependent",
        "Fine-grained volatility is less invariant than document-level PSI",
    )

    save(fig, "14_field_rank_encoder_robustness")


print("\n=== RESTYLE DONE ===")
print(OUT)

for p in sorted(OUT.glob("*.pdf")):
    print(p.name)
