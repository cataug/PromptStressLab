from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/tahiti/PromptStressLab")
SRC = ROOT / "outputs" / "reviewer_posthoc"
OUT = ROOT / "FIGURES_NEW"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "lines.linewidth": 2.5,
    "savefig.dpi": 260,
    "savefig.bbox": "tight",
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
    p = SRC / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def read_optional(name):
    p = SRC / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def annotate_vertical(ax, fmt=".3f", dy=0.008, fs=12):
    ymin, ymax = ax.get_ylim()
    dr = ymax - ymin
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.text(
                patch.get_x() + patch.get_width()/2,
                h + dy*dr,
                format(h, fmt),
                ha="center",
                va="bottom",
                fontsize=fs,
            )


print("=== LOAD RESULTS ===")

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

print("encoder_agreement:", encoder_agreement.shape)
print("psi_f1:", psi_f1.shape)
print("practical:", practical.shape)
print("posix_summary:", posix_summary.shape)
print("posix_adj_psi:", posix_adj_psi.shape)
print("posix_adj_f1:", posix_adj_f1.shape)
print("posix_adj:", posix_adj.shape)
print("sensitivity:", sens.shape)


# ============================================================
# STATISTICS
# ============================================================

print("\n=== BUILD SUMMARY STATISTICS ===")

stats = {}


# ------------------------------------------------------------
# Encoder agreement
# ------------------------------------------------------------

ea = encoder_agreement.copy()

stats["encoder_agreement_spearman_min"] = float(ea["spearman_rho"].min())
stats["encoder_agreement_spearman_max"] = float(ea["spearman_rho"].max())
stats["encoder_agreement_kendall_min"] = float(ea["kendall_tau"].min())
stats["encoder_agreement_kendall_max"] = float(ea["kendall_tau"].max())
stats["encoder_agreement_top10_jaccard_min"] = float(ea["top10pct_jaccard"].min())
stats["encoder_agreement_top10_jaccard_max"] = float(ea["top10pct_jaccard"].max())

ea.to_csv(OUT / "stats_encoder_agreement.csv", index=False)


# ------------------------------------------------------------
# Overall PSI <-> abs delta partial F1
# ------------------------------------------------------------

overall_psi = psi_f1[
    (psi_f1["level"] == "overall") &
    (psi_f1["target"] == "abs_delta_partial_f1")
].copy()

overall_psi["encoder_order"] = overall_psi["encoder"].map(
    {e:i for i,e in enumerate(ENC_ORDER)}
)
overall_psi = overall_psi.sort_values("encoder_order")

overall_psi[
    ["encoder", "n", "spearman_rho", "ci95_low", "ci95_high", "p_holm"]
].to_csv(
    OUT / "stats_psi_vs_abs_delta_partial_f1_overall.csv",
    index=False
)

semantic_overall = overall_psi[
    overall_psi["encoder"].isin(SEM_ENC)
]

stats["semantic_psi_abs_f1_rho_min"] = float(
    semantic_overall["spearman_rho"].min()
)
stats["semantic_psi_abs_f1_rho_max"] = float(
    semantic_overall["spearman_rho"].max()
)


# ------------------------------------------------------------
# Model-level PSI <-> abs delta partial F1
# ------------------------------------------------------------

model_psi = psi_f1[
    (psi_f1["level"] == "model") &
    (psi_f1["target"] == "abs_delta_partial_f1") &
    (psi_f1["encoder"].isin(SEM_ENC))
].copy()

model_psi[
    [
        "encoder", "model_id", "n",
        "spearman_rho", "ci95_low", "ci95_high", "p_holm"
    ]
].to_csv(
    OUT / "stats_semantic_psi_vs_abs_delta_partial_f1_by_model.csv",
    index=False
)


# ------------------------------------------------------------
# Signed delta check
# ------------------------------------------------------------

signed = psi_f1[
    (psi_f1["level"] == "overall") &
    (psi_f1["target"] == "delta_partial_f1") &
    (psi_f1["encoder"].isin(SEM_ENC))
].copy()

signed[
    ["encoder", "n", "spearman_rho", "ci95_low", "ci95_high", "p_holm"]
].to_csv(
    OUT / "stats_psi_vs_signed_delta_partial_f1.csv",
    index=False
)


# ------------------------------------------------------------
# Practical PSI utility
# ------------------------------------------------------------

practical.to_csv(
    OUT / "stats_psi_practical_utility_all.csv",
    index=False
)

practical_20 = practical[
    (practical["threshold_abs_delta_f1"] == 0.20) &
    (practical["encoder"].isin(SEM_ENC))
].copy()

practical_20.to_csv(
    OUT / "stats_psi_practical_utility_threshold_020.csv",
    index=False
)

stats["psi_utility_auc_020_min"] = float(practical_20["auroc"].min())
stats["psi_utility_auc_020_max"] = float(practical_20["auroc"].max())


# ------------------------------------------------------------
# Adjacent POSIX vs PSI
# ------------------------------------------------------------

adj_overall = posix_adj_psi[
    posix_adj_psi["level"] == "overall"
].copy()

adj_overall["encoder_order"] = adj_overall["encoder"].map(
    {e:i for i,e in enumerate(ENC_ORDER)}
)
adj_overall = adj_overall.sort_values("encoder_order")

adj_overall[
    ["encoder", "n", "spearman_rho", "p_value", "p_holm"]
].to_csv(
    OUT / "stats_adjacent_posix_vs_psi_overall.csv",
    index=False
)

adj_sem = adj_overall[
    adj_overall["encoder"].isin(SEM_ENC)
]

stats["adjacent_posix_semantic_psi_rho_min"] = float(
    adj_sem["spearman_rho"].min()
)
stats["adjacent_posix_semantic_psi_rho_max"] = float(
    adj_sem["spearman_rho"].max()
)


# ------------------------------------------------------------
# Adjacent POSIX vs F1
# ------------------------------------------------------------

adj_f1_main = posix_adj_f1[
    (posix_adj_f1["target"] == "abs_delta_partial_f1") &
    (posix_adj_f1["level"].isin(["overall", "model"]))
].copy()

adj_f1_main.to_csv(
    OUT / "stats_adjacent_posix_vs_abs_delta_partial_f1.csv",
    index=False
)

overall_posix_f1 = adj_f1_main[
    adj_f1_main["level"] == "overall"
]["spearman_rho"].iloc[0]

stats["adjacent_posix_abs_f1_rho_overall"] = float(overall_posix_f1)


# ------------------------------------------------------------
# Canonical POSIX
# ------------------------------------------------------------

posix_summary.to_csv(
    OUT / "stats_posix_by_model_dataset.csv",
    index=False
)


# ------------------------------------------------------------
# All-pairs POSIX comparisons
# ------------------------------------------------------------

posix_all.to_csv(
    OUT / "stats_posix_allpairs_comparisons.csv",
    index=False
)


# ------------------------------------------------------------
# Summary JSON / TXT
# ------------------------------------------------------------

with open(OUT / "stats_headline.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

with open(OUT / "stats_headline.txt", "w", encoding="utf-8") as f:
    f.write("REVIEWER POST-HOC HEADLINE STATISTICS\n")
    f.write("="*72 + "\n\n")

    f.write(
        "Document-level semantic-encoder PSI agreement, Spearman rho: "
        f"{stats['encoder_agreement_spearman_min']:.3f}--"
        f"{stats['encoder_agreement_spearman_max']:.3f}\n"
    )

    f.write(
        "Document-level semantic-encoder PSI agreement, Kendall tau: "
        f"{stats['encoder_agreement_kendall_min']:.3f}--"
        f"{stats['encoder_agreement_kendall_max']:.3f}\n"
    )

    f.write(
        "Top-10% PSI overlap, Jaccard: "
        f"{stats['encoder_agreement_top10_jaccard_min']:.3f}--"
        f"{stats['encoder_agreement_top10_jaccard_max']:.3f}\n\n"
    )

    f.write(
        "Semantic PSI vs |Delta partial F1|, overall rho: "
        f"{stats['semantic_psi_abs_f1_rho_min']:.3f}--"
        f"{stats['semantic_psi_abs_f1_rho_max']:.3f}\n"
    )

    f.write(
        "PSI screening AUROC for |Delta partial F1| >= 0.20: "
        f"{stats['psi_utility_auc_020_min']:.3f}--"
        f"{stats['psi_utility_auc_020_max']:.3f}\n"
    )

    f.write(
        "Adjacent POSIX vs semantic PSI, overall rho: "
        f"{stats['adjacent_posix_semantic_psi_rho_min']:.3f}--"
        f"{stats['adjacent_posix_semantic_psi_rho_max']:.3f}\n"
    )

    f.write(
        "Adjacent POSIX vs |Delta partial F1|, overall rho: "
        f"{stats['adjacent_posix_abs_f1_rho_overall']:.3f}\n"
    )


# ============================================================
# FIGURE 01 — ENCODER AGREEMENT HEATMAP
# ============================================================

encs = SEM_ENC

mat = pd.DataFrame(
    np.eye(len(encs)),
    index=encs,
    columns=encs,
)

for _, r in ea.iterrows():
    a = r["encoder_a"]
    b = r["encoder_b"]
    if a in encs and b in encs:
        mat.loc[a, b] = r["spearman_rho"]
        mat.loc[b, a] = r["spearman_rho"]

fig, ax = plt.subplots(figsize=(8.2, 7.1))

im = ax.imshow(
    mat.values,
    vmin=0.90,
    vmax=1.00,
    aspect="equal",
)

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
            j, i,
            f"{mat.iloc[i,j]:.3f}",
            ha="center",
            va="center",
            fontsize=19,
            color="black",
        )

cb = fig.colorbar(im, ax=ax)
cb.set_label("Spearman rho")

ax.set_title(
    "PSI Is Highly Stable Across Semantic Encoders\n"
    "4,605 adjacent prompt transitions"
)

save(fig, "01_encoder_agreement_heatmap")


# ============================================================
# FIGURE 02 — OVERALL CORRELATION WITH |DELTA F1|
# ============================================================

fig, ax = plt.subplots(figsize=(10.5, 7.5))

x = np.arange(len(overall_psi))
vals = overall_psi["spearman_rho"].to_numpy()

ax.bar(x, vals)

ax.set_xticks(x)
ax.set_xticklabels(
    [ENC_LABEL[e] for e in overall_psi["encoder"]],
    rotation=18,
    ha="right",
)

ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.78)

ax.set_title(
    "Sensitivity Measures Track the Magnitude of Extraction Change\n"
    "Association with |Δ partial F1| across 4,605 transitions"
)

annotate_vertical(ax)

save(fig, "02_sensitivity_vs_abs_delta_partial_f1")


# ============================================================
# FIGURE 03 — MODEL-LEVEL SEMANTIC PSI
# ============================================================

fig, ax = plt.subplots(figsize=(11.8, 7.8))

x = np.arange(len(MODEL_ORDER))
width = 0.22

for j, enc in enumerate(SEM_ENC):
    vals = []

    for model in MODEL_ORDER:
        row = model_psi[
            (model_psi["encoder"] == enc) &
            (model_psi["model_id"] == model)
        ]
        vals.append(
            row["spearman_rho"].iloc[0]
            if len(row) else np.nan
        )

    offset = (j - 1) * width

    ax.bar(
        x + offset,
        vals,
        width=width,
        label=ENC_LABEL[enc],
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.82)

ax.set_title(
    "Semantic PSI–Quality Association Persists Across Models\n"
    "PSI vs |Δ partial F1|"
)

ax.legend(frameon=True)

save(fig, "03_semantic_psi_vs_abs_delta_f1_by_model")


# ============================================================
# FIGURE 04 — PRACTICAL UTILITY AUROC
# ============================================================

fig, ax = plt.subplots(figsize=(11.8, 7.8))

x = np.arange(len(MODEL_ORDER))
width = 0.22

for j, enc in enumerate(SEM_ENC):

    vals = []

    for model in MODEL_ORDER:
        row = practical_20[
            (practical_20["encoder"] == enc) &
            (practical_20["model_id"] == model)
        ]

        vals.append(
            row["auroc"].iloc[0]
            if len(row) else np.nan
        )

    ax.bar(
        x + (j-1)*width,
        vals,
        width=width,
        label=ENC_LABEL[enc],
    )

ax.axhline(0.5, linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("AUROC")
ax.set_ylim(0.45, 0.93)

ax.set_title(
    "PSI as an Operational Risk Signal\n"
    "Detecting prompt transitions with |Δ partial F1| ≥ 0.20"
)

ax.legend(frameon=True)

save(fig, "04_psi_practical_utility_auroc")


# ============================================================
# FIGURE 05 — ADJACENT POSIX VS PSI
# ============================================================

fig, ax = plt.subplots(figsize=(10.5, 7.5))

x = np.arange(len(adj_overall))
vals = adj_overall["spearman_rho"].to_numpy()

ax.bar(x, vals)

ax.set_xticks(x)

ax.set_xticklabels(
    [ENC_LABEL[e] for e in adj_overall["encoder"]],
    rotation=18,
    ha="right",
)

ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.64)

ax.set_title(
    "Likelihood Sensitivity and Output Drift Are Related but Non-Equivalent\n"
    "Adjacent POSIX vs PSI on matched prompt transitions"
)

annotate_vertical(ax)

save(fig, "05_adjacent_posix_vs_psi")


# ============================================================
# FIGURE 06 — PSI VS POSIX FOR REALIZED QUALITY CHANGE
# ============================================================

rows = []

for _, r in semantic_overall.iterrows():
    rows.append({
        "measure": f"{ENC_LABEL[r['encoder']]} PSI",
        "rho": r["spearman_rho"],
    })

rows.append({
    "measure": "Adjacent POSIX",
    "rho": overall_posix_f1,
})

cmp = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10.2, 7.0))

cmp = cmp.sort_values("rho")

y = np.arange(len(cmp))

ax.hlines(
    y,
    xmin=0,
    xmax=cmp["rho"],
    linewidth=4,
)

ax.plot(
    cmp["rho"],
    y,
    "o",
    markersize=12,
)

ax.set_yticks(y)
ax.set_yticklabels(cmp["measure"])

ax.set_xlim(0, 0.74)

ax.set_xlabel(
    "Spearman rho with |Δ partial F1|"
)

ax.set_title(
    "Realized Semantic Drift Tracks Extraction Change\n"
    "More Closely Than Adjacent Likelihood Sensitivity"
)

for xv, yv in zip(cmp["rho"], y):
    ax.text(
        xv + 0.012,
        yv,
        f"{xv:.3f}",
        va="center",
        fontsize=14,
    )

save(fig, "06_psi_vs_posix_quality_association")


# ============================================================
# FIGURE 07 — DOCUMENT-LEVEL POSIX
# ============================================================

fig, ax = plt.subplots(figsize=(11.8, 7.8))

x = np.arange(len(DATASET_ORDER))
width = 0.22

for j, model in enumerate(MODEL_ORDER):

    vals = []

    for dataset in DATASET_ORDER:

        row = posix_summary[
            (posix_summary["model_id"] == model) &
            (posix_summary["dataset"] == dataset)
        ]

        vals.append(
            row["mean_posix"].iloc[0]
            if len(row) else np.nan
        )

    ax.bar(
        x + (j-1)*width,
        vals,
        width=width,
        label=MODEL_SHORT[model],
    )

ax.set_xticks(x)
ax.set_xticklabels(DATASET_ORDER)

ax.set_ylabel("Mean canonical POSIX")

ax.set_title(
    "Prompt-Likelihood Sensitivity Is Strongly Task Dependent\n"
    "Canonical all-pairs POSIX"
)

ax.legend(frameon=True)

save(fig, "07_posix_by_model_dataset")


# ============================================================
# FIGURE 08 — ADJACENT POSIX VS PSI BY MODEL HEATMAP
# ============================================================

tmp = posix_adj_psi[
    (posix_adj_psi["level"] == "model") &
    (posix_adj_psi["encoder"].isin(SEM_ENC))
].copy()

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

fig, ax = plt.subplots(figsize=(9.1, 7.2))

im = ax.imshow(
    matrix.values,
    vmin=0.35,
    vmax=0.62,
    aspect="auto",
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
            j, i,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=17,
            color="black",
        )

cb = fig.colorbar(im, ax=ax)
cb.set_label("Spearman rho")

ax.set_title(
    "Adjacent POSIX–PSI Association\n"
    "Consistent Across Models and Semantic Encoders"
)

save(fig, "08_adjacent_posix_psi_model_heatmap")


# ============================================================
# FIGURE 09 — CANONICAL ALL-PAIRS POSIX RELATIONS
# ============================================================

semantic_all = posix_all[
    (posix_all["comparison"] == "POSIX_vs_semantic_psi") &
    (posix_all["encoder"].isin(SEM_ENC))
].copy()

fig, ax = plt.subplots(figsize=(11.8, 7.8))

x = np.arange(len(MODEL_ORDER))
width = 0.22

for j, enc in enumerate(SEM_ENC):

    vals = []

    for model in MODEL_ORDER:

        row = semantic_all[
            (semantic_all["encoder"] == enc) &
            (semantic_all["model_id"] == model)
        ]

        vals.append(
            row["spearman_rho"].iloc[0]
            if len(row) else np.nan
        )

    ax.bar(
        x + (j-1)*width,
        vals,
        width=width,
        label=ENC_LABEL[enc],
    )

ax.axhline(0, linewidth=1.4)

ax.set_xticks(x)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in MODEL_ORDER]
)

ax.set_ylabel("Spearman rho")

ax.set_title(
    "Global POSIX and Local Semantic Drift Capture Different Sensitivity Scales\n"
    "Canonical all-pairs POSIX vs adjacent-output PSI"
)

ax.legend(frameon=True)

save(fig, "09_allpairs_posix_vs_semantic_psi")


# ============================================================
# FIGURE 10 — SCATTER: ADJACENT POSIX VS CLIP PSI
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

fig, ax = plt.subplots(figsize=(9.2, 7.6))

hb = ax.hexbin(
    sc["adjacent_posix"],
    sc["psi"],
    gridsize=45,
    mincnt=1,
)

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Transition count")

ax.set_xlabel("Adjacent POSIX")
ax.set_ylabel("CLIP PSI")

ax.set_title(
    "Likelihood Sensitivity vs Realized Semantic Drift\n"
    "4,605 matched adjacent transitions"
)

save(fig, "10_adjacent_posix_vs_clip_psi_hexbin")


# ============================================================
# FIGURE 11 — SCATTER PSI VS ABS DELTA F1
# ============================================================

fig, ax = plt.subplots(figsize=(9.2, 7.6))

hb = ax.hexbin(
    clip["psi"],
    clip["abs_delta_partial_f1"],
    gridsize=45,
    mincnt=1,
)

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Transition count")

ax.set_xlabel("CLIP PSI")
ax.set_ylabel("|Δ partial F1|")

ax.set_title(
    "Semantic Output Drift Predicts the Magnitude of Quality Change\n"
    "4,605 adjacent transitions"
)

save(fig, "11_clip_psi_vs_abs_delta_partial_f1_hexbin")


# ============================================================
# FIGURE 12 — MODEL-SPECIFIC POSIX VS ABS F1
# ============================================================

pf = posix_adj_f1[
    (posix_adj_f1["level"] == "model") &
    (posix_adj_f1["target"] == "abs_delta_partial_f1")
].copy()

fig, ax = plt.subplots(figsize=(9.5, 7.0))

pf["model_order"] = pf["model_id"].map(
    {m:i for i,m in enumerate(MODEL_ORDER)}
)

pf = pf.sort_values("model_order")

x = np.arange(len(pf))

ax.bar(
    x,
    pf["spearman_rho"],
)

ax.set_xticks(x)
ax.set_xticklabels(
    [MODEL_SHORT[m] for m in pf["model_id"]]
)

ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.53)

ax.set_title(
    "Adjacent POSIX Also Tracks Extraction Change,\n"
    "but More Weakly Than Semantic PSI"
)

annotate_vertical(ax)

save(fig, "12_adjacent_posix_vs_abs_delta_f1_by_model")


# ============================================================
# OPTIONAL TRANSITION / FIELD RANK ROBUSTNESS
# ============================================================

if encoder_transition is not None:

    tr = encoder_transition[
        encoder_transition["encoder_a"].isin(SEM_ENC) &
        encoder_transition["encoder_b"].isin(SEM_ENC)
    ].copy()

    tr["pair"] = (
        tr["encoder_a"].map(ENC_LABEL) +
        " vs " +
        tr["encoder_b"].map(ENC_LABEL)
    )

    tr_sum = (
        tr.groupby("pair", as_index=False)
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
        )
    )

    tr_sum.to_csv(
        OUT / "stats_transition_rank_robustness.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    x = np.arange(len(tr_sum))

    ax.bar(
        x,
        tr_sum["mean_spearman"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        tr_sum["pair"],
        rotation=18,
        ha="right",
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Spearman rho")

    ax.set_title(
        "Prompt-Transition Rankings Are Robust Across Encoders"
    )

    annotate_vertical(ax)

    save(fig, "13_transition_rank_encoder_robustness")


if encoder_field is not None:

    fr = encoder_field[
        encoder_field["encoder_a"].isin(SEM_ENC) &
        encoder_field["encoder_b"].isin(SEM_ENC)
    ].copy()

    fr["pair"] = (
        fr["encoder_a"].map(ENC_LABEL) +
        " vs " +
        fr["encoder_b"].map(ENC_LABEL)
    )

    fr_sum = (
        fr.groupby("pair", as_index=False)
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
        )
    )

    fr_sum.to_csv(
        OUT / "stats_field_rank_robustness.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    x = np.arange(len(fr_sum))

    ax.bar(
        x,
        fr_sum["mean_spearman"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        fr_sum["pair"],
        rotation=18,
        ha="right",
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Spearman rho")

    ax.set_title(
        "Field Rankings Show More Encoder Dependence\n"
        "than Document-Level PSI"
    )

    annotate_vertical(ax)

    save(fig, "14_field_rank_encoder_robustness")


# ============================================================
# MANIFEST
# ============================================================

generated = sorted(
    p.name
    for p in OUT.iterdir()
    if p.is_file()
)

with open(
    OUT / "MANIFEST.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PromptStressLab reviewer post-hoc figure pack\n"
    )

    f.write(
        "="*72 + "\n\n"
    )

    for name in generated:
        f.write(name + "\n")


print("\n=== DONE ===")
print("Output:", OUT)

print("\nHeadline statistics:")
print((OUT / "stats_headline.txt").read_text())

print("\nGenerated:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(p.name)
