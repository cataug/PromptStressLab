#!/usr/bin/env python3
from pathlib import Path
import re
import math
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================

ROOT_DEFAULT = "/home/tahiti/PromptStressLab"
DEFAULT_TEX = "PromptStressLab_PHASE1_FINAL_INLINE.tex"

VIR = cm.get_cmap("viridis")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "white",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# labels we want to convert to figures
TARGET_LABELS = {
    "tab:r1-ablations-full",
    "tab:r1-psi-jaccard-full",
    "tab:r1-idr-full",
    "tab:r1-condition-quality",
    "tab:r1-condition-tokens",
    "tab:r1-rq5-rank",
    "tab:r1-all-p-quality",
    "tab:r1-all-a-quality",
    "tab:r1-all-p-cost",
}

# ============================================================
# HELPERS
# ============================================================

def clean_tex(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\%", "%")
    s = s.replace(r"\_", "_")
    s = s.replace(r"\$", "$")
    s = s.replace(r"$<0.0001$", "<0.0001")
    s = s.replace(r"$\rightarrow$", "->")
    s = s.replace(r"$\to$", "->")
    s = s.replace(r"\rightarrow", "->")
    s = s.replace(r"\to", "->")
    s = s.replace(r"\mathrm{schema}", "schema")
    s = s.replace(r"\mathrm{tokens}", "tokens")
    s = s.replace(r"\mathrm{latency}", "latency")
    s = s.replace(r"\mathrm{PSI}", "PSI")
    s = s.replace(r"\rm", "")
    s = s.replace(r"\!", "")
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\caption\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("~", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_float(x):
    if x is None:
        return np.nan
    x = str(x).strip()
    if x in {"---", "--", "-", ""}:
        return np.nan
    if x.startswith("<"):
        try:
            val = float(x[1:])
            return val / 2.0
        except Exception:
            return np.nan
    x = x.replace(",", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_longtables(tex_text: str):
    pattern = re.compile(
        r"(\\begin\{longtable\}\{.*?\\label\{(?P<label>[^}]+)\}.*?\\end\{longtable\})",
        re.S
    )
    found = {}
    for m in pattern.finditer(tex_text):
        block = m.group(1)
        label = m.group("label").strip()
        if label not in TARGET_LABELS:
            continue
        found[label] = block
    return found

def extract_caption(block: str) -> str:
    m = re.search(r"\\caption\{(.*?)\}\\label\{", block, re.S)
    if not m:
        return ""
    return clean_tex(m.group(1))

def extract_rows(block: str):
    m = re.search(r"\\endfoot(.*?)\\end\{longtable\}", block, re.S)
    if not m:
        return []
    body = m.group(1)
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(r"\addlinespace"):
            continue
        if any(line.startswith(x) for x in [r"\toprule", r"\midrule", r"\bottomrule", r"\endfirsthead", r"\endhead"]):
            continue
        if "&" not in line:
            continue
        line = line.rstrip("\\").strip()
        if line.endswith(r"\\"):
            line = line[:-2].strip()
        parts = [clean_tex(p) for p in line.split("&")]
        lines.append(parts)
    return lines

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def savefig(fig, path_pdf: Path, path_png: Path):
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=220)
    plt.close(fig)

def wrapped_caption(s, width=118):
    return "\n".join(textwrap.wrap(s, width=width))

def replace_block(tex_text: str, old_block: str, new_tex: str) -> str:
    return tex_text.replace(old_block, new_tex)


def heatmap_with_text(
    ax,
    data,
    row_labels,
    col_labels,
    title="",
    fmt="{:.3f}",
    cmap="viridis",
    vmin=None,
    vmax=None,
    text_color="black",
    star_mask=None,
    nan_text=""
):
    """
    Scientific viridis heatmap:

    - translucent gradient background within each cell
    - thin black cell outlines
    - black text in white rounded boxes
    - larger font sizes
    - stronger separators between groups of three rows
    - explicit missing-value indication
    """

    from matplotlib.patches import Rectangle

    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {arr.shape}"
        )

    nr, nc = arr.shape

    valid = arr[np.isfinite(arr)]

    if valid.size:
        mn = (
            float(np.min(valid))
            if vmin is None
            else float(vmin)
        )
        mx = (
            float(np.max(valid))
            if vmax is None
            else float(vmax)
        )
    else:
        mn, mx = 0.0, 1.0

    if mx <= mn:
        mx = mn + 1e-9

    # --------------------------------------------------------
    # BASE VIRIDIS COLORS
    # --------------------------------------------------------

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f0f2f0")

    im = ax.imshow(
        np.ma.masked_invalid(arr),
        cmap=cmap_obj,
        vmin=mn,
        vmax=mx,
        interpolation="nearest",
        aspect="auto",
        alpha=0.86,
        zorder=1
    )

    # --------------------------------------------------------
    # GRADIENT TRANSPARENCY
    # --------------------------------------------------------
    # Within every cell:
    # left = lighter / more transparent
    # right = more saturated
    #
    # Data-dependent viridis hue is preserved.

    cell_w = 44
    cell_h = 26

    glaze = np.ones(
        (nr * cell_h, nc * cell_w, 4),
        dtype=np.float32
    )

    sheen = np.linspace(
        0.36,
        0.075,
        cell_w,
        dtype=np.float32
    )

    for j in range(nc):
        glaze[
            :,
            j * cell_w:(j + 1) * cell_w,
            3
        ] = sheen[None, :]

    ax.imshow(
        glaze,
        extent=(-0.5, nc - 0.5, nr - 0.5, -0.5),
        interpolation="bilinear",
        aspect="auto",
        zorder=2
    )

    # --------------------------------------------------------
    # BLACK CELL OUTLINES
    # --------------------------------------------------------

    for i in range(nr):
        for j in range(nc):

            border = Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#16221e",
                linewidth=0.78,
                zorder=3
            )

            ax.add_patch(border)

    # Model groups: Gemma / Mistral / Qwen
    if nr == 9:
        for boundary in (2.5, 5.5):

            ax.axhline(
                boundary,
                color="#081510",
                linewidth=2.0,
                zorder=4
            )

    # --------------------------------------------------------
    # LARGE NUMBERS INSIDE WHITE BBOX
    # --------------------------------------------------------

    if nc <= 6:
        number_font = 11.5
    elif nc == 7:
        number_font = 10.8
    else:
        number_font = 10.4

    for i in range(nr):
        for j in range(nc):

            val = arr[i, j]

            if np.isfinite(val):

                number = fmt.format(val)

                if (
                    star_mask is not None
                    and bool(star_mask[i, j])
                ):
                    number += "*"

            else:
                number = (
                    nan_text
                    if nan_text
                    else "—"
                )

            ax.text(
                j,
                i,
                number,
                ha="center",
                va="center",
                fontsize=number_font,
                fontweight="semibold",
                color="#101715",
                zorder=6,
                bbox=dict(
                    boxstyle=(
                        "round,pad=0.18,"
                        "rounding_size=0.10"
                    ),
                    facecolor=(
                        1.0,
                        1.0,
                        1.0,
                        0.92
                    ),
                    edgecolor="#111a16",
                    linewidth=0.88
                )
            )

    # --------------------------------------------------------
    # TITLES AND AXIS LABELS
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=15,
        fontweight="semibold",
        color="#101b15",
        pad=12
    )

    ax.set_xticks(np.arange(nc))

    ax.set_xticklabels(
        col_labels,
        rotation=0 if nc <= 7 else 22,
        ha="center" if nc <= 7 else "right",
        fontsize=11.2,
        fontweight="medium"
    )

    ax.set_yticks(np.arange(nr))

    ax.set_yticklabels(
        row_labels,
        fontsize=11.2,
        fontweight="medium"
    )

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        pad=6
    )

    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)

    # --------------------------------------------------------
    # OUTER FRAME
    # --------------------------------------------------------

    for spine in ax.spines.values():

        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_edgecolor("#111a16")

    return im


# ============================================================
# TABLE PARSERS
# ============================================================

def df_ablations(rows):
    cols = ["Model", "Dataset", "Contrast", "DeltaF1", "Holm_p_F1", "rrb", "Holm_p_schema"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_psi_jaccard(rows):
    cols = ["Model", "Dataset", "Measure", "n_d", "n_t", "rho", "CI", "Holm_p"]
    df = pd.DataFrame(rows, columns=cols)
    df["n_d"] = df["n_d"].map(to_float)
    df["n_t"] = df["n_t"].map(to_float)
    df["rho"] = df["rho"].map(to_float)
    df["Holm_p"] = df["Holm_p"].map(to_float)
    low = []
    high = []
    for x in df["CI"]:
        m = re.match(r"\[(.*?),(.*?)\]", x)
        if m:
            low.append(to_float(m.group(1)))
            high.append(to_float(m.group(2)))
        else:
            low.append(np.nan); high.append(np.nan)
    df["CI_low"] = low
    df["CI_high"] = high
    return df

def df_idr(rows):
    cols = ["Model", "Dataset", "Transition", "Disappearance", "Appearance", "Changed_output"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_condition_quality(rows):
    cols = ["Model", "Dataset", "Cond", "EntityF1", "RelationF1", "Schema", "Unsupported", "LegacyCap"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_condition_tokens(rows):
    cols = ["Model", "Dataset", "Cond", "MeanInput", "MeanPaddedWidth"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_rq5(rows):
    cols = ["Model", "Dataset", "tau_F1_schema", "tau_F1_tokens", "tau_F1_latency", "tau_F1_PSI"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[2:]:
        df[c] = df[c].map(to_float)
    return df

def df_all_p_quality(rows):
    cols = ["Model", "Data", "Prompt", "n", "PartialF1", "ExactF1", "RelationF1", "Schema", "JSON", "Unsupported"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_all_a_quality(rows):
    cols = ["Model", "Data", "Prompt", "n", "PartialF1", "RelationF1", "Schema", "JSON", "Unsupported"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

def df_all_p_cost(rows):
    cols = ["Model", "Data", "Prompt", "n", "InputTokens", "PaddedWidth", "SecPerItem", "BatchCap"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].map(to_float)
    return df

# ============================================================
# FIGURE BUILDERS
# ============================================================

def fig_ablations(df, out_pdf, out_png):
    models = ["Gemma", "Mistral", "Qwen"]
    datasets = ["EBM-NLP", "SciER", "SciERC"]
    contrasts = ["A1-A0", "A2-A0", "A3-A0", "A4-A0", "A5-A0", "A6-A0"]

    fig, axes = plt.subplots(3, 3, figsize=(13.6, 9.6), constrained_layout=True)
    v = np.nanmax(np.abs(df["DeltaF1"].to_numpy()))
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            ax = axes[i, j]
            sub = df[(df["Model"] == m) & (df["Dataset"] == d)].copy()
            sub = sub.set_index("Contrast").reindex(contrasts)
            arr = sub["DeltaF1"].to_numpy().reshape(1, -1)
            stars = (sub["Holm_p_F1"].to_numpy() < 0.05).reshape(1, -1)
            im = heatmap_with_text(
                ax, arr, [f"{m}\n{d}"], contrasts,
                title=f"{m} · {d}",
                fmt="{:.3f}",
                cmap="viridis",
                vmin=-v, vmax=v,
                star_mask=stars
            )
            ax.set_yticklabels(["ΔF1"])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    cbar.set_label("Δ Partial F1 versus A0")
    fig.suptitle("Ablation effects (A1–A6 versus A0)", y=1.02, fontsize=14)
    savefig(fig, out_pdf, out_png)

def fig_psi_jaccard(df, out_pdf, out_png):
    df = df.copy()
    df["Cell"] = df["Model"] + " · " + df["Dataset"]
    cells = [
        "Gemma · EBM-NLP", "Gemma · SciER", "Gemma · SciERC",
        "Mistral · EBM-NLP", "Mistral · SciER", "Mistral · SciERC",
        "Qwen · EBM-NLP", "Qwen · SciER", "Qwen · SciERC",
    ]
    y = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(11.8, 7.0), constrained_layout=True)
    offsets = {"PSI": -0.15, "Jaccard": +0.15}
    colors_map = {"PSI": VIR(0.28), "Jaccard": VIR(0.78)}
    for meas in ["PSI", "Jaccard"]:
        sub = df[df["Measure"] == meas].set_index("Cell").reindex(cells)
        x = sub["rho"].to_numpy()
        lo = x - sub["CI_low"].to_numpy()
        hi = sub["CI_high"].to_numpy() - x
        ax.errorbar(x, y + offsets[meas], xerr=[lo, hi], fmt="o",
                    color=colors_map[meas], ecolor=colors_map[meas],
                    capsize=4, lw=1.6, markersize=7, markeredgecolor="black", markeredgewidth=1.0, label=meas)

        for val, yy in zip(x, y + offsets[meas]):
            if np.isfinite(val):
                ax.annotate(
                    f"{val:.2f}",
                    xy=(val, yy),
                    xytext=(9, 3),
                    textcoords="offset points",
                    fontsize=9.8,
                    color="black",
                    fontweight="semibold",
                    bbox=dict(
                        boxstyle="round,pad=.15",
                        facecolor="white",
                        edgecolor="black",
                        linewidth=.65,
                        alpha=.92
                    )
                )
    ax.set_yticks(y)
    ax.set_yticklabels(cells)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Spearman ρ with |Δ partial F1|")
    ax.set_title("Changed-output association: PSI versus entity-set Jaccard")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.invert_yaxis()
    savefig(fig, out_pdf, out_png)

def fig_idr(df, out_pdf, out_png):
    df = df.copy()
    df["Row"] = df["Model"] + " · " + df["Dataset"]
    rows = [
        "Gemma · EBM-NLP", "Gemma · SciER", "Gemma · SciERC",
        "Mistral · EBM-NLP", "Mistral · SciER", "Mistral · SciERC",
        "Qwen · EBM-NLP", "Qwen · SciER", "Qwen · SciERC",
    ]
    trans = ["P1->P2", "P2->P3", "P3->P4", "P4->P5", "P5->P6"]
    df["Transition"] = df["Transition"].str.replace(" ", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 7.2), constrained_layout=True)
    metrics = [("Disappearance", "Field disappearance"),
               ("Appearance", "Field appearance"),
               ("Changed_output", "Changed extracted output")]
    for ax, (metric, title) in zip(axes, metrics):
        mat = []
        for r in rows:
            sub = df[df["Row"] == r].set_index("Transition").reindex(trans)
            mat.append(sub[metric].to_numpy())
        mat = np.array(mat, dtype=float)
        im = heatmap_with_text(ax, mat, rows, trans, title=title, fmt="{:.2f}",
                               cmap="viridis", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Rate")
    savefig(fig, out_pdf, out_png)

def fig_condition_quality(df, out_pdf, out_png):
    metrics = ["EntityF1", "RelationF1", "Schema", "Unsupported", "LegacyCap"]
    titles = ["Entity F1", "Relation F1", "Schema", "Unsupported span", "Legacy batch cap"]
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 7.0), constrained_layout=True)
    rows = []
    row_labels = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            row_labels.append(f"{model}\n{dataset}")
    for ax, metric, ttl in zip(axes, metrics, titles):
        mat = []
        for model in ["Gemma", "Mistral", "Qwen"]:
            for dataset in ["EBM-NLP", "SciER", "SciERC"]:
                sub = df[(df["Model"] == model) & (df["Dataset"] == dataset)].set_index("Cond")
                mat.append([to_float(sub.loc["A5", metric]) if "A5" in sub.index else np.nan,
                            to_float(sub.loc["P6", metric]) if "P6" in sub.index else np.nan])
        mat = np.array(mat, dtype=float)
        vmax = 1.0 if metric != "LegacyCap" else 1.0
        im = heatmap_with_text(ax, mat, row_labels, ["A5", "P6"], title=ttl,
                               fmt="{:.2f}", cmap="viridis", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Value")
    savefig(fig, out_pdf, out_png)

def fig_condition_tokens(df, out_pdf, out_png):
    metrics = ["MeanInput", "MeanPaddedWidth"]
    titles = ["Mean input tokens", "Mean padded width"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.2), constrained_layout=True)
    row_labels = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            row_labels.append(f"{model}\n{dataset}")
    for ax, metric, ttl in zip(axes, metrics, titles):
        mat = []
        for model in ["Gemma", "Mistral", "Qwen"]:
            for dataset in ["EBM-NLP", "SciER", "SciERC"]:
                sub = df[(df["Model"] == model) & (df["Dataset"] == dataset)].set_index("Cond")
                mat.append([to_float(sub.loc["A5", metric]) if "A5" in sub.index else np.nan,
                            to_float(sub.loc["P6", metric]) if "P6" in sub.index else np.nan])
        mat = np.array(mat, dtype=float)
        vmax = np.nanmax(mat)
        im = heatmap_with_text(ax, mat, row_labels, ["A5", "P6"], title=ttl,
                               fmt="{:.0f}", cmap="viridis", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Tokens")
    savefig(fig, out_pdf, out_png)

def fig_rq5(df, out_pdf, out_png):
    rows = []
    data = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            sub = df[(df["Model"] == model) & (df["Dataset"] == dataset)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            rows.append(f"{model}\n{dataset}")
            data.append([
                row["tau_F1_schema"],
                row["tau_F1_tokens"],
                row["tau_F1_latency"],
                row["tau_F1_PSI"],
            ])
    data = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    im = heatmap_with_text(
        ax, data, rows,
        [r"$\tau(F1,\mathrm{schema})$", r"$\tau(F1,\mathrm{tokens})$",
         r"$\tau(F1,\mathrm{latency})$", r"$\tau(F1,\mathrm{PSI})$"],
        title="Rank agreement across prompt conditions",
        fmt="{:.2f}", cmap="viridis", vmin=-1, vmax=1
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
    cbar.set_label("Kendall τ")
    savefig(fig, out_pdf, out_png)

def fig_all_p_quality(df, out_pdf, out_png):
    panels = [
        ("PartialF1", "Partial F1", 0, 0.5),
        ("RelationF1", "Relation F1", 0, 0.12),
        ("Schema", "Schema", 0, 1),
        ("Unsupported", "Unsupported span", 0, max(0.2, float(df["Unsupported"].max()))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2), constrained_layout=True)
    row_labels = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            row_labels.append(f"{model}\n{dataset}")
    prompts = ["P1", "P2", "P3", "P4", "P5", "P6"]
    for ax, (metric, title, vmin, vmax) in zip(axes.flat, panels):
        mat = []
        for model in ["Gemma", "Mistral", "Qwen"]:
            for dataset in ["EBM-NLP", "SciER", "SciERC"]:
                sub = df[(df["Model"] == model) & (df["Data"] == dataset)].set_index("Prompt")
                mat.append([to_float(sub.loc[p, metric]) if p in sub.index else np.nan for p in prompts])
        mat = np.array(mat, dtype=float)
        im = heatmap_with_text(ax, mat, row_labels, prompts, title=title,
                               fmt="{:.2f}", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01)
    cbar.set_label("Value")
    savefig(fig, out_pdf, out_png)

def fig_all_a_quality(df, out_pdf, out_png):
    panels = [
        ("PartialF1", "Partial F1", 0, max(0.5, float(df["PartialF1"].max()))),
        ("RelationF1", "Relation F1", 0, max(0.12, float(np.nanmax(df["RelationF1"])))),
        ("Schema", "Schema", 0, 1),
        ("Unsupported", "Unsupported span", 0, max(0.2, float(df["Unsupported"].max()))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.4), constrained_layout=True)
    row_labels = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            row_labels.append(f"{model}\n{dataset}")
    prompts = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    for ax, (metric, title, vmin, vmax) in zip(axes.flat, panels):
        mat = []
        for model in ["Gemma", "Mistral", "Qwen"]:
            for dataset in ["EBM-NLP", "SciER", "SciERC"]:
                sub = df[(df["Model"] == model) & (df["Data"] == dataset)].set_index("Prompt")
                mat.append([to_float(sub.loc[p, metric]) if p in sub.index else np.nan for p in prompts])
        mat = np.array(mat, dtype=float)
        im = heatmap_with_text(ax, mat, row_labels, prompts, title=title,
                               fmt="{:.2f}", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01)
    cbar.set_label("Value")
    savefig(fig, out_pdf, out_png)

def fig_all_p_cost(df, out_pdf, out_png):
    panels = [
        ("InputTokens", "Input tokens", 0, float(df["InputTokens"].max())),
        ("SecPerItem", "Seconds / item", 0, float(df["SecPerItem"].max())),
        ("BatchCap", "Legacy batch cap", 0, 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.6), constrained_layout=True)
    row_labels = []
    for model in ["Gemma", "Mistral", "Qwen"]:
        for dataset in ["EBM-NLP", "SciER", "SciERC"]:
            row_labels.append(f"{model}\n{dataset}")
    prompts = ["P1", "P2", "P3", "P4", "P5", "P6"]
    for ax, (metric, title, vmin, vmax) in zip(axes, panels):
        mat = []
        for model in ["Gemma", "Mistral", "Qwen"]:
            for dataset in ["EBM-NLP", "SciER", "SciERC"]:
                sub = df[(df["Model"] == model) & (df["Data"] == dataset)].set_index("Prompt")
                mat.append([to_float(sub.loc[p, metric]) if p in sub.index else np.nan for p in prompts])
        mat = np.array(mat, dtype=float)
        fmt = "{:.0f}" if metric != "SecPerItem" else "{:.1f}"
        im = heatmap_with_text(ax, mat, row_labels, prompts, title=title,
                               fmt=fmt, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    cbar.set_label("Value")
    savefig(fig, out_pdf, out_png)

# ============================================================
# TEX SNIPPETS
# ============================================================

def snippet_for(label, fig_rel, caption):
    return rf"""
\begin{{figure*}}[t]
    \centering
    \includegraphics[width=\textwidth]{{{fig_rel}}}
    \caption{{{caption}}}
    \label{{{label}}}
\end{{figure*}}
""".strip() + "\n"

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--tex", default=DEFAULT_TEX)
    args = ap.parse_args()

    root = Path(args.root)
    tex_path = root / args.tex
    if not tex_path.is_file():
        raise FileNotFoundError(f"Missing tex file: {tex_path}")

    out_root = root / "outputs" / "appendix_visuals"
    fig_dir = root / "figs" / "appendix_visuals"
    csv_dir = out_root / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    text = tex_path.read_text(encoding="utf-8")
    blocks = parse_longtables(text)

    missing = sorted(TARGET_LABELS - set(blocks))
    if missing:
        print("[WARN] Missing labels:", ", ".join(missing), flush=True)

    replacements = {}

    # 1) ablations
    if "tab:r1-ablations-full" in blocks:
        rows = extract_rows(blocks["tab:r1-ablations-full"])
        df = df_ablations(rows)
        save_df(df, csv_dir / "r1-ablations-full.csv")
        fig_ablations(df, fig_dir / "r1_ablations_full.pdf", fig_dir / "r1_ablations_full.png")
        replacements["tab:r1-ablations-full"] = snippet_for(
            "tab:r1-ablations-full",
            "figs/appendix_visuals/r1_ablations_full.pdf",
            "Visual summary of document-paired ablation effects. Each cell shows $\\Delta$ partial F1 for A1--A6 relative to A0; asterisk marks Holm-adjusted $p<0.05$ within the corresponding model--dataset family."
        )

    # 2) psi jaccard
    if "tab:r1-psi-jaccard-full" in blocks:
        rows = extract_rows(blocks["tab:r1-psi-jaccard-full"])
        df = df_psi_jaccard(rows)
        save_df(df, csv_dir / "r1-psi-jaccard-full.csv")
        fig_psi_jaccard(df, fig_dir / "r1_psi_jaccard_full.pdf", fig_dir / "r1_psi_jaccard_full.png")
        replacements["tab:r1-psi-jaccard-full"] = snippet_for(
            "tab:r1-psi-jaccard-full",
            "figs/appendix_visuals/r1_psi_jaccard_full.pdf",
            "Changed-output transitions only: associations between output-change measures and absolute changes in partial entity F1. Points show Spearman $\\rho$ with document-cluster bootstrap confidence intervals."
        )

    # 3) idr
    if "tab:r1-idr-full" in blocks:
        rows = extract_rows(blocks["tab:r1-idr-full"])
        df = df_idr(rows)
        save_df(df, csv_dir / "r1-idr-full.csv")
        fig_idr(df, fig_dir / "r1_idr_full.pdf", fig_dir / "r1_idr_full.png")
        replacements["tab:r1-idr-full"] = snippet_for(
            "tab:r1-idr-full",
            "figs/appendix_visuals/r1_idr_full.pdf",
            "Directional field disappearance, field appearance, and changed-output rates for each adjacent prompt transition, shown for all model--dataset cells."
        )

    # 4) A5/P6 quality
    if "tab:r1-condition-quality" in blocks:
        rows = extract_rows(blocks["tab:r1-condition-quality"])
        df = df_condition_quality(rows)
        save_df(df, csv_dir / "r1-condition-quality.csv")
        fig_condition_quality(df, fig_dir / "r1_condition_quality.pdf", fig_dir / "r1_condition_quality.png")
        replacements["tab:r1-condition-quality"] = snippet_for(
            "tab:r1-condition-quality",
            "figs/appendix_visuals/r1_condition_quality.pdf",
            "Observed A5/P6 condition-level quality and validity. Rows correspond to model--dataset cells; columns compare A5 and P6 for entity F1, relation F1, schema validity, unsupported-span rate, and the legacy batch-cap flag."
        )

    # 5) A5/P6 tokens
    if "tab:r1-condition-tokens" in blocks:
        rows = extract_rows(blocks["tab:r1-condition-tokens"])
        df = df_condition_tokens(rows)
        save_df(df, csv_dir / "r1-condition-tokens.csv")
        fig_condition_tokens(df, fig_dir / "r1_condition_tokens.pdf", fig_dir / "r1_condition_tokens.png")
        replacements["tab:r1-condition-tokens"] = snippet_for(
            "tab:r1-condition-tokens",
            "figs/appendix_visuals/r1_condition_tokens.pdf",
            "Observed A5/P6 token budgets measured with each model tokenizer. Panels compare mean input-token counts and mean padded widths over the matched evaluation subsets."
        )

    # 6) rq5 ranks
    if "tab:r1-rq5-rank" in blocks:
        rows = extract_rows(blocks["tab:r1-rq5-rank"])
        df = df_rq5(rows)
        save_df(df, csv_dir / "r1-rq5-rank.csv")
        fig_rq5(df, fig_dir / "r1_rq5_rank.pdf", fig_dir / "r1_rq5_rank.png")
        replacements["tab:r1-rq5-rank"] = snippet_for(
            "tab:r1-rq5-rank",
            "figs/appendix_visuals/r1_rq5_rank.pdf",
            "Kendall rank agreement between extraction quality and competing criteria across the six prompt configurations. Positive values denote better directional agreement."
        )

    # 7) all P1-P6 quality
    if "tab:r1-all-p-quality" in blocks:
        rows = extract_rows(blocks["tab:r1-all-p-quality"])
        df = df_all_p_quality(rows)
        save_df(df, csv_dir / "r1-all-p-quality.csv")
        fig_all_p_quality(df, fig_dir / "r1_all_p_quality.pdf", fig_dir / "r1_all_p_quality.png")
        replacements["tab:r1-all-p-quality"] = snippet_for(
            "tab:r1-all-p-quality",
            "figs/appendix_visuals/r1_all_p_quality.pdf",
            "All main P1--P6 conditions: condition-level quality and validity summaries shown as heatmaps over the nine model--dataset cells."
        )

    # 8) all A0-A6 quality
    if "tab:r1-all-a-quality" in blocks:
        rows = extract_rows(blocks["tab:r1-all-a-quality"])
        df = df_all_a_quality(rows)
        save_df(df, csv_dir / "r1-all-a-quality.csv")
        fig_all_a_quality(df, fig_dir / "r1_all_a_quality.pdf", fig_dir / "r1_all_a_quality.png")
        replacements["tab:r1-all-a-quality"] = snippet_for(
            "tab:r1-all-a-quality",
            "figs/appendix_visuals/r1_all_a_quality.pdf",
            "All ablation conditions A0--A6: condition-level quality and validity summaries shown as heatmaps over the nine model--dataset cells."
        )

    # 9) all P1-P6 cost
    if "tab:r1-all-p-cost" in blocks:
        rows = extract_rows(blocks["tab:r1-all-p-cost"])
        df = df_all_p_cost(rows)
        save_df(df, csv_dir / "r1-all-p-cost.csv")
        fig_all_p_cost(df, fig_dir / "r1_all_p_cost.pdf", fig_dir / "r1_all_p_cost.png")
        replacements["tab:r1-all-p-cost"] = snippet_for(
            "tab:r1-all-p-cost",
            "figs/appendix_visuals/r1_all_p_cost.pdf",
            "All main P1--P6 cost and execution summaries. Heatmaps report input tokens, observed seconds per item, and the legacy batch-cap indicator."
        )

    patched = text
    for label, block in blocks.items():
        if label in replacements:
            patched = replace_block(patched, block, replacements[label])

    out_tex = root / "PromptStressLab_visual_appendix.tex"
    out_tex.write_text(patched, encoding="utf-8")

    print("\n[OK] appendix visuals built", flush=True)
    print("[TEX ]", out_tex, flush=True)
    print("[FIG ]", fig_dir, flush=True)
    print("[CSV ]", csv_dir, flush=True)

if __name__ == "__main__":
    main()
