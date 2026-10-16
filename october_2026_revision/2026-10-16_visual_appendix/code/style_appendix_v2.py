#!/usr/bin/env python3

from pathlib import Path
import ast
import shutil

ROOT = Path("/home/tahiti/PromptStressLab")
SOURCE = ROOT / "build_appendix_visuals.py"
BACKUP = ROOT / "build_appendix_visuals_before_style_v2.py"

if not SOURCE.is_file():
    raise SystemExit(
        "[ERROR] build_appendix_visuals.py not found. "
        "First create it using the previous cat-block."
    )

text = SOURCE.read_text(encoding="utf-8")

if not BACKUP.exists():
    shutil.copy2(SOURCE, BACKUP)

start = text.find("def heatmap_with_text(")
end = text.find(
    "# ============================================================\n"
    "# TABLE PARSERS",
    start
)

if start < 0 or end < 0:
    raise SystemExit(
        "[ERROR] Cannot locate original heatmap function. "
        "No changes applied."
    )

# ============================================================
# NEW UNIFIED VIRIDIS STYLE
# ============================================================

helper = r'''
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


'''

# Replace plotting helper in previous script.
text = text[:start] + helper + text[end:]

# ============================================================
# GLOBAL FONT AND PANEL CONFIGURATION
# ============================================================

settings = {
    '"font.size": 10':
        '"font.size": 12',

    '"axes.titlesize": 12':
        '"axes.titlesize": 15',

    '"axes.labelsize": 10':
        '"axes.labelsize": 12',

    '"xtick.labelsize": 9':
        '"xtick.labelsize": 11',

    '"ytick.labelsize": 9':
        '"ytick.labelsize": 11',

    '"legend.fontsize": 9':
        '"legend.fontsize": 11',

    '"axes.facecolor": "#fbfcfc"':
        '"axes.facecolor": "#ffffff"',

    '"axes.spines.top": False':
        '"axes.spines.top": True',

    '"axes.spines.right": False':
        '"axes.spines.right": True',
}

for old, new in settings.items():
    text = text.replace(old, new)

# ============================================================
# CORRELATION FOREST PLOT: BLACK OUTLINED MARKERS + NUMBER BBOX
# ============================================================

needle = "capsize=3, lw=1.2, label=meas)"

if needle in text:

    text = text.replace(
        needle,
        'capsize=4, lw=1.6, markersize=7, '
        'markeredgecolor="black", '
        'markeredgewidth=1.0, label=meas)',
        1
    )

    anchor = "    ax.set_yticks(y)"

    addition = '''
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
'''

    function_start = text.find(
        "def fig_psi_jaccard("
    )

    position = text.find(
        anchor,
        function_start
    )

    if function_start >= 0 and position >= 0:
        text = (
            text[:position]
            + addition
            + text[position:]
        )

# ============================================================
# VALIDATE AND SAVE
# ============================================================

ast.parse(text)

SOURCE.write_text(
    text,
    encoding="utf-8"
)

print("[OK] Updated:", SOURCE)
print("[BACKUP]", BACKUP)
print("[STYLE] Viridis gradient transparency")
print("[STYLE] White number bbox, black outlines")
print("[STYLE] Large labels, model separators")
print("[OK] Ready to regenerate figures")
