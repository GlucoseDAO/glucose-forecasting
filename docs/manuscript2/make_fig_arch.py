"""SugarJEPA-288 architecture figure, matching jepa_paper/sugar_jepa.png style.

Only input shape and branch paths differ from the 128-step original:
shared window is (B, 288, 4); JEPA reads all 288 glucose steps (36 patches);
SugarOne still uses the last 128 steps (flatten 4096 unchanged).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Palette sampled from docs/manuscript2/jepa_paper/sugar_jepa.png
RED = "#F6B0A8"
YELLOW = "#F7E3A8"
GREEN = "#CFE6BF"
PURPLE = "#C4A6C8"
BLUE = "#DFE7F2"
GRAY = "#EFEFEF"
EDGE = "#515151"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    *,
    fontsize: float = 6.4,
    lw: float = 0.85,
    family: str = "DejaVu Sans",
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=facecolor,
            edgecolor=EDGE,
            linewidth=lw,
            linestyle=(0, (3.5, 2.2)) if dashed else "solid",
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=EDGE,
            family=family,
            linespacing=1.25,
        )


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, *, lw: float = 0.8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=EDGE,
            shrinkA=0.5,
            shrinkB=0.5,
        )
    )


def main() -> None:
    # Portrait, close to the original 1394x1624.
    fig, ax = plt.subplots(figsize=(6.97, 8.12), dpi=200)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 16.3)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ----- legend (bottom) -----
    legend = [
        (BLUE, "input / target"),
        (YELLOW, "Linear / FFN / Conv"),
        (GREEN, "embed / mix / norm"),
        (RED, "attention"),
        (PURPLE, "JEPA stream"),
    ]
    for i, (c, name) in enumerate(legend):
        x = 0.25 + i * 2.75
        ax.add_patch(Rectangle((x, 0.18), 0.38, 0.28, facecolor=c, edgecolor=EDGE, lw=0.6))
        ax.text(x + 0.46, 0.32, name, ha="left", va="center", fontsize=5.8, color=EDGE)

    # ----- inputs -----
    box(
        ax,
        0.25,
        1.55,
        9.15,
        1.55,
        "$x$  $(B,\\,288,\\,4)$   glucose, basal rate, bolus, carbs\n"
        "288 steps = 24 h @ 5 min, ends at $t=$ now\n"
        "SugarOne trunk: $x[:,\\,-128:,:]$   "
        "JEPA: $x[:,\\,:,0]$  glucose only",
        BLUE,
        fontsize=6.5,
    )

    ch = [
        (0.25, "glucose\n$(B,128,1)$"),
        (2.55, "basal rate\n$(B,128,1)$"),
        (4.85, "bolus\n$(B,128,1)$"),
        (7.15, "carbs\n$(B,128,1)$"),
    ]
    for x, label in ch:
        box(ax, x, 3.25, 2.15, 0.85, label, BLUE, fontsize=6.3)
        arrow(ax, x + 1.07, 3.25 + 0.85, x + 1.07, 4.22)

    # four Linear(1->32)
    for x, _ in ch:
        box(ax, x, 4.25, 2.15, 0.72, "Linear$(1\\to 32)$\n$(B,128,32)$", YELLOW, fontsize=6.0)
        arrow(ax, x + 1.07, 4.97, 4.82, 5.28)

    box(
        ax,
        0.25,
        5.32,
        9.15,
        0.78,
        "sinusoidal positional encoding  (fixed buffer $(1,128,32)$)\n"
        "$g_e,\\,b_e,\\,bo_e,\\,c_e$  each $(B,128,32)$",
        GREEN,
        fontsize=6.3,
    )
    arrow(ax, 4.82, 6.10, 4.82, 6.42)

    # ----- parallel block -----
    box(ax, 0.15, 6.45, 9.35, 6.55, "", GRAY, dashed=True, lw=1.05)
    ax.text(
        4.82,
        12.78,
        "SugarJepa parallel block  $\\times 5$   (batch-first — no permutes)\n"
        "only the glucose stream is updated; auxiliaries are re-read unchanged",
        ha="center",
        va="top",
        fontsize=6.4,
        color=EDGE,
    )

    # multi-scale
    box(
        ax,
        0.32,
        6.62,
        4.42,
        5.55,
        "Multi-scale self-attention  (glucose only)\n"
        "8 heads, $d{=}32$\n\n"
        "DS$=1$:  self-attn  $(B,128,32)$\n"
        "DS$=2$:  AvgPool$(2)$ $\\to$ $(B,64,32)$\n"
        "         interpolate back to 128\n"
        "DS$=4$:  AvgPool$(4)$ $\\to$ $(B,32,32)$\n"
        "         interpolate back to 128\n\n"
        "high $+$ up2 $+$ up4\n"
        "FFN $32\\to 128\\to 32$  (LN2 residual)\n"
        "$\\mathrm{ms\\_out}$  $(B,128,32)$",
        RED,
        fontsize=5.9,
    )

    # cross-attention
    box(
        ax,
        4.90,
        6.62,
        4.42,
        5.55,
        "Cross-attention  (4 auxiliaries)\n"
        "$Q=$ glucose $(B,128,32)$\n\n"
        "MHA basal   $K/V$ $(B,128,32)$\n"
        "MHA bolus   $K/V$ $(B,128,32)$\n"
        "MHA carbs   $K/V$ $(B,128,32)$\n"
        "MHA jepa    $K/V$ $(B,36,32)$\n"
        "each 32-d, 8 heads\n\n"
        "$\\mathrm{merged}=\\sum_i \\mathrm{softmax}(\\mathrm{mix\\_logits})[i]\\,\\mathrm{res}_i$\n"
        "shared ln1;  FFN $32\\to 128\\to 32$\n"
        "$\\mathrm{cross\\_out}$  $(B,128,32)$",
        RED,
        fontsize=5.8,
    )

    # ----- JEPA column -----
    box(ax, 9.70, 3.25, 4.05, 9.75, "", PURPLE, lw=1.1)
    ax.text(
        11.72,
        12.78,
        "JEPA encoder",
        ha="center",
        va="top",
        fontsize=7.2,
        color=EDGE,
        fontweight="bold",
    )
    box(
        ax,
        9.90,
        11.55,
        3.65,
        0.85,
        "glucose $x[:,\\,:,0]$\n$(B,288)$",
        BLUE,
        fontsize=6.2,
    )
    arrow(ax, 11.72, 11.55, 11.72, 11.28)
    box(
        ax,
        9.90,
        10.40,
        3.65,
        0.82,
        "Instance $z$-score\n(per-window)",
        GREEN,
        fontsize=6.2,
    )
    arrow(ax, 11.72, 10.40, 11.72, 10.12)
    box(
        ax,
        9.90,
        8.95,
        3.65,
        1.10,
        "Conv1d patchify\n$1\\to 96$, $k{=}8$, $s{=}8$\n$(B,36,96)$",
        YELLOW,
        fontsize=6.2,
    )
    arrow(ax, 11.72, 8.95, 11.72, 8.68)
    box(
        ax,
        9.90,
        6.55,
        3.65,
        2.05,
        "JepaBlock $\\times 3$\n"
        "MHA self-attn\n$d{=}96$, 6 heads, norm1\n"
        "MLP $96\\to 384\\to 96$\nnorm2 residual",
        PURPLE,
        fontsize=6.1,
    )
    arrow(ax, 11.72, 6.55, 11.72, 6.28)
    box(
        ax,
        9.90,
        4.95,
        3.65,
        1.25,
        "LayerNorm$(96)$\n"
        "jepa_proj: Linear$(96\\to 32)$\n"
        "$(B,36,32)$  $K/V$",
        YELLOW,
        fontsize=6.1,
    )
    arrow(ax, 9.90, 5.55, 9.32, 8.6)
    ax.text(9.48, 7.15, "$(B,36,32)$", ha="center", va="center", fontsize=5.6, color=EDGE, rotation=90)

    # ln_fuse
    arrow(ax, 2.53, 6.62, 4.50, 13.22)
    arrow(ax, 7.11, 6.62, 5.15, 13.22)
    box(
        ax,
        2.55,
        13.15,
        4.55,
        0.72,
        "ln_fuse:  $\\mathrm{ms\\_out}+\\mathrm{cross\\_out}$  then LN\n$(B,128,32)$",
        GREEN,
        fontsize=6.2,
    )
    arrow(ax, 4.82, 13.87, 4.82, 14.12)

    # head
    box(
        ax,
        0.25,
        14.15,
        9.15,
        0.78,
        "flatten: permute $(B,32,128)$ $\\to$ $(B,4096)$\n"
        "flatten_fc  Linear$(4096\\to 32)$  + GELU + Dropout$(0.1)$",
        YELLOW,
        fontsize=6.3,
    )
    arrow(ax, 4.82, 14.93, 4.82, 15.18)
    box(
        ax,
        0.25,
        15.20,
        9.15,
        0.85,
        "out_fc: Linear$(32\\to 12)$     $\\hat{y}$  $(B,12)$\n"
        "12 steps $\\times$ 5 min  =  60-minute horizon",
        BLUE,
        fontsize=6.5,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    out = Path(__file__).with_name("sugar_jepa.png")
    fig.savefig(out, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
