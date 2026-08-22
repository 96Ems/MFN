#!/usr/bin/env python
"""Figure 2 du papier : mécanisme du feedback — (a) inter-couches
(top-down t-1 + skip t, déroulé sur 2 pas de temps), (b) inter-threads
(latéral au même pas t, entre 3 threads). Sortie PDF vectorielle."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_IN, C_OUT, C_TD, C_SK, C_LAT = "#55a868", "#c44e52", "#8172b3", "0.35", "0.45"

def ball(ax, x, y, r, color, text=None, fs=9):
    ax.add_patch(Circle((x, y), r, fc=color, ec="k", lw=1.4, zorder=5))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=6)

def arrow(ax, a, b, color="k", style="-", lw=1.4, rad=0.0, ms=12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                 color=color, linestyle=style, lw=lw, zorder=3,
                 connectionstyle=f"arc3,rad={rad}"))

def box(ax, x, y, w, h, text, fc="#ffffff", ec="k", fs=8.5, lw=1.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)

def newpanel(ax, title):
    ax.set_title(title, fontsize=11.5, pad=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.8)
    ax.set_aspect("equal"); ax.axis("off")
    return ax

fig, axs = plt.subplots(1, 2, figsize=(12.7, 6.4))

# ============ (a) inter-layer : t-1 -> t ============
ax = newpanel(axs[0], "(a) Inter-layer feedback (top-down $t-1$ + skip $t$)")
xs = [2.6, 7.3]
ys = [2.6, 4.5, 6.4]
for c, x in enumerate(xs):
    for r, y in enumerate(ys):
        box(ax, x, y, 2.3, 1.35,
            f"L{r+1}\n({'$t-1$' if c == 0 else '$t$'})", fs=8.5)
ax.text(2.6, 7.32, "$t-1$", fontsize=10, ha="center", color="0.25")
ax.text(7.3, 7.32, "$t$", fontsize=10, ha="center", color="0.25")
arrow(ax, (4.35, 7.32), (5.95, 7.32), lw=1.2, color="0.25")
ax.text(5.15, 7.62, "time", fontsize=8, ha="center", color="0.25")
# feed-forward dans chaque colonne (y_l -> u_{l+1})
for x in xs:
    arrow(ax, (x, 3.28), (x, 3.82), lw=1.1)
    arrow(ax, (x, 5.18), (x, 5.72), lw=1.1)
ax.text(0.82, 5.55, "feed-forward\n$y_l \\to u_{l+1}$", fontsize=7.5,
        ha="center", color="0.3")
# top-down (t-1) : profond -> superficiel, en diagonale
arrow(ax, (3.75, 4.80), (6.15, 3.20), color=C_TD, style="--", lw=1.6)
arrow(ax, (3.75, 6.70), (6.15, 5.10), color=C_TD, style="--", lw=1.6)
ax.text(4.85, 6.85, "top-down:\n$\\mathrm{td} = W_{td}\\,y_{l+1}(t-1)$",
        fontsize=7.5, ha="center", color=C_TD)
# skip (t) : surface fraîche -> couches profondes (hors des boîtes)
arrow(ax, (5.90, 3.35), (5.90, 5.70), color=C_SK, style=":", lw=2.0)
ax.text(5.45, 2.45, "skip:\n$\\mathrm{sk} = W_{sk}\\,y_0(t)$",
        fontsize=7.5, ha="center", color=C_SK)
# formule de gate
ax.text(5.5, 0.72, "per-neuron gate: $g = \\sigma(u \\odot \\tilde h + "
        "v \\odot \\mathrm{fb} + b)$\n$m_{\\lambda} \\leftarrow m_{\\lambda} "
        "+ g \\odot \\mathrm{fb}$",
        fontsize=8, ha="center", color="0.15")
ball(ax, 2.6, 1.35, 0.38, C_IN, "$x_t$")
arrow(ax, (2.6, 1.73), (2.6, 1.90), lw=1.0)
ball(ax, 7.3, 7.45, 0.34, C_OUT, "$y$")
arrow(ax, (7.3, 7.08), (7.3, 7.12), lw=1.0)

# ============ (b) inter-thread ============
ax = newpanel(axs[1], "(b) Inter-thread lateral feedback (same step $t$)")
tx = [2.1, 5.0, 7.9]
for k, x in enumerate(tx):
    box(ax, x, 4.4, 2.3, 3.0, "", fs=8.5)
    ax.text(x, 5.55, f"thread {chr(65+k)}", fontsize=8.5, ha="center", zorder=6)
    ball(ax, x, 4.4, 0.40, C_OUT)
    ax.text(x, 3.35, "readout $y_l^{(k)}$", fontsize=7.5, ha="center", zorder=6)
for a, b in ((3.30, 3.82), (6.20, 6.72)):        # A->B puis B->C (vers le haut)
    arrow(ax, (a, 4.95), (b, 4.95), color=C_LAT, style="-.", lw=1.5, ms=10,
          rad=-0.25)
    arrow(ax, (b, 3.90), (a, 3.90), color=C_LAT, style="-.", lw=1.5, ms=10,
          rad=-0.25)
ax.text(5.0, 6.55, "$\\mathrm{lat}_{kj} = \\sigma(u_{kj} \\odot y_k + "
        "v_{kj} \\odot y_j + b_{kj}) \\odot W_{kj}(y_j)$\n"
        "$y_l^{(k)} \\leftarrow y_l^{(k)} + \\sum_{j \\neq k} "
        "\\mathrm{lat}_{kj}$", fontsize=7.5, ha="center", color="0.15")
ax.text(5.0, 1.05, "applied at every layer $l$, after the readouts,\n"
        "before feeding layer $l+1$ (and the head)",
        fontsize=7.5, ha="center", color="0.3")

handles = [
    Line2D([0], [0], color=C_TD, ls="--", lw=1.6),
    Line2D([0], [0], color=C_SK, ls=":", lw=2.0),
    Line2D([0], [0], color=C_LAT, ls="-.", lw=1.5),
]
labels = ["top-down $t-1$ (deep $\\to$ shallow)", "skip $t$ (surface $\\to$ deep)",
          "lateral same-$t$ (thread $\\to$ thread)"]
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(BASE, "papers", f"feedback_mech.{ext}"),
                bbox_inches="tight", dpi=180 if ext == "png" else None)
print("saved papers/feedback_mech.{pdf,png}")