#!/usr/bin/env python
"""Figure 1 du papier : schémas comparatifs (GRU / MFN 1L / 3L / 3L3T).
Design propre : grille 10x7.8 par panneau, aucune rotation de texte,
légende globale, couleurs sémantiques (vert=entrée, orange=λ, bleu=ψ,
rouge=sortie, violet pointillé=top-down, gris point-tiret=latéral)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_IN, C_LAM, C_PSI, C_OUT = "#55a868", "#dd8452", "#4c72b0", "#c44e52"
C_TD, C_LAT = "#8172b3", "0.45"

def ball(ax, x, y, r, color, text=None, fs=9):
    ax.add_patch(Circle((x, y), r, fc=color, ec="k", lw=1.4, zorder=5))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=6)

def arrow(ax, a, b, color="k", style="-", lw=1.4, rad=0.0, ms=12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                 color=color, linestyle=style, lw=lw, zorder=3,
                 connectionstyle=f"arc3,rad={rad}"))

def box(ax, x, y, w, h, text, fc="#ffffff", ec="k", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.3, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5,
            fontweight="bold" if bold else "normal")

def newpanel(ax, title):
    ax.set_title(title, fontsize=11.5, pad=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.8)
    ax.set_aspect("equal"); ax.axis("off")
    return ax

fig, axs = plt.subplots(2, 2, figsize=(12.7, 11.0))

# ---------------- (a) GRU ----------------
ax = newpanel(axs[0, 0], "(a) GRU — one stream, input-dependent gates")
ball(ax, 1.4, 3.9, 0.5, C_IN, "$x_t$")
box(ax, 5.0, 3.9, 4.3, 4.0,
    r"GRU cell" "\n\n" r"$h_t = \mathrm{GRU}(h_{t-1}, x_t)$" "\n\n"
    "r, z gates\none hidden stream", fs=9)
ball(ax, 8.75, 3.9, 0.5, C_OUT, "$y_t$")
ax.text(1.4, 2.85, "input", ha="center", fontsize=8, color="0.3")
ax.text(8.75, 2.85, "output", ha="center", fontsize=8, color="0.3")
arrow(ax, (1.9, 3.9), (2.85, 3.9))
arrow(ax, (7.15, 3.9), (8.25, 3.9))
arrow(ax, (4.1, 1.9), (5.9, 1.9), color=C_TD, style="--", lw=1.8, rad=-0.5)
ax.text(5.0, 0.62, "$h_{t-1}$  (recurrence)", ha="center", fontsize=8.5,
        color=C_TD)

# ---------------- (b) MFN 1L ----------------
ax = newpanel(axs[0, 1], "(b) MFN layer — dual streams + fatigue")
ball(ax, 1.3, 3.9, 0.5, C_IN, "$x_t$")
ax.text(1.3, 2.9, "input", ha="center", fontsize=8, color="0.3")
ball(ax, 4.2, 5.0, 0.62, C_LAM, "$\\lambda$")
ball(ax, 4.2, 2.8, 0.62, C_PSI, "$\\psi$")
ax.text(5.1, 5.0, "fast\nstream", fontsize=7.5, ha="left", va="center")
ax.text(5.1, 2.8, "slow\nstream", fontsize=7.5, ha="left", va="center")
box(ax, 4.2, 6.45, 3.4, 0.72, "fatigue $\\varphi_{\\lambda}$", fc="#fdf1e5", fs=8)
box(ax, 4.2, 1.35, 3.4, 0.72, "fatigue $\\varphi_{\\psi}$", fc="#e8eef7", fs=8)
arrow(ax, (4.2, 6.09), (4.2, 5.65), lw=1.1)
arrow(ax, (4.2, 1.71), (4.2, 2.15), lw=1.1)
arrow(ax, (1.8, 4.05), (3.6, 4.72))
arrow(ax, (1.8, 3.75), (3.6, 3.08))
ax.text(2.35, 4.98, "$\\alpha_{\\lambda}$", fontsize=8.5, ha="center")
ax.text(2.35, 2.75, "$\\alpha_{\\psi}^{\\beta}$", fontsize=8.5, ha="center")
arrow(ax, (4.95, 4.42), (4.95, 3.38), color=C_LAT, style="-.", lw=1.4, rad=-0.4, ms=10)
arrow(ax, (5.30, 3.38), (5.30, 4.42), color=C_LAT, style="-.", lw=1.4, rad=-0.4, ms=10)
ax.text(5.65, 2.05, "$g_{\\psi\\to\\lambda}, g_{\\lambda\\to\\psi}$",
        fontsize=7.5, ha="center", color="0.3")
ball(ax, 8.35, 3.9, 0.55, C_OUT, "$y_l$")
ax.text(8.35, 2.9, "$y_l = W_{ro}[\\tilde h_{\\lambda}; \\tilde h_{\\psi}]$",
        ha="center", fontsize=7.5)
arrow(ax, (4.82, 4.9), (7.8, 4.15))
arrow(ax, (4.82, 2.9), (7.8, 3.65))
arrow(ax, (8.9, 3.9), (9.55, 3.9), lw=1.1)
ax.text(9.45, 4.35, "next\nlayer", fontsize=7.5, ha="center", color="0.3")

# ---------------- (c) MFN 3L ----------------
ax = newpanel(axs[1, 0], "(c) MFN 3L — inter-layer feedback")
ball(ax, 5.0, 1.1, 0.45, C_IN, "$x_t$")
ys = [2.7, 4.5, 6.3]
for i, y in enumerate(ys):
    ball(ax, 4.5, y, 0.4, C_LAM)
    ball(ax, 5.5, y, 0.4, C_PSI)
    ax.text(4.5, y, "$\\lambda$", fontsize=7.5, color="white",
            ha="center", va="center", fontweight="bold")
    ax.text(5.5, y, "$\\psi$", fontsize=7.5, color="white",
            ha="center", va="center", fontweight="bold")
    ax.text(3.45, y, f"L{i+1}", fontsize=9, ha="center", va="center")
arrow(ax, (5.0, 1.55), (5.0, 2.28))
arrow(ax, (5.0, 3.12), (5.0, 4.08))
arrow(ax, (5.0, 4.92), (5.0, 5.88))
ball(ax, 5.0, 7.15, 0.45, C_OUT, "$y_t$")
arrow(ax, (5.0, 6.72), (5.0, 6.68))
# top-down (de plus profond -> plus superficiel), côté gauche
arrow(ax, (2.5, 6.55), (2.5, 4.95), color=C_TD, style="--", lw=1.4)
arrow(ax, (2.5, 4.55), (2.5, 2.95), color=C_TD, style="--", lw=1.4)
ax.text(1.35, 5.55, "top-down", fontsize=8.5, color=C_TD, ha="center")
ax.text(1.35, 5.05, "($t-1$)", fontsize=8, color=C_TD, ha="center")
# skip L1 -> L3, côté droit
arrow(ax, (7.5, 3.25), (7.5, 5.85), color="0.35", style=":", lw=1.8)
ax.text(8.45, 4.5, "skip\n($t$)", fontsize=8.5, color="0.35", ha="left",
        va="center")

# ---------------- (d) MFN 3L3T ----------------
ax = newpanel(axs[1, 1], "(d) MFN 3L3T — threaded grid (width of streams)")
cols = [2.5, 5.5, 8.5]
for k, x in enumerate(cols):
    box(ax, x, 4.75, 2.5, 4.55, f"stack {chr(65+k)}", fc="#f7f7f7", fs=9)
    for y in (3.6, 4.75, 5.9):
        ball(ax, x, y, 0.36, "#8da0cb")
arrow(ax, (cols[0], 4.0), (cols[0], 4.4), lw=1.0)
arrow(ax, (cols[0], 5.15), (cols[0], 5.55), lw=1.0)
arrow(ax, (cols[1], 4.0), (cols[1], 4.4), lw=1.0)
arrow(ax, (cols[1], 5.15), (cols[1], 5.55), lw=1.0)
arrow(ax, (cols[2], 4.0), (cols[2], 4.4), lw=1.0)
arrow(ax, (cols[2], 5.15), (cols[2], 5.55), lw=1.0)
# input partagé
ball(ax, 5.5, 1.05, 0.42, C_IN, "$x_t$")
ax.plot([2.5, 8.5], [1.75, 1.75], color="k", lw=1.2, zorder=2)
arrow(ax, (5.5, 1.47), (5.5, 1.75), lw=1.0)
arrow(ax, (2.5, 1.75), (2.5, 2.48), lw=1.0)
arrow(ax, (8.5, 1.75), (8.5, 2.48), lw=1.0)
ax.text(5.5, 0.45, "shared embedding", fontsize=7.5, ha="center", color="0.3")
# latéral à chaque étage
for y in (3.6, 4.75, 5.9):
    for a, b in ((3.72, 4.28), (6.72, 7.28)):
        arrow(ax, (a, y), (b, y), color=C_LAT, style="-.", lw=1.3, ms=9)
        arrow(ax, (b, y), (a, y), color=C_LAT, style="-.", lw=1.3, ms=9)
ax.text(5.5, 6.98, "lateral feedback (same $t$)", fontsize=7.5, ha="center",
        color="0.35")
# sortie concaténée
arrow(ax, (2.5, 6.55), (5.02, 7.02), lw=1.0)
arrow(ax, (5.5, 6.55), (5.5, 7.02), lw=1.0)
arrow(ax, (8.5, 6.55), (5.98, 7.02), lw=1.0)
ball(ax, 5.5, 7.35, 0.45, C_OUT, "$y$")
ax.text(6.45, 7.55, "concat $\\to$ head", fontsize=7.5, ha="left", va="center")

# ---------------- légende globale ----------------
from matplotlib.lines import Line2D
handles = [
    Circle((0, 0), 0.5, fc=C_IN, ec="k"), Circle((0, 0), 0.5, fc=C_LAM, ec="k"),
    Circle((0, 0), 0.5, fc=C_PSI, ec="k"), Circle((0, 0), 0.5, fc=C_OUT, ec="k"),
    Line2D([0], [0], color="k", lw=1.4), Line2D([0], [0], color=C_TD, ls="--", lw=1.4),
    Line2D([0], [0], color="0.35", ls=":", lw=1.8), Line2D([0], [0], color=C_LAT, ls="-.", lw=1.4),
]
labels = ["input $x_t$", "fast stream $\\lambda$", "slow stream $\\psi$",
          "output $y_t$ / head", "data flow", "top-down $t-1$", "skip $t$",
          "lateral (same $t$)", ]
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Schematic comparison of the recurrent cores "
             "(one ball = one state stream; arrows = learned projections)",
             fontsize=10.5, y=0.985)
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(BASE, "papers", f"arch_diagram.{ext}"),
                bbox_inches="tight", dpi=180 if ext == "png" else None)
print("saved papers/arch_diagram.{pdf,png}")