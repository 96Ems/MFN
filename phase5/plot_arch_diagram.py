#!/usr/bin/env python
"""Figure 2 du papier : schémas comparatifs des architectures (style
boule/lien) — GRU vs MFN 1L vs MFN 3L vs MFN 3L3T. Sortie PDF vectorielle.
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine repo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

C_LAM, C_PSI = "#dd8452", "#4c72b0"   # flux rapide / lent
C_IN, C_OUT, C_FB = "#55a868", "#c44e52", "#8172b3"

fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6))

def ball(ax, x, y, r, color, edge="k", lw=1.5):
    return ax.add_patch(Circle((x, y), r, fc=color, ec=edge, lw=lw, zorder=5))

def arrow(ax, a, b, color="k", style="-", lw=1.4, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                 color=color, linestyle=style, lw=lw,
                 connectionstyle=f"arc3,rad={rad}", zorder=z))

def box(ax, x, y, w, h, text, fc="#ffffff", ec="k", fs=8.5, z=4):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=1.2, zorder=z)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=z + 1)

# ============ (a) GRU ============
ax = axs[0, 0]; ax.set_title("(a) GRU — baseline", fontsize=11)
ball(ax, 0.10, 0.55, 0.055, C_IN); ax.text(0.10, 0.40, "$x_t$", ha="center", fontsize=10)
box(ax, 0.46, 0.55, 0.50, 0.34, "GRU cell\n$h_t = GRU(h_{t-1}, x_t)$\nr, z gates", fs=8)
ball(ax, 0.83, 0.55, 0.055, C_OUT); ax.text(0.83, 0.40, "$y_t$", ha="center", fontsize=10)
arrow(ax, (0.155, 0.55), (0.21, 0.55))
arrow(ax, (0.71, 0.55), (0.775, 0.55))
# boucle récurrente h_{t-1} -> h_t
arrow(ax, (0.46, 0.28), (0.46, 0.38), color=C_FB, rad=0.0, style="-", lw=1.2)
ax.text(0.46, 0.24, "$h_{t-1}$ (loop)", ha="center", fontsize=7.5, color=C_FB)
ax.text(0.30, 0.86, "one hidden stream,\ninput-dependent gates", fontsize=8, ha="center",
        color="0.25", style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ============ (b) MFN 1L — dual-stream layer ============
ax = axs[0, 1]; ax.set_title("(b) MFN layer — dual streams + fatigue", fontsize=11)
ball(ax, 0.10, 0.50, 0.05, C_IN); ax.text(0.10, 0.36, "$x_t$", ha="center", fontsize=10)
# fatigue boxes
box(ax, 0.34, 0.815, 0.34, 0.13, "fatigue $\\varphi_{\\lambda}$", fc="#fdf1e5", fs=7.5)
box(ax, 0.34, 0.185, 0.34, 0.13, "fatigue $\\varphi_{\\psi}$", fc="#e8eef7", fs=7.5)
ball(ax, 0.34, 0.66, 0.075, C_LAM)          # lambda
ball(ax, 0.34, 0.34, 0.075, C_PSI)          # psi
ax.text(0.34, 0.78, "$h^{\\lambda}$ (fast)", ha="center", fontsize=8)
ax.text(0.34, 0.215, "$h^{\\psi}$ (slow)", ha="center", fontsize=8)
arrow(ax, (0.15, 0.50), (0.265, 0.62), color="k", lw=1.1)
arrow(ax, (0.15, 0.50), (0.265, 0.38), color="k", lw=1.1)
ax.text(0.205, 0.655, "$\\alpha_{\\lambda}$", fontsize=7.5, color="k")
ax.text(0.205, 0.32, "$\\alpha_{\\psi}^{\\beta}$", fontsize=7.5, color="k")
# couplages croisés
arrow(ax, (0.415, 0.60), (0.415, 0.40), color="0.45", style="-.", lw=1.2, rad=-0.25)
arrow(ax, (0.43, 0.40), (0.43, 0.60), color="0.45", style="-.", lw=1.2, rad=-0.25)
ax.text(0.495, 0.51, "$g_{\\psi\\to\\lambda},\\; g_{\\lambda\\to\\psi}$",
        fontsize=7.5, color="0.35")
# readout
ball(ax, 0.68, 0.50, 0.06, C_OUT)
arrow(ax, (0.415, 0.66), (0.62, 0.53), lw=1.1)
arrow(ax, (0.415, 0.34), (0.62, 0.47), lw=1.1)
ax.text(0.68, 0.63, "$y_l = W_{ro}[\\tilde h_{\\lambda}; \\tilde h_{\\psi}]$",
        fontsize=7.5, ha="center")
arrow(ax, (0.74, 0.50), (0.80, 0.50), lw=1.1)
ax.text(0.87, 0.60, "to next\nlayer", fontsize=7.5, ha="center", color="0.3")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ============ (c) MFN 3L ============
ax = axs[1, 0]; ax.set_title("(c) MFN 3L — inter-layer feedback", fontsize=11)
xs = 0.42
ys = [0.20, 0.50, 0.80]
for i, y in enumerate(ys):
    ball(ax, xs - 0.09, y + 0.06, 0.045, C_LAM)
    ball(ax, xs + 0.09, y - 0.06, 0.045, C_PSI)
    ax.text(xs, y + 0.22, f"layer {i+1}", ha="center", fontsize=8)
# feed-forward y_l -> u_{l+1}
for i in range(2):
    arrow(ax, (xs + 0.09, ys[i] - 0.06 - 0.045), (xs, ys[i + 1] - 0.105 - 0.045 + 0.045),
          lw=1.1)
# top-down (t-1)
for i in range(2):
    arrow(ax, (xs - 0.09, ys[i + 1] - 0.105 - 0.045), (xs - 0.09, ys[i] + 0.105 + 0.045),
          color=C_FB, style="--", lw=1.1)
ax.text(0.10, 0.50, "top-down\n$(t-1)$", fontsize=7.5, color=C_FB,
        ha="center", rotation=90)
# skip layer 1 -> layer 3
arrow(ax, (0.30, ys[0] + 0.06), (0.30, ys[2] - 0.06), color=C_FB, style=":", lw=1.6)
ax.text(0.185, 0.50, "skip\n$(t)$", fontsize=7.5, color=C_FB, ha="center", rotation=90)
ball(ax, xs, 0.94, 0.045, C_OUT); ax.text(xs, 0.99, "$y_t$", fontsize=9, ha="center")
arrow(ax, (xs + 0.09, ys[2] + 0.105 + 0.045), (xs + 0.045, 0.94 - 0.045), lw=1.1)
ball(ax, xs, 0.05, 0.045, C_IN); ax.text(xs, 0.005, "$x_t$", fontsize=9, ha="center")
arrow(ax, (xs, 0.095), (xs - 0.045, ys[0] - 0.105), lw=1.1)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.axis("off")

# ============ (d) MFN 3L3T ============
ax = axs[1, 1]; ax.set_title("(d) MFN 3L3T — threaded grid (width of streams)",
                             fontsize=11)
tx = [0.22, 0.50, 0.78]
ty = [0.20, 0.50, 0.80]
for k, x in enumerate(tx):
    ax.text(x, 0.965, ["thread A", "thread B", "thread C"][k],
            ha="center", fontsize=8)
    for y in ty:
        ball(ax, x - 0.055, y + 0.045, 0.032, C_LAM)
        ball(ax, x + 0.055, y - 0.045, 0.032, C_PSI)
    for i in range(2):
        arrow(ax, (x + 0.055, ty[i] - 0.075), (x, ty[i + 1] - 0.16 + 0.075),
              lw=0.9)
# latéral à chaque étage
for y in ty:
    for ka in range(2):
        arrow(ax, (tx[ka] + 0.075, y), (tx[ka + 1] - 0.075, y),
              color="0.4", style="-.", lw=1.2)
ax.text(0.50, 0.90, "lateral feedback (same-$t$, per-neuron gates)",
        fontsize=7.5, ha="center", color="0.35", style="italic")
ball(ax, 0.50, 0.035, 0.045, C_OUT); ax.text(0.50, 0.002, "concat $\\to$ head",
        fontsize=7.5, ha="center")
for x in tx:
    arrow(ax, (x + 0.055, ty[2] + 0.075), (0.50 - 0.045, 0.08), lw=0.9)
ball(ax, 0.22, 0.045, 0.032, C_OUT); ball(ax, 0.78, 0.045, 0.032, C_OUT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.axis("off")

fig.suptitle("Schematic comparison of the recurrent cores (one neuron ball = one "
             "state stream; arrows = learned projections with the noted gating)",
             fontsize=10.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(BASE, "papers", "arch_diagram.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(BASE, "papers", "arch_diagram.png"), dpi=180, bbox_inches="tight")
print("saved papers/arch_diagram.{pdf,png}")