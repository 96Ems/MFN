"""Render a neuron-level representation of the dense MFN network (PNG + SVG).

Shown: 6 neurons per layer as representative of H = 144 (noted). Orange =
expressive stream Lambda, green = conceptual stream Psi. Each hidden neuron
carries: per-neuron decay alpha (input-conditioned), a fatigue accumulator phi
(multiplicative damping), a recurrent GRU self-loop, and cross-stream inputs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

H_VIS = 6       # neurons shown per layer (representative of H=144)
R = 0.055       # neuron radius
X_IN, X_LAM, X_PSI, X_OUT = 0.06, 0.40, 0.60, 0.845
Y0 = 0.12
YSTEP = (0.98 - 2 * Y0) / (H_VIS - 1)

fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def ypos(i):
    return 1.0 - Y0 - i * YSTEP


def neuron(ax, x, y, color, label, fs=8):
    c = Circle((x, y), R, facecolor=color, edgecolor="black", lw=1.2, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=6)


def edge(ax, x1, y1, x2, y2, color="#999999", lw=0.7, style="-", rad=0.0):
    e = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-", color=color,
                        lw=lw, linestyle=style, connectionstyle=f"arc3,rad={rad}",
                        zorder=2)
    ax.add_patch(e)


# ---------------------------------------------------------------- inputs ----
for i in range(H_VIS):
    neuron(ax, X_IN, ypos(i), "#e8f0fe", r"$e_{%d}$" % (i + 1))
ax.text(X_IN, 0.045, "embedding  e_t  (H=144)", ha="center", fontsize=9)

for i in range(H_VIS):
    for xh in (X_LAM, X_PSI):
        edge(ax, X_IN + R, ypos(i), xh - R, ypos(i), color="#aac8ff", lw=0.5,
             rad=0.0)
ax.text((X_IN + X_LAM) / 2, 1.02, "W_in  (H×H)", ha="center", fontsize=9)

# ------------------------------------------------------ lambda (expressif) ---
lam = ["#f8c98a", "#f6b96a", "#f4a94d", "#f29832", "#ef8a1a", "#e67f00"]
for i in range(H_VIS):
    x, y = X_LAM, ypos(i)
    neuron(ax, x, y, lam[i], r"$h^\Lambda_{%d}$" % (i + 1), fs=7)
    # recurrent self loop (GRUCell)
    edge(ax, x + R * 0.9, y + 0.018, x + R * 0.45, y + 0.095, color="#b06a00",
         lw=1.0, rad=0.35)
# alpha + fatigue labels above/below each neuron
for i in range(H_VIS):
    x, y = X_LAM, ypos(i)
    ax.text(x - 0.055, y + 0.075, r"$\alpha_\Lambda^{(i)}$", fontsize=7,
            color="#7a4d00", ha="center")
    c = Circle((x - 0.048, y - 0.075), 0.028, facecolor="white",
               edgecolor="#b06a00", lw=0.8, zorder=5)
    ax.add_patch(c)
    ax.text(x - 0.048, y - 0.075, r"$\phi$", fontsize=6, ha="center",
            va="center", zorder=6)
ax.text(X_LAM, 1.03, "flux expressif  Λ  (GRUCell, 12H²)", ha="center",
        fontsize=10, color="#7a4d00", fontweight="bold")

# --------------------------------------------------------- psi (conceptuel) ---
psi = ["#a7dcab", "#8fd295", "#77c87e", "#5fbe69", "#47b455", "#2faa42"]
for i in range(H_VIS):
    x, y = X_PSI, ypos(i)
    neuron(ax, x, y, psi[i], r"$h^{\Psi}_{%d}$" % (i + 1), fs=7)
    edge(ax, x - R * 0.9, y + 0.018, x - R * 0.45, y + 0.095, color="#1f7a2e",
         lw=1.0, rad=-0.35)
for i in range(H_VIS):
    x, y = X_PSI, ypos(i)
    ax.text(x + 0.055, y + 0.075, r"$\alpha_\Psi^{(i)}$", fontsize=7,
            color="#1f5a28", ha="center")
    c = Circle((x + 0.048, y - 0.075), 0.028, facecolor="white",
               edgecolor="#1f7a2e", lw=0.8, zorder=5)
    ax.add_patch(c)
    ax.text(x + 0.048, y - 0.075, r"$\phi$", fontsize=6, ha="center",
            va="center", zorder=6)
ax.text(X_PSI, 1.03, "flux conceptuel  Ψ  (GRUCell, 12H², α^β lent)",
        ha="center", fontsize=10, color="#1f5a28", fontweight="bold")

# ------------------------------------------------- cross-stream (W + gates) ---
# representative arcs: h_lambda(i) -> psi(j+1) and h_psi(i) -> lambda(j+1)
for i in range(H_VIS - 1):
    edge(ax, X_LAM + R, ypos(i), X_PSI - R, ypos(i + 1),
         color="#2faa42", lw=1.0, rad=0.22)
    edge(ax, X_PSI - R, ypos(i), X_LAM + R, ypos(i + 1),
         color="#e67f00", lw=1.0, rad=-0.22)
ax.text((X_LAM + X_PSI) / 2, 1.03, "W_Λ→Ψ  /  W_Ψ→Λ   (2H²)   +   portes g (4H²)",
        ha="center", fontsize=9)
# gates: small boxes between the two streams, in the gaps between neuron rows
for i in (0, 2, 4):
    gx, gy = 0.5, (ypos(i) + ypos(i + 1)) / 2
    g = plt.Rectangle((gx - 0.06, gy - 0.025), 0.12, 0.05, facecolor="#f0f0f0",
                      edgecolor="black", lw=0.9, zorder=4)
    ax.add_patch(g)
    ax.text(gx, gy, r"$g_{\Psi\to\Lambda}$" if i == 0 else
            (r"$g_{\Lambda\to\Psi}$" if i == 2 else r"$g$"),
            fontsize=6, ha="center", va="center", zorder=5)

# ---------------------------------------------------------------- output ----
for i in range(min(H_VIS, 5)):
    neuron(ax, X_OUT, ypos(i) + 0.02, "#f7f0ff", r"$y_{%d}$" % (i + 1), fs=7)
for i in range(min(H_VIS, 5)):
    for xh in (X_LAM, X_PSI):
        edge(ax, xh + R, ypos(i), X_OUT - R, ypos(i) + 0.02,
             color="#c9a0e8", lw=0.5)
ax.text(X_OUT, 1.03, "readout 2H→H  +  LN  →  head V (liée)  →  softmax",
        ha="center", fontsize=9, color="#6b3fa0")

# -------------------------------------------- annotations structurales
ax.text(0.5, -0.045, "poids partagés / temps : état (h, φ) propagé de t−1 à t ; "
        "décroissance α conditionnée par e_t, par neurone ; "
        "fatigue φ = Σ γφ+(1−γ)|h| → h_eff = h(1−φ) (rotation d'activité)",
        ha="center", fontsize=9, color="#444444")
ax.text(0.5, -0.095,
        "ordre : Λ mis à jour d'abord, puis Ψ lit h_Λ frais. "
        "Budget ≈ 22H² + (3d+2d_out+23)H + d_out  (H=144, V=4096)",
        ha="center", fontsize=9, color="#444444")

plt.tight_layout()
plt.savefig("MFN_neurons.png", dpi=200, bbox_inches="tight")
plt.savefig("MFN_neurons.svg", bbox_inches="tight")
print("saved MFN_neurons.png / .svg")