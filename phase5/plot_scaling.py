#!/usr/bin/env python
"""Première courbe de scaling MFN vs littérature — points mesurés uniquement.

Tous les points : mêmes données (phase4/data_subset, 22.6M tokens BPE-4096),
seed 0. Loss = val loss en nats/token. Aucun modèle externe ré-entraîné.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 1e6
TOK_PER_EP = 22.6 * M

# ---- Points "taille" (best val loss, convergés) — même jeu de données ----
size_points = [
    # (nom, N total, L val, couleur, marqueur)
    ("GRU 1L (H=215)", 1.159710 * M, 2.4351, "#4c72b0", "s"),
    ("MFN 1L (H=144)", 1.153440 * M, 2.4013, "#dd8452", "o"),
    ("GPT-2 mini 4L",  1.250640 * M, 3.3585, "#55a868", "^"),
    ("deep_2l (en cours, ep1)", 2.122368 * M, 3.0371, "#dd8452", "D"),
]

# ---- Trajectoires par epoch (axe données) ----
traj = {
    "MFN 1L":  (1.153440 * M, [2.6988, 2.5269, 2.4477, 2.4013], "#dd8452", "o"),
    "GRU 1L":  (1.159710 * M, [2.6583, 2.5280, 2.4708, 2.4351], "#4c72b0", "s"),
    "GPT-2 mini": (1.250640 * M, [3.6818, 3.4976, 3.4334, 3.3968, 3.3741, 3.3585],
                   "#55a868", "^"),
}

# ---- Exposants locaux ----
print("=== Exposants locaux L(D) (fit log-log, 22.6M ->", end=" ")
for name, (N, losses, c, mk) in traj.items():
    D = TOK_PER_EP * np.arange(1, len(losses) + 1)
    a, b = np.polyfit(np.log(D), np.log(losses), 1)
    # exposant moyen sur les 4 premières époques (iso-budget)
    D4 = TOK_PER_EP * np.arange(1, 5)
    a4, _ = np.polyfit(np.log(D4), np.log(losses[:4]), 1)
    print(f"{name}: alpha_D ~ {abs(a4):.3f} (4ep) / {abs(a):.3f} (toutes)", end="  ")
print(")")

# ---- Prévision deep_2l (même protocole, ep4) ----
N0, L0 = 1.153440 * M, 2.4013
N1 = 2.122368 * M
print("\n=== Prévision deep_2l ep4 (depuis P0 = mfn 1L) ===")
for a in (0.057, 0.076, 0.095):
    Lpred = L0 * (N0 / N1) ** a
    print(f"  alpha_N={a:.3f} (params totaux): val ~ {Lpred:.3f}")
# version non-emb (style Kaplan / réconciliation 2024)
emb0, emb1 = 4096 * 144, 4096 * 192
ne0, ne1 = N0 - emb0, N1 - emb1
for a in (0.076, 0.095):
    print(f"  alpha_N={a:.3f} (non-emb; ratio {ne1/ne0:.2f}x): val ~ "
          f"{L0 * (ne0/ne1) ** a:.3f}")

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=110)

# Panneau A : loss vs params
for name, N, L, c, mk in size_points:
    if "en cours" in name:
        ax1.scatter(N, L, marker=mk, s=90, facecolor="none", edgecolor=c, zorder=5)
        ax1.annotate("deep_2l ep1 (en cours)", (N, L), textcoords="offset points",
                     xytext=(6, 10), fontsize=9, color=c)
    else:
        ax1.scatter(N, L, marker=mk, s=90, color=c, zorder=5, label=name)
# droites de référence (pente littérature) ancrées sur P0
Ns = np.linspace(N0 * 0.9, 2.5 * M, 200)
for a, lab in ((0.057, r"$\alpha=0.057$ (Kaplan, bas)"),
               (0.076, r"$\alpha=0.076$ (Kaplan)"),
               (0.095, r"$\alpha=0.095$ (Chinchilla)")):
    ax1.plot(Ns, L0 * (N0 / Ns) ** a, "--", lw=1.2,
             color="grey" if a != 0.076 else "0.25",
             label=lab if a in (0.076, 0.095) else None)
ax1.axvspan(2.0 * M, 2.5 * M, color="#dd8452", alpha=0.08)
ax1.annotate("bande cible deep_2l ep4 :\n2.27–2.32 (α=0.076, emb/non-emb)",
             (2.12 * M, 2.62), fontsize=9, color="#8c4a1b", ha="center",
             bbox=dict(fc="#fdf3ea", ec="#dd8452", lw=0.8))
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Paramètres (log)")
ax1.set_ylabel("Val loss (nats/token, log)")
ax1.set_title("A — Axe taille (mêmes 22.6M tokens)")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, which="both", alpha=0.3)

# Panneau B : loss vs données vues
for name, (N, losses, c, mk) in traj.items():
    D = TOK_PER_EP * np.arange(1, len(losses) + 1)
    ax2.plot(D, losses, marker=mk, color=c, lw=1.6, label=name)
    a4, _ = np.polyfit(np.log(TOK_PER_EP * np.arange(1, 5)),
                       np.log(losses[:4]), 1)
    ax2.annotate(f"α_D ≈ {abs(a4):.3f}", (D[1], losses[1]), textcoords="offset points",
                 xytext=(6, -14), fontsize=9, color=c)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("Tokens vus (log)")
ax2.set_ylabel("Val loss (log)")
ax2.set_title("B — Axe données (mêmes modèles)")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)

fig.suptitle("Scaling MFN — 1ère ébauche (mêmes données/tokenizer/seed ; littérature : Kaplan/Chinchilla)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("phase5/scaling_curve.png", bbox_inches="tight")
print("\nFigure sauvegardée : phase5/scaling_curve.png")