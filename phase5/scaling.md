# scaling.md — Courbe de scaling MFN vs littérature (tracker)

**Règles** : mêmes données (22.6M tokens TinyStories-100K, `phase4/data_subset`),
même tokenizer (BPE-4096), seed 0. Loss = val loss en nats/token (ppl = e^loss).
Paramètres = comptage réel dédupliqué. Références citées en bas — aucun modèle
externe ré-entraîné.

## Points MFN mesurés

| Point | Config | Params | Epochs | Val loss | Ppl | BPC | Statut |
|---|---|---|---|---|---|---|---|
| P0 | mfn_dense H=144 (1 couche) | 1 153 440 | 4 @ 1e-3 | 2.4013 | 11.04 | 0.8813 | ✅ phase 4 |
| P1 | deep_2l [192,96] | 2 122 368 | 4 @ 5e-4 | 2.6022 | 13.49 | 0.9472 | ⚠️ sous-entraîné (bug + lr moitié) — invalide pour la pente |
| P2 | deep_3l [192,128,96] | 2 612 384 | 3 @ 5e-4 | 2.6582 | 14.27 | 0.9675 | ✅ test ppl 14.53 — invalide pour la pente (budget ≠ P0), info : bat 2l à iso-epoch |
| P3 | deep_2l [192,96] | 2 122 368 | 4 @ 1e-3 | — | — | — | ⏳ lane suivante (post-SFT) |
| P4 | deep_3l [192,128,96] | 2 612 384 | 4 @ 1e-3 | **2.4135** | **11.17** | **0.8785** | ✅ terminé 22/08 17h44 — TEST ppl **11.39**, bpc 0.8873. **Barre α_N falsifiée** : α_N = **−0.006** (total) / −0.004 (non-emb) vs barre 0.076 → 2.26 requis, obtenu 2.4135 (+0.15). Gains/ep décroissants ×0.5-0.6 (−0.23, −0.116, −0.072) ; asymptote estimée ≈ 2.30 > 2.26 même à epochs infinis sur ce corpus → signature **famine de données** (8.65 tok/param vs 19.6 pour le 1L), pas échec d'architecture. Le lr 1e-3 confirmé : chaque epoch bat le même epoch à 5e-4 (−0.22, −0.19, −0.17, −0.14) |

## Ancres locales (mêmes données, déjà entraînées — gratuites)

| Modèle | Params | Ppl test | BPC test | Source |
|---|---:|---:|---:|---|
| GRU + LN (H=215) | 1 159 710 | 11.58 | 0.8933 | phase4/results.json |
| GPT-2 mini (4×120) | 1 250 640 | 29.41 | 1.2333 | phase4/results.json |

## Droites de référence publiées (aucun entraînement)

1. **Kaplan et al. 2020** (WebText2, transformers) :
   - `L(N) ≈ (8.8×10¹³ / N)^0.076` (paramètres non-emb)
   - `L(D) ≈ (5.4×10¹³ / D)^0.095` (tokens)
   - `L(C) ≈ (1.6×10⁷ / C)^0.057` (PF-jours)
   - Résultat clé pour nous : les **LSTM égalent les transformers en début de
     contexte, décrochent plus loin** → le créneau « mémoire longue » de MFN.
2. **Hoffmann et al. 2022 (Chinchilla)** : à compute optimal, N ∝ C^0.5 et
   D ∝ C^0.5 (≈ 20 tokens/paramètre) ; 400+ modèles 70M–16B.
3. **Gu & Dao 2023 (Mamba)** : courbes comparatives récurrent vs transformer
   à ~1.4B/2.8B params — référence du « récurrent moderne qui scale ».
4. **Pearce & Song 2024 / Porian et al. 2024** : réconciliation Kaplan/Chinchilla ;
   mettent en garde sur la **courbure aux petites tailles** et le comptage
   emb-vs-non-emb → nos exposants à 0.5-8M params sont **indicatifs** (pente
   locale), pas une preuve globale.

## Analyse — 1ère ébauche (21 août, avant fin de deep_2l)

Voir `scaling_curve.png` (panneau A : axe taille ; panneau B : axe données).

### Axe données (exposants locaux α_D, mêmes modèles, même jeu)
| Modèle | Params | α_D (4 ep) | α_D (toutes ep) | Val ep4 |
|---|---:|---:|---:|---:|
| **MFN 1L** | 1 153 440 | **0.085** | 0.085 | 2.4013 |
| GRU 1L | 1 159 710 | 0.064 | 0.064 | 2.4351 |
| GPT-2 mini | 1 250 640 | 0.059 | 0.051 | 3.3585 |

→ **MFN tire le plus de chaque token** : α_D ~33% plus raide que GRU et ~65%
que GPT-2 mini (référence littérature : Kaplan α_D = 0.095). C'est le premier
signal chiffré de « barrière déplacée » — sur l'axe données.

### Axe taille (points convergés, mêmes 22.6M tokens)
| Modèle | Params | Val loss | Δ vs MFN |
|---|---:|---:|---:|
| **MFN 1L** | 1 153 440 | **2.4013** | — |
| GRU 1L | 1 159 710 | 2.4351 | +0.034 (≈ même N) |
| GPT-2 mini | 1 250 640 | 3.3585 | +0.96 (≈ même N) |
| deep_2l | 2 122 368 | (ep1 : 3.0371, en cours) | — |

### Prévision falsifiable — deep_3l @1e-3 (RÉSOLUE le 22/08, 17h44)

Depuis P0 (mfn 1L, 2.4013) avec la pente transformers α_N :
- α=0.057 (params totaux) → **2.29** ; α=0.076 → **2.26** ; α=0.095 → **2.22**
- α=0.076 en non-emb (ratio 3.24×) → **2.20**

**Issue : barre FALSIFIÉE.** Val ep4 = **2.4135** (ppl 11.17), TEST ppl 11.39.
- α_N mesuré (paire P0-P4) = **−0.006** total / **−0.004** non-emb — plat.
- La prédiction preregistrée a fait son travail : elle était falsifiable et falsifiée.
- Interprétation (déjà notée avant la fin du run) : à 22.6M tokens, le 3l voit
  8.65 tok/param vs 19.6 pour le 1L → sous-alimentation Chinchilla ×2.3 ; la
  capacité ajoutée ne peut pas s'exprimer. L'asymptote géométrique des gains
  (−0.23, −0.116, −0.072 ; ratio ~0.5-0.62) est ≈ 2.30 > 2.26 : plus d'epochs
  n'aurait pas suffi — il faut plus de tokens uniques (axe D).
- Effet lr confirmé indépendamment : 1e-3 bat 5e-4 à chaque epoch identique
  (−0.22 / −0.19 / −0.17 / −0.14).

### Suite
- Fit `log val_loss vs log params` complet quand P1 (deep_2l) et P2 (deep_3l)
  sortent : α_MFN vs 0.076.
- Ajustement compute : loss vs FLOPs (tok/s mesurés × params × steps) — déjà
  noté : MFN 5.99k tok/s vs GRU 40.8k (lourd en calcul par token, à pondérer).
- Conclusion prudente (courbure petite échelle) ; preuve définitive >100M
  params = cluster.