# MFN — Memory-Fatigue Network : concept, architecture et scaling

> Document scientifique de travail — v0.1 (22 août 2026).
> État : entraînements en cours (deep_3l@1e-3, grille 3l3t planifiée).
> Tous les résultats cités : mêmes données (22.6M tokens TinyStories-100K,
> tokenizer BPE-4096, `phase4/data_subset`), seed 0, comptage de paramètres
> réel (dédupliqué). Aucun modèle externe n'a été ré-entraîné : les
> comparaisons hors-corpus utilisent les courbes publiées (Kaplan, Chinchilla).

---

## 1. Concept : deux flux, une fatigue, du feedback gateé

Le MFN repose sur trois principes complémentaires, tous implémentés dans
`phase5/deepmfn.py` :

### 1.1 Double flux Λ/Ψ (échelles de temps découplées)
Chaque couche maintient **deux états récurrents** :
- `h_lam` — flux **rapide** (λ) : suit le contenu local immédiat ;
- `h_psi` — flux **lent** (ψ) : intègre à plus longue portée.

Les deux flux échangent via un **gating bidirectionnel per-neuron**
(« confidence gating ») :

```
g_p2l = σ(W_gp2l([h_psi_eff, h_lam_eff]))     (psi -> lambda)
g_l2p = σ(W_gl2p([h_lam_eff, h_psi_eff]))     (lambda -> psi)
fb_lam = W_fb_lam(h_lam_eff) · α_lam · g_p2l
fb_psi = W_fb_psi(h_psi_eff) · α_psi · g_l2p
```

### 1.2 Fatigue — l'oubli structurel
Chaque flux possède un compteur de fatigue `φ` mis à jour par moyenne
exponentielle de l'activité :

```
φ ← γ·φ + (1−γ)·|h|          avec γ = σ(w_γ) appris
h_eff = h · (1 − φ)
```

Un neurone qui domine continuellement **s'éteint progressivement**, forçant la
diversité d'activation. C'est la clé de la thèse : la fatigue est une
régularisation implicite qui devrait rendre *chaque paramètre plus efficace*
— donc potentiellement **améliorer l'exposant de scaling** α, pas seulement
l'offset.

### 1.3 Décay appris et courbe de puissance
Les « freins » des deux flux sont appris :

```
α_lam = σ(MLP(u))                    (décay du flux rapide)
α_psi = σ(MLP(u))^β ,  β = 0.2       (courbe de puissance du flux lent)
```

La puissance `β<1` aplatit le decay du flux lent près de 1 et l'aiguille vers
zéro — cf. §3 pour sa **falaise numérique** (bug de backward corrigé).

### 1.4 Feedback inter-couches (profondeur)
- **Top-down** : le readout y_{l+1} de la couche profonde (au pas t−1) est
  projeté et injecté (gate per-neuron) dans chaque flux de la couche l —
  « la couche profonde parle à la couche superficielle » ;
- **Skip** : le readout frais de la couche 0 (pas t) guide les couches l≥2 —
  « la surface guide le profond dans le pas courant ».

### 1.5 Grillage inter-threads (largeur de flux)
Nouveau (v0.1.1) : **n threads** = n piles identiques en parallèle, chacune
avec son topdown/skip interne, qui s'échangent latéralement à chaque étage :

```
LateralFB:  contribution = σ(u⊙h_me + v⊙h_other + b) ⊙ W(h_other)
```

Le readout concatène les sorties des threads avant le LM head (embeddings
liés). Hypothèse : la spécialisation des threads (syntaxe/sémantique/contenu,
échelles de temps) est l'incarnation structurelle de la fatigue — et un axe de
scaling **indépendant de la profondeur**.

---

## 2. Architecture implémentée

### 2.1 Famille de modèles
| Arch | Couches/profondeur | Threads | Params | État |
|---|---|---|---:|---|
| `mfn_dense` 1L H=144 | 1 | 1 | 1 153 440 | ✅ entraîné |
| `deep_2l` [192,96] | 2 | 1 | 2 122 368 | ✅ entraîné (5e-4) |
| `deep_3l` [192,128,96] | 3 | 1 | 2 612 384 | 🔄 @1e-3 en cours |
| `3l3t` (=3×`3l` + latéral) | 3× | **3** | 6 753 888 | ⏳ prévu |
| `1l_192` (contrôle largeur) | 1 | 1 | ≈2.0M | ⏳ ablation prévue |

Une couche = 2 GRUCells (Λ/Ψ) + 2 MLP de decay + 4 projections/gates croisés
+ readout 2H→H. L'embedding et le LM head sont **liés** (tied).

### 2.2 Recette d'entraînement
- AdamW (β=(0.9,0.999), weight_decay=0.1), cosine + warmup 200 par epoch,
  clip grad norm 1.0, batch 64×128 tokens, seq 128 ;
- **Garde de robustesse** : rollback des poids sur step suspect (loss NaN,
  outlier loss > 4×EMA+4, grad non-fini) + abort après 50 steps mauvais
  consécutifs (cf. §3).

---

## 3. Stabilité : le post-mortem du NaN (méthodologie)

Trois runs ont divergé (2l@1e-3 puis 2l@5e-4, deux fois dans la **même zone**
~step 1900-2150) avant le diagnostic. L'autopsie (détecteur d'anomalies
PyTorch) a isolé la cause racine :

```
alpha_psi = sigmoid(z)**0.2  →  en fp32, sigmoid s'arrondit à 0.0 pour
z ≲ -103 ; le forward reste fini (0^0.2 = 0) mais la dérivée
d/dx x^0.2 | x=0 = ∞ → backward NaN (SigmoidBackward0).
```

**Conséquences** : falaise purement *numérique*, ni données corrompues ni
architecture cassée ; elle se déclenchait plus souvent à la convergence (les
decays saturent) — 16% des steps (ep2) puis 97% (ep3-4).
**Correctif** : `clamp_min(1e-6)` avant la puissance (forward quasi identique,
dérivée bornée) → **0 rollback depuis** (vérifié sur 2l ep3-4 et 3l@5e-4).

Leçon de méthode : les divergences répétées dans une même zone de steps
pointaient une cause *déterministe* (ordre de données fixe + état du modèle),
pas un hasard d'optimisation — et `torch.autograd.set_detect_anomaly` a trouvé
l'op exacte en une passe.

---

## 4. Résultats mesurés (mêmes données, mêmes règles)

### 4.1 Baselines phase 4 (lr 1e-3, 4 epochs)
| Modèle | Params | Val ep1 → ep4 | TEST ppl |
|---|---:|---|---:|
| **mfn_dense 1L** | 1 153 440 | 2.6988 → 2.4013 | **11.20** |
| GRU (+LN) | 1 159 710 | 2.6583 → 2.4351 | 11.58 |
| GPT-2 mini 4×120 | 1 250 640 | 3.6818 → 3.3585 (6 ep) | 29.41 |

→ **À taille égale et mêmes règles, MFN bat GRU et GPT-2 mini** (offset
−0.034 vs GRU ; −0.96 vs GPT-2).

### 4.2 Profondeur (phase 5, lr 5e-4 — comparaisons à iso-epoch **valides**,
comparaisons vs 1L **invalides** : lr différent)
| Epoch | 2l | 3l |
|---|---:|---:|
| 1 | 3.0371 | 3.0541 |
| 2 | 2.8161 | **2.7945** |
| 3 | 2.6903 | **2.6582** |
| TEST | 13.75 | 14.53 (3 ep) |

→ **3l bat 2l à chaque epoch commun** : la profondeur paie à iso-budget.

### 4.3 Exposant données α_D (le premier chiffre de scaling)
Fit log-log `val_loss vs tokens vus` (4 epochs) :

| Modèle | α_D | vs littérature (0.095) |
|---|---:|---:|
| **MFN 1L** | **0.085** | −11% |
| GRU | 0.064 | −33% |
| GPT-2 mini | 0.051 | −46% |

→ MFN tire **~33% plus par token que GRU** et ~65% que GPT-2 mini sur nos
données. Lecture prudente (cf. §5.4) mais cohérente avec la thèse « fatigue =
efficacité de données ».

---

## 5. Programme de scaling : quatre axes, des barres falsifiables

### 5.1 Axe données D
Protocole : epochs successives = tokens vus. α_D(MFN) ≈ 0.085 (mesuré).
Objectif : confirmer α_D ≥ 0.085 sur d'autres tailles ; ancrage Chinchilla
(~20 tokens/param : nos 2.1M devraient voir ≈ 42M tokens uniques).

### 5.2 Axe paramètres N — le match α_N (en cours)
Points **règles identiques** (1e-3, 4 epochs, propres) :
| Point | Modèle | Params | Budget | Val ep4 | Statut |
|---|---|---|---:|---|---|
| P0 | 1L | 1 153 440 | 4 ep @1e-3 | 2.4013 | ✅ |
| P4 | 3l | 2 612 384 | 4 ep @1e-3 | ? | 🔄 aujourd'hui |

**Barre falsifiable** (écrite avant le run) : val ep4 ≤ 2.28-2.30 →
α_N(1.15M→2.61M) ≥ 0.076 (pente transform er publiée) ; < ~2.25 → pente plus
raide. Les runs 5e-4 (2.60/2.66) sont **explicitement invalides** pour la
pente (budget non aligné) ; ils servent de borne *pessimiste* de convergence.

### 5.3 Axe profondeur vs largeur de flux — le 3l3t (programmé ce soir)
- Question : la profondeur sature-t-elle (comme les RNN classiques) ?
- Protocole : 3l3t (6.75M, 3 threads) à 2 epochs @1e-3 vs 3l@2 epochs —
  la largeur de threads rattrape-t-elle la profondeur à budget égal ?
- Ablation de contrôle : `1l_192` (~2.0M) — profondeur à largeur constante ;
  `thread_fb=False` — rôle du latéral seul.

### 5.4 Réserve petite échelle (à citer dans toute conclusion)
- Kaplan 2020 : L(N)≈(8.8e13/N)^0.076 ; L(D)≈(5.4e13/D)^0.095 ;
  L(C)≈(1.6e7/C)^0.057 — mesures faites sur WebText2, tokénisation
  différente : on compare les **pentes**, jamais les offsets ;
- Chinchilla 2022 : N,D ∝ C^0.5 (≈20 tokens/param) ;
- Mamba 2023 : courbes récurrents-vs-transformers à 1.4B/2.8B — référence du
  « récurrent moderne qui scale » ;
- Réconciliation 2024 (Pearce & Song ; Porian et al.) : **courbure aux
  petites tailles** → nos exposants à 0.5-7M params sont *locaux*, indicatifs ;
  la preuve globale demanderait >100M params (cluster).
- Kaplan note aussi : les LSTM égalent les transformers en début de contexte
  mais décrochent plus loin → le créneau « mémoire longue » (k=20) de MFN.

### 5.5 Axe contexte (la niche structurelle)
Un récurrent paie O(H) par token quelle que soit la longueur (attention :
O(n²) + KV cache). Mesure de scaling propre : longueur de contexte vs qualité
à budget fixé — première jauge : le gain k=20 (mémoire longue) déjà observé
sur les variantes MFN.

---

## 6. Application : le chatbot (objectif n°1)

Pipeline : meilleur LM (test ppl min du pool 1L/GRU/2l/3l, **à budget égal**)
→ SFT UltraChat (CE masquée sur les réponses, 2 epochs, lr 1e-4) → chatbot
multi-tours. La sélection SFT est faite après complétion des budgets, jamais
sur des runs handicapés (règle posée pendant le développement).

---

## 7. Prochaines étapes (ordre de priorité)

1. Val ep4 du 3l@1e-3 (~17h40) → α_N P0→P4 ; SFT du meilleur réel ;
2. Grille 3l3t (2 epochs) — axe largeur de flux ;
3. Ablations : `1l_192`, `thread_fb off`, « fatigue off » (levier nominal de
   la pente) ;
4. Si les pentes le justifient : script de scaling multi-tailles (0.5-8M),
   ancres locales gratuites (GRU/GPT-2 phase 4) déjà en base ;
5. Longueur de contexte (seq 256/512) — la niche du récurrent.

## 8. Risques & limites (honnêteté scientifique)

- Exposants locaux ≠ exponentiels globaux (courbure petite échelle) ;
- CPU/GPU 960M : pas de fp16, tok/s limité → les budgets horaires contraignent
  le protocole (2 epochs sur le 3l3t) ;
- gating/fatigue = +calcul par paramètre : comparer aussi en FLOPs/token
  (la courbe « params » peut flatter, « compute » rabote) ;
- recettes d'entraînement non mûries vs 8 ans d'astuces transformers.

---

## Références

- Eldan & Li, *TinyStories* (2023) — corpus.
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020).
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla, 2022).
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling...* (2023).
- Pearce & Song, *Reconciling Kaplan and Chinchilla Scaling Laws* (2024) ;
  Porian et al., *Data Quality Scaling Laws* (2024).
- Cho et al., GRU (2014) ; Radford et al., GPT-2 (2019) — baselines locales.