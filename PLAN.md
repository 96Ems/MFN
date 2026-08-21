# PLAN — MFN profond : chatbot prioritaire + courbe de scaling vs littérature

Règles d'engagement :
1. **Objectif n°1 (prioritaire, socle) : le petit chatbot construit sur le meilleur
   des deep MFN.** Tout le reste est subordonné ; on ne sacrifie jamais le chatbot
   pour la recherche.
2. Objectif n°2 (fond de programme, en parallèle) : **mesurer la pente de scaling
   de MFN** et la comparer aux **courbes publiées** (Kaplan, Chinchilla, Mamba) —
   **sans ré-entraîner les modèles des autres** (long et cher) : on utilise les
   benchmarks existants + nos points déjà entraînés (GRU/GPT-2 phase 3-4) comme
   ancres locales gratuites.
3. Tout est reproductible : mêmes données (22.6M tokens, 100K histoires
   TinyStories, phase4/data_subset), même tokenizer (BPE-4096), seed 0, comptes
   de paramètres réels, commit git par étape.

---

## 🎯 Objectif 1 — Le petit chatbot deep MFN (prioritaire)

**Cette nuit (déjà en cours, automatique)**
| Étape | Statut | Livrable |
|---|---|---|
| `deep_2l` [192,96] lr 5e-4, 4 epochs | 🔄 epoch 2/4 (epoch 1 : val ppl 20.84) | checkpoint + results_deep.json |
| `deep_3l` [192,128,96] + skip, 3 epochs | ⏳ file | idem |
| Choix du vainqueur (meilleur test ppl) | ⏳ watcher `run_sft_after.sh` | — |
| **SFT UltraChat** (2 epochs, lr 1e-4, CE masquée) sur le vainqueur | ⏳ GPU libre ensuite | `sft_deep_*` + samples |

**Demain**
- `chatbot.py` (multi-tours, historique reformaté, generate_stream) branché sur
  `sft_deep_<vainqueur>` — c'est LE livrable utilisable.
- Comparaison rapide : deep gagne-t-il vs mfn_dense phase 4 (ppl 11.20) ?
  (Si oui → re-SFT optionnel depuis le deep si le SFT de la nuit a été fait sur
  l'autre base — le watcher choisit déjà le meilleur deep, donc normalement OK.)

**Critères de succès du chatbot**
- Réponses plausibles en style assistant (échantillons + test interactif).
- Val ppl SFT en baisse, pas de sur-apprentissage flagrant (2 epochs seulement).
- `git add` + commit du pipeline complet.

**Vigilances**
- Stabilité deep : garde anti-NaN active (3 non-finis → abort propre).
- RAM VRAM 2GB : batch 64 vérifié ; si OOM → batch 48/32.

---

## 📐 Objectif 2 — Courbe de scaling MFN (fond de programme)

### Points MFN à mesurer (mêmes données/tokenizer/seed)
| Point | Config | Params | Statut |
|---|---|---|---|
| P0 | mfn_dense H=144 (1 couche) | 1 153 440 | ✅ ppl 11.20 (phase 4) |
| P1 | deep_2l [192,96] | 2 122 368 | 🔄 cette nuit |
| P2 | deep_3l [192,128,96] | ~2.4M | 🔄 cette nuit |
| P3 | mfn_dense H=80 (petit) | ~0.5M | ⏳ nuit 2 |
| P4 | deep_4l [256,192,128,96] ou H=230 2 couches | ~4-5M | ⏳ nuit 2 |
| P5 (option) | ~8M | ⏳ nuit 3 |

Protocole : 2-4 epochs par point selon la taille (tokens/param ≈ 10-40),
cosinus + warmup, lr 5e-4 (profond) / 1e-3 (1 couche, déjà éprouvé).

### Droites de référence (littérature, rien à entraîner)
- **Kaplan 2020** : `L(N) ≈ (8.8e13/N)^0.076` (non-emb), `L(D) ≈ (5.4e13/D)^0.095`,
  `L(C) ≈ (1.6e7/C)^0.057`. Aussi mesuré : **les LSTM égalent les transformers en
  début de contexte mais décrochent plus loin** — le créneau mémoire longue de MFN.
- **Chinchilla 2022** : N,D ∝ C^0.5 (iso-compute), 400+ modèles 70M-16B.
- **Mamba 2023** : courbes recurrent-vs-transformer à échelle 1.4B/2.8B (référence
  « récurrent moderne »).
- **Réconciliation 2024** (Pearce & Song ; Porian et al.) : courbure aux petites
  tailles + comptage des emb vs non-emb → nos points 1-8M sont *indicatifs*
  (pente locale), la preuve définitive demanderait >100M (cluster).

### Analyse
1. Fit `log loss vs log params` sur P0-P5 → **α_MFN** (+ intervalle bootstrap).
2. Comparer à α = 0.076 (transformers) et aux pentes GRU/GPT-2 publiées.
   α_MFN > 0.076 → premier signal de barrière déplacée (pente plus raide).
3. Ajustement **compute** : loss vs FLOPs estimés (tok/s mesurés × params).
4. Ancre locale : nos GRU (ppl 11.58) et GPT-2 mini (29.4) phase 4 sur les mêmes
   données → alignement des offsets sur la même échelle.
5. Livrables : `phase5/scaling.md` (points, fits, comparaison chiffrée + citations)
   et le graphique dans `phase5/`.

### Vigilances scaling
- Exposant local ≠ global aux petites tailles (courbure) — conclusions prudentes.
- Params réels (dédupliqués), jamais les formules approchées.
- La courbe « params » peut flatter une archi coûteuse en FLOPs/token → toujours
  les deux axes (params et compute).

---

## 🗓️ Calendrier
- **Nuit 1 (en cours)** : deep_2l + deep_3l + SFT auto → chatbot testable au réveil.
- **Jour 1** : test du chatbot ; P0-P1-P2 dans scaling.md ; fit de première pente.
- **Nuit 2** : P3 (0.5M) + P4 (4-5M) → courbe à 5 points.
- **Jour 2** : fit α_MFN vs littérature, scaling.md finalisé, décision « barrière
  déplacée ou non » ; (option) re-SFT sur le meilleur absolu.
- **Nuit 3 (si utile)** : P5 (8M) et/ou entraînement plus long du chatbot.

## 🗂️ Fichiers
- `PLAN.md` (ce fichier), `phase5/scaling.md` (tracker courbe),
  `phase5/deepmfn.py`, `phase5/train_deep.py`, `phase5/train_sft.py`,
  `phase5/run_p5.sh`, `phase5/run_sft_after.sh`, `chatbot.py` (à créer),
  `phase5/results_deep.json`, `phase5/results_sft.json`, `phase5/checkpoints/`.
- Git : un commit par étape franchie.