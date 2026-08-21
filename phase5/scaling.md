# scaling.md — Courbe de scaling MFN vs littérature (tracker)

**Règles** : mêmes données (22.6M tokens TinyStories-100K, `phase4/data_subset`),
même tokenizer (BPE-4096), seed 0. Loss = val loss en nats/token (ppl = e^loss).
Paramètres = comptage réel dédupliqué. Références citées en bas — aucun modèle
externe ré-entraîné.

## Points MFN mesurés

| Point | Config | Params | Epochs | Val loss | Ppl | BPC | Statut |
|---|---|---|---|---|---|---|---|
| P0 | mfn_dense 1 couche H=144 | 1 153 440 | 4 | 2.4013 | 11.04 | 0.8813 | ✅ phase 4 |
| P1 | deep_2l [192,96] | 2 122 368 | — | — | — | — | 🔄 |
| P2 | deep_3l [192,128,96] | ~2.4M | — | — | — | — | ⏳ |
| P3 | mfn_dense 1 couche H=80 | ~0.5M | — | — | — | — | ⏳ |
| P4 | deep_4l / 2 couches larges | ~4-5M | — | — | — | — | ⏳ |
| P5 (opt) | — | ~8M | — | — | — | — | ⏳ |

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

## Analyse (à remplir quand P1-P2 sortent)

- Fit `log val_loss vs log params` sur les points disponibles : α_MFN = ?
- α ≷ 0.076 (référence transformers) ?
- Ajustement compute : loss vs FLOPs (tok/s mesurés × params × steps).
- Conclusion prudente (courbure petite échelle) + suite recommandée (>100M
  params = cluster) si la pente est prometteuse.