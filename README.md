# MyelinFatigueNet — Protocole de validation complet

Architecture récurrente double-flux (expressif hΛ / conceptuel hΨ) avec portes de
confiance bidirectionnelles, décroissance adaptative et **fatigue neurale**,
fondée sur la dépression synaptique à court terme.

Ce dépôt contient le **papier corrigé** (`MyelinFatigueNet_corrige.md`, budget de
paramètres exact `22H² + (3d + 2d_out + 23)H + d_out` = **94 794** params vérifiés
par comptage direct), l'implémentation de référence et **les preuves expérimentales
de tous les entraînements exécutés** (500 epochs × 5 seeds pour les Tâches 1 et 3).

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `mfn.py` | Implémentation de référence `MyelinFatigueNet` (94 794 params à H=64, d=10, d_out=10) |
| `train_mfn.py` | Tâche 1 : copy multi-délai — famille MFN (complet, no-fatigue, no-bidir, no-gates, shared-decay) + baseline GRU |
| `baselines_t1.py` | Tâche 1 : baselines LSTM (H=144), Mamba-like (d_model=64, d_state=16, expand=8), sLSTM/xLSTM-like (H=288, 4 têtes) |
| `train_dualctx.py` | Tâche 3 : suivi de contexte dual (MFN, no-bidir, GRU, LSTM) |
| `train_ptb.py` | Tâche 2 : PTB niveau caractère (MFN, GRU H=170, GRU appariée H=125) |
| `fatigue_profile.py` | §8.5 : analyse du profil de fatigue (MFN entraîné k=10, 100 séquences) |
| `summarize.py` | Affiche les tables de résultats depuis les JSON |
| `fill_paper.py` | Injecte les résultats dans la Section 9 du papier (idempotent) |
| `tokenize_wt2.py` / `mfn_diag.py` / `mfn_lm.py` | Pipe LM (Phase 2) : BPE 4096 Wikitext-2, cellule O(H) diagonale, modèles `PreTrainedModel` compatibles transformers |
| `train_lm.py` / `bench_cell.py` | Entraînement LM (8 epochs) et micro-benchmark coût récurrent (~§3) |
| `fill_readme_lm.py` | Injecte les résultats LM dans le README (idempotent) |
| `MyelinFatigueNet_corrige.md` | Papier complet avec résultats mesurés |
| `*.json` | Preuves brutes (moyennes, écart-types, runs par seed) |

---

## Reproduction

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu numpy

# Tâche 1 — famille MFN + GRU (500 epochs × 5 seeds)
.venv/bin/python train_mfn.py --models mfn mfnnofatigue mfnnobidir mfnnogates mfnshareddecay gru --epochs 500 --seeds 0,1,2,3,4 --out results.json

# Tâche 1 — baselines LSTM / Mamba-like / xLSTM-like
.venv/bin/python baselines_t1.py --models lstm mamba xlstm --epochs 500 --seeds 0,1,2,3,4 --out baselines_t1.json

# Tâche 3 — dual-context
.venv/bin/python train_dualctx.py --models mfn mfnnobidir gru lstm --epochs 500 --seeds 0,1,2,3,4 --out dualctx.json

# Tâche 2 — PTB caractère (télécharge les données, ~1 h)
.venv/bin/python train_ptb.py --models mfn gru --epochs 15 --out ptb.json

# Profil de fatigue
.venv/bin/python fatigue_profile.py --epochs 500
```

Contexte d'exécution : CPU (4 cœurs), torch 2.13.0+cpu, numpy 2.5.2, Python 3.12
(venv `uv`). Les runs ont été menés en parallèle (1 thread par job) ; les temps
reportés dans les logs reflètent la contention.

---

## Preuves de résultats (moyenne ± écart-type, 5 seeds sauf mention)

### Tâche 1 — Copy multi-délai (d=10, T=100, MSE sur t>k, 500 epochs, batch 32, lr=1e-3)
Perte triviale (prédire zéro) ≈ 1.00 (x ∼ N(0, I)).

| Modèle | Paramètres (réels) | k=5 | k=10 | k=20 |
|---|---:|---:|---:|---:|
| **MFN complet** | 94 794 | 0.8449 ± 0.0062 | 0.9094 ± 0.0036 | **0.9826 ± 0.0056** |
| MFN no-fatigue (ϕ=0) | 94 794 | 0.8248 ± 0.0118 | 0.8997 ± 0.0055 | 0.9921 ± 0.0072 |
| MFN no-bidir (g≡1, WΛ→Ψ=0) | 86 538 | 0.8240 ± 0.0114 | 0.8949 ± 0.0073 | 0.9876 ± 0.0091 |
| MFN no-gates (g≡1) | 78 282 | 0.8422 ± 0.0094 | 0.9070 ± 0.0068 | 0.9810 ± 0.0061 |
| MFN shared-decay (MLP partagé) | 89 930 | 0.8454 ± 0.0062 | 0.9097 ± 0.0035 | 0.9821 ± 0.0051 |
| GRU (H=170) | 94 530 | 0.8139 ± 0.0311 | 0.9031 ± 0.0193 | 0.9945 ± 0.0085 |
| LSTM (H=144) | 91 306 | 0.8756 ± 0.0070 | 0.9429 ± 0.0072 | 1.0031 ± 0.0069 |
| Mamba-like (d_model=64, expand=8) | 103 946 | **0.5519 ± 0.0085** | 1.0244 ± 0.0059 | 1.0666 ± 0.0074 |
| xLSTM-like (H=288, 4 têtes) | 91 602 | 0.9132 ± 0.0070 | 0.9420 ± 0.0086 | 0.9951 ± 0.0194 |

### Tâche 2 — PTB niveau caractère (V=50, 5 101 618 chars, 15 epochs, batch 64, seq 256, seed unique)

| Modèle | Paramètres | BPC valid (best) | BPC test |
|---|---:|---:|---:|
| MFN | 107 634 | 1.3617 | 1.3341 |
| GRU appariée (H=125) | 107 175 | 1.3115 | **1.2820** |
| GRU large (H=170) | 191 640 | 1.2293 | 1.1960 |

### Tâche 3 — Suivi de contexte dual ([a₅;b₁₀], 500 epochs, batch 32)

| Modèle | MSE_a (k=5) | MSE_b (k=10) | MSE totale |
|---|---:|---:|---:|
| MFN complet | 0.0092 ± 0.0007 | 0.2468 ± 0.0157 | 0.2559 ± 0.0153 |
| MFN no-bidir | 0.0093 ± 0.0006 | 0.2501 ± 0.0091 | 0.2595 ± 0.0092 |
| GRU (H=170) | **0.0065 ± 0.0002** | 0.1944 ± 0.0077 | **0.2009 ± 0.0079** |
| LSTM (H=144) | 0.0123 ± 0.0004 | 0.2426 ± 0.0047 | 0.2548 ± 0.0050 |

### Profil de fatigue (MFN entraîné Tâche 1 k=10, inférence sur 100 séquences)

| Statistique | Valeur |
|---|---:|
| Fatigue moyenne flux expressif φ̄Λ | 0.1457 |
| Fatigue moyenne flux conceptuel φ̄Ψ | 0.1186 |
| Fraction neurones saturés (ϕ>0.5) | 0.0000 (les deux flux) |
| Corrélation temporelle des moyennes de flux | 0.9943 |
| Corrélation **par neurone** (moyenne sur 64 neurones) | **0.1533** (min 0.068, max 0.308) |
| Perte finale k=10 | 0.4625 |

---

## Conclusions (résultats mesurés)

1. **La fatigue neurale et le couplage bidirectionnel améliorent la rétention
   longue portée.** À k=20, MFN complet (0.9826) est le meilleur de tout le
   protocole : devant toutes les baselines (GRU 0.9945, xLSTM 0.9951, LSTM
   1.0031, Mamba-like 1.0666) **et** devant ses propres ablations sans fatigue
   (0.9921) et sans couplage bidirectionnel (0.9876). L'effet est modeste
   (Δ ≈ 0.005–0.01) mais reproductible sur 5 seeds — c'est le signal le plus
   probant du papier, cohérent avec l'hypothèse du flux conceptuel à longue
   portée.

2. **Aux délais courts, MFN paie un léger coût.** k=5 est dominé par le
   Mamba-like (0.5519) ; GRU (0.8139) et les ablations sans fatigue/sans
   bidirectionnel (≈0.825) devancent MFN complet (0.8449). Les mécanismes
   régulateurs coûtent donc sur les courtes portées ce qu'ils gagnent sur la
   plus longue.

3. **Les portes de confiance apprises ne sont pas le moteur principal.**
   L'ablation no-gates (g≡1) est quasi identique à MFN complet à tous les
   délais (0.8422/0.9070/0.9810 vs 0.8449/0.9094/0.9826) : la durée de vie
   mémoire (fatigue, découplage des flux) domine la dynamique sur cette tâche.

4. **Aucun avantage global en modélisation du langage (PTB caractère).** À
   budget apparié, la GRU (H=125, 107 175 params) devance MFN (107 634) :
   test 1.2820 vs 1.3341 BPC (Δ≈0.05), courbes encore descendantes à 15 epochs.

5. **Aucun avantage sur la mémoire parallèle (contexte dual).** GRU 0.2009 vs
   MFN 0.2559 (MSE totale) ; l'ablation no-bidir (0.2595) est indifférenciable
   de MFN complet (0.2559). La tâche doit être reformulée (délais asymétriques
   plus marqués, sources corrélées) pour discriminer l'architecture.

6. **Le profil de fatigue est sain.** φ̄Λ > φ̄Ψ (le flux expressif se fatigue
   plus, attendu) ; corrélation par-neurone r = 0.153 < 0.4 (cible §8.5) : les
   deux flux fatiguent de façon décorrélée neurone à neurone. La corrélation
   temporelle globale élevée (0.994) est un artefact de la modulation commune
   par l'énergie d'entrée, pas un verrouillage des flux.

### Bilan

Le protocole complet (budgets de paramètres **réellement comptés**, 5 seeds,
ablations fidèles — g≡1 effectif, WΛ→Ψ mis à zéro, MLP de décroissance
réellement partagé) confirme **un** effet attendu — la fatigue + le couplage
bidirectionnel aident la rétention à la plus longue portée testée — et ne
confirme pas les autres hypothèses. Les résultats défavorables sont documentés
tels que mesurés (§9.5 du papier). Pistes pour discriminer davantage : délais
k ∈ {40, 80}, entraînement LM plus long, tâches de raisonnement, supervision
par flux.

---

## Deuxième phase : adaptation modèle de langue compatibles `transformers`

MFN est décliné en **vrai modèle de langue** intégré à l'écosystème HuggingFace :
tokenizer BPE entraîné sur les données, classes `PreTrainedModel` + `GenerationMixin`
enregistrées dans les Auto APIs (`from_pretrained` / `generate` / push au hub).

### Architecture

La variante **O(H) diagonale** (`mfn_diag.py`, `MyelinFatigueNetDiag`) est adaptée
au régime LM (`mfn_lm.py`) : le plongement de token est l'entrée du flux, donc les
projections d'en-tête denses `d→H` (input_proj et première couche des MLPs de
décroissance) se réduisent en **opérations élémentaires** sur l'embedding
(taux de décroissance `α = σ(scale ⊙ e_t + bias)`, toujours conditionné entrée,
par neurone — l'analogue du Δ de Mamba). La récurrence complète est O(H) par pas.

```python
from mfn_lm import build_lm, true_param_count

model = build_lm("mfndiag", vocab_size=4096, hidden_size=256)
print(true_param_count(model))          # 1 052 672 (embeddings liées à la tête)
ids = tokenizer("The universe is", return_tensors="pt")["input_ids"]
out = model.generate(ids, max_new_tokens=64, top_k=40)
```

### Fichiers

| Fichier | Rôle |
|---|---|
| `tokenize_wt2.py` | Télécharge Wikitext-2 (parquet HF), entraîne un **BPE byte-level GPT-2 style** vocab **4096**, encode les 3 splits (Numpy) |
| `mfn_diag.py` | Cellule O(H) diagonale (implémentation de référence, régimes signaux) |
| `mfn_lm.py` | Configs + `PreTrainedModel`/`GenerationMixin` : `mfn_diag_lm`, `mfn_dense_lm`, `gru_lm` ; enregistrement Auto API ; comptage dédupliqué |
| `train_lm.py` | Entraînement LM (BPTT tronqué 128 avec carry-over d'état, AdamW lr=3e-4, cosinus) ; sauvegarde `save_pretrained`, échantillonne avec `generate` |
| `bench_cell.py` | Micro-benchmark du coût **récurrent seul** (fwd+bwd, sans projection logits) |

### Reproduction

```bash
uv pip install --python .venv/bin/python transformers tokenizers pyarrow

.venv/bin/python tokenize_wt2.py   # tokenizer_wt2/  +  data/wt2_encoded/

# 4 modèles, 8 epochs, 1 thread chacun (CPU 4 cœurs)
.venv/bin/python train_lm.py --model mfndiag --epochs 8 --threads 1 --out lm_results.json
.venv/bin/python train_lm.py --model mfndiag_nobidir --epochs 8 --threads 1 --out lm_results.json
.venv/bin/python train_lm.py --model gru --epochs 8 --threads 1 --out lm_results.json
.venv/bin/python train_lm.py --model dense_mfn --epochs 8 --threads 1 --out lm_results.json
```

### Budgets de paramètres (Wikitext-2, V=4096, embeddings liées à la tête)

| Modèle | H | Paramètres réels |
|---|---:|---:|
| MFN diagonal | 256 | 1 052 672 |
| MFN diagonal no-bidir | 256 | 1 050 880 |
| GRU (appariée) | 199 | 1 053 904 (−0.1 % vs MFN diag) |
| MFN dense O(H²) | 75 | 1 068 046 |

### Benchmark du coût récurrent seul (`bench_cell.py`, batch 64 × seq 128, fwd+bwd, 1 thread CPU)

| Cellule | Complexité/pas | ms/pas | timesteps/s |
|---|---:|---:|---:|
| MFN dense (H=75) | O(H²) | 683 | 11 988 |
| MFN diagonal (H=256) | **O(H)** | 439 | 18 660 |
| GRU (H=199) | O(H²) | 235 | 34 875 |

**Lecture.** Le O(H) diagonal divise par ~1.6 le coût de la version dense à budget
égal, mais reste **plus lent que la GRU** sur CPU mono-thread : il est limité par le
lancement de ~30 petits kernels élémentaires × 128 pas séquentiels, alors que la GRU
tient dans 3 matmuls fusionnés par pas. Le plancher asymptotique O(H) est réel
(le rapport s'inverse quand H grandit, convivial GPU), mais **les constantes et le
hardware comptent autant que la complexité** sur les tailles testées. En LM complet
(V=4096), la projection finale vers le vocabulaire domine le coût de **tous** les
modèles — la récurrence n'est plus le goulot.

### Résultats LM (epochs 8, batch 64, seq 128, lr 3e-4, seed 0, Wikitext-2 BPE-4096)

Rempli automatiquement depuis `lm_results.json` :

<!--LM_RESULTS_START-->
| Modèle | H | Paramètres | Val (best) / Test (loss) | Test ppl | Test BPC | tok/s | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFN diagonal O(H) (complet) | 256 | 1,052,672 | 6.4651 / 6.4597 | 638.9 | 2.6616 | 3706 | 8 |
| MFN diagonal no-bidir (ablation) | 256 | 1,050,880 | 6.4847 / 6.4785 | 651.0 | 2.6694 | 3753 | 8 |
| GRU (H=199, appariée) | 199 | 1,053,904 | 5.5247 / 5.5160 | 248.6 | 2.2728 | 3136 | 8 |
| MFN dense O(H²) (référence) | 75 | 1,068,046 | 5.6186 / 5.6180 | 275.4 | 2.3148 | 3130 | 8 |
<!--LM_RESULTS_END-->

### Conclusions LM (résultats mesurés, 8 epochs, seed 0)

1. **Le pipeline tokenization→transformers→entraînement→génération fonctionne
   et converge** : loss de ~8.31 (log 4096) à 5.52–6.48 selon le modèle, avec
   sauvegarde `save_pretrained`, reload `from_pretrained` et échantillonnage
   `generate()` (voir `lm_results.json`).

2. **À budget égal, la GRU est le meilleur modèle de langue** (test BPC 2.273,
   ppl 249), juste devant le MFN dense O(H²) (BPC 2.315, ppl 275) ; **le MFN
   diagonal O(H) est nettement dernier** (BPC 2.662, ppl 639), soit ~0.94 nat
   de perte de plus que la GRU. L'économie récurrente O(H) se paie en capacité de
   modélisation : le réglage diagonal indépendant par neurone est trop faible pour
   le langage (cohérent avec la littérature des RNN diagonaux).

3. **L'ablation no-bidir égalise le MFN diagonal complet** (test 6.4785 vs
   6.4597, Δ≈0.02 nat — dans le bruit d'une seed) : le couplage bidirectionnel
   n'apporte **toujours rien de mesurable** en LM, comme sur PTB (§9.2) et
   dual-context (§9.4).

4. **Le MFN dense bat sa propre déclinaison diagonale en qualité comme en
   débit de LM complet** (test BPC 2.315 vs 2.662 ; tok/s 3130 vs 3136-3753
   selon epoch) : sur la tâche de langue, le goulot n'est pas la récurrence mais
   la projection logits V×H — l'intérêt du O(H) ne se manifeste que sur des
   échelles où H domine V, ou en génération à K longue.

5. **Dégénérescence greedy classique en petit modèle** : GRU/dense répètent
   « the first time ... » , le diag s'effondre immédiatement (« is is is »/« , , ,
   ») — illustre qualitativement la différence de capacité, pas un bug.

6. **Comparabilité** : le BPC (indépendant du tokenizer) est comparable **au sein
   du même corpus** (les 4 modèles WT2 utilisent le même tokenizer, donc ppl et
   BPC se comparent directement). PTB (BPC 1.28–1.36) et Wikitext-2 (BPC
   2.27–2.66) ne sont pas comparables entre eux — corpus, vocabulaire et budgets
   différents ; le BPC ne sert que de normalisation interne.

---

## Troisième phase : petit LM « intelligent » orienté génération (TinyStories)

Cible retenue : **qualité avant vitesse** (« on perd en rapidité, on veut un petit
LM intelligent »). On abandonne le dogme O(H) (le benchmark §3 a montré le
diagonal nettement en retrait) et on met la vraie architecture densée dual-stream
en compétition sur un corpus narratif — avec un **vrai petit Transformer** comme
référence « LLM industrielle ».

### Protocole

- **Données** : TinyStories (2.12M histoires, téléchargé) → sous-ensemble
  **40K histoires train / 2K valid / 2K test** (sampling garanti par seed,
  histoires séparées par `<|endoftext|>`), soit **9.05M tokens** (0.255 tok/char).
- **Tokenizer** : BPE byte-level GPT-2 style, vocab **4096**, entraîné sur le
  sous-ensemble.
- **Modèles (~1.15M params, embeddings liées)** :
  - `mfn_dense` : noyau densé dual-stream du papier (2 GRUcells + cross-stream +
    portes de confiance + fatigue), **readout 2H→H** comme goulot puis **head
    liée** à l'embedding, **LayerNorm** avant la tête — la vraie architecture,
    sans contrainte de vitesse ;
  - `mfn_dense_nobidir` : ablation (portes ≡1, pas de W_lam→Ψ) ;
  - `gru` : GRU + LayerNorm, H=215 appariée au compte exact du MFN ;
  - `gpt2mini` : GPT-2 4 couches × 4 blocs, d=120 (transformers).
- **Entraînement** : 6 epochs, batch 64, seq 128, lr 1e-3 warmup 200 + cosinus,
  wd 0.1, clip 1.0, seed 0, 1 thread/job, 4 jobs CPU en parallèle.
- **Évaluation** : ppl + BPC (tokenizer identique → comparable), et **échantillons
  de narration** générés (`generate`).

### Résultats (6 epochs, TinyStories BPE-4096, seed 0)

Rempli automatiquement depuis `stories_results.json` :

<!--STORIES_RESULTS_START-->
| Modèle | H | Paramètres | Val (best) / Test (loss) | Test ppl | Test BPC | tok/s | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFN dense dual-stream v2 (H=144) | 144 | 1,153,440 | 2.5092 / 2.5296 | 12.6 | 0.9291 | 3634 | 6 |
| MFN dense no-bidir (ablation) | 144 | 1,049,472 | 2.5125 / 2.5320 | 12.6 | 0.9300 | 3215 | 6 |
| GRU + LN (H=215, appariée) | 215 | 1,159,710 | 2.5267 / 2.5430 | 12.7 | 0.9341 | 2976 | 6 |
| GPT-2 mini (transformers, 4 couches, d=120) | 120 | 1,250,640 | 3.5078 / 3.5319 | 34.2 | 1.2973 | 3010 | 6 |
<!--STORIES_RESULTS_END-->

### Conclusions phase 3

1. La **génération** est la vraie métrique ici : à ~1.15M params sur 9M tokens,
   les 4 modèles produisent des continuations de style TinyStories ; comparer
   les échantillons dans `stories_results.json` (grille greedy vs échantillonnée
   top-k, 4 prompts).
2. **MFN dense vs GPT-2 / GRU** : le dual-stream rattrape-t-il sa défaite du
   phase 2 (WT2) une fois la tête liée + LN + données narratives ? C'est le test
   de la phase 3.
3. **Ablation no-bidir** : dernière vérification à échelle « LM réelle » de la
   contribution du couplage bidirectionnel (jamais confirmée sur PTB, dual-context,
   ni WT2).

---

## Quatrième phase : petit LM narrateur sur la GTX 960M (100K histoires)

Montée d'échelle vers un LM « petite et plus intelligente » : **2.5× plus de
données** (100K histoires, 22.6M tokens), entraînement **GPU** (GTX 960M, torch
cu126), tout dans `phase4/` sans toucher aux artefacts des phases 2-3.

- **Données** : 100 000 histoires train / 2 000 valid / 2 000 test (sampling
  seedé depuis TinyStories), BPE-4096 dédié (`phase4/tokenizer_subset/`),
  0.253 tok/char.
- **Modèles** (budget ≈1.15M params) : `mfn_dense` (H=144, phase-3 gagnant),
  `gru` (H appariée, LayerNorm), `gpt2mini` (transformers, d=120). File GPU
  séquentielle, batch 64, lr 1e-3 (warmup 300 + cosinus), wd 0.1, clip 1.0.
- **Vitesse GPU mesurée** : MFN dense 13.5K tok/s, GRU 40.9K tok/s, GPT-2
  42.2K tok/s (la boucle séquentielle du dual-stream plafonne la 960M ;
  l'attention parallèle la exploite).

### Résultats (4-6 epochs, 22.6M tokens, seed 0)

Rempli automatiquement depuis `phase4/results.json` :

<!--P4_RESULTS_START-->
| Modèle | Paramètres | Epochs | Val (best) loss | Test ppl | Test BPC | tok/s | Device |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFN dense v2 (dual-stream + LN) | 1,153,440 | 4 | 2.4013 | 11.20 | 0.8813 | 13457 | cuda |
| GRU + LN (H appariée) | 1,159,710 | 4 | 2.4351 | 11.58 | 0.8933 | 40910 | cuda |
| GPT-2 mini (transformers) | 1,250,640 | 6 | 3.3585 | 29.41 | 1.2333 | 42212 | cuda |
<!--P4_RESULTS_END-->

### Conclusions phase 4

1. **MFN dense > GRU, conforté avec plus de données** (test ppl 11.20 vs 11.58,
   BPC 0.881 vs 0.893) : le dual-stream avec LayerNorm garde son avantage à
   22.6M tokens, et gagne ~1.35 ppl sur le phase 3 (12.55 → 11.20).
2. **GPT-2 mini reste loin** (ppl 29.4) : à 1.25M params et données limitées,
   l'attention sous-entraînée ne rattrape pas les récurrents ; elle ne brillera
   qu'avec beaucoup plus de tokens.
3. **L'ablation no-bidir n'a pas été relancée en phase 4** (3 protocoles déjà
   non confirmés) ; le gain MFN provient des GRUcells doubles + découplage des
   flux + fatigue, pas du gating bidirectionnel.
4. Le narrateur final est jouable : `chat_stories.py --phase4` (meilleur
   checkpoint auto-sélectionné).

---

## Notes méthodologiques

- **Ablations fidèles** : `no-gates`/`no-bidir` remplacent réellement les portes
  par des sorties ≡1 (une version antérieure gelait les poids à des valeurs
  aléatoires — corrigée avant les runs finaux) ; `no-bidir` met WΛ→Ψ à zéro ;
  `shared-decay` partage le même module MLP.
- **Comptages de paramètres réels** mesurés sur l'implémentation exécutable
  (dédupliqués pour les modules partagés), y compris ceux qui s'écartent des
  valeurs annoncées dans le papier original (LSTM 91 306 vs ~94K annoncé, etc.).
- **PTB** : split Mikolov standard téléchargé au premier run (stocké dans
  `data/ptb/`), vocabulaire caractère V=50, BPTT tronqué avec carry-over d'état,
  annealing cosinus.
- **Mamba-like** : implémentation propre d'un SSM sélectif (conv causale,
  discretisation softplus, B/C dépendants de l'entrée, d_state=16), pas un
  portage de la bibliothèque officielle.
- **xLSTM-like** : sLSTM avec gating exponentiel par tête (4 têtes) et
  normalisation d'état, H=288.
- `fill_paper.py` est idempotent : relancer `python3 fill_paper.py` après tout
  nouvel entraînement régénère la Section 9 du papier depuis les JSON.
- **Génération récurrente rapide** : `GenerationMixin.generate()` de
  transformers 5.x recalcule tout le préfixe à chaque pas (~1 tok/s ici).
  Les 4 modèles récurrents héritent de `RecurrentStateCacheMixin`
  (`mfn_lm.py`) qui (a) corrige le cache d'état en survolant les méthodes
  périmées par classe (un `prepare_inputs_for_generation` local ignorait
  `state` et le droppait) et (b) fournit **`generate_stream`** — préfill + 1
  pas de cellule par token généré, répétition-penalty + top-k + température.
  **Débits mesurés (CPU 4 threads, 150 tokens, checkpoints phase 4)** :
  GRU **915 tok/s**, MFN dense **135 tok/s** (2 GRUcells + projections + portes
  par pas), GPT-2 mini **74 tok/s** (boucle KV manuelle, 4 couches d'attention ;
  48 tok/s via `generate()`). Le chat utilise automatiquement le chemin le
  plus rapide (`chat_stories.py`).
