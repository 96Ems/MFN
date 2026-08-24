# REPORT_MAC — Campagne deep_z30 sur MacBook Pro M1 16 Go

**Agent** : ox-alpha (autonome) · **Date de début** : 24/08/2026
**Mission** : prompt.md (bench → campagne deep_z30 → SFT → tests chat → push)

---

## 1. Environnement

| Composant | Version |
|---|---|
| macOS / machine | Apple M1, 8 cœurs, 16 Go unifiés |
| Python | 3.12.14 (Homebrew, venv `.venv`) |
| torch | 2.13.0 (build MPS) — **MPS disponible : True** |
| transformers | 5.15.1 (identique laptop) |
| numpy / tokenizers | 2.5.2 / 0.22.2 |

## 2. Bench

⏳ en cours (voir §7 décisions pour le contexte mémoire).

## 3. Courbe de loss

⏳ campagne non lancée.

## 4. Tests chat

⏳ SFT non lancé.

## 5. Bugs rencontrés et corrigés (commits séparés)

### 5.1 `train_sft.py` : BASES sans deep_z30 (`0390e3e`)
La mission demande `--base deep_z30` ; le choix n'existait pas dans `BASES`
→ argparse refusait. Ajout d'une ligne.

### 5.2 `build_bigdata_stream.py` : val/test quasi vides (`439eaed`)
Symptôme : validation **2 659 tokens** au lieu de 437 807 (stats laptop).
Cause racine : dans `build_split`, le paramètre `total_hint=2_119_718` par défaut
rendait le dict `hints` **mort** (`hints[tag] if total_hint is None else ...`) →
les 2000 indices val/test étaient tirés dans `range(2_119_718)` alors que
`TinyStories-valid.txt` ne contient que **21 989 histoires** → espérance de hits
≈ 20. Fix : pool = `count_stories(path)` réel (fidèle à la v1).
**Vérification** : régénération → val 437 807 / test 438 682 tokens = match exact
laptop. Le train (86 262 160 tokens) était correct et n'a pas été reconstruit.

### 5.3 `sft_data.py` : parquet UltraChat disparu (commit `c9e4486`… push)
Le repo `stingning/ultrachat` a été déplacé vers `openbmb/UltraChat` et réorganisé
en shards JSONL (`data/train_sft-00.parquet` → 404). Adaptation au shard
`train_0.jsonl` (~1 Go, format plat alterné `[q1, a1, q2, a2, ...]`), conversion en
messages `[{"role"},{"content"}]`, pipeline aval strictement inchangé.
Résultat : 150 000 dialogues → train 32 033 534 tokens / val 646 466.

### 5.4 Mémoire : z30 inentraînable en l'état sur 16 Go (`aac13f5`)
- Sans checkpointing : activations ≈ **59 Go à B128 seq128** (mesuré : 1 couche ×
  128 pas = 0.67 Go ; z30 = 120 couches-zones). OOM même à B32. La mention
  README « ~3 Go » n'était pas validée matériellement.
- Fix : flag `--grad-ckpt` (checkpointing par couche/pas, `use_reentrant=False`)
  + flag `--amp` (autocast fp16 + GradScaler, unscale avant clip). Défauts OFF →
  comportement laptop inchangé.
- **Parité bit-exacte vérifiée** (2zf CPU, ckpt off/on) : delta loss et gradnorm
  **0.00e+00** — le recompute est mathématiquement transparent.
- Au passage corrigé : le flag existait dans `MFNDeepConfig` mais n'était jamais lu
  dans `MFNDeepStack.__init__` (no-op silencieux si quelqu'un l'avait utilisé).

## 6. Écarts à la mission §3 (décisions autonomes tracées)

### 6.0 Optimisation « prochain run » (parité vérifiée)
- **Contiguïté emb** : la boucle temporelle nourrissait les couches avec
  `emb[:, t]` (vue stridée T·H → re-matérialisations MPS potentielles à chaque
  GEMM de chaque couche de chaque pas). Désormais une seule permutation
  `(T,B,H)` contiguë upfront. **SHA-256 des logits identique avant/après**
  (`ee1ff352121ae625`, 2zf CPU eval déterministe).
- Bench matrice prévue : B64/B128 × ckpt × amp × threads{2,4} — décision sur
  tok/s mesuré (règle 📊).


| Mission prévoyait | Réalité constatée | Décision |
|---|---|---|
| batch selon seuils tok/s (800/250) | plafond mémoire AVANT seuils débit | caler le plus gros batch qui tient avec `--grad-ckpt` (+`--amp`), puis mesurer tok/s |
| « z30 ~3-4 Go » | ~59 Go d'activations brutes | `--grad-ckpt` obligatoire (§5.4) |

### 6.1 Architecture : localité des feedbacks (décision chercheur)
Question soulevée : `skip_fb` fait communiquer la couche 0 directement avec les
couches l ≥ 2 (seul chemin non-adjacent ; topdown et latéral sont locaux).
Coût chiffré : 4.9M params (7.4 % de z30). **Décision : on garde `skip_fb=True`
tel quel** pour la campagne (comparabilité P0-P4/laptop). Une ablation
`z30ns` (sans skip) reste une option falsifiable documentée ici pour plus tard.

## 7. Prochaines étapes recommandées

1. Corriger la doc README (« z30 ~3 Go » → réalité avec/sans `--grad-ckpt`)
2. Sur laptop : même pipeline `--grad-ckpt --amp` devrait profiter à la 960M
3. data_big2 FULL (~530M tokens) après le SFT si la campagne est stable
