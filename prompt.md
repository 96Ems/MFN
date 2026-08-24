# Mission : campagne deep_z30 sur MacBook Pro M1 16 Go

Tu es l'agent d'entraînement du projet **MyelinFatigueNet (MFN)** — architecture
récurrente double-zone asymétrique (zone A rapide/large, zone B lente/étroite
rate 2). Tu opères **en totale autonomie** sur le MacBook : tu clônes, installes,
bencher, entraînes, SFT, testes, et **pousses tout sur GitHub**. Le laptop qui
a généré le repo ne te renvoie rien : tes résultats sont le seul canal de retour.

## 1. Récupération

```bash
git clone https://github.com/96Ems/MFN && cd MFN
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch transformers tokenizers pyarrow numpy huggingface_hub
```

Versions de référence (laptop) : python 3.12, transformers 5.15.1, torch 2.x
(build macOS avec MPS), numpy 2.5.2. En cas d'échec d'import d'un module réseau
(`deepmfn`, `lm2`, `mfn_lm`), vérifie la compatibilité transformers avant toute
autre action.

Vérifs indispensables :
```bash
.venv/bin/python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
#   -> doit afficher True, sinon STOP : MPS non dispo, rien ne peut tourner
.venv/bin/python -c "from transformers import AutoModelForCausalLM; print('HF ok')"
```

## 2. Données (auto-téléchargement)

Le corpus TinyStories (~2 Go) et le parquet UltraChat (~250 Mo) se
téléchargent tout seuls au premier run, puis restent en cache :

```bash
# corpus big 86.2M tokens -> phase4/data_big/  (~10-20 min + dl 2 Go)
.venv/bin/python phase5/build_bigdata_stream.py
# data SFT UltraChat -> phase5/data_sft/ (parquet auto-dl)
.venv/bin/python phase5/sft_data.py
```

Règle : `build_bigdata_stream.py` avec `MFN_FULL=1` construit le corpus COMPLET
(530M tokens, `phase4/data_big2/`) — **NE PAS lancer pendant un entraînement**
(compétition CPU). Si la campagne tourne stable, tu peux le lancer APRÈS le SFT.

## 3. Bench obligatoire (jamais de campagne sans mesure)

```bash
.venv/bin/python phase5/train_deep.py --arch z30 --bench --batch 64 --device mps \
  --data phase4/data_big --grad-ckpt --amp --limit-steps 30
```

**CARTE MÉMOIRE RÉELLE (ton propre fix, août 2026)** : activations fp32 z30 à
B128 ≈ **59 Go** → `--grad-ckpt --amp` OBLIGATOIRES pour z30+ (parité bit-exacte
vérifiée en CPU 2zf). B64 + ckpt + amp ≈ **3 Go** (✓ M1 16 Go) ; B128 ≈ 6 Go
(limite).

Lis le tok/s ET le `MPS alloc X.XX Go` affichés par le bench (`BENCH mps: ...`).
- MPS alloc > 10 Go → baisse le batch (B32) : zone swap
- OOM + lenteur = MÊME cause (pressure mémoire/swap) → baisse le batch d'abord
- pressure mémoire : `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`, fermer les apps lourdes
- tok/s ≥ 800 → garde ce batch ; 250-800 → batch 128 si alloc < 10 Go ;
  < 250 → batch 128, 1 seul epoch, puis SFT (voir §5)

## 4. Campagne deep_z30 (2 epochs, ~86M tokens par epoch)

```bash
.venv/bin/python phase5/train_deep.py --arch z30 --epochs 2 --batch 64 --lr 1e-3 \
  --device mps --data phase4/data_big --grad-ckpt --amp --tag deep_z30 \
  --out results_deep_m1.json > logs/p5_z30.log 2>&1 &
caffeinate -dimsu &   # CRITIQUE : empêche le Mac de dormir (runs de nuit)
```

- `batch` : selon §3.
- ESPÉRANCE : 300-1000 tok/s → 1-3 jours par epoch. Si un epoch dépasse 4 jours,
  tu arrêtes après l'epoch 1 et passes au SFT (la science reste valide).
- Monitoring : toutes les 30 min, `grep "epoch\|step " logs/p5_z30.log | tail`, et
  vérifier qu'il n'y a pas de rollback en boucle (shadow-rollback prévu pour
  absorber les batches aberrants : 1-2 rollbacks par run = normal, une cascade
  = problème → investiguer, sauver, et relaunch avec `--start-epoch`).
- Le checkpoint est auto-sauvegardé à chaque amélioration de val dans
  `phase5/checkpoints/deep_z30/` (et les shards écrits à la fin de chaque epoch
  grâce à save_pretrained — vérifie que le dossier grossit).
- Température : si le Mac chauffe/throttle (fréquence CPU qui chute, tokt/s qui
  s'effondre), `pmset -g thermlog` pour confirmer, puis baisse `--threads 1`
  (les threads CPU servent au launch-bound) et relance.
- Ne PAS utiliser torch.compile sur MPS (indisponible, fallback eager prévu).
- Ne PAS installer bitsandbytes (inutile sur Mac, AdamW fp32 suffit : 72.7M
  params × 12 octets ≈ 0.9 Go).

## 5. SFT puis test chat

```bash
.venv/bin/python phase5/train_sft.py --base deep_z30 --device mps --tag sft_deep_z30 \
  --epochs 2 > logs/p5_sft_z30.log 2>&1 &
```

(relance `caffeinate` si le run de nuit suivant repart). Puis batterie de test
FIXE (greedy, max-new 80) et sauvegarde des réponses :

```bash
for q in "Tell me a story about a cat." "What is the capital of France?" \
         "Write a short poem about the sea." "Why is the sky blue?"; do
  .venv/bin/python phase5/chat_ultra.py --tag sft_deep_z30 --ask "$q" --greedy --max-new 80
done
```

Attendu : réponses plus lisses que le modèle 1.8M (qui collait un moule unique
« Certainly! Here are some examples... »), débuts de distinction entre thèmes
(story/poème vs infos), encore loin d'un assistant. Chaque réponse va dans ton
rapport (voir §6) verbatim.

## 6. Livraison : rapport + push GitHub

Crée `REPORT_MAC.md` à la racine avec :
1. Versions (python, torch, transformers) + MPS OK
2. Bench : tok/s × batch × seq, machine (M1, RAM)
3. Courbe de loss : « train/val par epoch » (extrait du log), val finale, ppl,
   bpc ; durée réelle par epoch
4. Tests chat : les 4 questions ci-dessus, réponses verbatim, tes observations
5. Décisions prises et pourquoi (autonomie tracée)
6. Prochaines étapes recommandées (ex: data_big2 FULL pour z300, SFT 2e epoch, etc.)

Git :
```bash
git add -A
git commit -m "Mac M1 : campagne deep_z30 + SFT (rapport + checkpoints)"
git pull --rebase origin main    # par sécurité
git push origin main
```

Vérifie APRÈS push : `git status -sb` doit afficher `## main...origin/main`
sans commit en avance. Le laptop récupérera tout par `git pull`.

## 7. Règles d'autonomie (liberté ET garde-fous)

- ✅ Tu décides : batch, threads, arrêt après epoch 1, reprises (--start-epoch),
  ordre des tâches, second run de bench.
- ❌ Interdits : modifier `ARCHES`/les hyperparams dans train_deep.py sans le
  documenter dans REPORT_MAC.md ; supprimer des checkpoints profonds sans les
  avoir poussés ; lancer 2 entraînements GPU simultanés ; `git push --force` ;
  toucher aux branches autres que main ; effacer le cache HF (~2 Go, à re-télécharger).
- ⏱️ Bloqué > 2 h sur un problème : applique le plan B le plus raisonnable,
  documente-le dans le rapport, continue. Ne reste JAMAIS à attendre.
- 🛑 Bug dans le code phase5 : si le fix est triviale (chemin, typo), fais-le et
  commite-le séparément avec message clair ; sinon contourne et documente.
- 🔋 Le Mac doit rester branché en permanence pendant les runs de nuit — le
  signale dans le rapport si tu constates une décharge.
- 📊 Mesure d'abord, conclus ensuite : chaque décision de campagne s'appuie sur
  le bench, jamais sur une intuition.

Objectif final : un `deep_z30` (72.7M params) entraîné + SFT, une preuve de
valeurs sur le M1, et tout (checkpoints, rapport, résultats JSON) sur GitHub
pour que le laptop n'ait qu'à `git pull`.