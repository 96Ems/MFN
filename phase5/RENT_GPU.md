# Louer un GPU pour entraîner MFN — guide pratique (23 août 2026)

## Pourquoi louer
Notre code (cellule FAST, arch 2z/5l/10l/2zf, probing de zones) est prêt à tourner
sur du matériel moderne : fp16, torch.compile (Triton sm80+), gros batch. La 960M
locale (Maxwell) ne peut pas les exploiter ; une location 4090 transforme des runs
de 40h en 2-4h pour €1-3.

## Plateformes

| Plateforme | Prix 4090 | Stockage | UI | Notes |
|---|---|---|---|---|
| Vast.ai | 0.25-0.40 €/h | disque inclus + configurable | web/API/rsync | le moins cher ; vérifier notes/uptime du host |
| RunPod | 0.50-0.70 €/h | Network Volume (persistant, ~0.05€/GB/mois) | web + Jupyter | plus fiable, template "PyTorch 2.x CUDA 12" |
| Lambda | 1-1.3 €/h | disque | web/SSH | pro, cher |
| Modal | ~1 €/h | cloud | CLI | serverless, pas de serveur à gérer |

## Workflow typique (Vast.ai ou RunPod — même principe)

```bash
# 1. lancer une instance "RTX 4090 24GB" image pytorch:2.x-cuda12
#    (RunPod: template "pyTorch" ; Vast: template + SSH key)

# 2. envoyer le code et les données
git clone https://github.com/<votre>/MFN.git   # ou scp/rsync du dossier local
# corpus : rsync -av data_big/ user@host:/workspace/MFN/phase4/data_big/  (ou volume)

# 3. installer (une fois) — voir rented_setup.sh
cd MFN && ./phase5/rented_setup.sh

# 4. lancer le run (fp16 + compile + gros batch)
OMP_NUM_THREADS=8 .venv/bin/python phase5/train_deep.py --arch 10l \
  --epochs 1 --batch 128 --lr 1e-3 --cuda --compile \
  --tag deep_10l --out results_deep.json --eval-batch 64

# 5. surveiller (tmux + logs/ + notre dashboard si voulu en reverse-tunnel)

# 6. rapatrier le checkpoint puis ÉTEINDRE (la facture suit l'instance)
scp -r user@host:/workspace/MFN/phase5/checkpoints/deep_10l ./phase5/checkpoints/
# supervisord/cron : sauvegarde auto des ckpt vers le volume toutes les heures
```

## Budgets estimés pour nos runs (4090, fp16 + compile, ~120-160k tok/s)

| Run | Tokens | Durée | Vast | RunPod |
|---|---:|---:|---:|---:|
| 2ZF @22.6M (test local) | 22.6M | <10 min | — | local suffit |
| 10L validation | 22.6M | ~30 min | ~0.15 € | ~0.40 € |
| 30L asymétrique 72M | 1.4B | 2-4 h | **~1 €** | **~2-3 €** |
| 300M mini-généraliste | 6B | 1.5-2.5 j | ~15-25 € | ~30-50 € |
| 1B généraliste | 20B | 5-8 j (ou 2×GPU) | ~60-120 € | ~100-200 € |

## Règles d'or

1. **Sauvegarde des checkpoints toutes les heures** vers un volume persistant /
   Google Drive / S3 — les instances peuvent être récupérées ;
2. **Éteindre quand inactif** (billing horaire) ;
3. Vast : choisir des hosts avec notes hautes et +1000h d'uptime ; éviter les
   machines "RTX 4090 sous-refroidies" ;
4. Vérifier que le nvlink/multi-GPU n'est pas nécessaire au début (mono-GPU
   d'abord ; notre code est mono-`device` de toute façon) ;
5. Upload du corpus : compresser (tar.gz) — le réseau est souvent le goulot,
   pas le GPU.

## Ce qu'il faudra adapter au code (petit)

- [ ] `train_deep.py` : autocast fp16 (autocast=True + scaler) — 1 flag
- [ ] vérifier `torch.compile` sur sm89 (déjà codé, fallback eager propre)
- [ ] dataloading : les données sont déjà en .npy int32 → chargement direct
- [ ] (option) multi-GPU via DDP si deux cartes

Ces deux TODO sont ~30 lignes, faisables en une session avant le premier run loué.