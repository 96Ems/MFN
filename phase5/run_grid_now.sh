#!/usr/bin/env bash
# Grille 3l3t immédiate (skip SFT) : probe OOM -> 2 epochs @1e-3
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log
echo "=== GRILLE 3l3t DIRECTE (skip SFT, demande user) $(date)" >> "$Q"
B=24
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l3t --epochs 1 \
  --limit-steps 120 --batch 24 --lr 1e-3 --cuda --tag probe_g --out probe_none.json \
  > ../logs/probe_grid.log 2>&1
if grep -q "OutOfMemoryError" ../logs/probe_grid.log; then
  echo "  batch 24 : OOM -> batch 16" >> "$Q"; B=16
else
  echo "  batch 24 : OK" >> "$Q"
fi
echo "=== 3l3t FULL (2 ep, lr 1e-3, batch $B) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l3t --epochs 2 \
  --batch $B --lr 1e-3 --cuda --eval-batch 24 --out results_deep.json \
  > ../logs/p5_3l3t.log 2>&1
echo "=== done 3l3t exit=$? $(date)" >> "$Q"
