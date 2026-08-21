#!/usr/bin/env bash
# deep_3l nocturne — config validée GTX 960M : batch 64 (96 = OOM, compile =
# GPUTooOldForTriton), eval-batch 48. Écrit ALL_DONE pour déclencher le SFT.
set -u
cd "$(dirname "$0")"
echo "=== deep_3l (epochs=3 batch=64 eval-batch=48) $(date)" >> ../logs/p5_queue.log
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
  --arch 3l --epochs 3 --cuda --out results_deep.json --eval-batch 48 \
  > ../logs/p5_3l.log 2>&1
echo "=== done deep_3l exit=$? $(date)" >> ../logs/p5_queue.log
echo "ALL_DONE $(date)" >> ../logs/p5_queue.log