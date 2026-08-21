#!/usr/bin/env bash
# Phase 5 — GPU queue: deep MFN variants (sequential, one log per arch).
set -u
cd "$(dirname "$0")"
LOGF=../logs/p5
for spec in "2l 4" "3l 3"; do
  set -- $spec
  a=$1; ep=$2
  echo "=== deep_$a (epochs=$ep) $(date)" >> ../logs/p5_queue.log
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
    --arch "$a" --epochs "$ep" --cuda --out results_deep.json \
    > "$LOGF"_"$a".log 2>&1
  echo "=== done deep_$a exit=$? $(date)" >> ../logs/p5_queue.log
done
echo "ALL_DONE $(date)" >> ../logs/p5_queue.log