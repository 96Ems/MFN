#!/usr/bin/env bash
# Phase 4 — GPU queue (GTX 960M), sequential, one log per model.
set -u
cd "$(dirname "$0")"
LOGF=../logs/p4
for spec in "gpt2mini 6" "mfn_dense 4" "gru 4"; do
  set -- $spec
  m=$1; ep=$2
  echo "=== $m (epochs=$ep) $(date)" >> ../logs/p4_queue.log
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_full.py \
    --model "$m" --epochs "$ep" --cuda --out results.json \
    > "$LOGF"_"$m".log 2>&1
  echo "=== done $m exit=$? $(date)" >> ../logs/p4_queue.log
done
echo "ALL_DONE $(date)" >> ../logs/p4_queue.log