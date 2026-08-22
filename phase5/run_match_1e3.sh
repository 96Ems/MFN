#!/usr/bin/env bash
# MATCH scaling — deeps à lr 1e-3 (4 epochs, même budget que mfn_dense 1L P0).
# Vérifie la pente α_N : P3 = deep_2l@1e-3 (2.12M), P4 = deep_3l@1e-3 (2.61M),
# contre P0 = 1L@1e-3 (1.15M, val 2.4013). Code corrigé (clamp decay_psi),
# garde rollback active. Logs séparés. PAS d'ALL_DONE (pas de SFT auto de nuit).
set -u
cd "$(dirname "$0")"
LOGF=../logs/p5
echo "=== MATCH-1e3: deep_2l@1e-3 puis deep_3l@1e-3 $(date)" >> ../logs/p5_queue.log
for spec in "2l 4 2l_1e3" "3l 4 3l_1e3"; do
  set -- $spec
  a=$1; ep=$2; lg=$3
  echo "=== [$lg] epochs=$ep lr=1e-3 $(date)" >> ../logs/p5_queue.log
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
    --arch "$a" --epochs "$ep" --lr 1e-3 --cuda --eval-batch 48 \
    --out results_deep.json \
    > "$LOGF"_"$lg".log 2>&1
  echo "=== done [$lg] exit=$? $(date)" >> ../logs/p5_queue.log
done
echo "=== MATCH_ALL_DONE $(date)" >> ../logs/p5_queue.log