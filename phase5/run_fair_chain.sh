#!/usr/bin/env bash
# CHAÎNE ÉQUITABLE : 3l ep4 (complète le budget à 4 epochs comme 2l, depuis son
# ckpt ep3) -> SFT sur le VRAI meilleur (min test_ppl à budget égal) -> MATCH
# 1e-3 (4 epochs chacun). MATCH_ALL_DONE déclenche le PING 3.
set -u
cd "$(dirname "$0")"
echo "=== FAIR-CHAIN: 3l ep4 -> SFT gagnant -> match 1e3 $(date)" >> ../logs/p5_queue.log
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
  --arch 3l --epochs 4 --start-epoch 4 --cuda --eval-batch 48 \
  --out results_deep.json > ../logs/p5_3l_ep4.log 2>&1
echo "=== done 3l_ep4 exit=$? $(date)" >> ../logs/p5_queue.log
BEST=$(../.venv/bin/python - <<'PY'
import json
res = json.load(open('results_deep.json'))
best = min(res, key=lambda t: res[t]['test_ppl'])
print(best, res[best]['test_ppl'])
PY
)
set -- $BEST
echo "=== FAIR SFT from BEST=$1 (test_ppl=$2) $(date)" >> ../logs/p5_queue.log
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py \
  --base "$1" --epochs 2 --cuda --threads 2 --out results_sft.json \
  > ../logs/p5_sft_deep.log 2>&1
echo "=== SFT_DONE $1 exit=$? $(date)" >> ../logs/p5_queue.log
echo "=== MATCH 1e-3 (après SFT équitable) $(date)" >> ../logs/p5_queue.log
./run_match_1e3.sh