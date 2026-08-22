#!/usr/bin/env bash
# POST-3L : attend le TEST de deep_3l@1e-3 -> SFT sur le meilleur réel du pool
# (1L mfn_dense, GRU, deep_2l, deep_3l) -> GRILLE 3l3t (2 epochs, lr 1e-3,
# probe batch 24 -> 16 si OOM) -> MATCH_ALL_DONE.
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log
echo "=== POST-3L attendu: TEST 3l@1e-3 -> SFT -> grille 3l3t $(date)" >> "$Q"
while ! grep -aq "\[deep_3l\] TEST" ../logs/p5_3l_1e3.log; do sleep 60; done
sleep 5
BEST=$(../.venv/bin/python - <<'PY'
import json, sys
pool = {}
try:
    for t, r in json.load(open('../phase4/results.json')).items():
        pool[t] = r['test_ppl']
except Exception as e:
    print('p4 skipped:', e, file=sys.stderr)
for t, r in json.load(open('results_deep.json')).items():
    pool[t] = r['test_ppl']
best = min(pool, key=pool.get)
print(best, pool[best])
PY
)
set -- $BEST
echo "=== FAIR SFT from BEST=$1 (test_ppl=$2) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py \
  --base "$1" --epochs 2 --cuda --threads 2 --out results_sft.json \
  > ../logs/p5_sft_deep2.log 2>&1
echo "=== SFT_DONE $1 exit=$? $(date)" >> "$Q"

echo "=== GRILLE 3l3t (2 ep, lr 1e-3) $(date)" >> "$Q"
B=24
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l3t --epochs 1 \
  --limit-steps 120 --batch 24 --lr 1e-3 --cuda --tag probe_g --out probe_none.json \
  > ../logs/probe_grid.log 2>&1
if grep -q "OutOfMemoryError" ../logs/probe_grid.log; then
  echo "  batch 24 : OOM -> batch 16" >> "$Q"; B=16
else
  echo "  batch 24 : OK" >> "$Q"
fi
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l3t --epochs 2 \
  --batch $B --lr 1e-3 --cuda --eval-batch 24 --out results_deep.json \
  > ../logs/p5_3l3t.log 2>&1
echo "=== done 3l3t exit=$? $(date)" >> "$Q"
echo "=== MATCH_ALL_DONE $(date)" >> "$Q"