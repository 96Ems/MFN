#!/usr/bin/env bash
# CHAÎNE PRIORITAIRE 1e-3 (budgets égaux, meilleur modèle réel) :
# 1) deep_3l @1e-3 4ep (candidat n°1)  -> SFT sur le meilleur de TOUT le pool
#    (mfn_dense 1L, gru, deep_2l, deep_3l — min test_ppl réel)
# 2) deep_2l @1e-3 4ep (point de pente) -> re-SFT si le gagnant change
# 3) MATCH_ALL_DONE (PING 3)
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log

picked() {
  ../.venv/bin/python - <<'PY'
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
}

echo "=== PRIORITY-1e3: 3l -> SFT meilleur -> 2l -> (re-SFT) $(date)" >> "$Q"

echo "=== lane 1 : deep_3l @1e-3 (4 ep) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
  --arch 3l --epochs 4 --lr 1e-3 --cuda --eval-batch 48 \
  --out results_deep.json > ../logs/p5_3l_1e3.log 2>&1
echo "=== done 3l_1e3 exit=$? $(date)" >> "$Q"

set -- $(picked)
BEST=$1; PPL=$2
echo "=== FAIR SFT from BEST=$BEST (test_ppl=$PPL) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py \
  --base "$BEST" --epochs 2 --cuda --threads 2 --out results_sft.json \
  > ../logs/p5_sft_deep.log 2>&1
echo "=== SFT_DONE $BEST exit=$? $(date)" >> "$Q"

echo "=== lane 2 : deep_2l @1e-3 (4 ep) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
  --arch 2l --epochs 4 --lr 1e-3 --cuda --eval-batch 48 \
  --out results_deep.json > ../logs/p5_2l_1e3.log 2>&1
echo "=== done 2l_1e3 exit=$? $(date)" >> "$Q"

set -- $(picked)
BEST2=$1; PPL2=$2
if [ "$BEST2" != "$BEST" ]; then
  echo "=== RE-SFT sur nouveau gagnant $BEST2 (test_ppl=$PPL2) $(date)" >> "$Q"
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py \
    --base "$BEST2" --epochs 2 --cuda --threads 2 --out results_sft.json \
    > ../logs/p5_sft_2.log 2>&1
  echo "=== SFT_DONE2 $BEST2 exit=$? $(date)" >> "$Q"
fi
echo "=== MATCH_ALL_DONE $(date)" >> "$Q"