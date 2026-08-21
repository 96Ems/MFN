#!/usr/bin/env bash
# Waits for the deep-MFN GPU queue, then runs the SFT from the BEST deep model.
set -u
cd "$(dirname "$0")"
LOGF=../logs/p5_queue.log
while ! grep -q ALL_DONE "$LOGF" 2>/dev/null; do
  sleep 300
done
sleep 60
BEST=$(../.venv/bin/python - <<'PY'
import json
res = json.load(open('results_deep.json'))
best = min(res, key=lambda t: res[t]['test_ppl'])
print(best, res[best]['test_ppl'])
PY
)
set -- $BEST
echo "=== SFT from BEST_DEEP=$1 (test_ppl=$2) $(date)" >> "$LOGF"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py \
  --base "$1" --epochs 2 --cuda --threads 2 --out results_sft.json \
  > ../logs/p5_sft_deep.log 2>&1
echo "=== SFT_DONE $1 exit=$? $(date)" >> "$LOGF"