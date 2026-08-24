#!/usr/bin/env bash
# Chaîne nocturne post-reboot : 3l3t (2ep @22.6M) -> 3L big-corpus (1ep @90M)
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log
echo "=== NIGHT CHAIN relance post-reboot $(date)" >> "$Q"

# garde thermique : attendre GPU < 55C avant tout load
while true; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "$T" -lt 55 ] && break
  echo "  garde thermique: GPU ${T}C, attente" >> "$Q"; sleep 120
done

# moniteur température (1 ligne/min, coût négligeable)
( while true; do
    G=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    L=$(cut -d' ' -f1 /proc/loadavg)
    echo "$(date +%H:%M) gpu=$G load=$L" >> ../logs/temps.log
    sleep 60
  done ) &
TMPID=$!

# 1) probe OOM puis grille 3l3t
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

# 2) gros corpus anti-famine : 3L @1e-3, 1 epoch sur ~90M tokens (34 tok/param)
while [ ! -f ../phase4/data_big/stats.json ]; do sleep 120; done
echo "=== BIG RUN: deep_3l 1ep @data_big (~90M tokens) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l --epochs 1 \
  --batch 24 --lr 1e-3 --cuda --eval-batch 24 \
  --data ../phase4/data_big --tag deep_3l_big --out results_deep.json \
  > ../logs/p5_3l_big.log 2>&1
echo "=== done 3l_big exit=$? $(date)" >> "$Q"
echo "=== NIGHT_ALL_DONE $(date)" >> "$Q"
kill $TMPID 2>/dev/null
