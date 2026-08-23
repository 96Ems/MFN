#!/usr/bin/env bash
# Nuit : attendre data_big -> garde thermique -> 3L @B64 1ep @~90M tokens
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log
echo "=== BIG NIGHT: attente data_big puis 3L@B64 1ep $(date)" >> "$Q"
while [ ! -f ../phase4/data_big/stats.json ]; do sleep 60; done
echo "=== data_big prete $(cat ../phase4/data_big/stats.json | tr -d '\n' | head -c 120) $(date)" >> "$Q"
while true; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "$T" -lt 55 ] && break
  echo "  garde thermique GPU ${T}C" >> "$Q"; sleep 120
done
( while true; do
    G=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    L=$(cut -d' ' -f1 /proc/loadavg)
    echo "$(date +%H:%M) gpu=$G load=$L" >> ../logs/temps.log; sleep 60
  done ) &
TMPID=$!
echo "=== 3L @data_big B64 1ep GO $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 3l --epochs 1 \
  --batch 64 --lr 1e-3 --cuda --eval-batch 48 \
  --data ../phase4/data_big --tag deep_3l_big --out results_deep.json \
  > ../logs/p5_3l_big.log 2>&1
echo "=== done 3l_big exit=$? $(date)" >> "$Q"
kill $TMPID 2>/dev/null
