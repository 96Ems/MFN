#!/usr/bin/env bash
# Nuit 2Z : streaming data_big (CPU seul) -> garde thermique -> 2Z @22.6M 2ep -> TEST
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log
echo "=== 2Z NIGHT: build_bigdata streaming puis 2Z@subset B32 2ep $(date)" >> "$Q"

# 1) encodage SEUL sur CPU (idempotent : skip si déjà fait)
if [ ! -f ../phase4/data_big/stats.json ]; then
  ../.venv/bin/python -u build_bigdata_stream.py > ../logs/bigdata.log 2>&1
  echo "=== bigdata exit=$? $(tail -n1 ../logs/bigdata.log | head -c 80) $(date)" >> "$Q"
else
  echo "=== bigdata deja pret (skip) $(date)" >> "$Q"
fi

# 2) garde thermique
while true; do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "$T" -lt 55 ] && break
  echo "  garde thermique GPU ${T}C" >> "$Q"; sleep 120
done

# moniteur température
( while true; do
    G=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    L=$(cut -d' ' -f1 /proc/loadavg)
    echo "$(date +%H:%M) gpu=$G load=$L" >> ../logs/temps.log; sleep 60
  done ) &
TMPID=$!

# 3) 2ZF @22.6M : zones asymetriques + moteur FAST, B64 (bench: 2211 tok/s, x2.25 vs B32)
echo "=== 2ZF TRAIN GO (B64, 2ep, lr 1e-3) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch 2zf --epochs 2 \
  --batch 64 --lr 1e-3 --cuda --eval-batch 48 --tag deep_2zf --out results_deep.json \
  > ../logs/p5_2z.log 2>&1
echo "=== done 2zf exit=$? $(date)" >> "$Q"
kill $TMPID 2>/dev/null
