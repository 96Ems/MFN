#!/usr/bin/env bash
# POST-2ZF : SFT UltraChat sur deep_2zf -> Z10 (2 zones x 10 couches FAST, B64)
# Attendu : TEST 2zf (~05h10) -> SFT 2 ep (~8h) -> Z10 1 ep (~8h)
set -u
cd "$(dirname "$0")"
Q=../logs/p5_queue.log

echo "=== POST-2ZF : attente fin du run 2zf $(date)" >> "$Q"
while ! grep -aq "done 2zf" "$Q"; do sleep 60; done
echo "=== 2zf fini -> cooldown 10min $(date)" >> "$Q"
sleep 600

# moniteur de température global
( while true; do
    G=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    echo "$(date +%H:%M) gpu=$G" >> ../logs/temps.log; sleep 60
  done ) &
TMPID=$!

# garde thermique : SIGSTOP/SIGCONT au lieu de tuer (état préservé)
thermoguard() {
  local PID=$1
  while kill -0 "$PID" 2>/dev/null; do
    T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "$T" -ge 78 ]; then
      kill -STOP "$PID" 2>/dev/null
      echo "$(date +%H:%M) THERMO ${T}C -> PAUSE" >> ../logs/temps.log
      while :; do
        T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)
        [ "$T" -lt 62 ] && break
        sleep 90
      done
      kill -CONT "$PID" 2>/dev/null
      echo "$(date +%H:%M) THERMO ${T}C -> REPRISE" >> ../logs/temps.log
    fi
    sleep 180
  done
}

# 1) SFT UltraChat sur le 2ZF (2 epochs, batch 32 seq 256, lr 1e-4)
echo "=== SFT deep_2zf @UltraChat (2ep) $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_sft.py --base deep_2zf \
  --epochs 2 --batch 32 --seq 256 --lr 1e-4 --cuda --threads 2 \
  --out results_sft.json > ../logs/p5_sft_2zf.log 2>&1 &
SFT_PID=$!
thermoguard $SFT_PID &
wait $SFT_PID
echo "=== SFT_DONE exit=$? $(date)" >> "$Q"
sleep 600   # cooldown entre étages

# 2) Z10 : 2 zones asymetriques x 10 couches, FAST, B64, 1 epoch
echo "=== Z10 TRAIN GO $(date)" >> "$Q"
env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py --arch z10 --epochs 1 \
  --batch 64 --lr 1e-3 --cuda --eval-batch 48 --tag deep_z10 \
  --out results_deep.json > ../logs/p5_z10.log 2>&1 &
Z_PID=$!
thermoguard $Z_PID &
wait $Z_PID
echo "=== Z10 done exit=$? $(date)" >> "$Q"
echo "=== POST_2ZF_ALL_DONE $(date)" >> "$Q"
kill $TMPID 2>/dev/null