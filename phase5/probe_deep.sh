#!/usr/bin/env bash
# Sonde profondeur : 5l puis 10l — batch décroissant si OOM (VRAM 2 Go,
# les activations du graphe déroulé dominent, pas les poids).
# Résultat : plus grand batch qui tient + tok/s, écrit dans deep_probe.env.
set -u
cd "$(dirname "$0")"
LOG=../logs/probe.log
BEST_CFG=""

probe() {
  A=$1; B=$2
  : > "$LOG"
  echo "=== PROBE $A batch=$B $(date)"
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
    --arch "$A" --epochs 1 --limit-steps 250 --batch "$B" \
    --tag "${A}_probe" --out probe_none.json --cuda > "$LOG" 2>&1 &
  PID=$!
  SECS=0
  while ! grep -q "step 250/" "$LOG"; do
    if ! kill -0 $PID 2>/dev/null; then
      if grep -q "OutOfMemoryError" "$LOG"; then
        echo "  -> OOM à batch $B"
        return 1
      fi
      echo "  -> mort avant step 250:"; tail -2 "$LOG"; return 1
    fi
    sleep 10; SECS=$((SECS+10))
    [ $SECS -ge 1800 ] && { echo "  -> timeout 30 min"; kill $PID; return 1; }
  done
  T=$(grep -o "step 250/[0-9]* loss [0-9.]* \[[0-9]*s\]" "$LOG" |
      grep -o "\[[0-9]*s\]" | tr -d '[]s' | tail -1)
  echo "  -> batch $B OK en ${T}s => $((250 * 8192 / T)) tok/s"
  kill $PID 2>/dev/null; sleep 1
  BEST_CFG="$A $B"
}

for A in 5l 10l; do
  probe $A 64 || probe $A 48 || probe $A 32
done
echo "=== PROBE WINNER: $BEST_CFG $(date)"
echo "$BEST_CFG" > deep_probe.env