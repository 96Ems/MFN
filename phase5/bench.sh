#!/usr/bin/env bash
# Bench accélération GTX 960M — 3 configs sur arch 3l : tok/s sur 250 steps.
# batch 64 (baseline) / 96 / 96+compile (CUDA graphs).
# La config gagnante est écrite dans le .env pour le lancement du vrai run.
set -u
cd "$(dirname "$0")"
LOG=../logs/bench.log
BEST_TOK=0; BEST_CFG=""

run_and_measure() {
  B=$1; C=$2
  COMPILE=""; [ "$C" = "1" ] && COMPILE="--compile"
  : > "$LOG"
  echo "=== BENCH batch=$B compile=$C $(date)"
  env OMP_NUM_THREADS=2 ../.venv/bin/python -u train_deep.py \
    --arch 3l --epochs 3 --limit-steps 250 --batch "$B" $COMPILE \
    --tag "bench_b${B}c${C}" --out bench_none.json --cuda > "$LOG" 2>&1 &
  PID=$!
  # attendre la mesure (step 250) avec garde-fou 15 min (compile lente au 1er call)
  SECS=0
  while ! grep -q "step 250/" "$LOG"; do
    if ! kill -0 $PID 2>/dev/null; then
      echo "  -> run mort avant step 250 (voir $LOG), dernieres lignes:"
      tail -3 "$LOG"; return 1
    fi
    sleep 10; SECS=$((SECS+10))
    [ $SECS -ge 900 ] && { echo "  -> timeout 15 min"; kill $PID; return 1; }
  done
  T=$(grep -o "step 250/[0-9]* loss [0-9.]* \[[0-9]*s\]" "$LOG" |
      grep -o "\[[0-9]*s\]" | tr -d '[]s' | tail -1)
  L=$(grep -o "step 250/[0-9]* loss [0-9.]*" "$LOG" | tail -1)
  TOK_S=$((250 * 8192 / T))
  echo "  -> $L en ${T}s  =>  ${TOK_S} tok/s"
  kill $PID 2>/dev/null; sleep 1
  if [ "$TOK_S" -gt "$BEST_TOK" ]; then
    BEST_TOK=$TOK_S; BEST_CFG="$B $C"
  fi
}

run_and_measure 64 0
run_and_measure 96 0
run_and_measure 96 1

echo "=== WINNER: batch=${BEST_CFG% *} compile=${BEST_CFG#* } tok/s=$BEST_TOK $(date)"
echo "${BEST_CFG% *} ${BEST_CFG#* }" > bench_winner.env