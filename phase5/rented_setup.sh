#!/usr/bin/env bash
# Setup 1-commande d'une box GPU louée pour MFN (Ubuntu/Debian, CUDA 12).
# Usage : ./phase5/rented_setup.sh [repo_dir]
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"

echo "== GPU =="
nvidia-smi | head -12

echo "== dépendances système =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv git curl rsync tmux htop

echo "== venv + torch CUDA =="
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
# torch avec CUDA 12.x (sm89 pour RTX 40xx ; sm_86 pour 30xx)
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121

echo "== dépendances projet =="
.venv/bin/pip install -q transformers tokenizers pyarrow numpy

echo "== smoke test : 100 steps 10L compile (rapide) =="
OMP_NUM_THREADS=$(nproc) .venv/bin/python phase5/train_deep.py --arch 10l \
  --epochs 1 --limit-steps 100 --batch 128 --lr 1e-3 --cuda --compile \
  --tag smoke_rented --out probe_none.json --eval-batch 64 \
  > /tmp/smoke.log 2>&1 || tail -n 20 /tmp/smoke.log
grep -E "step 100|epoch 1" /tmp/smoke.log || tail -n 10 /tmp/smoke.log
echo "== OK : la box est prete. lancements possibles : =="
echo "OMP_NUM_THREADS=\$(nproc) .venv/bin/python phase5/train_deep.py --arch 10l --epochs 2 --batch 128 --lr 1e-3 --cuda --compile --tag deep_10l --out results_deep.json"