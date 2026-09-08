#!/usr/bin/env bash
# Train BOAR for one dataset, using configs/<dataset>.yaml.
#   scripts/train_boar.sh <dataset> [gpu]
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=$1
GPU=${2:-0}
PYTHON=${PYTHON:-python}

mkdir -p "runs/${DATASET}"
"${PYTHON}" ./src/main.py \
  --dataset "${DATASET}" \
  --device "cuda:${GPU}" \
  2>&1 | tee "runs/${DATASET}/boar.out"
