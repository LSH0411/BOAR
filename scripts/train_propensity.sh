#!/usr/bin/env bash
# Train the propensity network for one dataset, using configs/<dataset>.yaml.
#   scripts/train_propensity.sh <dataset> [gpu]
# Writes get_propensity/propensity_scores/<dataset>/propensity_scores.npy,
# which is what BOAR loads by default.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=$1
GPU=${2:-0}
PYTHON=${PYTHON:-python}

mkdir -p "runs/${DATASET}"
cd get_propensity
"${PYTHON}" ./src/main.py \
  --dataset "${DATASET}" \
  --device "cuda:${GPU}" \
  2>&1 | tee "../runs/${DATASET}/propensity.out"
