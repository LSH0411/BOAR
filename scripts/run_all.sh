#!/usr/bin/env bash
# Full pipeline: the propensity network per dataset, then BOAR per dataset.
# All runs go into one queue dispatched over the GPUs in $GPUS (one run per GPU
# at a time; the next run starts as soon as a GPU frees).
#
#   GPUS="1 2 3" scripts/run_all.sh
#   STAGE=propensity scripts/run_all.sh          # stage 1 only
#   STAGE=boar DATASETS=tmall scripts/run_all.sh # stage 2 only, one dataset
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"taobao jdata tmall"}
GPUS=${GPUS:-"0"}
STAGE=${STAGE:-all}
export PYTHON=${PYTHON:-python}

queue=$(mktemp)
trap 'rm -f "${queue}" "${queue}.lock" "${queue}.rest"' EXIT

if [ "${STAGE}" = all ] || [ "${STAGE}" = propensity ]; then
  echo "== Stage 1: propensity networks =="
  for dataset in ${DATASETS}; do
    echo "scripts/train_propensity.sh ${dataset}" >> "${queue}"
  done
  scripts/dispatch.sh "${queue}" ${GPUS}
fi

if [ "${STAGE}" = all ] || [ "${STAGE}" = boar ]; then
  echo "== Stage 2: BOAR =="
  : > "${queue}"
  for dataset in ${DATASETS}; do
    echo "scripts/train_boar.sh ${dataset}" >> "${queue}"
  done
  scripts/dispatch.sh "${queue}" ${GPUS}
fi

echo "Done. Logs: runs/<dataset>/ and log/<dataset>/"
