#!/usr/bin/env bash
# Shared work queue over a set of GPUs: every GPU runs one job at a time and
# pulls the next job as soon as it is free, so no GPU idles while work is left.
#
#   scripts/dispatch.sh <queue_file> <gpu> [gpu...]
#   scripts/dispatch.sh --worker <queue_file> <gpu>   # attach a single extra worker
#
# Each line of <queue_file> is a command to run with the GPU id appended as its
# last argument (train_propensity.sh and train_boar.sh both take the GPU last).
# Lines are consumed from the file, so a queue can be extended while it runs.
set -uo pipefail
cd "$(dirname "$0")/.."

pop_job() {
  local queue=$1
  exec 200>"${queue}.lock"
  flock 200
  local job
  job=$(head -n 1 "${queue}")
  if [ -n "${job}" ]; then
    tail -n +2 "${queue}" > "${queue}.rest" && mv "${queue}.rest" "${queue}"
  fi
  flock -u 200
  printf '%s\n' "${job}"
}

worker() {
  local queue=$1 gpu=$2 job
  while true; do
    job=$(pop_job "${queue}")
    [ -n "${job}" ] || break
    echo "[gpu ${gpu}] start: ${job}"
    if bash -c "${job} ${gpu}"; then
      echo "[gpu ${gpu}] done:  ${job}"
    else
      echo "[gpu ${gpu}] FAILED: ${job}" >&2
    fi
  done
}

if [ "${1:-}" = --worker ]; then
  worker "$2" "$3"
  exit $?
fi

QUEUE=$1
shift

pids=()
for gpu in "$@"; do
  worker "${QUEUE}" "${gpu}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
