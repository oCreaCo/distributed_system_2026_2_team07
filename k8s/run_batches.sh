#!/usr/bin/env bash
# Run calibration batches back to back: wait for the current experiment job, collect its logs,
# then launch the next config. Usage: k8s/run_batches.sh <name-of-running-job> <name:config.json> ...
#   e.g. k8s/run_batches.sh job1 job2:k8s/calib/job2.json job3:k8s/calib/job3.json
set -uo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=$PWD/team07.kubeconfig
LOG=A100_results/run_batches.log

wait_job() {  # returns when the job has a Complete or Failed condition (or disappears)
  while true; do
    cond=$(kubectl get job experiment -o jsonpath='{.status.conditions[*].type}' 2>/dev/null) || { echo "job gone"; return; }
    case "$cond" in *Complete*|*Failed*) echo "job condition: $cond"; return;; esac
    sleep 30
  done
}

collect() {
  python3 k8s/collect_results.py fetch "$1" && python3 k8s/collect_results.py split "$1"
  python3 k8s/collect_results.py summarize | tail -40
}

{
  current=$1; shift
  echo "$(date) waiting for running job ($current)"
  wait_job
  collect "$current"
  for spec in "$@"; do
    name=${spec%%:*}; cfg=${spec#*:}
    echo "$(date) launching $name ($cfg)"
    if ! k8s/sync.sh "$cfg" 2>&1 | grep -E 'created|Error|error'; then
      echo "$(date) sync failed for $name, retrying once"; sleep 30
      k8s/sync.sh "$cfg" 2>&1 | grep -E 'created|Error|error'
    fi
    wait_job
    collect "$name"
  done
  echo "$(date) all batches done"
} >> "$LOG" 2>&1
