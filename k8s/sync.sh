#!/usr/bin/env bash
# Sync local DT-FM code, test.py and a run config into /workspace, then launch experiment.yaml.
# Usage: k8s/sync.sh <run_config.json> [--no-run]
# The dev job is created only for the copy and deleted afterwards (it would hold 1 of the 8 GPUs).
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=$PWD/team07.kubeconfig

CONFIG=$1
RUN=${2:-}

kubectl delete job experiment --ignore-not-found
# a leftover dev job cannot be re-applied (immutable template), so always start from a fresh one
kubectl delete job dev --ignore-not-found
kubectl wait --for=delete pod -l batch.kubernetes.io/job-name=dev --timeout=120s || true
kubectl apply -f dev.yaml
kubectl wait --for=condition=Ready pod -l batch.kubernetes.io/job-name=dev --timeout=180s
DEV=$(kubectl get pods -l batch.kubernetes.io/job-name=dev -o name | cut -d/ -f2)

# pull the per-run logs written by earlier jobs (pod stdout is lost once a job fails)
(kubectl exec "$DEV" -- tar cf - -C /workspace/results --exclude=_sync . | tar xf - -C A100_results) || echo "results pull failed (continuing)"
tar cf - --exclude=.git DT-FM | kubectl exec -i "$DEV" -- tar xf - -C /workspace
kubectl cp k8s/test.py "$DEV":/workspace/test.py
kubectl cp k8s/nccl_bench.py "$DEV":/workspace/nccl_bench.py
kubectl cp "$CONFIG" "$DEV":/workspace/run_config.json
kubectl exec "$DEV" -- grep '"tag"' /workspace/run_config.json

kubectl delete job dev
kubectl wait --for=delete "pod/$DEV" --timeout=120s

if [ "$RUN" != "--no-run" ]; then
  kubectl apply -f experiment.yaml
  kubectl wait --for=condition=Ready pod -l batch.kubernetes.io/job-name=experiment --timeout=300s
  kubectl get pods -o wide
fi
