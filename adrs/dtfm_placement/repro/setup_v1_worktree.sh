#!/usr/bin/env bash
# Recreate the simulator-v1 environment in which adaevolve, EvoX and agent_cc were produced (2026-11-14/15).
#
#   adrs/dtfm_placement/repro/setup_v1_worktree.sh <dir>      e.g. ../team07_v1
#
# <dir> becomes a git worktree at e07b345: sim/stepsim.py v1 and the task (evaluator, topologies, initial_program,
# system message) as they were during the runs. Added on top:
#   - config.yaml / repro/config_evox.yaml from this checkout (OpenRouter instead of the old gpt-5 default;
#     the system message is unchanged),
#   - baselines.py and the evaluator with the vs_ga_sim / vs_greedy metrics (combined_score is unaffected),
#   - the v1 theta and the v1 reference/baselines caches from output/sim_v1/.
# Remove it afterwards with:  git worktree remove <dir>
set -euo pipefail
DIR=${1:?usage: setup_v1_worktree.sh <dir>}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TASK=$ROOT/adrs/dtfm_placement
V1=$TASK/output/sim_v1

git -C "$ROOT" worktree add --detach "$DIR" e07b345
DIR=$(cd "$DIR" && pwd)
cp "$TASK/config.yaml" "$TASK/evaluator.py" "$TASK/baselines.py" "$DIR/adrs/dtfm_placement/"
mkdir -p "$DIR/adrs/dtfm_placement/repro"
cp "$TASK/repro/config_evox.yaml" "$DIR/adrs/dtfm_placement/repro/"
cp "$V1/theta.json" "$DIR/sim/theta.json"
cp "$V1/reference.json" "$V1/baselines.json" "$DIR/adrs/dtfm_placement/"
# the environment (skydiscover/.venv) stays in this checkout; repro/run_skydiscover.sh takes ROOT=<dir>
echo "v1 worktree ready: $DIR"
echo "check: $ROOT/skydiscover/.venv/bin/python $DIR/adrs/dtfm_placement/evaluator.py \\"
echo "         $TASK/output/main_adaevolve_100/best/best_program.py train   # expect combined_score 1.2696"
