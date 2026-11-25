#!/usr/bin/env bash
# One SkyDiscover run on the placement task, as used for output/main_adaevolve_100 and output/main_evox_100.
#
#   adrs/dtfm_placement/repro/run_skydiscover.sh <search> <iterations> [config.yaml]
#     adaevolve 100                                   -> output/main_adaevolve_100   (task config.yaml)
#     evox 100 adrs/dtfm_placement/repro/config_evox.yaml -> output/main_evox_100
#
# Environment:
#   OPENROUTER_COURSE_PROJECT_KEY  required (course key, $50 cap). Never printed.
#   ROOT=<dir>                     task checkout to run against (default: this checkout). Use the worktree from
#                                  setup_v1_worktree.sh to rerun under simulator v1, as the original runs did.
# Prints wall time and the OpenRouter spend of the run (usage before/after from GET /api/v1/key).
# LLM sampling is not seeded, so a rerun gives a different trajectory and best program; see docs/reproduce.md.
set -euo pipefail
SEARCH=${1:?usage: run_skydiscover.sh <search> <iterations> [config]}
ITERS=${2:?usage: run_skydiscover.sh <search> <iterations> [config]}
HERE_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT=${ROOT:-$HERE_ROOT}
TASK=$ROOT/adrs/dtfm_placement
CFG=$(realpath "${3:-$TASK/config.yaml}")
OUT=$TASK/output/main_${SEARCH}_${ITERS}
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
: "${OPENROUTER_COURSE_PROJECT_KEY:?export OPENROUTER_COURSE_PROJECT_KEY first}"
# EvoX's meta-search LLM may read OPENAI_API_KEY; keep every call on the course OpenRouter key
export OPENAI_API_KEY="$OPENROUTER_COURSE_PROJECT_KEY"
unset OPENAI_API_BASE OPENAI_BASE_URL

usage() {
  curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_COURSE_PROJECT_KEY" |
    python3 -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])"
}
U0=$(usage); T0=$(date +%s)
echo "=== $SEARCH x$ITERS root=$ROOT cfg=$CFG start $(date -Is) usage_before=$U0"
mkdir -p "$TASK/output"
cd "$HERE_ROOT/skydiscover"
RC=0
.venv/bin/skydiscover optimize "$TASK/initial_program.py" "$TASK/evaluator.py" -c "$CFG" -s "$SEARCH" -i "$ITERS" \
  -o "$OUT" > "$OUT.log" 2>&1 || RC=$?
sleep 20  # let OpenRouter's usage counter catch up
U1=$(usage)
echo "=== done rc=$RC wall_s=$(( $(date +%s) - T0 )) usage_after=$U1 spent=$(python3 -c "print(round($U1 - $U0, 4))")"
exit $RC
