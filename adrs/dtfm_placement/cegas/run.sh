#!/usr/bin/env bash
# Run CEGAS (tcopt q_editeval) on the DT-FM placement task and log OpenRouter spend.
#   [CFG_SRC=<config.yaml>] adrs/dtfm_placement/cegas/run.sh <tag> [max_requests] [max_evals]
# Needs: cegas/config.yaml from prepare.py, TCResearch tcopt at $TCOPT (read-only, used via PYTHONPATH).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TASK=$ROOT/adrs/dtfm_placement
TCOPT=${TCOPT:-/home/yeomjy/TCResearch/tcopt}
TAG=${1:?tag}; REQ=${2:-100}; EVALS=${3:-100}
OUT=$TASK/output/cegas_$TAG
CFG=$OUT.config.yaml
export OPENROUTER_COURSE_PROJECT_KEY="${OPENROUTER_COURSE_PROJECT_KEY:-$(zsh -ic 'print -r -- $OPENROUTER_COURSE_PROJECT_KEY' 2>/dev/null | tail -1)}"
usage() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_COURSE_PROJECT_KEY" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])"; }
mkdir -p "$TASK/output"
sed -e "s/^  *max_requests: .*/    max_requests: $REQ/" -e "s/^max_iterations: .*/max_iterations: $EVALS/" "${CFG_SRC:-$TASK/cegas/config.yaml}" > "$CFG"
U0=$(usage); T0=$(date +%s)
echo "=== cegas $TAG requests=$REQ evals=$EVALS start $(date -Is) usage_before=$U0"
cd "$ROOT/skydiscover" && PYTHONPATH=$TCOPT .venv/bin/python -m tcopt.run "$TASK/initial_program.py" "$TASK/evaluator.py" \
    -c "$CFG" -i "$EVALS" -o "$OUT" > "$OUT.log" 2>&1
RC=$?; sleep 20; U1=$(usage)
echo "=== done rc=$RC wall_s=$(( $(date +%s)-T0 )) usage_after=$U1 spent=$(python3 -c "print(round($U1-$U0,4))")"
