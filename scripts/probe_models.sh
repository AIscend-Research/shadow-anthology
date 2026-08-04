#!/usr/bin/env bash
# Probe candidate models for logprob support before committing to a run.
#
#   bash scripts/probe_models.sh
#
# Being listed by /models is not enough: this method needs the tokens that
# LOST, and many served models return no logprobs, or return them only on one
# endpoint. Each probe is an 8-token generation -- a fraction of a cent.
#
# Pass your own list:  bash scripts/probe_models.sh accounts/fireworks/models/foo

set -uo pipefail
PY="${PY:-python}"
export PYTHONUNBUFFERED=1   # keep stderr and stdout in order when piped

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    accounts/fireworks/models/deepseek-v4-flash   # cheapest listed: $0.14/$0.28 per 1M
    accounts/fireworks/models/gpt-oss-20b         # open weights, >16B band
    accounts/fireworks/models/qwen3p7-plus
    accounts/fireworks/models/minimax-m2p7
    accounts/fireworks/models/gpt-oss-120b
  )
fi

for m in "${MODELS[@]}"; do
  printf '\n\033[1m--- %s\033[0m\n' "$m"
  $PY -m shadow_anthology.cli check --backend fireworks --model "$m" 2>&1 \
    | sed -n '/probing\|FAILED\|Traceback\|Error/,$p'
done

cat <<'EOF'

Pick a model that reports OK with >=2 candidates/step. Prefer, in order:
  1. "sampling_logprob (tempered)" over "raw logprob only" -- it is the
     distribution the sampler actually drew from, which is what we measure.
  2. an OK on the completions endpoint -- required for branching (E5).
  3. open weights, for the project's framing.

Then run:  MODEL=<id> bash scripts/run_all.sh
EOF
