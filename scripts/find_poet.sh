#!/usr/bin/env bash
# Probe every generative model on the account and report which ones write
# VERSE rather than a chain of thought. ~8 tokens per model, well under a cent.
#
#   bash scripts/find_poet.sh
#   bash scripts/find_poet.sh accounts/fireworks/models/foo   # specific ones
#
# Portable to bash 3.2 (the macOS default) -- no mapfile, no readarray.

set -uo pipefail
export PYTHONUNBUFFERED=1
PY="${PY:-python}"

if [ "$#" -gt 0 ]; then
  MODELS="$*"
else
  MODELS=$($PY -m shadow_anthology.cli check --backend fireworks \
    | grep -o 'accounts/[a-z0-9/._-]*' \
    | grep -vE 'embedding|reranker')      # not generative
fi

if [ -z "$MODELS" ]; then
  echo "No models found. Is FIREWORKS_API_KEY set in this shell?" >&2
  exit 1
fi

count=$(printf '%s\n' "$MODELS" | wc -l | tr -d ' ')
echo "probing $count models for verse-vs-preamble..."

for m in $MODELS; do
  printf '\n\033[1m--- %s\033[0m\n' "$m"
  $PY -m shadow_anthology.cli check --backend fireworks --model "$m" --endpoint chat 2>&1 \
    | sed -n '/probing/,$p' \
    | grep -vE '^probing|^OK --|^[[:space:]]*$'
done

cat <<'MSG'

Pick a model whose sample is VERSE (not "Okay, the user wants...") and which
reports OK with >=2 candidates/step.

  MODEL=<id> bash scripts/run_all.sh
MSG
