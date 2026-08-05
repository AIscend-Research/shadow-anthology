#!/usr/bin/env bash
# Run the full experiment suite.
#
#   bash scripts/run_all.sh                  # local weights (default)
#   BACKEND=fireworks bash scripts/run_all.sh
#
# Override anything with env vars:
#   SAMPLES=32 MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_all.sh
#
# Only E1, E3 and E5 generate. E2 and E4 re-analyse traces already on disk:
# they cost nothing, and they MUST reuse the same traces, because comparing
# rank 1 against rank 2 across different poems would destroy the pairing the
# whole design rests on.

set -euo pipefail

# ---------------------------------------------------------------- backend
# Default is LOCAL WEIGHTS. Measured on the hosted catalogue: every model
# there is a reasoning model that composes the poem while thinking and then
# transcribes it, so the poem span comes out deterministic (entropy 0.06 bits,
# margin 8.65 nats, 0% off-argmax). There are no sampler decisions in it --
# nothing for this project to measure. Local weights give full top-k, the
# exact tempered distribution, real forced-prefix branching, and no cost.
BACKEND="${BACKEND:-hf}"
SAMPLES="${SAMPLES:-16}"       # per prompt; 16 prompts x 16 = 256 poems per arm
PERM="${PERM:-20000}"          # permutation resamples
PROMPTS="${PROMPTS:-prompts/poems.txt}"
RUNS="${RUNS:-runs}"           # output dir; set it to compare models side by side
PY="${PY:-python}"

if [ "$BACKEND" = "hf" ]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
  CANDS="${CANDS:-20}"         # full top-k, not the hosted cap of ~5
  CONC="${CONC:-1}"            # one torch module cannot serve parallel loops
  MAXTOK="${MAXTOK:-224}"
  # Length guidance matters: without it, models that favour longer poems run
  # into the token cap and every text is truncated mid-line, which corrupts
  # repetition/TTR and reads badly. Ask for a length the cap can accommodate.
  SYSTEM="${SYSTEM:-Write only the poem. No title, no commentary, no explanation. Keep it under 12 short lines.}"
  CORPUS_BE=(--backend hf --model "$MODEL")
  BRANCH_BE=("${CORPUS_BE[@]}")          # branches natively
  SLICE=()                               # no reasoning to slice away
else
  MODEL="${MODEL:-accounts/fireworks/models/deepseek-v4-flash}"
  CANDS="${CANDS:-6}"
  CONC="${CONC:-8}"
  MAXTOK="${MAXTOK:-700}"                # room to think AND write
  MARKER="${MARKER:-===POEM===}"
  SYSTEM="${SYSTEM:-Think as long as you need. When you are ready, output the line ${MARKER} on its own, and after it write ONLY the poem: no title, no commentary, no explanation. Keep the poem under 12 short lines.}"
  CORPUS_BE=(--backend fireworks --model "$MODEL" --endpoint chat)
  BRANCH_BE=(--backend fireworks --model "$MODEL" --endpoint completions)
  SLICE=(--poem-marker "$MARKER")
  export MARKER RUNS
  if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    echo "FIREWORKS_API_KEY is not set." >&2
    echo "  echo \"export FIREWORKS_API_KEY='fw_...'\" >> ~/.zshrc && source ~/.zshrc" >&2
    exit 1
  fi
fi

SAMP=(--candidates "$CANDS" --max-tokens "$MAXTOK" --top-p 0.95 --system "$SYSTEM")

mkdir -p "$RUNS"
step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

# Resume: an arm whose summary.txt exists is complete and is skipped, so an
# interrupted run picks up where it stopped instead of regenerating hours of
# identical work. Seeds are deterministic, so a resumed arm is byte-identical
# to one produced in a single pass. FORCE=1 recomputes everything.
done_already() {
  if [ "${FORCE:-0}" != "1" ] && [ -f "$1/summary.txt" ]; then
    echo "  skip $(basename "$1") — already complete (FORCE=1 to redo)"
    return 0
  fi
  return 1
}
echo "backend=$BACKEND model=$MODEL samples=$SAMPLES candidates=$CANDS out=$RUNS"

# ---------------------------------------------------------------- smoke test
# One generation through the exact code path of the real run. A bad key, a
# wrong model id, a rejected parameter, or a model that writes reasoning
# instead of verse surfaces here rather than partway into a 256-generation arm.
step "SMOKE TEST"
$PY -m shadow_anthology.cli trace "${CORPUS_BE[@]}" "${SAMP[@]}" \
  --prompt "Write a short poem about salt." --out "$RUNS"/smoke.json >/dev/null

if BACKEND="$BACKEND" RUNS="$RUNS" $PY - <<'PYEOF'
import os, sys
from shadow_anthology import GenerationTrace
from shadow_anthology.corpus import REASONING_MARKERS, slice_to_poem

t = GenerationTrace.load(os.environ["RUNS"] + "/smoke.json")
if os.environ["BACKEND"] == "hf":
    poem = t
else:
    kept, rep = slice_to_poem([t], os.environ["MARKER"])
    if not kept:
        print("  never reached the poem:", rep)
        sys.exit(1)
    poem = kept[0]

print(f"  poem span: {len(poem)} tokens of {len(t)}")
print(f"  poem     : {poem.text.strip()[:80]!r}")
print(f"  entropy  : {poem.mean_entropy:.3f} bits, off-argmax {poem.offrank_fraction:.1%}")

if any(m in poem.text.strip()[:80].lower() for m in REASONING_MARKERS):
    print("  -> this is reasoning, not verse")
    sys.exit(1)
if len(poem) < 12:
    print("  -> too short to analyse")
    sys.exit(1)
# The whole project needs contested positions. A span the sampler never had a
# real choice in yields a shadow poem of tokens that were never in contention.
if poem.mean_entropy < 0.15:
    print("  -> DETERMINISTIC span (entropy < 0.15 bits): the sampler made no")
    print("     real choices here, so there is nothing to reconstruct.")
    sys.exit(1)
PYEOF
then
  echo "smoke test OK"
else
  printf '\n\033[1mSTOPPING: this model will not produce a usable corpus.\033[0m\n'
  echo "  Try the local backend:  pip install -e '.[hf]' && bash scripts/run_all.sh"
  exit 1
fi

# ------------------------------------------------------------------------ E1
step "E1 — main corpus, T=1.0  [GENERATES]"
done_already "$RUNS"/T1.0 || $PY -m shadow_anthology.cli corpus --prompts "$PROMPTS" \
  "${CORPUS_BE[@]}" "${SAMP[@]}" ${SLICE[@]+"${SLICE[@]}"} \
  --samples "$SAMPLES" --temperature 1.0 --concurrency "$CONC" \
  --permutations "$PERM" --out "$RUNS"/T1.0

if grep -q "CORPUS SANITY WARNING" "$RUNS"/T1.0/summary.txt; then
  printf '\n\033[1mSTOPPING: E1 did not produce poems.\033[0m\n'
  sed -n '/CORPUS SANITY WARNING/,/^!\{20,\}$/p' "$RUNS"/T1.0/summary.txt
  echo; echo "Inspect $RUNS/T1.0/pairs.txt before re-running."
  exit 1
fi

# ------------------------------------------------------------------------ E2
step "E2 — rank depth 1/2/3, reusing E1 traces  [FREE]"
for r in 1 2 3; do
  done_already "$RUNS/rank$r" || $PY -m shadow_anthology.cli corpus \
    --from-traces "$RUNS"/T1.0/traces.jsonl \
    --rank "$r" --permutations "$PERM" --out "$RUNS/rank$r"
done

# ------------------------------------------------------------------------ E3
step "E3 — temperature sweep  [GENERATES]"
for t in 0.3 0.7 1.3; do   # T=1.0 is E1; don't pay for it twice
  done_already "$RUNS/T$t" && continue
  $PY -m shadow_anthology.cli corpus --prompts "$PROMPTS" \
    "${CORPUS_BE[@]}" "${SAMP[@]}" ${SLICE[@]+"${SLICE[@]}"} \
    --samples "$SAMPLES" --temperature "$t" --concurrency "$CONC" \
    --permutations "$PERM" --out "$RUNS/T$t"
done

# ------------------------------------------------------------------------ E4
step "E4 — gated shadow (12 closest calls only), reusing E1 traces  [FREE]"
done_already "$RUNS"/gated || $PY -m shadow_anthology.cli corpus --from-traces "$RUNS"/T1.0/traces.jsonl \
  --top-n 12 --permutations "$PERM" --out "$RUNS"/gated

# ------------------------------------------------------------------------ E5
# Non-fatal: E1-E4 are the paired statistics and are already on disk by now.
# A branching failure must not discard them.
step "E5 — branching anthology  [GENERATES]"
if [ "${FORCE:-0}" != "1" ] && [ -f "$RUNS"/anthology.json ]; then
  echo "  skip E5 — already complete (FORCE=1 to redo)"
elif $PY -m shadow_anthology.cli trace "${BRANCH_BE[@]}" "${SAMP[@]}" \
      --prompt "Write a short poem about winter light on water." \
      --temperature 1.0 --out "$RUNS"/branch_trunk.json >/dev/null; then
  $PY -m shadow_anthology.cli branch --trace "$RUNS"/branch_trunk.json "${BRANCH_BE[@]}" \
    --points 8 --ranks 1 2 --budget 24 --max-tokens "$MAXTOK" \
    --out "$RUNS"/anthology.json || echo "E5 branching failed (see above)"
  $PY -m shadow_anthology.cli render --trace "$RUNS"/branch_trunk.json \
    --out "$RUNS"/poem.html --title "The Shadow Anthology" || true
else
  echo "E5 SKIPPED: this backend/endpoint cannot resume from a forced prefix."
  RUNS="$RUNS" $PY - <<'PYEOF' || true
import os
from shadow_anthology import load_traces, write_html
ts = load_traces(os.environ["RUNS"] + "/T1.0/traces.jsonl")
if ts:
    write_html(os.environ["RUNS"] + "/poem.html", ts[0], title="The Shadow Anthology")
    print("rendered "$RUNS"/poem.html from a corpus trace instead")
PYEOF
fi

# --------------------------------------------------------------------- recap
step "DONE"
echo "off-argmax fraction by temperature (the sampler's editorial footprint):"
for d in "$RUNS"/T0.3 "$RUNS"/T0.7 "$RUNS"/T1.0 "$RUNS"/T1.3; do
  [ -f "$d/summary.txt" ] && printf '  %-12s %s\n' "$(basename "$d")" \
    "$(grep offrank_fraction "$d/summary.txt" | awk '{print $2}')"
done
echo
echo "per-run detail : "$RUNS"/*/summary.txt"
echo "poem/shadow    : "$RUNS"/*/pairs.txt"
echo "reading page   : "$RUNS"/poem.html"
