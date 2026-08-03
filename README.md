# shadow-anthology

*what could've been*

Every token a language model writes is the survivor of a competition. The
sampler resolves that competition hundreds of times per poem, and the record of
what it discarded is destroyed the moment the text is displayed.

This reconstructs the record. From a single generation it recovers the **shadow
poem** — the text assembled from the nearest-rejected token at every position
the model actually wrote — and then measures whether the poem that exists
differs systematically from the poem that died at every step.

```bash
python -m shadow_anthology.cli demo      # full pipeline, no dependencies, no model
```

---

## What it produces

| Object | Cost | What it is |
|---|---|---|
| **Shadow poem** | free | best rejected token at every position |
| **Gated shadow** | free | the poem with only its *contested* hinges swapped — usually the legible one |
| **Shadow family** (ranks 1..k) | free | k near-poems off one generation, all sharing the original's shape |
| **Branching anthology** | one call per branch | poems the model genuinely *would* have written, forked at the closest calls |

### The one caveat that matters

**A shadow poem is a counterfactual comb, not a sample.** Each shadow token is
conditioned on the *written* prefix, not the shadow prefix — so it is not a text
the model would have produced had it diverged at position 0 and continued. It is
the pointwise record of every road not taken, laid end to end, holding the
original's exact length and metrical skeleton while replacing all of its
commitments.

That is a feature, not a defect to apologise for: it isolates the *choices* from
everything else about the poem. For texts the model would actually have written,
use `branch` — which pays for real continuations.

---

## Which models can be traced

The method needs one thing: the tokens that **lost**, not just the token that
won. That rules out most chat APIs.

| Backend | Depth | Branching | Notes |
|---|---|---|---|
| `hf` | full top-k | ✅ | local weights. Exact, seedable, unlimited rank depth. The reference backend. |
| `fireworks` | **5** (default) | ✅ | **best hosted option** — see below |
| `openai` | 20 | chat ✗ / completions ✅ | returns raw model logprobs, not the tempered sampling distribution |
| `mock` | full | ✅ | dependency-free fixture for tests and the demo |
| `anthropic` | — | — | **not possible**, raises with an explanation |

### Anthropic / Claude — why not

The Messages API exposes no `logprobs` or `top_logprobs`, so the runner-up token
is unrecoverable from a response. Independently, assistant prefill returns 400 on
Opus 4.6+, Sonnet 4.6+, Opus 5 and Fable 5, which also closes the black-box
workaround of forcing a prefix and resampling to estimate the distribution.

This is worth stating as a result rather than a gap: **the information needed to
credit the sampler exists at generation time in every deployed system, and is
discarded before display.** Where a provider withholds it, the sampler's
authorship cannot be read at all.

### Fireworks — the best hosted option

```bash
export FIREWORKS_API_KEY=...
python -m shadow_anthology.cli trace \
  --backend fireworks \
  --model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --prompt "Write a short poem about winter light on water." \
  --candidates 5 --temperature 1.0 --top-p 0.95 --out trace.json
```

Two advantages over the OpenAI API:

- it returns **`sampling_logprob`** alongside `logprob` — the *tempered*
  distribution the sampler actually drew from, which is exactly the quantity
  this project is about. Ranks therefore reflect the sampler's real preferences
  rather than the model's untempered ones. (Traces record which you got in
  `meta.logprobs_are_raw`.)
- its `/completions` endpoint over open-weights models takes a raw prompt, so a
  forced prefix is just a string — **branching works**.

**Constraint:** `top_logprobs` is capped by the deployment's `--max-logprobs`,
default **5**. That is the chosen token plus ~4 alternatives: enough for rank-1
through rank-3 shadows, not enough for deep-rank anthologies. Raise it on a
dedicated deployment and pass `max_top_logprobs=` to match. Requests above the
cap are clamped and flagged in `meta.server_capped_candidates`, never silently
honoured.

Note the `completions` endpoint does no chat templating — format instruct-model
prompts yourself, or use `--endpoint chat` and give up branching.

---

## Experiments

The analysis pipeline is complete and tested. **No real model has been run yet**
— everything below is ready to execute against `fireworks` or `hf`.

```bash
# E1 — main corpus: poem vs rank-1 shadow, paired tests over 16 prompts
python -m shadow_anthology.cli corpus --prompts prompts/poems.txt \
  --backend fireworks --model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --samples 8 --candidates 5 --temperature 1.0 --out runs/main

# E2 — rank depth: does divergence grow monotonically with rank?
for r in 1 2 3; do
  python -m shadow_anthology.cli corpus --prompts prompts/poems.txt \
    --backend fireworks --rank $r --out runs/rank$r
done

# E3 — temperature sweep: how much of the poem is the sampler?
for t in 0.3 0.7 1.0 1.3; do
  python -m shadow_anthology.cli corpus --prompts prompts/poems.txt \
    --backend fireworks --temperature $t --out runs/T$t
done

# E4 — gated vs full shadow
python -m shadow_anthology.cli corpus --prompts prompts/poems.txt \
  --backend fireworks --top-n 12 --out runs/gated

# E5 — branching anthology: poems the model would have written
python -m shadow_anthology.cli branch --trace trace.json \
  --backend fireworks --points 8 --ranks 1 2 --budget 24 --out anthology.json
```

**E3 is the load-bearing one.** `offrank_fraction` — the share of positions where
the sampler did *not* take the model's preferred token — is the sampler's
editorial footprint, and it is zero under greedy decoding by construction. How it
and the poem/shadow effect sizes move together across temperature is the direct
measurement of how much of the poem belongs to the draw rather than the model.

### Use real lexical norms

The built-in lexicons are **seeds for demonstration only**, and every result is
stamped `seed_lexicons: true` until you replace them. Before reporting anything:

```bash
--concreteness-csv Concreteness_ratings_Brysbaert_et_al_BRM.csv \
--vad-csv Ratings_Warriner_et_al.csv \
--frequency-csv SUBTLEXus.csv
```

---

## How the measurement stays honest

A poem and its shadow share prompt, seed, model, sampling parameters, token count
and line structure, differing only in which branch of each decision was taken —
so nearly every confound of a between-texts comparison is held fixed by
construction. Differences are tested with a **sign-flip permutation test**
(exact in its null, no distributional assumptions), **Holm-corrected** across the
metric family, with percentile bootstrap intervals. No scipy required.

Three deliberate refusals:

1. **A metric whose lexicon covers too little of a text returns `None`, not
   zero.** Averaging three matched words and calling it "tone" is the standard
   way this kind of study manufactures an effect. Under-covered pairs are dropped
   and the dropped count is printed beside every test.
2. **Model surprisal is excluded from composite risk.** The written token
   outranks its shadow by construction, so "the shadow is more surprising" is a
   tautology of the selection rule, not a finding. It's reported as a sanity
   channel only.
3. **There is a null control, asserted as a test.** The mock backend's
   distributions are hashed noise, so no effect exists to find. `pytest` fails if
   the pipeline ever reports a significant difference on it. A paired design over
   matched texts is exactly the setting where a subtly broken pipeline produces
   beautiful spurious results — so this guard runs on every commit.

Aesthetic quality is not measured. The claim is that chosen and rejected poems
*differ*, and that the difference is attributable to the sampler — not that
either is better.

---

## Install

```bash
pip install -e .                 # core: zero dependencies
pip install -e '.[hf]'           # local models (torch + transformers)
pip install -e '.[openai]'       # hosted endpoints (httpx) — incl. Fireworks
pip install -e '.[dev]' && pytest
```

## Library

```python
from shadow_anthology import get_backend, shadow_poem, gated_shadow, write_html

be = get_backend("fireworks", model="accounts/fireworks/models/llama-v3p1-8b-instruct")
trace = be.generate_trace("Write a short poem about salt.", seed=0, candidates=5)

print(trace.text)
print(shadow_poem(trace).text)              # every position swapped
print(gated_shadow(trace, top_n=8).text)    # only the closest calls swapped

print(f"{trace.offrank_fraction:.1%} of tokens were not the model's first choice")
write_html("poem.html", trace)              # self-contained reading interface
```

`write_html` produces a standalone page: the poem as written, every token
carrying its rejected alternatives on hover, contested tokens marked by weight
rather than decoration, and a poem/shadow/both toggle. No external fetches, works
in light and dark.
