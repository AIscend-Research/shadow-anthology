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
| `hf` **(default)** | full top-k | ✅ | local weights. Exact, seedable, unlimited rank depth, free. **The only backend measured to work end to end.** |
| `fireworks` | ~5 | endpoint-dependent | hosted, but see the finding below — the catalogue tested could not support the method |
| `openai` | 20 | chat ✗ / completions ✅ | returns raw model logprobs, not the tempered sampling distribution |
| `mock` | full | ✅ | dependency-free fixture for tests and the demo |
| `anthropic` | — | — | **not possible**, raises with an explanation |

### Measured finding: hosted reasoning models cannot be traced

Probing all 18 generative models on one Fireworks account (2026): three return
no logprobs at all, and **every** remaining one is a reasoning model. That is
fatal in a non-obvious way. A reasoning model composes the poem *inside its
chain of thought* and then transcribes it, so the poem you receive is a copy:

| span | mean entropy | mean margin | off-argmax |
|---|---|---|---|
| reasoning | 0.999 bits | 2.77 nats | 28.5% |
| **poem** | **0.056 bits** | **8.65 nats** | **0.0%** |

The sampler made no choices in the poem span. Reconstructing its "rejected"
tokens would mean reporting alternatives that were never in contention. The
run aborts on this now (`entropy < 0.15 bits`) rather than producing a
confident, meaningless result.

This sharpens the agency argument rather than blocking it: it is not only that
providers *withhold* the choice record, it is that an architecture can move
the real choosing somewhere the API never exposes.

### Anthropic / Claude — why not

The Messages API exposes no `logprobs` or `top_logprobs`, so the runner-up token
is unrecoverable from a response. Independently, assistant prefill returns 400 on
Opus 4.6+, Sonnet 4.6+, Opus 5 and Fable 5, which also closes the black-box
workaround of forcing a prefix and resampling to estimate the distribution.

This is worth stating as a result rather than a gap: **the information needed to
credit the sampler exists at generation time in every deployed system, and is
discarded before display.** Where a provider withholds it, the sampler's
authorship cannot be read at all.

### Fireworks — hosted, with caveats

```bash
export FIREWORKS_API_KEY=...
bash scripts/find_poet.sh          # probe every model: logprobs? verse or reasoning?
BACKEND=fireworks bash scripts/run_all.sh
```

Genuine advantages when a suitable model exists: it returns **`sampling_logprob`**
alongside `logprob` — the *tempered* distribution the sampler drew from, which is
exactly what this project measures and which OpenAI does not expose (traces record
which you got in `meta.logprobs_are_raw`). Its `/completions` endpoint also takes a
raw prompt, so a forced prefix is just a string and branching works.

Constraints: `top_logprobs` is capped by the deployment's `--max-logprobs`
(default **5** — chosen token plus ~4 alternatives, enough for rank-1..3);
requests above it are clamped and flagged in `meta.server_capped_candidates`,
never silently honoured. `/completions` does no chat templating, so format
instruct prompts yourself or use `--endpoint chat` and give up branching. And
on the catalogue tested, no model cleared the reasoning problem above.

---

## Experiments

```bash
pip install -e '.[hf,dev]'
bash scripts/get_norms.sh                  # once
SAMPLES=2 PERM=2000 bash scripts/run_all.sh   # validate (~5 min)
SAMPLES=16 bash scripts/run_all.sh            # full suite
```

Only **E1, E3 and E5 generate**. E2 and E4 re-analyse traces already on disk —
free, and *required* to work that way: regenerating per rank would compare rank 1
against rank 2 across different poems and silently destroy the pairing the whole
design rests on.

| | what | generates? |
|---|---|---|
| **E1** | main corpus, T=1.0, poem vs rank-1 shadow | yes |
| **E2** | rank depth 1/2/3 — does divergence grow monotonically? | no, reuses E1 |
| **E3** | temperature sweep 0.3/0.7/1.0/1.3 | yes |
| **E4** | gated shadow — only the 12 closest calls swapped | no, reuses E1 |
| **E5** | branching anthology + HTML reading page | yes |

**E3 is the load-bearing one.** `offrank_fraction` — the share of positions where
the sampler did *not* take the model's preferred token — is the sampler's
editorial footprint, and it is zero under greedy decoding by construction. How it
and the poem/shadow effect sizes move together across temperature is the direct
measurement of how much of the poem belongs to the draw rather than the model.

The runner refuses to produce a meaningless corpus: the smoke test validates its
own output before E1, a sanity gate aborts if the corpus is prose or reasoning
rather than verse, and a deterministic-span check stops runs where the sampler
had no real choices to make.

### Lexical norms — required

```bash
bash scripts/get_norms.sh     # ~5MB, once
```

Fetches Brysbaert concreteness (39,954 lemmas) and Warriner valence/arousal
(13,915 lemmas); word frequency is derived from the SUBTLEX counts shipped inside
the concreteness file, so there is no third download. `data/norms/` is picked up
automatically — no flags — and overridable with `--concreteness-csv` / `--vad-csv`
/ `--frequency-csv`.

**Not optional.** Without them the seed lexicons cover too little real vocabulary
and the coverage guard drops every pair: concreteness, imagery, valence and
arousal all report `n=0`, leaving only the lexical-risk half of the study. With
them, coverage on generated poetry runs ~82% concreteness and ~42%
valence/arousal. Every result carries `seed_lexicons: true/false`, so
demonstration numbers can never be mistaken for norm-backed ones.

### Cost and runtime

Local (`BACKEND=hf`, the default) costs **nothing** and runs on CPU/MPS. Expect
roughly **1–1.5 hours** for all four arms at `SAMPLES=16` — there is no API
concurrency, since one torch module cannot serve parallel generate loops. Use
`SAMPLES=8` (~40 min) or `Qwen/Qwen2.5-0.5B-Instruct` (2–3× faster, weaker verse)
if that is too slow; since cost is zero, `SAMPLES=32` is equally viable if you can
leave it running.

Hosted (`BACKEND=fireworks`) is minutes rather than hours at `--concurrency 8`,
and cents rather than dollars — an 8B-class model sits in the $0.20/1M band, so
the full suite is well under $1 even with reasoning tokens. The blocker there is
model availability, not price.

---

## Results

Qwen2.5-1.5B-Instruct, 16 prompts x 16 samples = 256 poems per arm, four
temperatures, ranks 1-3, published norms. Full output in `runs/`.

**1. How much of the poem is available to decide is itself a function of
temperature.** This is the finding.

| T | off-argmax | **positions with any alternative** | entropy |
|---|---|---|---|
| 0.3 | 12.7% | **47.3%** | 0.46 bits |
| 0.7 | 33.3% | **81.2%** | 1.45 bits |
| 1.0 | 48.2% | **92.6%** | 2.30 bits |
| 1.3 | 59.7% | **97.1%** | 2.91 bits |

At T=0.3 more than half the poem has no second candidate at all. Those tokens
were not chosen over alternatives; they were the only thing in the sampler's
support. The sampler's authorship does not merely weaken at low temperature --
across most of the text it does not exist.

**2. The poem and its nearest-rejected sibling do not differ in imagery or
tone.** Every lexical difference that reaches significance is an artifact of
probability rank, established by two controls:

- `scripts/rank_control.py` — comparing the poem against rank-1, 2 and 3
  shadows. Six of eleven metrics (rarity, repetition, risk, type-token ratio,
  word length, concreteness) grow monotonically with rank, so they track how
  far down the ranking you went, not what the sampler chose. `type_token_ratio`
  and `mean_word_length` **reverse sign** between rank 1 and rank 2.
- `scripts/across_arm_correction.py` — Holm and Benjamini-Hochberg over all 77
  unique tests, not 11 per arm. `valence` and `arousal`, the only imagery/tone
  metrics that reached significance in any single arm, **do not survive**
  (Holm p = 0.14 and 0.13); they hold in 1 of 4 temperature conditions.

`imagery`, `sensory_density` and `abstract_ratio` are null at every rank and
every temperature.

> The sampler's choice does not measurably change what kind of poem you get.
> What it changes is whether there was a poem to choose between at all.

This is a negative result on the original hypothesis. It is reported as one.

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
pip install -e '.[hf,dev]'       # local models (torch + transformers) + pytest
bash scripts/get_norms.sh        # psycholinguistic norms, ~5MB
pytest                           # 53 tests

pip install -e .                 # core only: zero dependencies, mock backend
pip install -e '.[openai]'       # hosted endpoints (httpx) — incl. Fireworks
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
