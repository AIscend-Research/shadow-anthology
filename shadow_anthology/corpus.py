"""Corpus-scale comparison of poems against their nearest-rejected siblings.

One poem and one shadow is an anecdote. The claim --- that the chosen poem
differs systematically from the poem rejected at every step --- is a claim
about a distribution, so this module runs the pipeline over many prompts and
seeds, pairs each poem with its own shadow, and applies paired tests with
family-wise correction.

The pairing is the point. Poem and shadow share prompt, seed, model, sampling
parameters, token count and line structure; they differ only in which branch
of each decision was taken. Almost every confound you would worry about in a
between-texts comparison is held fixed by construction.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .backends import Backend
from .lexicons import DEFAULT, Lexicons
from .metrics import COMPARABLE, Comparison, compare, measure_shadow_surprisal
from .shadow import ShadowPoem, coin_flips, gated_shadow, shadow_poem
from .stats import TestResult, holm_bonferroni, paired_permutation_test
from .trace import GenerationTrace, save_traces


@dataclass
class CorpusResult:
    """Everything a corpus run produced, plus the tests over it."""

    comparisons: list[Comparison] = field(default_factory=list)
    tests: list[TestResult] = field(default_factory=list)
    traces: list[GenerationTrace] = field(default_factory=list, repr=False)
    shadows: list[ShadowPoem] = field(default_factory=list, repr=False)
    decision_stats: dict[str, float] = field(default_factory=dict)
    sanity: dict[str, Any] = field(default_factory=dict)
    """Whether these texts are plausibly poems at all. See `text_sanity`."""
    config: dict[str, Any] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    """Per-metric count of pairs excluded for insufficient lexicon coverage."""

    def summary(self) -> str:
        warns = sanity_warnings(self.sanity)
        lines = []
        if warns:
            lines += ["!" * 70, "CORPUS SANITY WARNING -- read before the statistics below:"]
            for w in warns:
                lines.append("  * " + w)
            lines += ["!" * 70, ""]
        lines += [
            f"shadow-anthology corpus: {len(self.comparisons)} poem/shadow pairs",
            f"  model={self.config.get('model')} backend={self.config.get('backend')} "
            f"rank={self.config.get('rank')} T={self.config.get('temperature')}",
            "",
            "Sampler footprint (how much of the text is the draw, not the model):",
        ]
        for k, v in self.decision_stats.items():
            lines.append(f"  {k:<28} {v:.4f}")
        lines += ["", "Paired tests, poem minus shadow (Holm-corrected):"]
        for t in sorted(self.tests, key=lambda t: t.p_adjusted or t.p_value):
            lines.append("  " + str(t))
            if self.dropped.get(t.name):
                lines.append(f"      ({self.dropped[t.name]} pairs dropped: low coverage)")
        if any(c.poem.seed_lexicons for c in self.comparisons):
            lines += [
                "",
                "NOTE: scored with built-in SEED lexicons. Load published norms",
                "      (Brysbaert / Warriner / SUBTLEX) before reporting these.",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "n_pairs": len(self.comparisons),
            "decision_stats": self.decision_stats,
            "sanity": self.sanity,
            "sanity_warnings": sanity_warnings(self.sanity),
            "dropped": self.dropped,
            "tests": [t.to_dict() for t in self.tests],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)


def generate_traces(
    backend: Backend,
    prompts: Sequence[str],
    *,
    samples_per_prompt: int = 4,
    max_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.95,
    candidates: int = 20,
    system: str | None = None,
    seed0: int = 0,
    concurrency: int = 1,
    checkpoint: str | None = None,
    on_progress: Callable[[int, int, GenerationTrace], None] | None = None,
) -> list[GenerationTrace]:
    """Generate the (prompt x sample) grid. This is the only part that costs.

    Seeds are `seed0 + i` over the flattened grid, so a run is reproducible
    from `seed0` alone and any single poem can be regenerated on its own.

    `concurrency > 1` issues requests in parallel, which is worth doing against
    a hosted API (generation is the entire wall-clock cost) and is **not** safe
    for a single local `hf` model --- one torch module cannot serve concurrent
    generate loops. Results are always returned in grid order regardless of
    completion order, so a run is deterministic no matter the concurrency.

    `checkpoint` is a JSONL path each trace is appended to as it completes.
    On restart, already-generated grid positions are loaded and skipped, so an
    interrupted arm resumes mid-way instead of starting over. Because seeds are
    a deterministic function of grid position, a resumed arm is identical to
    one produced in a single pass.
    """
    grid = [
        (i, p_i, s_i, prompt)
        for i, (p_i, s_i, prompt) in enumerate(
            (p_i, s_i, prompt)
            for p_i, prompt in enumerate(prompts)
            for s_i in range(samples_per_prompt)
        )
    ]

    def one(item: tuple[int, int, int, str]) -> tuple[int, GenerationTrace]:
        i, _p, _s, prompt = item
        return i, backend.generate_trace(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidates=candidates,
            seed=seed0 + i,
            system=system,
        )

    out: dict[int, GenerationTrace] = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out[int(rec["_grid_index"])] = GenerationTrace.from_dict(rec)
                except Exception:
                    continue  # truncated final line from a hard kill
        if out:
            print(
                f"  resuming: {len(out)}/{len(grid)} already generated",
                file=sys.stderr,
            )
        grid = [g for g in grid if g[0] not in out]

    ckpt_fh = open(checkpoint, "a", encoding="utf-8") if checkpoint else None
    lock = threading.Lock()

    def record(i: int, tr: GenerationTrace) -> None:
        out[i] = tr
        if ckpt_fh is not None:
            with lock:
                d = tr.to_dict()
                d["_grid_index"] = i
                ckpt_fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                ckpt_fh.flush()

    done = len(out)
    total = done + len(grid)
    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, item) for item in grid]
            for fut in as_completed(futures):
                i, tr = fut.result()
                record(i, tr)
                done += 1
                if on_progress:
                    on_progress(done, total, tr)
    else:
        for item in grid:
            i, tr = one(item)
            record(i, tr)
            done += 1
            if on_progress:
                on_progress(done, total, tr)

    if ckpt_fh is not None:
        ckpt_fh.close()
    return [out[i] for i in sorted(out)]


def analyse_traces(
    traces: Sequence[GenerationTrace],
    *,
    rank: int = 1,
    gated: bool = False,
    gate_top_n: int | None = None,
    gate_max_cost: float | None = None,
    lex: Lexicons | None = None,
    n_iter: int = 20000,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> CorpusResult:
    """Pair each poem with its shadow and test. Costs nothing --- no generation.

    Rank sweeps and gating comparisons **must** go through this function on one
    fixed set of traces. Regenerating per rank would compare rank 1 against
    rank 2 across different poems, which silently destroys the pairing that the
    whole design rests on.
    """
    lex = lex or DEFAULT
    res = CorpusResult(
        config={
            **{
                "model": traces[0].model if traces else "?",
                "backend": traces[0].backend if traces else "?",
                "n_traces": len(traces),
                "rank": rank,
                "gated": gated,
                "gate_top_n": gate_top_n,
                "gate_max_cost": gate_max_cost,
                "analysed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            **(config or {}),
        }
    )

    for i, trace in enumerate(traces):
        sh = (
            gated_shadow(trace, rank, top_n=gate_top_n, max_cost=gate_max_cost)
            if gated
            else shadow_poem(trace, rank)
        )
        res.traces.append(trace)
        res.shadows.append(sh)
        res.comparisons.append(
            compare(trace.text, sh.text, trace=trace, lex=lex, rank=rank, label=f"t{i}")
        )

    res.decision_stats = _decision_stats(res.traces, res.shadows)
    res.sanity = text_sanity(res.traces)
    res.tests, res.dropped = _run_tests(res.comparisons, n_iter=n_iter, seed=seed)
    return res


def run_corpus(
    backend: Backend,
    prompts: Sequence[str],
    *,
    samples_per_prompt: int = 4,
    rank: int = 1,
    gated: bool = False,
    gate_top_n: int | None = None,
    gate_max_cost: float | None = None,
    max_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.95,
    candidates: int = 20,
    system: str | None = None,
    seed0: int = 0,
    concurrency: int = 1,
    lex: Lexicons | None = None,
    n_iter: int = 20000,
    on_progress: Callable[[int, int, GenerationTrace], None] | None = None,
) -> CorpusResult:
    """Generate a corpus and analyse it: `generate_traces` then `analyse_traces`.

    To sweep rank or gating, call `generate_traces` once and `analyse_traces`
    repeatedly over the same traces --- do not call this function per setting.
    """
    traces = generate_traces(
        backend, prompts,
        samples_per_prompt=samples_per_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        candidates=candidates,
        system=system,
        seed0=seed0,
        concurrency=concurrency,
        on_progress=on_progress,
    )
    return analyse_traces(
        traces,
        rank=rank,
        gated=gated,
        gate_top_n=gate_top_n,
        gate_max_cost=gate_max_cost,
        lex=lex,
        n_iter=n_iter,
        seed=seed0,
        config={
            "n_prompts": len(prompts),
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "candidates": candidates,
            "max_tokens": max_tokens,
            "seed0": seed0,
            "concurrency": concurrency,
        },
    )


def _run_tests(
    comparisons: Sequence[Comparison], *, n_iter: int, seed: int
) -> tuple[list[TestResult], dict[str, int]]:
    tests: list[TestResult] = []
    dropped: dict[str, int] = {}
    for metric in COMPARABLE:
        vals = [c.deltas.get(metric) for c in comparisons]
        usable = [v for v in vals if v is not None]
        dropped[metric] = len(vals) - len(usable)
        tests.append(
            paired_permutation_test(usable, n_iter=n_iter, seed=seed, name=metric)
        )
    return holm_bonferroni(tests), dropped


def _decision_stats(
    traces: Sequence[GenerationTrace], shadows: Sequence[ShadowPoem]
) -> dict[str, float]:
    if not traces:
        return {}
    n = len(traces)

    def avg(f: Callable[[Any], float], xs: Sequence[Any]) -> float:
        return sum(f(x) for x in xs) / len(xs) if xs else 0.0

    coin = [len(coin_flips(t, 0.1)) / max(1, len(t)) for t in traces]
    surp_gap = []
    for t in traces:
        sh = measure_shadow_surprisal(t, 1)
        if sh is not None and len(t):
            poem = sum(s.chosen.surprisal for s in t.steps) / len(t)
            surp_gap.append(sh - poem)

    return {
        "mean_tokens": avg(len, traces),
        "mean_entropy_bits": avg(lambda t: t.mean_entropy, traces),
        "mean_top1_top2_margin_nats": avg(lambda t: t.mean_margin, traces),
        "offrank_fraction": avg(lambda t: t.offrank_fraction, traces),
        "coin_flip_rate_at_0.1nats": sum(coin) / n,
        "shadow_minus_poem_surprisal_bits": (
            sum(surp_gap) / len(surp_gap) if surp_gap else 0.0
        ),
        "mean_shadow_divergence_rate": avg(lambda s: s.divergence_rate, shadows),
    }


POEM_MARKER = "===POEM==="
"""Delimiter we ask the model to emit between its reasoning and the poem.

Every model on some providers' catalogues is a reasoning model whose visible
output is its chain of thought. Rather than fight that, we let the model think
and ask it to mark where the poem begins, then analyse only the poem's tokens.
"""


def slice_to_poem(
    traces: Sequence[GenerationTrace],
    marker: str = POEM_MARKER,
    *,
    min_tokens: int = 12,
) -> tuple[list[GenerationTrace], dict[str, Any]]:
    """Keep only the post-marker span of each trace.

    Traces with no marker, or with too little text after it, are dropped
    rather than analysed whole --- a trace whose marker never appeared is a
    trace where the model never got to the poem, and including it would put
    reasoning prose back into the corpus.

    Returns (sliced_traces, report). The report is surfaced in the summary so
    a high drop rate is visible instead of quietly shrinking the corpus.
    """
    kept: list[GenerationTrace] = []
    no_marker = too_short = looped = 0
    for t in traces:
        spans = t.marker_spans(marker)
        if not spans:
            no_marker += 1
            continue
        start = spans[0][1]
        # Models that loop re-emit the whole block. Cut at whichever loop
        # signal comes FIRST: the next marker, or an echo of the system or
        # user prompt. Cutting on the marker alone is not enough, because the
        # repetition usually begins by echoing the prompt *before* reaching
        # the marker again.
        cuts = [s[0] for s in spans[1:]]
        for echo in (t.system, t.prompt):
            if not echo:
                continue
            probe = echo.strip()[:40]
            if len(probe) < 12:
                continue
            cuts += [s[0] for s in t.marker_spans(probe) if s[0] > start]
        end = min(cuts) if cuts else None
        if cuts:
            looped += 1
        sub = t.slice_steps(start, end)
        if len(sub) < min_tokens:
            too_short += 1
            continue
        kept.append(sub)
    return kept, {
        "input_traces": len(traces),
        "kept": len(kept),
        "dropped_no_marker": no_marker,
        "dropped_too_short": too_short,
        "truncated_at_repeat": looped,
        "marker": marker,
    }


REASONING_MARKERS = (
    "okay,", "okay ", "hmm,", "the user wants", "the user is asking",
    "i should", "i need to", "let me", "first, i", "we need to",
    "this is a request", "let's ", "wait,",
)


def text_sanity(traces: Sequence[GenerationTrace]) -> dict[str, Any]:
    """Is this corpus actually made of poems?

    Written after a 256-pair run in which every "poem" turned out to be a
    reasoning model's visible chain of thought, truncated before it wrote any
    verse. The paired statistics were valid and meant nothing, because the
    texts were not poems. A pipeline that can measure the wrong genre without
    complaining is worse than one that crashes.
    """
    if not traces:
        return {}
    n = len(traces)
    reasoning = json_like = truncated = 0
    line_lens: list[float] = []

    for t in traces:
        head = t.text.strip()[:120].lower()
        if any(m in head for m in REASONING_MARKERS):
            reasoning += 1
        if t.text.count('":') > 2 or t.text.count("{") > 2:
            json_like += 1
        cap = t.params.get("max_tokens")
        if cap and len(t) >= cap:
            truncated += 1
        lines = [ln for ln in t.text.splitlines() if ln.strip()]
        if lines:
            line_lens.append(sum(len(ln) for ln in lines) / len(lines))

    return {
        "reasoning_preamble_rate": reasoning / n,
        "json_like_rate": json_like / n,
        "hit_token_cap_rate": truncated / n,
        "mean_line_chars": sum(line_lens) / len(line_lens) if line_lens else 0.0,
    }


def sanity_warnings(s: dict[str, Any]) -> list[str]:
    """Human-readable objections to treating this corpus as poetry."""
    out = []
    if s.get("reasoning_preamble_rate", 0) > 0.2:
        out.append(
            f"{s['reasoning_preamble_rate']:.0%} of texts open with reasoning-model "
            "preamble ('Okay,', 'The user wants', 'I should'). These are chains of "
            "thought, not poems -- the comparison below is measuring the wrong genre. "
            "Use a non-reasoning model, or a system prompt that forbids preamble."
        )
    if s.get("json_like_rate", 0) > 0.2:
        out.append(
            f"{s['json_like_rate']:.0%} of texts look like JSON/structured data, not "
            "verse. This is the signature of an untemplated completions prompt: the "
            "base model continued your prompt as data rather than answering it."
        )
    if s.get("hit_token_cap_rate", 0) > 0.5:
        out.append(
            f"{s['hit_token_cap_rate']:.0%} of generations hit the max_tokens cap, so "
            "the texts are truncated mid-thought rather than finished poems."
        )
    if s.get("mean_line_chars", 0) > 90:
        out.append(
            f"mean line length is {s['mean_line_chars']:.0f} characters -- that is "
            "prose, not lineated verse."
        )
    return out


def load_prompts(path: str) -> list[str]:
    """One prompt per line; blank lines and `#` comments ignored."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def save_corpus(res: CorpusResult, outdir: str) -> dict[str, str]:
    """Write results, traces and readable text side by side."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "results": os.path.join(outdir, "results.json"),
        "traces": os.path.join(outdir, "traces.jsonl"),
        "texts": os.path.join(outdir, "pairs.txt"),
        "summary": os.path.join(outdir, "summary.txt"),
    }
    res.save(paths["results"])
    save_traces(res.traces, paths["traces"])
    with open(paths["texts"], "w", encoding="utf-8") as fh:
        for t, s, c in zip(res.traces, res.shadows, res.comparisons):
            fh.write(f"=== {c.label}  seed={t.seed}  prompt={t.prompt!r}\n")
            fh.write("--- poem\n" + t.text.strip() + "\n")
            fh.write(f"--- shadow (rank {s.rank}, {s.n_substitutions} subs)\n")
            fh.write(s.text.strip() + "\n\n")
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        fh.write(res.summary() + "\n")
    # The arm is complete; the resume checkpoint is now redundant.
    partial = os.path.join(outdir, "traces.partial.jsonl")
    if os.path.exists(partial):
        os.remove(partial)
    return paths
