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
    config: dict[str, Any] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    """Per-metric count of pairs excluded for insufficient lexicon coverage."""

    def summary(self) -> str:
        lines = [
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
            "dropped": self.dropped,
            "tests": [t.to_dict() for t in self.tests],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)


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
    lex: Lexicons | None = None,
    n_iter: int = 20000,
    on_progress: Callable[[int, int, GenerationTrace], None] | None = None,
) -> CorpusResult:
    """Generate a corpus, pair each poem with its shadow, and test.

    Seeds are `seed0 + i` over the flattened (prompt, sample) grid, so a run is
    reproducible from `seed0` alone and any single pair can be regenerated.
    """
    lex = lex or DEFAULT
    res = CorpusResult(
        config={
            "model": getattr(backend, "model", "?"),
            "backend": backend.name,
            "n_prompts": len(prompts),
            "samples_per_prompt": samples_per_prompt,
            "rank": rank,
            "gated": gated,
            "gate_top_n": gate_top_n,
            "gate_max_cost": gate_max_cost,
            "temperature": temperature,
            "top_p": top_p,
            "candidates": candidates,
            "max_tokens": max_tokens,
            "seed0": seed0,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )

    total = len(prompts) * samples_per_prompt
    i = 0
    for p_i, prompt in enumerate(prompts):
        for s_i in range(samples_per_prompt):
            seed = seed0 + i
            trace = backend.generate_trace(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                candidates=candidates,
                seed=seed,
                system=system,
            )
            if gated:
                sh = gated_shadow(
                    trace, rank, top_n=gate_top_n, max_cost=gate_max_cost
                )
            else:
                sh = shadow_poem(trace, rank)

            cmp_ = compare(
                trace.text,
                sh.text,
                trace=trace,
                lex=lex,
                rank=rank,
                label=f"p{p_i}s{s_i}",
            )
            res.traces.append(trace)
            res.shadows.append(sh)
            res.comparisons.append(cmp_)
            i += 1
            if on_progress:
                on_progress(i, total, trace)

    res.decision_stats = _decision_stats(res.traces, res.shadows)
    res.tests, res.dropped = _run_tests(res.comparisons, n_iter=n_iter, seed=seed0)
    return res


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
    return paths
