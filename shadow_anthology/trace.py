"""Data model for a sampler trace.

A trace is the record of every decision the sampler made while writing one poem:
at each position, which token was emitted, and which tokens were in contention.
Everything downstream --- shadow poems, branching anthologies, comparative
metrics --- is a view over this object.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    """One token in contention at a single position."""

    token_id: int
    text: str
    logprob: float
    """Log-probability under the distribution the sampler actually drew from
    (i.e. after temperature and any truncation). See `Candidate.raw_logprob`
    for the model's untempered value where the backend can supply it."""
    raw_logprob: float | None = None

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def surprisal(self) -> float:
        """Bits of surprise. Monotone in -logprob; reported in bits for readability."""
        return -self.logprob / math.log(2)


@dataclass
class TokenStep:
    """One decision point.

    `candidates` is ordered by descending logprob and always contains the chosen
    token. `chosen_rank` is the chosen token's index in that list --- 0 for a
    greedy-agreeing draw, higher when the sampler took a road less probable.
    """

    index: int
    chosen: Candidate
    candidates: list[Candidate]
    chosen_rank: int

    # ---- decision geometry -------------------------------------------------

    @property
    def top(self) -> Candidate:
        return self.candidates[0]

    @property
    def margin(self) -> float:
        """logprob gap between the top-ranked and second-ranked candidate.

        Small margin == a genuine coin-flip; large margin == the model was
        never really choosing. This is the primary knob for locating the
        moments where the poem could most plausibly have gone otherwise.
        """
        if len(self.candidates) < 2:
            return float("inf")
        return self.candidates[0].logprob - self.candidates[1].logprob

    @property
    def entropy(self) -> float:
        """Shannon entropy (bits) of the renormalised candidate distribution.

        NOTE: computed over the retained top-k only, so it is a *lower bound*
        on the true entropy. With k >= 20 the truncated mass is usually
        negligible for poetry-scale distributions, but the bound is worth
        remembering when comparing across backends with different k.
        """
        ps = [c.prob for c in self.candidates]
        z = sum(ps)
        if z <= 0:
            return 0.0
        h = 0.0
        for p in ps:
            q = p / z
            if q > 0:
                h -= q * math.log(q, 2)
        return h

    @property
    def retained_mass(self) -> float:
        """Total probability captured by the retained candidates."""
        return min(1.0, sum(c.prob for c in self.candidates))

    def alternatives(self) -> list[Candidate]:
        """Candidates that would render differently, best-first.

        `alternatives()[0]` is *the* nearest-rejected token: the shadow.

        Candidates whose text is identical to the chosen token's are excluded,
        not merely the chosen index. Real tokenizers routinely carry several
        distinct token ids that decode to the same string, and one of those
        surfacing as the "runner-up" produces a substitution that changes
        nothing on the page --- an invisible divergence that would still be
        counted, inflating the substitution rate and diluting every measured
        poem-vs-shadow difference toward zero.
        """
        return [
            c
            for i, c in enumerate(self.candidates)
            if i != self.chosen_rank and c.text != self.chosen.text
        ]

    def alternative(self, rank: int = 1) -> Candidate | None:
        """The `rank`-th rejected token (1 = runner-up, 2 = third choice, ...)."""
        alts = self.alternatives()
        if rank < 1 or rank > len(alts):
            return None
        return alts[rank - 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chosen_rank": self.chosen_rank,
            "candidates": [asdict(c) for c in self.candidates],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TokenStep":
        cands = [Candidate(**c) for c in d["candidates"]]
        rank = d["chosen_rank"]
        return TokenStep(
            index=d["index"], chosen=cands[rank], candidates=cands, chosen_rank=rank
        )


@dataclass
class GenerationTrace:
    """A complete generation plus the decision record behind it."""

    text: str
    steps: list[TokenStep]
    prompt: str
    model: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)
    system: str | None = None
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[TokenStep]:
        return iter(self.steps)

    # ---- aggregate decision statistics -------------------------------------

    @property
    def mean_entropy(self) -> float:
        return _mean([s.entropy for s in self.steps])

    @property
    def mean_margin(self) -> float:
        finite = [s.margin for s in self.steps if math.isfinite(s.margin)]
        return _mean(finite)

    @property
    def offrank_fraction(self) -> float:
        """Share of positions where the sampler did *not* take the argmax.

        This is the sampler's visible editorial footprint: at temperature 0 it
        is zero by construction, and every unit above zero is a place where the
        text exists because of the draw rather than the model's preference.
        """
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.chosen_rank != 0) / len(self.steps)

    def decision_points(self, n: int | None = None, by: str = "margin") -> list[TokenStep]:
        """Steps ranked by how contested they were, most contested first.

        by='margin'  -> smallest top-1/top-2 gap (closest call)
        by='entropy' -> highest entropy (most diffuse field of options)
        by='rank'    -> sampler deviated furthest from the argmax
        """
        if by == "margin":
            ordered = sorted(self.steps, key=lambda s: s.margin)
        elif by == "entropy":
            ordered = sorted(self.steps, key=lambda s: -s.entropy)
        elif by == "rank":
            ordered = sorted(self.steps, key=lambda s: (-s.chosen_rank, s.margin))
        else:
            raise ValueError(f"unknown ranking: {by!r}")
        return ordered[:n] if n else ordered

    def prefix_text(self, upto: int) -> str:
        """The generated text up to (not including) step `upto`."""
        return "".join(s.chosen.text for s in self.steps[:upto])

    def slice_steps(self, start: int, end: int | None = None) -> "GenerationTrace":
        """A new trace over `steps[start:end]`, re-indexed from zero.

        The per-position decision record is unchanged --- each step keeps the
        candidates the sampler actually saw, which were conditioned on the full
        preceding context. Only which positions we *analyse* changes.

        This is what makes reasoning models usable: their visible output is
        chain-of-thought followed by the answer, and slicing to the answer span
        lets us measure the poem's decisions without the monologue's.
        """
        sub = self.steps[start:end]
        steps = [
            TokenStep(
                index=i,
                chosen=s.chosen,
                candidates=s.candidates,
                chosen_rank=s.chosen_rank,
            )
            for i, s in enumerate(sub)
        ]
        return GenerationTrace(
            text="".join(s.chosen.text for s in steps),
            steps=steps,
            prompt=self.prompt,
            model=self.model,
            backend=self.backend,
            params=dict(self.params),
            system=self.system,
            seed=self.seed,
            meta={**self.meta, "sliced_from": [start, end], "original_len": len(self)},
        )

    def marker_spans(self, marker: str) -> list[tuple[int, int]]:
        """Step spans of every occurrence of `marker`, as (start, end_exclusive).

        Matching is on accumulated text, not token boundaries, since a marker is
        usually split across several tokens and rarely aligns with any of them.
        `end` is the first step index *after* the marker finishes --- i.e. where
        the text following it begins.
        """
        if not marker:
            return []
        # char offset at which each step's text begins
        offsets, acc = [], 0
        for s in self.steps:
            offsets.append(acc)
            acc += len(s.chosen.text)
        text = self.text if len(self.text) == acc else "".join(
            s.chosen.text for s in self.steps
        )

        def step_of(char_i: int) -> int:
            lo, hi = 0, len(offsets) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if offsets[mid] <= char_i:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        spans, at = [], text.find(marker)
        while at != -1:
            spans.append((step_of(at), min(len(self.steps), step_of(at + len(marker)) + 1)))
            at = text.find(marker, at + len(marker))
        return spans

    def find_marker(self, marker: str) -> int | None:
        """First step index *after* `marker` finishes, or None if absent."""
        spans = self.marker_spans(marker)
        return spans[0][1] if spans else None

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "text": self.text,
            "prompt": self.prompt,
            "system": self.system,
            "model": self.model,
            "backend": self.backend,
            "params": self.params,
            "seed": self.seed,
            "meta": self.meta,
            "steps": [s.to_dict() for s in self.steps],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GenerationTrace":
        v = d.get("schema_version", 1)
        if v != SCHEMA_VERSION:
            raise ValueError(f"trace schema v{v} != supported v{SCHEMA_VERSION}")
        return GenerationTrace(
            text=d["text"],
            steps=[TokenStep.from_dict(s) for s in d["steps"]],
            prompt=d["prompt"],
            model=d["model"],
            backend=d["backend"],
            params=d.get("params", {}),
            system=d.get("system"),
            seed=d.get("seed"),
            meta=d.get("meta", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)

    @staticmethod
    def load(path: str) -> "GenerationTrace":
        with open(path, encoding="utf-8") as fh:
            return GenerationTrace.from_dict(json.load(fh))


def save_traces(traces: Sequence[GenerationTrace], path: str) -> None:
    """Write many traces as JSON Lines (one trace per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: str) -> list[GenerationTrace]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(GenerationTrace.from_dict(json.loads(line)))
    return out


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else 0.0
