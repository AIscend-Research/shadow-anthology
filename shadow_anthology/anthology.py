"""Branching anthologies: the poems the model *would* have written.

`shadow.py` reconstructs counterfactual tokens for free, but its shadow poems
are combs, not samples --- each substitution is conditioned on the original
prefix. To get texts the model would genuinely have produced, we have to pay:
fork at a decision point, force the rejected token, and let generation run to
the end from there.

That gives a tree. The trunk is the poem that exists; every branch is a poem
that was one draw away at a nameable moment. Branch points are chosen where
the decision was closest, so the anthology concentrates on the places where
authorship was least determined --- not on arbitrary positions.

Cost is explicit and bounded: `n_points * len(ranks)` continuations per level,
`budget` caps the total. Nothing is silently truncated; anything the budget
drops is recorded in `Anthology.dropped`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .backends import Backend, BackendUnsupported
from .shadow import Substitution, _substitution
from .trace import Candidate, GenerationTrace


@dataclass
class Branch:
    """One realised alternative poem, plus where and why it split off."""

    branch_at: int
    rank: int
    forced: Candidate
    displaced: Candidate
    cost: float
    """Signed: written-token logprob minus forced-token logprob."""
    gap: float
    """Unsigned distance --- how close this fork was to having happened."""
    entropy: float
    trace: GenerationTrace = field(repr=False)
    parent: "Branch | None" = field(default=None, repr=False)
    depth: int = 1

    @property
    def text(self) -> str:
        return self.trace.text

    @property
    def shared_prefix(self) -> str:
        return self.trace.prefix_text(self.branch_at)

    @property
    def divergent_tail(self) -> str:
        """Everything after the split --- the part that is genuinely new."""
        return self.trace.text[len(self.shared_prefix):]

    def lineage(self) -> list[int]:
        """Branch indices from the trunk down to this branch."""
        out, node = [], self
        while node is not None:
            out.append(node.branch_at)
            node = node.parent
        return list(reversed(out))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_at": self.branch_at,
            "rank": self.rank,
            "depth": self.depth,
            "forced": self.forced.text,
            "displaced": self.displaced.text,
            "cost": self.cost,
            "gap": self.gap,
            "entropy": self.entropy,
            "lineage": self.lineage(),
            "text": self.text,
            "divergent_tail": self.divergent_tail,
        }


@dataclass
class Anthology:
    """A trunk poem and the tree of near-poems around it."""

    trunk: GenerationTrace = field(repr=False)
    branches: list[Branch] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    """Branch points the budget refused. Recorded so coverage is never
    silently overstated."""
    calls: int = 0

    def __len__(self) -> int:
        return len(self.branches)

    def by_depth(self, depth: int) -> list[Branch]:
        return [b for b in self.branches if b.depth == depth]

    def nearest(self, n: int = 5) -> list[Branch]:
        """Branches that were closest to having happened."""
        return sorted(self.branches, key=lambda b: b.gap)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trunk": {
                "text": self.trunk.text,
                "prompt": self.trunk.prompt,
                "model": self.trunk.model,
                "seed": self.trunk.seed,
            },
            "calls": self.calls,
            "n_branches": len(self.branches),
            "dropped": self.dropped,
            "branches": [b.to_dict() for b in self.branches],
        }


def branch_anthology(
    backend: Backend,
    trunk: GenerationTrace,
    *,
    n_points: int = 6,
    ranks: Sequence[int] = (1,),
    by: str = "gap",
    depth: int = 1,
    budget: int | None = None,
    max_tokens: int | None = None,
    min_gap: int = 4,
    seed: int | None = None,
) -> Anthology:
    """Fork the trunk at its most contested positions and generate onward.

    n_points -- branch points per parent, taken from the most contested first.
    ranks    -- which rejected tokens to force (1 = runner-up, 2 = third, ...).
    by       -- 'gap' (closest call, unsigned) or 'entropy' (most diffuse).
    depth    -- recursion depth. depth=2 branches the branches; cost grows
                multiplicatively, so `budget` is strongly recommended above 1.
    min_gap  -- minimum token distance between branch points on one parent,
                so the anthology samples the whole poem instead of clustering
                inside a single contested phrase.
    budget   -- hard ceiling on backend calls. Excess points are recorded in
                `Anthology.dropped`, never dropped silently.
    """
    if not getattr(backend, "supports_forced_prefix", False):
        raise BackendUnsupported(
            f"backend {backend.name!r} cannot resume from a forced prefix, so it "
            "cannot build a branching anthology. Use shadow.shadow_family() for "
            "the zero-cost alternative, or switch to the 'hf' backend."
        )

    anth = Anthology(trunk=trunk)
    frontier: list[tuple[GenerationTrace, Branch | None]] = [(trunk, None)]

    for level in range(1, depth + 1):
        next_frontier: list[tuple[GenerationTrace, Branch | None]] = []
        for parent_trace, parent_branch in frontier:
            points = _select_points(parent_trace, n_points, by, min_gap)
            for step_index, sub in points:
                for rank in ranks:
                    alt = parent_trace.steps[step_index].alternative(rank)
                    if alt is None:
                        continue
                    if budget is not None and anth.calls >= budget:
                        anth.dropped.append(
                            {
                                "branch_at": step_index,
                                "rank": rank,
                                "depth": level,
                                "reason": "budget_exhausted",
                            }
                        )
                        continue

                    child = backend.continue_from(
                        parent_trace,
                        step_index,
                        alt,
                        max_tokens=max_tokens or len(parent_trace),
                        seed=seed,
                    )
                    anth.calls += 1
                    signed = parent_trace.steps[step_index].chosen.logprob - alt.logprob
                    b = Branch(
                        branch_at=step_index,
                        rank=rank,
                        forced=alt,
                        displaced=parent_trace.steps[step_index].chosen,
                        cost=signed,
                        gap=abs(signed),
                        entropy=parent_trace.steps[step_index].entropy,
                        trace=child,
                        parent=parent_branch,
                        depth=level,
                    )
                    anth.branches.append(b)
                    next_frontier.append((child, b))
        frontier = next_frontier
        if not frontier:
            break

    return anth


def _select_points(
    trace: GenerationTrace, n: int, by: str, min_gap: int
) -> list[tuple[int, Substitution]]:
    """Most-contested positions, thinned so they are at least `min_gap` apart."""
    scored: list[tuple[float, int, Substitution]] = []
    for step in trace.steps:
        alt = step.alternative(1)
        if alt is None:
            continue
        sub = _substitution(trace, step, alt)
        score = -step.entropy if by == "entropy" else sub.gap
        if not math.isfinite(score):
            continue
        scored.append((score, step.index, sub))

    scored.sort(key=lambda t: t[0])
    picked: list[tuple[int, Substitution]] = []
    for _, idx, sub in scored:
        if any(abs(idx - j) < min_gap for j, _ in picked):
            continue
        picked.append((idx, sub))
        if len(picked) >= n:
            break
    return sorted(picked, key=lambda t: t[0])
