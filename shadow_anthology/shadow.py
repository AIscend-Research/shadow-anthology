"""Shadow poems: the text assembled from what the sampler rejected.

Given one trace, we can read off, at every position, the token that came
second. Concatenating those gives the *shadow poem* --- a text that was never
generated but was, at every single step, one draw away.

A caveat that matters, and that we do not paper over:

    The shadow poem is a COUNTERFACTUAL COMB, NOT A SAMPLE.

Each shadow token is conditioned on the *actual* prefix, not on the shadow
prefix. So the shadow poem is not a text the model would have produced had it
diverged at position 0 and kept going --- it is the pointwise record of every
road not taken, laid end to end. That is a legitimate and, we argue, more
interesting object than a resampled poem: it holds the original poem's exact
shape while replacing its every commitment. For texts the model *would* have
written, branch instead --- see `anthology.py`, which pays for real
continuations.

Two reading modes are provided:

  * `shadow_poem(trace)` --- substitute everywhere. Maximum contrast; often
    reads as an aphasic double of the original.
  * `gated_shadow(trace, ...)` --- substitute only where the decision was
    genuinely close. Reads as the original poem with its hinges swapped, which
    is usually the more legible artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .trace import Candidate, GenerationTrace, TokenStep


@dataclass
class Substitution:
    """One position where the shadow diverges from the poem."""

    index: int
    chosen: Candidate
    shadow: Candidate
    margin: float
    entropy: float
    chosen_rank: int
    prefix: str = ""
    """Trailing context before this position, for display."""

    @property
    def cost(self) -> float:
        """Signed log-probability difference, poem minus shadow.

        Positive: the poem took the likelier road and the shadow was the
        sacrifice. Negative: the sampler had already gone off-argmax here, and
        the shadow is the token the *model* preferred --- the poem is the
        sacrifice. Both are interesting; the sign says which kind of moment
        this was.
        """
        return self.chosen.logprob - self.shadow.logprob

    @property
    def gap(self) -> float:
        """Unsigned distance between the two readings.

        This, not `cost`, is the measure of how close the call was: a gap near
        zero means the poem and its shadow were all but interchangeable to the
        model, whichever one the sampler happened to take.
        """
        return abs(self.cost)

    @property
    def model_preferred_shadow(self) -> bool:
        """True where the written token was *not* the model's preference."""
        return self.cost < 0

    def __str__(self) -> str:
        return f"[{self.index}] {self.chosen.text!r} <- {self.shadow.text!r} (Δlp={self.cost:+.3f})"


@dataclass
class ShadowPoem:
    """A reconstructed near-poem plus the record of how it differs."""

    text: str
    rank: int
    substitutions: list[Substitution]
    source: GenerationTrace = field(repr=False)
    gated: bool = False
    gate: dict[str, Any] = field(default_factory=dict)

    @property
    def n_substitutions(self) -> int:
        return len(self.substitutions)

    @property
    def divergence_rate(self) -> float:
        """Fraction of positions that differ from the poem."""
        return self.n_substitutions / len(self.source) if len(self.source) else 0.0

    @property
    def total_cost(self) -> float:
        """Signed log-probability difference summed over substitutions.

        Positive means the poem is, in aggregate, the likelier of the two
        texts. It goes negative when the sampler spent much of the poem off
        the model's preferred path --- in which case the shadow is closer to
        what the model "wanted" than the poem is.
        """
        return sum(s.cost for s in self.substitutions)

    @property
    def total_gap(self) -> float:
        """Summed unsigned distance --- how far apart the two texts are."""
        return sum(s.gap for s in self.substitutions)

    @property
    def mean_gap(self) -> float:
        return self.total_gap / self.n_substitutions if self.substitutions else 0.0

    @property
    def contested_fraction(self) -> float:
        """Share of substitutions where the model itself preferred the shadow."""
        if not self.substitutions:
            return 0.0
        return sum(s.model_preferred_shadow for s in self.substitutions) / len(
            self.substitutions
        )

    def closest_calls(self, n: int = 10) -> list[Substitution]:
        """Substitutions where the two readings were nearest to interchangeable."""
        return sorted(self.substitutions, key=lambda s: s.gap)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "rank": self.rank,
            "gated": self.gated,
            "gate": self.gate,
            "divergence_rate": self.divergence_rate,
            "total_cost": self.total_cost,
            "total_gap": self.total_gap,
            "contested_fraction": self.contested_fraction,
            "substitutions": [
                {
                    "index": s.index,
                    "chosen": s.chosen.text,
                    "shadow": s.shadow.text,
                    "cost": s.cost,
                    "gap": s.gap,
                    "margin": s.margin,
                    "entropy": s.entropy,
                }
                for s in self.substitutions
            ],
        }


# --------------------------------------------------------------------------


def is_structural(text: str) -> bool:
    """True for tokens that carry line structure rather than lexical content."""
    return text.strip() == ""


def _word_initial(text: str) -> bool:
    """True if this token starts a new word rather than continuing one."""
    return text[:1].isspace() or text[:1] in "\n\t" or not text[:1].isalnum()


def _occupies_whole_word(trace: GenerationTrace, i: int, alt: Candidate) -> bool:
    """True if swapping step i for `alt` replaces a complete word.

    Subword tokenisers split words into pieces, so replacing one piece welds a
    new fragment onto the leftovers -- "peta|l" with "peta" swapped for
    " flower" becomes "floweral". Requiring that the position, its
    replacement, AND the following token are all word-initial keeps whole
    words intact.
    """
    if not _word_initial(trace.steps[i].chosen.text) or not _word_initial(alt.text):
        return False
    nxt = trace.steps[i + 1] if i + 1 < len(trace.steps) else None
    return nxt is None or _word_initial(nxt.chosen.text)


def shadow_poem(
    trace: GenerationTrace,
    rank: int = 1,
    *,
    preserve_structure: bool = False,
    word_aligned: bool = False,
) -> ShadowPoem:
    """The poem written from the `rank`-th rejected token at every position.

    rank=1 is the nearest-rejected sibling; rank=2, 3, ... walk further down
    the model's preference order, producing progressively stranger doubles.
    Positions with no candidate at that depth keep the original token (and are
    not counted as substitutions), so the result stays the same length.

    `preserve_structure=True` leaves whitespace-only tokens (line breaks,
    indentation) alone. The sampler did choose those, so substituting them is
    defensible --- but it conflates a decision about *lineation* with one about
    *wording*, and it makes the shadow illegible as verse because lines fuse.
    Default False so the statistics measure every decision the sampler made;
    the renderer sets True so the artifact can be read.

    `word_aligned=True` additionally skips positions where the swap would land
    mid-word, which is what produces fragments like "floweral". It yields a
    readable shadow at the cost of measuring fewer decisions: it is a
    *different object*, a word-level shadow rather than a token-level one, and
    should be labelled as such wherever it is shown.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1 (rank 0 is the poem itself)")

    pieces: list[str] = []
    subs: list[Substitution] = []
    for step in trace.steps:
        if preserve_structure and is_structural(step.chosen.text):
            pieces.append(step.chosen.text)
            continue
        alt = step.alternative(rank)
        if preserve_structure and alt is not None and is_structural(alt.text):
            # Swapping a word for a line break also destroys the shape.
            alt = next(
                (a for a in step.alternatives()[rank - 1 :] if not is_structural(a.text)),
                None,
            )
        if word_aligned and alt is not None:
            alt = next(
                (
                    a
                    for a in step.alternatives()[rank - 1 :]
                    if not (preserve_structure and is_structural(a.text))
                    and _occupies_whole_word(trace, step.index, a)
                ),
                None,
            )
        if alt is None:
            pieces.append(step.chosen.text)
            continue
        pieces.append(alt.text)
        subs.append(_substitution(trace, step, alt))
    return ShadowPoem(
        text="".join(pieces), rank=rank, substitutions=subs, source=trace
    )


def gated_shadow(
    trace: GenerationTrace,
    rank: int = 1,
    *,
    max_cost: float | None = None,
    top_n: int | None = None,
    min_entropy: float | None = None,
    by: str = "gap",
    preserve_structure: bool = False,
) -> ShadowPoem:
    """Substitute only at *contested* positions; keep the poem elsewhere.

    This is the reading most people find legible: the original poem with its
    genuine hinges swapped out, everywhere else untouched. Gate by any of:

      max_cost    -- only where the two readings were within this many nats
                     of each other (unsigned; e.g. 0.35 ~ the rejected token
                     held >=70% of the written token's probability).
      top_n       -- only the n most contested positions.
      min_entropy -- only where the field of options was diffuse (bits).

    Gates compose; `top_n` is applied last, to whatever survives the others.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")

    eligible: list[tuple[TokenStep, Candidate, Substitution]] = []
    for step in trace.steps:
        if preserve_structure and is_structural(step.chosen.text):
            continue
        alt = step.alternative(rank)
        if preserve_structure and alt is not None and is_structural(alt.text):
            alt = next(
                (a for a in step.alternatives()[rank - 1 :] if not is_structural(a.text)),
                None,
            )
        if alt is None:
            continue
        sub = _substitution(trace, step, alt)
        if max_cost is not None and sub.gap > max_cost:
            continue
        if min_entropy is not None and step.entropy < min_entropy:
            continue
        eligible.append((step, alt, sub))

    if top_n is not None:
        key = (lambda t: -t[0].entropy) if by == "entropy" else (lambda t: t[2].gap)
        eligible = sorted(eligible, key=key)[:top_n]

    chosen_idx = {s.index: (alt, sub) for s, alt, sub in eligible}
    pieces, subs = [], []
    for step in trace.steps:
        if step.index in chosen_idx:
            alt, sub = chosen_idx[step.index]
            pieces.append(alt.text)
            subs.append(sub)
        else:
            pieces.append(step.chosen.text)

    return ShadowPoem(
        text="".join(pieces),
        rank=rank,
        substitutions=sorted(subs, key=lambda s: s.index),
        source=trace,
        gated=True,
        gate={"max_cost": max_cost, "top_n": top_n, "min_entropy": min_entropy, "by": by},
    )


def shadow_family(trace: GenerationTrace, depth: int = 4) -> list[ShadowPoem]:
    """Ranks 1..depth as a family of near-poems off a single generation.

    This is the cheap anthology: `depth` texts for the price of one forward
    pass, all sharing the original's exact metrical skeleton.
    """
    return [shadow_poem(trace, r) for r in range(1, depth + 1)]


def divergence_profile(trace: GenerationTrace) -> list[dict[str, float]]:
    """Per-position decision geometry, for plotting the poem's contested spine."""
    out = []
    for s in trace.steps:
        alt = s.alternative(1)
        out.append(
            {
                "index": float(s.index),
                "entropy": s.entropy,
                "margin": s.margin if math.isfinite(s.margin) else float("nan"),
                "cost": (s.chosen.logprob - alt.logprob) if alt else float("nan"),
                "gap": abs(s.chosen.logprob - alt.logprob) if alt else float("nan"),
                "chosen_rank": float(s.chosen_rank),
                "retained_mass": s.retained_mass,
            }
        )
    return out


def coin_flips(trace: GenerationTrace, threshold: float = 0.1) -> list[Substitution]:
    """Positions where the top two candidates were within `threshold` nats.

    These are the moments where the poem's identity is, for practical purposes,
    the sampler's decision rather than the model's.
    """
    out = []
    for step in trace.steps:
        alt = step.alternative(1)
        if alt is None:
            continue
        sub = _substitution(trace, step, alt)
        if sub.gap <= threshold:
            out.append(sub)
    return out


def _substitution(
    trace: GenerationTrace, step: TokenStep, alt: Candidate, context: int = 40
) -> Substitution:
    prefix = trace.prefix_text(step.index)
    return Substitution(
        index=step.index,
        chosen=step.chosen,
        shadow=alt,
        margin=step.margin,
        entropy=step.entropy,
        chosen_rank=step.chosen_rank,
        prefix=prefix[-context:],
    )
