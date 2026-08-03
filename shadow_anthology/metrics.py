"""Comparative measurement of a poem and its nearest-rejected sibling.

Three families, matching the three properties the project claims to compare:

  imagery -- concreteness and sensory density: is the text made of things you
             can see, or of ideas about things?
  tone    -- valence and arousal: where does it sit affectively?
  risk    -- lexical rarity, repetition avoidance, and (where a trace is
             available) the model's own surprisal: how far from the safe
             centre of the distribution does the text sit?

Design commitments worth stating, because they are where this kind of
measurement usually goes wrong:

  * A metric whose lexicon covers too little of the text returns `None`, not
    zero. Averaging over three matched words and calling it "tone" is how you
    manufacture an effect.
  * Model surprisal is reported but flagged: by construction the chosen token
    outranks its shadow at every position, so "the shadow is more surprising"
    is a tautology, not a finding. It is included as a sanity channel and
    excluded from the composite risk score. See `RISK_EXCLUDES_SURPRISAL`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Mapping, Sequence

from .lexicons import DEFAULT, Lexicons, tokenize_words
from .trace import GenerationTrace

MIN_COVERAGE = 0.15
"""Below this share of covered word tokens, a lexicon metric reports None."""

RISK_EXCLUDES_SURPRISAL = True
"""Composite risk deliberately omits model surprisal; see module docstring."""


@dataclass
class PoemMetrics:
    """Measurements for a single text. `None` means 'not enough coverage'."""

    n_words: int = 0
    n_types: int = 0
    n_lines: int = 0
    type_token_ratio: float = 0.0
    mean_word_length: float = 0.0
    mean_line_length: float = 0.0

    concreteness: float | None = None
    abstract_ratio: float | None = None
    sensory_density: float = 0.0
    imagery: float | None = None

    valence: float | None = None
    arousal: float | None = None

    rarity: float = 0.0
    repetition: float = 0.0
    risk: float | None = None

    mean_surprisal: float | None = None
    """Bits, from the trace. Tautologically ordered between poem and shadow."""

    coverage: dict[str, float] = field(default_factory=dict)
    seed_lexicons: bool = True
    """True if scored against the built-in seed lexicons rather than norms."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(
    text: str,
    *,
    trace: GenerationTrace | None = None,
    lex: Lexicons | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> PoemMetrics:
    """Score one text. Pass `trace` to include model-derived surprisal."""
    lex = lex or DEFAULT
    words = tokenize_words(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    m = PoemMetrics(seed_lexicons=lex.is_seed)

    if not words:
        return m

    m.n_words = len(words)
    m.n_types = len(set(words))
    m.n_lines = len(lines)
    m.type_token_ratio = m.n_types / m.n_words
    m.mean_word_length = sum(len(w) for w in words) / m.n_words
    m.mean_line_length = (
        sum(len(tokenize_words(ln)) for ln in lines) / len(lines) if lines else 0.0
    )
    m.coverage = lex.coverage(words)

    # -- imagery -----------------------------------------------------------
    conc = [lex.concreteness[w] for w in words if w in lex.concreteness]
    if m.coverage["concreteness"] >= min_coverage and conc:
        m.concreteness = sum(conc) / len(conc)
        m.abstract_ratio = sum(1 for c in conc if c < 3.0) / len(conc)
    m.sensory_density = sum(1 for w in words if w in lex.sensory) / m.n_words
    if m.concreteness is not None:
        # normalise concreteness to 0..1 over its 1..5 scale, then blend
        m.imagery = 0.7 * ((m.concreteness - 1.0) / 4.0) + 0.3 * min(
            1.0, m.sensory_density * 8.0
        )

    # -- tone --------------------------------------------------------------
    val = [lex.valence[w] for w in words if w in lex.valence]
    aro = [lex.arousal[w] for w in words if w in lex.arousal]
    if m.coverage["valence"] >= min_coverage and val:
        m.valence = sum(val) / len(val)
    if m.coverage["arousal"] >= min_coverage and aro:
        m.arousal = sum(aro) / len(aro)

    # -- risk --------------------------------------------------------------
    m.rarity = sum(lex.rarity(w) for w in words) / m.n_words
    m.repetition = _repetition(words)
    risk_parts = [m.rarity, 1.0 - m.repetition, m.type_token_ratio]
    if m.arousal is not None:
        risk_parts.append(min(1.0, abs(m.arousal - 5.0) / 4.0))
    m.risk = sum(risk_parts) / len(risk_parts)

    if trace is not None and len(trace):
        m.mean_surprisal = sum(s.chosen.surprisal for s in trace.steps) / len(trace)

    return m


def measure_shadow_surprisal(trace: GenerationTrace, rank: int = 1) -> float | None:
    """Mean surprisal (bits) of the rank-k shadow tokens under the model.

    Reported alongside `PoemMetrics.mean_surprisal` for transparency. It is
    always >= the poem's by construction; the informative quantity is the
    *gap*, which measures how much probability the poem's identity actually
    cost --- not which text is "more surprising".
    """
    vals = []
    for s in trace.steps:
        alt = s.alternative(rank)
        if alt is not None:
            vals.append(alt.surprisal)
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------


COMPARABLE = (
    "concreteness",
    "abstract_ratio",
    "sensory_density",
    "imagery",
    "valence",
    "arousal",
    "rarity",
    "repetition",
    "risk",
    "type_token_ratio",
    "mean_word_length",
)


@dataclass
class Comparison:
    """Paired measurement of a poem against one of its shadows."""

    poem: PoemMetrics
    shadow: PoemMetrics
    deltas: dict[str, float | None]
    rank: int = 1
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rank": self.rank,
            "poem": self.poem.to_dict(),
            "shadow": self.shadow.to_dict(),
            "deltas": self.deltas,
        }


def compare(
    poem_text: str,
    shadow_text: str,
    *,
    trace: GenerationTrace | None = None,
    lex: Lexicons | None = None,
    rank: int = 1,
    label: str = "",
) -> Comparison:
    """Measure both texts and take poem-minus-shadow deltas.

    A delta is `None` whenever either side lacked coverage, so downstream
    aggregation can drop the pair instead of imputing a zero difference.
    """
    a = measure(poem_text, trace=trace, lex=lex)
    b = measure(shadow_text, lex=lex)
    deltas: dict[str, float | None] = {}
    for k in COMPARABLE:
        va, vb = getattr(a, k), getattr(b, k)
        deltas[k] = (va - vb) if (va is not None and vb is not None) else None
    return Comparison(poem=a, shadow=b, deltas=deltas, rank=rank, label=label)


def _repetition(words: Sequence[str], n: int = 2) -> float:
    """Share of n-grams that are not the first occurrence of that n-gram."""
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / len(grams)
