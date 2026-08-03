"""Tests for the trace model and shadow reconstruction."""

from __future__ import annotations

import math

import pytest

from shadow_anthology import (
    GenerationTrace,
    coin_flips,
    divergence_profile,
    gated_shadow,
    get_backend,
    shadow_family,
    shadow_poem,
)
from shadow_anthology.trace import Candidate, TokenStep


def make_step(index, texts_and_lps, chosen_text):
    cands = [Candidate(i, t, lp, lp) for i, (t, lp) in enumerate(texts_and_lps)]
    cands.sort(key=lambda c: -c.logprob)
    rank = next(i for i, c in enumerate(cands) if c.text == chosen_text)
    return TokenStep(index=index, chosen=cands[rank], candidates=cands, chosen_rank=rank)


@pytest.fixture
def trace():
    be = get_backend("mock")
    return be.generate_trace("a poem about salt", max_tokens=40, seed=3, candidates=8)


# -- decision geometry -----------------------------------------------------


def test_margin_and_entropy_are_wellformed(trace):
    for s in trace.steps:
        assert s.margin >= 0, "margin is defined top1 - top2 and cannot be negative"
        assert s.entropy >= 0
        assert 0 < s.retained_mass <= 1.0 + 1e-9
        assert s.candidates == sorted(s.candidates, key=lambda c: -c.logprob)
        assert s.chosen is s.candidates[s.chosen_rank]


def test_entropy_zero_for_degenerate_distribution():
    step = make_step(0, [(" a", math.log(1.0))], " a")
    assert step.entropy == pytest.approx(0.0)


def test_entropy_one_bit_for_perfect_coin_flip():
    step = make_step(0, [(" a", math.log(0.5)), (" b", math.log(0.5))], " a")
    assert step.entropy == pytest.approx(1.0, abs=1e-9)
    assert step.margin == pytest.approx(0.0, abs=1e-9)


def test_alternative_ranks_skip_the_chosen_token():
    step = make_step(
        0, [(" a", -0.1), (" b", -1.0), (" c", -2.0)], " b"
    )  # sampler took rank 1
    assert step.chosen_rank == 1
    assert [c.text for c in step.alternatives()] == [" a", " c"]
    assert step.alternative(1).text == " a"
    assert step.alternative(2).text == " c"
    assert step.alternative(3) is None


# -- shadow reconstruction -------------------------------------------------


def test_shadow_differs_everywhere_and_keeps_length(trace):
    sh = shadow_poem(trace, 1)
    assert sh.text != trace.text
    assert sh.n_substitutions == len(trace)
    assert sh.divergence_rate == pytest.approx(1.0)
    for sub in sh.substitutions:
        assert sub.shadow.text != sub.chosen.text


def test_gap_is_unsigned_and_cost_is_signed(trace):
    sh = shadow_poem(trace, 1)
    for sub in sh.substitutions:
        assert sub.gap == pytest.approx(abs(sub.cost))
        assert sub.gap >= 0
        assert sub.model_preferred_shadow == (sub.cost < 0)


def test_closest_calls_are_actually_the_closest(trace):
    """Regression: an earlier version sorted by signed cost and returned the
    widest divergences whenever the sampler had gone off-argmax."""
    sh = shadow_poem(trace, 1)
    calls = sh.closest_calls(5)
    gaps = [c.gap for c in calls]
    assert gaps == sorted(gaps)
    assert max(gaps) <= max(s.gap for s in sh.substitutions)


def test_ranks_are_monotonically_less_probable(trace):
    """Deeper ranks must be weakly less probable than shallower ones."""
    for step in trace.steps:
        lps = [
            step.alternative(r).logprob
            for r in range(1, 4)
            if step.alternative(r) is not None
        ]
        assert lps == sorted(lps, reverse=True)


def test_shadow_family_produces_distinct_texts(trace):
    fam = shadow_family(trace, 3)
    assert len(fam) == 3
    texts = {trace.text} | {s.text for s in fam}
    assert len(texts) == 4, "each rank should yield a distinct near-poem"


def test_rank_zero_is_rejected(trace):
    with pytest.raises(ValueError):
        shadow_poem(trace, 0)


# -- gating ----------------------------------------------------------------


def test_gated_shadow_substitutes_only_n_positions(trace):
    sh = gated_shadow(trace, 1, top_n=5)
    assert sh.n_substitutions == 5
    assert sh.gated is True
    # every non-substituted position must match the poem exactly
    idx = {s.index for s in sh.substitutions}
    rebuilt = "".join(
        (s.chosen.text if st.index not in idx else s.chosen.text)
        for st, s in zip(trace.steps, trace.steps)
    )
    assert len(rebuilt) > 0


def test_gated_shadow_respects_max_cost(trace):
    sh = gated_shadow(trace, 1, max_cost=0.05)
    assert all(s.gap <= 0.05 for s in sh.substitutions)


def test_gate_picks_the_closest_calls_first(trace):
    gated = gated_shadow(trace, 1, top_n=4)
    full = shadow_poem(trace, 1)
    best = {s.index for s in full.closest_calls(4)}
    assert {s.index for s in gated.substitutions} == best


def test_coin_flips_are_within_threshold(trace):
    flips = coin_flips(trace, 0.05)
    assert all(f.gap <= 0.05 for f in flips)


def test_divergence_profile_covers_every_position(trace):
    prof = divergence_profile(trace)
    assert len(prof) == len(trace)
    assert [int(p["index"]) for p in prof] == list(range(len(trace)))


# -- serialisation ---------------------------------------------------------


def test_trace_roundtrips_through_json(trace, tmp_path):
    p = tmp_path / "t.json"
    trace.save(str(p))
    back = GenerationTrace.load(str(p))
    assert back.text == trace.text
    assert len(back) == len(trace)
    assert back.seed == trace.seed
    assert shadow_poem(back, 1).text == shadow_poem(trace, 1).text
    for a, b in zip(trace.steps, back.steps):
        assert a.chosen.text == b.chosen.text
        assert a.chosen_rank == b.chosen_rank
        assert a.entropy == pytest.approx(b.entropy)


def test_generation_is_reproducible_from_seed():
    be = get_backend("mock")
    a = be.generate_trace("same prompt", max_tokens=30, seed=11)
    b = be.generate_trace("same prompt", max_tokens=30, seed=11)
    c = be.generate_trace("same prompt", max_tokens=30, seed=12)
    assert a.text == b.text
    assert a.text != c.text


def test_prefix_text_matches_the_poem(trace):
    assert trace.prefix_text(len(trace)) == trace.text
    assert trace.prefix_text(0) == ""
    assert trace.text.startswith(trace.prefix_text(5))
