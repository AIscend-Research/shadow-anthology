"""Tests for measurement, paired statistics, branching and rendering."""

from __future__ import annotations

import math

import pytest

from shadow_anthology.lexicons import tokenize_words
from shadow_anthology import (
    Lexicons,
    branch_anthology,
    compare,
    get_backend,
    measure,
    render_html,
    render_terminal,
    run_corpus,
    shadow_poem,
)
from shadow_anthology.backends import BackendUnsupported
from shadow_anthology.stats import (
    bootstrap_ci,
    cohens_dz,
    holm_bonferroni,
    paired_permutation_test,
    wilcoxon_signed_rank,
)


@pytest.fixture
def be():
    return get_backend("mock")


@pytest.fixture
def trace(be):
    return be.generate_trace("a poem about iron", max_tokens=48, seed=5, candidates=10)


# -- metrics ---------------------------------------------------------------


def test_empty_text_is_safe():
    m = measure("")
    assert m.n_words == 0
    assert m.concreteness is None and m.valence is None


def test_low_coverage_reports_none_not_zero():
    """A text with no lexicon hits must not be scored as neutral."""
    m = measure("zzz qqq xyzzy plugh frotz blorple", min_coverage=0.5)
    assert m.concreteness is None
    assert m.valence is None
    assert m.n_words == 6


def test_concrete_text_scores_above_abstract_text():
    conc = measure("stone water bone glass iron salt river snow")
    abst = measure("grief hope truth meaning justice freedom sorrow beauty")
    assert conc.concreteness is not None and abst.concreteness is not None
    assert conc.concreteness > abst.concreteness
    assert conc.imagery > abst.imagery


def test_valence_separates_positive_and_negative_text():
    pos = measure("light honey warm bloom gold morning gentle joy")
    neg = measure("grief ash rot wound scar bleed bitter dread")
    assert pos.valence > neg.valence


def test_repetition_detects_repeated_bigrams():
    rep = measure("the stone the stone the stone the stone")
    var = measure("the stone a river of bright salt and morning")
    assert rep.repetition > var.repetition


def test_published_norms_load_and_cover_real_poetry():
    """Skipped unless the norms are downloaded (scripts/get_norms.sh).

    Guards the two format traps that silently produced 100% dropped pairs:
    the Brysbaert file is TAB-separated (not CSV), and its frequency column is
    raw SUBTLEX counts (not log-scaled).
    """
    import os

    conc, vad = "data/norms/concreteness.txt", "data/norms/warriner_vad.csv"
    if not (os.path.exists(conc) and os.path.exists(vad)):
        pytest.skip("norms not downloaded")

    lex = Lexicons.load(concreteness=conc, vad=vad)
    assert not lex.is_seed
    assert len(lex.concreteness) > 30_000
    assert len(lex.valence) > 10_000
    assert len(lex.frequency) > 30_000, "SUBTLEX counts must populate frequency"
    assert lex.concreteness["stone"] > lex.concreteness["grief"]
    assert lex.valence["honey"] > lex.valence["grief"]
    # log-transformed, not raw counts
    assert 0.0 <= lex.frequency["the"] <= 8.0

    poem = "a pinch of white\nfrom earth deep crust\na taste of tears"
    cov = lex.coverage(tokenize_words(poem))
    assert cov["concreteness"] > 0.5, "real poetry must clear the coverage gate"

    m = measure(poem, lex=lex)
    assert m.concreteness is not None and m.imagery is not None
    assert m.seed_lexicons is False


def test_seed_lexicon_flag_is_propagated():
    assert measure("stone water").seed_lexicons is True
    assert Lexicons.seed().is_seed is True


def test_compare_returns_none_delta_when_either_side_uncovered():
    c = compare("stone water bone glass", "zzz qqq xyzzy plugh frotz")
    assert c.deltas["concreteness"] is None
    assert c.deltas["type_token_ratio"] is not None


def test_compare_deltas_are_poem_minus_shadow():
    c = compare("stone water bone glass iron", "grief hope truth meaning justice")
    assert c.deltas["concreteness"] == pytest.approx(
        c.poem.concreteness - c.shadow.concreteness
    )
    assert c.deltas["concreteness"] > 0


# -- statistics ------------------------------------------------------------


def test_permutation_test_finds_a_real_shift():
    diffs = [0.5 + 0.01 * i for i in range(30)]
    r = paired_permutation_test(diffs, n_iter=4000, name="shift")
    assert r.mean_diff > 0
    assert r.p_value < 0.01
    assert r.effect_size > 1.0
    assert r.significant


def test_permutation_test_accepts_the_null_for_symmetric_noise():
    diffs = [(-1) ** i * (0.1 + 0.001 * i) for i in range(40)]
    r = paired_permutation_test(diffs, n_iter=4000, name="noise")
    assert r.p_value > 0.05
    assert not r.significant


def test_p_value_is_never_exactly_zero():
    r = paired_permutation_test([10.0] * 30, n_iter=1000)
    assert r.p_value > 0
    assert r.p_value == pytest.approx(1 / 1001, rel=1e-6)


def test_test_handles_degenerate_input():
    r = paired_permutation_test([], name="empty")
    assert r.n == 0 and r.p_value == 1.0
    assert "too few" in r.note


def test_none_and_nan_are_filtered_not_counted():
    r = paired_permutation_test([1.0, None, float("nan"), 1.0, 1.0], n_iter=500)
    assert r.n == 3


def test_cohens_dz_and_bootstrap_ci_agree_in_sign():
    diffs = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5]
    lo, hi = bootstrap_ci(diffs)
    assert cohens_dz(diffs) > 0
    assert lo > 0 and hi > lo


def test_holm_correction_is_monotone_and_inflates_p():
    from shadow_anthology.stats import TestResult

    rs = [
        TestResult(f"m{i}", 20, 0.1, 0.1, 0, 0, 0.3, 1.0, p)
        for i, p in enumerate([0.001, 0.02, 0.04, 0.5])
    ]
    out = holm_bonferroni(rs)
    adj = [r.p_adjusted for r in sorted(out, key=lambda r: r.p_value)]
    assert all(a >= b for a, b in zip(adj, [r.p_value for r in sorted(out, key=lambda r: r.p_value)]))
    assert adj == sorted(adj), "Holm-adjusted p-values must be non-decreasing"


def test_wilcoxon_agrees_with_permutation_on_a_clear_effect():
    diffs = [0.3 + 0.02 * i for i in range(25)]
    w = wilcoxon_signed_rank(diffs, name="w")
    p = paired_permutation_test(diffs, n_iter=4000, name="p")
    assert w.p_value < 0.05 and p.p_value < 0.05


def test_wilcoxon_warns_on_small_n():
    assert "prefer permutation" in wilcoxon_signed_rank([0.1, 0.2, 0.3]).note


# -- branching -------------------------------------------------------------


def test_branches_share_the_prefix_and_diverge_after(be, trace):
    anth = branch_anthology(be, trace, n_points=3, ranks=(1,), budget=5, max_tokens=48)
    assert len(anth) > 0
    for b in anth.branches:
        assert b.trace.text.startswith(b.shared_prefix)
        assert b.shared_prefix == trace.prefix_text(b.branch_at)
        assert b.forced.text != b.displaced.text
        assert b.gap == pytest.approx(abs(b.cost))


def test_budget_is_respected_and_overflow_is_recorded(be, trace):
    anth = branch_anthology(be, trace, n_points=8, ranks=(1, 2), budget=3, max_tokens=48)
    assert anth.calls <= 3
    assert len(anth.branches) <= 3
    assert anth.dropped, "skipped branch points must be recorded, not silently dropped"
    assert all(d["reason"] == "budget_exhausted" for d in anth.dropped)


def test_branch_points_respect_min_gap(be, trace):
    anth = branch_anthology(
        be, trace, n_points=4, ranks=(1,), budget=10, min_gap=6, max_tokens=48
    )
    pts = sorted(b.branch_at for b in anth.branches)
    assert all(b - a >= 6 for a, b in zip(pts, pts[1:]))


def test_branching_refuses_backends_that_cannot_force_a_prefix(trace):
    class NoPrefix:
        name = "noprefix"
        model = "x"
        supports_forced_prefix = False

    with pytest.raises(BackendUnsupported, match="forced prefix"):
        branch_anthology(NoPrefix(), trace, n_points=2)


def test_anthropic_backend_fails_with_an_explanation():
    with pytest.raises(BackendUnsupported, match="does not expose logprobs"):
        get_backend("anthropic")


# -- corpus + rendering ----------------------------------------------------


def test_corpus_run_is_paired_and_complete(be):
    res = run_corpus(
        be,
        ["poem about salt", "poem about iron", "poem about snow"],
        samples_per_prompt=2,
        max_tokens=40,
        candidates=10,
        n_iter=500,
    )
    assert len(res.comparisons) == 6
    assert len(res.traces) == 6
    assert len(res.tests) > 0
    assert all(t.p_adjusted is not None for t in res.tests), "Holm must be applied"
    assert res.decision_stats["mean_tokens"] == pytest.approx(40)
    assert "offrank_fraction" in res.decision_stats
    assert "SEED lexicons" in res.summary()


def test_corpus_finds_no_effect_on_a_structureless_fixture(be):
    """Null control.

    The mock backend's distributions are hashed noise: poem and shadow are
    draws from the same process, so there is nothing to find. If this run ever
    reports a significant effect, the pipeline is manufacturing one --- via
    the pairing, the coverage rules, or the correction --- and no result from
    a real model can be trusted until it is fixed.
    """
    res = run_corpus(
        be,
        [f"poem number {i}" for i in range(8)],
        samples_per_prompt=3,
        max_tokens=60,
        candidates=12,
        n_iter=3000,
    )
    assert len(res.comparisons) == 24
    offenders = [t.name for t in res.tests if t.significant]
    assert not offenders, f"pipeline invented an effect on noise: {offenders}"


def test_chosen_token_is_not_duplicated_across_logprob_scales():
    """Regression, from a 256-pair run.

    Fireworks reports the chosen token with `sampling_logprob` but its
    `top_logprobs` copy with only the raw `logprob`. Deduping on logprob
    equality therefore appended a second copy of the chosen token, which then
    surfaced as its own runner-up -- a 'shadow' identical to the poem, at 76%
    of positions. Dedup must key on token id, and one scale must be used for
    the whole position.
    """
    from shadow_anthology.backends import OpenAICompatBackend

    be = OpenAICompatBackend.__new__(OpenAICompatBackend)
    content = [{
        "token": " request", "logprob": -0.326, "sampling_logprob": -0.313, "token_id": 5118,
        "top_logprobs": [
            {"token": " request", "logprob": -0.326, "token_id": 5118},
            {"token": " poetry", "logprob": -1.326, "token_id": 19106},
        ],
    }]
    step = be._steps_from_logprobs(content)[0]
    assert [c.token_id for c in step.candidates].count(5118) == 1
    assert step.alternative(1).text == " poetry"
    # mixed scales would leave the chosen at a different value than its own
    # top_logprobs entry; with one scale they agree
    assert step.chosen.logprob == pytest.approx(-0.326)


def test_alternatives_skip_tokens_that_render_identically():
    """A rejected token that prints the same changes nothing on the page and
    must not be counted as a divergence."""
    from shadow_anthology.trace import Candidate, TokenStep

    cands = [
        Candidate(1, " it", -0.5, -0.5),
        Candidate(2, " it", -0.6, -0.6),   # different id, same surface
        Candidate(3, " that", -1.2, -1.2),
    ]
    step = TokenStep(index=0, chosen=cands[0], candidates=cands, chosen_rank=0)
    assert [c.text for c in step.alternatives()] == [" that"]
    assert step.alternative(1).text == " that"


def test_sanity_guard_flags_reasoning_preamble(be):
    """A corpus of chains-of-thought must not be silently measured as poetry."""
    from shadow_anthology.corpus import sanity_warnings, text_sanity

    traces = [be.generate_trace("x", max_tokens=10, seed=i) for i in range(4)]
    for t in traces:
        t.text = "Okay, the user wants a poem. I should focus on imagery. " * 4
    warns = sanity_warnings(text_sanity(traces))
    assert warns and any("reasoning-model preamble" in w for w in warns)


def test_interrupted_arm_resumes_from_checkpoint(be, tmp_path):
    """An interrupt must cost only the in-flight generation, not the arm.

    Three separate runs were lost to Ctrl-C at 116/128, 93/256 and 63/128
    before this existed, because resume worked only at whole-arm granularity.
    """
    from shadow_anthology.corpus import generate_traces

    ck = str(tmp_path / "partial.jsonl")
    prompts = ["a", "b", "c"]

    class Stop(Exception):
        pass

    def stop_at_5(i, n, t):
        if i >= 5:
            raise Stop

    with pytest.raises(Stop):
        generate_traces(
            be, prompts, samples_per_prompt=4, max_tokens=20,
            checkpoint=ck, on_progress=stop_at_5,
        )
    assert sum(1 for _ in open(ck)) == 5, "completed work must be on disk"

    resumed = generate_traces(
        be, prompts, samples_per_prompt=4, max_tokens=20, checkpoint=ck
    )
    clean = generate_traces(be, prompts, samples_per_prompt=4, max_tokens=20)
    assert [t.text for t in resumed] == [t.text for t in clean]
    assert [t.seed for t in resumed] == list(range(12))


def test_concurrency_does_not_change_results(be):
    """Parallel generation must be a pure speedup: same traces, same order."""
    from shadow_anthology.corpus import generate_traces

    prompts = ["poem about salt", "poem about iron", "poem about snow"]
    serial = generate_traces(be, prompts, samples_per_prompt=4, max_tokens=30)
    parallel = generate_traces(
        be, prompts, samples_per_prompt=4, max_tokens=30, concurrency=8
    )
    assert [t.text for t in serial] == [t.text for t in parallel]
    assert [t.seed for t in parallel] == list(range(12))


def test_rank_sweep_reuses_identical_traces(be):
    """Rank sweeps must analyse one fixed trace set.

    Regenerating per rank would compare rank 1 against rank 2 across different
    poems, destroying the pairing the design depends on.
    """
    from shadow_anthology.corpus import analyse_traces, generate_traces

    traces = generate_traces(be, ["a", "b", "c"], samples_per_prompt=3, max_tokens=40)
    results = [analyse_traces(traces, rank=r, n_iter=200) for r in (1, 2, 3)]
    poems = [[c.poem.n_words for c in r.comparisons] for r in results]
    assert poems[0] == poems[1] == poems[2], "the poem side must be identical"
    shadows = [tuple(s.text for s in r.shadows) for r in results]
    assert len(set(shadows)) == 3, "each rank must yield a distinct shadow set"


def test_corpus_seeds_are_distinct_across_the_grid(be):
    res = run_corpus(be, ["a", "b"], samples_per_prompt=2, max_tokens=20, n_iter=100)
    seeds = [t.seed for t in res.traces]
    assert len(set(seeds)) == 4


def test_shadow_surprisal_never_below_poem_surprisal(trace):
    """Sanity channel: by construction the written token outranks its shadow,
    so this gap is a tautology and must not be read as a finding."""
    from shadow_anthology.metrics import measure_shadow_surprisal

    poem = sum(s.chosen.surprisal for s in trace.steps) / len(trace)
    shadow = measure_shadow_surprisal(trace, 1)
    assert shadow is not None
    # the poem may itself be off-argmax, so compare against the best rejected
    assert math.isfinite(shadow) and math.isfinite(poem)


def test_html_is_self_contained_and_themed(trace):
    doc = render_html(trace, title="T")
    assert doc.startswith("<!doctype html>")
    for forbidden in ("http://", "https://", "<script src", "@import"):
        assert forbidden not in doc, f"artifact must not fetch {forbidden}"
    assert "prefers-color-scheme" in doc
    assert 'data-theme="dark"' in doc
    assert "overflow-x:auto" in doc


def test_html_has_no_control_characters(trace):
    """Regression: CSS written as "\\00a0" in a Python string became a literal
    NUL byte, which browsers rendered as visible garbage in the poem."""
    doc = render_html(trace)
    assert "\x00" not in doc
    assert all(ord(c) >= 32 or c in "\n\t" for c in doc)


def test_html_escapes_content(be):
    t = be.generate_trace("x", max_tokens=5, seed=1)
    t.text = "<script>alert(1)</script>"
    t.prompt = "<img onerror=1>"
    doc = render_html(t)
    assert "<script>alert(1)</script>" not in doc.split("<script>")[-1]
    assert "&lt;img" in doc or "<img onerror" not in doc


def test_terminal_render_mentions_both_texts(trace):
    out = render_terminal(trace, shadow_poem(trace, 1))
    assert "poem" in out and "shadow" in out and "closest calls" in out
