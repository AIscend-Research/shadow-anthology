"""Paired statistics, dependency-free.

Every comparison in this project is naturally paired: a poem and the shadow
reconstructed from the same trace share prompt, seed, model, length and
metrical shape. That pairing is the design's main strength, so the tests here
are paired ones, and the default is a sign-flip permutation test --- exact in
its null (under the null, the sign of each within-pair difference is
exchangeable) and free of distributional assumptions poetry metrics will not
satisfy.

`scipy` is not required. If it is installed nothing here changes; these
implementations are the reference.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Sequence


@dataclass
class TestResult:
    name: str
    n: int
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    effect_size: float
    """Cohen's d_z --- mean difference over the SD of the differences."""
    statistic: float
    p_value: float
    p_adjusted: float | None = None
    note: str = ""

    @property
    def significant(self) -> bool:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        star = "*" if p < 0.05 else " "
        return (
            f"{star} {self.name:<20} n={self.n:<4} "
            f"Δ={self.mean_diff:+.4f} [{self.ci_low:+.4f},{self.ci_high:+.4f}] "
            f"d_z={self.effect_size:+.3f} p={p:.4g}"
        )


def paired_permutation_test(
    diffs: Sequence[float],
    *,
    n_iter: int = 20000,
    seed: int = 0,
    name: str = "",
) -> TestResult:
    """Two-sided sign-flip permutation test on within-pair differences.

    p is computed with the standard +1 correction in numerator and denominator,
    so it is never reported as exactly zero --- with `n_iter` resamples the
    floor is 1/(n_iter+1).
    """
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    n = len(d)
    if n < 2:
        return TestResult(
            name=name, n=n, mean_diff=(d[0] if d else 0.0), median_diff=(d[0] if d else 0.0),
            ci_low=float("nan"), ci_high=float("nan"), effect_size=0.0,
            statistic=0.0, p_value=1.0, note="too few pairs",
        )

    obs = _mean(d)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_iter):
        s = _mean([x if rng.random() < 0.5 else -x for x in d])
        if abs(s) >= abs(obs) - 1e-15:
            extreme += 1
    p = (extreme + 1) / (n_iter + 1)

    lo, hi = bootstrap_ci(d, seed=seed)
    return TestResult(
        name=name,
        n=n,
        mean_diff=obs,
        median_diff=_median(d),
        ci_low=lo,
        ci_high=hi,
        effect_size=cohens_dz(d),
        statistic=obs,
        p_value=p,
        note=f"sign-flip permutation, {n_iter} resamples",
    )


def wilcoxon_signed_rank(diffs: Sequence[float], *, name: str = "") -> TestResult:
    """Wilcoxon signed-rank with tie-corrected normal approximation.

    Provided for readers who expect it. For n < ~20 the normal approximation is
    poor and `paired_permutation_test` should be preferred; the note field says
    so on the result itself rather than leaving it to the reader.
    """
    d = [x for x in diffs if x is not None and math.isfinite(x) and x != 0.0]
    n = len(d)
    if n < 2:
        return TestResult(name, n, 0.0, 0.0, float("nan"), float("nan"), 0.0, 0.0, 1.0,
                          note="too few non-zero pairs")

    ranks = _average_ranks([abs(x) for x in d])
    w_plus = sum(r for x, r in zip(d, ranks) if x > 0)
    w_minus = sum(r for x, r in zip(d, ranks) if x < 0)
    w = min(w_plus, w_minus)

    mu = n * (n + 1) / 4.0
    tie_corr = sum(t**3 - t for t in _tie_group_sizes(ranks)) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    z = (w - mu) / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))

    lo, hi = bootstrap_ci(d)
    note = "normal approximation" + ("; n<20, prefer permutation" if n < 20 else "")
    return TestResult(name, n, _mean(d), _median(d), lo, hi, cohens_dz(d), w,
                      min(1.0, max(0.0, p)), note=note)


def bootstrap_ci(
    xs: Sequence[float], *, alpha: float = 0.05, n_iter: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    d = [x for x in xs if x is not None and math.isfinite(x)]
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(d)
    means = [_mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(n_iter)]
    means.sort()
    lo = means[int((alpha / 2) * n_iter)]
    hi = means[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return (lo, hi)


def cohens_dz(diffs: Sequence[float]) -> float:
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    if len(d) < 2:
        return 0.0
    m = _mean(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 0 else 0.0


def holm_bonferroni(results: Sequence[TestResult]) -> list[TestResult]:
    """Holm step-down correction across a family of tests, in place.

    Applied by default in `corpus.py`: comparing eleven metrics without
    correction would make a spurious hit near-certain.
    """
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, results[i].p_value * (m - rank))
        running = max(running, adj)  # enforce monotonicity
        results[i].p_adjusted = running
    return list(results)


# -- helpers ---------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _average_ranks(xs: Sequence[float]) -> list[float]:
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _tie_group_sizes(ranks: Sequence[float]) -> list[int]:
    counts: dict[float, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    return [c for c in counts.values() if c > 1]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
