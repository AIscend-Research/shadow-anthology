"""shadow-anthology — reconstructing the poems a language model almost wrote.

    from shadow_anthology import get_backend, shadow_poem, render_html

    be = get_backend("hf", model="gpt2")
    trace = be.generate_trace("Write a short poem about winter light.", seed=0)
    sh = shadow_poem(trace)          # the nearest-rejected sibling
    print(trace.text); print(sh.text)

See README.md for the argument the code is making, and `shadow demo` for a
runnable end-to-end example with no dependencies.
"""

from .anthology import Anthology, Branch, branch_anthology
from .backends import Backend, BackendUnsupported, get_backend
from .corpus import (
    CorpusResult,
    analyse_traces,
    generate_traces,
    load_prompts,
    run_corpus,
    save_corpus,
)
from .lexicons import Lexicons
from .metrics import Comparison, PoemMetrics, compare, measure
from .render import render_html, render_terminal, write_html
from .shadow import (
    ShadowPoem,
    Substitution,
    coin_flips,
    divergence_profile,
    gated_shadow,
    shadow_family,
    shadow_poem,
)
from .stats import TestResult, holm_bonferroni, paired_permutation_test
from .trace import Candidate, GenerationTrace, TokenStep, load_traces, save_traces

__version__ = "0.1.0"

__all__ = [
    "Anthology", "Backend", "BackendUnsupported", "Branch", "Candidate",
    "Comparison", "CorpusResult", "GenerationTrace", "Lexicons", "PoemMetrics",
    "ShadowPoem", "Substitution", "TestResult", "TokenStep", "branch_anthology",
    "coin_flips", "compare", "divergence_profile", "gated_shadow", "get_backend",
    "holm_bonferroni", "load_prompts", "load_traces", "measure",
    "paired_permutation_test", "render_html", "render_terminal", "run_corpus",
    "save_corpus", "save_traces", "shadow_family", "shadow_poem", "write_html",
    "analyse_traces", "generate_traces",
]
