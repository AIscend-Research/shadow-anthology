"""Command line interface.

    shadow demo                        # runs end-to-end with no dependencies
    shadow trace  --prompt "..."       # generate and save a sampler trace
    shadow shadow --trace t.json       # reconstruct the nearest-rejected poem
    shadow branch --trace t.json       # generate the branching anthology
    shadow corpus --prompts p.txt      # corpus run with paired statistics
    shadow render --trace t.json       # standalone HTML artifact
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from .anthology import branch_anthology
from .backends import BackendUnsupported, get_backend
from .corpus import load_prompts, run_corpus, save_corpus
from .lexicons import Lexicons
from .metrics import compare
from .render import render_html, render_terminal, write_html
from .shadow import gated_shadow, shadow_family, shadow_poem
from .trace import GenerationTrace


def _backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="mock", choices=["mock", "hf", "openai", "anthropic"])
    p.add_argument("--model", default=None, help="model id for the chosen backend")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--endpoint", default="chat", choices=["chat", "completions"])
    p.add_argument("--device", default=None)


def _sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--candidates", type=int, default=20, help="top-k alternatives to retain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system", default=None)


def _lex_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concreteness-csv", default=None, help="Brysbaert et al. norms")
    p.add_argument("--vad-csv", default=None, help="Warriner et al. norms")
    p.add_argument("--frequency-csv", default=None, help="SUBTLEX frequencies")


def _make_backend(a: argparse.Namespace) -> Any:
    kw: dict[str, Any] = {}
    if a.backend == "hf":
        kw = {"model": a.model or "gpt2", "device": a.device}
    elif a.backend == "openai":
        if not a.model:
            raise SystemExit("--model is required for the openai backend")
        kw = {"model": a.model, "base_url": a.base_url, "endpoint": a.endpoint}
    elif a.backend == "mock" and a.model:
        kw = {"model": a.model}
    return get_backend(a.backend, **kw)


def _make_lex(a: argparse.Namespace) -> Lexicons:
    if any([a.concreteness_csv, a.vad_csv, a.frequency_csv]):
        return Lexicons.load(
            concreteness=a.concreteness_csv, vad=a.vad_csv, frequency=a.frequency_csv
        )
    return Lexicons.seed()


# -- commands --------------------------------------------------------------


def cmd_demo(a: argparse.Namespace) -> int:
    be = get_backend("mock")
    trace = be.generate_trace(
        a.prompt, max_tokens=a.max_tokens, seed=a.seed, candidates=12
    )
    sh = shadow_poem(trace, 1)
    print(render_terminal(trace, sh))
    print()
    print("shadow family (ranks 1-3):")
    for s in shadow_family(trace, 3):
        print(f"  rank {s.rank}: {s.text.strip()[:110]!r}")

    anth = branch_anthology(be, trace, n_points=3, ranks=(1,), budget=3, max_tokens=a.max_tokens)
    print(f"\nbranched {len(anth)} alternate poems in {anth.calls} calls")

    cmp_ = compare(trace.text, sh.text, trace=trace)
    print("\npoem vs shadow (seed lexicons):")
    for k, v in cmp_.deltas.items():
        print(f"  {k:<20} {'n/a' if v is None else f'{v:+.4f}'}")

    out = a.out or "demo.html"
    write_html(out, trace, shadow=sh, anthology=anth, title="The Shadow Anthology — demo")
    print(f"\nwrote {out}")
    return 0


def cmd_trace(a: argparse.Namespace) -> int:
    be = _make_backend(a)
    t = be.generate_trace(
        a.prompt,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        candidates=a.candidates,
        seed=a.seed,
        system=a.system,
    )
    t.save(a.out)
    print(t.text.strip())
    print(f"\n[{len(t)} tokens, off-argmax {t.offrank_fraction:.1%}] -> {a.out}")
    return 0


def cmd_shadow(a: argparse.Namespace) -> int:
    t = GenerationTrace.load(a.trace)
    sh = (
        gated_shadow(t, a.rank, top_n=a.top_n, max_cost=a.max_cost)
        if (a.top_n or a.max_cost)
        else shadow_poem(t, a.rank)
    )
    if a.json:
        print(json.dumps(sh.to_dict(), ensure_ascii=False, indent=1))
    else:
        print(render_terminal(t, sh))
    return 0


def cmd_branch(a: argparse.Namespace) -> int:
    t = GenerationTrace.load(a.trace)
    be = _make_backend(a)
    try:
        anth = branch_anthology(
            be, t,
            n_points=a.points,
            ranks=tuple(a.ranks),
            depth=a.depth,
            budget=a.budget,
            max_tokens=a.max_tokens,
            seed=a.seed,
        )
    except BackendUnsupported as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(anth.to_dict(), fh, ensure_ascii=False, indent=1)
    print(f"{len(anth)} branches in {anth.calls} calls -> {a.out}")
    for b in anth.nearest(5):
        print(
            f"\n@{b.branch_at} {b.displaced.text!r} -> {b.forced.text!r} "
            f"(gap {b.gap:.3f} nats)"
        )
        print("  " + b.divergent_tail.strip().replace("\n", "\n  ")[:200])
    if anth.dropped:
        print(f"\n{len(anth.dropped)} branch points skipped (budget).")
    return 0


def cmd_corpus(a: argparse.Namespace) -> int:
    prompts = load_prompts(a.prompts)
    if not prompts:
        raise SystemExit(f"no prompts found in {a.prompts}")
    be = _make_backend(a)

    def progress(i: int, n: int, _t: Any) -> None:
        print(f"\r  generating {i}/{n}", end="", file=sys.stderr, flush=True)

    res = run_corpus(
        be, prompts,
        samples_per_prompt=a.samples,
        rank=a.rank,
        gated=bool(a.top_n or a.max_cost),
        gate_top_n=a.top_n,
        gate_max_cost=a.max_cost,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        candidates=a.candidates,
        system=a.system,
        seed0=a.seed,
        lex=_make_lex(a),
        n_iter=a.permutations,
        on_progress=progress,
    )
    print(file=sys.stderr)
    print(res.summary())
    paths = save_corpus(res, a.out)
    print("\nwrote: " + ", ".join(paths.values()))
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    t = GenerationTrace.load(a.trace)
    sh = shadow_poem(t, a.rank)
    write_html(a.out, t, shadow=sh, title=a.title, depth=a.depth)
    print(f"wrote {a.out}")
    return 0


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shadow",
        description="Reconstruct the poems a language model almost wrote.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end run on the built-in mock sampler")
    d.add_argument("--prompt", default="Write a short poem about winter light.")
    d.add_argument("--max-tokens", type=int, default=60)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", default=None)
    d.set_defaults(fn=cmd_demo)

    t = sub.add_parser("trace", help="generate a poem and record the sampler trace")
    t.add_argument("--prompt", required=True)
    t.add_argument("--out", default="trace.json")
    _backend_args(t); _sampling_args(t)
    t.set_defaults(fn=cmd_trace)

    s = sub.add_parser("shadow", help="reconstruct the shadow poem from a trace")
    s.add_argument("--trace", required=True)
    s.add_argument("--rank", type=int, default=1)
    s.add_argument("--top-n", type=int, default=None, help="gate: only n most contested")
    s.add_argument("--max-cost", type=float, default=None, help="gate: max nats given up")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_shadow)

    b = sub.add_parser("branch", help="generate poems that fork at contested points")
    b.add_argument("--trace", required=True)
    b.add_argument("--out", default="anthology.json")
    b.add_argument("--points", type=int, default=6)
    b.add_argument("--ranks", type=int, nargs="+", default=[1])
    b.add_argument("--depth", type=int, default=1)
    b.add_argument("--budget", type=int, default=24)
    _backend_args(b); _sampling_args(b)
    b.set_defaults(fn=cmd_branch)

    c = sub.add_parser("corpus", help="corpus run with paired statistics")
    c.add_argument("--prompts", required=True)
    c.add_argument("--samples", type=int, default=4)
    c.add_argument("--rank", type=int, default=1)
    c.add_argument("--top-n", type=int, default=None)
    c.add_argument("--max-cost", type=float, default=None)
    c.add_argument("--permutations", type=int, default=20000)
    c.add_argument("--out", default="runs/latest")
    _backend_args(c); _sampling_args(c); _lex_args(c)
    c.set_defaults(fn=cmd_corpus)

    r = sub.add_parser("render", help="write the standalone HTML artifact")
    r.add_argument("--trace", required=True)
    r.add_argument("--out", default="shadow.html")
    r.add_argument("--rank", type=int, default=1)
    r.add_argument("--depth", type=int, default=3)
    r.add_argument("--title", default="The Shadow Anthology")
    r.set_defaults(fn=cmd_render)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except BackendUnsupported as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
