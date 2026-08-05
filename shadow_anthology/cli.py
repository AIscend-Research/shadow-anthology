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
from .backends import APIRequestFailed, BackendUnsupported, get_backend
from .corpus import (
    analyse_traces,
    generate_traces,
    load_prompts,
    run_corpus,
    save_corpus,
)
from .lexicons import Lexicons
from .metrics import compare
from .render import render_html, render_terminal, write_html
from .shadow import gated_shadow, shadow_family, shadow_poem
from .trace import GenerationTrace, load_traces


def _backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "hf", "fireworks", "openai", "anthropic"],
    )
    p.add_argument("--model", default=None, help="model id for the chosen backend")
    p.add_argument("--base-url", default=None, help="override the backend's default")
    p.add_argument(
        "--endpoint", default=None, choices=["chat", "completions"],
        help="completions keeps forced-prefix branching available; chat does not",
    )
    p.add_argument(
        "--max-top-logprobs", type=int, default=None,
        help="server cap on retained alternatives (Fireworks default 5)",
    )
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
    elif a.backend in ("openai", "fireworks"):
        if a.backend == "openai" and not a.model:
            raise SystemExit("--model is required for the openai backend")
        if a.model:
            kw["model"] = a.model
        if a.base_url:
            kw["base_url"] = a.base_url
        # Default to /completions on Fireworks so branching stays available;
        # OpenAI's chat endpoint remains the default there.
        kw["endpoint"] = a.endpoint or ("completions" if a.backend == "fireworks" else "chat")
        if a.backend == "fireworks" and a.max_top_logprobs:
            kw["max_top_logprobs"] = a.max_top_logprobs
    elif a.backend == "mock" and a.model:
        kw = {"model": a.model}
    return get_backend(a.backend, **kw)


DEFAULT_NORMS = {
    "concreteness": "data/norms/concreteness.txt",
    "vad": "data/norms/warriner_vad.csv",
}


def _make_lex(a: argparse.Namespace) -> Lexicons:
    """Published norms if available, seed lexicons otherwise.

    Falls back to `data/norms/` automatically so a normal run is norm-backed
    without extra flags. Whether it succeeded is stamped on every result as
    `seed_lexicons`, so the distinction is never lost.
    """
    conc = a.concreteness_csv or DEFAULT_NORMS["concreteness"]
    vad = a.vad_csv or DEFAULT_NORMS["vad"]
    lex = Lexicons.load(concreteness=conc, vad=vad, frequency=a.frequency_csv)
    if lex.is_seed:
        print(
            "warning: no published norms found (looked in data/norms/). "
            "Imagery and tone will be scored with SEED lexicons and will "
            "mostly report n=0. Run: bash scripts/get_norms.sh",
            file=sys.stderr,
        )
    return lex


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


def cmd_check(a: argparse.Namespace) -> int:
    """Preflight: is the key valid, and is the model reachable?

    These are checked separately and in that order, because the APIs conflate
    them: Fireworks resolves the model before authenticating, so an unknown
    model returns 404 even with a bad key. Diagnosing them together is how you
    end up debugging the wrong one.
    """
    import httpx

    key_env = {"fireworks": "FIREWORKS_API_KEY", "openai": "OPENAI_API_KEY"}.get(
        a.backend
    )
    key = os.environ.get(key_env or "", "")
    print(f"backend   {a.backend}")
    print(f"{key_env or 'api key':<9} " + (f"set, {len(key)} chars" if key else "NOT SET"))
    if not key:
        print(f"\nFAIL: {key_env} is empty. Create a key and export it.")
        return 2

    base = a.base_url or (
        "https://api.fireworks.ai/inference/v1"
        if a.backend == "fireworks"
        else "https://api.openai.com/v1"
    )
    hdr = {"Authorization": f"Bearer {key}"}

    # 1. Auth, isolated from any model lookup.
    try:
        r = httpx.get(f"{base}/models", headers=hdr, timeout=30.0)
    except Exception as e:
        print(f"\nFAIL: could not reach {base}: {e}")
        return 2
    if r.status_code == 401:
        print(f"\nFAIL: key rejected -- {r.text[:200]}")
        print("  The whole key must be copied; they are long and easily truncated.")
        return 2
    if r.status_code >= 400:
        print(f"\nWARN: /models returned {r.status_code}: {r.text[:200]}")
    else:
        print("auth      OK")

    # 2. Model reachability, and what is actually available.
    ids = []
    try:
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass
    if not a.model:
        if ids:
            print(f"\nModels available to this account ({len(ids)}):")
            for m in ids[:40]:
                print(f"  {m}")
            print("\nRe-run with --model <id> to probe one for logprob support.")
        return 0

    print(f"model     {a.model} -- {'listed' if a.model in ids else 'NOT in /models'}")
    if ids and a.model not in ids:
        print("\n  Available to this account:")
        for m in ids[:40]:
            print(f"    {m}")
        return 2

    # 3. Live probe. Being listed is not enough: this method needs the losing
    #    tokens, and plenty of served models return no logprobs at all. An
    #    8-token generation settles it for a fraction of a cent.
    print("\nprobing (8 tokens)...", flush=True)
    ok_any = False
    for endpoint in ([a.endpoint] if a.endpoint else ["chat", "completions"]):
        try:
            be = get_backend(
                a.backend, model=a.model, endpoint=endpoint,
                **({"max_top_logprobs": a.max_top_logprobs} if a.max_top_logprobs else {}),
            )
            t = be.generate_trace(
                "Write a short poem about salt.", max_tokens=8, candidates=5,
                temperature=1.0,
            )
        except (APIRequestFailed, BackendUnsupported) as e:
            print(f"  {endpoint:<12} FAILED: {str(e).splitlines()[0]}", flush=True)
            continue
        except Exception as e:
            # Anything else --- a timeout, or a response shape we cannot parse
            # (some served models return no `logprobs` object at all). Report
            # it as a probe result rather than a traceback: the point of this
            # command is to answer "can this model be traced", and a crash
            # answers that badly.
            print(
                f"  {endpoint:<12} FAILED: {type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )
            continue

        from .corpus import REASONING_MARKERS
        head = t.text.strip()[:80].lower()
        preamble = any(m in head for m in REASONING_MARKERS)
        n_alts = [len(s.candidates) for s in t.steps]
        avg = sum(n_alts) / len(n_alts) if n_alts else 0
        tempered = not t.meta.get("logprobs_are_raw", True)
        usable = bool(t.steps) and avg >= 2
        print(
            f"  {endpoint:<12} {'OK ' if usable else 'NO '} "
            f"{len(t.steps)} steps, {avg:.1f} candidates/step, "
            f"{'sampling_logprob (tempered)' if tempered else 'raw logprob only'}",
            flush=True,
        )
        print(f"      sample: {t.text.strip()[:70]!r}")
        if preamble:
            print("      -> WRITES PREAMBLE, not verse. A reasoning model's visible")
            print("         output is its chain of thought; unusable as poems.")
        if not usable:
            print("      -> no runner-up tokens returned; this model cannot be traced")
        else:
            ok_any = True
            if endpoint == "completions":
                print("      -> branching available on this endpoint")

    if not ok_any:
        print("\nFAIL: this model does not expose usable logprobs. Try another.")
        return 2
    print("\nOK -- ready to run.")
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
    gate = dict(
        rank=a.rank,
        gated=bool(a.top_n or a.max_cost),
        gate_top_n=a.top_n,
        gate_max_cost=a.max_cost,
        lex=_make_lex(a),
        n_iter=a.permutations,
        seed=a.seed,
    )

    def maybe_slice(traces):
        if not a.poem_marker:
            return traces
        from .corpus import slice_to_poem
        kept, rep = slice_to_poem(traces, a.poem_marker)
        print(
            f"marker slice: kept {rep['kept']}/{rep['input_traces']} "
            f"(no marker: {rep['dropped_no_marker']}, too short: "
            f"{rep['dropped_too_short']})", file=sys.stderr,
        )
        if not kept:
            raise SystemExit(
                f"No trace contained {a.poem_marker!r}. The model never emitted "
                "it -- check the system prompt, or raise --max-tokens so it has "
                "room to finish thinking and write."
            )
        return kept

    if a.from_traces:
        # Re-analysis only: no generation, no cost. This is the correct path
        # for rank and gating sweeps -- it guarantees every setting is
        # compared on the *same* traces, preserving the pairing.
        traces = maybe_slice(load_traces(a.from_traces))
        print(f"analysing {len(traces)} existing traces (no generation)", file=sys.stderr)
        res = analyse_traces(traces, **gate)
    else:
        prompts = load_prompts(a.prompts) if a.prompts else []
        if not prompts:
            raise SystemExit("need --prompts (to generate) or --from-traces (to re-analyse)")
        be = _make_backend(a)
        if a.concurrency > 1 and a.backend == "hf":
            print(
                "warning: --concurrency > 1 is unsafe for a single local hf model; "
                "forcing 1", file=sys.stderr,
            )
            a.concurrency = 1

        def progress(i: int, n: int, _t: Any) -> None:
            print(f"\r  generating {i}/{n}", end="", file=sys.stderr, flush=True)

        traces = generate_traces(
            be, prompts,
            samples_per_prompt=a.samples,
            max_tokens=a.max_tokens,
            temperature=a.temperature,
            top_p=a.top_p,
            candidates=a.candidates,
            system=a.system,
            seed0=a.seed,
            concurrency=a.concurrency,
            on_progress=progress,
        )
        print(file=sys.stderr)
        traces = maybe_slice(traces)
        res = analyse_traces(
            traces,
            config={
                "n_prompts": len(prompts),
                "samples_per_prompt": a.samples,
                "temperature": a.temperature,
                "top_p": a.top_p,
                "candidates": a.candidates,
                "max_tokens": a.max_tokens,
                "seed0": a.seed,
                "concurrency": a.concurrency,
            },
            **gate,
        )

    print(res.summary())
    paths = save_corpus(res, a.out)
    print("\nwrote: " + ", ".join(paths.values()))
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    # A corpus file holds hundreds of poems; pick one rather than being stuck
    # with whichever trace happens to be lying around.
    # Detect a corpus by PARSING, not by filename or line count: a single
    # trace is written pretty-printed across many lines, so counting lines
    # misclassifies it, and the extension can be anything.
    with open(a.trace, encoding="utf-8") as _fh:
        _raw = _fh.read()
    try:
        json.loads(_raw)
        _is_corpus = False          # parses whole -> one trace
    except json.JSONDecodeError:
        _is_corpus = True           # trailing data -> JSON Lines
    if _is_corpus:
        traces = load_traces(a.trace)
        if a.pick == "best":
            # most near-ties = the most genuinely contested poem, which is the
            # one worth reading beside its shadow
            t = max(traces, key=lambda x: sum(1 for s in x.steps if s.margin < 0.15))
        else:
            t = traces[int(a.pick)]
        print(f"picked trace {traces.index(t)} of {len(traces)}", file=sys.stderr)
    else:
        t = GenerationTrace.load(a.trace)

    # Gated by default. The full comb substitutes at every position and reads
    # as word salad -- true to the method, but the wrong thing to put in front
    # of a reader as the headline artifact.
    # preserve_structure keeps line breaks intact so both readings stay verse.
    keep = not a.token_level
    gated = gated_shadow(
        t, a.rank, top_n=a.gate_top_n, max_cost=a.gate_max_cost,
        preserve_structure=keep,
    )
    full = shadow_poem(
        t, a.rank, preserve_structure=keep, word_aligned=keep
    )
    # Both layers ship in the page: gated is legible, full is what the method
    # recovers. --full only changes which one the page opens on.
    write_html(a.out, t, shadow=gated, full_shadow=full, title=a.title, depth=a.depth)
    print(
        f"wrote {a.out} — modes: poem / gated ({gated.n_substitutions} swapped) "
        f"/ full ({full.n_substitutions} swapped) / both"
    )
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

    k = sub.add_parser("check", help="preflight: validate the key and the model id")
    _backend_args(k)
    k.set_defaults(fn=cmd_check)

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
    c.add_argument("--prompts", default=None, help="prompt file (omit with --from-traces)")
    c.add_argument(
        "--from-traces",
        default=None,
        help="re-analyse an existing traces.jsonl instead of generating. "
        "Use this for rank/gating sweeps so every setting is compared on "
        "identical traces (and costs nothing).",
    )
    c.add_argument(
        "--concurrency", type=int, default=1,
        help="parallel generation requests; safe for hosted APIs, not for local hf",
    )
    c.add_argument(
        "--poem-marker", default=None,
        help="analyse only the tokens AFTER this marker in each generation. "
        "Use with reasoning models: let them think, ask for the marker, and "
        "measure only the poem. Traces lacking it are dropped and reported.",
    )
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
    r.add_argument(
        "--gate-top-n", type=int, default=10,
        help="swap only the N most contested positions (default). This is the "
        "legible artifact; the full comb reads as word salad.",
    )
    r.add_argument("--gate-max-cost", type=float, default=None,
                   help="swap only where the two readings were within N nats")
    r.add_argument(
        "--token-level", action="store_true",
        help="substitute at EVERY token, including line breaks and mid-word "
        "subword pieces -- what the statistics measure. Illegible, but it is "
        "the object the method actually recovers.",
    )
    r.add_argument(
        "--break-lines", action="store_true",
        help="also substitute whitespace/newline tokens (as the statistics do). "
        "Illegible, but shows every decision the sampler made.",
    )
    r.add_argument("--full", action="store_true",
                   help="substitute at EVERY position (the full counterfactual comb)")
    r.add_argument("--pick", default="best",
                   help="with a .jsonl corpus: 'best' (most near-ties) or an index")
    r.set_defaults(fn=cmd_render)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (BackendUnsupported, APIRequestFailed) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
