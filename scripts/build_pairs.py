"""Build a blinded (poem, branch) pair set for the human study.

    python scripts/build_pairs.py --traces runs/T1.0/traces.jsonl --n 100

WHY BRANCHES AND NOT SHADOW POEMS
---------------------------------
The obvious human test is poem vs gated shadow. It does not work. A shadow
token is chosen for its position but everything after it was written assuming
the original word, so the text breaks grammatically:

    poem   : "In trees and caves where light does not gleam"
    shadow : "In trees it caves where light does not gleam"

A rater spots that instantly and scores ~95%, having measured local
incoherence rather than any aesthetic difference. The null would be rejected
for entirely the wrong reason.

A *branch* forks at one contested token and lets the model write the rest
itself. Both texts are then fluent, both are texts the model would genuinely
have produced, and they differ only in which way one near-tied decision went.
That is the comparison the project's claim is actually about.

Each pair records which side is the original, but the sides are shuffled, so
the rater is blind. Catch trials (a full shadow comb, which IS word salad) are
mixed in to detect inattentive raters -- anyone who cannot spot those is not
reading, and their data should be dropped.
"""

from __future__ import annotations

import argparse
import json
import random

from shadow_anthology import get_backend, load_traces, shadow_poem
from shadow_anthology.shadow import _substitution


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="runs/T1.0/traces.jsonl")
    ap.add_argument("--n", type=int, default=100, help="number of real pairs")
    ap.add_argument("--catch", type=int, default=10, help="attention-check pairs")
    ap.add_argument("--max-gap", type=float, default=0.15,
                    help="only fork where the two readings were within this many nats")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study/pairs.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    traces = load_traces(a.traces)
    be = get_backend(a.backend, model=a.model)

    # Rank traces by how contested their closest call was: forking on a true
    # near-tie is the whole point, so prefer poems that contain one.
    scored = []
    for t in traces:
        best = None
        for step in t.steps:
            alt = step.alternative(1)
            if alt is None or step.index < 4:
                continue          # too early to fork: nothing shared to compare
            sub = _substitution(t, step, alt)
            if best is None or sub.gap < best.gap:
                best = sub
        if best is not None and best.gap <= a.max_gap:
            scored.append((best.gap, t, best))
    scored.sort(key=lambda x: x[0])
    print(f"{len(scored)}/{len(traces)} traces have a fork within {a.max_gap} nats")

    pairs = []
    for i, (gap, t, sub) in enumerate(scored[: a.n]):
        alt = t.steps[sub.index].alternative(1)
        branch = be.continue_from(
            t, sub.index, alt,
            max_tokens=t.params.get("max_tokens", 160), seed=a.seed + i,
        )
        original, other = t.text.strip(), branch.text.strip()
        if not other or other == original:
            continue
        flip = rng.random() < 0.5
        pairs.append({
            "id": f"p{i}",
            "kind": "branch",
            "prompt": t.prompt,
            "a": other if flip else original,
            "b": original if flip else other,
            "original_side": "b" if flip else "a",
            "fork_index": sub.index,
            "gap": gap,
            "written": sub.chosen.text,
            "rejected": sub.shadow.text,
        })
        print(f"\r  built {len(pairs)}/{a.n}", end="", flush=True)
    print()

    # Attention checks: the full comb really is word salad, so a rater who is
    # reading will always pick it out. Anyone at chance here was not reading.
    for j, (_, t, _) in enumerate(scored[a.n : a.n + a.catch]):
        salad = shadow_poem(t, 1, preserve_structure=True).text.strip()
        original = t.text.strip()
        flip = rng.random() < 0.5
        pairs.append({
            "id": f"c{j}", "kind": "catch", "prompt": t.prompt,
            "a": salad if flip else original,
            "b": original if flip else salad,
            "original_side": "b" if flip else "a",
        })

    rng.shuffle(pairs)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"pairs": pairs, "model": a.model, "source": a.traces}, fh,
                  ensure_ascii=False, indent=1)
    n_real = sum(1 for p in pairs if p["kind"] == "branch")
    print(f"wrote {a.out}: {n_real} branch pairs + {len(pairs)-n_real} catch trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
