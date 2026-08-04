"""Rank-ladder control: is 'poem vs shadow' just 'more probable vs less probable'?

    python scripts/rank_control.py runs/rank1 runs/rank2 runs/rank3

The paired design holds prompt, seed, model and length fixed, so a significant
poem-vs-shadow difference is tempting to read as "the sampler systematically
prefers X". But the shadow is *defined* as a lower-probability token, and token
probability correlates with corpus frequency, with repetition, and with word
length. So some differences are guaranteed by the selection rule alone.

This control separates the two. Compare the poem against successively deeper
rejected tokens (rank 1, 2, 3):

  * If a metric's delta GROWS MONOTONICALLY with rank, it is tracking
    probability depth. The poem-vs-shadow result is then a special case of
    "more probable vs less probable" and says nothing specific about the
    sampler's editorial choice.

  * If a metric's delta is FLAT or NON-MONOTONE across ranks, it is not
    explained by depth alone, and is a candidate for a real difference between
    what was chosen and what was rejected.

Report this table alongside any poem-vs-shadow claim. Without it, the headline
effects are uninterpretable.
"""

from __future__ import annotations

import json
import sys

METRICS = (
    "rarity", "repetition", "risk", "type_token_ratio", "mean_word_length",
    "concreteness", "abstract_ratio", "imagery", "sensory_density",
    "valence", "arousal",
)


def load(run_dir: str) -> dict[str, dict]:
    with open(f"{run_dir}/results.json", encoding="utf-8") as fh:
        return {t["name"]: t for t in json.load(fh)["tests"]}


def monotone(vals: list[float]) -> bool:
    inc = all(b >= a for a, b in zip(vals, vals[1:]))
    dec = all(b <= a for a, b in zip(vals, vals[1:]))
    return (inc or dec) and abs(vals[-1]) > abs(vals[0]) * 1.5


def main(dirs: list[str]) -> int:
    if len(dirs) < 2:
        print(__doc__)
        return 2
    runs = [load(d) for d in dirs]
    names = [d.rsplit("/", 1)[-1] for d in dirs]

    print("Rank-ladder control — poem minus rank-k shadow")
    print("  * = significant after Holm correction\n")
    print(f"  {'metric':<18}" + "".join(f"{n:>13}" for n in names) + "   verdict")
    print("  " + "-" * (18 + 13 * len(names) + 30))

    for m in METRICS:
        if not all(m in r for r in runs):
            continue
        deltas = [r[m]["mean_diff"] for r in runs]
        cells = ""
        for r in runs:
            star = "*" if (r[m].get("p_adjusted") or 1.0) < 0.05 else " "
            cells += f"{r[m]['mean_diff']:>+12.4f}{star}"
        any_sig = any((r[m].get("p_adjusted") or 1.0) < 0.05 for r in runs)
        if monotone(deltas) and any_sig:
            verdict = "DEPTH ARTIFACT — grows with rank"
        elif any_sig:
            verdict = "candidate real effect"
        else:
            verdict = "no effect"
        print(f"  {m:<18}{cells}   {verdict}")

    print(
        "\nA 'DEPTH ARTIFACT' verdict means the metric tracks how far down the\n"
        "probability ranking you go, so it cannot support a claim about what\n"
        "the sampler chose. Report it as such, or drop it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
