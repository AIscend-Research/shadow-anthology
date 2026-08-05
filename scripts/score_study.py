"""Score the human study.

    python scripts/score_study.py --ratings ratings.json --pairs study/pairs.json

Two questions, each tested against chance (0.5) with the same paired
permutation machinery as the rest of the project:

  which_original -- can a reader tell the written poem from one that diverged
                    at a single near-tied decision? A null here is the strong
                    result: it says the sampler's choice is imperceptible.

  which_vivid    -- is the written poem preferred? A null says the sampler's
                    choice is not an improvement either.

Raters who fail the attention checks are dropped before anything is computed,
and the drop is reported rather than quietly applied. A null result is only
worth something if you can show the rater was awake.
"""

from __future__ import annotations

import argparse
import json

from shadow_anthology.stats import bootstrap_ci, holm_bonferroni, paired_permutation_test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", nargs="+", required=True,
                    help="one or more ratings.json files (one per rater)")
    ap.add_argument("--pairs", default="study/pairs.json")
    ap.add_argument("--catch-threshold", type=float, default=0.8,
                    help="min accuracy on attention checks to keep a rater")
    ap.add_argument("--n-iter", type=int, default=20000)
    a = ap.parse_args()

    kept, dropped = [], []
    for path in a.ratings:
        with open(path, encoding="utf-8") as fh:
            rs = json.load(fh)["ratings"]
        catch = [r for r in rs
                 if r["kind"] == "catch" and r["question"] == "which_original"
                 and r["chose"]]
        acc = (sum(r["chose"] == r["original_side"] for r in catch) / len(catch)
               if catch else 0.0)
        (kept if acc >= a.catch_threshold else dropped).append((path, rs, acc))

    for path, _, acc in dropped:
        print(f"DROPPED {path}: {acc:.0%} on attention checks "
              f"(threshold {a.catch_threshold:.0%}) — not reading carefully")
    if not kept:
        print("\nNo rater passed the attention checks. Nothing to report.")
        return 1
    print(f"{len(kept)} rater(s) kept, {len(dropped)} dropped\n")

    results = []
    for qname, label in (("which_original", "discrimination"),
                         ("which_vivid", "preference")):
        # One value per (rater, pair): 1 if the original was picked, else 0.
        # Skips are excluded rather than counted as errors -- "can't tell" is
        # not the same as guessing wrong.
        vals = []
        for _, rs, _ in kept:
            for r in rs:
                if r["kind"] != "branch" or r["question"] != qname or not r["chose"]:
                    continue
                vals.append(1.0 if r["chose"] == r["original_side"] else 0.0)
        if len(vals) < 5:
            print(f"{label}: too few responses ({len(vals)})")
            continue
        diffs = [v - 0.5 for v in vals]
        res = paired_permutation_test(diffs, n_iter=a.n_iter, name=label)
        acc = sum(vals) / len(vals)
        lo, hi = bootstrap_ci(vals)
        results.append((label, acc, lo, hi, res, len(vals)))

    for r in holm_bonferroni([x[4] for x in results]):
        pass  # correction applied in place across the two questions

    print(f"  {'question':<16}{'n':>6}{'accuracy':>11}{'95% CI':>18}{'p (Holm)':>11}")
    print("  " + "-" * 62)
    for label, acc, lo, hi, res, n in results:
        p = res.p_adjusted if res.p_adjusted is not None else res.p_value
        star = "*" if p < 0.05 else " "
        print(f"{star} {label:<16}{n:>6}{acc:>10.1%}{f'[{lo:.1%}, {hi:.1%}]':>18}{p:>11.4g}")

    print()
    for label, acc, lo, hi, res, n in results:
        p = res.p_adjusted if res.p_adjusted is not None else res.p_value
        if p >= 0.05:
            print(f"  {label}: indistinguishable from chance. With n={n}, the CI "
                  f"[{lo:.1%}, {hi:.1%}] bounds any real effect.")
        else:
            print(f"  {label}: reliably above chance at {acc:.1%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
