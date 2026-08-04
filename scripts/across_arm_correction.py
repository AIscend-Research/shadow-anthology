"""Family-wise correction across ALL arms, not just within each one.

    python scripts/across_arm_correction.py runs/*/

Each corpus run Holm-corrects its own 11 metrics. But a study reports many
arms -- 4 temperatures, 3 ranks, a gated condition -- so the real family is
~88 tests, not 11. Correcting within arms and then reading the starred rows
across arms is a well-known way to manufacture findings: with 88 tests at
alpha=0.05 you expect ~4 false positives even if nothing is true.

This pools every p-value, applies Holm and Benjamini-Hochberg to the whole
family, and reports what survives. Holm controls the family-wise error rate
(the probability of ANY false positive) and is the conservative choice;
Benjamini-Hochberg controls the false discovery rate (the expected share of
claims that are wrong) and is the usual choice when you have many tests and
expect several real effects. Report whichever you pre-specified -- and say
which, because choosing after seeing the results is the same error one level up.
"""

from __future__ import annotations

import glob
import json
import sys


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (m - rank)))
        out[i] = running
    return out


def benjamini_hochberg(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    out = [1.0] * m
    running = 1.0
    for pos, i in enumerate(order):
        rank = m - pos
        running = min(running, min(1.0, ps[i] * m / rank))
        out[i] = running
    return out


def main(paths: list[str]) -> int:
    rows = []
    for d in sorted(paths):
        d = d.rstrip("/")
        try:
            with open(f"{d}/results.json", encoding="utf-8") as fh:
                res = json.load(fh)
        except FileNotFoundError:
            continue
        arm = d.rsplit("/", 1)[-1]
        for t in res["tests"]:
            if t["n"] < 2:
                continue
            rows.append({
                "arm": arm, "metric": t["name"], "n": t["n"],
                "delta": t["mean_diff"], "dz": t["effect_size"],
                "p_raw": t["p_value"], "p_within": t.get("p_adjusted"),
            })

    if not rows:
        print("no results found; pass run directories, e.g. runs/*/")
        return 2

    # rank1 IS T1.0 re-analysed at rank 1 -- the identical computation on the
    # identical traces. Counting it twice inflates the family and double-counts
    # the same evidence, so identical (metric, n, delta) rows collapse to one.
    seen, unique, dupes = {}, [], 0
    for r in rows:
        key = (r["metric"], r["n"], round(r["delta"], 10), round(r["p_raw"], 12))
        if key in seen:
            seen[key]["arm"] += f"={r['arm']}"
            dupes += 1
            continue
        seen[key] = r
        unique.append(r)
    if dupes:
        print(f"  (collapsed {dupes} duplicate tests: identical analyses in two arms)\n")
    rows = unique

    ps = [r["p_raw"] for r in rows]
    for r, h, b in zip(rows, holm(ps), benjamini_hochberg(ps)):
        r["p_holm_all"], r["p_bh_all"] = h, b

    n_within = sum(1 for r in rows if (r["p_within"] or 1) < 0.05)
    n_holm = sum(1 for r in rows if r["p_holm_all"] < 0.05)
    n_bh = sum(1 for r in rows if r["p_bh_all"] < 0.05)

    print(f"{len(rows)} tests across {len({r['arm'] for r in rows})} arms\n")
    print(f"  significant, corrected WITHIN each arm only : {n_within}")
    print(f"  significant, Holm across the whole family   : {n_holm}")
    print(f"  significant, Benjamini-Hochberg (FDR)       : {n_bh}")
    print(f"\n  {n_within - n_holm} claims do not survive family-wide correction.\n")

    print(f"  {'arm':<8}{'metric':<18}{'n':>5}{'delta':>10}{'d_z':>8}"
          f"{'p(within)':>11}{'p(Holm)':>10}{'p(BH)':>9}   verdict")
    print("  " + "-" * 96)
    for r in sorted(rows, key=lambda r: r["p_raw"]):
        if (r["p_within"] or 1) >= 0.05 and r["p_bh_all"] >= 0.05:
            continue  # never significant under any scheme: omit for brevity
        if r["p_holm_all"] < 0.05:
            verdict = "survives Holm"
        elif r["p_bh_all"] < 0.05:
            verdict = "survives FDR only"
        else:
            verdict = "LOST — was within-arm only"
        print(f"  {r['arm']:<8}{r['metric']:<18}{r['n']:>5}{r['delta']:>+10.4f}"
              f"{r['dz']:>+8.3f}{(r['p_within'] or 1):>11.4g}"
              f"{r['p_holm_all']:>10.4g}{r['p_bh_all']:>9.4g}   {verdict}")

    print(
        "\nAny row marked LOST was significant only because its arm was corrected\n"
        "in isolation. Do not report those as findings."
    )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("runs/*/"))
    raise SystemExit(main(args))
