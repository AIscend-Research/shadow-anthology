"""Build the blinded rating interface from a pair set.

    python scripts/build_rater.py --pairs study/pairs.json --out study/rater.html

Produces one self-contained HTML file. Open it, rate, click Download; it writes
ratings.json, which scripts/score_study.py feeds into the same paired
permutation test as everything else.

Design constraints, all of them there to stop the study measuring the wrong
thing:

  * Sides are shuffled per pair at build time and never labelled.
  * No feedback, ever. Feedback would let a rater learn the model's tics and
    turn a perception test into a training task.
  * Two questions per pair: discrimination ("which is the original?") and
    preference ("which is more vivid?"). They answer different things --- one
    asks whether the sampler's choice is perceptible, the other whether it is
    better.
  * Catch trials are mixed in. A rater at chance on those was not reading, and
    the scorer drops them.
  * Keyboard-driven, because 120 pairs by mouse is how attention dies.
"""

from __future__ import annotations

import argparse
import html
import json

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Anthology — rating</title><style>
:root{--bg:#faf8f4;--fg:#16130f;--mut:#6d6459;--line:#e0d9cd;--sel:#b23a2c;--panel:#fffdf9}
@media(prefers-color-scheme:dark){:root{--bg:#12100e;--fg:#ece6dc;--mut:#8b8177;--line:#2b2724;--sel:#e4705f;--panel:#191614}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 Georgia,'Iowan Old Style',serif}
.wrap{max-width:62rem;margin:0 auto;padding:1.5rem 1.25rem 4rem}
.bar{height:3px;background:var(--line);margin-bottom:1.25rem}
.bar div{height:3px;background:var(--sel);transition:width .2s}
.hd{display:flex;justify-content:space-between;color:var(--mut);font-size:.78rem;
 font-family:ui-monospace,Menlo,monospace;margin-bottom:1rem}
.q{font-size:.95rem;margin:0 0 .75rem}
.q b{font-weight:400;border-bottom:1px solid var(--sel)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:44rem){.cols{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:1rem 1.1rem;
 white-space:pre-wrap;cursor:pointer;font-size:1.02rem;line-height:1.85}
.card:hover{border-color:var(--mut)}
.card.on{border-color:var(--sel);box-shadow:inset 0 0 0 1px var(--sel)}
.key{float:right;color:var(--mut);font:11px ui-monospace,Menlo,monospace;border:1px solid var(--line);
 border-radius:2px;padding:0 .3rem;margin-left:.5rem}
.foot{margin-top:1.25rem;color:var(--mut);font-size:.8rem;display:flex;gap:1rem;align-items:center}
button{font:inherit;font-size:.85rem;background:transparent;color:var(--fg);border:1px solid var(--line);
 border-radius:2px;padding:.35rem .8rem;cursor:pointer}
button:hover{border-color:var(--fg)}
#done{display:none;text-align:center;padding:4rem 0}
#done h2{font-weight:400}
</style></head><body><div class="wrap">
<div class="bar"><div id="prog" style="width:0"></div></div>
<div class="hd"><span id="count"></span><span id="phase"></span></div>
<div id="task">
  <p class="q" id="question"></p>
  <div class="cols">
    <div class="card" id="cardA" data-side="a"><span class="key">1</span><span id="txtA"></span></div>
    <div class="card" id="cardB" data-side="b"><span class="key">2</span><span id="txtB"></span></div>
  </div>
  <div class="foot">
    <button id="skip">can't tell / skip &nbsp;<span class="key">3</span></button>
    <span>keys: 1 = left, 2 = right, 3 = skip</span>
  </div>
</div>
<div id="done"><h2>Done — thank you.</h2>
  <p class="q">Nothing was uploaded. Download the file and send it on.</p>
  <button id="dl">Download ratings.json</button></div>
</div><script>
const PAIRS = __PAIRS__;
const QUESTIONS = [
  ["which_original", "One of these is the poem the model wrote. The other diverges from it at a single word, after which the model continued on its own. <b>Which is the original?</b>"],
  ["which_vivid",    "<b>Which is more vivid?</b> Judge the writing, not which you think is the original."]
];
let i = 0, q = 0, out = [], t0 = Date.now();
const $ = id => document.getElementById(id);

function draw(){
  if (i >= PAIRS.length) return finish();
  const p = PAIRS[i];
  $("question").innerHTML = QUESTIONS[q][1];
  $("txtA").textContent = p.a;
  $("txtB").textContent = p.b;
  $("cardA").classList.remove("on"); $("cardB").classList.remove("on");
  $("count").textContent = `pair ${i+1} of ${PAIRS.length}`;
  $("phase").textContent = `question ${q+1} of 2`;
  $("prog").style.width = (100*(i + q/2)/PAIRS.length) + "%";
  t0 = Date.now();
}
function answer(side){
  const p = PAIRS[i];
  out.push({id:p.id, kind:p.kind, question:QUESTIONS[q][0], chose:side,
            original_side:p.original_side, ms:Date.now()-t0});
  if (side) { $(side==="a"?"cardA":"cardB").classList.add("on"); }
  q++; if (q >= QUESTIONS.length) { q = 0; i++; }
  setTimeout(draw, side ? 120 : 0);
}
function finish(){
  $("task").style.display="none"; $("done").style.display="block";
  $("prog").style.width="100%"; $("count").textContent=""; $("phase").textContent="";
}
$("cardA").onclick = () => answer("a");
$("cardB").onclick = () => answer("b");
$("skip").onclick  = () => answer(null);
document.addEventListener("keydown", e => {
  if (e.key==="1") answer("a"); else if (e.key==="2") answer("b");
  else if (e.key==="3") answer(null);
});
$("dl").onclick = () => {
  const blob = new Blob([JSON.stringify({ratings:out, n_pairs:PAIRS.length}, null, 1)],
                        {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "ratings.json"; a.click();
};
draw();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="study/pairs.json")
    ap.add_argument("--out", default="study/rater.html")
    a = ap.parse_args()

    with open(a.pairs, encoding="utf-8") as fh:
        data = json.load(fh)

    # Strip the answer key of everything the rater must not see, but keep
    # original_side: the page needs it to record correctness, and a rater
    # determined to cheat could read the source either way. Blinding here is
    # about not being told, not about cryptography.
    pairs = [
        {k: p[k] for k in ("id", "kind", "a", "b", "original_side")}
        for p in data["pairs"]
    ]
    page = _PAGE.replace("__PAIRS__", json.dumps(pairs, ensure_ascii=False))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(page)

    n_real = sum(1 for p in pairs if p["kind"] == "branch")
    print(f"wrote {a.out}: {n_real} branch pairs + {len(pairs)-n_real} catch trials")
    print(f"  ~{len(pairs)*2*12//60} min to rate at 12s per question")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
