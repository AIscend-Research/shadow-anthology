"""Rendering: making the sampler's choices visible.

Two surfaces.

`render_terminal` is for working: the poem, its shadow, and a table of the
closest calls, in ANSI.

`render_html` is the artwork. It shows the poem as written, with every token
carrying its own rejected alternatives underneath. Contested tokens are marked
by weight, not by decoration, so the page still reads as a poem rather than as
a visualisation of one. The reader can hold the poem and its unwritten
siblings in view at once --- which is the whole argument, made legible instead
of argued.

The HTML is fully self-contained (no fonts, scripts or styles fetched) and
renders in both light and dark.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any, Sequence

from .anthology import Anthology
from .shadow import ShadowPoem, gated_shadow, shadow_poem
from .trace import GenerationTrace

# -- ANSI ------------------------------------------------------------------

_RESET, _DIM, _BOLD = "\033[0m", "\033[2m", "\033[1m"
_RED, _YEL, _CYA, _GRY = "\033[31m", "\033[33m", "\033[36m", "\033[90m"


def render_terminal(
    trace: GenerationTrace, shadow: ShadowPoem | None = None, *, n_calls: int = 12
) -> str:
    sh = shadow or shadow_poem(trace, 1)
    out = [
        f"{_BOLD}poem{_RESET} {_GRY}({trace.model}, seed={trace.seed}, "
        f"{len(trace)} tokens){_RESET}",
        trace.text.strip(),
        "",
        f"{_BOLD}shadow{_RESET} {_GRY}(rank {sh.rank}, {sh.n_substitutions} "
        f"substitutions, Σgap={sh.total_gap:.2f} nats, "
        f"{sh.contested_fraction:.0%} model-preferred){_RESET}",
        sh.text.strip(),
        "",
        f"{_BOLD}closest calls{_RESET} {_GRY}(where the poem was nearest to "
        f"going otherwise){_RESET}",
    ]
    for s in sh.closest_calls(n_calls):
        colour = _RED if s.gap < 0.05 else (_YEL if s.gap < 0.3 else _CYA)
        ctx = s.prefix.replace("\n", "⏎")[-28:]
        out.append(
            f"  {_GRY}…{ctx}{_RESET} {colour}{s.chosen.text!r}{_RESET} "
            f"{_DIM}↔{_RESET} {colour}{s.shadow.text!r}{_RESET} "
            f"{_GRY}gap={s.gap:.3f} nats, H={s.entropy:.2f} bits{_RESET}"
        )
    out += [
        "",
        f"{_GRY}sampler footprint: {trace.offrank_fraction:.1%} of tokens were "
        f"not the model's first choice; mean entropy {trace.mean_entropy:.2f} bits"
        f"{_RESET}",
    ]
    return "\n".join(out)


# -- HTML ------------------------------------------------------------------

_CSS = """
:root{--bg:#faf8f4;--fg:#16130f;--mut:#6d6459;--line:#e0d9cd;--hot:#b23a2c;--warm:#c58a1e;--cool:#4d6b8a;--panel:#fffdf9}
@media (prefers-color-scheme:dark){:root{--bg:#12100e;--fg:#ece6dc;--mut:#8b8177;--line:#2b2724;--hot:#e4705f;--warm:#d9a441;--cool:#7fa5c9;--panel:#191614}}
:root[data-theme="dark"]{--bg:#12100e;--fg:#ece6dc;--mut:#8b8177;--line:#2b2724;--hot:#e4705f;--warm:#d9a441;--cool:#7fa5c9;--panel:#191614}
:root[data-theme="light"]{--bg:#faf8f4;--fg:#16130f;--mut:#6d6459;--line:#e0d9cd;--hot:#b23a2c;--warm:#c58a1e;--cool:#4d6b8a;--panel:#fffdf9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.7 Georgia,'Iowan Old Style','Times New Roman',serif;overflow-x:hidden}
.wrap{max-width:52rem;margin:0 auto;padding:3rem 1.25rem 6rem}
h1{font-size:1.6rem;font-weight:400;letter-spacing:.01em;margin:0 0 .25rem}
.sub{color:var(--mut);font-size:.9rem;margin:0 0 2rem;font-style:italic}
.meta{color:var(--mut);font-size:.78rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
 border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:.6rem 0;margin:0 0 2rem;
 display:flex;flex-wrap:wrap;gap:.35rem 1.25rem}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;margin:0 0 1.75rem}
button{font:inherit;font-size:.82rem;background:transparent;color:var(--mut);border:1px solid var(--line);
 border-radius:2px;padding:.3rem .7rem;cursor:pointer}
button[aria-pressed="true"]{color:var(--fg);border-color:var(--fg)}
.poem{white-space:pre-wrap;font-size:1.16rem;line-height:2.1;margin:0 0 2.5rem}
.tok{position:relative;border-radius:2px;transition:background .12s}
.tok.c1{box-shadow:inset 0 -2px 0 var(--hot)}
.tok.c2{box-shadow:inset 0 -2px 0 var(--warm)}
.tok.c3{box-shadow:inset 0 -1px 0 var(--cool)}
.tok:hover{background:color-mix(in srgb,var(--fg) 8%,transparent)}
.tok:hover .pop{display:block}
.pop{display:none;position:absolute;left:0;bottom:1.9em;z-index:9;min-width:14rem;max-width:min(22rem,88vw);
 background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.55rem .7rem;
 font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fg);box-shadow:0 6px 22px rgba(0,0,0,.16);
 white-space:normal;text-align:left}
.pop .hd{color:var(--mut);margin-bottom:.3rem}
.pop .row{display:flex;justify-content:space-between;gap:.75rem}
.pop .row.sel{color:var(--hot)}
.pop .bar{display:inline-block;height:3px;background:var(--cool);vertical-align:middle;margin-right:.4rem}
.shadow{display:none}
body.mode-shadow .chosen{display:none}
body.mode-shadow .shadow{display:inline}
body.mode-both .shadow{display:inline;color:var(--mut);font-style:italic}
body.mode-both .shadow::before{content:"/"}
h2{font-size:.78rem;font-weight:400;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
 border-top:1px solid var(--line);padding-top:1.1rem;margin:3rem 0 1rem}
table{width:100%;border-collapse:collapse;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
td,th{text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:400}
td.num{text-align:right;white-space:nowrap;color:var(--mut)}
.branch{border-left:2px solid var(--line);padding:.1rem 0 .1rem 1rem;margin:0 0 1.5rem}
.branch .at{font:12px ui-monospace,Menlo,monospace;color:var(--mut);margin-bottom:.35rem}
.branch .txt{white-space:pre-wrap}
.branch .txt .pre{color:var(--mut)}
footer{color:var(--mut);font-size:.75rem;margin-top:4rem;border-top:1px solid var(--line);padding-top:1rem}
"""

_JS = """
(function(){
 var b=document.body;
 document.querySelectorAll('[data-mode]').forEach(function(btn){
  btn.addEventListener('click',function(){
   b.className='mode-'+btn.dataset.mode;
   document.querySelectorAll('[data-mode]').forEach(function(o){
    o.setAttribute('aria-pressed',String(o===btn));});
  });
 });
})();
"""


def render_html(
    trace: GenerationTrace,
    *,
    shadow: ShadowPoem | None = None,
    anthology: Anthology | None = None,
    title: str = "The Shadow Anthology",
    depth: int = 3,
    standalone: bool = True,
) -> str:
    """Render one trace as a readable, self-contained page.

    `depth` controls how many rejected alternatives each token exposes.
    Set `standalone=False` to emit body content only (for embedding).
    """
    sh = shadow or shadow_poem(trace, 1)
    body = _body(trace, sh, anthology, title, depth)
    if not standalone:
        return body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body class=\"mode-poem\">\n{body}\n<script>{_JS}</script>\n</body>\n</html>\n"
    )


def _body(
    trace: GenerationTrace,
    sh: ShadowPoem,
    anth: Anthology | None,
    title: str,
    depth: int,
) -> str:
    parts = [
        '<div class="wrap">',
        f"<h1>{html.escape(title)}</h1>",
        '<p class="sub">The poem as written, and the poems rejected at every step.</p>',
        _meta(trace, sh),
        _controls(),
        _poem_markup(trace, sh, depth),
        _calls_table(sh),
    ]
    if anth is not None and len(anth):
        parts.append(_branches(anth))
    parts.append(
        '<footer>Every alternative shown was in the sampler&rsquo;s support at that '
        'position. Shadow tokens are conditioned on the written prefix, so the shadow '
        'poem is a record of rejected choices, not a text the model would have '
        'produced end to end &mdash; for those, see the branches.</footer>'
    )
    parts.append("</div>")
    return "\n".join(parts)


def _meta(trace: GenerationTrace, sh: ShadowPoem) -> str:
    p = trace.params
    bits = [
        f"model {html.escape(str(trace.model))}",
        f"backend {html.escape(str(trace.backend))}",
        f"seed {trace.seed}",
        f"T {p.get('temperature')}",
        f"top_p {p.get('top_p')}",
        f"{len(trace)} tokens",
        f"off-argmax {trace.offrank_fraction:.1%}",
        f"mean H {trace.mean_entropy:.2f} bits",
        f"shadow distance {sh.total_gap:.1f} nats",
    ]
    return '<div class="meta">' + "".join(f"<span>{b}</span>" for b in bits) + "</div>"


def _controls() -> str:
    return (
        '<div class="controls">'
        '<button data-mode="poem" aria-pressed="true">poem</button>'
        '<button data-mode="shadow" aria-pressed="false">shadow</button>'
        '<button data-mode="both" aria-pressed="false">both</button>'
        "</div>"
    )


def _poem_markup(trace: GenerationTrace, sh: ShadowPoem, depth: int) -> str:
    sub_at = {s.index: s for s in sh.substitutions}
    out = ['<div class="poem">']
    for step in trace.steps:
        chosen = step.chosen
        alts = step.alternatives()[:depth]
        cls = "tok"
        if alts:
            gap = abs(chosen.logprob - alts[0].logprob)
            cls += " c1" if gap < 0.10 else (" c2" if gap < 0.40 else (" c3" if gap < 1.0 else ""))
        shadow_tok = sub_at[step.index].shadow.text if step.index in sub_at else chosen.text

        rows = [f'<div class="hd">position {step.index} &middot; '
                f'H={step.entropy:.2f} bits &middot; rank taken {step.chosen_rank}</div>']
        for c in step.candidates[: depth + 1]:
            sel = " sel" if c.token_id == chosen.token_id and c.text == chosen.text else ""
            w = max(1, int(round(math.exp(c.logprob) * 60)))
            rows.append(
                f'<div class="row{sel}"><span>'
                f'<span class="bar" style="width:{w}px"></span>'
                f"{html.escape(_vis(c.text))}</span>"
                f"<span>{math.exp(c.logprob):.3f}</span></div>"
            )
        pop = f'<span class="pop">{"".join(rows)}</span>'
        out.append(
            f'<span class="{cls}">'
            f'<span class="chosen">{html.escape(chosen.text)}</span>'
            f'<span class="shadow">{html.escape(shadow_tok)}</span>'
            f"{pop}</span>"
        )
    out.append("</div>")
    return "".join(out)


def _calls_table(sh: ShadowPoem, n: int = 15) -> str:
    rows = [
        "<h2>Closest calls</h2>",
        '<div class="scroll"><table><tr><th>pos</th><th>context</th><th>written</th>'
        "<th>rejected</th><th>gap nats</th><th>H bits</th></tr>",
    ]
    for s in sh.closest_calls(n):
        rows.append(
            f'<tr><td class="num">{s.index}</td>'
            f"<td>&hellip;{html.escape(_vis(s.prefix[-26:]))}</td>"
            f"<td>{html.escape(_vis(s.chosen.text))}</td>"
            f"<td>{html.escape(_vis(s.shadow.text))}</td>"
            f'<td class="num">{s.gap:.3f}</td>'
            f'<td class="num">{s.entropy:.2f}</td></tr>'
        )
    rows.append("</table></div>")
    return "".join(rows)


def _branches(anth: Anthology, n: int = 8) -> str:
    out = ["<h2>Branches &mdash; poems the model would have written</h2>"]
    for b in anth.nearest(n):
        out.append(
            '<div class="branch">'
            f'<div class="at">split at token {b.branch_at}: '
            f"{html.escape(_vis(b.displaced.text))} &rarr; "
            f"{html.escape(_vis(b.forced.text))} "
            f"(gap {b.gap:.3f} nats)</div>"
            f'<div class="txt"><span class="pre">{html.escape(b.shared_prefix)}</span>'
            f"{html.escape(b.divergent_tail)}</div></div>"
        )
    if anth.dropped:
        out.append(
            f'<p class="sub">{len(anth.dropped)} further branch points were not '
            "generated (call budget).</p>"
        )
    return "".join(out)


def _vis(s: str) -> str:
    return s.replace("\n", "⏎").replace("\t", "⇥")


def write_html(path: str, *args: Any, **kwargs: Any) -> str:
    doc = render_html(*args, **kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
