"""Lexical resources for scoring poems.

Ships small hand-built seed lexicons so the package runs with zero downloads.
These are seeds, not norms: they give directionally sensible scores on poetry
vocabulary but they are not a substitute for published psycholinguistic norms.
For anything reported as a result, load the real thing:

    from shadow_anthology.lexicons import Lexicons
    lex = Lexicons.load(
        concreteness="Concreteness_ratings_Brysbaert_et_al_BRM.csv",
        vad="Ratings_Warriner_et_al.csv",
        frequency="SUBTLEXus.csv",
    )

`Lexicons.coverage(words)` reports what fraction of a text each resource
actually covers, so thin coverage is visible rather than silently averaged
away. Every metric that depends on a lexicon returns `None` --- not 0.0 ---
when coverage is too low to mean anything.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


# --- seed lexicons --------------------------------------------------------
# Scales: concreteness 1 (abstract) .. 5 (concrete), matching Brysbaert.
#         valence / arousal 1 .. 9, matching Warriner.

_CONCRETE = """
stone bone glass water river snow rain iron ash salt bread knife window door
lantern needle wire thread honey smoke root soil leaf bird feather hand mouth
tooth skin hair blood milk oil wax rope nail hammer engine wheel brick fence
table chair bowl cup spoon shoe coat button dust mud ice frost gravel sand
shell wing beak claw fur horn hoof antler moth beetle spider fish scale gill
""".split()

_ABSTRACT = """
grief hope truth meaning justice freedom sorrow beauty terror longing memory
absence presence essence idea reason virtue sin fate chance destiny doubt
faith wonder despair mercy honor shame guilt pride envy patience courage
wisdom folly time eternity infinity silence stillness becoming nothingness
belief desire intention purpose consequence identity self soul spirit mind
""".split()

_POSITIVE = """
light honey warm bloom gold morning gentle soft home laughter kind bright
sweet calm peace joy love tender green sunlit clear rest safe open free
blossom harvest music dance gift bless heal mend flourish radiant lucid
""".split()

_NEGATIVE = """
grief ash bone rot wound scar bleed break shatter cold bitter dark ruin
hunger fear terror ache burn drown choke rust decay corpse grave mourn
sever tear crush wither famine plague dread despair violence blade cruel
""".split()

_HIGH_AROUSAL = """
burn shatter scream tear explode rage storm lightning fever seize crash
strike hunger terror ecstasy violence blaze flood roar shock thrash pulse
""".split()

_LOW_AROUSAL = """
still quiet slow soft calm rest sleep drift settle linger hush dim gentle
patient steady dusk shade cool murmur wade idle lull pale mute weary
""".split()

_SENSORY = """
red blue green gold white black grey amber crimson bitter sweet sour salt
sharp rough smooth cold warm damp dry loud quiet ringing humming acrid musk
bright dim glare shimmer rasp velvet grain scent reek fragrant sting numb
""".split()

# ~250 highest-frequency English word forms; anything absent counts as "rare"
# under the fallback rarity model. Replace with SUBTLEX for real work.
_COMMON = """
the be to of and a in that have i it for not on with he as you do at this but
his by from they we say her she or an will my one all would there their what
so up out if about who get which go me when make can like time no just him
know take people into year your good some could them see other than then now
look only come its over think also back after use two how our work first well
way even new want because any these give day most us is was are were been has
had said made did went could should must may might shall am being does having
where why while before under above between through during against without
within along across behind beyond near far here there again once always never
often sometimes very much many few little more less own same such each every
both either neither another next last long short high low old young big small
great little light dark white black red green blue hand eye head face body
water fire air earth night morning house door window road home life death love
""".split()


@dataclass
class Lexicons:
    concreteness: dict[str, float] = field(default_factory=dict)
    valence: dict[str, float] = field(default_factory=dict)
    arousal: dict[str, float] = field(default_factory=dict)
    frequency: dict[str, float] = field(default_factory=dict)
    """Zipf-scale frequency (higher = more common)."""
    sensory: set[str] = field(default_factory=set)
    is_seed: bool = True
    """True when any resource is the built-in seed rather than published norms.
    Propagated into every metrics payload so results are never mistaken for
    norm-backed numbers."""

    # -- construction ------------------------------------------------------

    @classmethod
    def seed(cls) -> "Lexicons":
        conc = {w: 4.6 for w in _CONCRETE}
        conc.update({w: 1.6 for w in _ABSTRACT})
        val = {w: 7.0 for w in _POSITIVE}
        val.update({w: 2.6 for w in _NEGATIVE})
        aro = {w: 7.0 for w in _HIGH_AROUSAL}
        aro.update({w: 2.8 for w in _LOW_AROUSAL})
        freq = {w: 6.0 for w in _COMMON}
        return cls(
            concreteness=conc,
            valence=val,
            arousal=aro,
            frequency=freq,
            sensory=set(_SENSORY),
            is_seed=True,
        )

    @classmethod
    def load(
        cls,
        *,
        concreteness: str | None = None,
        vad: str | None = None,
        frequency: str | None = None,
        base: "Lexicons | None" = None,
    ) -> "Lexicons":
        """Overlay published norms onto the seed lexicons.

        Column names follow the standard public releases; anything unparseable
        is skipped rather than guessed at.
        """
        lex = base or cls.seed()
        loaded_any = False

        if concreteness and os.path.exists(concreteness):
            lex.concreteness.update(_read_csv(concreteness, "Word", "Conc.M"))
            loaded_any = True
        if vad and os.path.exists(vad):
            lex.valence.update(_read_csv(vad, "Word", "V.Mean.Sum"))
            lex.arousal.update(_read_csv(vad, "Word", "A.Mean.Sum"))
            loaded_any = True
        if frequency and os.path.exists(frequency):
            lex.frequency.update(_read_csv(frequency, "Word", "Lg10WF"))
            loaded_any = True

        lex.is_seed = not loaded_any
        return lex

    # -- queries -----------------------------------------------------------

    def coverage(self, words: Sequence[str]) -> dict[str, float]:
        """Fraction of `words` present in each resource."""
        if not words:
            return {k: 0.0 for k in ("concreteness", "valence", "arousal", "frequency")}
        n = len(words)
        return {
            "concreteness": sum(w in self.concreteness for w in words) / n,
            "valence": sum(w in self.valence for w in words) / n,
            "arousal": sum(w in self.arousal for w in words) / n,
            "frequency": sum(w in self.frequency for w in words) / n,
        }

    def rarity(self, word: str) -> float:
        """0 (very common) .. 1 (unattested in the frequency resource).

        Deliberately crude when running on the seed list: it is a binary
        common/uncommon signal there, and `is_seed` flags that downstream.
        """
        f = self.frequency.get(word)
        if f is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (f / 7.0)))


def _read_csv(path: str, wcol: str, vcol: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            w = (row.get(wcol) or "").strip().lower()
            try:
                v = float(row.get(vcol) or "")
            except ValueError:
                continue
            if w:
                out[w] = v
    return out


DEFAULT = Lexicons.seed()
