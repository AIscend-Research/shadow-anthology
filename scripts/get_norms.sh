#!/usr/bin/env bash
# Download the published psycholinguistic norms the metrics need.
#
#   bash scripts/get_norms.sh
#
# Without these, imagery/tone/concreteness/valence report n=0: the seed
# lexicons in the package cover too little real vocabulary to mean anything,
# and the coverage guard correctly refuses to invent numbers from them.
#
#   Brysbaert, Warriner & Kuperman (2014), concreteness for ~40k lemmas
#   Warriner, Kuperman & Brysbaert (2013), valence/arousal for ~14k lemmas
#
# The concreteness release also carries raw SUBTLEX counts, which the loader
# log-transforms for word frequency -- so no separate SUBTLEX download.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/norms

get() {  # url dest
  if [ -s "$2" ]; then echo "  have $2"; return; fi
  echo "  fetching $(basename "$2") ..."
  curl -sL --max-time 180 -o "$2" "$1"
}

get "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt" \
    data/norms/concreteness.txt
get "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv" \
    data/norms/warriner_vad.csv

python - <<'PYEOF'
from shadow_anthology.lexicons import Lexicons
lex = Lexicons.load(concreteness="data/norms/concreteness.txt",
                    vad="data/norms/warriner_vad.csv")
assert not lex.is_seed, "norms failed to load"
print(f"  concreteness {len(lex.concreteness):>6}")
print(f"  valence      {len(lex.valence):>6}")
print(f"  arousal      {len(lex.arousal):>6}")
print(f"  frequency    {len(lex.frequency):>6}")
print("norms OK")
PYEOF
