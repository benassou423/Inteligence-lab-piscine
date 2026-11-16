"""Recompte, pour chaque dossier de jugement publié, les notes par modèle à partir de notes_finales.json.

    python experiments/chatbot_rtx/jugements/recompter_jugements.py [DOSSIER]

Par modèle : nombre de réponses, F, R, S (votes à la majorité) et T = réponses où F, R et S valent 1.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/chatbot_rtx/jugements")
for nf in sorted(base.glob("*/notes_finales.json")):
    d = json.loads(nf.read_text(encoding="utf-8"))
    tot = defaultdict(lambda: defaultdict(int))
    for ident, n in d["notes"].items():
        m = d["cle"][ident]["modele"]
        tot[m]["n"] += 1
        for c in "FRS":
            tot[m][c] += n[c]
        tot[m]["T"] += int(n["F"] and n["R"] and n["S"])
    print(nf.parent.name)
    for m, t in tot.items():
        print(f"   {m:10s} n={t['n']:3d}  F={t['F']:3d}  R={t['R']:3d}  S={t['S']:3d}  T={t['T']:3d}")
