"""Comparaison appariée de deux modèles dans un jugement à l'aveugle (même message, même graine, même tour).

    python experiments/chatbot_rtx/jugements/apparie.py DOSSIER_AVEUGLE TEMOIN [CRITERE]

CRITERE : T (sans défaut : F, R et S valent 1 ; par défaut), F, R ou S. Pour chaque autre modèle : nombre de
réponses où il est meilleur que le témoin, moins bon, et la probabilité (test des signes, bilatéral) d'un
écart au moins aussi grand si les deux modèles se valaient.
"""
import json
import sys
from math import comb
from pathlib import Path

d = json.loads((Path(sys.argv[1]) / "notes_finales.json").read_text(encoding="utf-8"))
temoin = sys.argv[2]
crit = sys.argv[3] if len(sys.argv) > 3 else "T"
note = (lambda n: int(n["F"] and n["R"] and n["S"])) if crit == "T" else (lambda n: n[crit])
par = {}
for ident, n in d["notes"].items():
    c = d["cle"][ident]
    par.setdefault(c["modele"], {})[(c["job"], c["graine"], c["tour"])] = note(n)
for m in sorted(par):
    if m == temoin:
        continue
    cles = sorted(set(par[m]) & set(par[temoin]))
    mieux = sum(par[m][k] > par[temoin][k] for k in cles)
    moins = sum(par[m][k] < par[temoin][k] for k in cles)
    n = mieux + moins
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(mieux, moins) + 1)) / 2 ** n) if n else 1.0
    print(f"{m:6s} contre {temoin} ({crit}) : {len(cles)} paires, meilleur {mieux}, moins bon {moins}, "
          f"écart {mieux - moins:+d}, p = {p:.4f}")
