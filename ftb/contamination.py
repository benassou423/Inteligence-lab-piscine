"""Mesure le recouvrement entre tes données d'entraînement et un jeu d'évaluation.

    python -m ftb.contamination --inputs mes_donnees/*.jsonl
    python -m ftb.contamination --inputs ... --against data/validation.jsonl --out rapport.json

Pour chaque document du jeu d'évaluation, on compte la part de ses 13-grammes de mots (en
minuscules) présents quelque part dans tes données. Un document est signalé au-delà de 10 %.

Entraîner sur la validation n'est pas interdit en soi par le classement (elle est publique), mais
ton BPB de validation devient alors faux, et tu perds ton seul repère fiable. Sur le test caché,
l'orga lance le même contrôle sur le dataset des meilleurs : un recouvrement anormal disqualifie.
"""

import argparse
import json
import re
import sys
from hashlib import blake2b

from ftb import VALIDATION_PATH
from ftb.data import iter_documents

N = 13
FLAG = 0.10
WORD = re.compile(r"\w+", re.UNICODE)


def ngram_hashes(text, n=N):
    words = WORD.findall(text.lower())
    out = []
    for i in range(len(words) - n + 1):
        out.append(blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest())
    return out


def contamination(inputs, against, n=N):
    with open(against, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    owner = {}
    totals = []
    for i, d in enumerate(docs):
        hs = set(ngram_hashes(d["text"], n))
        totals.append(len(hs))
        for h in hs:
            owner.setdefault(h, []).append(i)
    hits = [set() for _ in docs]
    for k, text in enumerate(iter_documents(inputs)):
        for h in ngram_hashes(text, n):
            if h in owner:
                for i in owner[h]:
                    hits[i].add(h)
        if k % 10000 == 0:
            print(f"\r{k:,} documents lus", end="", file=sys.stderr)
    print(file=sys.stderr)
    rows = []
    for d, tot, hit in zip(docs, totals, hits):
        frac = len(hit) / tot if tot else 0.0
        rows.append({"id": d.get("id"), "genre": d.get("genre"), "title": d.get("title"),
                     "recouvrement": round(frac, 4)})
    flagged = [r for r in rows if r["recouvrement"] > FLAG]
    return {"n_documents_evalues": len(rows), "n_signales": len(flagged),
            "recouvrement_moyen": round(sum(r["recouvrement"] for r in rows) / max(1, len(rows)), 5),
            "signales": sorted(flagged, key=lambda r: -r["recouvrement"]), "detail": rows}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.contamination", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--against", default=str(VALIDATION_PATH))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    res = contamination(args.inputs, args.against)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
