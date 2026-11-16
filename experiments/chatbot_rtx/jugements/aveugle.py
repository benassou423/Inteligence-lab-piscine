"""Notation à l'aveugle de réponses de chatbots (jeu « protocole » de tools/nuit_moliere.py).

prepare : lit les rows.jsonl de plusieurs modèles, garde les graines demandées, mélange (graine fixe)
          et écrit items.json (ce que voient les juges : identifiant, échanges précédents, message,
          réponse) et cle.json (identifiant -> modèle, jamais montré aux juges).
agrege  : lit les notes des juges {id: {F, R, S}} (plusieurs juges), vote à la majorité, dévoile.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare(a):
    items, cle = [], {}
    for spec in a.modeles:
        nom, chemin = spec.split("=", 1)
        rows = [json.loads(l) for l in Path(chemin).read_text(encoding="utf-8").splitlines() if l.strip()]
        convs = defaultdict(list)
        for r in rows:
            convs[(r["job"], r["base_seed"])].append(r)
        for (job, graine), conv in convs.items():
            if graine not in a.graines:
                continue
            conv.sort(key=lambda r: r["turn"])
            for i, r in enumerate(conv):
                items.append({"modele": nom, "job": job, "graine": graine, "tour": r["turn"],
                              "avant": [{"message": p["message"], "reponse": p["reply"]} for p in conv[:i]],
                              "message": r["message"], "reponse": r["reply"]})
    random.Random(a.melange).shuffle(items)
    vus = []
    for n, it in enumerate(items, 1):
        ident = f"r{n:03d}"
        cle[ident] = {k: it[k] for k in ("modele", "job", "graine", "tour")}
        vus.append({"id": ident, "avant": it["avant"], "message": it["message"], "reponse": it["reponse"]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(vus, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "cle.json").write_text(json.dumps(cle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(vus)} réponses mélangées ({', '.join(s.split('=')[0] for s in a.modeles)}) -> {out}")


def agrege(a):
    out = Path(a.out)
    cle = json.loads((out / "cle.json").read_text(encoding="utf-8"))
    juges = json.loads(Path(a.notes).read_text(encoding="utf-8"))  # liste de {id: {F,R,S}} par juge
    votes = defaultdict(lambda: defaultdict(list))
    for j in juges:
        for ident, n in j.items():
            for c in "FRS":
                if n.get(c) in (0, 1):
                    votes[ident][c].append(n[c])
    final, manquants = {}, []
    for ident in cle:
        v = votes.get(ident)
        if not v or any(len(v[c]) == 0 for c in "FRS"):
            manquants.append(ident)
            continue
        final[ident] = {c: int(sum(v[c]) * 2 > len(v[c])) for c in "FRS"}
        final[ident]["juges"] = {c: v[c] for c in "FRS"}
    tot = defaultdict(lambda: defaultdict(int))
    par_job = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    accord = defaultdict(lambda: [0, 0])
    for ident, n in final.items():
        m = cle[ident]["modele"]
        tot[m]["n"] += 1
        for c in "FRS":
            tot[m][c] += n[c]
            accord[c][0] += len(set(n["juges"][c])) == 1
            accord[c][1] += 1
        par_job[m][cle[ident]["job"]]["n"] += 1
        par_job[m][cle[ident]["job"]]["R"] += n["R"]
    res = {"modeles": {m: dict(d) for m, d in tot.items()},
           "R_par_message": {m: {str(j): f"{d['R']}/{d['n']}" for j, d in sorted(jobs.items())}
                             for m, jobs in par_job.items()},
           "accord_unanime": {c: f"{a_}/{b_}" for c, (a_, b_) in accord.items()},
           "juges": len(juges), "manquants": manquants}
    (out / "notes_finales.json").write_text(json.dumps({"resume": res, "notes": final, "cle": cle},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--modeles", nargs="+", required=True, metavar="NOM=rows.jsonl")
    a.add_argument("--graines", nargs="+", type=int, default=[11, 22, 33])
    a.add_argument("--melange", type=int, default=20261113)
    a.add_argument("--out", required=True)
    b = sub.add_parser("agrege")
    b.add_argument("--notes", required=True)
    b.add_argument("--out", required=True)
    x = p.parse_args()
    {"prepare": prepare, "agrege": agrege}[x.cmd](x)


if __name__ == "__main__":
    main()
