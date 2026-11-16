"""Union de deux fichiers de synthèse déjà fusionnés et refiltrés (campagnes v4 et v5, consignes différentes).

Le premier fichier est gardé tel quel. Une conversation du second est écartée si, par rapport à TOUT ce qui est
déjà gardé (premier fichier + conversations du second déjà acceptées), elle est un doublon exact, répète une
réplique longue, ou contient une formule de 5 mots déjà trop répandue (mêmes seuils que chatbot.synthetic.Dedup,
lus dans les consignes du second fichier). Le plafond par premier mot n'est PAS rejoué ici : il a été appliqué
dans chaque campagne, et le rejouer sur l'union dépendrait de l'ordre de lecture.

Usage (racine du clone) :
    python experiments/chatbot_rtx/outils/unir_synthese.py PREMIER.jsonl.zst SECOND.jsonl.zst CONSIGNES_DU_SECOND.json DST.jsonl.zst
"""
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path.cwd()))
from chatbot import synthetic as s  # noqa: E402

STYLE = "moliere"


def lire(chemin):
    with open(chemin, "rb") as f:
        for ligne in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            if ligne.strip():
                yield json.loads(ligne)


def main():
    premier, second, consignes, dst = sys.argv[1:5]
    dst = Path(dst)
    if dst.exists():
        sys.exit(f"REFUS : {dst} existe déjà")
    cfg = s.load_config(consignes)
    dedup = s.Dedup.depuis(cfg, STYLE)
    dedup.ouverture_max = None  # voir l'en-tête : pas de plafond par premier mot sur l'union
    gardes, ecartes, origine = [], collections.Counter(), collections.Counter()
    for d in lire(premier):
        dedup.ajoute([t["text"] for t in d["turns"]])
        gardes.append(d)
        origine["premier"] += 1
    lues = 0
    for d in lire(second):
        lues += 1
        textes = [t["text"] for t in d["turns"]]
        raison = dedup.raison(textes)
        if raison:
            ecartes[raison] += 1
            continue
        dedup.ajoute(textes)
        gardes.append(d)
        origine["second"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f, zstandard.ZstdCompressor(level=10).stream_writer(f) as z:
        for d in gardes:
            z.write((json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8"))
    bilan = {"premier": Path(premier).as_posix(), "second": Path(second).as_posix(),
             "consignes_des_seuils": Path(consignes).as_posix(),
             "gardees_du_premier": origine["premier"], "lues_du_second": lues,
             "gardees_du_second": origine["second"], "ecartees_du_second": dict(ecartes),
             "total": len(gardes), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    Path(str(dst).replace(".jsonl.zst", ".union.json")).write_text(
        json.dumps(bilan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(bilan, ensure_ascii=False))


main()
