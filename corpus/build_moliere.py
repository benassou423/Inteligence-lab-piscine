"""Reconstruit le corpus Molière du kit depuis le Projet Gutenberg.

    python corpus/build_moliere.py

Source : « Œuvres complètes de J.-B. Poquelin Molière », édition Philarète Chasles (Calmann Lévy,
1888), tomes 1 à 4, numérisés par le Projet Gutenberg (ebooks 40086, 43535, 50173, 57270).
Domaine public. Les notices et notes de l'éditeur sont retirées : seules les pièces restent.

Sorties :
  corpus/moliere_train.txt   toutes les pièces sauf trois
  corpus/moliere_val.txt     trois pièces tenues à l'écart (vers, prose, comédie-ballet)
  corpus/moliere_manifest.json
Les pièces sont séparées par une ligne vide puis une ligne « ===== ».
"""

import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOMES = [40086, 43535, 50173, 57270]
VAL_PLAYS = {"LE MISANTHROPE", "L'AVARE", "LE SICILIEN"}
SEP = "\n\n=====\n\n"


def download(n):
    cache = HERE / "cache" / f"pg{n}.txt"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        req = urllib.request.Request(f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt",
                                     headers={"User-Agent": "FeedTheBeastKit/1.0"})
        cache.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return cache.read_text(encoding="utf-8")


def play_title(fin_line):
    t = fin_line.strip().rstrip(".").strip()
    t = re.sub(r"^FIN (DU |DE LA |DE L'|DES |DE |D')", lambda m: {
        "FIN DU ": "LE ", "FIN DE LA ": "LA ", "FIN DE L'": "L'", "FIN DES ": "LES ",
        "FIN DE ": "", "FIN D'": ""}[m.group(0)], t)
    return t


def clean_play(text):
    lines = text.replace("\r\n", "\n").split("\n")
    out, in_note = [], False
    for ln in lines:
        s = ln.strip()
        if re.match(r"^\[\d+\]", s):  # début de note de bas de page
            in_note = True
            continue
        if in_note:
            if not s:
                in_note = False
            continue
        s = re.sub(r"\[\d+\]", "", s)  # appels de note
        s = s.replace("_", "")
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return unicodedata.normalize("NFC", text).strip()


def split_plays(raw):
    s = raw.index("\n", raw.index("*** START OF")) + 1
    e = raw.index("*** END OF")
    body = raw[s:e].replace("\r\n", "\n")
    # Table des matières et corrections du transcripteur, en fin de volume
    body = body[: [m.start() for m in re.finditer(r"\n[ \t]*TABLE[ \t]*\n", body)][-1]]
    lines = body.split("\n")
    plays, start = [], 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*FIN D", ln) and "TABLE" not in ln:
            seg = lines[start:i]
            title = play_title(ln)
            # La notice de l'éditeur précède la pièce : on commence à la liste des personnages
            k = next((j for j, x in enumerate(seg) if x.strip().startswith(("PERSONNAGES", "ACTEURS"))), None)
            if k is None:
                start = i + 1
                continue  # segment sans pièce (livret de ballet seul) : ignoré
            text = clean_play("\n".join(seg[k:]))
            plays.append((title, f"{title}\n\n{text}"))
            start = i + 1
    return plays


def main():
    train, val, manifest = [], [], []
    for n in TOMES:
        for title, text in split_plays(download(n)):
            dest = "val" if title in VAL_PLAYS else "train"
            (val if dest == "val" else train).append(text)
            manifest.append({"titre": title, "ebook_gutenberg": n, "split": dest,
                             "caracteres": len(text)})
    for name, docs in (("moliere_train.txt", train), ("moliere_val.txt", val)):
        (HERE / name).write_text(SEP.join(docs) + "\n", encoding="utf-8")
    info = {
        "source": "Projet Gutenberg, Œuvres complètes de Molière (éd. Chasles, 1888), tomes 1 à 4",
        "ebooks": [f"https://www.gutenberg.org/ebooks/{n}" for n in TOMES],
        "licence": "Domaine public",
        "traitement": "notices, notes et corrections du transcripteur retirées ; une pièce par bloc",
        "pieces": manifest,
        "sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("moliere_train.txt", "moliere_val.txt")},
    }
    (HERE / "moliere_manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    n_tr, n_va = sum(map(len, train)), sum(map(len, val))
    print(f"{len(train)} pièces en train ({n_tr:,} car.), {len(val)} en validation ({n_va:,} car.)")


if __name__ == "__main__":
    main()
