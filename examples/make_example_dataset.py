"""Exemple minimal de dataset (baseline) : à battre, pas à copier.

    pip install datasets
    python examples/make_example_dataset.py --out example_data/ --mo 450

Mélange naïf, sans nettoyage, sans déduplication, sans filtrage qualité :
  70 % FineWeb-2 (config fra_Latn, web français déjà filtré par Hugging Face, ODC-By)
  25 % Wikipédia FR (dump Hugging Face wikimedia/wikipedia 20231101.fr, CC BY-SA 4.0)
   5 % Molière (corpus du kit, domaine public)
Les documents sont simplement pris dans l'ordre du flux, puis mélangés.

Ton travail commence là : du Common Crawl brut, tes propres filtres, tes propres proportions,
des preuves (ablations) que chaque étape aide. Ce script sert seulement à vérifier que toute
la chaîne tourne (prepare -> train -> evaluate -> submit) et à fixer un point de comparaison.
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def take(stream, field, n_bytes):
    got = 0
    for row in stream:
        text = row[field]
        if not text:
            continue
        yield text
        got += len(text.encode("utf-8"))
        if got >= n_bytes:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="example_data")
    p.add_argument("--mo", type=float, default=450, help="taille totale visée en Mo de texte")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset

    total = int(args.mo * 1e6)
    docs = []
    print("FineWeb-2 fra_Latn...")
    fw = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    docs += [("fineweb2", t) for t in take(fw, "text", int(total * 0.70))]
    print("Wikipédia FR...")
    wp = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    docs += [("wikipedia", t) for t in take(wp, "text", int(total * 0.25))]
    plays = (HERE / "corpus" / "moliere_train.txt").read_text(encoding="utf-8").split("\n\n=====\n\n")
    reps = max(1, int(total * 0.05 / sum(len(x.encode()) for x in plays)))
    docs += [("moliere", t) for t in plays for _ in range(reps)]

    random.Random(args.seed).shuffle(docs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "example.jsonl", "w", encoding="utf-8") as f:
        for src, text in docs:
            f.write(json.dumps({"source": src, "text": text}, ensure_ascii=False) + "\n")
    by_src = {}
    for src, text in docs:
        by_src[src] = by_src.get(src, 0) + len(text.encode("utf-8"))
    print(json.dumps({"documents": len(docs), "octets_par_source": by_src}, indent=2))


if __name__ == "__main__":
    main()
