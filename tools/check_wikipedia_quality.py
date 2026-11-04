"""Mesure l'essai Wikipédia et prépare 20 lectures ; aucune approbation visuelle automatique."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from pipeline.collect.common import read_json, write_json
from pipeline.io import read_docs

HOLE = re.compile(r" (le|du|de|en|à|par|était de|s'élevait à) [.,]")
FIRST_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lost_apostrophe(text, raw):
    """Perte confirmée par le wikitexte : pas de faux positif sur Le, Dans, Des, etc."""
    match = FIRST_WORD.search(text)
    if not match:
        return False
    first = match[0]
    if len(first) < 3 or first[0] not in "LDld" or "'" in first or "’" in first:
        return False
    # Une apostrophe ou l'ouverture du gras suit L/D dans la source, puis le même mot.
    pattern = (r"(?<!\w)" + re.escape(first[0]) + r"(?:['’]+|\{\{\s*['’]\s*\}\})"
               + r"'*(?:\[\[)?" + re.escape(first[1:]) + r"\b")
    return bool(re.search(pattern, raw))


def select_manual(docs):
    """20 IDs déterministes, répartis sur les parties, sans sélectionner les textes faciles."""
    groups = defaultdict(list)
    for doc in docs:
        groups[doc["meta"]["dump_part"]].append(doc)
    key = lambda d: sha(d["id"])
    selected = [min(group, key=key) for _, group in sorted(groups.items())]
    selected_ids = {d["id"] for d in selected}
    selected += sorted((d for d in docs if d["id"] not in selected_ids), key=key)[:20 - len(selected)]
    return selected[:20]


def measure(source, output, expected=2000):
    source, output = Path(source), Path(output)
    state = read_json(source / "_state.json", {})
    if not state.get("complete") or state.get("mode") != "sample":
        raise ValueError("Il faut un échantillon terminé, pas une collecte en cours")
    docs = [d for p in sorted(source.glob("part-*.jsonl.zst")) for d in read_docs(p)]
    raw = {"wikipedia_fr:" + d["pid"]: d["wikitext"]
           for p in sorted((source / "_sample_input").glob("*.jsonl.zst")) for d in read_docs(p)}
    if len(docs) != expected or len({d["id"] for d in docs}) != expected:
        raise ValueError(f"Attendu {expected} articles distincts, reçu {len(docs)}")
    if any(d["id"] not in raw for d in docs):
        raise ValueError("Wikitexte original manquant")
    holes, glued, counts, examples = [], [], Counter(), []
    for doc in docs:
        text = doc["text"]
        matches = list(HOLE.finditer(text))
        if matches:
            holes.append(doc["id"])
            examples.append({"id": doc["id"], "title": doc["meta"]["title"], "matches": [
                text[max(0, m.start()-80):m.end()+80] for m in matches]})
        if lost_apostrophe(text, raw[doc["id"]]):
            glued.append(doc["id"])
        counts.update(doc["meta"].get("unknown_templates", {}))
    selected = select_manual(docs)
    review = read_json(output / "manual_review.json", {})
    validated = [d for d in selected if review.get(d["id"], {}).get("sha256") == sha(d["text"])
                 and review[d["id"]].get("verdict") == "sans_trou_visible"]
    numeric_ok = len(holes)/expected < .05 and len(glued)/expected < .002
    dates_ok = all(d["date"] <= "2026-10-01" and d["meta"].get("last_revision") for d in docs)
    report = {"docs": expected, "dump_parts": len({d["meta"]["dump_part"] for d in docs}),
              "hole_regex": HOLE.pattern, "hole_docs": len(holes), "hole_percent": 100*len(holes)/expected,
              "glued_first_word_docs": len(glued), "glued_first_word_percent": 100*len(glued)/expected,
              "glued_method": "premier mot rendu comparé au wikitexte d'origine (apostrophe ou gras après L/D)",
              "date_min": min(d["date"] for d in docs), "date_max": max(d["date"] for d in docs),
              "unknown_top30": counts.most_common(30), "unknown_types": len(counts),
              "manual_ids": [d["id"] for d in selected], "manual_validated": len(validated),
              "numeric_pass": numeric_ok, "dates_pass": dates_ok,
              "gate_pass": numeric_ok and dates_ok and len(validated) == 20,
              "hole_ids": holes, "glued_ids": glued}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_json(output / "hole_examples.json", examples)
    write_json(output / "manual_selection.json", [{"id": d["id"], "title": d["meta"]["title"],
                "sha256": sha(d["text"]), "chars": len(d["text"])} for d in selected])
    text = "# Essai Wikipédia — lecture intégrale des 20 articles\n\n"
    for i, doc in enumerate(selected, 1):
        text += f"## {i}. {doc['meta']['title']} — {doc['id']}\n\n{doc['text']}\n\n"
    (output / "manual_articles.md").write_text(text, encoding="utf-8")
    unknown = "# Modèles inconnus — essai de 2 000 articles\n\n| Modèle | Occurrences |\n|---|---:|\n"
    unknown += "".join(f"| `{name}` | {n} |\n" for name, n in counts.most_common(30))
    (output / "unknown_top30.md").write_text(unknown, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("manual_ids", "hole_ids", "glued_ids")}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/wikipedia_fr")
    parser.add_argument("--output", default="data/stages/wikipedia_quality_2000")
    args = parser.parse_args()
    measure(args.source, args.output)


if __name__ == "__main__":
    main()
