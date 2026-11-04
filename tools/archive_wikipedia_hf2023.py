"""Archive locale HF 2023 : change uniquement source/ID, vérifie chaque document avant remplacement."""
import hashlib
from itertools import zip_longest
from pathlib import Path

from pipeline.collect.common import read_json, write_json
from pipeline.io import read_docs, write_docs

OLD, NEW = "wikipedia_fr", "wikipedia_fr_hf2023"


def relabel(doc):
    if doc["source"] not in (OLD, NEW):
        raise ValueError(f"Source inattendue : {doc['source']}")
    prefix = doc["source"] + ":"
    if not doc["id"].startswith(prefix):
        raise ValueError(f"ID inattendu : {doc['id']}")
    return {**doc, "source": NEW, "id": NEW + ":" + doc["id"][len(prefix):]}


def main():
    raw = Path("data/raw")
    if not raw.is_symlink() or not raw.resolve().name.endswith(".nosync"):
        raise RuntimeError("data/raw doit rester un lien vers *.nosync")
    old, dest = raw / OLD, raw / NEW
    if old.exists() and dest.exists():
        raise RuntimeError("Les deux versions existent déjà : aucune migration implicite")
    if old.exists():
        old.rename(dest)
    if not dest.is_dir():
        raise RuntimeError("Archive absente")
    manifest = read_json(dest / "_migration_hf2023.json", {})
    if not manifest:
        manifest = {"original_stats": read_json(dest / "_stats.json"), "parts": {}}
    for part in sorted(dest.glob("part-*.jsonl.zst")):
        if part.name in manifest["parts"]:
            continue
        tmp = part.with_name(part.name + ".migrate")
        write_docs(tmp, (relabel(d) for d in read_docs(part)))
        digest, n, size = hashlib.sha256(), 0, 0
        for before, after in zip_longest(read_docs(part), read_docs(tmp)):
            if before is None or after is None or relabel(before) != after:
                raise RuntimeError(f"Différence non autorisée dans {part}")
            text = before["text"].encode("utf-8")
            digest.update(len(text).to_bytes(8, "big")); digest.update(text)
            n += 1; size += len(text)
        tmp.replace(part)
        manifest["parts"][part.name] = {"docs": n, "bytes": size, "text_sha256": digest.hexdigest()}
        write_json(dest / "_migration_hf2023.json", manifest)
        print(f"{part.name}: {n} documents, textes et métadonnées vérifiés", flush=True)
    original = manifest["original_stats"]
    total = {k: sum(v[k] for v in manifest["parts"].values()) for k in ("docs", "bytes")}
    if any(total[k] != original[k] for k in total):
        raise RuntimeError(f"Totaux différents : {total}")
    write_json(dest / "_stats.json", {**original, "source": NEW})
    manifest["complete"] = True
    write_json(dest / "_migration_hf2023.json", manifest)
    print(f"ARCHIVE VALIDÉE : {total}", flush=True)


if __name__ == "__main__":
    main()
