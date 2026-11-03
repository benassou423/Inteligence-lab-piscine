"""Lecture et écriture des documents : JSONL compressé zstd, un document par ligne."""
import json
from pathlib import Path

import zstandard


def read_docs(path):
    """Renvoie les documents d'un fichier .jsonl.zst, un par un."""
    with zstandard.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_docs(path, docs):
    """Écrit les documents dans un fichier .jsonl.zst et renvoie leur nombre."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zstandard.open(path, "wt", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n
