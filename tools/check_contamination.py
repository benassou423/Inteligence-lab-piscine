"""Vérifie une sortie du pipeline avec le contrôle officiel du kit (ftb.contamination).

    uv run python -m tools.check_contamination --config nettoye [--stage 07_exact_dedup] [--max-docs N]
    uv run python -m tools.check_contamination --config nettoye --fast     # tout le corpus, en parallèle

Mode normal : le kit lit .jsonl ou .jsonl.gz, pas .jsonl.zst. On exporte l'étape dans un
dossier temporaire (supprimé à la fin), puis on lance `python -m ftb.contamination`.
Le kit hache tout en un seul processus : sur tout le corpus, utiliser --max-docs.

Mode --fast : même mesure (13-grammes hachés par ftb.contamination.ngram_hashes, union sur
tout le corpus, document de val signalé au-delà de 10 %), calculée par 8 workers.
Attendu après l'étape decontaminate : 0 document signalé.
"""
import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

from pipeline.io import read_docs

ROOT = Path(__file__).resolve().parents[1]
_INDEX = None


def _shared_in_file(path):
    hits = set()
    for doc in read_docs(path):
        hits |= _INDEX.shared(doc.get("text") or "")
    return hits


def fast_check(stage_dir, workers=8, flag=None):
    """Union des 13-grammes de val présents dans l'étape ; même sortie que le kit (sans détail)."""
    global _INDEX
    from ftb import VALIDATION_PATH
    from ftb.contamination import FLAG

    from pipeline.dedup.decontaminate import ValIndex, load_reference

    flag = FLAG if flag is None else flag
    _INDEX = ValIndex(load_reference([VALIDATION_PATH]))
    files = sorted(Path(stage_dir).glob("*/part-*.jsonl.zst"))
    union = set()
    with get_context("fork").Pool(workers) as pool:
        for hits in pool.imap_unordered(_shared_in_file, files):
            union |= hits
    rows = []
    for d, size in zip(_INDEX.docs, _INDEX.sizes):
        hs = set(_INDEX.ngram_hashes(d["text"]))
        frac = len(hs & union) / size if size else 0.0
        rows.append({"id": d.get("id"), "genre": d.get("genre"), "title": d.get("title"),
                     "recouvrement": round(frac, 4)})
    flagged = [r for r in rows if r["recouvrement"] > flag]
    return {"n_fichiers": len(files), "n_documents_evalues": len(rows), "n_signales": len(flagged),
            "recouvrement_moyen": round(sum(r["recouvrement"] for r in rows) / max(1, len(rows)), 5),
            "recouvrement_max": max(r["recouvrement"] for r in rows),
            "signales": sorted(flagged, key=lambda r: -r["recouvrement"]),
            "plus_hauts": sorted(rows, key=lambda r: -r["recouvrement"])[:5]}


def export(stage_dir, out_dir, max_docs=None):
    """Une copie .jsonl.gz par source (champ text seulement). Renvoie la liste des fichiers."""
    files = []
    for src_dir in sorted(p for p in Path(stage_dir).iterdir() if p.is_dir() and not p.name.startswith("_")):
        dest = Path(out_dir) / f"{src_dir.name}.jsonl.gz"
        n = 0
        with gzip.open(dest, "wt", encoding="utf-8", compresslevel=1) as f:
            for part in sorted(src_dir.glob("part-*.jsonl.zst")):
                for doc in read_docs(part):
                    f.write(json.dumps({"text": doc["text"]}, ensure_ascii=False) + "\n")
                    n += 1
                    if max_docs and n >= max_docs:
                        break
                if max_docs and n >= max_docs:
                    break
        print(f"{src_dir.name} : {n} documents exportés", file=sys.stderr)
        files.append(str(dest))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="nettoye")
    ap.add_argument("--stage", help="dossier d'étape (défaut : dernière étape du manifest)")
    ap.add_argument("--max-docs", type=int, help="documents max par source (échantillon)")
    ap.add_argument("--out", help="rapport JSON complet du kit")
    ap.add_argument("--fast", action="store_true", help="tout le corpus, en parallèle (même mesure que le kit)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    base = ROOT / "data" / "stages" / args.config
    if args.stage:
        stage_dir = base / args.stage
    else:
        stage_dir = ROOT / json.loads((base / "manifest.json").read_text(encoding="utf-8"))["final_dir"]
    if args.fast:
        res = fast_check(stage_dir, args.workers)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if res["n_signales"] == 0 else 1
    with tempfile.TemporaryDirectory(prefix="ftb_contam_") as tmp:
        files = export(stage_dir, tmp, args.max_docs)
        cmd = [sys.executable, "-m", "ftb.contamination", "--inputs", *files]
        if args.out:
            cmd += ["--out", args.out]
        return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
