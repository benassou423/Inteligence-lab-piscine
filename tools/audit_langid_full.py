"""Audit en lecture seule : effet exact des règles, textes complets et deux détecteurs.

Les sorties restent dans data/stages ; elles ne remplacent jamais le corpus actif.
Les détecteurs servent au diagnostic, pas à fabriquer une vérité terrain humaine.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
from types import SimpleNamespace

import zstandard

from pipeline.collect.common import read_json, write_json
from pipeline.io import read_docs, write_docs

ROOT = Path(__file__).resolve().parents[1]
MISSING_SOURCES = {"wikisource_fr", "gutenberg_fr", "assemblee_nationale"}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
MARKUP = re.compile(r"\{\{|\}\}|\{\||\|\}|\balign\s*=|\browspan\s*=|\bcolspan\s*=")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(path):
    s = Path(path).stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}


def decisions(doc, threshold=.65):
    meta = doc["meta"]
    fr, family = float(meta["langid_fr"]), float(meta["langid_family"])
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (fr, family)) or family + 1e-6 < fr:
        raise ValueError(f"Scores invalides : {doc['id']}")
    return fr >= threshold, family >= threshold


def windows(text, size=3000):
    """Tranches disjointes, sans trou ni troncature ; séparation sur une espace si possible."""
    if size < 20:
        raise ValueError("Une fenêtre doit contenir au moins 20 caractères")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            split = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if split >= start + size // 2:
                end = split + 1
        yield start, end, text[start:end]
        start = end


def quality_features(text):
    """Signaux descriptifs : aucun n'est appliqué comme filtre de qualité dans cet audit."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = [w.casefold() for w in WORD.findall(text)]
    counts = Counter(lines)
    return {"nonempty_lines": len(lines), "markup_lines": sum(bool(MARKUP.search(x)) for x in lines),
            "numeric_lines": sum(not WORD.search(x) for x in lines),
            "unique_line_fraction": len(counts) / max(1, len(lines)),
            "word_count": len(words), "unique_word_fraction": len(set(words)) / max(1, len(words)),
            "max_repeated_word_run": max((sum(1 for _ in g) for _, g in itertools.groupby(words)), default=0),
            "sentence_punctuation": sum(text.count(c) for c in ".!?"),
            "letter_fraction": sum(c.isalpha() for c in text) / max(1, len(text))}


def scan_unit(args):
    path, output, targets, threshold, code = args
    path, output = Path(path), Path(output)
    source = path.parent.name
    unit = output / "scan" / source / path.name
    receipt = unit.with_suffix(".receipt.json")
    original = unit.with_suffix(".original.zst")
    candidate = unit.with_suffix(".candidate.zst")
    signature = {"input": file_signature(path), "code": code, "threshold": threshold,
                 "targets": sha("\n".join(sorted(targets)))}
    old = read_json(receipt, {})
    if old.get("signature") == signature and original.exists() and candidate.exists():
        return old
    unit.parent.mkdir(parents=True, exist_ok=True)
    originals, candidates = original.with_suffix(".tmp"), candidate.with_suffix(".tmp")
    counts = Counter()
    controls, largest = [], []
    with ExitStack() as stack:
        fo = stack.enter_context(zstandard.open(originals, "wt", encoding="utf-8"))
        fc = stack.enter_context(zstandard.open(candidates, "wt", encoding="utf-8"))
        for line, doc in enumerate(read_docs(path)):
            if doc["source"] != source:
                raise ValueError(f"Source incohérente : {path}, {doc['id']}")
            current, wider = decisions(doc, threshold)
            size = len(doc["text"].encode("utf-8"))
            counts["docs"] += 1
            counts["bytes"] += size
            for name, keep in (("current", current), ("candidate", wider), ("added", wider and not current)):
                if keep:
                    counts[name + "_docs"] += 1
                    counts[name + "_bytes"] += size
            record = {**doc, "audit_location": {"file": str(path), "row": line}}
            if doc["id"] in targets:
                fo.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["original_docs"] += 1
            if wider and not current:
                fc.write(json.dumps(record, ensure_ascii=False) + "\n")
            if source in MISSING_SOURCES and current:
                ref = {"id": doc["id"], "file": str(path), "row": line, "source": source,
                       "bytes": size, "rank": sha("control-v1|" + doc["id"])}
                controls = sorted(controls + [ref], key=lambda r: r["rank"])[:4]
                largest = sorted(largest + [ref], key=lambda r: (-r["bytes"], r["id"]))[:1]
    if file_signature(path) != signature["input"]:
        raise RuntimeError(f"Entrée modifiée pendant la lecture : {path}")
    originals.replace(original)
    candidates.replace(candidate)
    result = {"signature": signature, "source": source, "input": str(path), "counts": dict(counts),
              "original_path": str(original), "candidate_path": str(candidate),
              "controls": controls, "largest": largest}
    write_json(receipt, result)
    return result


def extract(input_dir, sample_path, output, workers=8, threshold=.65):
    input_dir, output = Path(input_dir), Path(output)
    samples = {d["id"]: d for d in map(json.loads, Path(sample_path).read_text().splitlines())}
    files = sorted(input_dir.glob("*/part-*.jsonl.zst"))
    if not files or len(samples) != 300:
        raise ValueError("Entrée absente ou échantillon différent de 300 IDs")
    code = sha(Path(__file__).read_text())
    tasks = [(str(p), str(output), set(samples), threshold, code) for p in files]
    receipts = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(scan_unit, tasks), 1):
            receipts.append(result)
            if i % 10 == 0 or i == len(files):
                print(f"Lecture du corpus : {i}/{len(files)} fichiers", flush=True)
    receipts.sort(key=lambda r: r["input"])
    sources = defaultdict(Counter)
    for r in receipts:
        sources[r["source"]].update(r["counts"])
    total = sum(sources.values(), Counter())
    expected = read_json(input_dir / "stats.json")["total"]
    if total["docs"] != expected["docs_out"] or total["bytes"] != expected["bytes_out"]:
        raise RuntimeError("Le recensement ne correspond pas aux statistiques de l'étape")

    chosen = {}
    def add(doc, role):
        if doc["id"] not in chosen:
            chosen[doc["id"]] = {**doc, "audit_roles": []}
        if role not in chosen[doc["id"]]["audit_roles"]:
            chosen[doc["id"]]["audit_roles"].append(role)

    found = set()
    for r in receipts:
        for doc in read_docs(r["original_path"]):
            old = samples[doc["id"]]
            if doc["id"] in found or not doc["text"].startswith(old["text"]) or len(doc["text"].encode()) != old["text_bytes"]:
                raise RuntimeError(f"Échantillon initial incohérent : {doc['id']}")
            found.add(doc["id"])
            add(doc, "original_300")
    if found != set(samples):
        raise RuntimeError("Des documents de l'échantillon initial sont absents")
    wiki = defaultdict(list)
    for r in receipts:
        for doc in read_docs(r["candidate_path"]):
            if doc["source"] != "wikipedia_fr":
                add(doc, "all_added_non_wikipedia")
            elif doc["id"] not in samples:
                n = len(doc["text"])
                bucket = "short" if n <= 1500 else "medium" if n <= 6000 else "long"
                wiki[bucket].append(doc)
                wiki[bucket].sort(key=lambda d: sha("new-wiki-v1|" + d["id"]))
                del wiki[bucket][20:]
    for bucket, docs in wiki.items():
        for doc in docs:
            add(doc, "new_wikipedia_" + bucket)
    refs = defaultdict(list)
    for source in sorted(MISSING_SOURCES):
        randoms = [d for r in receipts if r["source"] == source for d in r["controls"]]
        largest = [d for r in receipts if r["source"] == source for d in r["largest"]]
        selected = sorted(randoms, key=lambda d: d["rank"])[:4] + sorted(largest, key=lambda d: (-d["bytes"], d["id"]))[:1]
        for ref in {d["id"]: d for d in selected}.values():
            refs[ref["file"]].append(ref)
    for path, entries in sorted(refs.items()):
        wanted = {r["id"] for r in entries}
        for line, doc in enumerate(read_docs(path)):
            if doc["id"] in wanted:
                add({**doc, "audit_location": {"file": path, "row": line}}, "control_missing_source")
                wanted.remove(doc["id"])
                if not wanted:
                    break
        if wanted:
            raise RuntimeError("Contrôle introuvable")
    selected = [chosen[i] for i in sorted(chosen)]
    write_docs(output / "selected_full.jsonl.zst", selected)
    manifest = {"input": str(input_dir), "sample_path": str(sample_path), "threshold": threshold,
                "by_source": {k: dict(v) for k, v in sources.items()}, "total": dict(total),
                "original_found": len(found), "selected_docs": len(selected),
                "selected_bytes": sum(len(d["text"].encode()) for d in selected),
                "selection_roles": dict(Counter(role for d in selected for role in d["audit_roles"])),
                "input_signatures": {r["input"]: r["signature"]["input"] for r in receipts},
                "code_sha256": code}
    write_json(output / "extraction.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("input_signatures", "by_source")}, ensure_ascii=False), flush=True)
    return manifest


def analyse_doc(doc, glot, independent, window_chars=3000):
    text = doc["text"]
    full_glot = glot.score(text)
    full_second = independent.score(text)
    ranges = []
    for start, end, part in windows(text, window_chars):
        ranges.append({"start": start, "end": end, "sha256": sha(part),
                       "glot": glot.score(part), "lid176": independent.score(part)})
    assert sum(r["end"]-r["start"] for r in ranges) == len(text)
    if ranges:
        assert ranges[0]["start"] == 0 and ranges[-1]["end"] == len(text)
        assert all(a["end"] == b["start"] for a, b in zip(ranges, ranges[1:]))
    cached = {k: v for k, v in doc.get("meta", {}).items() if k.startswith("langid_")}
    delta = max((abs(cached[k]-full_glot[k]) for k in ("langid_fr", "langid_family") if k in cached), default=0) if full_glot else 0
    if cached.get("langid_model") == "glotlid" and delta > 1e-5:
        raise RuntimeError(f"Le score GlotLID ne se reproduit pas : {doc['id']} ({delta})")
    new = decisions(doc)[1] and not decisions(doc)[0]
    return {"id": doc["id"], "source": doc["source"], "roles": doc["audit_roles"], "added_by_family": new,
            "text_sha256": sha(text), "chars": len(text), "bytes": len(text.encode()),
            "features": quality_features(text), "cached": cached, "full_glot": full_glot,
            "full_lid176": full_second, "cached_glot_delta": delta,
            "prefix_glot": glot.score(text[:1500]), "prefix_lid176": independent.score(text[:1500]),
            "coverage_chars": sum(r["end"]-r["start"] for r in ranges), "segments": ranges}


_MODELS = None


def init_models():
    global _MODELS
    from pipeline.clean.langid_score import LangIdScoreStage
    models = [LangIdScoreStage({"model": name, "fallback": None, "download": False},
                              SimpleNamespace(root=ROOT)) for name in ("glotlid", "lid176")]
    for model in models:
        model.setup()
    _MODELS = models


def analyse_unit(args):
    doc, output, config = args
    path = Path(output) / "diagnostics" / (sha(doc["id"]) + ".json")
    old = read_json(path, {})
    if old.get("audit_config") == config and old.get("text_sha256") == sha(doc["text"]):
        return old
    result = analyse_doc(doc, *_MODELS, config["window_chars"])
    result["audit_config"] = config
    write_json(path, result)
    return result


def analyse(output, workers=2, window_chars=3000):
    output = Path(output)
    model_paths = {"glotlid": ROOT / "data/models/glotlid/model.bin",
                   "lid176": ROOT / "data/models/lid176/lid.176.bin"}
    hashes = {}
    for name, path in model_paths.items():
        with path.open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    config = {"window_chars": window_chars, "models_sha256": hashes, "code_sha256": sha(Path(__file__).read_text())}
    counts = Counter()
    sources = defaultdict(Counter)
    results = []
    tasks = ((doc, str(output), config) for doc in read_docs(output / "selected_full.jsonl.zst"))
    with mp.get_context("spawn").Pool(workers, initializer=init_models) as pool:
        for result in pool.imap_unordered(analyse_unit, tasks, chunksize=1):
            results.append(result)
            counts["docs"] += 1
            counts["chars"] += result["chars"]
            counts["coverage_chars"] += result["coverage_chars"]
            counts["segments"] += len(result["segments"])
            sources[result["source"]]["docs"] += 1
            sources[result["source"]]["bytes"] += result["bytes"]
            if counts["docs"] % 25 == 0:
                print(f"Audit intégral : {counts['docs']} documents, {counts['segments']} passages", flush=True)
    expected = read_json(output / "extraction.json")
    if counts["docs"] != expected["selected_docs"] or counts["chars"] != counts["coverage_chars"]:
        raise RuntimeError("Couverture incomplète")
    summary = {"complete": True, "counts": dict(counts), "by_source": {s: dict(v) for s, v in sources.items()},
               "config": config, "human_validation": False, "decision": "diagnostic_only"}
    write_json(output / "analysis.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "analyse"))
    parser.add_argument("--input", default="data/stages/nettoye/04_langid_score")
    parser.add_argument("--sample", default="data/annotations/ia-20261104/langid_sample.jsonl")
    parser.add_argument("--output", default="data/stages/langid_full_audit_20261104")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if not (ROOT / "data/stages").is_symlink():
        raise RuntimeError("data/stages doit rester un lien vers son dossier nosync")
    if args.stage == "extract":
        extract(args.input, args.sample, args.output, args.workers or 8)
    else:
        analyse(args.output, args.workers or 2)


if __name__ == "__main__":
    main()
