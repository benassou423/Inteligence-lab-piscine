"""Mix : sorties du pipeline → dataset ORDONNÉ pour le kit (`data/mixes/<run>/train.jsonl.gz`).

    uv run python -m bench.mix --config configs/bench/<run>.yaml

Le kit (`ftb.prepare`) garde l'ordre des documents : c'est ici qu'on choisit quoi prendre,
combien de fois, et dans quel ordre.
1. index     : lit chaque source (sortie d'une config pipeline, ou chemins) ;
2. sélection : plafond d'octets par source, répétition (ou poids en octets), plafond global ;
3. ordre     : `shuffle` (mélange déterministe), `blocks` (blocs dans l'ordre croissant de
               `block`, mélangés à l'intérieur : « garder le meilleur pour la fin »), `as_is` ;
4. écriture  : `train.jsonl.gz` (champ `text`), `tokenizer_sample.jsonl.gz` (échantillon pour le
               tokenizer) et `manifest.json` (octets et documents par source, sha256 des fichiers).
Option : décontamination contre `data/validation.jsonl` (13-grammes, gabarits ignorés), filet de
sécurité si une source ne passe pas par l'étape `decontaminate` du pipeline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import multiprocessing as mp
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zstandard
from numpy.lib.stride_tricks import sliding_window_view

from bench.common import (
    REPO, doc_files, final_stage_dir, get_logger, h64, human_bytes, resolve_path, sha256_file,
    sha256_json, stage_limit, write_json, read_json,
)

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
SALT_REP, SALT_TOK = np.uint64(0x5EED0001), np.uint64(0x5EED0002)
POW_BASE = np.uint64(1_000_003)
INPUT_SUFFIXES = (".jsonl.zst", ".jsonl.gz", ".jsonl", ".txt")

MIX_DEFAULTS = {
    "sources": [],
    "order": "shuffle",                # shuffle | blocks | as_is
    "max_bytes": None,                 # plafond global (sous-échantillon uniforme, proportions gardées)
    "total_bytes": None,               # taille visée quand les sources ont un `weight`
    "prefilter": True,                 # avec max_bytes : saute sans les décoder les documents hors plafond
    "tokenizer_sample_bytes": 100_000_000,
    "decontaminate": {"enabled": False, "against": "data/validation.jsonl", "ngram": 13,
                      "min_matches": 50, "max_df": 10},
    "workers": 8,
}
SOURCE_DEFAULTS = {
    "name": None,
    "path": None,          # fichier, dossier ou motif glob (.jsonl.zst, .jsonl.gz, .jsonl, .txt)
    "pipeline": None,      # config pipeline : sa dernière étape terminée (data/stages/<config>/)
    "stage": None,         # ou une étape précise, ex. 04_langid
    "sources": None,       # filtre sur le champ `source` des documents
    "repeat": 1.0,         # facteur de répétition (0,5 = moitié tirée au hasard, 2 = deux fois)
    "weight": None,        # alternative : part du mix en octets (avec total_bytes)
    "max_bytes": None,     # plafond d'octets pris dans cette source (avant répétition)
    "block": 0,            # ordre `blocks` : les blocs se suivent dans l'ordre croissant
}


# --------------------------------------------------------------------------- hachage vectorisé

def splitmix64(x) -> np.ndarray:
    """Mélangeur 64 bits (dérive des tirages indépendants d'un même hachage)."""
    z = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        z += GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return z


def to_unit(x) -> np.ndarray:
    """uint64 → flottant uniforme dans [0, 1)."""
    return (np.asarray(x, dtype=np.uint64) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# --------------------------------------------------------------------------- lecture des fichiers

def iter_lines(path):
    """(n° de ligne, ligne JSON brute) d'un fichier .jsonl[.zst|.gz] ; un .txt = un seul document."""
    path = str(path)
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            yield 0, json.dumps({"id": f"file:{Path(path).name}", "text": f.read()}, ensure_ascii=False)
        return
    if path.endswith(".zst"):
        opener = zstandard.open(path, "rt", encoding="utf-8")
    elif path.endswith(".gz"):
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, encoding="utf-8")
    with opener as f:
        for ln, line in enumerate(f):
            if line.strip():
                yield ln, line


def _doc_id(doc, path, ln) -> str:
    return str(doc.get("id") or f"{Path(path).name}:{ln}")


# --------------------------------------------------------------------------- décontamination

_WORD_RE = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Mots en minuscules, sans ponctuation ni apostrophes (comme `ftb/contamination.py`)."""
    return _WORD_RE.findall(text.lower())


def _ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hachage polynomial de chaque fenêtre de n identifiants de mots."""
    powers = np.array([int(POW_BASE) ** (n - 1 - j) % (1 << 64) for j in range(n)], dtype=np.uint64)
    win = sliding_window_view(ids.astype(np.uint64), n)
    with np.errstate(over="ignore"):
        return (win * powers).sum(axis=1, dtype=np.uint64)


def build_val_ngrams(texts: list[str], n: int) -> tuple[dict, np.ndarray]:
    """Vocabulaire de la val et n-grammes propres à un seul document de la val (sinon : gabarit)."""
    vocab: dict[str, int] = {}
    counts: Counter = Counter()
    for t in texts:
        w = words(t)
        if len(w) < n:
            continue
        ids = np.array([vocab.setdefault(x, len(vocab)) + 1 for x in w], dtype=np.int64)
        counts.update(np.unique(_ngram_hashes(ids, n)).tolist())
    kept = np.array(sorted(h for h, c in counts.items() if c == 1), dtype=np.uint64)
    return vocab, kept


def shared_ngrams(text: str, vocab: dict, val_hashes: np.ndarray, n: int) -> np.ndarray:
    """n-grammes de mots (hachés, distincts) du texte qui sont aussi dans la val."""
    if len(val_hashes) == 0:
        return np.zeros(0, np.uint64)
    w = words(text)
    if len(w) < n:
        return np.zeros(0, np.uint64)
    ids = np.fromiter((vocab.get(x, -1) + 1 for x in w), dtype=np.int64, count=len(w))
    bad = np.concatenate([[0], np.cumsum(ids == 0)])
    ok = (bad[n:] - bad[:-n]) == 0  # fenêtres sans mot inconnu de la val
    if not ok.any():
        return np.zeros(0, np.uint64)
    h = _ngram_hashes(ids, n)[ok]
    pos = np.searchsorted(val_hashes, h)
    pos[pos == len(val_hashes)] = 0
    return np.unique(h[val_hashes[pos] == h])


def contaminated_docs(results: list[dict], min_matches: int, max_df: int) -> tuple[set, int]:
    """Documents à retirer : ≥ `min_matches` n-grammes partagés avec la val, sans compter ceux
    présents dans plus de `max_df` documents du mix (gabarits : rubrique « Climat » des communes
    générée par robot, mentions légales…), comme le filtre de contamination de GPT-3.
    Renvoie ({(tâche, ligne)}, nombre de n-grammes de gabarit ignorés)."""
    df: Counter = Counter()
    for r in results:
        df.update(dict(zip(r["df_hash"].tolist(), r["df_count"].tolist())))
    frequent = np.array(sorted(h for h, c in df.items() if c > max_df), dtype=np.uint64)
    out = set()
    for r in results:
        start = 0
        for ln, k in zip(r["cand_lines"].tolist(), r["cand_len"].tolist()):
            m = r["cand_hash"][start:start + k]
            start += k
            if len(frequent):
                m = m[~np.isin(m, frequent)]
            if len(m) >= min_matches:
                out.add((r["tid"], ln))
    return out, len(frequent)


# --------------------------------------------------------------------------- travailleurs

_G: dict = {}
_ID_RE = re.compile(r'^\{"id":\s*"((?:[^"\\]|\\.)*)"')


def _init_worker(state: dict):
    _G.clear()
    _G.update(state)


def _index_task(task):
    """Index d'un fichier : (ligne, octets, hachages, source) des documents non vides.

    Avec un pré-filtre (`prefilter_rate` < 1), un document dont le hachage de sélection dépasse
    le seuil est sauté sans décoder son JSON : le plafond global garde les plus petits hachages,
    il n'aurait donc pas été retenu (la sélection finale est identique, voir les tests)."""
    tid, path, wanted = task
    seed = _G["seed"]
    rate = _G.get("prefilter_rate")
    limit = int(rate * 2.0**64) if rate is not None and rate < 1.0 else None
    lines, nbytes, h_sel, h_ord, codes = [], [], [], [], []
    names: dict[str, int] = {}
    seen = empty = skipped = excluded = 0
    for ln, line in iter_lines(path):
        if limit is not None:
            m = _ID_RE.match(line)
            if m and h64(seed, "sel", json.loads(f'"{m.group(1)}"')) >= limit:
                skipped += 1
                continue
        doc = json.loads(line)
        src = doc.get("source") or Path(path).parent.name
        if wanted and src not in wanted:
            continue
        seen += 1
        did = _doc_id(doc, path, ln)
        if did in _G.get("excluded_ids", ()):
            excluded += 1
            continue
        text = doc.get("text") or ""
        if not text.strip():
            empty += 1
            continue
        lines.append(ln)
        nbytes.append(len(text.encode("utf-8")))
        h_sel.append(h64(seed, "sel", did))
        h_ord.append(h64(seed, "ord", did))
        codes.append(names.setdefault(src, len(names)))
    return {
        "tid": tid, "lines": np.array(lines, np.int64), "nbytes": np.array(nbytes, np.int64),
        "h_sel": np.array(h_sel, np.uint64), "h_ord": np.array(h_ord, np.uint64),
        "codes": np.array(codes, np.int32), "names": list(names), "seen": seen, "empty": empty,
        "skipped": skipped, "excluded_heldout": excluded,
    }


def _extract_task(task):
    """Écrit les documents retenus d'un fichier en lignes JSON `{"text": …}` (shard temporaire).

    Note aussi les n-grammes de la val trouvés (décontamination)."""
    tid, path, lines, out_shard = task
    dec = _G.get("decontam")
    want = set(lines.tolist())
    kept, nbyt = [], []
    cand_lines, cand_len, cand_hash = [], [], []
    df: Counter = Counter()
    tmp = Path(str(out_shard) + ".part")
    with open(tmp, "wb") as f:
        for ln, line in iter_lines(path):
            if ln not in want:
                continue
            doc = json.loads(line)
            text = doc["text"]
            if dec:
                m = shared_ngrams(text, dec["vocab"], dec["hashes"], dec["n"])
                if len(m):
                    df.update(m.tolist())
                    if len(m) >= dec["min_matches"]:
                        cand_lines.append(ln)
                        cand_len.append(len(m))
                        cand_hash.append(m)
            record = {"text": text}
            if _G.get("keep_metadata"):
                record.update({k: v for k, v in doc.items() if k != "html"})
            rec = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(rec)
            kept.append(ln)
            nbyt.append(len(rec))
    tmp.replace(out_shard)
    return {"tid": tid, "lines": np.array(kept, np.int64), "reclen": np.array(nbyt, np.int64),
            "cand_lines": np.array(cand_lines, np.int64), "cand_len": np.array(cand_len, np.int64),
            "cand_hash": np.concatenate(cand_hash) if cand_hash else np.zeros(0, np.uint64),
            "df_hash": np.array(list(df.keys()), np.uint64), "df_count": np.array(list(df.values()), np.int64)}


def _pool_map(fn, tasks, workers, state, log=None, label=""):
    """Applique `fn` aux tâches (en parallèle si utile) ; résultats dans l'ordre des tâches."""
    t0 = time.time()
    if workers <= 1 or len(tasks) <= 1:
        _init_worker(state)
        out = [fn(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(state,)) as pool:
            out = pool.map(fn, tasks, chunksize=1)
    if log and label:
        log.info(f"{label} : {len(tasks)} fichiers en {time.time() - t0:.1f} s")
    return out


# --------------------------------------------------------------------------- sources

def _input_files(p: Path) -> list[Path]:
    if p.is_dir():
        return [f for f in sorted(p.rglob("*")) if f.is_file() and f.name.endswith(INPUT_SUFFIXES)
                and not any(part.startswith(("_", ".")) for part in f.relative_to(p).parts)]
    if any(c in str(p) for c in "*?["):
        return sorted(Path(x) for x in glob.glob(str(p), recursive=True))
    return [p] if p.exists() else []


def resolve_sources(mix_cfg: dict, data_root: Path, log) -> list[dict]:
    """Fichiers de chaque source déclarée dans `mix.sources`."""
    entries = []
    for i, s in enumerate(mix_cfg["sources"]):
        s = {**SOURCE_DEFAULTS, **s}
        warn = None
        if s["path"]:
            files = _input_files(resolve_path(s["path"]))
            origin = str(s["path"])
            if not (mix_cfg.get("decontaminate") or {}).get("enabled"):
                # mesuré le 03/11 : un mix tiré de data/raw/wikinews_fr contenait 32 des 104 articles de presse de la val
                warn = (f"{origin} n'est pas passé par l'étape decontaminate du pipeline : la val peut y être "
                        f"(mettre mix.decontaminate.enabled: true)")
        elif s["pipeline"]:
            name = Path(str(s["pipeline"])).stem
            root = data_root / "stages" / name
            stage = root / s["stage"] if s["stage"] else final_stage_dir(data_root / "stages", name)
            if stage is None or not stage.is_dir():
                raise SystemExit(f"mix : source '{s.get('name')}' : aucune étape terminée dans {root}")
            files = doc_files(stage)
            origin = str(stage.relative_to(data_root))
            limit = stage_limit(stage)
            if limit is not None:
                warn = f"{origin} vient d'un run d'essai du pipeline (--limit {limit}) : ce n'est qu'un échantillon"
        else:
            raise SystemExit("mix : chaque source demande `path` ou `pipeline`")
        wanted = set(s["sources"]) if s["sources"] else None
        if wanted:
            by_dir = [f for f in files if f.parent.name in wanted]
            files = by_dir or files  # sinon le filtre se fait document par document
        s["name"] = s["name"] or (s["sources"][0] if s["sources"] and len(s["sources"]) == 1 else f"source{i}")
        if warn:
            log.warning(f"source '{s['name']}' : {warn}")
        if not files:
            raise SystemExit(f"mix : source '{s['name']}' : aucun fichier ({origin})")
        entries.append({**s, "files": files, "origin": origin, "warning": warn, "wanted": wanted})
    return entries


def _prefilter_rate(mix_cfg, entries) -> float | None:
    """Seuil du pré-filtre par hachage quand un plafond global est fixé (estimation prudente :
    ≥ 2 octets de texte par octet compressé, marge ×2). Désactivé avec des poids ou une répétition
    non entière : la part prise d'une source dépend alors de TOUTE la source."""
    cap = mix_cfg.get("max_bytes")
    if not cap or not mix_cfg.get("prefilter", True) or any(e["weight"] is not None for e in entries):
        return None
    if any(float(e["repeat"]) != int(float(e["repeat"])) for e in entries):
        return None
    comp = sum(f.stat().st_size for e in entries for f in e["files"])
    r_min = max(min(float(e["repeat"]) for e in entries), 0.05)
    rate = cap / (comp * r_min) if comp else 1.0
    return None if rate >= 1.0 else rate


# --------------------------------------------------------------------------- sélection

def select_copies(entries, index, mix_cfg, log, info: dict | None = None) -> dict[int, np.ndarray]:
    """Nombre de copies de chaque document indexé (0 = écarté).

    `info["precap_bytes"]` reçoit les octets (copies comprises) avant le plafond global."""
    by_entry = defaultdict(list)
    for t in index.values():
        by_entry[t["entry"]].append(t["tid"])
    copies = {tid: np.zeros(len(t["lines"]), np.int64) for tid, t in index.items()}
    keep = {tid: np.ones(len(t["lines"]), bool) for tid, t in index.items()}

    # plafond par source (avant répétition) : les documents de plus petit hachage
    avail = {i: 0 for i in range(len(entries))}
    for e_i, e in enumerate(entries):
        tids = by_entry.get(e_i, [])
        nb = np.concatenate([index[t]["nbytes"] for t in tids]) if tids else np.zeros(0, np.int64)
        if e["max_bytes"] and nb.sum() > e["max_bytes"]:
            hs = np.concatenate([index[t]["h_sel"] for t in tids])
            order = np.argsort(hs, kind="stable")
            ok = np.zeros(len(nb), bool)
            ok[order[np.cumsum(nb[order]) <= e["max_bytes"]]] = True
            start = 0
            for t in tids:
                n = len(index[t]["lines"])
                keep[t] = ok[start:start + n]
                start += n
        avail[e_i] = int(sum(index[t]["nbytes"][keep[t]].sum() for t in tids))

    # facteur de répétition : `repeat`, ou déduit des poids (part du mix en octets)
    weights = [e["weight"] for e in entries]
    if any(w is not None for w in weights):
        if any(w is None for w in weights):
            raise SystemExit("mix : mettre `weight` sur toutes les sources ou sur aucune")
        total = mix_cfg["total_bytes"] or sum(avail.values())
        wsum = float(sum(weights))
        factors = {i: (weights[i] / wsum * total / avail[i]) if avail[i] else 0.0 for i in range(len(entries))}
    else:
        factors = {i: float(e["repeat"]) for i, e in enumerate(entries)}
    # r = 2,4 : 2 copies de chaque document, plus une 3e copie pour 40 % des OCTETS de la source.
    # Ces documents sont pris dans un ordre aléatoire fixe (hachage) jusqu'à 0,4 × octets, au
    # demi-document près : la part visée est tenue en octets, même avec de gros documents (livres,
    # séances), là où un tirage document par document s'en écartait de ±20 %.
    for e_i, tids in by_entry.items():
        r = factors[e_i]
        base, frac = int(np.floor(r)), r - np.floor(r)
        nb = np.concatenate([index[t]["nbytes"] * keep[t] for t in tids]) if tids else np.zeros(0, np.int64)
        extra = np.zeros(len(nb), np.int64)
        if frac > 0 and nb.sum():
            u = np.concatenate([splitmix64(index[t]["h_sel"] ^ SALT_REP) for t in tids])
            order = np.argsort(u, kind="stable")
            took = np.cumsum(nb[order]) - nb[order] / 2 <= frac * nb.sum()
            extra[order[took]] = 1
        start = 0
        for t in tids:
            n = len(index[t]["lines"])
            copies[t] = (base + extra[start:start + n]) * keep[t]
            start += n

    # plafond global : sous-échantillon uniforme (mêmes proportions en moyenne)
    cap = mix_cfg["max_bytes"]
    if info is not None:
        info["precap_bytes"] = int(sum((index[t]["nbytes"] * copies[t]).sum() for t in index))
    if cap:
        tids = sorted(index)
        hs = np.concatenate([index[t]["h_sel"] for t in tids])
        w = np.concatenate([index[t]["nbytes"] * copies[t] for t in tids])
        if w.sum() > cap:
            order = np.argsort(hs, kind="stable")
            drop = np.zeros(len(w), bool)
            drop[order[np.cumsum(w[order]) > cap]] = True
            start = 0
            for t in tids:
                n = len(index[t]["lines"])
                copies[t][drop[start:start + n]] = 0
                start += n
    for e_i, e in enumerate(entries):
        log.info(f"source '{e['name']}' : {human_bytes(avail.get(e_i, 0))} disponibles, facteur {factors.get(e_i, 0):.3g}")
    return copies


# --------------------------------------------------------------------------- construction

def _fingerprint(mix_cfg, seed, entries, val_path) -> str:
    files = [[str(f), f.stat().st_size, int(f.stat().st_mtime)] for e in entries for f in e["files"]]
    cfg = {k: v for k, v in mix_cfg.items() if k not in ("workers", "prefilter")}
    dec = mix_cfg["decontaminate"]
    val = sha256_file(val_path) if dec.get("enabled") and val_path.exists() else None
    # version 2 : poids et répétitions non entières exacts en octets, segments du flux
    return sha256_json({"mix": cfg, "seed": seed, "files": files, "val": val, "version": 2})


def build_mix(mix_cfg: dict, out_dir, seed: int = 1337, data_root=None, force: bool = False, log=None,
              *, excluded_ids=None, keep_metadata=False) -> dict:
    """Construit le mix ; renvoie le manifeste (déjà fait et inchangé : relu tel quel)."""
    mix_cfg = {**MIX_DEFAULTS, **mix_cfg,
               "decontaminate": {**MIX_DEFAULTS["decontaminate"], **(mix_cfg.get("decontaminate") or {})}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or get_logger("bench", out / "mix.log")
    data_root = Path(data_root) if data_root else REPO / "data"
    t0 = time.time()
    workers = int(mix_cfg.get("workers") or 1)
    entries = resolve_sources(mix_cfg, data_root, log)
    dec_cfg = mix_cfg["decontaminate"]
    val_path = resolve_path(dec_cfg["against"])
    fp = _fingerprint(mix_cfg, seed, entries, val_path)
    excluded_ids = frozenset(excluded_ids or ())
    if excluded_ids or keep_metadata:
        fp = sha256_json({"base": fp, "heldout_ids": sorted(excluded_ids), "keep_metadata": keep_metadata})
    man_path, train_path = out / "manifest.json", out / "train.jsonl.gz"
    if not force and man_path.exists() and train_path.exists():
        man = read_json(man_path)
        if man.get("fingerprint") == fp and man.get("train_sha256") == sha256_file(train_path):
            log.info(f"mix déjà fait et inchangé ({man['train_sha256'][:12]}…), --force pour refaire")
            return man

    # 1. index (avec pré-filtre exact quand un plafond global est fixé)
    tasks, task_entry = [], {}
    for e_i, e in enumerate(entries):
        for f in e["files"]:
            task_entry[len(tasks)] = e_i
            tasks.append((len(tasks), str(f), e["wanted"]))
    rate = _prefilter_rate(mix_cfg, entries)
    while True:
        index = {}
        for res in _pool_map(_index_task, tasks, workers, {"seed": seed, "prefilter_rate": rate,
                             "excluded_ids": excluded_ids}, log, "index"):
            res["entry"] = task_entry[res["tid"]]
            index[res["tid"]] = res
        skipped = sum(t["skipped"] for t in index.values())
        if rate is None or not skipped:
            copies = select_copies(entries, index, mix_cfg, log)
            break
        # exact si le plafond global coupe DANS les documents gardés
        sel_info: dict = {}
        copies = select_copies(entries, index, mix_cfg, log, sel_info)
        if sel_info["precap_bytes"] > mix_cfg["max_bytes"]:
            log.info(f"pré-filtre : {skipped:,} documents sautés sans décodage (seuil {rate:.4f}), sélection identique")
            break
        rate = min(1.0, rate * 4)
        log.info(f"pré-filtre trop serré : relecture avec un seuil {rate:.4f}")
    names = sorted({n for t in index.values() for n in t["names"]})
    code = {n: i for i, n in enumerate(names)}
    for t in index.values():
        remap = np.array([code[n] for n in t["names"]] or [0], np.int32)
        t["gcodes"] = remap[t["codes"]] if len(t["codes"]) else t["codes"]
    log.info(f"{sum(t['seen'] for t in index.values()):,} documents lus, {sum(t['empty'] for t in index.values()):,} vides")

    # 2. décontamination (optionnelle) : n-grammes de la val
    decontam = None
    if dec_cfg.get("enabled"):
        with open(val_path, encoding="utf-8") as f:
            vtexts = [json.loads(line)["text"] for line in f if line.strip()]
        vocab, hashes = build_val_ngrams(vtexts, int(dec_cfg["ngram"]))
        decontam = {"vocab": vocab, "hashes": hashes, "n": int(dec_cfg["ngram"]), "min_matches": int(dec_cfg["min_matches"])}

    # 3. extraction des documents retenus (shards temporaires, en parallèle)
    tmp = out / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    etasks = [(tid, tasks[tid][1], index[tid]["lines"][copies[tid] > 0], str(tmp / f"{tid:05d}.jsonl"))
              for tid in sorted(index) if (copies[tid] > 0).any()]
    if not etasks:
        raise SystemExit("mix : la sélection est vide")
    done = {r["tid"]: r for r in _pool_map(_extract_task, etasks, workers,
            {"decontam": decontam, "keep_metadata": keep_metadata}, log, "extraction")}
    drop, n_frequent = set(), 0
    if decontam:
        drop, n_frequent = contaminated_docs(list(done.values()), decontam["min_matches"], int(dec_cfg["max_df"]))

    # 4. ordre final et écriture
    man = _write(mix_cfg, seed, entries, index, copies, done, drop, tmp, out, names, log)
    shutil.rmtree(tmp, ignore_errors=True)
    man.update({
        "fingerprint": fp,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "inputs": [{"name": e["name"], "origin": e["origin"], "files": len(e["files"]), "warning": e["warning"]}
                   for e in entries],
        "decontamination": {"enabled": bool(decontam), "against": str(dec_cfg["against"]),
                            "template_ngrams_ignored": n_frequent},
        "prefilter": {"rate": rate, "skipped_docs": sum(t["skipped"] for t in index.values())},
        "config": mix_cfg,
        "seconds": round(time.time() - t0, 1),
    })
    if excluded_ids or keep_metadata:
        man["heldout_exclusion"] = {"listed_ids": len(excluded_ids),
                                    "ids_sha256": sha256_json(sorted(excluded_ids)),
                                    "excluded_in_index": sum(t["excluded_heldout"] for t in index.values())}
        man["metadata_preserved"] = bool(keep_metadata)
    write_json(man_path, man)
    log.info(f"mix : {man['documents']:,} documents, {human_bytes(man['bytes'])} → {train_path} "
             f"(sha256 {man['train_sha256'][:12]}…, {man['seconds']} s)")
    return man


def _write(mix_cfg, seed, entries, index, copies, done, drop, tmp, out, names, log) -> dict:
    """Écrit train.jsonl.gz dans l'ordre final, et l'échantillon du tokenizer."""
    tids, offs, lens, reps, hords, blocks, ents, srcs, nbytes, hsel = ([] for _ in range(10))
    contaminated: Counter = Counter()
    for tid in sorted(done):
        r, t = done[tid], index[tid]
        off = np.concatenate([[0], np.cumsum(r["reclen"])[:-1]]).astype(np.int64)
        keep = np.array([(tid, int(ln)) not in drop for ln in r["lines"]], bool) if drop else np.ones(len(r["lines"]), bool)
        pos = np.searchsorted(t["lines"], r["lines"])
        if (~keep).any():
            contaminated.update(names[c] for c in t["gcodes"][pos[~keep]])
        pos = pos[keep]
        n = len(pos)
        tids.append(np.full(n, tid, np.int64))
        offs.append(off[keep])
        lens.append(r["reclen"][keep])
        reps.append(copies[tid][pos])
        hords.append(t["h_ord"][pos])
        hsel.append(t["h_sel"][pos])
        ents.append(np.full(n, t["entry"], np.int64))
        blocks.append(np.full(n, int(entries[t["entry"]]["block"]), np.int64))
        srcs.append(t["gcodes"][pos])
        nbytes.append(t["nbytes"][pos])
    cat = lambda xs, dt_: np.concatenate(xs) if xs else np.zeros(0, dt_)  # noqa: E731
    tids, offs, lens, reps = cat(tids, np.int64), cat(offs, np.int64), cat(lens, np.int64), cat(reps, np.int64)
    hords, hsel, ents, blocks = cat(hords, np.uint64), cat(hsel, np.uint64), cat(ents, np.int64), cat(blocks, np.int64)
    srcs, nbytes = cat(srcs, np.int32), cat(nbytes, np.int64)
    if contaminated:
        log.info(f"décontamination : {sum(contaminated.values())} documents retirés {dict(contaminated)}")

    # une ligne par copie de document, puis ordre final
    doc_of = np.repeat(np.arange(len(reps)), reps)
    first = np.concatenate([[0], np.cumsum(reps)[:-1]]) if len(reps) else np.zeros(0, np.int64)
    k = np.arange(len(doc_of)) - first[doc_of]
    with np.errstate(over="ignore"):
        key = splitmix64(hords[doc_of] + k.astype(np.uint64) * GOLDEN)
    order_mode = mix_cfg["order"]
    if order_mode == "shuffle":
        order = np.argsort(key, kind="stable")
    elif order_mode == "blocks":
        order = np.lexsort((key, blocks[doc_of]))  # blocs croissants, mélangés à l'intérieur
    elif order_mode == "as_is":
        order = np.lexsort((offs[doc_of], tids[doc_of], k, ents[doc_of]))
    else:
        raise SystemExit(f"mix.order inconnu : {order_mode!r} (shuffle | blocks | as_is)")
    seq = doc_of[order]
    if len(seq) == 0:
        raise SystemExit("mix : aucun document (tout a été filtré)")
    # segments du flux (ordres blocks et as_is) : le banc vérifie ensuite que le budget de tokens
    # ne coupe pas le dernier bloc (bench.run.stream_coverage)
    segments = []
    if order_mode != "shuffle":
        lab = (blocks if order_mode == "blocks" else ents)[seq]
        cut = np.flatnonzero(np.diff(lab)) + 1
        for a, b in zip(np.r_[0, cut].tolist(), np.r_[cut, len(seq)].tolist()):
            v = int(lab[a])
            segments.append({"label": f"bloc {v}" if order_mode == "blocks" else f"source '{entries[v]['name']}'",
                             "documents": b - a, "bytes": int(nbytes[seq[a:b]].sum())})

    # échantillon du tokenizer : tirage de Bernoulli déterministe parmi les documents distincts
    uniq_bytes = int(nbytes.sum())
    p = min(1.0, float(mix_cfg["tokenizer_sample_bytes"] or 0) / max(uniq_bytes, 1))
    in_sample = to_unit(splitmix64(hsel ^ SALT_TOK)) < p

    shards = {tid: np.memmap(tmp / f"{tid:05d}.jsonl", dtype=np.uint8, mode="r")
              for tid in np.unique(tids).tolist() if (tmp / f"{tid:05d}.jsonl").stat().st_size}
    train_path, sample_path = out / "train.jsonl.gz", out / "tokenizer_sample.jsonl.gz"
    sampled = np.zeros(len(reps), bool)
    sample_docs = sample_bytes = 0
    # gzip déterministe (mtime=0, pas de nom de fichier) : même mix → même fichier, même sha256
    with open(train_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                      compresslevel=6) as gz, \
         open(sample_path, "wb") as raw_s, gzip.GzipFile(filename="", mode="wb", fileobj=raw_s, mtime=0,
                                                         compresslevel=6) as gz_s:
        for i in seq.tolist():
            rec = shards[int(tids[i])][int(offs[i]):int(offs[i]) + int(lens[i])].tobytes()
            gz.write(rec)
            if in_sample[i] and not sampled[i]:
                sampled[i] = True
                gz_s.write(rec)
                sample_docs += 1
                sample_bytes += int(nbytes[i])
    del shards

    by_src = {}
    for c in np.unique(srcs).tolist():
        m = srcs == c
        by_src[names[c]] = {"documents": int(reps[m].sum()), "unique_documents": int((reps[m] > 0).sum()),
                            "bytes": int((nbytes[m] * reps[m]).sum())}
    total = int((nbytes * reps).sum())
    for v in by_src.values():
        v["share_bytes"] = round(v["bytes"] / max(total, 1), 4)
    by_entry = {}
    for e_i, e in enumerate(entries):
        m = ents == e_i
        by_entry[e["name"]] = {"documents": int(reps[m].sum()), "bytes": int((nbytes[m] * reps[m]).sum()),
                               "block": int(e["block"])}
    return {
        "documents": int(reps.sum()), "unique_documents": int((reps > 0).sum()), "bytes": total,
        "order": order_mode, "by_source": by_src, "by_entry": by_entry, "segments": segments,
        "contaminated_by_source": dict(contaminated),
        "train_file": train_path.name, "train_sha256": sha256_file(train_path),
        "tokenizer_sample": {"file": sample_path.name, "documents": sample_docs, "bytes": sample_bytes,
                             "sha256": sha256_file(sample_path)},
    }


def main(argv=None):
    from bench.config import load_run_config, mix_dir, data_root as cfg_data_root

    ap = argparse.ArgumentParser(description="Construit le dataset ordonné d'un run (data/mixes/<run>/)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_run_config(args.config)
    build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], cfg_data_root(cfg), force=args.force)


if __name__ == "__main__":
    main()
