"""Outils communs aux collecteurs : config, journal, HTTP poli, écriture en parts, reprise.

Chaque source est un module qui expose :
- NAME, DOMAIN, LICENSE ;
- prepare(cfg, log, sample) -> ctx (dict picklable) ;
- units(ctx) -> liste triée d'unités de travail (fichiers, lots…) ;
- process(unit, ctx, tally, log) -> itérateur de documents.
Le lanceur (run_source) gère l'échantillon (--limit), les workers, la reprise et la finalisation.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import multiprocessing as mp
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import yaml

from pipeline.io import write_docs

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "configs" / "collect.yaml"

FREEZE_DATE = "2026-10-16"  # gel des données : rien de daté d'après
MAX_PART_BYTES = 100_000_000  # octets UTF-8 de `text` au plus par fichier
USER_AGENT = "FeedTheBeast-ECE-Paris/0.1 (projet etudiant piscine I Lab, ECE Paris; https://www.ece.fr)"
RETRY_STATUS = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------- config et journal

def load_config(path=None) -> dict:
    """Lit configs/collect.yaml."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    global FREEZE_DATE, MAX_PART_BYTES
    FREEZE_DATE = str(cfg.get("freeze_date", FREEZE_DATE))
    MAX_PART_BYTES = int(cfg.get("max_part_text_bytes", MAX_PART_BYTES))
    return cfg


def source_dir(name: str) -> Path:
    return RAW_DIR / name


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Journal dans <log_dir>/_collect.log (par défaut data/raw/<name>/) et sur stderr."""
    log_dir = Path(log_dir) if log_dir else source_dir(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"collect.{name}")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(process)d] %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_dir / "_collect.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        if sys.stderr.isatty():  # en tâche détachée (nohup … >> _collect.log 2>&1), pas de doublon
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            log.addHandler(sh)
        log.propagate = False
    return log


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- HTTP

def make_session(user_agent: str = USER_AGENT) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


def http_get(session, url, *, log=None, stream=False, headers=None, timeout=(30, 300),
             max_tries=8, base_delay=2.0, max_delay=300.0, ok=(200, 206)):
    """GET avec backoff exponentiel sur 429/5xx (dont 503 « Slow Down ») et erreurs réseau.

    Les autres codes d'erreur (404…) lèvent requests.HTTPError tout de suite."""
    delay = base_delay
    for attempt in range(1, max_tries + 1):
        try:
            r = session.get(url, stream=stream, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            err = repr(e)
        else:
            if r.status_code in ok:
                return r
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()
                return r
            err = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After", "")
            r.close()
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        if attempt == max_tries:
            raise RuntimeError(f"échec après {max_tries} essais : {url} ({err})")
        wait = min(max_delay, delay) * (0.75 + random.random() / 2)
        if log:
            log.warning("%s sur %s : nouvel essai %d/%d dans %.0f s", err, url, attempt + 1, max_tries, wait)
        time.sleep(wait)
        delay *= 2
    raise AssertionError("inaccessible")


def download(session, url, dest, *, log=None, chunk=1 << 20, max_tries=10):
    """Télécharge url vers dest, avec reprise (en-tête Range) après coupure.

    On abandonne après max_tries coupures de suite sans aucun octet de plus."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    delay = 5.0
    attempt, last_size = 0, -1
    while True:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if have > last_size:  # progrès depuis la dernière coupure : compteur remis à zéro
            attempt, last_size, delay = 1, have, 5.0
        headers = {"Range": f"bytes={have}-"} if have else None
        try:
            r = http_get(session, url, stream=True, headers=headers, log=log, ok=(200, 206, 416), timeout=(30, 60))
            if r.status_code == 416:  # déjà complet
                r.close()
                part.rename(dest)
                return dest
            total = None
            if r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
                total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                mode = "ab"
            else:
                have = 0
                mode = "wb"
                if r.headers.get("Content-Length", "").isdigit():
                    total = int(r.headers["Content-Length"])
            with open(part, mode) as f:
                for b in r.iter_content(chunk):
                    f.write(b)
            r.close()
            size = part.stat().st_size
            if total is not None and size < total:
                raise OSError(f"téléchargement incomplet ({size}/{total} octets)")
            part.rename(dest)
            return dest
        except (requests.RequestException, OSError) as e:
            if attempt >= max_tries and (part.stat().st_size if part.exists() else 0) <= last_size:
                raise
            if log:
                log.warning("téléchargement interrompu (%s) : reprise dans %.0f s", e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)


# ---------------------------------------------------------------- documents

def make_doc(source, domain, key, text, *, url, date, license, meta=None, html=None) -> dict:
    """Construit un document au format commun (docs/ARCHITECTURE.md)."""
    doc = {
        "id": f"{source}:{key}",
        "source": source,
        "domain": domain,
        "text": text,
        "url": url,
        "date": date,
        "license": license,
        "meta": meta or {},
    }
    if html is not None:
        doc["html"] = html
    return doc


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def date_ok(date) -> bool:
    """Vrai si la date (AAAA-MM-JJ, jour qui existe) ne dépasse pas la date de gel.

    Format vérifié : la comparaison de chaînes n'est juste qu'entre dates AAAA-MM-JJ complètes."""
    d = str(date)[:10] if date else ""
    if not _ISO_DATE.fullmatch(d):
        return False
    try:
        dt.date.fromisoformat(d)
    except ValueError:  # « 2012-02-31 »
        return False
    return d <= FREEZE_DATE


def text_bytes(doc) -> int:
    return len(doc["text"].encode("utf-8"))


class Tally:
    """Compteurs d'une unité de travail : documents gardés, octets, dates, rejets par raison."""

    def __init__(self):
        self.docs = 0
        self.bytes = 0
        self.date_min = None
        self.date_max = None
        self.removed = Counter()
        self.extra = {}

    def add(self, doc):
        self.docs += 1
        self.bytes += text_bytes(doc)
        d = doc.get("date")
        if d:
            self.date_min = d if self.date_min is None or d < self.date_min else self.date_min
            self.date_max = d if self.date_max is None or d > self.date_max else self.date_max

    def skip(self, reason, n=1):
        self.removed[reason] += n

    def to_dict(self):
        return {"docs": self.docs, "bytes": self.bytes, "date_min": self.date_min, "date_max": self.date_max,
                "removed_by_reason": dict(self.removed), "extra": self.extra}

    @staticmethod
    def merge(dicts):
        out = {"docs": 0, "bytes": 0, "date_min": None, "date_max": None, "removed_by_reason": Counter(), "extra": {}}
        for d in dicts:
            out["docs"] += d["docs"]
            out["bytes"] += d["bytes"]
            for k, f in (("date_min", min), ("date_max", max)):
                if d.get(k):
                    out[k] = d[k] if out[k] is None else f(out[k], d[k])
            out["removed_by_reason"].update(d.get("removed_by_reason", {}))
        out["removed_by_reason"] = dict(out["removed_by_reason"])
        return out


def write_parts(docs, out_dir, stem="part", start=0, max_bytes=None):
    """Écrit les documents en fichiers <stem>-NNNNN.jsonl.zst de ≤ max_bytes de texte chacun.

    Écriture via pipeline.io.write_docs, dans un .tmp renommé à la fin (pas de fichier à moitié écrit)."""
    max_bytes = max_bytes or MAX_PART_BYTES
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    it = iter(docs)
    state = {"head": next(it, None)}
    paths = []
    k = start
    while state["head"] is not None:
        path = out_dir / f"{stem}-{k:05d}.jsonl.zst"
        tmp = path.with_name(path.name + ".tmp")

        def chunk():
            size = 0
            while state["head"] is not None:
                b = text_bytes(state["head"])
                if size and size + b > max_bytes:
                    return
                size += b
                d, state["head"] = state["head"], None
                yield d
                state["head"] = next(it, None)

        write_docs(tmp, chunk())
        tmp.rename(path)
        paths.append(path)
        k += 1
    return paths


# ---------------------------------------------------------------- état, reprise, finalisation

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)


def safe_name(unit: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", unit)[:150]


def read_done(work: Path) -> dict:
    """Unités déjà traitées (une ligne JSON par unité dans _work/done.jsonl)."""
    done = {}
    f = work / "done.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["unit"]] = rec["stats"]
    return done


def append_done(work: Path, unit: str, stats: dict):
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "done.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"unit": unit, "stats": stats}, ensure_ascii=False) + "\n")


def replace_parts(src: Path, files):
    """Remplace les part-*.jsonl.zst de src par files, renumérotés dans l'ordre donné."""
    for p in src.glob("part-*.jsonl.zst"):
        p.unlink()
    for i, f in enumerate(files):
        Path(f).rename(src / f"part-{i:05d}.jsonl.zst")


def write_stats(src: Path, name: str, mode: str, merged: dict, extra=None):
    stats = {"source": name, "mode": mode, "finished": now_iso(), **merged}
    if extra:
        stats["extra"] = {**stats.get("extra", {}), **extra}
    write_json(src / "_stats.json", stats)
    return stats


def _run_unit(args):
    """Traite une unité dans un worker : écrit _work/<unité>/part-*.jsonl.zst et renvoie ses compteurs."""
    modname, unit, ctx, work = args
    load_config(ctx.get("_config_path"))  # resynchronise FREEZE_DATE et MAX_PART_BYTES dans le worker
    mod = importlib.import_module(modname)
    log = get_logger(mod.NAME)
    out = Path(work) / safe_name(unit)
    if out.exists():
        shutil.rmtree(out)  # reste d'une exécution interrompue
    tally = Tally()

    def gen():
        for d in mod.process(unit, ctx, tally, log):
            tally.add(d)
            yield d

    write_parts(gen(), out)
    return unit, tally.to_dict()


def run_source(mod, cfg, *, limit=None, workers=None, force=False):
    """Lance la collecte d'une source : échantillon (limit) ou complète (avec reprise)."""
    log = get_logger(mod.NAME)
    src = source_dir(mod.NAME)
    state = read_json(src / "_state.json", {}) or {}
    scfg = cfg["sources"][mod.NAME]
    workers = workers or int(scfg.get("workers", 1))
    t0 = time.time()

    if limit is not None:
        if state.get("mode") == "full" and state.get("complete") and not force:
            log.warning("collecte complète déjà présente : échantillon ignoré (utiliser --force)")
            return state
        log.info("échantillon de %d documents", limit)
        ctx = mod.prepare(cfg, log, sample=True)
        tally = Tally()

        def gen():
            n = 0
            for u in mod.units(ctx):
                for d in mod.process(u, ctx, tally, log):
                    tally.add(d)
                    yield d
                    n += 1
                    if n >= limit:
                        return

        tmp = src / "_sample_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        files = write_parts(gen(), tmp)
        replace_parts(src, files)
        shutil.rmtree(tmp, ignore_errors=True)
        stats = write_stats(src, mod.NAME, "sample", tally.to_dict())
        write_json(src / "_state.json", {"mode": "sample", "limit": limit, "complete": True, "finished": now_iso()})
        log.info("échantillon écrit : %d docs, %.1f Mo de texte, %.0f s", tally.docs, tally.bytes / 1e6, time.time() - t0)
        return stats

    if state.get("mode") == "full" and state.get("complete") and not force:
        log.info("collecte complète déjà faite (--force pour refaire)")
        return state
    work = src / "_work"
    if force and work.exists():
        shutil.rmtree(work)
    write_json(src / "_state.json", {"mode": "full", "complete": False, "started": now_iso()})
    ctx = mod.prepare(cfg, log, sample=False)
    units = mod.units(ctx)
    done = read_done(work)
    pending = [u for u in units if u not in done]
    log.info("collecte complète : %d unités, %d déjà faites, %d à faire, %d workers",
             len(units), len(done), len(pending), workers)
    args = [(mod.__name__, u, ctx, str(work)) for u in pending]
    n_done = len(done)
    if workers > 1 and len(args) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(args))) as pool:
            for unit, st in pool.imap_unordered(_run_unit, args):
                append_done(work, unit, st)
                done[unit] = st
                n_done += 1
                log.info("unité finie %s (%d/%d) : %d docs, %.1f Mo", unit, n_done, len(units), st["docs"], st["bytes"] / 1e6)
    else:
        for a in args:
            unit, st = _run_unit(a)
            append_done(work, unit, st)
            done[unit] = st
            n_done += 1
            log.info("unité finie %s (%d/%d) : %d docs, %.1f Mo", unit, n_done, len(units), st["docs"], st["bytes"] / 1e6)

    files = []
    for u in units:
        files.extend(sorted((work / safe_name(u)).glob("part-*.jsonl.zst")))
    replace_parts(src, files)
    merged = Tally.merge([done[u] for u in units])
    extra = mod.summarize(done, units) if hasattr(mod, "summarize") else None
    stats = write_stats(src, mod.NAME, "full", merged, extra)
    write_json(src / "_state.json", {"mode": "full", "complete": True, "finished": now_iso(), "units": len(units)})
    shutil.rmtree(work, ignore_errors=True)
    log.info("collecte complète finie : %d docs, %.1f Mo de texte, %d fichiers, %.0f s",
             merged["docs"], merged["bytes"] / 1e6, len(files), time.time() - t0)
    return stats
