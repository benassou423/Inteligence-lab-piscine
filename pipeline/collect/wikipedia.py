"""Wikipédia FR : dump officiel découpé, lecture XML en flux, rendu local du wikitexte."""
from __future__ import annotations

import bz2
from collections import Counter
from contextlib import nullcontext
import datetime as dt
import hashlib
import logging
import math
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote

from pipeline.collect import common, wikitext as wt
from pipeline.collect.common import date_ok, download, http_get, make_doc, make_session, source_dir
from pipeline.collect.wikisource import iter_pages
from pipeline.collect.wikipedia_render import WikipediaRenderer
from pipeline.io import read_docs, write_docs

NAME = "wikipedia_fr"
DOMAIN = "encyclopedique"
LICENSE = "CC BY-SA 4.0"
PART = re.compile(r"frwiki-(\d{8})-pages-articles-multistream\d+\.xml-p(\d+)p(\d+)\.bz2$")
DISAMBIG = re.compile(r"\{\{\s*(?:homonymie|homonymie de [^|}]+|désambiguïsation|disambig|disambiguation|patronymie|toponymie)(?=\s*[|}])", re.I)
REDIRECT = re.compile(r"^\s*(?:<!--.*?-->\s*)*#(?:redirect|redirection)\b", re.I | re.S)


def trim_empty_fields(text):
    """Enlève uniquement les champs de liste déjà vides dans la source, sans effacer de phrase."""
    lines = text.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines)
                   if not (re.fullmatch(r"\* [^:\n]{1,80}:[ \t]*\n?", line)
                           and (i + 1 == len(lines) or not lines[i + 1].startswith("**"))))


def part_files(base, dump, session, log):
    """Uniquement les parties XML multistream ; jamais l'index ni le fichier recombiné."""
    status = http_get(session, f"{base}/{dump}/dumpstatus.json", log=log).json()
    job = status["jobs"]["articlesmultistreamdump"]
    if job.get("status") != "done":
        raise RuntimeError(f"Dump frwiki-{dump} incomplet : {job.get('status')}")
    parts = []
    for name, info in job["files"].items():
        match = PART.fullmatch(name)
        if match and match[1] == dump:
            parts.append({"name": name, "url": f"{base}/{dump}/{name}", "size": int(info["size"]),
                          "sha1": info["sha1"], "first_page": int(match[2])})
    if not parts:
        raise RuntimeError("Aucune partie XML multistream trouvée")
    return sorted(parts, key=lambda p: p["first_page"])


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    dump, base = str(c["dump"]), c["base_url"].rstrip("/")
    cutoff = dt.datetime.strptime(dump, "%Y%m%d").date().isoformat()
    if not date_ok(cutoff):
        raise ValueError("Le dump dépasse le gel des données")
    parts = part_files(base, dump, make_session(), log)
    src = source_dir(NAME)
    common.write_json(src / "_dump_manifest.json", {"dump": dump, "parts": parts})
    budget = int(float(cfg.get("raw_budget_gb", 30)) * 1e9)
    used = sum(p.stat().st_size for p in common.RAW_DIR.rglob("*") if p.is_file())
    download_bytes = sum(p["size"] for p in parts)
    # Réserve prudente : tous les bz2 + 8 Go de sortie. Les bz2 seront libérés par partie.
    reserve = download_bytes + 8_000_000_000
    if not sample and (used + reserve > budget or shutil.disk_usage(src).free < reserve):
        raise RuntimeError(f"Place insuffisante : {used / 1e9:.1f} Go utilisés, réserve {reserve / 1e9:.1f} Go, budget {budget / 1e9:.1f} Go")
    sample_limit = int(cfg.get("_sample_limit", 2000))
    return {"dump": dump, "cutoff": cutoff, "parts": {p["name"]: p for p in parts},
            "sample": sample, "sample_per_part": math.ceil(sample_limit / len(parts)),
            "min_chars": int(c.get("min_chars", 1)), "download_lock": cfg.get("_download_lock")}


def units(ctx):
    return list(ctx["parts"])


def article_doc(pid, title, ts, text, ctx, renderer=None):
    """Garde la dernière révision antérieure au 2 octobre ; aucune date inventée."""
    if REDIRECT.match(text):
        return None, "redirection"
    if DISAMBIG.search(text):
        return None, "homonymie"
    try:
        timestamp = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Horodatage sans fuseau")
        date = timestamp.astimezone(dt.timezone.utc).date().isoformat()
    except (ValueError, TypeError):
        return None, "date_invalide"
    cutoff = ctx.get("cutoff", dt.datetime.strptime(ctx["dump"], "%Y%m%d").date().isoformat())
    if not date_ok(date) or date > cutoff:
        return None, "revision_apres_dump"
    renderer = renderer or WikipediaRenderer()
    renderer.population = renderer.population_year = ""
    body = wt.to_text(trim_empty_fields(text), renderer)
    body = wt.tidy(re.sub(r"(?m)^[ \t.,;:·–—-]+$", "", body))  # ponctuation seule après retrait des références
    if len(body) < ctx["min_chars"]:
        return None, "texte_vide_ou_court"
    meta = {"title": title, "dump": f"frwiki-{ctx['dump']}", "last_revision": ts,
            "date_source": "derniere_revision"}
    url = "https://fr.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:,()'’")
    # Le contrat commun attend AAAA-MM-JJ ; l'horodatage complet est conservé dans meta.
    return make_doc(NAME, DOMAIN, pid, body, url=url, date=date, license=LICENSE, meta=meta), None


def _sample_pages(unit, ctx, log):
    """Cache de wikitexte pour rejouer le même essai sans recontacter Wikipédia."""
    cache = source_dir(NAME) / "_sample_input" / f"{unit}.jsonl.zst"
    if cache.exists():
        yield from read_docs(cache)
        return
    part = ctx["parts"][unit]
    # On ne télécharge que le début nécessaire de chaque partie (connexion fermée au seuil).
    with make_session() as session:
        with http_get(session, part["url"], stream=True, log=log) as response:
            response.raw.decode_content = True
            with bz2.BZ2File(response.raw, "rb") as stream:
                for ns, pid, title, ts, text in iter_pages(stream):
                    if ns == 0:
                        yield {"pid": pid, "title": title, "ts": ts, "wikitext": text}


def process(unit, ctx, tally, log):
    renderer = WikipediaRenderer()
    counts = Counter(tally.extra.get("unknown_templates", {}))
    dest = source_dir(NAME) / "_dl" / unit
    sample_records = []
    n = 0

    def render(records):
        nonlocal n
        for rec in records:
            doc, why = article_doc(rec["pid"], rec["title"], rec["ts"], rec["wikitext"], ctx, renderer)
            if doc is None:
                tally.skip(why)
                renderer.unknown.clear()
                continue
            if ctx["sample"]:
                sample_records.append(rec)
            doc["meta"]["dump_part"] = unit
            # Mise à jour avant yield : le lanceur peut s'arrêter au 2 000e document.
            if renderer.unknown:
                doc["meta"]["unknown_templates"] = dict(renderer.unknown)
            counts.update(renderer.unknown)
            tally.extra["unknown_templates"] = counts
            renderer.unknown.clear()
            n += 1
            yield doc
            if ctx["sample"] and n >= ctx["sample_per_part"]:
                return

    if ctx["sample"]:
        cache = source_dir(NAME) / "_sample_input" / f"{unit}.jsonl.zst"
        try:
            yield from render(_sample_pages(unit, ctx, log))
        finally:
            if not cache.exists() and sample_records:
                tmp = cache.with_name(cache.name + ".tmp")
                write_docs(tmp, sample_records)
                tmp.replace(cache)
        return
    part = ctx["parts"][unit]
    with ctx.get("download_lock") or nullcontext():
        download(make_session(), part["url"], dest, log=log)
    with dest.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha1").hexdigest()
    if dest.stat().st_size != part["size"] or checksum != part["sha1"]:
        raise RuntimeError(f"Taille/empreinte officielle incorrecte : {dest}")
    with bz2.open(dest, "rb") as stream:
        records = ({"pid": pid, "title": title, "ts": ts, "wikitext": text}
                   for ns, pid, title, ts, text in iter_pages(stream) if ns == 0)
        yield from render(records)
    dest.unlink()  # seulement après lecture complète réussie


def summarize(done, names):
    counts = Counter()
    for name in names:
        counts.update(done[name].get("extra", {}).get("unknown_templates", {}))
    return {"unknown_templates": dict(counts.most_common())}


def run(cfg, *, limit=None, workers=None, force=False):
    """Le lanceur commun garde reprise et processus ; deux téléchargements simultanés maximum."""
    log = common.get_logger(NAME)
    if not sys.stderr.isatty():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)  # le journal de nohup reçoit aussi la progression
    cfg = dict(cfg)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit doit être positif")
        cfg["_sample_limit"] = limit
        return common.run_source(sys.modules[__name__], cfg, limit=limit, workers=workers, force=force)
    with mp.Manager() as manager:
        cfg["_download_lock"] = manager.BoundedSemaphore(2)
        return common.run_source(sys.modules[__name__], cfg, workers=workers, force=force)
