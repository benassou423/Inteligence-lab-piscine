"""Project Gutenberg, livres en français, via un miroir officiel (mirror.cs.odu.edu).

Le site www.gutenberg.org interdit l'aspiration par robot : on lit le catalogue hors ligne
(pg_catalog.csv.gz) et les fichiers texte sur un miroir de la liste officielle (MIRRORS.ALL).
Le texte est gardé brut, avec l'en-tête et le pied de page Gutenberg (retirés au nettoyage)."""
import csv
import gzip
from concurrent.futures import ThreadPoolExecutor

import requests

from pipeline.collect.common import date_ok, download, http_get, make_doc, make_session, source_dir

NAME = "gutenberg_fr"
DOMAIN = "litteraire"
LICENSE = "Project Gutenberg License (domaine public aux États-Unis)"


def read_catalog(path, languages, log=None):
    """Livres de type Text dont une des langues est dans `languages`, triés par numéro."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    books = []
    for r in rows:
        langs = [x.strip() for x in (r.get("Language") or "").split(";") if x.strip()]
        if r.get("Type") != "Text" or not set(langs) & set(languages):
            continue
        books.append({"n": int(r["Text#"]), "issued": r.get("Issued") or None, "title": r.get("Title"),
                      "authors": r.get("Authors"), "languages": langs, "subjects": r.get("Subjects"),
                      "bookshelves": r.get("Bookshelves"), "locc": r.get("LoCC")})
    books.sort(key=lambda b: b["n"])
    if log:
        log.info("catalogue : %d livres en %s (type Text)", len(books), "/".join(languages))
    return books


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    dest = source_dir(NAME) / "_pg_catalog.csv.gz"
    if not dest.exists():
        download(make_session(), c["catalog"], dest, log=log)
    books = read_catalog(dest, c["languages"], log)
    size = int(c.get("batch_size", 100))
    batches = {f"books-{i // size:05d}": books[i:i + size] for i in range(0, len(books), size)}
    return {"mirror_epub": c["mirror_epub"].rstrip("/"), "mirror_main": c["mirror_main"].rstrip("/"),
            "batches": batches, "threads": int(c.get("threads", 4))}


def units(ctx):
    return sorted(ctx["batches"])


def candidate_urls(mirror_epub, mirror_main, n):
    """Fichier texte généré (UTF-8) puis anciens fichiers de la collection principale."""
    s = str(n)
    d = "/".join(s[:-1]) if len(s) > 1 else "0"
    base = f"{mirror_main}/{d}/{s}"
    return [f"{mirror_epub}/{s}/pg{s}.txt", f"{base}/{s}-0.txt", f"{base}/{s}-8.txt", f"{base}/{s}.txt"]


def decode_text(raw: bytes) -> str:
    """UTF-8 si possible, sinon Windows-1252 (sur-ensemble de Latin-1). BOM retiré, fins de ligne en \\n."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def fetch_book(session, ctx, n, log):
    for url in candidate_urls(ctx["mirror_epub"], ctx["mirror_main"], n):
        try:
            r = http_get(session, url, log=log, max_tries=5)
        except requests.HTTPError:
            continue  # 404 : on essaie le fichier suivant
        except RuntimeError as e:  # miroir indisponible après plusieurs essais : livre compté comme introuvable
            if log:
                log.warning("livre %d : %s", n, e)
            continue
        return url, r.content
    return None, None


def process(unit, ctx, tally, log):
    books = ctx["batches"][unit]
    session = make_session()

    def job(b):
        if not date_ok(b["issued"]):
            return b, None, None
        url, raw = fetch_book(session, ctx, b["n"], log)
        return b, url, raw

    with ThreadPoolExecutor(ctx["threads"]) as ex:
        for b, url, raw in ex.map(job, books):  # map garde l'ordre : sortie déterministe
            if not date_ok(b["issued"]):
                tally.skip("date_apres_gel")
                continue
            if raw is None:
                tally.skip("texte_introuvable")
                log.warning("livre %d : aucun fichier texte sur le miroir", b["n"])
                continue
            text = decode_text(raw)
            if not text.strip():
                tally.skip("texte_vide")
                continue
            meta = {k: b[k] for k in ("title", "authors", "languages", "subjects", "bookshelves", "locc")}
            meta["mirror_url"] = url
            yield make_doc(NAME, DOMAIN, str(b["n"]), text, url=f"https://www.gutenberg.org/ebooks/{b['n']}",
                           date=b["issued"], license=LICENSE, meta=meta)
