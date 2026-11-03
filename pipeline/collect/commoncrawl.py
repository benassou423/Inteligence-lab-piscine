"""Common Crawl : pages françaises lues en flux dans les fichiers WARC (rien n'est stocké sur disque).

Une page est gardée si les langues cld2 de son enregistrement `metadata` contiennent le français
(à n'importe quel rang) ou si son domaine finit par .fr. Les pages qui réservent la fouille de
textes (TDMRep : en-tête HTTP ou <meta name="tdm-reservation" content="1">) sont écartées et comptées."""
import gzip
import json
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import requests
import urllib3
from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from pipeline.collect import common, htmltext
from pipeline.collect.common import date_ok, download, http_get, make_doc, make_session, source_dir

NAME = "commoncrawl"
DOMAIN = "web"
LICENSE = "Common Crawl Terms of Use ; droits des pages réservés à leurs auteurs (exception TDM, art. L122-5-3 CPI)"
DATA_URL = "https://data.commoncrawl.org/"
COLLINFO = "https://index.commoncrawl.org/collinfo.json"
HTML_TYPES = ("text/html", "application/xhtml+xml")
NET_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, OSError, EOFError, zlib.error, ArchiveLoadFailed)


def latest_crawl_before(freeze, log=None):
    """Dernier crawl dont la fin (`to`) précède la date de gel."""
    crawls = make_session().get(COLLINFO, timeout=60).json()
    ok = [c for c in crawls if c.get("to", "9999")[:10] < freeze]
    best = max(ok, key=lambda c: c["to"])
    if log:
        log.info("collinfo : dernier crawl fini avant le %s = %s (%s → %s)", freeze, best["id"], best["from"], best["to"])
    return best["id"]


def pick_warc_files(paths, n):
    """WARC retenus, déterministes : ordre trié de warc.paths.gz, premier segment, n premiers fichiers.

    Renvoie (fichiers, premier segment, nb de fichiers du segment, nb total)."""
    paths = sorted(p.strip() for p in paths if p.strip())
    first_segment = paths[0].split("/")[3]
    seg = [p for p in paths if p.split("/")[3] == first_segment]
    return seg[:n], first_segment, len(seg), len(paths)


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    crawl = c.get("crawl")
    try:
        # common.FREEZE_DATE lu ici (et non importé) : c'est la valeur de la config chargée par load_config
        auto = latest_crawl_before(common.FREEZE_DATE, log)
        if crawl and crawl != auto:
            log.warning("crawl figé dans la config (%s) ≠ dernier crawl avant le gel (%s)", crawl, auto)
        crawl = crawl or auto
    except (requests.RequestException, ValueError) as e:
        if not crawl:
            raise
        log.warning("collinfo.json illisible (%s) : on garde %s", e, crawl)
    dest = source_dir(NAME) / f"_{crawl}.warc.paths.gz"
    if not dest.exists():
        download(make_session(), f"{DATA_URL}crawl-data/{crawl}/warc.paths.gz", dest, log=log)
    with gzip.open(dest, "rt") as f:
        files, first_segment, n_seg, n_all = pick_warc_files(f, int(c.get("n_warc_files", 10)))
    log.info("%s : %d fichiers WARC, premier segment %s (%d fichiers), on en prend %d",
             crawl, n_all, first_segment, n_seg, len(files))
    return {"crawl": crawl, "files": files, "keep_tld": c.get("keep_tld", ".fr")}


def units(ctx):
    return ctx["files"]


def parse_metadata(raw: bytes) -> dict:
    """Champs « clé: valeur » d'un enregistrement metadata (dont languages-cld2 en JSON)."""
    out = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        k, sep, v = line.partition(":")
        if sep:
            out[k.strip()] = v.strip()
    if "languages-cld2" in out:
        try:
            out["languages-cld2"] = json.loads(out["languages-cld2"])
        except ValueError:
            out["languages-cld2"] = {}
    return out


def host_of(url):
    try:
        return (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def _decide(p, meta, ctx, tally, warc_name):
    """Document à garder, ou None (la raison est comptée)."""
    if p.get("skip"):
        tally.skip(p["skip"])
        return None
    cld2 = meta.get("languages-cld2") or {}
    langs = [x.get("code") for x in cld2.get("languages", [])]
    by_lang = "fr" in langs
    by_tld = p["host"].endswith(ctx["keep_tld"])
    if not (by_lang or by_tld):
        tally.skip("pas_francais")
        return None
    date = p["date"]
    if not date_ok(date):
        tally.skip("date_apres_gel")
        return None
    html, charset = htmltext.decode_html(p["payload"], p["content_type"], meta.get("charset-detected"))
    if htmltext.tdm_reserved(html, p["http_headers"]):
        tally.skip("tdm_reservation")
        return None
    root = htmltext.parse_html(html)
    text = htmltext.visible_text(root=root)
    if not text:
        tally.skip("texte_vide")
        return None
    kept_by = "+".join(k for k, v in (("langue", by_lang), ("tld", by_tld)) if v)
    tally.extra["kept_by_" + kept_by] = tally.extra.get("kept_by_" + kept_by, 0) + 1
    m = {"warc_file": warc_name, "cld2_languages": [{"code": x.get("code"), "text_covered": x.get("text-covered")}
                                                     for x in cld2.get("languages", [])],
         "cld2_reliable": cld2.get("reliable"), "content_type": p["content_type"], "charset": charset,
         "kept_by": kept_by, "host": p["host"], "title": htmltext.page_title(root)}
    if p["truncated"]:
        m["warc_truncated"] = p["truncated"]
    return make_doc(NAME, DOMAIN, p["rid"].strip("<>"), text, url=p["url"], date=date, license=LICENSE,
                    meta=m, html=html)


class WarcScanner:
    """Parcourt les enregistrements d'un WARC (request → response → metadata) et produit les documents.

    Garde l'état utile à une reprise : `resume` = début du dernier enregistrement « request » vu,
    `decided` = réponses déjà traitées (pas de doublon si on relit un bout du fichier)."""

    def __init__(self, ctx, tally, warc_name):
        self.ctx, self.tally, self.warc_name = ctx, tally, warc_name
        self.resume = 0
        self.decided = set()
        self.n = {"records": 0, "responses": 0, "html": 0, "kept": 0}

    def scan(self, stream, base=0):
        """Documents gardés du flux `stream` (qui commence à l'octet `base` du fichier)."""
        pending = None
        it = ArchiveIterator(stream)
        for rec in it:
            self.n["records"] += 1
            kind = rec.rec_type
            if kind == "request":
                self.resume = base + it.get_record_offset()
                pending = None
            elif kind == "response":
                pending = self._response(rec)
            elif kind == "metadata" and pending is not None:
                if rec.rec_headers.get_header("WARC-Concurrent-To") != pending["rid"]:
                    continue
                doc = _decide(pending, parse_metadata(rec.content_stream().read()), self.ctx, self.tally,
                              self.warc_name)
                self.decided.add(pending["rid"])
                pending = None
                if doc is not None:
                    self.n["kept"] += 1
                    yield doc

    def _response(self, rec):
        rid = rec.rec_headers.get_header("WARC-Record-ID")
        if rid in self.decided:
            return None
        self.n["responses"] += 1
        uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
        hh = rec.http_headers
        ctype = hh.get_header("Content-Type") if hh else None
        status = hh.get_statuscode() if hh else None
        ident = (rec.rec_headers.get_header("WARC-Identified-Payload-Type") or "").lower()
        p = {"rid": rid, "url": uri, "host": host_of(uri), "content_type": ctype,
             "date": (rec.rec_headers.get_header("WARC-Date") or "")[:10],
             "truncated": rec.rec_headers.get_header("WARC-Truncated"),
             "http_headers": hh.headers if hh else []}
        if status != "200":
            p["skip"] = "statut_http_non_200"
        elif not (any(t in (ctype or "").lower() for t in HTML_TYPES) or ident in HTML_TYPES):
            p["skip"] = "pas_html"
        else:
            self.n["html"] += 1
            p["payload"] = rec.content_stream().read()
        return p


def process(unit, ctx, tally, log):
    """Lit un fichier WARC en flux HTTP ; reprise par requête Range après une coupure réseau."""
    url = DATA_URL + unit
    warc_name = Path(unit).name
    session = make_session()
    scanner = WarcScanner(ctx, tally, warc_name)
    total_size = None
    tries = 0
    t0 = time.time()
    while True:
        base = scanner.resume
        r = http_get(session, url, stream=True, headers={"Range": f"bytes={base}-"} if base else None, log=log)
        if base and r.status_code != 206:
            r.close()
            raise RuntimeError(f"le serveur ignore Range pour {url}")
        if total_size is None:
            total_size = int(r.headers.get("Content-Length", 0)) + base
        try:
            yield from scanner.scan(r.raw, base)
            break  # fin du fichier
        except NET_ERRORS as e:
            tries += 1
            if tries > 10:
                raise
            wait = min(300, 5 * 2 ** tries)
            log.warning("%s : coupure (%r) ; reprise à l'octet %d dans %d s", warc_name, e, scanner.resume, wait)
            time.sleep(wait)
        finally:
            r.close()
    secs = time.time() - t0
    mb = (total_size or 0) / 1e6
    n = scanner.n
    tally.extra.update({"warc_file": warc_name, "warc_mb": round(mb, 1), "seconds": round(secs, 1),
                        "mb_per_s": round(mb / secs, 2) if secs else None, "reconnects": tries, **n,
                        "tdm_reservation": tally.removed.get("tdm_reservation", 0)})
    log.info("%s : %.0f Mo en %.0f s (%.1f Mo/s), %d réponses, %d HTML, %d pages gardées, %d réserves TDM",
             warc_name, mb, secs, mb / secs if secs else 0, n["responses"], n["html"], n["kept"],
             tally.removed.get("tdm_reservation", 0))


def summarize(done, units_):
    """Débit et pages gardées par fichier WARC (pour _stats.json et la fiche)."""
    per_file = [done[u]["extra"] for u in units_ if done.get(u, {}).get("extra")]
    tot_mb = sum(x.get("warc_mb", 0) for x in per_file)
    tot_s = sum(x.get("seconds", 0) for x in per_file)
    return {"per_warc": per_file, "total_warc_mb": round(tot_mb, 1),
            "mean_mb_per_s_per_download": round(tot_mb / tot_s, 2) if tot_s else None,
            "pages_kept_per_file": round(sum(x.get("kept", 0) for x in per_file) / max(1, len(per_file)), 1)}
