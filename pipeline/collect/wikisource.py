"""Wikisource en français : dump officiel frwikisource (pages-articles, XML bz2).

Pourquoi pas le dataset HF wikimedia/wikisource (20231201.fr) ? Il ne contient que l'espace principal,
dont la plupart des pages ne font que transclure l'espace « Page: » : 23 201 pages, texte médian de
146 caractères (liens interlangues). Le vrai texte relu est dans l'espace Page: du dump.

Deux phases (reprise possible à chaque unité) :
1. par fichier du dump : pages « Page:<livre>/<n> » relues (niveau ≥ 3) converties en texte et rangées
   dans 64 seaux selon le livre ; pages de l'espace principal qui ont leur propre texte → documents ;
   métadonnées des livres (espace « Livre: ») ;
2. par seau : pages regroupées par livre, triées par numéro, recollées → un document par livre."""
import bz2
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import mwparserfromhell as mwp
import zstandard
from lxml import etree

from pipeline.collect import wikitext as wt
from pipeline.collect.common import (Tally, append_done, date_ok, download, get_logger, http_get, load_config,
                                     make_doc, make_session, now_iso, read_done, read_json, replace_parts,
                                     safe_name, source_dir, write_json, write_parts, write_stats)
from pipeline.io import read_docs

NAME = "wikisource_fr"
DOMAIN = "litteraire"
LICENSE = "Domaine public (œuvres) ; transcription Wikisource CC BY-SA 4.0"
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
SKIP_NS0 = re.compile(r"<pages\b|\{\{\s*page\s*:|\{\{\s*(liste documents|homonymie|éditions|voir éditions)", re.I)
INDEX_FIELDS = {"Titre": "title", "Auteur": "author", "Annee": "year", "Type": "type", "Volume": "volume",
                "Editeur": "publisher", "Lieu": "place", "Traducteur": "translator"}


def part_files(base_url, dump, session, log):
    """Fichiers « pages-articlesN.xml-pXpY.bz2 » du dump (dumpstatus.json), triés."""
    st = http_get(session, f"{base_url}/{dump}/dumpstatus.json", log=log).json()
    job = st["jobs"]["articlesdump"]
    if job.get("status") != "done":
        raise RuntimeError(f"dump {dump} incomplet : {job.get('status')}")
    files = job["files"]

    def first_page(name):
        m = re.search(r"-p(\d+)p\d+\.bz2$", name)
        return int(m.group(1)) if m else 0

    return [{"name": n, "url": f"{base_url}/{dump}/{n}", "size": files[n].get("size")}
            for n in sorted(files, key=first_page)]


def bucket_of(index: str, n_buckets: int) -> int:
    return int(hashlib.md5(index.encode("utf-8")).hexdigest(), 16) % n_buckets


def split_page_title(title: str):
    """« Page:Livre.djvu/12 » → ("Livre.djvu", 12) ; page sans numéro → (titre, 0)."""
    name = title.split(":", 1)[1] if ":" in title else title
    idx, sep, num = name.rpartition("/")
    if sep and num.isdigit():
        return idx, int(num)
    return name, 0


def index_meta(wikitext: str) -> dict:
    """Champs utiles du modèle Proofreadpage_index_template d'une page Livre:."""
    out = {}
    try:
        code = mwp.parse(wikitext)
    except Exception:  # noqa: BLE001
        return out
    r = wt.Renderer()
    for tpl in code.filter_templates(recursive=False):
        if "proofreadpage_index_template" not in str(tpl.name).lower():
            continue
        for p in tpl.params:
            key = str(p.name).strip()
            if key in INDEX_FIELDS:
                v = wt.tidy(r.render(p.value)).replace("\n", " ").strip()
                if v:
                    out[INDEX_FIELDS[key]] = v
    return out


def iter_pages(fileobj):
    """(ns, id, titre, horodatage, wikitexte) pour chaque page du dump XML."""
    for _, el in etree.iterparse(fileobj, events=("end",), tag=f"{MW_NS}page"):
        ns = int(el.findtext(f"{MW_NS}ns") or -1)
        rev = el.find(f"{MW_NS}revision")
        if rev is not None and ns in (0, 104, 112):
            yield (ns, el.findtext(f"{MW_NS}id"), el.findtext(f"{MW_NS}title"),
                   rev.findtext(f"{MW_NS}timestamp") or "", rev.findtext(f"{MW_NS}text") or "")
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]


def page_url(title: str) -> str:
    return "https://fr.wikisource.org/wiki/" + quote(title.replace(" ", "_"), safe="/:,()'’")


def ns0_doc(pid, title, ts, text, dump, min_chars):
    """Document d'une page de l'espace principal qui porte son propre texte, ou (None, raison)."""
    head = text.lstrip()[:40].lower()
    if head.startswith("#redirect"):
        return None, "ns0_redirection"
    if SKIP_NS0.search(text):
        return None, "ns0_transclusion_ou_liste"
    body = wt.to_text(text)
    if len(body) < min_chars:
        return None, "ns0_trop_court"
    date = ts[:10]
    if not date_ok(date):
        return None, "date_apres_gel"
    meta = {"title": title, "dump": f"frwikisource-{dump}", "kind": "page", "is_theatre": wt.is_theatre(body)}
    return make_doc(NAME, DOMAIN, f"page:{pid}", body, url=page_url(title), date=date, license=LICENSE, meta=meta), None


# ---------------------------------------------------------------- phase 1

class Buckets:
    """Seaux JSONL compressés zstd. Un compresseur PAR seau : un même ZstdCompressor partagé entre
    plusieurs flux ouverts en même temps mélange leurs trames (fichiers illisibles)."""

    def __init__(self, out_dir, n):
        self.out_dir, self.n, self.w = Path(out_dir), n, {}

    def write(self, b, rec):
        if b not in self.w:
            f = open(self.out_dir / f"b{b:03d}.jsonl.zst", "wb")
            self.w[b] = zstandard.ZstdCompressor(level=3).stream_writer(f)  # ferme aussi f à la fin
        self.w[b].write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    def close(self):
        for w in self.w.values():
            w.close()
        self.w = {}


def phase1(args):
    """Convertit un fichier du dump : seaux de pages Page:, documents de l'espace principal, index Livre:."""
    path, out_dir, ctx = args
    load_config(ctx.get("_config_path"))
    log = get_logger(NAME)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    tally = Tally()
    nb = ctx["buckets"]
    buckets = Buckets(out_dir, nb)

    ns0_docs = []
    index_out = open(out_dir / "index.jsonl", "w", encoding="utf-8")
    renderer = wt.Renderer()
    t0, n = time.time(), 0
    with bz2.open(path, "rb") as f:
        for ns, pid, title, ts, text in iter_pages(f):
            n += 1
            if n % 200_000 == 0:
                log.info("%s : %d pages lues (%.0f s)", Path(path).name, n, time.time() - t0)
            if ns == 104:
                idx, num = split_page_title(title)
                level, body = wt.page_body(text)
                rec = {"i": idx, "n": num, "q": level, "ts": ts[:10]}
                if level is None or level < ctx["min_quality"]:
                    tally.skip(f"page_niveau_{level}")
                else:
                    rec["t"] = wt.to_text(body, renderer)
                    tally.skip("page_gardee", 0)
                buckets.write(bucket_of(idx, nb), rec)
            elif ns == 112:
                meta = index_meta(text)
                name = title.split(":", 1)[1] if ":" in title else title
                index_out.write(json.dumps({"i": name, "id": pid, **meta}, ensure_ascii=False) + "\n")
            elif ns == 0:
                doc, why = ns0_doc(pid, title, ts, text, ctx["dump"], ctx["min_chars"])
                if doc is None:
                    tally.skip(why)
                else:
                    ns0_docs.append(doc)
    buckets.close()
    index_out.close()
    tally2 = Tally()
    for d in ns0_docs:
        tally2.add(d)
    write_parts(iter(ns0_docs), out_dir / "ns0")
    st = tally.to_dict()
    st.update({"docs": tally2.docs, "bytes": tally2.bytes, "date_min": tally2.date_min, "date_max": tally2.date_max,
               "pages_read": n, "seconds": round(time.time() - t0, 1),
               "unknown_templates": dict(sorted(renderer.unknown.items(), key=lambda x: -x[1])[:40])})
    log.info("%s fini : %d pages, %d documents de l'espace principal, %.0f s", Path(path).name, n, tally2.docs,
             time.time() - t0)
    return st


# ---------------------------------------------------------------- phase 2

def load_indexes(p1_dirs):
    idx = {}
    for d in p1_dirs:
        f = Path(d) / "index.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    idx[rec["i"]] = rec
    return idx


def read_bucket_file(path):
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f, read_across_frames=True) as r:
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


def book_docs(pages_by_index, indexes, ctx, tally):
    """Un document par livre : pages relues triées par numéro et recollées."""
    for idx in sorted(pages_by_index):
        pages = sorted(pages_by_index[idx], key=lambda r: r["n"])
        kept = [p for p in pages if p.get("t")]
        skipped = len(pages) - len(kept)
        if not kept:
            tally.skip("livre_sans_page_relue")
            continue
        text = wt.tidy(wt.join_pages(p["t"] for p in kept))
        if len(text) < ctx["min_chars"]:
            tally.skip("livre_trop_court")
            continue
        date = max(p["ts"] for p in kept)
        if not date_ok(date):
            tally.skip("date_apres_gel")
            continue
        info = indexes.get(idx, {})
        meta = {"index": idx, "kind": "livre", "dump": f"frwikisource-{ctx['dump']}", "n_pages": len(kept),
                "n_pages_skipped_quality": skipped, "is_theatre": wt.is_theatre(text)}
        for k in ("title", "author", "year", "type", "volume", "publisher", "translator"):
            if info.get(k):
                meta[k] = info[k]
        yield make_doc(NAME, DOMAIN, f"livre:{idx}", text, url=page_url("Livre:" + idx), date=date,
                       license=LICENSE, meta=meta)


def phase2(args):
    """Regroupe un seau : documents par livre écrits dans out_dir/part-*.jsonl.zst."""
    b, p1_dirs, out_dir, ctx = args
    load_config(ctx.get("_config_path"))
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    indexes = load_indexes(p1_dirs)
    pages = {}
    for d in p1_dirs:
        f = Path(d) / f"b{b:03d}.jsonl.zst"
        if f.exists():
            for rec in read_bucket_file(f):
                pages.setdefault(rec["i"], []).append(rec)
    tally = Tally()

    def gen():
        for doc in book_docs(pages, indexes, ctx, tally):
            tally.add(doc)
            yield doc

    write_parts(gen(), out_dir)
    return b, tally.to_dict()


# ---------------------------------------------------------------- lancement

def _ctx(cfg):
    c = cfg["sources"][NAME]
    return {"dump": str(c["dump"]), "base_url": c["base_url"].rstrip("/"), "min_quality": int(c["min_page_quality"]),
            "min_chars": int(c["min_chars"]), "buckets": int(c["buckets"]), "workers": int(c.get("workers", 4)),
            "parallel_downloads": int(c.get("parallel_downloads", 2))}


def run_sample(cfg, limit, log):
    """Échantillon : lecture en flux du premier fichier du dump, regroupement en mémoire."""
    ctx = _ctx(cfg)
    src = source_dir(NAME)
    session = make_session()
    files = part_files(ctx["base_url"], ctx["dump"], session, log)
    r = http_get(session, files[0]["url"], stream=True, log=log)
    r.raw.decode_content = False
    tally = Tally()
    pages, indexes, ns0 = {}, {}, []
    renderer = wt.Renderer()
    with bz2.open(r.raw, "rb") as f:
        for ns, pid, title, ts, text in iter_pages(f):
            if ns == 104:
                idx, num = split_page_title(title)
                level, body = wt.page_body(text)
                rec = {"i": idx, "n": num, "q": level, "ts": ts[:10]}
                if level is not None and level >= ctx["min_quality"]:
                    rec["t"] = wt.to_text(body, renderer)
                else:
                    tally.skip(f"page_niveau_{level}")
                pages.setdefault(idx, []).append(rec)
            elif ns == 112:
                name = title.split(":", 1)[1]
                indexes[name] = {"i": name, **index_meta(text)}
            elif ns == 0 and len(ns0) < limit // 2:
                doc, why = ns0_doc(pid, title, ts, text, ctx["dump"], ctx["min_chars"])
                if doc:
                    ns0.append(doc)
                else:
                    tally.skip(why)
            if len(ns0) >= limit // 2 and sum(1 for v in pages.values() if any(p.get("t") for p in v)) >= limit:
                break
    r.close()
    books = list(book_docs(pages, indexes, ctx, tally))
    docs = (ns0 + books)[:limit]
    for d in docs:
        d["meta"]["sample_partial"] = True  # livres possiblement incomplets (lecture partielle du dump)
        tally.add(d)
    tmp = src / "_sample_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    replace_parts(src, write_parts(iter(docs), tmp))
    shutil.rmtree(tmp, ignore_errors=True)
    stats = write_stats(src, NAME, "sample", tally.to_dict(),
                        {"unknown_templates": dict(sorted(renderer.unknown.items(), key=lambda x: -x[1])[:40])})
    write_json(src / "_state.json", {"mode": "sample", "limit": limit, "complete": True, "finished": now_iso()})
    log.info("échantillon écrit : %d docs (%d pages principales, %d livres), %.1f Mo", tally.docs,
             min(len(ns0), limit), max(0, len(docs) - len(ns0)), tally.bytes / 1e6)
    return stats


def run(cfg, limit=None, workers=None, force=False):
    import multiprocessing as mp

    log = get_logger(NAME)
    src = source_dir(NAME)
    state = read_json(src / "_state.json", {}) or {}
    if limit is not None:
        if state.get("mode") == "full" and state.get("complete") and not force:
            log.warning("collecte complète déjà présente : échantillon ignoré (utiliser --force)")
            return state
        return run_sample(cfg, limit, log)
    if state.get("mode") == "full" and state.get("complete") and not force:
        log.info("collecte complète déjà faite (--force pour refaire)")
        return state
    ctx = _ctx(cfg)
    workers = workers or ctx["workers"]
    work = src / "_work"
    if force and work.exists():
        shutil.rmtree(work)
    write_json(src / "_state.json", {"mode": "full", "complete": False, "started": now_iso()})
    session = make_session()
    files = part_files(ctx["base_url"], ctx["dump"], session, log)
    done = read_done(work)
    t0 = time.time()
    log.info("dump frwikisource-%s : %d fichiers, %.1f Go compressés", ctx["dump"], len(files),
             sum(f["size"] or 0 for f in files) / 1e9)

    # phase 1 : téléchargements (2 à la fois) enchaînés avec les conversions (1 processus par fichier)
    p1_dirs = [work / "p1" / safe_name(f["name"]) for f in files]
    todo = [(f, d) for f, d in zip(files, p1_dirs) if f"p1:{f['name']}" not in done]
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool, \
                ThreadPoolExecutor(ctx["parallel_downloads"]) as dl:
            def fetch(f):
                dest = work / "dl" / f["name"]
                download(make_session(), f["url"], dest, log=log)
                return dest

            futs = [(f, d, dl.submit(fetch, f)) for f, d in todo]
            results = []
            for f, d, fut in futs:
                dest = fut.result()
                log.info("téléchargé %s (%.0f Mo)", f["name"], dest.stat().st_size / 1e6)
                results.append((f, dest, pool.apply_async(phase1, ((str(dest), str(d), ctx),))))
            for f, dest, res in results:
                st = res.get()
                append_done(work, f"p1:{f['name']}", st)
                done[f"p1:{f['name']}"] = st
                dest.unlink(missing_ok=True)
                log.info("phase 1 finie pour %s", f["name"])

    # phase 2 : un seau à la fois par worker
    todo2 = [b for b in range(ctx["buckets"]) if f"p2:{b:03d}" not in done]
    log.info("phase 2 : %d seaux à regrouper", len(todo2))
    args = [(b, [str(d) for d in p1_dirs], str(work / "p2" / f"b{b:03d}"), ctx) for b in todo2]
    with mp.get_context("spawn").Pool(min(workers, max(1, len(args)))) as pool:
        for b, st in pool.imap_unordered(phase2, args):
            append_done(work, f"p2:{b:03d}", st)
            done[f"p2:{b:03d}"] = st

    # finalisation : pages principales (par fichier du dump), puis livres (par seau)
    out = []
    for d in p1_dirs:
        out.extend(sorted((d / "ns0").glob("part-*.jsonl.zst")))
    for b in range(ctx["buckets"]):
        out.extend(sorted((work / "p2" / f"b{b:03d}").glob("part-*.jsonl.zst")))
    replace_parts(src, out)
    keys = [f"p1:{f['name']}" for f in files] + [f"p2:{b:03d}" for b in range(ctx["buckets"])]
    merged = Tally.merge([done[k] for k in keys])
    unknown = {}
    for f in files:
        for k, v in done[f"p1:{f['name']}"].get("unknown_templates", {}).items():
            unknown[k] = unknown.get(k, 0) + v
    extra = {"pages_read": sum(done[f"p1:{f['name']}"].get("pages_read", 0) for f in files),
             "unknown_templates_top": dict(sorted(unknown.items(), key=lambda x: -x[1])[:40])}
    stats = write_stats(src, NAME, "full", merged, extra)
    write_json(src / "_state.json", {"mode": "full", "complete": True, "finished": now_iso()})
    shutil.rmtree(work, ignore_errors=True)
    log.info("collecte complète finie : %d docs, %.1f Mo de texte, %d fichiers, %.0f s",
             merged["docs"], merged["bytes"] / 1e6, len(out), time.time() - t0)
    return stats


# accès en lecture pour les tests et le débogage
def read_part(path):
    return list(read_docs(path))
