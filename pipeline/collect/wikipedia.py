"""Wikipédia en français : dataset Hugging Face wikimedia/wikipedia (fichiers parquet)."""
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.collect.common import date_ok, download, make_doc, make_session, source_dir

NAME = "wikipedia_fr"
DOMAIN = "encyclopedique"
LICENSE = "CC BY-SA 4.0"


def _hf_api_files(repo, revision, config):
    """Liste triée des fichiers parquet de la config (API Hugging Face)."""
    s = make_session()
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{config}"
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return sorted(x["path"] for x in r.json() if x["path"].endswith(".parquet"))


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    files = _hf_api_files(c["repo"], c["revision"], c["config"])
    log.info("%s %s@%s : %d fichiers parquet", c["repo"], c["config"], c["revision"][:10], len(files))
    return {"repo": c["repo"], "revision": c["revision"], "config": c["config"], "dump_date": c["dump_date"],
            "files": files, "sample": sample, "batch": int(c.get("batch_size", 1000))}


def units(ctx):
    return ctx["files"]


def _docs_from_table(pf, ctx, tally, row_groups=None):
    date = ctx["dump_date"]
    for batch in pf.iter_batches(batch_size=ctx["batch"], columns=["id", "url", "title", "text"], row_groups=row_groups):
        for r in batch.to_pylist():
            text = r["text"] or ""
            if not text.strip():
                tally.skip("texte_vide")
                continue
            if not date_ok(date):
                tally.skip("date_apres_gel")
                continue
            yield make_doc(NAME, DOMAIN, r["id"], text, url=r["url"], date=date, license=LICENSE,
                           meta={"title": r["title"], "hf_dataset": f"{ctx['repo']}/{ctx['config']}",
                                 "hf_revision": ctx["revision"]})


def process(unit, ctx, tally, log):
    if ctx["sample"]:
        # échantillon : lecture en flux (requêtes Range), sans tout télécharger
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with fs.open(f"datasets/{ctx['repo']}@{ctx['revision']}/{unit}", "rb") as f:
            yield from _docs_from_table(pq.ParquetFile(f), ctx, tally, row_groups=[0])
        return
    url = f"https://huggingface.co/datasets/{ctx['repo']}/resolve/{ctx['revision']}/{unit}"
    dest = source_dir(NAME) / "_work" / "dl" / Path(unit).name
    log.info("téléchargement %s", url)
    download(make_session(), url, dest, log=log)
    try:
        yield from _docs_from_table(pq.ParquetFile(dest), ctx, tally)
    finally:
        dest.unlink(missing_ok=True)  # fichier temporaire supprimé après usage
