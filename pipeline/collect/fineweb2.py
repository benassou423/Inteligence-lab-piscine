"""FineWeb-2, français (`fra_Latn`) : web déjà filtré par Hugging Face, sous-ensemble déterministe et réparti.

On prend `n_files` fichiers parquet à pas régulier parmi les 135 (ils sont rangés par dump Common Crawl,
donc on couvre 2013 → 2024), puis dans chacun des groupes de lignes à pas régulier, jusqu'à
target_text_gb / n_files de texte par fichier. Lecture en flux (requêtes Range), rien n'est téléchargé en entier."""
import math

import pyarrow.parquet as pq

from pipeline.collect.common import date_ok, make_doc, make_session

NAME = "fineweb2_fra"
DOMAIN = "web"
LICENSE = "ODC-By 1.0 (FineWeb-2) ; Common Crawl Terms of Use ; droits des pages réservés à leurs auteurs"
COLUMNS = ["text", "id", "url", "date", "dump", "language_score", "minhash_cluster_size"]


def spread(n_total, k):
    """k indices répartis régulièrement dans range(n_total), premier et dernier compris."""
    if k >= n_total:
        return list(range(n_total))
    if k == 1:
        return [0]
    return sorted({round(i * (n_total - 1) / (k - 1)) for i in range(k)})


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    s = make_session()
    rev = c.get("revision")
    if not rev:
        rev = s.get(f"https://huggingface.co/api/datasets/{c['repo']}", timeout=60).json()["sha"]
    r = s.get(f"https://huggingface.co/api/datasets/{c['repo']}/tree/{rev}/data/{c['config']}/train", timeout=60)
    r.raise_for_status()
    files = sorted(x["path"] for x in r.json() if x["path"].endswith(".parquet"))
    picked = [files[i] for i in spread(len(files), int(c["n_files"]))]
    per_file = float(c["target_text_gb"]) * 1e9 / len(picked)
    log.info("%s %s@%s : %d fichiers, on en lit %d (%.0f Mo de texte chacun)", c["repo"], c["config"], rev[:10],
             len(files), len(picked), per_file / 1e6)
    return {"repo": c["repo"], "revision": rev, "files": picked, "per_file_bytes": per_file, "sample": sample}


def units(ctx):
    return ctx["files"]


def process(unit, ctx, tally, log):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{ctx['repo']}@{ctx['revision']}/{unit}", "rb", block_size=8 << 20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        text_col = pf.schema_arrow.get_field_index("text")
        sizes = [md.row_group(i).column(text_col).total_uncompressed_size for i in range(md.num_row_groups)]
        avg = max(1, sum(sizes) / len(sizes))
        need = 1 if ctx["sample"] else min(md.num_row_groups, math.ceil(ctx["per_file_bytes"] / avg * 1.1))
        groups = spread(md.num_row_groups, need)
        got = 0
        for g in groups:
            for r in pf.read_row_group(g, columns=COLUMNS).to_pylist():
                date = (r.get("date") or "")[:10]
                if not date_ok(date):
                    tally.skip("date_apres_gel")
                    continue
                text = r["text"] or ""
                if not text.strip():
                    tally.skip("texte_vide")
                    continue
                meta = {"dump": r.get("dump"), "language_score": r.get("language_score"),
                        "minhash_cluster_size": r.get("minhash_cluster_size"), "file": unit, "row_group": g}
                doc = make_doc(NAME, DOMAIN, (r["id"] or "").strip("<>"), text, url=r.get("url"), date=date,
                               license=LICENSE, meta=meta)
                got += len(text.encode("utf-8"))
                yield doc
            if got >= ctx["per_file_bytes"]:
                break
    log.info("%s : %d groupes de lignes lus sur %d, %.0f Mo de texte", unit, len(groups), md.num_row_groups, got / 1e6)
