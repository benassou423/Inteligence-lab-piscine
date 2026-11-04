"""`python -m pipeline.collect stats` : volumes collectés par source → docs/sources/volumes.md."""
import importlib
from pathlib import Path

from pipeline.collect.common import ROOT, Tally, now_iso, read_json, source_dir
from pipeline.io import read_docs

OUT = ROOT / "docs" / "sources" / "volumes.md"


def scan_parts(src: Path) -> dict:
    """Compte documents, octets et dates en relisant les fichiers (si _stats.json manque)."""
    t = Tally()
    for p in sorted(src.glob("part-*.jsonl.zst")):
        for d in read_docs(p):
            t.add(d)
    return t.to_dict()


def source_volume(name: str) -> dict | None:
    src = source_dir(name)
    parts = sorted(src.glob("part-*.jsonl.zst"))
    if not parts:
        return None
    st = read_json(src / "_stats.json") or scan_parts(src)
    state = read_json(src / "_state.json", {}) or {}
    return {"docs": st["docs"], "bytes": st["bytes"], "date_min": st.get("date_min"), "date_max": st.get("date_max"),
            "files": len(parts), "disk": sum(p.stat().st_size for p in parts),
            "mode": "complet" if state.get("mode") == "full" and state.get("complete") else
            ("échantillon" if state.get("mode") == "sample" else "en cours"),
            "removed": st.get("removed_by_reason", {})}


def write_volumes(cfg) -> str:
    from pipeline.collect.__main__ import MODULES

    rows, tot = [], {"docs": 0, "bytes": 0, "disk": 0}
    active = dict(tot)
    archives = []
    # Une source désactivée peut être une archive déjà présente : elle occupe du disque.
    names = dict.fromkeys([*cfg.get("order", MODULES), *cfg.get("sources", {})])
    for name in names:
        if name == "moliere" or name not in MODULES:
            continue
        v = source_volume(name)
        if v is None:
            continue
        domain = importlib.import_module(MODULES[name]).DOMAIN
        period = f"{v['date_min']} → {v['date_max']}" if v["date_min"] else "—"
        enabled = cfg.get("sources", {}).get(name, {}).get("enabled", True)
        mode = v["mode"] if enabled else v["mode"] + " · archive"
        if not enabled:
            archives.append(name)
        rows.append(f"| `{name}` | {domain} | {mode} | {v['docs']:,} | {v['bytes'] / 1e6:,.1f} | {period} | "
                    f"{v['files']} | {v['disk'] / 1e6:,.0f} |".replace(",", " "))
        for k in tot:
            tot[k] += v[k]
            if enabled:
                active[k] += v[k]
    budget = float(cfg.get("raw_budget_gb", 30))
    raw_disk = sum(p.stat().st_size for p in source_dir("").rglob("*") if p.is_file())
    lines = [
        "Prochaine action : comparer les volumes actifs et les archives — 1 min.",
        "",
        "# Volumes collectés (`data/raw/`)",
        "",
        f"Généré par `uv run python -m pipeline.collect stats` le {now_iso()[:16].replace('T', ' ')}. "
        "Mo de texte = octets UTF-8 du champ `text` (le HTML brut de Common Crawl n'est pas compté). "
        "Période = plus petite et plus grande valeur du champ `date` (gel au "
        f"{cfg.get('freeze_date', '2026-10-16')}).",
        "",
        "| Source | Domaine | État | Documents | Mo de texte | Période couverte | Fichiers | Disque (Mo, zstd) |",
        "|---|---|---|---:|---:|---|---:|---:|",
        *rows,
        f"| **Total stocké (archives comprises)** | | | **{tot['docs']:,}** | **{tot['bytes'] / 1e6:,.1f}** | | | **{tot['disk'] / 1e6:,.0f}** |"
        .replace(",", " "),
        "",
        f"Corpus actif : **{active['docs']:,} documents** ; **{active['bytes'] / 1e6:,.1f} Mo de texte**."
        .replace(",", " "),
        "",
        f"Budget disque de `data/raw/` : **{raw_disk / 1e9:.2f} Go utilisés sur {budget:.0f} Go**, "
        "fichiers de travail et journaux compris.",
    ]
    if archives:
        lines += ["", "Archives exclues de la collecte `all` et du corpus actif : "
                  + ", ".join(f"`{name}`" for name in archives) + "."]
    mol = read_json(ROOT / "data" / "moliere" / "_stats.json")
    if mol:
        n = lambda x: f"{x:,}".replace(",", " ")  # noqa: E731
        lines += ["", f"Corpus Molière provisoire (`data/moliere/moliere.txt`, remplacé par `corpus/` du kit) : "
                      f"{mol['plays']} pièces, {n(mol['lines'])} lignes, {n(mol['chars'])} caractères."]
    lines += ["", "Détail par source (origine, licence, risques) : voir les fiches `docs/sources/<source>.md`."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)
