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
    for name in cfg.get("order", MODULES):
        if name == "moliere" or name not in MODULES:
            continue
        v = source_volume(name)
        if v is None:
            continue
        domain = importlib.import_module(MODULES[name]).DOMAIN
        period = f"{v['date_min']} → {v['date_max']}" if v["date_min"] else "—"
        rows.append(f"| `{name}` | {domain} | {v['mode']} | {v['docs']:,} | {v['bytes'] / 1e6:,.1f} | {period} | "
                    f"{v['files']} | {v['disk'] / 1e6:,.0f} |".replace(",", " "))
        for k in tot:
            tot[k] += v[k]
    budget = float(cfg.get("raw_budget_gb", 30))
    lines = [
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
        f"| **Total** | | | **{tot['docs']:,}** | **{tot['bytes'] / 1e6:,.1f}** | | | **{tot['disk'] / 1e6:,.0f}** |"
        .replace(",", " "),
        "",
        f"Budget disque de `data/raw/` : {tot['disk'] / 1e9:.1f} Go utilisés sur {budget:.0f} Go.",
    ]
    mol = read_json(ROOT / "data" / "moliere" / "_stats.json")
    if mol:
        n = lambda x: f"{x:,}".replace(",", " ")  # noqa: E731
        lines += ["", f"Corpus Molière provisoire (`data/moliere/moliere.txt`, remplacé par `corpus/` du kit) : "
                      f"{mol['plays']} pièces, {n(mol['lines'])} lignes, {n(mol['chars'])} caractères."]
    lines += ["", "Détail par source (origine, licence, risques) : voir les fiches `docs/sources/<source>.md`."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)
