"""Statistiques d'étape, échantillons déterministes et entonnoir lisible (funnel.md)."""
import heapq
import json
from collections import Counter
from pathlib import Path

import xxhash

CORE_KEYS = ("docs_in", "docs_out", "bytes_in", "bytes_out")


# --------------------------------------------------------------------------- stats

def new_source_stats():
    """Statistiques vides pour une source (format de stats.json)."""
    return {
        "docs_in": 0,
        "docs_out": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "removed_by_reason": {},
        "docs_modified": 0,  # documents gardés dont le texte a changé
        "bytes_removed_docs": 0,  # octets des documents retirés
        "counters": {},  # compteurs propres à l'étape
        "cpu_seconds": 0.0,
        "units": 0,
    }


def merge_source_stats(a, b):
    """Additionne b dans a (même source, unités différentes)."""
    for k in CORE_KEYS + ("docs_modified", "bytes_removed_docs", "units"):
        a[k] += b.get(k, 0)
    a["cpu_seconds"] += b.get("cpu_seconds", 0.0)
    for field in ("removed_by_reason", "counters"):
        c = Counter(a[field])
        c.update(b.get(field, {}))
        a[field] = dict(c)
    return a


def total_stats(per_source):
    """Somme sur toutes les sources."""
    tot = new_source_stats()
    for st in per_source.values():
        merge_source_stats(tot, st)
    return tot


def rates(st):
    """Débits par cœur (docs/s et Mo/s), calculés sur le temps CPU des workers."""
    cpu = st.get("cpu_seconds") or 0.0
    if cpu <= 0:
        return {"docs_per_s_core": None, "mb_per_s_core": None}
    return {
        "docs_per_s_core": round(st["docs_in"] / cpu, 1),
        "mb_per_s_core": round(st["bytes_in"] / 1e6 / cpu, 3),
    }


# ------------------------------------------------------------------ échantillons

def sample_hash(salt, doc_id):
    """Hash 64 bits stable d'un document : sert au tirage déterministe."""
    return xxhash.xxh64_intdigest(f"{salt}|{doc_id}".encode("utf-8"))


class BottomK:
    """Garde les k éléments de plus petit hash.

    Comme le hash est pseudo-aléatoire, c'est un tirage uniforme sans remise,
    identique quel que soit l'ordre de lecture ou le découpage en workers.
    """

    def __init__(self, k):
        self.k = k
        self._heap = []  # (-hash, id, record) : le plus grand hash est en tête

    def wants(self, h):
        return self.k > 0 and (len(self._heap) < self.k or h < -self._heap[0][0])

    def add(self, h, doc_id, record):
        item = (-h, doc_id, record)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, item)
        elif h < -self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def items(self):
        """[(hash, record)] triés par hash croissant."""
        return sorted(((-nh, rec) for nh, _, rec in self._heap), key=lambda x: x[0])


def merge_bottom_k(item_lists, k):
    """Fusionne des listes [(hash, record)] et garde les k plus petits hashes."""
    merged = [it for lst in item_lists for it in lst]
    merged.sort(key=lambda x: (x[0], x[1].get("id", "")))
    return merged[:k]


# ------------------------------------------------------------- formats lisibles

def fmt_int(n):
    """12345 -> '12 345'."""
    return f"{int(n):,}".replace(",", " ")


def fmt_mb(nbytes, digits=1):
    """Octets -> Mo au format français ('12,3')."""
    mb = nbytes / 1e6
    if 0 < mb < 0.1 and digits == 1:
        digits = 2 if mb >= 0.01 else 3
    return f"{mb:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_pct(part, whole, digits=1):
    """Pourcentage au format français ('98,5 %'), tiret si indéfini."""
    if not whole:
        return "—"
    return f"{100 * part / whole:.{digits}f} %".replace(".", ",")


def fmt_rate(x):
    if x is None:
        return "—"
    return fmt_int(round(x)) if x >= 10 else f"{x:.1f}".replace(".", ",")


def fmt_seconds(s):
    s = float(s or 0)
    if s < 60:
        return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s".replace(".", ",")
    if s < 3600:
        return f"{s // 60:.0f} min {s % 60:02.0f} s"
    return f"{s // 3600:.0f} h {(s % 3600) // 60:02.0f} min"


def top_reasons(removed, n=3):
    """'not_french 12 · low_score 3' (raisons les plus fréquentes)."""
    items = sorted(removed.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return " · ".join(f"{k} {fmt_int(v)}" for k, v in items)


# ---------------------------------------------------------------- stats.json

def write_json(path, obj):
    """Écriture JSON atomique (fichier temporaire puis renommage)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ funnel.md

def render_funnel(run, stages):
    """Construit funnel.md.

    run : dict (config, input, limit, workers, sources, generated_at, wall_seconds)
    stages : liste des stats.json des étapes, dans l'ordre.
    """
    sources = run["sources"]
    lines = []
    title = f"# Entonnoir « {run['config']} »"
    lines.append(title)
    lines.append("")
    scope = (
        f"échantillon : **{fmt_int(run['limit'])} premiers documents par source** (`--limit`)"
        if run.get("limit")
        else "corpus complet"
    )
    lines.append(
        f"Entrée `{run['input']}` · {scope} · {run['workers']} workers · "
        f"durée {fmt_seconds(run.get('wall_seconds'))} · généré le {run['generated_at']}."
    )
    lines.append("")
    lines.append(
        "Lecture : une ligne par étape. « gardés » = part qui passe l'étape ; "
        "« restant » = part des octets de l'entrée encore là. "
        "Débit mesuré en temps CPU par cœur (lecture + traitement + écriture)."
    )
    lines.append("")
    if not stages:
        lines.append("_Aucune étape exécutée._")
        return "\n".join(lines) + "\n"

    first = stages[0]

    def src_stats(stage, s):
        return stage["sources"].get(s)

    # ---- total
    lines.append("## Total")
    lines.append("")
    lines.append("| Étape | Documents | Mo | Docs gardés | Mo gardés | Mo restant | Débit (docs/s/cœur) | Retirés ou modifiés |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|---|")
    t0 = first["total"]
    lines.append(
        f"| 00 entrée | {fmt_int(t0['docs_in'])} | {fmt_mb(t0['bytes_in'])} | | | 100 % | | |"
    )
    for st in stages:
        t = st["total"]
        detail = []
        if t["removed_by_reason"]:
            detail.append(top_reasons(t["removed_by_reason"]))
        if t.get("docs_modified"):
            detail.append(f"{fmt_int(t['docs_modified'])} docs modifiés")
        lines.append(
            f"| {st['dir']} | {fmt_int(t['docs_out'])} | {fmt_mb(t['bytes_out'])} | "
            f"{fmt_pct(t['docs_out'], t['docs_in'])} | {fmt_pct(t['bytes_out'], t['bytes_in'])} | "
            f"{fmt_pct(t['bytes_out'], t0['bytes_in'])} | {fmt_rate(st['timing'].get('docs_per_s_core'))} | "
            f"{' · '.join(detail)} |"
        )
    lines.append("")

    # ---- matrice étape × source
    lines.append("## Étape × source")
    lines.append("")
    lines.append("Chaque case : documents · Mo · part des octets d'entrée de la source encore présente.")
    lines.append("")
    lines.append("| Étape | " + " | ".join(sources) + " |")
    lines.append("|---|" + "---|" * len(sources))
    row = ["00 entrée"]
    for s in sources:
        a = src_stats(first, s)
        row.append(f"{fmt_int(a['docs_in'])} · {fmt_mb(a['bytes_in'])} Mo · 100 %" if a else "—")
    lines.append("| " + " | ".join(row) + " |")
    for st in stages:
        row = [st["dir"]]
        for s in sources:
            a, a0 = src_stats(st, s), src_stats(first, s)
            if not a or not a0:
                row.append("—")
                continue
            row.append(
                f"{fmt_int(a['docs_out'])} · {fmt_mb(a['bytes_out'])} Mo · {fmt_pct(a['bytes_out'], a0['bytes_in'])}"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- détail par source
    lines.append("## Détail par source")
    lines.append("")
    for s in sources:
        a0 = src_stats(first, s)
        if not a0:
            continue
        lines.append(f"### {s}")
        lines.append("")
        lines.append("| Étape | Documents | Mo | Docs gardés | Mo gardés | Débit (docs/s/cœur) | Détail |")
        lines.append("|---|--:|--:|--:|--:|--:|---|")
        lines.append(f"| 00 entrée | {fmt_int(a0['docs_in'])} | {fmt_mb(a0['bytes_in'])} | | | | |")
        for st in stages:
            a = src_stats(st, s)
            if not a:
                continue
            detail = a.get("summary") or ""
            if not detail and a["removed_by_reason"]:
                detail = top_reasons(a["removed_by_reason"])
            lines.append(
                f"| {st['dir']} | {fmt_int(a['docs_out'])} | {fmt_mb(a['bytes_out'])} | "
                f"{fmt_pct(a['docs_out'], a['docs_in'])} | {fmt_pct(a['bytes_out'], a['bytes_in'])} | "
                f"{fmt_rate(a.get('docs_per_s_core'))} | {detail} |"
            )
        lines.append("")

    # ---- notes d'étape (rapports globaux)
    notes = [(st, st.get("report_md")) for st in stages if st.get("report_md")]
    if notes:
        lines.append("## Notes d'étape")
        lines.append("")
        for st, md in notes:
            lines.append(f"### {st['dir']}")
            lines.append("")
            lines.append(md.rstrip())
            lines.append("")

    lines.append("## Où regarder")
    lines.append("")
    lines.append(
        "- `stats.json` de chaque étape : chiffres exacts, compteurs, débits.\n"
        "- `_rejected_sample/` : documents retirés (tirage déterministe, 2 000 max par source) "
        "et `apercu.md` pour un coup d'œil rapide.\n"
        "- `_changed_sample/` : avant/après des documents modifiés (encodage, boilerplate)."
    )
    return "\n".join(lines) + "\n"
