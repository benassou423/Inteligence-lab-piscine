"""Annotation manuelle du filtre de langue : une touche par document, sans Entrée.

    uv run python -m tools.annotate_langid --config nettoye [--n 300]

Touches : f = français · n = pas français · m = mélangé · s = passer · u = annuler · q = quitter.
On peut quitter à tout moment : la reprise repart au premier document non annoté.

Tirage stratifié et déterministe : ~60 % parmi les documents gardés (répartis par
tranches de score), ~40 % parmi les rejetés (dont ceux que la règle « famille » sauverait).
Chaque document porte un poids (taille de sa strate / nombre tiré) pour que les
estimations de tools.langid_report valent pour tout le corpus, pas seulement pour l'échantillon.

Annotation à l'aveugle : l'écran ne montre pas la note du modèle (elle influencerait la
réponse). Chaque annotation garde toutes les notes du document (`langid_fr`,
`langid_family`…) : le rapport peut ainsi comparer les deux règles et tous les seuils.
"""
import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

import xxhash

from pipeline.io import read_docs

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"f": "français", "n": "pas français", "m": "mélangé", "s": "passé"}
# Ce qu'une annotation doit garder pour évaluer n'importe quelle règle et n'importe quel seuil.
SCORE_FIELDS = ("langid_label", "langid_score", "langid_fr", "langid_family", "langid_group", "langid_top")
GUIDE = ("f = le texte est en français (les noms propres étrangers ne comptent pas)\n"
         "n = autre langue, ou pas de phrases du tout (code, chiffres, liste de noms)\n"
         "m = vrai mélange : une bonne part en français ET une bonne part dans une autre langue")


# ------------------------------------------------------------------- tirage

def find_langid_dir(config, stages_root):
    dirs = sorted((Path(stages_root) / config).glob("*_langid"))
    if not dirs:
        sys.exit(f"Pas d'étape langid dans {Path(stages_root) / config}. "
                 f"Lancer d'abord : uv run python -m pipeline.run --config configs/pipeline/{config}.yaml")
    return dirs[-1]


def _hash(doc_id, salt="annot"):
    return xxhash.xxh64_intdigest(f"{salt}|{doc_id}".encode("utf-8"))


def _largest_remainder(total, weights, caps):
    """Répartit `total` proportionnellement aux poids, sans dépasser les plafonds.

    Méthode des plus forts restes : parts entières d'abord, puis une unité à la
    fois à la plus grosse part restante. Ce qu'un plafond bloque va aux autres.
    """
    alloc = {k: 0 for k in weights}
    remaining = min(total, sum(caps.get(k, 0) for k in weights if weights[k] > 0))
    while remaining > 0:
        active = sorted(k for k in weights if weights[k] > 0 and alloc[k] < caps.get(k, 0))
        if not active:
            break
        wsum = sum(weights[k] for k in active)
        quotas = {k: remaining * weights[k] / wsum for k in active}
        step = {k: min(int(quotas[k]), caps[k] - alloc[k]) for k in active}
        if sum(step.values()) == 0:
            best = max(active, key=lambda k: quotas[k])  # premier en cas d'égalité (liste triée)
            step = {best: 1}
        for k, v in step.items():
            alloc[k] += v
            remaining -= v
    return alloc


def draw_sample(stage_dir, n=300, kept_share=0.6):
    """Échantillon stratifié et pondéré à partir des sorties de l'étape langid."""
    stats = json.loads((Path(stage_dir) / "stats.json").read_text(encoding="utf-8"))
    threshold = float(stats["params"].get("threshold", 0.65))
    rule = stats["params"].get("rule", "fr")
    pop = {}  # (source, strate) -> nombre de documents dans le corpus
    for source, entry in stats["sources"].items():
        for stratum, count in entry["details"].get("strata", {}).items():
            pop[(source, stratum)] = count
    names = sorted({h for _, h in pop})
    kept_names = [h for h in names if h.startswith("kept")]
    rejected_names = [h for h in names if h.startswith("rejected")]
    pool = {}  # (source, strate) -> enregistrements disponibles, dans un ordre pseudo-aléatoire stable
    for f in sorted((Path(stage_dir) / "_strata_sample").glob("*.jsonl.zst")):
        for rec in read_docs(f):
            if "langid_fr" not in rec or "langid_family" not in rec:
                sys.exit(f"{f} : échantillon sans langid_fr / langid_family (ancienne version de l'étape langid). "
                         "Relancer le pipeline (étapes langid_score puis langid) avant de tirer.")
            if rec.get("text", "").strip():
                pool.setdefault((rec["source"], rec["stratum"]), []).append(rec)
    for key in pool:
        pool[key].sort(key=lambda r: _hash(r["id"]))

    def side(names, total):
        avail = {h: sum(len(pool.get((s, h2), [])) for (s, h2) in pool if h2 == h) for h in names}
        size = {h: sum(c for (s, h2), c in pop.items() if h2 == h) for h in names}
        live = [h for h in names if avail[h] > 0 and size[h] > 0]
        # Répartition égale entre strates (les strates rares sont sur-représentées, puis repondérées)
        per = _largest_remainder(total, {h: 1 for h in live}, {h: avail[h] for h in live})
        chosen = []
        for h in live:
            cells = {s: pop.get((s, h), 0) for (s, h2) in pool if h2 == h}
            caps = {s: len(pool[(s, h)]) for s in cells}
            alloc = _largest_remainder(per[h], cells, caps)
            for s, k in alloc.items():
                for rec in pool[(s, h)][:k]:
                    chosen.append({**rec, "weight": pop.get((s, h), 0) / k})
        return chosen

    n_kept = round(n * kept_share)
    items = side(kept_names, n_kept) + side(rejected_names, n - n_kept)
    items.sort(key=lambda r: _hash(r["id"], "ordre"))  # ordre mélangé : gardés et rejetés alternent
    meta = {"stage_dir": str(stage_dir), "threshold": threshold, "rule": rule, "n": len(items),
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "population": {f"{s}|{h}": c for (s, h), c in sorted(pop.items())}}
    return items, meta


# ----------------------------------------------------------- annotations

def load_annotations(path):
    """Dernière annotation par document (les lignes « undo » ont été retirées du fichier)."""
    ann = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                ann[rec["id"]] = rec
    return ann


def append_annotation(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def undo_last(path):
    """Retire la dernière annotation ; renvoie son ID (ou None)."""
    p = Path(path)
    if not p.exists():
        return None
    lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    p.write_text("".join(line + "\n" for line in lines[:-1]), encoding="utf-8")
    return last["id"]


# --------------------------------------------------------------- terminal

def getch():
    """Une touche, sans Entrée (mode cbreak). Repli sur input() hors terminal."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1] if line else "q"
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def render(item, done, total, counts, width, preview_chars=600):
    bar_w = max(10, min(40, width - 40))
    filled = int(bar_w * done / max(1, total))
    lines = [
        f"\033[1mAnnotation langue\033[0m  {done}/{total}  [{'█' * filled}{'░' * (bar_w - filled)}]  "
        f"{100 * done / max(1, total):.0f} %",
        f"f {counts['f']} · n {counts['n']} · m {counts['m']} · s {counts['s']}",
        "",
        # Pas de note du modèle à l'écran : annotation à l'aveugle.
        f"source : \033[1m{item['source']}\033[0m   taille : {item.get('text_bytes', 0):,} octets".replace(",", " "),
        "─" * min(width, 100),
    ]
    text = item.get("text", "")[:preview_chars]
    paras = [p for p in text.replace("\r", "").split("\n") if p.strip()]
    for p in paras[:14]:
        lines += textwrap.wrap(p.strip(), width=min(width, 100) - 2) or [""]
    if len(item.get("text", "")) > preview_chars:
        lines.append("…")
    lines += ["─" * min(width, 100),
              "\033[1m[f]\033[0m français   \033[1m[n]\033[0m pas français   \033[1m[m]\033[0m mélangé   "
              "\033[1m[s]\033[0m passer   \033[1m[u]\033[0m annuler   \033[1m[q]\033[0m quitter"]
    return "\033[2J\033[H" + "\n".join(lines) + "\n"


def run_session(sample_path, ann_path, out=sys.stdout, key_reader=getch):
    items = [json.loads(line) for line in Path(sample_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(items)
    while True:
        ann = load_annotations(ann_path)
        todo = [it for it in items if it["id"] not in ann]
        counts = {k: sum(1 for a in ann.values() if a["label"] == k) for k in KEYS}
        if not todo:
            out.write(f"\n✅ Les {total} documents sont annotés.\n"
                      "Rapport : uv run python -m tools.langid_report\n")
            return 0
        item = todo[0]
        width = shutil.get_terminal_size((100, 30)).columns
        out.write(render(item, total - len(todo), total, counts, width))
        out.flush()
        key = (key_reader() or "").lower()
        if key == "q" or key == "\x1b":
            out.write(f"\nPause. {total - len(todo)}/{total} annotés. Relancer la même commande pour reprendre.\n")
            return 0
        if key == "u":
            undo_last(ann_path)
            continue
        if key in KEYS:
            append_annotation(ann_path, {
                "id": item["id"], "label": key, "source": item["source"], "stratum": item["stratum"],
                "keep": item["keep"], **{k: item.get(k) for k in SCORE_FIELDS},
                "weight": item["weight"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })


def main(argv=None):
    ap = argparse.ArgumentParser(description="Annoter ~300 documents pour mesurer le filtre de langue.")
    ap.add_argument("--config", default="nettoye")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stages", default=str(ROOT / "data" / "stages"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "annotations"))
    ap.add_argument("--reset", action="store_true", help="refait le tirage (les annotations sont gardées)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path, meta_path = out_dir / "langid_sample.jsonl", out_dir / "langid_sample.meta.json"
    ann_path = out_dir / "langid.jsonl"
    if args.reset or not sample_path.exists():
        stage_dir = find_langid_dir(args.config, args.stages)
        items, meta = draw_sample(stage_dir, args.n)
        meta["config"] = args.config
        sample_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Tirage : {len(items)} documents ({stage_dir}).")
    if sys.stdin.isatty():
        print(GUIDE + "\n\nAppuie sur une touche pour commencer…")
        getch()
    return run_session(sample_path, ann_path)


if __name__ == "__main__":
    sys.exit(main())
