"""Précision et rappel du filtre de langue, estimés à partir des annotations de Ben.

    uv run python -m tools.langid_report

Lit data/annotations/langid_sample.jsonl (tirage) et data/annotations/langid.jsonl
(annotations), écrit data/annotations/langid_report.md.

Compare, sur les mêmes annotations, deux règles et plusieurs seuils :
- `fr`     : probabilité du français >= seuil (règle actuelle) ;
- `family` : probabilité « français + parlers proches » (normand, franco-provençal…) >= seuil.

Chaque document annoté compte pour toute sa cellule (source × strate) :
poids = documents de la cellule dans le corpus / documents annotés de la cellule.
Les intervalles de confiance à 95 % viennent d'un bootstrap dans chaque cellule.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
RULES = {"fr": "langid_fr", "family": "langid_family"}  # règle -> score comparé au seuil
RULE_NAMES = {"fr": "français seul", "family": "français + parlers proches"}


def load(out_dir):
    out_dir = Path(out_dir)
    sample_path, meta_path, ann_path = (out_dir / "langid_sample.jsonl", out_dir / "langid_sample.meta.json",
                                        out_dir / "langid.jsonl")
    if not sample_path.exists():
        sys.exit("Pas de tirage : lancer d'abord uv run python -m tools.annotate_langid")
    items = {}
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ann = {}
    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                ann[rec["id"]] = rec["label"]
    return items, meta, ann


def annotated_rows(items, meta, ann):
    """Documents annotés (hors « passé »), avec leur poids recalculé."""
    pop = {tuple(k.split("|", 1)): v for k, v in meta["population"].items()}
    rows = [dict(items[i], label=lab) for i, lab in ann.items() if i in items and lab in ("f", "n", "m")]
    missing = [r["id"] for r in rows if not all(f in r for f in RULES.values())]
    if missing:
        sys.exit(f"{len(missing)} documents du tirage sans langid_fr / langid_family (tirage d'une ancienne "
                 "version) : relancer le pipeline puis uv run python -m tools.annotate_langid --reset")
    per_cell = Counter((r["source"], r["stratum"]) for r in rows)
    for r in rows:
        cell = (r["source"], r["stratum"])
        r["cell"] = cell
        r["w"] = pop.get(cell, 0) / per_cell[cell]
    covered = sum(pop.get(c, 0) for c in per_cell)
    total = sum(pop.values())
    return rows, covered / total if total else 0.0


def is_fr(label, lenient):
    return label == "f" or (lenient and label == "m")


def kept_at(r, rule=None, threshold=None):
    """Le document est-il gardé ? Sans règle ni seuil : la décision réelle du pipeline."""
    if rule is None and threshold is None:
        return bool(r["keep"])
    return r[RULES[rule or "fr"]] >= threshold


def estimate(rows, lenient=False, threshold=None, rule=None):
    """(précision, rappel, français perdus estimés, docs gardés estimés) pour une règle et un seuil."""
    if threshold is not None and rule is None:
        rule = "fr"
    kept_w = kept_fr = all_fr = 0.0
    for r in rows:
        fr = is_fr(r["label"], lenient)
        if kept_at(r, rule, threshold):
            kept_w += r["w"]
            kept_fr += r["w"] * fr
        all_fr += r["w"] * fr
    precision = kept_fr / kept_w if kept_w else float("nan")
    recall = kept_fr / all_fr if all_fr else float("nan")
    return precision, recall, all_fr - kept_fr, kept_w


def bootstrap(rows, lenient=False, threshold=None, n_boot=1000, seed=0, rule=None):
    """Intervalles à 95 % : rééchantillonnage avec remise dans chaque cellule."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    precs, recs = [], []
    for _ in range(n_boot):
        boot = [rng.choice(members) for members in cells.values() for _ in members]
        p, rc, _, _ = estimate(boot, lenient, threshold, rule)
        precs.append(p)
        recs.append(rc)

    def ci(xs):
        xs = sorted(x for x in xs if x == x)
        if not xs:
            return float("nan"), float("nan")
        return xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]

    return ci(precs), ci(recs)


def pct(x, digits=1):
    return "—" if x != x else f"{100 * x:.{digits}f} %".replace(".", ",")


def dec(x, digits=2):
    """0.65 -> '0,65'."""
    return f"{x:.{digits}f}".replace(".", ",")


def num(x):
    return f"{round(x):,}".replace(",", " ")


def snippet(text, n=160):
    s = " ".join((text or "").split())
    return (s[:n] + "…" if len(s) > n else s).replace("|", "\\|")


def scores(r):
    return f"{r['langid_label']} {dec(r['langid_score'])} · fr {dec(r['langid_fr'])} · famille {dec(r['langid_family'])}"


def build_report(items, meta, ann, n_boot=1000):
    rows, coverage = annotated_rows(items, meta, ann)
    labels = Counter(ann.get(i) for i in items if i in ann)
    threshold, rule = float(meta["threshold"]), meta.get("rule", "fr")
    lines = ["# Filtre de langue : précision et rappel mesurés à la main", ""]
    lines.append(
        f"Config `{meta.get('config', '?')}` · règle actuelle **{RULE_NAMES[rule]}** · seuil **{dec(threshold)}** · "
        f"{len(items)} documents tirés · annotés : **{len(rows)}** (f {labels['f']} · n {labels['n']} · "
        f"m {labels['m']} · passés {labels['s']}) · part du corpus couverte par les strates annotées : {pct(coverage)}."
    )
    lines.append("")
    if not rows:
        lines.append("_Pas encore d'annotation : lancer `uv run python -m tools.annotate_langid`._")
        return "\n".join(lines) + "\n", {}
    if len(rows) < 0.8 * len(items) or coverage < 0.9:
        lines += [f"> ⚠️ **Résultats provisoires** : {len(rows)} annotations sur {len(items)}, strates annotées "
                  f"= {pct(coverage)} du corpus. Finir l'annotation avant de conclure.", ""]

    res = {}
    for lenient in (False, True):
        p, r, lost, kept = estimate(rows, lenient)
        (plo, phi), (rlo, rhi) = bootstrap(rows, lenient, n_boot=n_boot)
        res[lenient] = dict(precision=p, recall=r, lost=lost, kept=kept, p_ci=(plo, phi), r_ci=(rlo, rhi))
    s, l_ = res[False], res[True]
    lines += [
        "## Résultat (règle actuelle)",
        "",
        "| | Strict (m = pas français) | Souple (m = français) |",
        "|---|---|---|",
        f"| **Précision** : part des documents gardés qui sont en français | {pct(s['precision'])} "
        f"[{pct(s['p_ci'][0])} – {pct(s['p_ci'][1])}] | {pct(l_['precision'])} "
        f"[{pct(l_['p_ci'][0])} – {pct(l_['p_ci'][1])}] |",
        f"| **Rappel** : part des documents français qui sont gardés | {pct(s['recall'])} "
        f"[{pct(s['r_ci'][0])} – {pct(s['r_ci'][1])}] | {pct(l_['recall'])} "
        f"[{pct(l_['r_ci'][0])} – {pct(l_['r_ci'][1])}] |",
        f"| Documents français perdus (estimation, corpus entier) | {num(s['lost'])} | {num(l_['lost'])} |",
        "",
        f"En clair (strict) : sur 100 documents gardés, environ **{dec(100 * (1 - s['precision']), 1)}** ne sont "
        f"pas en français ; sur 100 documents français, environ **{dec(100 * (1 - s['recall']), 1)}** sont jetés "
        "à tort.",
        "Entre crochets : intervalle de confiance à 95 % (bootstrap).",
        "",
    ]

    # Matrice de confusion
    conf_w, conf_n = defaultdict(float), Counter()
    for r in rows:
        key = ("Gardé" if r["keep"] else "Rejeté", r["label"])
        conf_w[key] += r["w"]
        conf_n[key] += 1
    lines += ["## Matrice de confusion", "",
              "Nombre estimé de documents dans tout le corpus (entre parenthèses : documents annotés).", "",
              "| Décision du filtre \\ Annotation | français (f) | mélangé (m) | pas français (n) |",
              "|---|--:|--:|--:|"]
    for decision in ("Gardé", "Rejeté"):
        cells = [f"{num(conf_w[(decision, k)])} ({conf_n[(decision, k)]})" for k in ("f", "m", "n")]
        lines.append(f"| {decision} | " + " | ".join(cells) + " |")
    lines.append("")

    # Les deux règles, plusieurs seuils, mêmes annotations
    sweep = {}
    lines += ["## Deux règles, plusieurs seuils (mêmes annotations)", "",
              "**fr** = probabilité du français ≥ seuil (règle actuelle). **famille** = probabilité "
              "« français + normand, franco-provençal, picard, wallon » ≥ seuil. Strict (m = pas français). "
              "Entre crochets : intervalles à 95 %.", "",
              "| Seuil | Précision fr | Rappel fr | Précision famille | Rappel famille | Français perdus fr → famille |",
              "|--:|--:|--:|--:|--:|--:|"]
    for t in sorted(set(SWEEP) | {threshold}):
        cols = []
        for rl in ("fr", "family"):
            p, rc, lost, kept = estimate(rows, False, t, rl)
            (plo, phi), (rlo, rhi) = bootstrap(rows, False, t, n_boot=n_boot, rule=rl)
            sweep[(rl, t)] = dict(precision=p, recall=rc, lost=lost, kept=kept, p_ci=(plo, phi), r_ci=(rlo, rhi))
            cols += [f"{pct(p)} [{pct(plo)}–{pct(phi)}]", f"{pct(rc)} [{pct(rlo)}–{pct(rhi)}]"]
        mark = " ◀ actuel" if abs(t - threshold) < 1e-9 else ""
        lost = f"{num(sweep[('fr', t)]['lost'])} → {num(sweep[('family', t)]['lost'])}"
        lines.append(f"| {dec(t)}{mark} | " + " | ".join(cols) + f" | {lost} |")
    lines += ["", "Lecture : à seuil égal, la règle famille garde au moins autant de documents que la règle fr "
              "(rappel ≥, précision ≤ ou ≥ selon ce qu'elle repêche). Choisir le couple (règle, seuil) qui "
              "garde le rappel haut sans faire tomber la précision.", "",
              "Les cas que seule la règle famille garde sont surtout dans les strates « rejected_fam_* » ; plus il y a "
              "d'annotations dans ces strates, plus la comparaison est fiable.", ""]

    # Par source
    lines += ["## Par source (seuil actuel)", "",
              "| Source | Annotés | Précision fr | Rappel fr | Précision famille | Rappel famille |",
              "|---|--:|--:|--:|--:|--:|"]
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for src, rs in sorted(by_source.items()):
        pf, rf, _, _ = estimate(rs, False, threshold, "fr")
        pfa, rfa, _, _ = estimate(rs, False, threshold, "family")
        lines.append(f"| {src} | {len(rs)} | {pct(pf)} | {pct(rf)} | {pct(pfa)} | {pct(rfa)} |")
    lines.append("")

    # Par strate
    lines += ["## Par strate", "", "| Strate | Annotés | f | m | n |", "|---|--:|--:|--:|--:|"]
    by_stratum = defaultdict(Counter)
    for r in rows:
        by_stratum[r["stratum"]][r["label"]] += 1
    for h in sorted(by_stratum):
        c = by_stratum[h]
        tot = sum(c.values())
        lines.append(f"| {h} | {tot} | {pct(c['f'] / tot, 0)} | {pct(c['m'] / tot, 0)} | {pct(c['n'] / tot, 0)} |")
    lines.append("")

    # Erreurs et cas repêchés
    fp = [r for r in rows if r["keep"] and r["label"] == "n"]
    fn = [r for r in rows if not r["keep"] and r["label"] == "f"]
    rescued = [r for r in rows if not kept_at(r, "fr", threshold) and kept_at(r, "family", threshold)]
    lines += ["## Erreurs à regarder", "", f"### Gardés alors qu'ils ne sont pas en français ({len(fp)})", ""]
    for r in sorted(fp, key=lambda r: r["langid_score"]):
        lines.append(f"- `{r['id']}` {scores(r)} — {snippet(r.get('text'))}")
    lines += ["", f"### Rejetés alors qu'ils sont en français ({len(fn)})", ""]
    for r in sorted(fn, key=lambda r: -r["langid_family"]):
        lines.append(f"- `{r['id']}` {scores(r)} — {snippet(r.get('text'))}")
    lines += ["", f"### Repêchés par la règle famille au seuil {dec(threshold)} ({len(rescued)} annotés : "
              f"f {sum(r['label'] == 'f' for r in rescued)} · m {sum(r['label'] == 'm' for r in rescued)} · "
              f"n {sum(r['label'] == 'n' for r in rescued)})", ""]
    for r in sorted(rescued, key=lambda r: -r["langid_family"]):
        lines.append(f"- `{r['id']}` **{r['label']}** {scores(r)} — {snippet(r.get('text'))}")
    lines += ["", "## Méthode", "",
              "- **Tirage stratifié** : ~60 % de documents gardés répartis par tranches de score, ~40 % de rejetés "
              "(score « famille » haut ; moyen ; autre langue incertaine ; autre langue sûre).",
              "- **Pondération** : chaque document annoté représente sa cellule (source × strate) : "
              "poids = taille de la cellule / documents annotés dans la cellule. Sans ce poids, les strates rares "
              "(sur-représentées exprès) fausseraient le résultat. Les strates découpent tout le corpus : "
              "l'estimation vaut pour n'importe quelle règle, pas seulement celle du tirage.",
              "- **Intervalles** : bootstrap (1 000 tirages avec remise dans chaque cellule), percentiles 2,5 et 97,5.",
              "- **À l'aveugle** : l'écran d'annotation ne montre pas la note du modèle.",
              ""]
    summary = {"n_annotated": len(rows), "strict": {k: v for k, v in s.items()},
               "lenient": {k: v for k, v in l_.items()},
               "sweep": {f"{rl}@{t:.2f}": v for (rl, t), v in sweep.items()}}
    return "\n".join(lines) + "\n", summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rapport de précision/rappel du filtre de langue.")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "annotations"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    items, meta, ann = load(args.out_dir)
    report, summary = build_report(items, meta, ann, n_boot=args.boot)
    path = Path(args.out_dir) / "langid_report.md"
    path.write_text(report, encoding="utf-8")
    if summary:
        s = summary["strict"]
        print(f"{summary['n_annotated']} annotations · précision {pct(s['precision'])} "
              f"[{pct(s['p_ci'][0])} – {pct(s['p_ci'][1])}] · rappel {pct(s['recall'])} "
              f"[{pct(s['r_ci'][0])} – {pct(s['r_ci'][1])}] (strict, règle actuelle)")
    print(f"Rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
