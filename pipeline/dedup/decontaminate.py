"""Étape `decontaminate` : l'entraînement ne doit pas recouvrir la validation du kit.

Même définition que le contrôle officiel (ftb/contamination.py) : 13-grammes de mots
en minuscules (regex \\w+), hachés par `ftb.contamination.ngram_hashes`.

Étape globale en deux passes :
1. chaque worker liste, pour chaque document, les 13-grammes qu'il partage avec la val ;
2. le processus principal décide, document par document :
   - **copie** : le document recouvre à lui seul >= `doc_min_coverage` (20 %) d'un
     document de val -> retiré en entier (séance, livre ou article source de la val) ;
   - **formules** : un 13-gramme présent dans >= `formula_min_docs` (20) documents
     d'entraînement est une formule (« le scrutin est annoncé dans l'enceinte de
     l'Assemblée nationale ») : il reste seulement si l'ensemble de ces exemptions
     ne dépasse pas le seuil officiel de recouvrement d'un document de val ;
   - **recouvrement résiduel** : les lignes qui portent les autres 13-grammes partagés
     sont retirées, le reste du document est gardé.

Mesures qui fixent ces seuils : docs/comprendre/decontamination.md.
Vitesse : une fenêtre de 13 mots n'est hachée que si son couple (premier mot, dernier
mot) existe dans la val ; résultat identique au hachage de toutes les fenêtres.
"""
import glob
import json
from collections import Counter
from pathlib import Path

import xxhash

from pipeline.clean.base import Outcome, Stage

DEFAULT_AGAINST = "data/validation.jsonl"


def _kit():
    from ftb.contamination import N, WORD, ngram_hashes

    return N, WORD, ngram_hashes


def load_reference(paths):
    """Documents de référence (val) : liste de dicts avec au moins `text`."""
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs += [json.loads(line) for line in f if line.strip()]
    return docs


class ValIndex:
    """Index des 13-grammes de la val : hash -> documents de val qui le contiennent."""

    def __init__(self, docs):
        n, word, ngram_hashes = _kit()
        self.n, self.word, self.ngram_hashes = n, word, ngram_hashes
        self.docs = docs
        self.owner, self.text_of, self.pairs = {}, {}, set()
        self.sizes = []
        for i, d in enumerate(docs):
            hs = set(ngram_hashes(d.get("text") or ""))
            self.sizes.append(len(hs))
            for h in hs:
                self.owner.setdefault(h, []).append(i)
            w = word.findall((d.get("text") or "").lower())
            for j in range(len(w) - n + 1):
                self.pairs.add((w[j], w[j + n - 1]))
                self.text_of.setdefault(ngram_hashes(" ".join(w[j:j + n]))[0], " ".join(w[j:j + n]))

    def _windows(self, words):
        """(position, hash) des fenêtres de 13 mots présentes dans la val."""
        n, pairs, owner = self.n, self.pairs, self.owner
        for i in range(len(words) - n + 1):
            if (words[i], words[i + n - 1]) in pairs:
                h = self.ngram_hashes(" ".join(words[i:i + n]))[0]
                if h in owner:
                    yield i, h

    def shared(self, text):
        """13-grammes que ce texte partage avec la val (même résultat que ngram_hashes)."""
        return {h for _, h in self._windows(self.word.findall(text.lower()))}

    def coverage(self, hits):
        """(document de val le plus recouvert, nombre de 13-grammes communs, part recouverte).

        « Le plus recouvert » = la plus grande part de ses 13-grammes, pas le plus grand nombre :
        un petit document de val copié en entier doit l'emporter sur un gros à peine effleuré.
        À égalité, le plus petit index : le résultat ne dépend pas de l'ordre de `hits` (un set
        de bytes, dont l'ordre change d'un processus à l'autre).
        """
        per = Counter(i for h in hits for i in self.owner[h])
        i, k = max(per.items(), key=lambda ik: (ik[1] / max(1, self.sizes[ik[0]]), -ik[0]))
        return i, k, k / max(1, self.sizes[i])

    def drop_lines(self, text, targets):
        """Retire les lignes qui portent un 13-gramme de `targets`. Renvoie (texte, lignes retirées)."""
        lines = text.split("\n")
        words, line_of = [], []
        for li, line in enumerate(lines):  # même découpage que sur le texte entier (\n n'est pas un mot)
            ws = self.word.findall(line.lower())
            words += ws
            line_of += [li] * len(ws)
        drop = set()
        for i, h in self._windows(words):
            if h in targets:
                drop.update(range(line_of[i], line_of[i + self.n - 1] + 1))
        if not drop:
            return text, 0
        return "\n".join(line for li, line in enumerate(lines) if li not in drop), len(drop)


class DecontaminateStage(Stage):
    name = "decontaminate"
    kind = "global"
    scan_kind = "records"  # passe 1 : liste de 13-grammes par document (et non une clé fixe)
    filters = True

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        against = self.params.get("against", DEFAULT_AGAINST)
        patterns = against if isinstance(against, list) else [against]
        self.paths = []
        for pat in patterns:
            full = pat if str(pat).startswith("/") else str(ctx.root / pat)
            self.paths += sorted(glob.glob(full))
        if not self.paths:
            raise FileNotFoundError(f"decontaminate : aucun fichier de référence pour {against}")
        self.doc_min_coverage = float(self.params.get("doc_min_coverage", 0.20))
        self.formula_min_docs = int(self.params.get("formula_min_docs", 20))
        self.mask_lines = bool(self.params.get("mask_lines", True))
        self.index = None
        self.decisions = {}
        self.formulas = set()
        self.df = Counter()
        self.guarded_val_docs = set()
        self.unexempted_formulas = 0

    def setup(self):
        if self.index is None:
            self.index = ValIndex(load_reference(self.paths))

    def teardown(self):
        self.index = None
        self.decisions = {}

    def fingerprint_extra(self, source=None):
        h = xxhash.xxh64()
        for p in self.paths:
            h.update(Path(p).read_bytes())
        return {"against": [str(p) for p in self.paths], "digest": h.hexdigest()}

    # ---- passe 1 (workers)
    def scan_doc(self, doc):
        hits = self.index.shared(doc.get("text") or "")
        return sorted(int.from_bytes(h, "big") for h in hits) if hits else None

    # ---- décision (processus principal)
    def plan(self, scanned):
        """scanned : [(unit_key, source, {"n": docs, "records": [[ligne, [hashes]], …]})]."""
        self.setup()
        self.df = Counter(h for _, _, data in scanned for _, hs in data["records"] for h in hs)
        self.formulas = {h for h, c in self.df.items() if c >= self.formula_min_docs}
        # Des phrases répétées dans beaucoup d'articles peuvent, ensemble, recouvrir
        # trop de validation. La fréquence ne doit pas contourner le contrôle du kit.
        from ftb.contamination import FLAG

        covered = Counter(i for h in self.formulas for i in self.index.owner[h.to_bytes(8, "big")])
        self.guarded_val_docs = {i for i, k in covered.items() if k / max(1, self.index.sizes[i]) > FLAG}
        unsafe = {h for h in self.formulas
                  if any(i in self.guarded_val_docs for i in self.index.owner[h.to_bytes(8, "big")])}
        self.unexempted_formulas = len(unsafe)
        self.formulas -= unsafe
        self.decisions = {}
        for unit_key, source, data in scanned:
            for line, hs in data["records"]:
                hits = {h.to_bytes(8, "big") for h in hs}
                i, k, cov = self.index.coverage(hits)
                specific = frozenset(hits - {h.to_bytes(8, "big") for h in hs if h in self.formulas})
                if cov >= self.doc_min_coverage:
                    action = "copy"
                elif specific and self.mask_lines:
                    action = "mask"
                else:
                    action = "formula_only"
                self.decisions[(unit_key, line)] = (action, i, len(hits), round(cov, 4), specific)
        h = xxhash.xxh64()
        for key in sorted(self.decisions):
            a, i, k, cov, spec = self.decisions[key]
            h.update(f"{key}|{a}|{i}|{k}|{cov}|{sorted(spec)}".encode())
        return h.hexdigest()

    # ---- passe 2 (workers)
    def apply_at(self, doc, unit_key, line):
        dec = self.decisions.get((unit_key, line))
        if dec is None:
            return Outcome(doc)
        action, i, k, cov, specific = dec
        val_doc = self.index.docs[i]
        meta = doc.setdefault("meta", {})
        meta["decontaminate_shared"] = k
        meta["decontaminate_val_id"] = val_doc.get("id")
        meta["decontaminate_val_frac"] = cov
        if action == "copy":
            return Outcome(doc, keep=False, reason="val_copy",
                           counters={f"copy_genre:{val_doc.get('genre')}": 1})
        if action == "formula_only":
            return Outcome(doc, counters={"formula_only_docs": 1})
        text, total = doc.get("text") or "", 0
        for _ in range(3):  # recoller deux lignes peut créer un nouveau 13-gramme : on revérifie
            text, n = self.index.drop_lines(text, specific)
            total += n
            rest = {h for h in self.index.shared(text) if int.from_bytes(h, "big") not in self.formulas}
            if not rest:
                break
            specific = frozenset(rest)
        meta["decontaminate_dropped_lines"] = total
        counters = {"masked_docs": 1, "masked_lines": total}
        if not text.strip():
            return Outcome(doc, keep=False, reason="val_overlap_empty", counters=counters)
        doc["text"] = text
        return Outcome(doc, counters=counters)

    # ---- fin d'étape (processus principal)
    def finalize(self, captures, samples):
        actions = Counter(a for a, *_ in self.decisions.values())
        top = sorted(((c, h) for h, c in self.df.items() if h in self.formulas), reverse=True)[:8]
        report = {"docs_with_overlap": len(self.decisions), "actions": dict(actions),
                  "n_formulas": len(self.formulas), "formula_min_docs": self.formula_min_docs,
                  "doc_min_coverage": self.doc_min_coverage,
                  "formula_guard_val_ids": [self.index.docs[i].get("id") for i in sorted(self.guarded_val_docs)],
                  "unexempted_formulas": self.unexempted_formulas,
                  "top_formulas": [{"docs": c, "text": self.index.text_of.get(h.to_bytes(8, "big"), "")}
                                   for c, h in top] if self.index else []}
        lines = [f"Documents qui partagent au moins un 13-gramme avec la val : **{len(self.decisions)}** "
                 f"(copies retirées {actions.get('copy', 0)}, lignes retirées dans {actions.get('mask', 0)}, "
                 f"formules seules {actions.get('formula_only', 0)}). "
                 f"Formules (13-grammes dans >= {self.formula_min_docs} documents) : {len(self.formulas)}.", ""]
        if self.unexempted_formulas:
            lines += [f"Exemptions retirées pour respecter le seuil officiel : {self.unexempted_formulas} "
                      f"13-grammes, sur {len(self.guarded_val_docs)} document(s) de validation.", ""]
        if report["top_formulas"]:
            lines += ["| Documents | Formule la plus fréquente |", "|--:|---|"]
            lines += [f"| {t['docs']} | {t['text']} |" for t in report["top_formulas"]]
        return report, "\n".join(lines)

    def describe(self, st):
        c = st.get("counters", {})
        r = st["removed_by_reason"]
        return (f"copies de la val retirées {r.get('val_copy', 0)} ; lignes retirées dans "
                f"{c.get('masked_docs', 0)} docs ({c.get('masked_lines', 0)} lignes) ; "
                f"formules seules {c.get('formula_only_docs', 0)}")
