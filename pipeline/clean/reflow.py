"""Étape `reflow` : mettre les livres dans la forme de la val.

Les fichiers texte du Projet Gutenberg sont coupés tous les ~70 caractères, au milieu
des phrases, et marquent l'italique par des soulignés (_ainsi_). Dans la val du kit,
chaque paragraphe tient sur une ligne (874 paragraphes sur 890) et il n'y a aucun
souligné. On recolle donc les lignes des paragraphes de prose et on retire les
soulignés d'italique. Les vers, tables et blocs centrés restent tels quels ; une ligne
qui commence par un tiret de dialogue (« -- ») reste une nouvelle ligne.
"""
import re
from collections import Counter

from pipeline.clean.base import Outcome, Stage

DEFAULT_METHODS = {"gutenberg_fr": "gutenberg"}

_PARA_SPLIT = re.compile(r"\n[ \t]*\n")
_ITALIC = re.compile(r"(?<![\w_])_(?=\S)([^_\n]*?\S)_(?![\w_])")
_DIALOGUE = ("--", "—", "–")


def wrap_width(text):
    """Largeur de coupe du livre : 95e centile des longueurs de ligne."""
    lens = sorted(len(line) for line in text.split("\n") if line.strip())
    return lens[int(0.95 * (len(lens) - 1))] if lens else 0


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def is_wrapped(lines, width, full_ratio=0.7, min_full=0.8, max_upper=0.7):
    """Paragraphe de prose coupé à largeur fixe (et non vers, table ou bloc centré)."""
    if len(lines) < 2 or width < 40:
        return False
    if len({_indent(line) for line in lines[1:]}) > 1:  # indentation variable : bloc mis en forme
        return False
    body = lines[:-1]
    if sum(len(line.rstrip()) >= full_ratio * width for line in body) < min_full * len(body):
        return False  # lignes courtes : vers, liste, table des matières
    starts = [line.lstrip()[:1] for line in lines[1:]]
    if len(lines) >= 4 and sum(s.isupper() for s in starts) > max_upper * len(starts):
        return False  # chaque ligne commence par une majuscule : des vers
    return True


def _join(lines):
    out = lines[0].strip()
    for line in lines[1:]:
        s = line.strip()
        if out.endswith("-") and not out.endswith("--") and out[-2:-1].isalpha() and s[:1].islower():
            out += s  # mot coupé en fin de ligne (« peut-\nêtre »)
        else:
            out += " " + s
    return out


def reflow_paragraph(par, width):
    """Recolle un paragraphe ; les répliques (« -- ») restent sur des lignes séparées."""
    lines = par.split("\n")
    groups, cur = [], []
    for line in lines:
        if cur and line.lstrip().startswith(_DIALOGUE):
            groups.append(cur)
            cur = []
        cur.append(line)
    groups.append(cur)
    out, joined = [], 0
    for g in groups:
        if is_wrapped(g, width):
            out.append(_join(g))
            joined += len(g) - 1
        else:
            out.extend(line.rstrip() for line in g)
    return "\n".join(out), joined


def reflow_book(text):
    """Renvoie (texte, infos) : lignes recollées, soulignés retirés."""
    width = wrap_width(text)
    paras = _PARA_SPLIT.split(text)
    out, joined = [], 0
    for p in paras:
        r, j = reflow_paragraph(p, width)
        out.append(r)
        joined += j
    new = "\n\n".join(out)
    new, n_italic = _ITALIC.subn(r"\1", new)
    return new, {"width": width, "lines_joined": joined, "italics": n_italic}


class ReflowStage(Stage):
    name = "reflow"
    sample_sizes = {"changed": 100}

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.methods = {**DEFAULT_METHODS, **(self.params.get("methods") or {})}

    def apply(self, doc):
        if self.methods.get(doc["source"]) != "gutenberg":
            return Outcome(doc)
        text = doc.get("text") or ""
        new, info = reflow_book(text)
        if new == text:
            return Outcome(doc, counters={"unchanged": 1})
        counters = Counter({"docs_reflowed": 1, "lines_joined": info["lines_joined"], "italics_removed": info["italics"]})
        mid = len(text) // 2
        record = {"id": doc["id"], "source": doc["source"], "width": info["width"],
                  "before": text[mid:mid + 400], "after": new[len(new) // 2: len(new) // 2 + 400]}
        doc["text"] = new
        return Outcome(doc, counters=dict(counters), samples=[("changed", "", record)])

    def describe(self, st):
        c = st.get("counters", {})
        if not c.get("docs_reflowed"):
            return "source inchangée"
        return (f"{c['docs_reflowed']} livres remis en paragraphes : {c.get('lines_joined', 0)} lignes recollées, "
                f"{c.get('italics_removed', 0)} italiques _ _ retirés")
