"""Étape `boilerplate` : retire ce qui n'est pas le contenu, selon la source.

- commoncrawl  : extraction du texte principal du HTML avec trafilatura
                 (menus, pieds de page, bandeaux cookies et commentaires écartés).
- gutenberg_fr : retrait de l'en-tête et de la licence Project Gutenberg.
- wikipedia_fr : retrait des sections finales sans prose (références, liens…).
- autres       : inchangées.
"""
import logging
import re
from collections import Counter

from pipeline.clean.base import Outcome, Stage, utf8_len
from pipeline.clean.encoding import clean_text as fix_encoding  # ftfy + espaces, comme l'étape encoding

DEFAULT_METHODS = {
    "commoncrawl": "trafilatura",
    "gutenberg_fr": "gutenberg",
    "wikipedia_fr": "wikipedia",
}

# ----------------------------------------------------------------- trafilatura

TRAFILATURA_DEFAULTS = {
    "include_comments": False,  # commentaires des lecteurs : bruit
    "include_tables": False,  # tableaux rendus en lignes « a | b | c » : bruit pour un LM
    "favor_precision": True,  # comme FineWeb : mieux vaut perdre un peu que garder du menu
    "favor_recall": False,
    "fast": True,  # sans algorithmes de secours : 1,6x plus rapide ; ce qu'ils ajoutent est surtout du rebut
    "deduplicate": False,  # True utilise un cache global : résultat dépendant des workers
    "max_html_bytes": 3_000_000,  # au-delà : page anormale (scripts, données), retirée
    "fix_encoding": True,  # même réparation que l'étape encoding (ftfy + espaces), sur le texte extrait
}


def extract_main_text(html, url=None, params=None):
    """Texte principal d'une page HTML (None si rien d'extractible)."""
    import trafilatura

    p = {**TRAFILATURA_DEFAULTS, **(params or {})}
    return trafilatura.extract(
        html,
        url=url,
        output_format="txt",
        include_comments=p["include_comments"],
        include_tables=p["include_tables"],
        favor_precision=p["favor_precision"],
        favor_recall=p["favor_recall"],
        fast=p["fast"],
        deduplicate=p["deduplicate"],
        include_links=False,
        include_images=False,
        with_metadata=False,
    )


# ------------------------------------------------------------------- gutenberg

# Marqueurs de fin d'en-tête (on coupe après la DERNIÈRE occurrence dans le début du livre).
GUTENBERG_START = (
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "***START OF THE PROJECT GUTENBERG",
    "***START OF THIS PROJECT GUTENBERG",
    "*** START OF THE COPYRIGHTED PROJECT GUTENBERG",
    "*END*THE SMALL PRINT",
    "*END THE SMALL PRINT",
    "***END OF THE SMALL PRINT",
    "*** END OF THE SMALL PRINT",
)
# Marqueurs de début de pied de page (on coupe à la PREMIÈRE occurrence après l'en-tête).
GUTENBERG_END = (
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "***END OF THIS PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED PROJECT GUTENBERG",
    "END OF THE PROJECT GUTENBERG EBOOK",
    "END OF THIS PROJECT GUTENBERG EBOOK",
    "END OF PROJECT GUTENBERG",
    "END OF THE PROJECT GUTENBERG",
    "END OF THIS PROJECT GUTENBERG",
    "FIN DU PROJET GUTENBERG",
    "FIN DE CE PROJET GUTENBERG",
)
# Paragraphes de crédits juste après l'en-tête (« Produced by … pgdp.net … gallica »), et restes
# d'un ancien en-tête recopié dans le livre (« The Project Gutenberg EBook of… Copyright laws… »).
GUTENBERG_CREDITS = (
    "PRODUCED BY",
    "CREDITS:",
    "THIS ETEXT WAS PREPARED BY",
    "THIS ETEXT WAS PRODUCED BY",
    "THIS EBOOK WAS PRODUCED BY",
    "THIS EBOOK WAS PREPARED BY",
    "E-TEXT PREPARED BY",
    "ETEXT PREPARED BY",
    "TRANSCRIBED FROM",
    "THIS FILE WAS PRODUCED FROM",
    "(THIS FILE WAS PRODUCED FROM",
    "THE PROJECT GUTENBERG EBOOK",
    "THE PROJECT GUTENBERG ETEXT",
    "PROJECT GUTENBERG'S",
    "PROJECT GUTENBERG’S",
    "COPYRIGHT LAWS ARE CHANGING",
    "THIS HEADER SHOULD BE THE FIRST THING",
)


def _marker(line, markers):
    s = line.lstrip().upper()
    return any(s.startswith(m) for m in markers)


def _paragraphs(lines, start, limit):
    """[(début, fin)] des `limit` premiers paragraphes (blocs de lignes non vides) à partir de `start`."""
    out, i = [], start
    while i < len(lines) and len(out) < limit:
        if not lines[i].strip():
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip():
            j += 1
        out.append((i, j))
        i = j
    return out


def strip_gutenberg(text, max_header_lines=1000, credits_paragraphs=4):
    """Retire en-tête, crédits et licence Project Gutenberg.

    Les crédits (« Produced by… », « Credits: … », ancien en-tête recopié) peuvent tenir
    en plusieurs paragraphes : on coupe après le DERNIER paragraphe de crédits parmi les
    `credits_paragraphs` premiers paragraphes du livre.
    Renvoie (corps, en-tête retiré, pied retiré, infos) ; infos dit ce qui a été trouvé.
    """
    lines = text.split("\n")
    info = Counter()
    start = 0
    for i, line in enumerate(lines[:max_header_lines]):
        if _marker(line, GUTENBERG_START):
            start = i + 1
    if start:
        info["start_marker"] += 1
        credits = [(a, b) for a, b in _paragraphs(lines, start, credits_paragraphs)
                   if _marker(lines[a], GUTENBERG_CREDITS)]
        if credits:
            start = credits[-1][1]
            info["credits"] += 1
            info["credits_paragraphs"] += len(credits)
    end = len(lines)
    for i in range(start, len(lines)):
        if _marker(lines[i], GUTENBERG_END):
            end = i
            info["end_marker"] += 1
            break
    if not start and end == len(lines):
        return text, "", "", info
    header, footer = "\n".join(lines[:start]), "\n".join(lines[end:])
    body = "\n".join(lines[start:end]).strip("\n")
    info["bytes_header"] = utf8_len(header)
    info["bytes_footer"] = utf8_len(footer)
    return body, header, footer, info


# ------------------------------------------------------------------- wikipedia

# Titres des sections finales. Un titre est « final » si tous ses mots viennent de
# ce vocabulaire et qu'il contient un mot-clé : « Notes et références »,
# « Notes, sources et références », « Liens internes », « Voir aussi »…
WIKIPEDIA_END_KEYWORDS = (
    "notes", "note", "références", "référence", "sources", "source", "bibliographie",
    "liens", "lien", "voir", "articles", "article", "annexes", "annexe", "webographie",
    "approfondir", "documentation",
)
WIKIPEDIA_END_VOCAB = WIKIPEDIA_END_KEYWORDS + (
    "et", "aussi", "externes", "externe", "internes", "interne", "connexes", "connexe",
    "pour", "autres", "sites", "site", "ressources", "principales", "générales", "de",
)
# Résidus fréquents en fin d'article dans le dataset HF (catégories, tableaux supprimés).
_RESIDUE_PREFIXES = ("homonymie de", "portail ", "catégorie:", "catégorie :", "sur les autres projets wikimedia")
_HEADING_STRIP = re.compile(r"^[=#\s]+|[=#:\s]+$")
_WORDS = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"[.!?…»)][\s\"»]|[.!?…]$")
_FINAL_PUNCT = (".", "!", "?", "…", "»", '"', ")", ":", ";", ",")


def _norm_heading(line):
    return _HEADING_STRIP.sub("", line).strip().lower()


def is_end_heading(line, keywords=WIKIPEDIA_END_KEYWORDS, vocab=WIKIPEDIA_END_VOCAB):
    """Ligne de titre d'une section finale (références, liens, annexes…)."""
    if len(line) >= 60 or line.startswith(" "):
        return False
    words = _WORDS.findall(_norm_heading(line))
    return bool(words) and all(w in vocab for w in words) and any(w in keywords for w in words)


def _is_prose(line, min_chars):
    """Paragraphe de prose : long, avec au moins deux fins de phrase."""
    s = line.strip()
    return len(s) >= min_chars and len(_SENTENCE_END.findall(s)) >= 2


_TABLE_MARKUP = re.compile(r"^\s*(\{\||\|\}|\|-|\|\+)")
_ONE_LINE_TEMPLATE = re.compile(r"^\s*\{\{.*\}\}\s*$")


def strip_wikitext_markup(lines, max_block=300):
    """Retire le wikitexte resté brut : lignes de tableau ({| |- |}) et modèles {{…}}.

    Un modèle sur plusieurs lignes (infobox) part en entier, jusqu'à sa ligne « }} ».
    Renvoie (lignes, nombre de lignes retirées).
    """
    out, removed, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if _TABLE_MARKUP.match(line) or _ONE_LINE_TEMPLATE.match(line):
            removed += 1
            i += 1
            continue
        if s.startswith("{{") and "}}" not in s:
            end = next((j for j in range(i + 1, min(len(lines), i + max_block)) if lines[j].strip().startswith("}}")), None)
            if end is not None:
                removed += end - i + 1
                i = end + 1
                continue
        out.append(line)
        i += 1
    return out, removed


def _is_residue(lines, i):
    """Dernière ligne sans contenu : titre vide, reste de tableau, catégorie, ponctuation seule."""
    s = lines[i].strip()
    low = s.lower()
    if not any(ch.isalnum() for ch in s):  # « . », « |} »
        return True
    if low.startswith(_RESIDUE_PREFIXES):
        return True
    # Titre de section resté sans contenu : ligne courte, isolée par une ligne vide, sans ponctuation finale.
    prev_blank = i == 0 or not lines[i - 1].strip()
    return prev_blank and not lines[i].startswith(" ") and len(s) < 80 and not s.endswith(_FINAL_PUNCT)


def strip_wikipedia(text, prose_min_chars=300, keywords=WIKIPEDIA_END_KEYWORDS, vocab=WIKIPEDIA_END_VOCAB):
    """Retire les sections finales (références, liens…) et les résidus de fin d'un article.

    On coupe au premier titre de section finale après lequel il n'y a plus
    aucun paragraphe de prose : une section « Notes » longue et rédigée est
    gardée, les listes de références et de liens partent. Puis on retire les
    dernières lignes sans contenu (titres vides, restes de tableaux, catégories).
    Le wikitexte resté brut ({| |- |}, {{modèles}}) est retiré d'abord.
    Renvoie (texte, titre où l'on a coupé ou None, {residue_lines, markup_lines}).
    """
    lines = text.split("\n")
    n_markup = 0
    if "{" in text or "|" in text:
        lines, n_markup = strip_wikitext_markup(lines)
    cut = None
    candidates = [i for i, line in enumerate(lines) if is_end_heading(line, keywords, vocab)]
    if candidates:
        last_prose = -1  # dernière ligne de prose après le premier titre final
        for i in range(len(lines) - 1, candidates[0], -1):
            if _is_prose(lines[i], prose_min_chars):
                last_prose = i
                break
        for c in candidates:
            if c > last_prose:
                cut = _norm_heading(lines[c])
                lines = lines[:c]
                break
    n_residue = 0
    while lines:
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) > 1 and _is_residue(lines, len(lines) - 1):
            lines.pop()
            n_residue += 1
        else:
            break
    info = {"residue_lines": n_residue, "markup_lines": n_markup}
    if cut is None and not n_residue and not n_markup:
        return text, None, info
    while lines and not lines[0].strip():  # lignes vides laissées en tête par un modèle retiré
        lines.pop(0)
    return "\n".join(lines).rstrip(), cut, info


# ----------------------------------------------------------------------- étape

class BoilerplateStage(Stage):
    name = "boilerplate"
    uses_html = True
    filters = True
    sample_sizes = {"changed": 200}

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.methods = {**DEFAULT_METHODS, **(self.params.get("methods") or {})}
        self.traf = {**TRAFILATURA_DEFAULTS, **(self.params.get("trafilatura") or {})}
        self.guten = dict(self.params.get("gutenberg") or {})
        wiki = dict(self.params.get("wikipedia") or {})
        self.wiki = {"prose_min_chars": int(wiki.get("prose_min_chars", 300))}
        if wiki.get("keywords"):
            self.wiki["keywords"] = tuple(wiki["keywords"])
        if wiki.get("vocab"):
            self.wiki["vocab"] = tuple(wiki["vocab"])
        self.min_chars = int(self.params.get("min_chars", 1))

    def setup(self):
        logging.getLogger("trafilatura").setLevel(logging.ERROR)

    def replaces_text_from_html(self, source):
        return self.methods.get(source) == "trafilatura"

    def apply(self, doc):
        method = self.methods.get(doc["source"])
        if method == "trafilatura":
            return self._trafilatura(doc)
        if method == "gutenberg":
            return self._gutenberg(doc)
        if method == "wikipedia":
            return self._wikipedia(doc)
        return Outcome(doc)

    # -- web
    def _trafilatura(self, doc):
        html = doc.get("html")
        raw = doc.get("text") or ""
        if not html:
            return Outcome(doc, counters={"web:no_html_kept_text": 1})
        html_bytes = utf8_len(html)
        counters = {"web:html_bytes": html_bytes, "web:raw_text_bytes": utf8_len(raw)}
        if html_bytes > self.traf["max_html_bytes"]:
            return Outcome(doc, keep=False, reason="html_too_large", counters=counters)
        try:
            text = extract_main_text(html, url=doc.get("url"), params=self.traf)
        except Exception:  # pages cassées : lxml peut lever des erreurs variées
            counters["web:extract_error"] = 1
            text = None
        if not text or len(text.strip()) < self.min_chars:
            return Outcome(doc, keep=False, reason="no_extraction", counters=counters)
        if self.traf.get("fix_encoding", True):
            fixed = fix_encoding(text)
            if fixed != text:
                counters["web:encoding_fixed"] = 1
            text = fixed
        counters["web:extracted"] = 1
        doc["text"] = text
        record = {"id": doc["id"], "source": doc["source"], "url": doc.get("url"),
                  "raw_text_head": raw[:400], "extracted_head": text[:400],
                  "raw_bytes": utf8_len(raw), "extracted_bytes": utf8_len(text), "html_bytes": html_bytes}
        return Outcome(doc, counters=counters, samples=[("changed", "", record)])

    # -- livres
    def _gutenberg(self, doc):
        text = doc.get("text") or ""
        body, header, footer, info = strip_gutenberg(text, **self.guten)
        counters = {f"gutenberg:{k}": v for k, v in info.items()}
        if body == text:
            counters["gutenberg:no_marker"] = 1
            return Outcome(doc, counters=counters)
        if len(body.strip()) < self.min_chars:
            return Outcome(doc, keep=False, reason="empty_after_boilerplate", counters=counters)
        record = {"id": doc["id"], "source": doc["source"],
                  "removed_header_tail": header[-600:], "body_head": body[:300],
                  "body_tail": body[-300:], "removed_footer_head": footer[:600],
                  "bytes_removed": utf8_len(text) - utf8_len(body)}
        doc["text"] = body
        return Outcome(doc, counters=counters, samples=[("changed", "", record)])

    # -- encyclopédie
    def _wikipedia(self, doc):
        text = doc.get("text") or ""
        body, cut, info = strip_wikipedia(text, **self.wiki)
        n_residue, n_markup = info["residue_lines"], info["markup_lines"]
        if body == text:
            return Outcome(doc, counters={"wikipedia:unchanged": 1})
        counters = Counter()
        if cut is not None:
            counters["wikipedia:cut"] += 1
            counters[f"wikipedia:cut_at:{cut}"] += 1
        if n_residue:
            counters["wikipedia:residue_docs"] += 1
            counters["wikipedia:residue_lines"] += n_residue
        if n_markup:
            counters["wikipedia:markup_docs"] += 1
            counters["wikipedia:markup_lines"] += n_markup
        if len(body.strip()) < self.min_chars:
            return Outcome(doc, keep=False, reason="empty_after_boilerplate", counters=dict(counters))
        removed = text[len(body):] if text.startswith(body) else ""
        counters["wikipedia:bytes_removed"] += utf8_len(text) - utf8_len(body)
        record = {"id": doc["id"], "source": doc["source"], "cut_at": cut or "", "residue_lines": n_residue,
                  "markup_lines": n_markup, "kept_tail": body[-300:], "removed": removed[:800],
                  "bytes_removed": utf8_len(text) - utf8_len(body)}
        doc["text"] = body
        return Outcome(doc, counters=dict(counters), samples=[("changed", "", record)])

    # -- résumé lisible
    def summarize(self, counters):
        out, cut_at = {}, Counter()
        for k, v in counters.items():
            if k.startswith("wikipedia:cut_at:"):
                cut_at[k[len("wikipedia:cut_at:"):]] += v
            else:
                out[k] = v
        if cut_at:
            out["wikipedia:cut_at"] = dict(cut_at.most_common(25))
        return dict(sorted(out.items()))

    def describe(self, st):
        c = st.get("counters", {})
        removed_kept = st["bytes_in"] - st["bytes_out"] - st.get("bytes_removed_docs", 0)
        mb = f"{removed_kept / 1e6:.2f}".replace(".", ",")
        if c.get("web:extracted") or c.get("web:html_bytes"):
            raw = c.get("web:raw_text_bytes", 0) / 1e6
            out = st["bytes_out"] / 1e6
            return (f"trafilatura : {c.get('web:extracted', 0)} pages extraites sur {st['docs_in']} ; "
                    f"texte brut {raw:.2f} Mo -> extrait {out:.2f} Mo".replace(".", ","))
        if any(k.startswith("gutenberg:") for k in c):
            return (f"en-tête trouvé {c.get('gutenberg:start_marker', 0)}, pied {c.get('gutenberg:end_marker', 0)}, "
                    f"sans marqueur {c.get('gutenberg:no_marker', 0)} ; {mb} Mo retirés")
        if any(k.startswith("wikipedia:") for k in c):
            return (f"sections finales coupées dans {c.get('wikipedia:cut', 0)} articles, "
                    f"résidus de fin dans {c.get('wikipedia:residue_docs', 0)}, "
                    f"wikitexte brut dans {c.get('wikipedia:markup_docs', 0)} ; {mb} Mo retirés")
        return "source inchangée"
