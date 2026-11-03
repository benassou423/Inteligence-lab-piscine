"""Étape `encoding` : réparation du mojibake (ftfy), normalisation NFC et des espaces.

La typographie française est conservée : apostrophe ’, guillemets « », tirets — et –,
points de suspension …. D'où `uncurl_quotes=False`.

Les espaces Unicode (insécable U+00A0, fine U+202F, fine U+2009…) deviennent une
espace simple et les caractères invisibles (trait d'union conditionnel, espace sans
chasse) disparaissent : la val publique du kit n'en contient aucun (0 sur 813 Ko),
alors que ’ (4 383), « (357), – (195) et — (26) y sont.

ftfy est lent (~1 Mo/s par cœur). Un pré-filtre exact évite de l'appeler sur
les textes qu'il ne modifierait pas : chaque correcteur de ftfy ne s'active que
sur des caractères précis, et la réparation d'encodage seulement si
`ftfy.badness.is_bad` détecte du mojibake. Résultat identique, ~5x plus rapide.

Un faux positif de ftfy est refusé : il lit « »Éminence » (guillemet de continuation des
vieux livres) comme du MacRoman et en fait « ȃminence » (15 livres Gutenberg abîmés,
mesuré sur tout le corpus). Une réparation MacRoman ou cp437 qui fait disparaître un
guillemet « ou » est donc annulée pour cette ligne (les autres corrections restent).
"""
import re
import unicodedata
from collections import Counter

import ftfy
from ftfy import TextFixerConfig, chardata
from ftfy.badness import is_bad

from pipeline.clean.base import Outcome, Stage, excerpt_diff


def _char_class(codepoints, surrogates=True):
    """Construit une classe de caractères regex à partir d'une liste de points de code."""
    cps = sorted(set(codepoints))
    parts, i = [], 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        a, b = chr(cps[i]), chr(cps[j])
        parts.append(re.escape(a) if i == j else f"{re.escape(a)}-{re.escape(b)}")
        i = j + 1
    return "[" + "".join(parts) + ("\ud800-\udfff" if surrogates else "") + "]"  # surrogates : fix_surrogates


# Caractères qui déclenchent un correcteur de ftfy (hors mojibake et NFC) :
# contrôles C1 (fix_c1_controls), ligatures, pleine chasse, caractères de contrôle,
# fins de ligne \r U+2028 U+2029 (fix_line_breaks), échappements terminal \x1b.
_TRIGGER_CHARS = (
    set(chardata.LIGATURES)
    | set(chardata.WIDTH_MAP)
    | set(chardata.CONTROL_CHARS)
    | set(range(0x80, 0xA0))
    | {0x0D, 0x2028, 0x2029, 0x1B}
)
_TRIGGER_RE = re.compile(_char_class(_TRIGGER_CHARS))
_ENTITY_RE = chardata.HTML_ENTITY_RE
# ftfy traite le texte ligne par ligne ; le seul motif ancré de sa regex de
# mojibake (`^[ÃÂ][ ]`) doit donc être cherché en début de chaque ligne.
_LINE_START_MOJIBAKE_RE = re.compile("^[ÃÂ] ", re.MULTILINE)


def make_config(params=None):
    """Configuration ftfy du projet. `uncurl_quotes` reste False : on garde ’ “ ”."""
    p = dict(params or {})
    return TextFixerConfig(
        uncurl_quotes=bool(p.get("uncurl_quotes", False)),
        normalization=p.get("normalization", "NFC"),
        unescape_html=p.get("unescape_html", "auto"),
        fix_latin_ligatures=bool(p.get("fix_latin_ligatures", True)),
        fix_character_width=bool(p.get("fix_character_width", True)),
        explain=False,
    )


DEFAULT_CONFIG = make_config()


def triggers(text, config=DEFAULT_CONFIG):
    """Liste des raisons pour lesquelles ftfy pourrait modifier ce texte (vide = rien à faire)."""
    found = []
    if _TRIGGER_RE.search(text):
        found.append("chars")
    if config.unescape_html and _ENTITY_RE.search(text):
        found.append("html_entity")
    if config.normalization and not unicodedata.is_normalized(config.normalization, text):
        found.append("not_nfc")
    if not text.isascii() and (is_bad(text) or _LINE_START_MOJIBAKE_RE.search(text)):
        found.append("mojibake")
    return found


# Encodages d'où viennent les faux positifs mesurés (« »Éminence » -> « ȃminence »).
RISKY_ENCODINGS = ("macroman", "cp437")
_GUILLEMETS = ("«", "»")


def _risky(segment, fixed, steps):
    """Réparation à refuser : relecture MacRoman/cp437 qui fait disparaître un guillemet « ou »."""
    if not steps or not any(step == ("encode", enc) for step in steps for enc in RISKY_ENCODINGS):
        return False
    return sum(fixed.count(g) for g in _GUILLEMETS) < sum(segment.count(g) for g in _GUILLEMETS)


def _fix_lines(text, config):
    """ftfy.fix_text, ligne par ligne exactement comme ftfy, sans les réparations risquées."""
    cfg = config._replace(explain=True)
    out, pos = [], 0
    while pos < len(text):
        end = text.find("\n", pos) + 1 or len(text)
        end = min(end, pos + cfg.max_decode_length)
        segment = text[pos:end]
        if cfg.unescape_html == "auto" and "<" in segment:  # même règle que ftfy.fix_text
            cfg = cfg._replace(unescape_html=False)
        fixed, steps = ftfy.fix_and_explain(segment, cfg)
        if _risky(segment, fixed, steps):
            fixed, _ = ftfy.fix_and_explain(segment, cfg._replace(fix_encoding=False))
        out.append(fixed)
        pos = end
    return "".join(out)


def fix_text(text, config=DEFAULT_CONFIG):
    """Répare le texte. Identique à ftfy.fix_text(text, config), en plus rapide, sauf les
    réparations MacRoman/cp437 qui mangent un guillemet (faux positifs, refusées)."""
    if not text:
        return text
    # Fins de ligne Windows (très fréquentes) : remplacées d'avance comme le ferait
    # fix_line_breaks, sauf s'il reste des \r isolés (ftfy s'en charge alors).
    if "\r\n" in text:
        unix = text.replace("\r\n", "\n")
        if "\r" not in unix:
            text = unix
    found = triggers(text, config)
    if not found:
        return text
    if "mojibake" not in found:  # sans mojibake détecté, ftfy ne relit pas l'encodage
        return ftfy.fix_text(text, config)
    return _fix_lines(text, config)


# Toutes les espaces Unicode (catégorie Zs) sauf l'espace ASCII -> espace simple ;
# caractères invisibles (trait d'union conditionnel, espace sans chasse, gluon, BOM) -> supprimés.
SPACE_CHARS = tuple(cp for cp in range(0x110000) if unicodedata.category(chr(cp)) == "Zs" and cp != 0x20)
INVISIBLE_CHARS = (0x00AD, 0x200B, 0x2060, 0xFEFF)
_SPACE_TABLE = {**{cp: " " for cp in SPACE_CHARS}, **{cp: None for cp in INVISIBLE_CHARS}}
_SPACE_RE = re.compile(_char_class(SPACE_CHARS + INVISIBLE_CHARS, surrogates=False))


def normalize_spaces(text):
    """Espaces Unicode -> espace simple ; invisibles supprimés."""
    return text.translate(_SPACE_TABLE) if _SPACE_RE.search(text) else text


def clean_text(text, config=DEFAULT_CONFIG, spaces=True):
    """Réparation ftfy (identique à ftfy.fix_text) puis normalisation des espaces."""
    fixed = fix_text(text, config)
    return normalize_spaces(fixed) if spaces else fixed


def explain_fix(text, config=DEFAULT_CONFIG):
    """Compte les corrections appliquées, ligne par ligne (seulement pour les docs modifiés)."""
    kinds = Counter()
    cfg = config._replace(explain=True)
    for line in text.replace("\r\n", "\n").splitlines(keepends=True):
        if not triggers(line, config):
            continue
        fixed, steps = ftfy.fix_and_explain(line, cfg)
        if _risky(line, fixed, steps):  # réparation refusée par fix_text : on ne la compte pas
            fixed, steps = ftfy.fix_and_explain(line, cfg._replace(fix_encoding=False))
        if fixed == line or not steps:
            continue
        names = set()
        for step in steps:
            action, detail = step
            if action in ("encode", "decode", "transcode"):
                names.add("mojibake")
            elif action == "normalize":
                names.add(f"normalize_{detail}")
            else:
                names.add(detail)
        kinds.update(names)
    return kinds


class EncodingStage(Stage):
    """Réparation d'encodage et NFC sur `text`.

    Les documents dont le texte sera ré-extrait du HTML plus loin (commoncrawl)
    sont laissés tels quels ici : l'étape boilerplate applique la même
    réparation sur le texte extrait.
    """

    name = "encoding"
    sample_sizes = {"changed": 200}

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.config = make_config(self.params)
        if self.config.uncurl_quotes:
            raise ValueError("encoding : uncurl_quotes doit rester False (typographie française du test)")
        self.spaces = bool(self.params.get("normalize_spaces", True))
        self.deferred_sources = set()

    def bind(self, later_stages):
        super().bind(later_stages)
        self.deferred_sources = {
            s for s in self.ctx.sources if any(st.replaces_text_from_html(s) for st in later_stages)
        }

    def fingerprint_extra(self, source=None):
        # Ce que fait l'étape pour une source dépend des étapes suivantes (texte ré-extrait du HTML
        # plus loin ou non) : sans cela, changer la méthode de boilerplate laisserait un cache périmé.
        return {"deferred": source in self.deferred_sources}

    def apply(self, doc):
        if doc["source"] in self.deferred_sources and doc.get("html"):
            return Outcome(doc, counters={"deferred_to_boilerplate": 1})
        text = doc.get("text") or ""
        repaired = fix_text(text, self.config)
        fixed = normalize_spaces(repaired) if self.spaces else repaired
        if fixed == text:
            return Outcome(doc)
        counters = Counter({"modified": 1})
        if repaired != text:
            for kind in explain_fix(text, self.config):
                counters[f"fix:{kind}"] += 1
            if "\r\n" in text:
                counters["fix:crlf"] += 1
        if fixed != repaired:
            counters["fix:spaces"] += 1
        before, after = excerpt_diff(text, fixed)
        doc["text"] = fixed
        record = {"id": doc["id"], "source": doc["source"], "before": before, "after": after,
                  "fixes": sorted(k[4:] for k in counters if k.startswith("fix:"))}
        return Outcome(doc, counters=dict(counters), samples=[("changed", "", record)])

    def describe(self, st):
        c = st.get("counters", {})
        if c.get("deferred_to_boilerplate"):
            return f"{c['deferred_to_boilerplate']} docs réparés après extraction HTML (étape boilerplate)"
        if not c.get("modified"):
            return "aucun document modifié"
        fixes = sorted(((k[4:], v) for k, v in c.items() if k.startswith("fix:")), key=lambda kv: -kv[1])
        detail = ", ".join(f"{k} {v}" for k, v in fixes[:4])
        return f"{c['modified']} docs modifiés ({detail})"
