"""Wikitexte de Wikisource → texte brut (mwparserfromhell).

On garde le texte affiché : contenu des modèles de mise en forme ({{c|…}}, {{sc|…}}, {{Centré|…}}),
noms des personnages ({{Personnage|Alceste}} → ALCESTE.), mots coupés entre deux pages
({{tiret|mo|t}} + {{tiret2|mo|t}} → « mot »). On retire la navigation, les notes (<ref>),
les catégories, les liens interlangues et les modèles de mise en page. Conversion imparfaite :
les modèles rares ou inconnus gardent leur premier paramètre s'il ressemble à du texte ; ceux dont le
texte affiché est ailleurs ({{lp}}, {{style}}, {{StdT}}…) sont dans DISPLAY_PARAM ; les balises HTML
orphelines et les longueurs CSS seules sur leur ligne sont retirées après rendu (scrub)."""
import re

import mwparserfromhell as mwp
from mwparserfromhell.nodes import (Argument, Comment, ExternalLink, Heading, HTMLEntity, Tag, Template, Text,
                                    Wikilink)

NOINCLUDE = re.compile(r"<noinclude\s*>.*?</noinclude\s*>", re.S | re.I)
QUALITY = re.compile(r"<pagequality\s+level\s*=\s*[\"']?(\d)", re.I)
MAGIC = re.compile(r"__[A-ZÉÈ]+__")
CSSISH = re.compile(r"^\s*[\d.,]+\s*(em|ex|px|pt|%|cm|mm)?\s*$", re.I)
CSS_LEN = re.compile(r"^\s*[\d.,]+\s*(em|ex|px|pt)\s*$", re.I)  # « 0.1em » : une longueur, jamais du texte
# Restes après rendu : balise HTML ouverte sur une page et fermée sur une autre (mwparserfromhell la laisse
# en texte), ligne qui n'est qu'une longueur CSS ou un paramètre d'image (« 200px|center »).
_ATTRS = r"""(?:\s+[\w:-]+(?:\s*=\s*(?:"[^"\n]*"|'[^'\n]*'|[^\s"'<>]+))?)*"""
LEFTOVER_TAG = re.compile(r"</?(div|p|center|blockquote|table|tr|td|th|br|hr|poem|span|font|small|big|abbr|b|i|u|s"
                          r"|sup|sub|section|includeonly|noinclude|onlyinclude)\b" + _ATTRS + r"\s*/?>", re.I)
LEFTOVER_LINE = re.compile(r"^[ \t]*(?:[\d.,]+[ \t]*(?:em|ex|px|pt)|\d+px[ \t]*\|[^\n]*)[ \t]*(?:\n|$)", re.M | re.I)
HAS_LETTER = re.compile(r"[^\W\d_]")
_SPACES = re.compile(r"[ \t​]+")

DROP_TAGS = {"ref", "references", "math", "gallery", "score", "timeline", "syntaxhighlight", "source", "hiero",
             "pages", "section", "templatestyles", "imagemap", "graph", "mapframe", "maplink", "chem", "ce",
             "noinclude", "indicator", "categorytree", "inputbox", "pagequality", "img", "style", "script"}
BLOCK_TAGS = {"p", "div", "center", "blockquote", "poem", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
              "pre", "ul", "ol", "dl", "caption"}
CELL_TAGS = {"td", "th"}
LINK_DROP = ("catégorie:", "category:", "fichier:", "file:", "image:", "média:", "media:")
INTERPROJECT = {"w", "wikt", "s", "b", "q", "n", "v", "c", "commons", "d", "mw", "meta", "oldwikisource",
                "wikipedia", "wiktionary", "wikisource", "wikilivres", "wikiquote", "auteur", "author", "page",
                "livre", "portail", "wikisource", "special", "spécial"}
INTERLANG = re.compile(r"^:?[a-z]{2,3}(-[a-z]{2,8})?:", re.I)

# Modèles sans paramètre qui affichent un texte (abréviations typographiques)
ZERO_ARG = {"e": "e", "er": "er", "re": "re", "ers": "ers", "res": "res", "es": "es", "me": "me", "mes": "mes",
            "nd": "nd", "nde": "nde", "o": "o", "os": "os", "a": "a", "1er": "1er", "1re": "1re", "1o": "1o",
            "2o": "2o", "3o": "3o", "m.": "M.", "mM.": "MM.", "mme": "Mme", "mmes": "Mmes", "mlle": "Mlle",
            "mlles": "Mlles", "mgr": "Mgr", "dr": "Dr", "cie": "Cie", "n°": "n°", "no": "no", "nos": "nos",
            "lié": " ", "br0": "\n", "br": "\n", "saut": "\n", "!": "|", "=": "=", "etc": "etc.",
            "œ": "œ", "æ": "æ", "sic": "[sic]", "vs": "vs", "id.": "Id.", "ibid.": "Ibid."}
SUPERSCRIPT = {"e", "er", "re", "exp", "sup", "ex", "o", "me", "nd"}
TITLE_ABBR = {"m.", "mM.", "mme", "mmes", "mlle", "mlles", "mgr", "dr"}
# Modèles « bloc » (rendus comme un <div>) : sur leur propre ligne
BLOCK_TEMPLATES = {"c", "centré", "center", "c3", "c2", "t1", "t2", "t3", "t4", "t5", "t6", "droite", "d", "gauche",
                   "bloc centré", "centré-bloc", "titre-centré", "citation", "bloc", "encadré", "retrait", "alinéa",
                   "paragraphe", "bloc smaller", "texteADroite", "texte à droite", "centrer", "personnages",
                   "titreModifié", "titrePoeme2", "discours", "colG", "colD", "sA", "sansAlinéa"}
POEM_TEMPLATES = {"poem", "poème", "poeme", "poésie", "vers", "pom"}  # {{Pom|vers…|df}} = <poem>
# Modèles dont le texte affiché n'est PAS le 1er paramètre (vérifié sur leur code, fr.wikisource, 03/11) :
# indices des paramètres positionnels à essayer, dans l'ordre ; le premier non vide est affiché.
DISPLAY_PARAM = {"lp": (3, 2), "lien page": (3, 2),        # {{lp|cible|page|texte|affiché}}
                 "lia": (2,), "lien ancre": (2,),          # {{lia|cible|ancre|texte|page}}
                 "hwp": (1, 0), "dPRML": (1, 0),           # {{hwp|cible|affiché}}, sinon la cible
                 "style": (1, 0),                          # {{style|css|texte}} ou {{style|texte}}
                 "stdT": (1,)}                             # {{StdT|s|texte}} : 1er paramètre = taille
LIGATURES = {"ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "et": "&"}  # {{ligat|o|e}} ; autres lettres recollées
# Modèles en ligne connus (premier paramètre = texte affiché) : pas comptés comme inconnus
KNOWN_INLINE = {"sc", "pc", "g", "gras", "i", "it", "italique", "u", "souligné", "t", "taille", "fs", "small",
                "smaller", "larger", "big", "x-larger", "xx-larger", "espacé", "esp", "es", "sp", "rom", "rom-maj",
                "rom-min", "romain", "lettrine", "polytonique", "grec", "he", "hébreu", "ar", "arabe", "vert", "bleu",
                "rouge", "encL", "encL2", "p", "nobr", "inline", "abréviation", "abr", "sic", "exp", "sup", "ind",
                "souscrit", "gr", "lang-grc", "latin", "la", "aN", "essaisA", "essaisB", "essaisC", "a", "x",
                "petites capitales", "gris", "cach"}
SINGLE_NL = re.compile(r"(?<!\n)\n(?!\n)")
BETWEEN_BLOCKS = re.compile(r"\n[ \t]+\n")
# Modèles à retirer (navigation, mise en page, méta)
DROP = {"textquality", "navigateur", "interprojet", "interwiki-info", "version sonore", "validé", "voir homonymes",
        "homonymie", "autreVersion", "versionsLaFontaine", "titre sous-page", "chapitreNav", "chapitre", "export",
        "bon pour export", "l2s", "l2S", "livre2Scanné", "nav", "indexAlpha2", "références", "option", "ancre",
        "ancre+", "sautDePage", "sommaire2ADroite", "p début", "p fin", "page", "hhakAvantTransclusion",
        "hhakAprèsTransclusion", "hhakPaletteNavigation", "dP-ONU", "4/4", "3/4", "2/4", "1/4", "0/4", "—", "-",
        "brn", "interligne", "pli", "sep", "séparateur", "statGr", "modernisation", "bASEPAGENAME", "pAGENAME",
        "sUBPAGENAME", "fULLPAGENAME", "img float", "nAD", "œuvre", "éditions", "voir éditions",
        "voir traductions", "document", "liste documents début", "liste documents fin", "auteur", "caché", "em",
        "refl", "refa", "rDDM2", "tSP", "numVers", "nTD", "légal", "ordinal", "angle", "tDM", "tdm", "clear",
        "efface", "haut2Colonnes", "deuxColonnes", "colonnes", "fin colonnes", "début colonnes", "nobr début",
        "nobr fin", "fs début", "fs fin", "taille début", "taille fin", "bloc début", "bloc fin", "centré début",
        "centré fin", "pointsdesuite", "sommaire", "sommaire début", "sommaire fin", "erratum", "corrections",
        "proofreadpage_index_template", "mediaWiki:Proofreadpage_index_template", ":MediaWiki:Proofreadpage_index_template",
        "nr début", "nr fin", "img", "image", "illustration", "gravure", "figure", "pagination", "nobr",
        "espace", "espacement", "vertical", "dts", "iA", "gallica", "googleBooks", "bNF", "commons",
        "wikidata", "lien web", "c-début", "c-fin", "romain début", "romain fin", "hr", "fin", "début",
        "personnages début", "personnages fin", "rom-maj début", "rom-maj fin", "titre-ligne", "sommaire2",
        "refancre", "manchette droite", "manchette gauche", "manchette", "petiteNote", "note", "nDT", "nDE",
        "ref", "réf", "réf.", "appel", "noExport", "navigateurFin", "portail", "traductions", "chercher",
        "art libre", "traduction en cours", "index", "articleCatégorie", "wD", "séparateur personnalisé",
        "ts", "tableauStyle"}  # {{ts|ac|bb}} : attributs de style d'une cellule de tableau


def _canon(name: str) -> str:
    """Nom de modèle canonique : sans espaces autour, '_' → ' ', première lettre en minuscule."""
    n = name.strip().replace("_", " ")
    if n.lower().startswith("modèle:") or n.lower().startswith("template:"):
        n = n.split(":", 1)[1].strip()
    return n[:1].lower() + n[1:] if n else n


def _positional(tpl):
    return [p.value for p in tpl.params if not p.showkey]


def _named(tpl, key):
    for p in tpl.params:
        if p.showkey and str(p.name).strip() == key:
            return p.value
    return None


class Renderer:
    """Rend un arbre mwparserfromhell en texte ; compte les modèles inconnus (diagnostic).

    Comme MediaWiki : hors <poem>, un saut de ligne simple est une espace (même paragraphe) ;
    une ligne vide sépare deux paragraphes."""

    def __init__(self):
        self.unknown = {}
        self.poem = 0

    def render(self, code) -> str:
        if code is None:
            return ""
        return "".join(self.node(n) for n in code.nodes)

    def node(self, n) -> str:
        if isinstance(n, Text):
            v = MAGIC.sub("", str(n.value))
            return v if self.poem else SINGLE_NL.sub(" ", v)
        if isinstance(n, Template):
            return self.template(n)
        if isinstance(n, Wikilink):
            return self.wikilink(n)
        if isinstance(n, Tag):
            return self.tag(n)
        if isinstance(n, HTMLEntity):
            try:
                return n.normalize()
            except ValueError:
                return ""
        if isinstance(n, ExternalLink):
            return self.render(n.title) if n.title is not None else ""
        if isinstance(n, Heading):
            return "\n" + self.render(n.title).strip() + "\n"
        if isinstance(n, (Comment, Argument)):
            return ""
        return ""

    def wikilink(self, n) -> str:
        target = str(n.title).strip()
        low = target.lower().lstrip(":")
        if low.startswith(LINK_DROP):
            return ""
        if n.text is not None:
            return self.render(n.text)
        m = INTERLANG.match(target)
        if m:
            prefix = target.lstrip(":").split(":", 1)[0].lower()
            if prefix not in INTERPROJECT:
                return ""  # lien interlangue [[en:…]] : non affiché
            return target.lstrip(":").split(":", 1)[1]
        return target.lstrip(":").split("#")[0]

    def tag(self, n) -> str:
        name = str(n.tag).strip().lower()
        if name in DROP_TAGS:
            return ""
        if name in ("br", "hr"):
            return "\n"
        if name in ("li", "dd", "dt") and n.wiki_markup:
            return "\n"  # puce de liste wiki : le contenu suit dans le texte, sur sa propre ligne
        if name == "poem":
            self.poem += 1
            try:
                inner = self.render(n.contents) if n.contents is not None else ""
            finally:
                self.poem -= 1
            return "\n" + inner + "\n"
        inner = self.render(n.contents) if n.contents is not None else ""
        if name in CELL_TAGS:
            return " " + inner.strip() + " "
        if name in BLOCK_TAGS or name in ("li", "dd", "dt"):
            return "\n" + inner + "\n"
        return inner

    def first_text(self, tpl, idx=0):
        pos = _positional(tpl)
        return self.render(pos[idx]) if len(pos) > idx else ""

    def template(self, t) -> str:
        raw = str(t.name).strip()
        name = _canon(raw)
        if name.startswith("#") or name.split(":", 1)[0].upper() in ("DEFAULTSORT", "DISPLAYTITLE", "CLEDETRI"):
            return ""  # fonctions parseur ({{#if:…}}, {{#tag:ref…}}) et mots magiques
        if name in POEM_TEMPLATES:
            self.poem += 1
            try:
                return "\n" + "\n".join(self.render(p) for p in _positional(t)[:1]) + "\n"
            finally:
                self.poem -= 1
        if name == "mpom":  # groupe de vers dans un poème ; {{Mpom|—|4}} = filet séparateur
            self.poem += 1
            try:
                txt = self.render(_positional(t)[0]) if _positional(t) else ""
            finally:
                self.poem -= 1
            return "" if txt.strip() in ("", "—") else txt
        if name in ("Œ", "œ", "Æ", "æ") or raw in ("Œ", "Æ"):
            return raw
        pos = _positional(t)
        if name in DROP or name.lower() in DROP:
            return ""
        if name == "ligat":  # {{ligat|f|f}} → « ff » (avant : « f », une lettre perdue)
            s = "".join(self.render(p).strip() for p in pos[:3])
            return LIGATURES.get(s, s)
        if name in DISPLAY_PARAM:
            for i in DISPLAY_PARAM[name]:
                txt = self.render(pos[i]) if len(pos) > i else ""
                if txt.strip():
                    return txt
            return ""
        if name in SUPERSCRIPT and pos:
            return self.render(pos[0])
        if name in ZERO_ARG and not pos:
            return ZERO_ARG[name]
        if name in TITLE_ABBR:  # {{M.|Dupont}} → « M. Dupont »
            return ZERO_ARG[name] + "\u00a0" + " ".join(self.render(p).strip() for p in pos)
        if name.endswith(".") and 1 < len(name) <= 6 and name[:-1].isalpha() and pos:
            # abréviations {{fig.|21}} → « fig. 21 », {{t.|3}} → « t. 3 » ; options « cap » (majuscule), « inv »
            args = [self.render(p).strip() for p in pos]
            abbr = raw[:1].upper() + raw[1:] if "cap" in args[1:] else raw
            if not args[0]:
                return abbr
            return f"{args[0]} {abbr}" if "inv" in args[1:] else f"{abbr} {args[0]}"
        if name == "lié":
            return "\u00a0".join(self.render(p).strip() for p in pos)
        if name in ("tiret",):
            return "".join(self.render(p).strip() for p in pos[:2])
        if name in ("tiret2",):
            return ""
        if name in ("corr", "corrigé", "coquille", "corrigé2", "sic2"):
            return self.render(pos[1]) if len(pos) > 1 else (self.render(pos[0]) if pos else "")
        if name in ("lang", "langue", "lang-fr", "lien langue"):
            v = _named(t, "texte")
            return self.render(v) if v is not None else (self.render(pos[1]) if len(pos) > 1 else "")
        if name in ("personnage", "personnageD", "personnage2", "pers", "Pers"):
            who = self.render(pos[0]).strip().upper() if pos else ""
            didasc = self.render(pos[2]).strip() if len(pos) > 2 else ""
            if not who:
                return ""
            return "\n" + who + (", " + didasc if didasc else "") + ("" if who.endswith(".") else ".") + "\n"
        if name in ("acteurs",):
            return "\n" + (self.render(pos[0]).strip().upper() if pos else "") + "\n"
        if name in ("scène", "scene"):
            return "\nSCÈNE " + (self.render(pos[0]).strip() if pos else "") + ".\n"
        if name in ("acte",):
            return "\nACTE " + (self.render(pos[0]).strip() if pos else "") + ".\n"
        if name in ("titre", "titre2", "titreCentré"):
            return "\n" + "\n".join(x for x in (self.render(p).strip() for p in pos[:3]) if x) + "\n"
        if name in ("table", "table2", "tableau"):
            v = _named(t, "titre")
            if v is None and pos:
                v = pos[-1]
            return ("\n" + self.render(v).strip() + "\n") if v is not None else ""
        if name in ("s", "siècle", "s-"):
            return (self.render(pos[0]).strip() + "e siècle") if pos else ""
        if name in ("roi", "nr", "ligne", "gauche-droite", "centre-droite", "brisé", "nr2"):
            return " ".join(x for x in (self.render(p).strip() for p in pos) if x)
        if name in ("uc", "maj", "majuscules", "capitales"):
            return self.first_text(t).upper()
        if name in ("lc", "minuscules"):
            return self.first_text(t).lower()
        if name in ("coloré", "couleur", "color"):
            return self.first_text(t, 1)
        if name in ("didascalie", "didas", "did", "di"):
            return self.first_text(t)
        # par défaut : premier paramètre positionnel s'il ressemble à du texte
        if pos:
            txt = self.render(pos[0])
            if CSS_LEN.match(txt):
                return ""
            if name in BLOCK_TEMPLATES:
                return "\n" + txt + "\n"
            if name in KNOWN_INLINE:
                return txt
            if HAS_LETTER.search(txt) and not CSSISH.match(txt):
                self.unknown[name] = self.unknown.get(name, 0) + 1
                return txt
        return ""


def tidy(text: str) -> str:
    """Espaces normalisés par ligne, au plus une ligne vide de suite."""
    lines = [_SPACES.sub(" ", line).strip() for line in text.replace("\r", "").split("\n")]
    out, blank = [], 0
    for line in lines:
        if line:
            out.append(line)
            blank = 0
        else:
            blank += 1
            if blank == 1 and out:
                out.append("")
    return "\n".join(out).strip()


def scrub(text: str) -> str:
    """Retire les restes de balisage après rendu : balises HTML orphelines (<br> → saut de ligne),
    lignes qui ne sont qu'une longueur CSS (« 0.1em ») ou un paramètre d'image (« 200px|center »)."""
    text = LEFTOVER_TAG.sub(lambda m: "\n" if m.group(1).lower() in ("br", "hr") else "", text)
    return LEFTOVER_LINE.sub("", text)


def to_text(wikitext: str, renderer: Renderer | None = None) -> str:
    """Wikitexte → texte brut."""
    renderer = renderer or Renderer()
    try:
        code = mwp.parse(wikitext, skip_style_tags=False)
    except Exception:  # noqa: BLE001 — wikitexte pathologique : on rend le texte nu
        return tidy(re.sub(r"\{\{[^{}]*\}\}|\[\[[^\]]*\]\]|<[^>]+>", "", wikitext))
    # un saut de ligne simple entre deux blocs ({{c|…}}\n{{c|…}}) ne crée pas de paragraphe
    return tidy(BETWEEN_BLOCKS.sub("\n", scrub(renderer.render(code))))


def page_body(wikitext: str):
    """(niveau de relecture, corps) d'une page de l'espace Page: (en-tête et pied <noinclude> retirés)."""
    m = QUALITY.search(wikitext)
    level = int(m.group(1)) if m else None
    return level, NOINCLUDE.sub("", wikitext)


def join_pages(texts) -> str:
    """Recolle les pages d'un livre comme la transclusion de Wikisource : la phrase continue si la page
    suivante commence par une minuscule (espace) ou par une ponctuation de fin (collée) ; sinon saut de ligne."""
    out = ""
    for t in texts:
        if not t:
            continue
        if not out:
            out = t
            continue
        last = out.rstrip()[-1:]
        first = t.lstrip()[:1]
        if last == "-" and out.rstrip()[-2:-1].isalpha() and first.islower():
            # mot coupé en fin de page sans {{tiret}} : « principa- » + « lement » → « principalement »
            out = out.rstrip()[:-1] + t.lstrip()
            continue
        if first in ",.)…]":
            sep = ""
        elif first in ";:!?»" or (last and (last.isalnum() or last in ",-") and first.islower()):
            sep = " "
        else:
            sep = "\n"
        out = out.rstrip() + sep + t.lstrip()
    return out


SPEAKER = re.compile(r"^[A-ZÀ-ÖØ-ÞŒ][A-ZÀ-ÖØ-ÞŒ' \-]{1,40}(,[^\n]{0,80})?\.?$")
SCENE = re.compile(r"^(ACTE|SCÈNE|SCENE)\b", re.I)


def theatre_stats(text: str):
    """(lignes « NOM DU PERSONNAGE » en capitales, marqueurs ACTE/SCÈNE, lignes non vides)."""
    n_sp = n_sc = n = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        n += 1
        if SCENE.match(line):
            n_sc += 1
        elif SPEAKER.match(line) and sum(c.isupper() for c in line.split(",")[0]) >= 2:
            n_sp += 1
    return n_sp, n_sc, n


def is_theatre(text: str) -> bool:
    """Heuristique : beaucoup de lignes « NOM. » en capitales (≥ 20 et ≥ 4 % des lignes) et au moins une scène."""
    n_sp, n_sc, n = theatre_stats(text)
    return n_sp >= 20 and n and n_sp / n >= 0.04 and n_sc >= 1
