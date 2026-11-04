"""Modèles Wikipédia : valeurs conservées, décor supprimé, inconnus comptés.

Pas d'exécution de Lua/Wikidata : une valeur absente du dump n'est jamais inventée.
"""
import re

from pipeline.collect import wikitext as wt

MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
SUPER = str.maketrans("0123456789+-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻")
DROP = {"références", "references", "référence souhaitée", "référence à confirmer", "refconf", "note", "efn",
        "article détaillé", "article connexe", "articles connexes", "voir aussi", "voir homonymes", "autre",
        "autre4", "autres projets", "homonyme", "homonymie", "à ne pas confondre", "confusion", "voir famille",
        "sommaire", "sommaire flottant", "sommaire à droite", "clr", "clear", "ancre", "ancre+", "sfn",
        "harvsp", "harv", "harvnb", "harvtxt", "ouvrage", "article", "lien web", "lien vidéo", "lien brisé",
        "lien archive", "légende", "légende plume", "plume", "images", "image", "double image", "triple image",
        "début de colonnes", "fin de colonnes", "colonnes", "colonnes début", "colonnes fin", "réflist",
        "début citation", "fin citation", "début encadré", "fin encadré", "début tableau", "fin tableau",
        "titre en italique", "titre mis en forme", "titre incorrect", "bon article", "article de qualité",
        "coord", "coordonnées", "géolocalisation", "autorité", "bases", "dictionnaires", "notice",
        "commons", "wikiquote", "wiktionnaire", "wikisource", "wikidata", "liens", "réf", "référence",
        ",", "...", "traduction/référence", "bioref", "sommaire alphabétique", "langue du titre",
        "boîte déroulante/début", "boîte déroulante/fin", "fonctionnaire", "fstats", "foot résultat",
        "foot classement", "volley résultat 2", "palmarès wta/ligne", "sources à lier", "rédaction",
        "à recycler", "à wikifier", "section vide", "section incomplète", "section vide ou incomplète",
        "sources secondaires", "problèmes multiples", "promotionnel", "ethno", "uicn", "itis", "ncbi",
        "catalogueoflife", "biolib", "adw", "bases sport", "bases audiovisuel", "site officiel",
        "élu", "historique tournoi tennis/base", "fstats total", "aéroport-dest-liste", "…", "-d"}
DROP_PREFIX = ("infobox", "palette", "portail", "ébauche", "ébauches", "références/",
               "section à", "article à", "à sourcer", "à vérifier", "admissibilité", "suppression", "maintenance",
               "bandeau", "taxobox", "navigation", "début dynastie", "fin dynastie", "succession")
INLINE = {"nobr", "nobreak", "nobr rom", "nowrap", "insécable", "incise", "petites capitales", "pc", "sc",
          "italique", "italiques", "ital", "italics", "langue du titre", "souligné", "surligné", "gras",
          "small", "petit", "sic", "sic!", "refnec", "réf nécessaire", "référence nécessaire",
          "pas clair", "non neutre", "citation nécessaire", "quoi", "lequel", "lesquels", "précision nécessaire",
          "page h'", "page h"}
BASE_KNOWN = (set(wt.ZERO_ARG) | wt.KNOWN_INLINE | wt.BLOCK_TEMPLATES | wt.SUPERSCRIPT |
              set(wt.DISPLAY_PARAM) | wt.DROP | {"lié", "corr", "corrigé", "coquille", "sic2", "uc", "lc"})
COUNTRIES = {name.lower(): name for name in (
    "France", "Italie", "Allemagne", "Espagne", "Portugal", "Belgique", "Suisse", "Autriche", "Royaume-Uni",
    "États-Unis", "Canada", "Mexique", "Brésil", "Venezuela", "Guyane", "Guyana", "Suriname", "Colombie",
    "Équateur", "Pérou", "Bolivie", "Argentine", "Chili", "Paraguay", "Uruguay", "Chine", "Japon", "Inde",
    "Australie", "Nouvelle-Zélande", "Russie", "Ukraine", "Suède", "Norvège", "Danemark", "Finlande",
    "Pologne", "Tchéquie", "Pays-Bas", "Grèce", "Turquie", "Égypte", "Maroc", "Algérie", "Tunisie")}


def number(value):
    """Format français sans float ni arrondi : tous les chiffres restent."""
    value = value.strip().replace("\u00a0", " ").replace("\u202f", " ")
    sci = re.fullmatch(r"(.+?)\s*[eE]([+−-]?\d+)", value)
    if sci:
        return number(sci[1]) + " × 10" + sci[2].translate(SUPER)
    match = re.fullmatch(r"([+−-]?)([\d .,]+)([^\d .,].*)?", value)
    if not match:
        return value
    digits = match[2].strip().replace(" ", "")
    if "." in digits and "," in digits:
        digits = digits.replace(",", "")  # syntaxe documentée : 1,002.3
    digits = digits.replace(".", ",")
    if not re.fullmatch(r"\d+(?:,\d+)?", digits):
        return value
    whole, sep, decimals = digits.partition(",")
    grouped = re.sub(r"(?<=\d)(?=(?:\d{3})+$)", " ", whole)
    return match[1] + grouped + ("," + decimals if sep else "") + (" " + match[3].strip() if match[3] else "")


def date_text(args):
    """Jour/mois/année ou date en un paramètre ; sans qualification d'éphéméride."""
    if len(args) == 1:
        s = args[0]
        iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if iso:
            return date_text([iso[3], iso[2], iso[1]])
        # Le tiret final masque le mois/l'année dans MediaWiki, il n'est pas affiché.
        s = re.sub(r"\b(?:\d{3,4}|" + "|".join(MONTHS) + r")-(?=\s|$)", "", s, flags=re.I)
        s = " ".join(s.split())
        return re.sub(r"^0?1(?=\s+[A-Za-zéû])", "1er", s)
    if len(args) >= 3:
        day, month, year = args[:3]
        month = "" if month.endswith("-") else month
        year = "" if year.endswith("-") else year
        if re.fullmatch(r"-\d+", year):
            year = year[1:] + " av. J.-C."
        if day in ("1", "01"):
            day = "1er"
        elif day.isdigit():
            day = str(int(day))
        if month.isdigit() and 1 <= int(month) <= 12:
            month = MONTHS[int(month) - 1]
        return " ".join(x for x in (day, month, year) if x)
    if len(args) == 2:
        if args[0].isdigit() and 1 <= int(args[0]) <= 12 and re.fullmatch(r"-?\d{3,4}", args[1]):
            return MONTHS[int(args[0]) - 1] + " " + args[1]
        return date_text(args[:1])
    return ""


class WikipediaRenderer(wt.Renderer):
    """Spécialisation sans changement du moteur Wikisource/Wikinews."""

    def __init__(self):
        super().__init__()
        self.population = ""
        self.population_year = ""

    def render(self, code):
        if code is None:
            return ""
        chunks, previous = [], None
        for node in code.nodes:
            # Apostrophe française confondue avec le début du gras : L + texte en gras.
            if (isinstance(previous, wt.Text) and re.search(r"(?<!\w)[LlDd]$", str(previous.value))
                    and isinstance(node, wt.Tag) and str(node.tag) in ("b", "i") and node.wiki_markup):
                chunks.append("'")
            rendered = self.node(node)
            if (isinstance(node, wt.Template) and wt._canon(str(node.name)).lower() in ("etc", "etc.")
                    and chunks and chunks[-1].rstrip().endswith(",") and rendered.startswith(",")):
                rendered = rendered[1:]
            chunks.append(rendered)
            previous = node
        return "".join(chunks)

    def tag(self, node):
        if str(node.tag).strip().lower() in ("table", "gallery"):
            return "\n\n"
        if str(node.tag).strip().lower() in ("math", "chem", "ce"):
            return str(node.contents or "").strip()  # garder la formule, en notation textuelle TeX
        if str(node.tag).strip().lower() in ("syntaxhighlight", "source"):
            return "\n" + str(node.contents or "").strip() + "\n"
        return super().tag(node)

    def node(self, node):
        if isinstance(node, wt.ExternalLink) and node.title is None:
            return str(node.url)  # une URL sans libellé reste du contenu, pas un trou entre parenthèses
        return super().node(node)

    def args(self, tpl):
        # Les clés numériques explicites (1=...) sont aussi des paramètres positionnels.
        positional, named = {}, {}
        for param in tpl.params:
            key = str(param.name).strip()
            if key.isdigit() and 0 < int(key) < 100:
                positional[int(key)] = self.render(param.value).strip()
            else:
                named[key.lower()] = self.render(param.value).strip()
        args = [positional.get(i, "") for i in range(1, max(positional, default=0) + 1)]
        return args, named

    def template(self, tpl):
        raw = str(tpl.name).strip()
        name = wt._canon(raw).lower()
        # Ne pas parcourir tous les modèles imbriqués des infobox/références supprimées.
        if name.startswith("infobox"):
            for param in tpl.params:
                key, val = str(param.name).strip().lower(), str(param.value).strip()
                if key in ("population", "population totale") and re.fullmatch(r"[\d ,.]+", val):
                    self.population = number(val)
                if key in ("année pop", "année_pop", "date population", "population date") and re.fullmatch(r"\d{4}", val):
                    self.population_year = val
            return ""
        if name in DROP or name.startswith(DROP_PREFIX):
            return ""
        if name.split(":", 1)[0] in ("defaultsort", "displaytitle", "cledetri"):
            return ""
        args, named = self.args(tpl)
        first = args[0] if args else ""
        if name in ("'", "’"):
            return "'"
        if name in ("etc", "etc."):
            return first + ", etc."
        if name in COUNTRIES:
            return COUNTRIES[name]
        if name in ("av jc", "av. j.-c."):
            return "av. J.-C."
        if name == "ap jc":
            return "apr. J.-C."
        if re.fullmatch(r"[a-z]{3}-d", name):
            return ""  # icône de drapeau, sans texte
        if name in ("en", "fr", "de", "es", "it", "pt", "ru", "la", "ja", "ar", "pl", "nl") and not args:
            return "(" + name + ")"
        if name in ("isbn", "issn", "doi", "pmid"):
            return name.upper() + (" " + ", ".join(args) if first else "")
        if name == "cas":
            return "-".join(x for x in args if x)
        if name == "doublage":
            return " (" + "; ".join(k.upper() + " : " + v for k, v in named.items() if v) + ")" if named else first
        if name in ("noble", "noble-"):
            return re.sub(r" \([^)]*\)$", "", first) + (" " + args[1] if len(args) > 1 and args[1] else "")
        if name == "code aéroport":
            return "(" + ", ".join(label + " : " + value for label, value in zip(("AITA", "OACI"), args) if value) + ") "
        if name in ("math", "grec ancien", "commentaire biblio", "graphie", "api", "apib", "sapi", "msapi",
                    "c'est à dire", "lnobr"):
            return first
        if name == "vol.":
            return "vol. " + first
        # Modèles dont le nom contient directement le nombre : {{2e}}, {{XIXe|s}}, {{IXe siècle}}.
        ordinal = re.fullmatch(r"([ivxlcdm]+|\d+)(er|re|e|ème|eme)?(?: (siècle(?: av\. j\.-c\.)?|s|dynastie égyptienne))?", name)
        if ordinal and (ordinal[2] or ordinal[3] or raw.isupper()):
            value = ordinal[1].upper() + (ordinal[2] or "")
            tail = ordinal[3] or first
            if tail == "s":
                tail = "siècle" if not ordinal[3] else ""
            return value + (" " + tail if tail else "")
        if name.startswith("formatnum:"):
            return number(self.render(wt.mwp.parse(raw.split(":", 1)[1])))
        if name in ("date", "date-", "date+", "date de naissance", "date de décès", "date décès", "date naissance"):
            return date_text(args)
        if name in ("nombre", "unité", "unite", "nb", "num", "formatnum", "tmp"):
            value = number(first)
            for key in ("à", "–", "et", "±", "×", "××", "/"):
                if named.get(key):
                    value += " " + key.replace("××", "×") + " " + number(named[key])
            if named.get("e"):
                value += " × 10" + named["e"].translate(SUPER)
            unit = []
            for i in range(1, len(args), 2):
                symbol = args[i]
                power = args[i + 1] if i + 1 < len(args) else ""
                unit.append(symbol + (power.translate(SUPER) if re.fullmatch(r"[+−-]?\d+", power) else power))
            return " ".join(x for x in (value, *unit) if x)
        if name in ("lang", "langue", "lang-fr"):
            return named.get("texte", args[1] if len(args) > 1 else "")
        if name.startswith("lang-"):
            return named.get("texte", first)
        if name in ("langue avec nom", "lang-avec-nom"):
            languages = {"en": "anglais", "de": "allemand", "fr": "français", "es": "espagnol",
                         "it": "italien", "ar": "arabe", "ja": "japonais", "la": "latin", "pt": "portugais"}
            label = languages.get(first, first)
            content = named.get("texte", args[1] if len(args) > 1 else "")
            return label + " : " + content + (", " + named["trans"] if named.get("trans") else "")
        if name == "japonais":
            extra = [a for a in args[1:4] if a]
            if named.get("extra"):
                extra.append(named["extra"])
            if not first and extra:
                first = extra.pop(0)
            return first + (" (" + ", ".join(extra) + ")" if extra else "")
        if name in ("lien", "lien web traduit"):
            return named.get("texte") or named.get("fr") or first or named.get("trad", "")
        if name in ("er", "e", "re", "ème", "eme"):
            suffix = "e" if name in ("ème", "eme") else name
            return first + suffix if first else suffix
        if name in ("1er", "1re", "1ers", "1res"):
            return name
        century = name.replace(" mini", "").replace("mini", "").strip("-")
        if century in ("s", "siècle", "s2", "sp", "sav", "sap"):
            values = [a for a in args if a not in ("e", "er", "")]
            vals = values[:2] if century == "s2" else values[:1]
            joiner = " et "
            plural = century == "s2"
            if century == "sp" and len(values) >= 3:
                vals, joiner, plural = [values[0], values[2]], " " + values[1] + " ", "s" in values[3:]
            result = joiner.join(v.upper() + ("er" if v.upper() in ("I", "1") else "e") for v in vals)
            if "mini" not in name and vals:
                result += " siècles" if plural else " siècle"
            if "mini" not in name and name.startswith("-") and vals:
                result += " av. J.-C."
            if century == "sap" and vals:
                result += " apr. J.-C."
            return result
        if name in ("heure", "heures"):
            return " ".join(v + " " + unit for v, unit in zip(args[:3], ("h", "min", "s")) if v)
        if name in ("citation", "citation bloc", "citation étrangère", "guil", "guillemets"):
            text = named.get("texte") or named.get("citation") or first
            return "« " + text + " »" if text else ""
        if name in ("population", "population de france/dernière pop"):
            explicit = named.get("population", first)
            if explicit and re.fullmatch(r"[\d ,.]+", explicit):
                return number(explicit)
            if self.population and not first and not named.get("qid"):
                return self.population
            self.unknown[name] = self.unknown.get(name, 0) + 1
            return ""
        if name == "population de france/dernière année":
            if self.population_year:
                return self.population_year
            self.unknown[name] = self.unknown.get(name, 0) + 1
            return ""
        if name in INLINE:
            return named.get("texte", first)
        if name in ("exp", "sup"):
            return first.translate(SUPER) if re.fullmatch(r"[+−-]?\d+", first) else first
        if name in ("ind", "sub"):
            return first
        if name in ("abr", "abbr", "abréviation"):
            return first
        if name in ("rom", "romain", "rom-maj", "rom-min"):
            return first + (args[1] if len(args) > 1 else "")
        if name in ("numéro", "n°", "no"):
            return "n°" + (" " + first if first else "")
        if name in ("nº", "nos", "numéros"):
            return "nos" + (" " + first if first else "")
        if name in ("page", "p.", "p"):
            return "p." + (" " + "–".join(args[:2]) if first else "")
        if name in ("drapeau", "drapeau2"):
            return first if name == "drapeau2" else ""
        if name in ("nbsp", "espace"):
            return " "
        if name in ("sp", "spp"):
            return " ".join(args)
        if name == "!":
            return "|"
        canon = wt._canon(raw)
        if canon in BASE_KNOWN:
            # Réutiliser les paramètres déjà rendus : les imbriqués ne sont comptés qu'une fois.
            if canon in wt.DROP:
                return ""
            if canon in wt.ZERO_ARG and not args:
                return wt.ZERO_ARG[canon]
            if canon in wt.TITLE_ABBR:
                return wt.ZERO_ARG[canon] + (" " + " ".join(args) if args else "")
            if canon in wt.DISPLAY_PARAM:
                return next((args[i] for i in wt.DISPLAY_PARAM[canon] if i < len(args) and args[i]), "")
            if canon in ("corr", "corrigé", "coquille", "sic2"):
                return args[1] if len(args) > 1 else first
            if canon == "lié":
                return " ".join(args)
            if canon == "uc":
                return first.upper()
            if canon == "lc":
                return first.lower()
            return "\n" + first + "\n" if canon in wt.BLOCK_TEMPLATES else first
        # Compter aussi les inconnus sans paramètre ou contenant seulement un nombre.
        self.unknown[name] = self.unknown.get(name, 0) + 1
        value = named.get("texte") or named.get("nom") or first
        return "" if wt.CSS_LEN.fullmatch(value) else value
