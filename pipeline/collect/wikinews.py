"""Wikinews en français : dump officiel frwikinews (pages-articles), domaine presse.

Wikinews FR est fermé depuis mai 2026 : le dump du 2026-10-01 contient tous les articles.
Un document = un article de l'espace principal (hors redirections et page d'accueil).
`date` = date de publication (modèle {{Date|1er janvier 2024}}, sinon {{Publication|jour|mois|année}},
sinon titre « Brèves : 15 avril 2005 ») ; à défaut, dernière révision (meta.date_source le dit).
Texte : comme la validation du kit, la date en première ligne, paragraphes séparés par une ligne vide,
sources, tableaux et mise en forme retirés."""
import bz2
import datetime as dt
import re
from urllib.parse import quote

from pipeline.collect import wikitext as wt
from pipeline.collect.common import date_ok, download, http_get, make_doc, make_session, source_dir
from pipeline.collect.wikisource import iter_pages

NAME = "wikinews_fr"
DOMAIN = "presse"
LICENSE = "CC BY 2.5"
MONTHS = {m: i for i, m in enumerate(["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                                      "septembre", "octobre", "novembre", "décembre"], 1)}
FR_DATE = re.compile(r"(\d{1,2})(?:er|ᵉʳ)?\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", re.I)
DATE_TPL = re.compile(r"\{\{\s*[Dd]ate\s*\|([^}|]*)")
PUB_TPL = re.compile(r"\{\{\s*[Pp]ublication(.*?)\}\}", re.S)
DROP = {"source", "source rian", "source sans url", "source2", "afp", "reuters", "voa", "publication", "archive",
        "thème", "interprojet", "wikinews", "sommaire flottant", "clr", "ligne grise", "notes", "page rugby",
        "page football", "page france", "wikipédiapar", "traduction", "reportage original", "brèves", "brève",
        "évènements", "dynastie", "résultat-sport", "carton jaune", "rugby à xv", "football", "accueil/Entête",
        "développement", "relecture", "en cours", "image", "photo", "infobox", "portail", "voir aussi", "réf",
        "commons", "wikiquote", "wikisource", "wiktionnaire"}


class NewsRenderer(wt.Renderer):
    """Rendu Wikinews : date sur sa propre ligne, {{w|cible|libellé}} → libellé, sources retirées."""

    def template(self, t):
        name = wt._canon(str(t.name))
        pos = wt._positional(t)
        if name in DROP or name.lower() in DROP:
            return ""
        if name == "date":
            return ("\n" + self.render(pos[0]).strip() + "\n") if pos else ""
        if name in ("w", "wikipédia", "wp", "lien", "wikt"):
            return self.render(pos[1] if len(pos) > 1 else pos[0]) if pos else ""
        if name == "guil":
            return "« " + self.render(pos[0]).strip() + " »" if pos else ""
        if name in ("unité", "nombre", "nb"):
            return " ".join(self.render(p).strip() for p in pos)
        if name in ("ème", "eme"):
            return "e"
        return super().template(t)


def iso_date(y, mon, d):
    """AAAA-MM-JJ, ou None si le jour n'existe pas (« 31 février 2012 », page parodique)."""
    try:
        return dt.date(int(y), int(mon), int(d)).isoformat()
    except ValueError:
        return None


def parse_fr_date(s):
    m = FR_DATE.search(s or "")
    if not m:
        return None
    return iso_date(m.group(3), MONTHS[m.group(2).lower()], m.group(1))


def publication_date(title, text):
    """(date AAAA-MM-JJ, origine) de publication d'un article."""
    m = DATE_TPL.search(text)
    if m and parse_fr_date(m.group(1)):
        return parse_fr_date(m.group(1)), "modele_date"
    m = PUB_TPL.search(text)
    if m:
        body = m.group(1)
        j = re.search(r"jour\s*=\s*(\d{1,2})", body)
        mo = re.search(r"mois\s*=\s*([^\s|}]+)", body)
        y = re.search(r"année\s*=\s*(\d{4})", body)
        if j and mo and y and mo.group(1).lower() in MONTHS:
            d = iso_date(y.group(1), MONTHS[mo.group(1).lower()], j.group(1))
            if d:
                return d, "modele_publication"
    d = parse_fr_date(title)
    if d:
        return d, "titre"
    return None, None


def statut(text):
    m = PUB_TPL.search(text)
    if m:
        s = re.search(r"statut\s*=\s*([^|\n}]*)", m.group(1))
        if s and s.group(1).strip():
            return s.group(1).strip().lower()
    return None


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    base, dump = c["base_url"].rstrip("/"), str(c["dump"])
    st = http_get(make_session(), f"{base}/{dump}/dumpstatus.json", log=log).json()["jobs"]["articlesdump"]
    if st.get("status") != "done":
        raise RuntimeError(f"dump frwikinews-{dump} incomplet")
    files = sorted(st["files"])
    return {"base": base, "dump": dump, "files": files, "min_chars": int(c.get("min_chars", 200))}


def units(ctx):
    return ctx["files"]


def article_doc(pid, title, ts, text, ctx, renderer=None):
    """Document d'un article, ou (None, raison)."""
    if text.lstrip()[:12].lower().startswith(("#redirect", "#redirection")):
        return None, "redirection"
    if title in ("Accueil",) or title.startswith("Accueil/"):
        return None, "page_accueil"
    body = wt.to_text(text, renderer or NewsRenderer())
    if len(body) < ctx["min_chars"]:
        return None, "trop_court"
    date, origin = publication_date(title, text)
    if date is None:
        date, origin = ts[:10], "derniere_revision"
    if not date_ok(date):
        return None, "date_apres_gel"
    meta = {"title": title, "statut": statut(text), "date_source": origin, "last_revision": ts[:10],
            "dump": f"frwikinews-{ctx['dump']}"}
    url = "https://fr.wikinews.org/wiki/" + quote(title.replace(" ", "_"), safe="/:,()'’")
    return make_doc(NAME, DOMAIN, pid, body, url=url, date=date, license=LICENSE, meta=meta), None


def process(unit, ctx, tally, log):
    dest = source_dir(NAME) / "_dl" / unit
    download(make_session(), f"{ctx['base']}/{ctx['dump']}/{unit}", dest, log=log)
    renderer = NewsRenderer()
    try:
        with bz2.open(dest, "rb") as f:
            for ns, pid, title, ts, text in iter_pages(f):
                if ns != 0:
                    continue
                doc, why = article_doc(pid, title, ts, text, ctx, renderer)
                if doc is None:
                    tally.skip(why)
                    continue
                yield doc
    finally:
        dest.unlink(missing_ok=True)
