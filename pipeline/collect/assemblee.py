"""Assemblée nationale : comptes rendus des débats en séance publique (open data, XML Syceron).

Un document = une séance, au format de la validation du kit (data/validation.jsonl, genre dialogue) :
une intervention par paragraphe, « M. Nom : propos », une ligne vide entre deux paragraphes.
Les mentions sans orateur (« (Il est procédé au scrutin.) ») sont des paragraphes à part ;
les didascalies en ligne (« (Applaudissements…) ») restent dans le propos. Comme dans la validation,
pas de qualité après le nom, pas de titres de points de l'ordre du jour, pas de signature finale."""
import re
import struct
import zipfile
import zlib

from lxml import etree

from pipeline.collect.common import date_ok, download, make_doc, make_session, source_dir

NAME = "assemblee_nationale"
DOMAIN = "dialogue"
LICENSE = "Licence Ouverte 2.0 (Etalab)"
NS = "{http://schemas.assemblee-nationale.fr/referentiel}"
_WS = re.compile(r"\s+")


def prepare(cfg, log, sample):
    c = cfg["sources"][NAME]
    legs = [int(x) for x in c["legislatures"]]
    return {"legislatures": legs, "url_template": c["url_template"]}


def units(ctx):
    return [f"leg{leg}" for leg in ctx["legislatures"]]  # ordre de la config : 17e d'abord


def local(el):
    return etree.QName(el).localname if isinstance(el.tag, str) else None


def inline_text(el) -> str:
    """Texte d'un <texte> : <br/> → espace, italique/exposant/indice gardés en ligne."""
    parts = []

    def rec(e):
        if e.text:
            parts.append(e.text)
        for c in e:
            if local(c) == "br":
                parts.append(" ")
            elif local(c) is not None:
                rec(c)
            if c.tail:
                parts.append(c.tail)

    rec(el)
    return _WS.sub(" ", "".join(parts)).strip()


def speaker(par) -> str | None:
    """Nom de l'orateur tel qu'imprimé (« M. Jean Dupont », « Mme la présidente »), sans sa qualité."""
    names = [_WS.sub(" ", o.findtext(f"{NS}nom") or "").strip() for o in par.findall(f"{NS}orateurs/{NS}orateur")]
    return " et ".join(n for n in names if n) or None


def seance_to_doc(root, source_file):
    """Document d'une séance, ou (None, raison)."""
    md = root.find(f"{NS}metadonnees")
    uid = root.findtext(f"{NS}uid")
    raw_date = (md.findtext(f"{NS}dateSeance") or "") if md is not None else ""
    date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) >= 8 else None
    if not date_ok(date):
        return None, "date_apres_gel"
    contenu = root.find(f"{NS}contenu")
    if contenu is None:
        return None, "sans_contenu"
    paras, speakers, n_inter, n_titles = [], set(), 0, 0
    for el in contenu.iter(f"{NS}paragraphe", f"{NS}point"):
        if local(el) == "point":
            n_titles += 1  # titre de point de l'ordre du jour : absent du format de la validation
            continue
        if el.get("code_style") == "Signature droite":
            continue  # « Le directeur des comptes rendus … »
        texte = el.find(f"{NS}texte")
        t = inline_text(texte) if texte is not None else ""
        if not t:
            continue
        who = speaker(el)
        if who:
            paras.append(f"{who} : {t}")
            speakers.add(who)
            n_inter += 1
        else:
            paras.append(t)
    if not paras:
        return None, "texte_vide"
    leg = md.findtext(f"{NS}legislature")
    pres = md.findtext(f"{NS}sommaire/{NS}presidentSeance")
    meta = {"legislature": int(leg) if leg and leg.isdigit() else leg, "session": md.findtext(f"{NS}session"),
            "num_seance": md.findtext(f"{NS}numSeance"), "jour": md.findtext(f"{NS}dateSeanceJour"),
            "presidence": _WS.sub(" ", pres).strip() if pres else None, "etat": md.findtext(f"{NS}etat"),
            "version": md.findtext(f"{NS}version"), "n_interventions": n_inter, "n_orateurs": len(speakers),
            "n_points": n_titles, "xml_file": source_file}
    url = f"https://www.assemblee-nationale.fr/dyn/{leg}/comptes-rendus/seance/{uid}"
    return make_doc(NAME, DOMAIN, uid, "\n\n".join(paras), url=url, date=date, license=LICENSE, meta=meta), None


def iter_partial_zip(path):
    """(nom, contenu) des fichiers lisibles d'une archive zip tronquée (en-têtes locaux lus un par un)."""
    with open(path, "rb") as f:
        while True:
            h = f.read(30)
            if len(h) < 30 or h[:4] != b"PK\x03\x04":
                return
            _, _, flag, method, _, _, _, csize, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", h)
            name = f.read(nlen).decode("utf-8", "replace")
            f.read(xlen)
            if flag & 0x08:
                return  # tailles inconnues (descripteur de données) : on s'arrête
            data = f.read(csize)
            if len(data) < csize:
                return
            yield name, zlib.decompress(data, -15) if method == 8 else data


def _members(dest, partial):
    if not partial:
        with zipfile.ZipFile(dest) as zf:
            for n in sorted(x for x in zf.namelist() if x.endswith(".xml")):
                yield n, zf.read(n)
    else:
        yield from sorted((n, b) for n, b in iter_partial_zip(dest) if n.endswith(".xml"))


def process(unit, ctx, tally, log):
    leg = int(unit[3:])
    url = ctx["url_template"].format(leg=leg)
    dest = source_dir(NAME) / "_dl" / f"syceron_leg{leg}.xml.zip"
    log.info("téléchargement %s", url)
    partial = False
    try:
        download(make_session(), url, dest, log=log, max_tries=4)
    except Exception as e:  # noqa: BLE001 — serveur qui coupe l'archive : on garde ce qui est lisible
        part = dest.with_name(dest.name + ".part")
        if not part.exists():
            raise
        log.error("législature %d : archive incomplète après plusieurs reprises (%s) ; "
                  "on lit les %.0f premiers Mo seulement", leg, e, part.stat().st_size / 1e6)
        part.rename(dest)
        partial = True
        tally.extra[f"leg{leg}_archive_partielle_mo"] = round(dest.stat().st_size / 1e6, 1)
    try:
        n = 0
        for name, raw in _members(dest, partial):
            n += 1
            try:
                root = etree.fromstring(raw)
            except etree.XMLSyntaxError as e:
                log.warning("%s illisible : %s", name, e)
                tally.skip("xml_illisible")
                continue
            doc, why = seance_to_doc(root, name.rsplit("/", 1)[-1])
            if doc is None:
                tally.skip(why)
                continue
            if partial:
                doc["meta"]["archive_partielle"] = True
            yield doc
        log.info("législature %d : %d fichiers XML%s", leg, n, " (archive partielle)" if partial else "")
    finally:
        dest.unlink(missing_ok=True)
