"""Corpus Molière provisoire (bloc 1) : théâtre de Molière depuis Wikisource → data/moliere/moliere.txt.

Édition suivie : Œuvres complètes de Molière, éd. Charles Louandre, Charpentier, 1910 (3 tomes),
dans l'ordre de ses tables des matières. Pour chaque pièce : page principale et sous-pages (actes)
rendues en HTML par l'API MediaWiki, puis converties en texte (noms des personnages en capitales,
vers un par ligne, didascalies gardées ; notes, numéros de vers et de page retirés).
Les pièces sont à la suite, séparées par une ligne vide ; aucune ligne vide à l'intérieur d'une pièce."""
import re
import time
from urllib.parse import quote


from pipeline.collect import htmltext
from pipeline.collect.common import ROOT, get_logger, http_get, make_session, now_iso, read_json, write_json

NAME = "moliere"
LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
TABLE = re.compile(r"\{\{\s*table\s*\|(.*?)\}\}\s*$", re.M | re.S)
REMOVE_XPATH = ("//*[contains(concat(' ', normalize-space(@class), ' '), ' ws-noexport ')]"
                "|//*[@id='headertemplate']|//*[@id='ws-data']|//style|//script|//link|//meta"
                "|//sup[contains(@class,'reference')]|//*[contains(@class,'mw-references-wrap')]"
                "|//ol[contains(@class,'references')]|//*[contains(@class,'references-small')]"
                "|//*[contains(@class,'pagenum')]|//*[contains(@class,'mw-editsection')]"
                "|//*[contains(@class,'noprint')]|//*[contains(@class,'navbox')]")
NOTICE = re.compile(r"^(NOTICE|AVERTISSEMENT DE L’ÉDITEUR|AVERTISSEMENT DE L'ÉDITEUR)\b", re.I)
CAST = re.compile(r"^(PERSONNAGES|ACTEURS)\b", re.I)


class Api:
    """Client poli de l'API MediaWiki (1 requête par seconde)."""

    def __init__(self, url, sleep_s, log):
        self.url, self.sleep_s, self.log = url, sleep_s, log
        self.s = make_session()
        self.last = 0.0
        self.n = 0

    def parse(self, page, prop):
        wait = self.sleep_s - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        params = {"action": "parse", "page": page, "prop": prop, "format": "json", "formatversion": "2",
                  "redirects": "1", "disableeditsection": "1", "disablelimitreport": "1"}
        r = http_get(self.s, self.url + "?" + "&".join(f"{k}={quote(v)}" for k, v in params.items()), log=self.log)
        self.last = time.time()
        self.n += 1
        d = r.json()
        if "error" in d:
            raise KeyError(f"{page} : {d['error'].get('info')}")
        return d["parse"]


def toc_entries(api, tables, exclude):
    """(cible, libellé) des tables des matières des 3 tomes, dans l'ordre."""
    out = []
    for t in tables:
        w = api.parse(t, "wikitext")["wikitext"]
        for m in TABLE.finditer(w):
            body = m.group(1)
            mt = re.search(r"titre\s*=\s*(.*?)(\|\s*page\s*=|$)", body, re.S)
            if not mt:
                continue
            lm = LINK.search(mt.group(1))
            if not lm:
                continue
            target, label = lm.group(1).strip(), (lm.group(2) or lm.group(1)).strip()
            if label in exclude or target in exclude:
                continue
            out.append((target, label))
    return out


EDITIONS = re.compile(r"\{\{\s*(liste documents début|œuvre\s*\|\s*éditions|éditions|molière\s*\|)", re.I)
NOT_PLAY = re.compile(r"/(Notice|Avertissement|Lettre de|Lettre à)", re.I)


def resolve_edition(api, target):
    """Page d'une pièce : suit une page « éditions » (liste, ou modèle {{molière|…}}) vers l'édition
    Louandre si elle existe, sinon la première édition transcrite."""
    p = api.parse(target, "wikitext|links")
    title, w = p["title"], p["wikitext"]
    if EDITIONS.search(w):
        base = title.split("/")[0]
        links = [x["title"] for x in p.get("links", []) if x.get("ns") == 0 and x.get("exists", True)
                 and x["title"].startswith(base + "/")]
        links += [m.group(1).strip() for m in LINK.finditer(w) if m.group(1).strip().startswith(base + "/")]
        pick = next((x for x in links if "Louandre" in x), next((x for x in links if "dition" in x), None))
        if pick and pick != title:
            p = api.parse(pick, "wikitext")
            title, w = p["title"], p["wikitext"]
    return title, w


def subpages(title, w):
    """Sous-pages liées depuis la page de la pièce (actes…), dans l'ordre, sans notices ni lettres d'autrui."""
    base = title.split("/")[0]
    seen, out = {title}, []
    for m in LINK.finditer(w):
        t = m.group(1).strip().rstrip("/")
        if t.startswith("/"):
            t = title + t
        elif t.startswith("../"):
            continue
        if not t.startswith(base + "/") or t in seen or NOT_PLAY.search(t):
            continue
        if not t.startswith(title + "/") and "dition" in t[len(base):]:
            continue  # autre édition de la même pièce
        seen.add(t)
        out.append(t)
    return out


def html_to_lines(html):
    """HTML rendu d'une page Wikisource → lignes de texte (personnages en capitales)."""
    root = htmltext.parse_html(html)
    if root is None:
        return []
    for el in root.xpath(REMOVE_XPATH):
        el.getparent().remove(el) if el.getparent() is not None else None
    for el in root.xpath("//*[contains(@class,'personnage')]"):
        txt = "".join(el.itertext()).strip()
        for c in list(el):
            el.remove(c)
        el.text = txt.upper()
    text = htmltext.visible_text(root=root)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"[\d\s.\-–—◄►|]+", line):
            continue  # numéros de page, flèches de navigation
        lines.append(line)
    return lines


def trim_front_matter(lines):
    """Retire une notice de l'éditeur placée avant la liste des personnages."""
    for i, line in enumerate(lines[:400]):
        if CAST.match(line):
            head = lines[:i]
            if any(NOTICE.match(x) for x in head) or len(head) > 60:
                return lines[i:], len(head)
            return lines, 0
    return lines, 0


def play_text(api, target, label, log):
    title, w = resolve_edition(api, target)
    pages = [title] + subpages(title, w)
    lines, got = [], []
    for p in pages:
        try:
            lines.extend(html_to_lines(api.parse(p, "text")["text"]))
            got.append(p)
        except KeyError as e:
            log.warning("%s : page ignorée (%s)", label, e)
    pages = got
    lines, cut = trim_front_matter(lines)
    head = label.upper()
    if not any(head in x.upper() for x in lines[:5]):
        lines.insert(0, head)
    log.info("%s : %d pages, %d lignes%s", label, len(pages), len(lines), f" ({cut} lignes de notice retirées)" if cut else "")
    return title, pages, lines


def run(cfg, limit=None, workers=None, force=False):
    c = cfg["sources"][NAME]
    out_dir = ROOT / c["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(NAME, out_dir)
    done = read_json(out_dir / "_stats.json")
    if done and limit is None and not force and (out_dir / "moliere.txt").exists():
        log.info("moliere.txt déjà construit (%d pièces) : --force pour refaire", done.get("plays", 0))
        return done
    api = Api(c["api"], float(c.get("sleep_s", 1.0)), log)
    entries = toc_entries(api, c["tables"], set(c.get("exclude", [])))
    if limit is not None:
        entries = entries[:limit]
    log.info("%d pièces à récupérer", len(entries))
    plays = []
    for target, label in entries:
        try:
            title, pages, lines = play_text(api, target, label, log)
        except (KeyError, RuntimeError) as e:
            log.warning("%s : échec (%s)", label, e)
            continue
        plays.append({"label": label, "title": title, "pages": pages, "lines": lines})
    text = "\n\n".join("\n".join(p["lines"]) for p in plays) + "\n"
    (out_dir / "moliere.txt").write_text(text, encoding="utf-8")
    readme = [
        "# Corpus Molière provisoire (bloc 1)",
        "",
        "Provisoire : le kit officiel fournira le corpus de référence. Généré par "
        "`uv run python -m pipeline.collect moliere` le " + now_iso()[:10] + ".",
        "",
        "- **Source** : Wikisource, *Œuvres complètes de Molière*, éd. Charles Louandre, Charpentier, 1910 "
        "(ordre des tables des matières des 3 tomes ; quelques pièces liées à une autre édition).",
        "- **Licence** : textes du domaine public ; transcription Wikisource sous CC BY-SA 4.0.",
        "- **Format** : UTF-8, pièces à la suite séparées par une ligne vide ; noms des personnages en capitales, "
        "un vers par ligne, didascalies gardées ; notes de l'éditeur, numéros de vers et de page retirés.",
        f"- **Volume** : {len(plays)} pièces, {sum(len(p['lines']) for p in plays)} lignes, "
        f"{len(text)} caractères, {len(text.encode('utf-8'))} octets.",
        "",
        "| # | Pièce | Lignes | Pages Wikisource |",
        "|---|---|---|---|",
    ]
    for i, p in enumerate(plays, 1):
        urls = " ".join(f"[{j}](https://fr.wikisource.org/wiki/{quote(x.replace(' ', '_'), safe='/:,()')})"
                        for j, x in enumerate(p["pages"], 1))
        readme.append(f"| {i} | {p['label']} | {len(p['lines'])} | {urls} |")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_json(out_dir / "_stats.json", {"plays": len(plays), "lines": sum(len(p["lines"]) for p in plays),
                                         "chars": len(text), "api_requests": api.n, "finished": now_iso()})
    log.info("moliere.txt : %d pièces, %d caractères (%d requêtes API)", len(plays), len(text), api.n)
    return {"plays": len(plays), "chars": len(text)}
