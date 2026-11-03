"""HTML brut → texte visible naïf (menus et pieds de page compris), décodage du charset, réserve TDM."""
import codecs
import re

from lxml import etree
from lxml import html as lhtml

# Balises dont le contenu n'est jamais affiché
SKIP_TAGS = {"script", "style", "noscript", "template", "head", "title", "svg", "math", "iframe",
             "object", "embed", "canvas", "audio", "video", "select", "datalist"}
# Balises de bloc : saut de ligne avant et après
BLOCK_TAGS = {"address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "details", "dialog", "div",
              "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
              "header", "hr", "html", "legend", "li", "main", "menu", "nav", "ol", "option", "p", "pre", "section",
              "summary", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul", "button", "label"}
_WS = re.compile(r"[ \t\r\f\v ​]+")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9._:-]+)""", re.I)
_HTTP_CHARSET = re.compile(r"""charset\s*=\s*["']?\s*([A-Za-z0-9._:-]+)""", re.I)
_META_TAG = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR = re.compile(r"""([a-zA-Z_:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""")
# Étiquettes que le standard HTML5 fait lire comme windows-1252
_AS_CP1252 = {"iso-8859-1", "iso8859-1", "latin-1", "latin1", "l1", "us-ascii", "ascii", "iso_8859-1", "cp819"}


def _norm_charset(name):
    if not name:
        return None
    n = name.strip().strip("\"'").lower()
    if n in _AS_CP1252:
        return "cp1252"
    try:
        return codecs.lookup(n).name
    except LookupError:
        return None


def decode_html(payload: bytes, content_type: str | None = None, detected: str | None = None):
    """Décode le HTML : BOM, puis UTF-8 s'il est valide, puis charset déclaré (HTTP, <meta>),
    puis charset détecté par Common Crawl, puis windows-1252. Renvoie (texte, charset utilisé)."""
    for bom, enc in ((codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")):
        if payload.startswith(bom):
            return payload[len(bom):].decode(enc, errors="replace"), enc
    try:
        return payload.decode("utf-8"), "utf-8"  # UTF-8 valide : presque toujours le bon choix
    except UnicodeDecodeError:
        pass
    cands = []
    m = _HTTP_CHARSET.search(content_type or "")
    if m:
        cands.append(m.group(1))
    m = _META_CHARSET.search(payload[:8192])
    if m:
        cands.append(m.group(1).decode("ascii", "ignore"))
    cands.append(detected)
    for c in cands:
        enc = _norm_charset(c)
        if enc and enc not in ("utf-8",):
            try:
                return payload.decode(enc), enc
            except (UnicodeDecodeError, LookupError):
                continue
    return payload.decode("cp1252", errors="replace"), "cp1252"


def tdm_reserved(html: str, headers=None) -> bool:
    """Vrai si la page réserve la fouille de textes (protocole TDMRep) : en-tête HTTP
    `tdm-reservation: 1` ou balise <meta name="tdm-reservation" content="1">."""
    if headers:
        for k, v in headers:
            if k.lower() == "tdm-reservation" and v.strip() == "1":
                return True
    low = html[:200_000]
    if "tdm-reservation" not in low.lower():
        return False
    for tag in _META_TAG.findall(low):
        attrs = {m.group(1).lower(): (m.group(2) or m.group(3) or m.group(4) or "") for m in _ATTR.finditer(tag)}
        if attrs.get("name", "").strip().lower() == "tdm-reservation" and attrs.get("content", "").strip() == "1":
            return True
    return False


def _hidden(el) -> bool:
    if el.get("hidden") is not None or el.get("aria-hidden") == "true":
        return True
    style = (el.get("style") or "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def parse_html(html: str):
    """Arbre lxml ou None (HTML vide ou illisible)."""
    if not html or not html.strip():
        return None
    try:
        parser = lhtml.HTMLParser(encoding="utf-8", remove_comments=False)
        return lhtml.document_fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except (etree.ParserError, ValueError):
        return None


def visible_text(html: str = None, root=None) -> str:
    """Tout le texte affiché de la page, une ligne par bloc, espaces normalisés (version « brute »)."""
    if root is None:
        root = parse_html(html)
    if root is None:
        return ""
    out = []
    skip = 0
    for ev, el in etree.iterwalk(root, events=("start", "end")):
        tag = el.tag.lower() if isinstance(el.tag, str) else None  # None : commentaire, instruction
        hide = tag is None or tag in SKIP_TAGS or (ev == "start" and not skip and _hidden(el))
        if ev == "start":
            if skip or hide:
                skip += 1
                continue
            if tag in BLOCK_TAGS:
                out.append("\n")
            if el.text:
                out.append(el.text)
        else:
            if skip:
                # fin d'un élément ignoré : on sort du masquage quand on remonte au niveau visible
                skip -= 1
                if not skip and el.tail:
                    out.append(el.tail)
                continue
            if tag in BLOCK_TAGS:
                out.append("\n")
            if el.tail:
                out.append(el.tail)
    lines = (_WS.sub(" ", line).strip() for line in "".join(out).split("\n"))
    return "\n".join(line for line in lines if line)


def page_title(root) -> str | None:
    if root is None:
        return None
    t = root.find(".//title")
    if t is not None and t.text:
        return _WS.sub(" ", t.text).strip() or None
    return None
