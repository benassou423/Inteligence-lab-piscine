"""Tests Common Crawl : WARC minuscule écrit avec warcio, filtre langue/domaine, réserve TDM, charset."""
import json
from io import BytesIO

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from pipeline.collect import htmltext
from pipeline.collect.commoncrawl import WarcScanner, host_of, parse_metadata
from pipeline.collect.common import Tally

CTX = {"keep_tld": ".fr"}


def cld2(*codes):
    langs = [{"code": c, "code-iso-639-3": "x", "text-covered": 0.5, "score": 1.0, "name": c} for c in codes]
    return json.dumps({"reliable": True, "text-bytes": 100, "languages": langs})


PAGES = [
    # (url, cld2, content-type, corps, en-têtes HTTP en plus)
    ("https://www.exemple.com/fr", cld2("fr"), "text/html; charset=utf-8",
     "<html><head><title>Accueil</title></head><body><nav>Menu</nav><p>Bonjour à tous</p></body></html>".encode(), []),
    ("https://site.fr/en", cld2("en"), "text/html", b"<html><body><p>Hello from a .fr domain</p></body></html>", []),
    ("https://site.com/en", cld2("en"), "text/html", b"<html><body><p>Hello world</p></body></html>", []),
    ("https://reserve.com/a", cld2("fr"), "text/html",
     '<html><head><meta name="tdm-reservation" content="1"></head><body><p>Réservé</p></body></html>'.encode(), []),
    ("https://reserve.com/b", cld2("fr"), "text/html", "<html><body><p>Réservé aussi</p></body></html>".encode(),
     [("tdm-reservation", "1")]),
    ("https://image.fr/x.png", cld2("fr"), "image/png", b"\x89PNG....", []),
    ("https://latin.com/p", cld2("en", "fr"), "text/html; charset=ISO-8859-1",
     "<html><body><p>Café crème à côté</p></body></html>".encode("latin-1"), []),
]


def make_warc(path):
    with open(path, "wb") as f:
        w = WARCWriter(f, gzip=True)
        for url, langs, ctype, body, extra in PAGES:
            req = w.create_warc_record(url, "request", payload=BytesIO(b"GET / HTTP/1.1\r\n\r\n"))
            w.write_record(req)
            http = StatusAndHeaders("200 OK", [("Content-Type", ctype), *extra], protocol="HTTP/1.1")
            resp = w.create_warc_record(url, "response", payload=BytesIO(body), http_headers=http,
                                        warc_headers_dict={"WARC-Date": "2026-09-04T14:43:16Z"})
            w.write_record(resp)
            rid = resp.rec_headers.get_header("WARC-Record-ID")
            meta = f"fetchTimeMs: 10\r\ncharset-detected: UTF-8\r\nlanguages-cld2: {langs}\r\n".encode()
            w.write_record(w.create_warc_record(url, "metadata", payload=BytesIO(meta),
                                                warc_headers_dict={"WARC-Concurrent-To": rid,
                                                                   "Content-Type": "application/warc-fields"}))


def test_warc_scanner_filters(tmp_path):
    path = tmp_path / "mini.warc.gz"
    make_warc(path)
    tally = Tally()
    sc = WarcScanner(CTX, tally, "mini.warc.gz")
    with open(path, "rb") as f:
        docs = list(sc.scan(f))
    urls = [d["url"] for d in docs]
    assert urls == ["https://www.exemple.com/fr", "https://site.fr/en", "https://latin.com/p"]
    assert tally.removed == {"pas_francais": 1, "tdm_reservation": 2, "pas_html": 1}
    d = docs[0]
    assert d["id"].startswith("commoncrawl:urn:uuid:") and d["source"] == "commoncrawl" and d["domain"] == "web"
    assert d["date"] == "2026-09-04" and d["license"]
    assert d["text"] == "Menu\nBonjour à tous"  # texte brut : menus compris, pas le <title>
    assert d["html"].startswith("<html>") and d["meta"]["title"] == "Accueil"
    assert d["meta"]["kept_by"] == "langue" and d["meta"]["warc_file"] == "mini.warc.gz"
    assert docs[1]["meta"]["kept_by"] == "tld"
    assert docs[2]["text"] == "Café crème à côté" and docs[2]["meta"]["charset"] == "cp1252"
    assert docs[2]["meta"]["kept_by"] == "langue"  # français au 2e rang dans cld2
    assert sc.n["responses"] == len(PAGES) and sc.n["kept"] == 3


def test_warc_scanner_resume_no_duplicates(tmp_path):
    """Après une coupure, on relit depuis `resume` : les réponses déjà traitées ne ressortent pas."""
    path = tmp_path / "mini.warc.gz"
    make_warc(path)
    sc = WarcScanner(CTX, Tally(), "mini.warc.gz")
    with open(path, "rb") as f:
        first = list(sc.scan(f))
    assert sc.resume > 0
    with open(path, "rb") as f:
        f.seek(sc.resume)
        again = list(sc.scan(f, base=sc.resume))
    assert len(first) == 3 and again == []


def test_parse_metadata_and_host():
    m = parse_metadata(b'fetchTimeMs: 76\r\ncharset-detected: UTF-8\r\nlanguages-cld2: {"languages":[{"code":"fr"}]}\r\n')
    assert m["charset-detected"] == "UTF-8" and m["languages-cld2"]["languages"][0]["code"] == "fr"
    assert host_of("https://WWW.Exemple.FR.:443/x") == "www.exemple.fr"


def test_visible_text_and_tdm():
    html = ("<html><head><title>T</title><style>a{}</style></head><body><ul><li>Accueil</li><li>Contact</li></ul>"
            "<!-- commentaire --><div>Bonjour <b>le</b> monde<script>x=1</script> fin</div>"
            "<p hidden>caché</p><p style='display: none'>caché</p><footer>© 2026</footer></body></html>")
    assert htmltext.visible_text(html) == "Accueil\nContact\nBonjour le monde fin\n© 2026"
    assert htmltext.tdm_reserved('<meta content="1" name="TDM-Reservation">')
    assert not htmltext.tdm_reserved('<meta name="tdm-reservation" content="0">')
    assert htmltext.tdm_reserved("<p>x</p>", [("TDM-Reservation", "1")])


def test_decode_html():
    assert htmltext.decode_html("é".encode("cp1252"), "text/html; charset=ISO-8859-1") == ("é", "cp1252")
    assert htmltext.decode_html("é".encode("utf-8"), "text/html; charset=ISO-8859-1") == ("é", "utf-8")
    meta = '<meta charset="iso-8859-15"><p>€</p>'.encode("iso-8859-15")
    assert htmltext.decode_html(meta)[0].endswith("€</p>")
    assert htmltext.decode_html(b"\xef\xbb\xbfabc") == ("abc", "utf-8")
