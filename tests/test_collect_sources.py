"""Tests des collecteurs sur petits fixtures : Assemblée nationale (XML), Gutenberg (en-tête, catalogue),
Wikipédia (parquet), Molière (HTML Wikisource)."""
import gzip
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from pipeline.collect import assemblee, gutenberg, moliere, wikipedia_hf2023
from pipeline.collect.common import Tally

FIX = Path(__file__).parent / "fixtures" / "collect"


# ---------------------------------------------------------------- Assemblée nationale

def test_assemblee_seance_dialogue_format():
    root = etree.parse(str(FIX / "an_seance.xml")).getroot()
    doc, why = assemblee.seance_to_doc(root, "an_seance.xml")
    assert why is None
    assert doc["id"] == "assemblee_nationale:CRSANR5L17S2026O1N999"
    assert doc["date"] == "2026-10-15" and doc["domain"] == "dialogue"
    assert doc["url"] == "https://www.assemblee-nationale.fr/dyn/17/comptes-rendus/seance/CRSANR5L17S2026O1N999"
    assert doc["text"].split("\n\n") == [
        "Mme la présidente : La séance est ouverte.",
        "(La séance est ouverte à quinze heures.)",
        "Mme la présidente : La parole est à M. Jean Exemple.",
        "M. Jean Exemple : Madame la ministre, voici le 1er point. Et voici le second. "
        "(Applaudissements sur quelques bancs.)",
        "Mme la présidente : La séance est levée.",
    ]
    assert "\n\n\n" not in doc["text"] and doc["meta"]["n_points"] == 1
    assert doc["meta"]["n_interventions"] == 4 and doc["meta"]["n_orateurs"] == 2
    assert doc["license"].startswith("Licence Ouverte")


def test_assemblee_after_freeze_dropped():
    xml = (FIX / "an_seance.xml").read_bytes().replace(b"20261015150000000", b"20261020150000000")
    doc, why = assemblee.seance_to_doc(etree.fromstring(xml), "x.xml")
    assert doc is None and why == "date_apres_gel"


# ---------------------------------------------------------------- Gutenberg

def test_gutenberg_keeps_header_and_footer():
    text = gutenberg.decode_text((FIX / "gutenberg_header.txt").read_bytes())
    assert text.startswith("The Project Gutenberg eBook of Le Test")  # BOM retiré, en-tête gardé
    assert "*** START OF THE PROJECT GUTENBERG EBOOK LE TEST ***" in text
    assert "*** END OF THE PROJECT GUTENBERG EBOOK LE TEST ***" in text
    assert "\r" not in text and "é, è, à, ç, œ" in text


def test_gutenberg_latin1_fallback():
    assert gutenberg.decode_text("Café\r\n".encode("latin-1")) == "Café\n"


def test_gutenberg_catalog_and_process(tmp_path, monkeypatch):
    cat = tmp_path / "pg_catalog.csv.gz"
    with gzip.open(cat, "wb") as f:
        f.write((FIX / "pg_catalog_mini.csv").read_bytes())
    books = gutenberg.read_catalog(cat, ["fr"])
    assert [b["n"] for b in books] == [796, 12000, 79999]  # texte, une des langues = fr ; pas le son
    assert books[1]["languages"] == ["en", "fr"]
    raw = (FIX / "gutenberg_header.txt").read_bytes()
    monkeypatch.setattr(gutenberg, "fetch_book", lambda s, ctx, n, log: (f"https://miroir/{n}.txt", raw))
    ctx = {"batches": {"books-00000": books}, "threads": 2, "mirror_epub": "m", "mirror_main": "m"}
    tally = Tally()
    docs = list(gutenberg.process("books-00000", ctx, tally, None))
    assert [d["id"] for d in docs] == ["gutenberg_fr:796", "gutenberg_fr:12000"]
    assert tally.removed == {"date_apres_gel": 1}  # sorti le 2026-10-22, après le gel
    d = docs[0]
    assert d["url"] == "https://www.gutenberg.org/ebooks/796" and d["date"] == "1997-01-01"
    assert d["meta"]["title"] == "La Chartreuse De Parme" and d["text"].startswith("The Project Gutenberg")


def test_gutenberg_mirror_urls():
    urls = gutenberg.candidate_urls("https://e", "https://m", 796)
    assert urls == ["https://e/796/pg796.txt", "https://m/7/9/796/796-0.txt", "https://m/7/9/796/796-8.txt",
                    "https://m/7/9/796/796.txt"]
    assert gutenberg.candidate_urls("https://e", "https://m", 5)[1] == "https://m/0/5/5-0.txt"


# ---------------------------------------------------------------- Wikipédia

def test_wikipedia_docs_from_parquet(tmp_path):
    t = pa.table({"id": ["3", "4"], "url": ["https://fr.wikipedia.org/wiki/A", "https://fr.wikipedia.org/wiki/B"],
                  "title": ["A", "B"], "text": ["Texte de A.", "  "]})
    pq.write_table(t, tmp_path / "x.parquet")
    ctx = {"dump_date": "2023-11-01", "batch": 10, "repo": "wikimedia/wikipedia", "config": "20231101.fr",
           "revision": "abc"}
    tally = Tally()
    docs = list(wikipedia_hf2023._docs_from_table(pq.ParquetFile(tmp_path / "x.parquet"), ctx, tally))
    assert len(docs) == 1 and tally.removed == {"texte_vide": 1}
    assert docs[0]["id"] == "wikipedia_fr_hf2023:3" and docs[0]["meta"]["title"] == "A"
    assert docs[0]["license"] == "CC BY-SA 4.0" and docs[0]["date"] == "2023-11-01"


# ---------------------------------------------------------------- Molière

def test_moliere_html_to_lines():
    lines = moliere.html_to_lines((FIX / "moliere_acte.html").read_text(encoding="utf-8"))
    assert lines == ["PHILINTE", "Qu’est-ce donc ? Qu’avez-vous ?", "ALCESTE, assis.", "Laissez-moi, je vous prie.",
                     "PHILINTE", "Mais encor, dites-moi", "Quelle bizarrerie…"]


def test_moliere_subpages_and_front_matter():
    w = "* [[Le Misanthrope/Édition Louandre, 1910/Acte I|Acte I]]\n* [[/Acte II]]\n* [[Le Misanthrope/Notice]]"
    assert moliere.subpages("Le Misanthrope/Édition Louandre, 1910", w) == [
        "Le Misanthrope/Édition Louandre, 1910/Acte I", "Le Misanthrope/Édition Louandre, 1910/Acte II"]
    lines = ["NOTICE", "Texte de Louandre."] + ["PERSONNAGES", "ALCESTE.", "Vers."]
    assert moliere.trim_front_matter(lines) == (["PERSONNAGES", "ALCESTE.", "Vers."], 2)
    assert moliere.trim_front_matter(["PRÉFACE", "PERSONNAGES"]) == (["PRÉFACE", "PERSONNAGES"], 0)


def test_fixture_dir_exists():
    assert shutil.which is not None and FIX.is_dir()


def test_assemblee_partial_zip(tmp_path):
    """Archive tronquée par le serveur : on relit les fichiers complets du début."""
    import zipfile
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k in range(5):
            zf.writestr(f"xml/compteRendu/S{k}.xml", "<a>" + "texte " * 2000 + f"{k}</a>")
    raw = z.read_bytes()
    cut = tmp_path / "cut.zip"
    cut.write_bytes(raw[: len(raw) // 2])
    got = list(assemblee.iter_partial_zip(cut))
    assert 1 <= len(got) < 5
    assert got[0][0] == "xml/compteRendu/S0.xml" and got[0][1].endswith(b"0</a>")


def test_wikinews_article_format_and_date():
    from pipeline.collect import wikinews
    wt_src = ("{{Date|1er janvier 2024}}\nLe {{w|Japon}} a été secoué ce lundi 1{{er}} janvier,\nà {{w|Heure normale du Japon|"
              "heure locale}}.\n\nDeuxième paragraphe.\n\n== Sources ==\n* {{source|url=http://x|titre=X}}\n\n"
              "{{Publication|statut=brève|jour=1|mois=janvier|année=2024}}\n[[Catégorie:Japon]]")
    ctx = {"dump": "20261001", "min_chars": 20}
    doc, why = wikinews.article_doc("42", "Séisme au Japon", "2024-01-06T19:13:09Z", wt_src, ctx)
    assert why is None and doc["id"] == "wikinews_fr:42" and doc["domain"] == "presse"
    assert doc["date"] == "2024-01-01" and doc["meta"]["date_source"] == "modele_date"
    assert doc["meta"]["statut"] == "brève" and doc["license"] == "CC BY 2.5"
    assert doc["text"] == ("1er janvier 2024\nLe Japon a été secoué ce lundi 1er janvier, à heure locale.\n\n"
                           "Deuxième paragraphe.\n\nSources")
    assert wikinews.publication_date("Brèves : 15 avril 2005", "texte")[0] == "2005-04-15"
    assert wikinews.article_doc("1", "X", "2020-01-01T00:00:00Z", "#REDIRECTION [[Y]]", ctx) == (None, "redirection")
