"""Tests Wikisource : conversion wikitexte → texte, et les deux phases sur un mini-dump XML (bz2)."""
import bz2
from pathlib import Path

from pipeline.collect import wikisource as ws
from pipeline.collect import wikitext as wt
from pipeline.io import read_docs

FIX = Path(__file__).parent / "fixtures" / "collect"


def test_templates_and_markup():
    src = ("Le grand\nsilence des {{sc|jours}} d’été<ref>note</ref>\n\nNouveau {{M.|Dupont}} {{c|CENTRÉ}}\n"
           "<poem>Vers un\nVers deux</poem>\n{{Personnage|Alceste|c}}\nOui {{lié|de|la}} [[Auteur:X|X]]"
           "[[Catégorie:Y]][[en:Z]] {{TextQuality|100%}}{{e}} {{corr|faute|correct}} {{lang|la|rosa}}")
    out = wt.to_text(src).replace("\u00a0", " ")
    assert out.split("\n") == ["Le grand silence des jours d’été", "", "Nouveau M. Dupont", "CENTRÉ",
                               "Vers un", "Vers deux", "ALCESTE.", "Oui de la X e correct rosa"]


def test_page_body_and_titles():
    level, body = wt.page_body('<noinclude><pagequality level="3" user="u" />{{nr||12|}}</noinclude>Texte'
                               '<noinclude>{{références}}</noinclude>')
    assert (level, body) == (3, "Texte")
    assert ws.split_page_title("Page:Livre - Titre.djvu/12") == ("Livre - Titre.djvu", 12)
    assert ws.split_page_title("Page:Feuille.jpg") == ("Feuille.jpg", 0)


def test_join_pages():
    assert wt.join_pages(["fin de la page avec un", "mot qui continue.", "Nouvelle page."]) == \
        "fin de la page avec un mot qui continue.\nNouvelle page."
    assert wt.join_pages(["d’honneur", ", on ne lâche"]) == "d’honneur, on ne lâche"


def test_is_theatre():
    play = "\n".join(["ACTE I.", "SCÈNE I."] + [f"ALCESTE.\nVers numéro {i}." for i in range(30)])
    assert wt.is_theatre(play)
    assert not wt.is_theatre("Un roman.\n" * 100)


def test_two_phases_on_mini_dump(tmp_path, monkeypatch):
    from pipeline.collect import common
    monkeypatch.setattr(common, "RAW_DIR", tmp_path / "raw")  # le journal ne va pas dans data/raw
    dump = tmp_path / "mini.xml.bz2"
    dump.write_bytes(bz2.compress((FIX / "frwikisource_mini.xml").read_bytes()))
    ctx = {"dump": "20261001", "min_quality": 3, "min_chars": 20, "buckets": 4}
    p1 = tmp_path / "p1"
    st = ws.phase1((str(dump), str(p1), ctx))
    assert st["docs"] == 1  # « Un conte » ; « La Pièce » (transclusion) et la redirection sont écartées
    assert st["removed_by_reason"]["page_niveau_1"] == 1
    assert st["removed_by_reason"]["ns0_redirection"] == 1
    assert st["removed_by_reason"]["ns0_transclusion_ou_liste"] == 1
    ns0 = [d for p in sorted((p1 / "ns0").glob("part-*.jsonl.zst")) for d in read_docs(p)]
    assert ns0[0]["id"] == "wikisource_fr:page:20" and ns0[0]["date"] == "2023-06-07"
    assert ns0[0]["text"] == ("Un conte\nJean Auteur\n1700\nPremier chapitre\n"
                              "Il était une fois un roi qui vivait dans un château.\n\nLe roi avait une fille.")

    books = []
    for b in range(ctx["buckets"]):
        out = tmp_path / f"p2_{b}"
        _, st2 = ws.phase2((b, [str(p1)], str(out), ctx))
        books += [d for p in sorted(out.glob("part-*.jsonl.zst")) for d in read_docs(p)]
    assert len(books) == 1
    d = books[0]
    assert d["id"] == "wikisource_fr:livre:Auteur - Pièce.djvu"
    assert d["url"] == "https://fr.wikisource.org/wiki/Livre:Auteur_-_Pi%C3%A8ce.djvu"
    assert d["date"] == "2025-03-04"  # dernière révision des pages gardées
    assert d["text"] == ("ACTE PREMIER\nSCÈNE PREMIÈRE\nPHILINTE.\nQu'est-ce donc ? qu'avez-vous ?\n"
                         "ALCESTE.\nMais je veux qu'on soit sincère, et qu'en homme d'honneur, "
                         "on ne lâche aucun mot qui ne parte du cœur.")
    m = d["meta"]
    assert (m["n_pages"], m["n_pages_skipped_quality"]) == (3, 1)
    assert (m["title"], m["author"], m["year"], m["type"]) == ("La Pièce", "Jean Auteur", "1666", "book")


def test_buckets_interleaved_writes(tmp_path):
    """Écritures entremêlées dans plusieurs seaux : chaque fichier reste lisible et complet."""
    b = ws.Buckets(tmp_path, 3)
    recs = [{"i": f"livre{k % 3}", "n": k, "t": "x" * 2000 + str(k)} for k in range(3000)]
    for r in recs:
        b.write(r["n"] % 3, r)
    b.close()
    back = [r for k in range(3) for r in ws.read_bucket_file(tmp_path / f"b{k:03d}.jsonl.zst")]
    assert sorted(r["n"] for r in back) == list(range(3000))
    assert all(r["t"].endswith(str(r["n"])) for r in back)
