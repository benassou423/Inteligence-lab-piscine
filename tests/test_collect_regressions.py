"""Régressions trouvées à la relecture de la collecte (03/11) : chaque test aurait attrapé un défaut réel.

Les appels de modèles viennent de vraies pages de fr.wikisource (citées en commentaire)."""
import gzip

import pytest

from pipeline.collect import common, commoncrawl, wikinews
from pipeline.collect import wikitext as wt


# ---------------------------------------------------------------- dates et gel

def test_date_ok_rejects_malformed_and_impossible_dates():
    assert common.date_ok("2026-10-16") and common.date_ok("2026-10-16T23:59:59Z")
    assert not common.date_ok("2026-9-30")  # comparaison de chaînes : passait avant avec un mois sur 1 chiffre
    assert not common.date_ok("2012-02-31")  # jour inexistant
    assert not common.date_ok("16/10/2026") and not common.date_ok("")


def test_wikinews_impossible_date_falls_back():
    # page parodique « Évènements du 31 février 2012 » : la date du titre n'existe pas
    assert wikinews.parse_fr_date("31 février 2012") is None
    assert wikinews.parse_fr_date("1er mars 2012") == "2012-03-01"
    txt = "{{Publication|jour=31|mois=février|année=2012}}\n" + "Un texte d'article assez long. " * 10
    doc, why = wikinews.article_doc("7", "Évènements du 31 février 2012", "2012-03-02T10:00:00Z", txt,
                                    {"dump": "20261001", "min_chars": 50})
    assert why is None and doc["date"] == "2012-03-02" and doc["meta"]["date_source"] == "derniere_revision"


def test_commoncrawl_uses_current_freeze_date(tmp_path, monkeypatch):
    """Le gel lu dans la config (load_config) sert au choix du crawl, pas une copie figée à l'import."""
    monkeypatch.setattr(common, "RAW_DIR", tmp_path)
    monkeypatch.setattr(common, "FREEZE_DATE", "2026-09-01")
    seen = []

    def fake_latest(freeze, log=None):
        seen.append(freeze)
        return "CC-MAIN-TEST"

    monkeypatch.setattr(commoncrawl, "latest_crawl_before", fake_latest)
    (tmp_path / "commoncrawl").mkdir()
    lines = [f"crawl-data/CC-MAIN-TEST/segments/{s}/warc/f{k}.warc.gz" for s in ("222.2", "111.1") for k in (3, 1, 2)]
    with gzip.open(tmp_path / "commoncrawl" / "_CC-MAIN-TEST.warc.paths.gz", "wt") as f:
        f.write("\n".join(lines) + "\n")

    class Log:
        def info(self, *a): pass
        def warning(self, *a): pass

    cfg = {"sources": {"commoncrawl": {"crawl": None, "n_warc_files": 2}}}
    ctx = commoncrawl.prepare(cfg, Log(), sample=True)
    assert seen == ["2026-09-01"]
    # liste déterministe : ordre trié, premier segment seulement, n premiers fichiers
    assert ctx["files"] == ["crawl-data/CC-MAIN-TEST/segments/111.1/warc/f1.warc.gz",
                            "crawl-data/CC-MAIN-TEST/segments/111.1/warc/f2.warc.gz"]


# ---------------------------------------------------------------- Wikisource : modèles mal rendus

@pytest.mark.parametrize("src, expected", [
    # Page:Diderot - Encyclopedie 1ere edition tome 4.djvu/1002 : une lettre perdue (« afaire »)
    ("a{{ligat|f|f}}aire", "affaire"),
    ("c{{ligat|o|e}}ur et {{Ligat|a|e}}", "cœur et æ"),
    # Page:Bertrand - Mémoires historiques… tremblemens de terre.djvu/26 : le CSS sortait à la place du titre
    ("{{style|font-size:120%; font-variant:small-caps;|Relation chronologique}}", "Relation chronologique"),
    ("{{style|Texte seul}}", "Texte seul"),
    # Page:Vidocq - Les Voleurs - Tome 1.djvu/223 : la cible « dico1/C » à la place du texte affiché
    ("les {{lp|dico1/C|137|chevalier d’industrie|chevaliers d’industrie}}, les", "les chevaliers d’industrie, les"),
    ("{{lp||12|Chapitre I}}", "Chapitre I"),
    # Page:Milton - Cheadle - Voyage de l’Atlantique au Pacifique.djvu/401 : titre perdu
    ("{{lia|17|front|Halte des voyageurs dans la montagne.|396}}", "Halte des voyageurs dans la montagne."),
    # Page:Arrhenius - Le Destin des étoiles, 1921.djvu/196 : texte affiché = un autre modèle ({{fig.|21}})
    ("(voy. {{lia|ch-5|f21|{{fig.|21}}<noinclude>, {{pg|149}}</noinclude>|194}})", "(voy. fig. 21)"),
    ("{{t.|3}}, {{Fig.|2|cap}}", "t. 3, Fig. 2"),
    ("{{hwp|Paris|la capitale}} et {{hwp|Lyon}}", "la capitale et Lyon"),
    ("{{DPRML|abo|abô}}", "abô"),
    ("{{StdT|s|petit texte}}", "petit texte"),
    ("{{ts|ac|bb}}Cellule", "Cellule"),
    ("{{SA|Sans alinéa}}", "Sans alinéa"),
])
def test_wikisource_templates_rendered_like_wikisource(src, expected):
    assert wt.to_text(src).replace(" ", " ") == expected  # espaces insécables : remplacées au nettoyage


def test_pom_keeps_verse_lines():
    src = "{{Pom|Je suis le ténébreux, le veuf,\nl’inconsolé,\nLe prince d’Aquitaine|df}}"
    assert wt.to_text(src) == "Je suis le ténébreux, le veuf,\nl’inconsolé,\nLe prince d’Aquitaine"
    assert wt.to_text("<poem>Vers un\n{{Mpom|Vers deux\nVers trois|c}}\n{{Mpom|—|4}}</poem>") == \
        "Vers un\nVers deux\nVers trois"


def test_leftover_html_and_css_lines_removed():
    # HTML ouvert sur une page et fermé sur la suivante, valeurs CSS seules sur leur ligne (200 livres du 03/11)
    page1 = 'Déposé en 1926.\n<div class="titre">\n\nALAIN\n\nLE CITOYEN'
    page2 = 'CONTRE LES POUVOIRS</div>\n{{c|0.1em}}\n{{c|SAGITTAIRE}}\nTexte <span style="color:red">rouge</span>.'
    out = wt.tidy(wt.join_pages([wt.to_text(page1), wt.to_text(page2)]))
    assert out == "Déposé en 1926.\n\nALAIN\n\nLE CITOYEN\nCONTRE LES POUVOIRS\nSAGITTAIRE\nTexte rouge."
    assert wt.scrub("A\n0.05em\n200px|center\nB<br/>C") == "A\nB\nC"
    assert wt.to_text("si a <b, alors c> d") == "si a <b, alors c> d"  # pas une balise : on n'y touche pas
    assert wt.to_text("{{c|1872}}") == "1872"  # une année centrée reste (seules les longueurs CSS partent)


def test_join_pages_rejoins_word_cut_at_page_end():
    assert wt.join_pages(["c’est principa-", "lement vrai."]) == "c’est principalement vrai."
    assert wt.join_pages(["fin de page -", "suite"]) == "fin de page - suite"  # tiret isolé : inchangé
