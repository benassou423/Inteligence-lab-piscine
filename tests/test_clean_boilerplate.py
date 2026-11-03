"""Étape boilerplate : HTML -> texte principal, licence Gutenberg retirée, fin d'article Wikipédia."""
from pathlib import Path

from pipeline.clean.boilerplate import (
    BoilerplateStage, extract_main_text, is_end_heading, strip_gutenberg, strip_wikipedia,
)
from pipeline.run import Context

FX = Path(__file__).parent / "fixtures" / "clean"


def _stage():
    st = BoilerplateStage({}, Context(root=Path("."), config_name="t", sources=["commoncrawl"]))
    st.setup()
    return st


def test_html_menu_et_pied_de_page_retires():
    text = extract_main_text((FX / "page_actu.html").read_text(encoding="utf-8"))
    assert "Dans le quartier de la Croix-Rousse" in text
    assert "Tant qu’on pourra semer, on sèmera." in text  # typographie gardée
    for junk in ("Accueil", "Se connecter", "cookies", "Mentions légales", "Tous droits réservés",
                 "À lire aussi", "Jean-Pierre69", "Commentaires"):
        assert junk not in text, junk


def test_html_sans_contenu_retire():
    doc = {"id": "commoncrawl:x", "source": "commoncrawl", "text": "",
           "html": "<html><head><script>var a=1;</script></head><body><img src='a.png'></body></html>"}
    out = _stage().apply(doc)
    assert not out.keep and out.reason == "no_extraction"


def test_gutenberg_moderne():
    text = (FX / "gutenberg_moderne.txt").read_text(encoding="utf-8")
    body, header, footer, info = strip_gutenberg(text)
    assert body.startswith("LE VOYAGE DE MONSIEUR FERRAND")
    assert body.rstrip().endswith("FIN")
    for junk in ("Project Gutenberg", "*** START", "*** END", "Produced by", "gallica", "LICENSE"):
        assert junk not in body, junk
    assert info["start_marker"] == 1 and info["end_marker"] == 1 and info["credits"] == 1
    assert "*** START OF THE PROJECT GUTENBERG EBOOK" in header and "FULL LICENSE" in footer


def test_gutenberg_ancien_format():
    text = (FX / "gutenberg_ancien.txt").read_text(encoding="utf-8")
    body, _, _, info = strip_gutenberg(text)
    assert body.startswith("CONTES DU SOIR")
    assert body.rstrip().endswith("roue son chant.")
    assert "SMALL PRINT" not in body and "Etext" not in body and "prepared by" not in body


def test_gutenberg_credits_en_plusieurs_paragraphes():
    """Cas réels (10 livres sur 4 228) : deux paragraphes de crédits, « Credits: », ancien en-tête recopié."""
    head = "Titre: X\n\n*** START OF THE PROJECT GUTENBERG EBOOK LA VAMPIRE ***\n\n"
    livre = "LA VAMPIRE\n\npar\n\nPAUL FÉVAL\n\nCeci est une étrange histoire.\n"
    cas = [
        "This file was produced from images generously made available by the\nBibliothèque nationale de "
        "France (BnF/Gallica) at http://gallica.bnf.fr.\n\nProduced by Carlo Traverso, Anne Dreze and the PG "
        "Online Distributed\nProofreaders.\n\n\n\n",
        "Credits: Christine De Ryck and the Online Distributed Proofreading Team.\n\n\n",
        "Produced by Carlo Traverso.\n\n\nThe Project Gutenberg EBook of Pot-bouille, by Emile Zola\n\n"
        "Copyright laws are changing all over the world. Be sure to check the\ncopyright laws.\n\n"
        "This header should be the first thing seen when viewing this Project\nGutenberg file.\n\n\n",
    ]
    for credits in cas:
        body, _, _, info = strip_gutenberg(head + credits + livre + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
        assert body.startswith("LA VAMPIRE") and info["credits"] == 1, body[:80]
    # Un livre sans crédits n'est pas entamé, même s'il parle de crédits plus loin
    body, _, _, info = strip_gutenberg(head + livre + "\n\n" * 8 + "Produced by personne.\n")
    assert body.startswith("LA VAMPIRE") and not info["credits"]


def test_gutenberg_sans_marqueur_inchange():
    text = "Un texte sans aucun marqueur.\nDeuxième ligne."
    assert strip_gutenberg(text)[0] == text


def test_wikipedia_sections_finales():
    text = (FX / "wikipedia_article.txt").read_text(encoding="utf-8")
    body, cut, _ = strip_wikipedia(text)
    assert cut == "notes et références"
    assert body.endswith("permis la production d'électricité.")
    for junk in ("Bibliographie", "Liens externes", "Articles connexes", "Notices d'autorité"):
        assert junk not in body


def test_wikipedia_section_notes_redigee_gardee():
    prose = ("Ceci est une longue note rédigée qui explique un point du texte. " * 6).strip()
    text = ("Intro de l'article. Elle est courte.\n\nNotes\n\n" + prose +
            "\n\nLiens externes\n Site officiel\n Autre lien")
    body, cut, _ = strip_wikipedia(text)
    assert cut == "liens externes"
    assert prose in body and "Site officiel" not in body


def test_wikipedia_residus_de_fin():
    text = "Le village compte trois fermes. Il est calme.\n\nHéraldique\n\n.\n\nHomonymie de communes en France"
    body, cut, info = strip_wikipedia(text)
    assert body == "Le village compte trois fermes. Il est calme." and cut is None
    assert info["residue_lines"] == 3


def test_wikipedia_wikitexte_brut_retire():
    text = ("{{Titre mis en forme|}}\n\nLe club joue à Abha. Il a été fondé en 1966.\n\n{{Infobox Club\n"
            "| nom = Abha Club\n| pays = Arabie saoudite\n}}\nSon stade compte dix mille places.\n\n"
            "Effectif actuel \n|-\n\n|-\n\n{|\n|}\n\nNotes et références")
    body, cut, info = strip_wikipedia(text)
    assert body == "Le club joue à Abha. Il a été fondé en 1966.\n\nSon stade compte dix mille places."
    assert info["markup_lines"] == 9 and cut == "notes et références"


def test_titres_de_fin():
    for t in ("Notes et références", "Notes, sources et références", "Voir aussi", "Liens externes ",
              "Articles connexes", "Pour approfondir", "Liens internes"):
        assert is_end_heading(t), t
    for t in ("Histoire", "Géographie", " Liens externes", "Notes de l'auteur sur la traduction"):
        assert not is_end_heading(t), t
