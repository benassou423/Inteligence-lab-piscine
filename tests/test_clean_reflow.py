"""Étape reflow : paragraphes recollés, vers et répliques gardés, italiques retirés."""
from pathlib import Path

from pipeline.clean.reflow import ReflowStage, reflow_book
from pipeline.run import Context

PROSE = ("Monsieur Ferrand avait résolu de partir un mardi, parce que le mardi,\n"
         "disait-il, les diligences sont moins pleines et les aubergistes moins\n"
         "pressés. Il prépara sa malle la veille, y rangea trois chemises, un\n"
         "habit de drap et un exemplaire _fatigué_ des fables de La Fontaine.")
VERS = ("Ô rage ! ô désespoir ! ô vieillesse ennemie !\n"
        "N'ai-je donc tant vécu que pour cette infamie ?\n"
        "Et ne suis-je blanchi dans les travaux guerriers\n"
        "Que pour voir en un jour flétrir tant de lauriers ?")
DIALOGUE = ("--Vous reviendrez, au moins? demanda madame Lebrun en le regardant\n"
            "faire depuis sa fenêtre ouverte sur la rue.\n"
            "--On revient toujours, répondit-il.")


def test_prose_recollee_italiques_retires():
    out, info = reflow_book(PROSE + "\n\n" + PROSE)
    para = out.split("\n\n")[0]
    assert "\n" not in para and "un exemplaire fatigué des fables" in para
    assert info["italics"] == 2 and info["lines_joined"] == 6


def test_vers_gardes():
    out, _ = reflow_book(PROSE + "\n\n" + VERS + "\n\n" + PROSE)
    assert VERS in out


def test_repliques_sur_des_lignes_separees():
    out, _ = reflow_book(PROSE + "\n\n" + DIALOGUE)
    assert out.split("\n\n")[1] == ("--Vous reviendrez, au moins? demanda madame Lebrun en le regardant "
                                    "faire depuis sa fenêtre ouverte sur la rue.\n--On revient toujours, répondit-il.")


def test_mot_coupe_en_fin_de_ligne():
    text = ("Il marchait depuis le matin sans rencontrer âme qui vive, et c'était peut-\n"
            "être la première fois de sa vie qu'il goûtait un pareil silence au cœur.\n"
            "Fin.")
    out, _ = reflow_book(text)
    assert "peut-être la première fois" in out


def test_autres_sources_inchangees():
    st = ReflowStage({}, Context(root=Path("."), config_name="t", sources=["wikipedia_fr"]))
    doc = {"id": "w:1", "source": "wikipedia_fr", "text": PROSE}
    assert st.apply(doc).doc["text"] == PROSE
