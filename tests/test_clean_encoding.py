"""Étape encoding : mojibake réparé, typographie française intacte, résultat identique à ftfy."""
import random

import ftfy
import pytest

from pipeline.clean.encoding import DEFAULT_CONFIG, EncodingStage, clean_text, fix_text, normalize_spaces, triggers

TYPO = "L’été « chaud » : c’est ça — vraiment… « Oui ! » dit-elle ; 5 €."


def test_mojibake_repare():
    assert fix_text("Ã©tÃ©") == "été"
    assert fix_text("Lâ€™Ã©tÃ© est chaud : câ€™est la saison des fÃªtes.") == \
        "L’été est chaud : c’est la saison des fêtes."
    assert fix_text("Ã‰cole normale") == "École normale"


def test_typographie_francaise_conservee_espaces_normalisees():
    # ftfy seul ne touche à rien (ftfy n'est même pas appelé)...
    assert fix_text(TYPO) == TYPO
    assert triggers(TYPO) == []
    # ... puis les espaces insécables deviennent des espaces simples (la val du kit n'en a aucune),
    # mais apostrophe et guillemets courbes, tiret cadratin et points de suspension restent.
    assert clean_text(TYPO) == "L’été « chaud » : c’est ça — vraiment… « Oui ! » dit-elle ; 5 €."


def test_espaces_et_invisibles():
    assert normalize_spaces("a b c d　e") == "a b c d e"
    assert normalize_spaces("anti­constitution​nel") == "anticonstitutionnel"
    assert normalize_spaces("rien à faire – ici") == "rien à faire – ici"  # tiret demi-cadratin gardé


def test_nfc():
    assert fix_text("café") == "café"  # e + accent combinant -> é précomposé


def test_fins_de_ligne_windows():
    assert fix_text("ligne 1\r\nligne 2\r\n") == "ligne 1\nligne 2\n"


def test_guillemet_de_continuation_pas_pris_pour_du_macroman():
    """ftfy lit « »É » comme du MacRoman et en fait « ȃ » : 15 livres Gutenberg abîmés sur le corpus."""
    for s in ("»Éminence, je vous prie.", "»Ça va.", "»Ébloui", "texte normal\n»Écrivez-moi, je vous prie."):
        assert ftfy.fix_text(s, DEFAULT_CONFIG) != s  # le faux positif de ftfy existe bien...
        assert fix_text(s) == s  # ... et il est refusé
    # Le reste du texte est toujours réparé, ligne par ligne
    assert fix_text("»Écoute.\nLâ€™Ã©tÃ©.") == "»Écoute.\nL’été."
    # Une vraie réparation MacRoman (sans guillemet mangé) reste faite
    assert fix_text("caf√© cr√®me") == ftfy.fix_text("caf√© cr√®me", DEFAULT_CONFIG) == "café crème"


@pytest.mark.parametrize("seed", [0, 1])
def test_identique_a_ftfy(seed):
    """Le pré-filtre rapide ne change pas le résultat de ftfy (cas adverses aléatoires), sauf
    les faux positifs MacRoman/cp437 refusés : là, ftfy mange un guillemet et nous non."""
    rng = random.Random(seed)
    pieces = ["La Révolution", " française", "« Oui » ", "c’est l’été", " : ", " ;", " — ", "…",
              "Ã©tÃ©", "Ã ", "Ã la maison", "câ€™est", "Ã‰cole", "Â« ", "â€”", "\r\n", "\n", "\r", "&amp;",
              "&eacute;", "<b>", "ﬁ", "ＡＢ", "\x00", "\x85", "\x92", " ", "é", "œuvre", "😀",
              "\x1b[31m", "ÃƒÂ©", "Il est parti.\n", "\nÃ ", "√©", "»É", "»Ça", "«é"]
    for _ in range(3000):
        s = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 10)))
        ours, ref = fix_text(s), ftfy.fix_text(s, DEFAULT_CONFIG)
        if ours != ref:
            n_guill = [sum(x.count(g) for g in "«»") for x in (s, ours, ref)]
            assert n_guill[2] < n_guill[0] == n_guill[1] or n_guill[2] < n_guill[1], repr(s)


def test_uncurl_quotes_interdit():
    from pipeline.run import Context
    from pathlib import Path

    ctx = Context(root=Path("."), config_name="t", sources=["s"])
    with pytest.raises(ValueError):
        EncodingStage({"uncurl_quotes": True}, ctx)


def test_etape_compte_les_modifications():
    from pipeline.run import Context
    from pathlib import Path

    stage = EncodingStage({}, Context(root=Path("."), config_name="t", sources=["wikipedia_fr"]))
    stage.bind([])
    out = stage.apply({"id": "w:1", "source": "wikipedia_fr", "text": "Ã©tÃ© Ã  Paris"})
    assert out.keep and out.doc["text"] == "été à Paris"
    assert out.counters["modified"] == 1 and out.counters["fix:mojibake"] == 1
    out = stage.apply({"id": "w:2", "source": "wikipedia_fr", "text": TYPO})
    assert out.doc["text"] == clean_text(TYPO) and out.counters == {"modified": 1, "fix:spaces": 1}
    out = stage.apply({"id": "w:3", "source": "wikipedia_fr", "text": "Rien à corriger ici."})
    assert not out.counters
