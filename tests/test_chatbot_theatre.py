"""Tests du découpage en scènes et répliques (chatbot/theatre.py), sur de petits extraits écrits ici."""
import json

import numpy as np

from chatbot import theatre
from chatbot.theatre import (classify, classify_text, dedupe_across_documents, distinctive_speakers, doc_metrics,
                             exclusion_report, ngram_hashes, parse_scenes, same_title, scene_records, shared,
                             split_long, theatre_verdict, turns_text)
from pipeline.io import write_docs

# Typographie du kit : « NOM. » seul, ligne vide, réplique en prose coupée à 72 colonnes.
KIT = """LE MÉDECIN VOLANT

PERSONNAGES

GORGIBUS, père de Lucile.
LUCILE, fille de Gorgibus.

SCÈNE I.--VALÈRE, SABINE.

VALÈRE.

Eh bien, Sabine, quel conseil me donnes-tu?

SABINE.

Vraiment, il y a bien des nouvelles. Mon oncle veut résolûment que ma
cousine épouse Villebrequin, et les affaires sont tellement avancées.

VALÈRE, à part.

(Haut.)

C'est un lourdaud qui gâtera tout.

SCÈNE II.--VALÈRE, SGANARELLE.

VALÈRE.

Ah! mon pauvre Sganarelle.

SGANARELLE.

Ce que je sais faire, monsieur?
"""

# Typographie Wikisource : « NOM » seul, réplique sur les lignes suivantes, didascalies entre parenthèses.
WIKISOURCE = """Scène II
Le Docteur, Le Barbouillé.
LE BARBOUILLÉ
Je m’en allois vous chercher.
LE DOCTEUR
Il faut que tu sois bien mal appris.
(Il s’en va.)
LE BARBOUILLÉ
Ma foi, je m’y suis mépris.
LE DOCTEUR, troussant sa robe.
Sache, mon ami, que je ne veux point d'argent.
"""

# Pièce en vers : un vers par ligne, chaque vers commence par une capitale.
VERSE = """LÉLIE.

Ah! Mascarille!

MASCARILLE.

Quoi?

LÉLIE.

Voici bien des affaires;
J'ai dans ma passion toutes choses contraires.
"""


def test_kit_typography_prose_joined_and_cast_skipped():
    scenes = parse_scenes(KIT, strict=False)
    assert len(scenes) == 2
    first = scenes[0]
    assert [t["speaker"] for t in first] == ["VALÈRE", "SABINE", "VALÈRE"]
    # prose coupée recollée en une ligne ; didascalie « (Haut.) » retirée ; « , à part. » retiré du nom
    assert first[1]["text"] == ("Vraiment, il y a bien des nouvelles. Mon oncle veut résolûment que ma "
                                "cousine épouse Villebrequin, et les affaires sont tellement avancées.")
    assert first[2]["text"] == "C'est un lourdaud qui gâtera tout."
    assert all(t["speaker"] not in ("GORGIBUS", "LUCILE") for s in scenes for t in s)   # liste des personnages


def test_wikisource_typography_and_didascalies():
    (scene,) = parse_scenes(WIKISOURCE, strict=False)
    assert [t["speaker"] for t in scene] == ["LE BARBOUILLÉ", "LE DOCTEUR", "LE BARBOUILLÉ", "LE DOCTEUR"]
    assert scene[1]["text"] == "Il faut que tu sois bien mal appris."        # « (Il s’en va.) » retiré
    assert scene[3]["text"].startswith("Sache, mon ami")


def test_verse_keeps_line_breaks():
    (scene,) = parse_scenes(VERSE)
    assert scene[2]["text"] == "Voici bien des affaires;\nJ'ai dans ma passion toutes choses contraires."
    for t in scene:
        assert t["text"] == t["text"].strip() and "\n\n" not in t["text"]


def test_speaker_with_stage_direction_and_inline_form():
    text = ("HARPAGON, à part.\n\nAu voleur!\n\nVALÈRE.\n\nQu'avez-vous?\n\nANTONIO.--Qui est là?\n\n"
            "GRATIANO.--C'est vous?\n\nMASCARILLE.\n\nAh! vicomte!\nJODELET. (Ils s'embrassent.)\nAh! marquis!\n"
            "\nCYRANO (riant):\n(A lui-même):\nLâche!\n")
    (scene,) = parse_scenes(text)
    assert [t["speaker"] for t in scene] == ["HARPAGON", "VALÈRE", "ANTONIO", "GRATIANO", "MASCARILLE", "JODELET",
                                             "CYRANO"]
    assert scene[2]["text"] == "Qui est là?"
    assert scene[4]["text"] == "Ah! vicomte!" and scene[5]["text"] == "Ah! marquis!"
    assert scene[6]["text"] == "Lâche!"                     # didascalie « (A lui-même): » retirée


def test_strict_mode_cuts_on_narrative_prose():
    # Un vrai bloc de récit (plus de 5 paragraphes avant le nom suivant) ferme la scène en mode strict.
    # (Un seul paragraphe court suivi d'un nom est désormais gardé : voir le test suivant.)
    narrative = "".join(f"Le récit continue, paragraphe {i}, la lettre avait été concertée entre elles.\n\n"
                        for i in range(6))
    text = ("LE MARQUIS.\n\nRendez-moi ce service.\n\nMADAME.\n\nLa chose est difficile.\n\n"
            + narrative + "LE MARQUIS.\n\nSeul.\n\nMADAME.\n\nTant pis.\n")
    scenes = parse_scenes(text, strict=True)
    assert [len(s) for s in scenes] == [2, 2]
    assert "lettre" not in turns_text(scenes)
    assert scenes[0][1]["text"] == "La chose est difficile."


def test_strict_mode_keeps_verses_separated_by_blank_lines():
    # Format Wikisource : une ligne vide entre les vers. La réplique doit rester entière (avant la
    # correction, la scène était fermée sur « Bonjour, Anaïk, » et la suite perdue).
    text = ("MARIA.\n\nBonjour, Anaïk,\n\nqui mourut, il y a trois mois d'hier matin,\n\n"
            "les grenouilles ne chantent plus à l'étang.\n\nANAÏK.\n\nBonjour, Maria.\n\n"
            "MARIA.\n\nTu ne dis rien ?\n")
    (scene,) = parse_scenes(text, strict=True)
    assert [t["speaker"] for t in scene] == ["MARIA", "ANAÏK", "MARIA"]
    assert scene[0]["text"].endswith("les grenouilles ne chantent plus à l'étang.")
    assert "trois mois" in scene[0]["text"]


def test_strict_mode_drops_short_unparenthesized_stage_direction():
    text = "HORACE.\n\nAdieu !\n\nIl sort.\n\nAGNÈS.\n\nIl part.\n\nHORACE.\n\nJe reviens.\n"
    (scene,) = parse_scenes(text, strict=True)
    assert scene[0]["text"] == "Adieu !"


def test_dictionary_false_positive_rejected():
    # Un dictionnaire de botanique : des « NOMS. » en capitales, mais chacun n'apparaît qu'une fois.
    entries = [f"{w}.\n\nPlante herbacée de la famille des {w.lower()}.\n" for w in
               ("ABRICOT", "ABSINTHE", "ACACIA", "ACANTHE", "ACONIT", "AGAVE", "ALOÈS", "ANIS", "ARNICA",
                "ARUM", "AULNE", "AVOINE", "BAMBOU", "BASILIC", "BEGONIA", "BLÉ", "BOULEAU", "BRUYÈRE",
                "BUIS", "CACTUS", "CAFÉIER", "CAMÉLIA", "CANNE", "CÂPRIER")]
    text = "\n".join(entries)
    scenes = parse_scenes(text)
    m = doc_metrics(scenes, text)
    assert theatre_verdict(m) is not None


def test_real_play_accepted_by_detector():
    lines = []
    for i in range(30):
        lines += ["ALCESTE.", "", "Réplique d'Alceste, " + "assez " * (i % 4) + "courte.", "",
                  "PHILINTE.", "", "Réponse de Philinte, " + "fort " * (i % 3) + "brève.", ""]
    text = "\n".join(lines)
    scenes = parse_scenes(text)
    assert theatre_verdict(doc_metrics(scenes, text)) is None


def test_long_scene_split_in_chunks_of_60():
    turns = [{"speaker": "A" if i % 2 else "B", "text": f"r{i}"} for i in range(130)]
    chunks = split_long(turns)
    assert len(chunks) == 3 and all(len(c) <= 60 for c in chunks)
    assert sum(len(c) for c in chunks) == 130
    recs = scene_records([turns], "gutenberg_fr", "123", "Pièce", "gutenberg_fr:123", None, None, False)
    assert [r["id"] for r in recs] == ["gutenberg_fr:123:0001-1", "gutenberg_fr:123:0001-2", "gutenberg_fr:123:0001-3"]


def test_overlap_with_validation_detected():
    val = ("Non, je ne puis souffrir cette lâche méthode qu'affectent la plupart de vos gens à la mode "
           "et je ne hais rien tant que les contorsions de tous ces grands faiseurs de protestations")
    ref = ngram_hashes(val)
    # autre édition : ponctuation, majuscules et accents différents → mêmes 13-grammes
    other = "PHILINTE.\n" + val.upper().replace("â", "a").replace(",", ";") + "\nFIN."
    assert shared(ngram_hashes(other), ref) >= 3
    unrelated = "Le petit chat dort au soleil sur le rebord de la fenêtre pendant que la pluie tombe dehors"
    assert shared(ngram_hashes(unrelated), ref) == 0
    assert shared(np.empty(0, dtype=np.uint64), ref) == 0


def test_titles_from_different_editions_match():
    assert same_title("LE FESTIN DE PIERRE", "Don Juan, ou le Festin de Pierre")
    assert same_title("LE TARTUFFE", "Tartuffe, ou l’Imposteur")
    assert same_title("LA PASTORALE COMIQUE", "Pastorale comique")
    assert not same_title("L'AVARE", "Le Bourgeois gentilhomme")


# ---------------------------------------------------------------------------------------------
# Corrections de la revue du 12/11 (un test par correction qui change le résultat)
# ---------------------------------------------------------------------------------------------

def test_shared_verse_echo_removed_in_chain():
    # Chantecler : chaque réplique recopiait la fin de la précédente avant sa moitié de vers.
    text = ("LE MERLE.\nEncor ! Cachez-moi !\nCHANTECLER.\nEncor ! Cachez-moi ! Ciel !\n"
            "PATOU.\nEncor ! Cachez-moi ! Ciel ! Un faisan doré !\nLE MERLE.\nOui.\nCHANTECLER.\nOui.\n"
            "PATOU.\nNon.\n")
    (scene,) = parse_scenes(text, strict=True)          # corpus seulement (mode strict), jamais chez Molière
    assert [t["text"] for t in scene[:3]] == ["Encor ! Cachez-moi !", "Ciel !", "Un faisan doré !"]


def test_full_echo_written_by_author_is_kept():
    text = "ARLEQUIN.\nBonjour, monsieur.\nPIERROT.\nBonjour, monsieur.\nARLEQUIN.\nAdieu.\n"
    (scene,) = parse_scenes(text, strict=True)
    assert scene[1]["text"] == "Bonjour, monsieur."


def test_name_without_punctuation_recognized_if_already_a_speaker():
    # Dumas : « D’ÉPERNON Et toi ? » (pas de point après le nom) devient une réplique à part.
    text = "\n".join(["D’ÉPERNON.", "Allons !", "JOYEUSE.", "Oui.", "D’ÉPERNON.", "Courage !", "JOYEUSE.", "Soit.",
                     "D’ÉPERNON.", "Bien.", "RUGGIERI.", "Voici le philtre.", "D’ÉPERNON Et toi ?",
                     "JOYEUSE.", "Moi ?", ""])
    (scene,) = parse_scenes(text, strict=False)
    assert [t["speaker"] for t in scene][-3:] == ["RUGGIERI", "D’ÉPERNON", "JOYEUSE"]
    assert scene[-2]["text"] == "Et toi ?"


def test_capitalised_prose_sentence_stays_text():
    # « PARIS Est une grande ville » : PARIS n'est jamais locuteur, la ligne reste du texte.
    text = "LÉLIE.\nPARIS Est une grande ville.\nMASCARILLE.\nSans doute.\n"
    (scene,) = parse_scenes(text, strict=False)
    assert scene[0] == {"speaker": "LÉLIE", "text": "PARIS Est une grande ville."}


def test_section_labels_reject_document():
    # Spinoza, Éthique : DÉMONSTRATION, SCOLIE… reviennent comme des « personnages ».
    lines = []
    for i in range(30):
        lines += ["DÉMONSTRATION.", "", f"La substance est cause de soi, voir la proposition {'x' * (i % 3)}.", "",
                  "SCOLIE.", "", "Ceci suit de ce qui précède.", ""]
    text = "\n".join(lines)
    m = doc_metrics(parse_scenes(text), text)
    assert m["sections"] == 1.0 and theatre_verdict(m) == "etiquettes_de_section"


def test_singing_and_staging_labels_are_not_speakers():
    text = ("LES MÊMES, Plumard entre.\n\nPLUMARD.\n\nMe voici !\n\nCOLETTE.\n\nEnfin.\n\n"
            "PLUMARD.\n\nAIR : Ô ma tendre musette.\n\nJe chante pour vous.\n\nCOLETTE.\n\nMerci.\n")
    (scene,) = parse_scenes(text)
    assert [t["speaker"] for t in scene] == ["PLUMARD", "COLETTE", "PLUMARD", "COLETTE"]
    assert scene[2]["text"] == "Je chante pour vous."


def test_cast_heading_requires_capital():
    assert classify("personnages qu'il représente, et se servir des mêmes traits")[0] == "text"
    assert classify("PERSONNAGES DU BALLET.")[0] == "cast"
    assert classify("Personnages")[0] == "cast"


def test_leading_verse_numbers_removed_from_numbered_play():
    lines = []
    for k in range(12):
        lines += ["CIDIPPE.", f"{540 + 5 * k} À ne vous point mentir, je sens que dans mon cœur",
                  "AGLAURE.", "Ma sœur, je vous entends."]
    scenes = parse_scenes("\n".join(lines), strict=False)
    assert all(not t["text"][0].isdigit() for s in scenes for t in s)
    assert scenes[0][0]["text"] == "À ne vous point mentir, je sens que dans mon cœur"


def test_dates_and_amounts_are_not_verse_numbers():
    text = "\n".join(["MADAME.", "14 novembre 1863, il pleuvait.", "MONSIEUR.", "162 francs !",
                     "MADAME.", "C'est cher."])
    (scene,) = parse_scenes(text, strict=False)
    assert scene[0]["text"] == "14 novembre 1863, il pleuvait." and scene[1]["text"] == "162 francs !"


def test_wrapped_stage_direction_does_not_enter_reply():
    # Kit coupé à 72 colonnes : la fin de la didascalie était collée au début de la réplique.
    text = ("SGANARELLE, chantant, puis se retournant tout à coup sans\n"
            "apercevoir Valère ni Lucas.\n\nLa, la, la.\n\nVALÈRE.\n\nC'est lui.\n\n"
            "MERCURE, sur un nuage; LA NUIT, dans un char traîné par\ndeux chevaux.\n\n"
            "Tout beau! charmante Nuit.\n")
    (scene,) = parse_scenes(text, strict=False)
    assert scene[0] == {"speaker": "SGANARELLE", "text": "La, la, la."}
    assert scene[2] == {"speaker": "MERCURE", "text": "Tout beau! charmante Nuit."}


def test_wikisource_short_stage_direction_unchanged():
    # Wikisource : didascalie courte, réplique juste en dessous, sans ligne vide : rien n'est consommé.
    events = classify_text("SGANARELLE, sortant de la maison, en médecin\nVoilà qui est fait.\nGÉRONTE\nBien.")
    assert [k for k, _ in events] == ["speaker", "text", "speaker", "text"]


def _play(a, b, n=30, extra=""):
    """Petite pièce de deux personnages, assez longue pour passer le détecteur du corpus."""
    lines = []
    for i in range(n):
        lines += [f"{a}.", "", f"Réplique numéro {'un ' * (i % 4)}de {a.lower()}, assez courte {i * 'x'} ?", "",
                  f"{b}.", "", f"Réponse de {b.lower()}, {'fort ' * (i % 3)}brève {i * 'y'} !", ""]
    return "\n".join(lines) + extra


def test_dedupe_scenes_across_editions():
    turns = [{"speaker": "A", "text": "un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze"},
             {"speaker": "B", "text": "quinze seize dix-sept dix-huit dix-neuf vingt"}]
    other = [{"speaker": "A", "text": "rien de commun avec la scène précédente, mais assez de mots pour un essai"},
             {"speaker": "B", "text": "vraiment rien du tout, ni les mots ni l'ordre de ces mots ici"}]
    recs = [{"doc_id": "ws:1", "turns": turns}, {"doc_id": "ws:1", "turns": turns},   # refrain : gardé
            {"doc_id": "gut:2", "turns": turns}, {"doc_id": "gut:2", "turns": other}]
    kept, dropped = dedupe_across_documents(recs)
    assert [r["doc_id"] for r in kept] == ["ws:1", "ws:1", "gut:2"] and kept[2]["turns"] is other
    assert dropped == {("ws:1", "gut:2"): 1}


def _ref(val_text, mol_text="", val_speakers=None):
    val, mol = ngram_hashes(val_text), ngram_hashes(mol_text)
    return {"val": val, "val12": ngram_hashes(val_text, 12), "mol": mol, "val_mol": np.union1d(val, mol),
            "val_speakers": val_speakers or {}, "exclusions": {"gutenberg_fr:999": "erotique_ou_obscene"}}


VERS_VAL = "Ah! tout est ruiné;\nJe suis, je suis trahi, je suis assassiné;\nUn homme, ah! quel affront!"


def test_twelve_word_verse_from_validation_removes_scene():
    # Don Garcie / Le Misanthrope : vers repris de 12 mots, invisible aux 13-grammes.
    theatre._init_worker(_ref(VERS_VAL))
    leak = [{"speaker": "DON GARCIE", "text": "Ah! tout est ruiné;\nJe suis, je suis trahi, je suis assassiné;"},
            {"speaker": "ÉLISE", "text": "Seigneur, calmez-vous."}]
    assert shared(ngram_hashes(turns_text([leak])), ngram_hashes(VERS_VAL)) == 0     # 13 mots : rien vu
    assert theatre.scene_reason(leak) == "scenes_retirees_val"


def test_validation_characters_scene_removed_generic_roles_kept():
    cast = {"LE MISANTHROPE": distinctive_speakers([[{"speaker": s, "text": "x"} for s in
                                                     ("ALCESTE", "PHILINTE", "PREMIER MUSICIEN", "TROISIÈME MUSICIEN",
                                                      "LE GARDE")]])}
    assert cast["LE MISANTHROPE"] == {"ALCESTE", "PHILINTE"}
    theatre._init_worker(_ref("texte sans rapport", val_speakers=cast))
    assert theatre.scene_reason([{"speaker": "ALCESTE", "text": "Moi ?"},
                                 {"speaker": "PHILINTE", "text": "Vous."}]) == "scenes_retirees_personnages_val"
    assert theatre.scene_reason([{"speaker": "PREMIER MUSICIEN", "text": "La, la !"},
                                 {"speaker": "TROISIÈME MUSICIEN", "text": "Do ?"}]) is None


def test_corpus_document_filters(tmp_path):
    val_text = ("Non, je ne puis souffrir cette lâche méthode qu'affectent la plupart de vos gens à la mode "
                "et je ne hais rien tant que les contorsions de tous ces grands faiseurs de protestations")
    note = "\nNOTE. Molière écrivait : " + val_text + "\n"
    docs = [
        # non-Molière, une note cite la validation : le document garde ses scènes (ancienne règle : exclu entier)
        {"id": "wikisource_fr:a", "source": "wikisource_fr", "text": _play("LEBONNARD", "ROBERT", extra=note),
         "meta": {"title": "Le Père Lebonnard", "authors": "Aicard"}},
        # pièce SUR Molière, sans recouvrement : gardée, mais is_moliere reste faux (pas une pièce DE Molière)
        {"id": "wikisource_fr:c", "source": "wikisource_fr", "text": _play("ARMANDE", "BARON"),
         "meta": {"title": "La Fille de Molière", "authors": "Fournier"}},
        # édition de Molière (auteur Molière) qui recouvre la validation : exclue entière
        {"id": "gutenberg_fr:b", "source": "gutenberg_fr", "text": _play("ALCESTE", "PHILINTE", extra=note * 3),
         "meta": {"title": "Œuvres", "authors": "Molière"}},
        # document de la liste d'exclusion : écarté avant toute analyse
        {"id": "gutenberg_fr:999", "source": "gutenberg_fr", "text": _play("DORANTE", "CLITANDRE"), "meta": {}},
    ]
    path = tmp_path / "part-0000.jsonl.zst"
    write_docs(path, docs)
    theatre._init_worker(_ref(val_text, val_speakers={"LE MISANTHROPE": {"ALCESTE", "PHILINTE"}}))
    _, records, rows, n_docs, _, excluded = theatre.process_file(path)
    assert n_docs == 4 and excluded == ["gutenberg_fr:999"]
    by_id = {r["id"]: r for r in rows}
    assert by_id["wikisource_fr:a"]["reason"] == "retenu" and by_id["wikisource_fr:a"]["shared_val"] >= 3
    assert by_id["gutenberg_fr:b"]["reason"] == "recouvrement_val"
    assert by_id["wikisource_fr:c"]["reason"] == "retenu"
    assert records and {r["doc_id"] for r in records} == {"wikisource_fr:a", "wikisource_fr:c"}
    assert all(r["is_moliere"] is False for r in records)


def test_exclusion_report_warns_on_missing_ids():
    rep = exclusion_report({"a": "erotique_ou_obscene", "b": "pas_du_dialogue", "c": "erotique_ou_obscene"},
                           ["a", "c"])
    assert rep["documents_ecartes"] == 2 and rep["par_raison"] == {"erotique_ou_obscene": 2}
    assert rep["introuvables"] == ["b"] and len(rep["avertissements"]) == 1


# ---------------------------------------------------------------------------------------------
# Corrections après la vérification des corrections (12/11, second tour)
# ---------------------------------------------------------------------------------------------

def test_strict_never_glues_a_reply_written_name_dot_text():
    # « MÉNECHME. Salut. » (nom, point, réplique) n'est pas reconnu comme un nom : avant cette correction,
    # le recollage « nom proche » le mettait dans la réplique de LÉLIE (plusieurs personnages dans un tour).
    text = "VALÈRE.\nQui va là ?\nLÉLIE.\nBonjour.\n\nMÉNECHME. Salut.\nCLÉANTE.\nAu revoir.\nVALÈRE.\nAdieu.\n"
    scenes = parse_scenes(text, strict=True)
    assert scenes[0] == [{"speaker": "VALÈRE", "text": "Qui va là ?"}, {"speaker": "LÉLIE", "text": "Bonjour."}]
    assert "MÉNECHME" not in turns_text(scenes)
    assert theatre.reply_like("LÉONTINE (2), à Duchotel (3).") and theatre.reply_like("ROXANE. Battu !")
    assert not theatre.reply_like("M. Dupont est arrivé.") and not theatre.reply_like("OUI, monsieur.")


def test_named_stage_direction_removed_but_validation_frozen():
    # L'Étourdi : « Lélie sort. » restait dans la réplique en mode non strict (Molière).
    text = ("LÉLIE.\n\nAdieu, je cours.\n\nLélie sort.\n\nMASCARILLE.\n\nLe voilà parti.\nLélie frappe à sa porte.\n"
            "Holà !\n\nLÉLIE.\n\nMe revoici.\n")
    (scene,) = parse_scenes(text, strict=False)
    assert scene[0]["text"] == "Adieu, je cours." and scene[1]["text"] == "Le voilà parti.\nHolà !"
    # Le Tartuffe : « Il entre. » est une vraie réplique de Dorine ; le pronom seul ne suffit pas chez Molière.
    (tartuffe,) = parse_scenes("DAMIS.\n\nEt, s'il falloit...\n\nDORINE.\n\nIl entre.\n", strict=False)
    assert tartuffe[1] == {"speaker": "DORINE", "text": "Il entre."}
    # Validation figée (cleanup=False) : règles d'origine, le texte n'est pas touché.
    (frozen,) = parse_scenes(text, strict=False, cleanup=False)
    assert frozen[0]["text"] == "Adieu, je cours.\nLélie sort."
    # Un nom qui n'est pas un personnage de la pièce n'est pas une didascalie.
    (other,) = parse_scenes("LÉLIE.\nValère sort.\nMASCARILLE.\nTant mieux.\n", strict=False)
    assert other[0]["text"] == "Valère sort."


def test_dedupe_second_pass_keeps_the_longest_copy():
    letters = "abcdefghijklmnpqrstuvwxz"                   # sans « o » ni « y » (repli d'orthographe)
    words = [f"mot{letters[i % 24]}{letters[i // 24]}" for i in range(60)]
    short = [{"speaker": "A", "text": " ".join(words[:20])}, {"speaker": "B", "text": "oui"}]
    long_ = [{"speaker": "A", "text": " ".join(words[:20])}, {"speaker": "B", "text": " ".join(words[20:60])}]
    recs = [{"doc_id": "ws:1", "turns": short}, {"doc_id": "gut:2", "turns": long_}]
    kept, _ = dedupe_across_documents(recs)
    assert len(kept) == 2                                  # premier passage : la scène longue n'est couverte qu'en partie
    kept, dropped = theatre.dedupe_both_ways(kept)
    assert [r["doc_id"] for r in kept] == ["gut:2"] and dropped == {("gut:2", "ws:1"): 1}
    # Deux copies identiques dans deux documents : une seule est retirée (la plus tardive).
    kept, _ = theatre.dedupe_both_ways([{"doc_id": "a", "turns": long_}, {"doc_id": "b", "turns": long_}])
    assert [r["doc_id"] for r in kept] == ["a"]


def test_single_exclusion_list_in_json_file(tmp_path):
    # Une seule liste : le fichier JSON (les anciens ajouts du code y ont été déplacés, Lysistrata ajoutée).
    path = tmp_path / "exclusions.json"
    path.write_text('{"documents": {"gutenberg_fr:1": "pas_du_dialogue"}}', encoding="utf-8")
    assert theatre.load_exclusions(path) == {"gutenberg_fr:1": "pas_du_dialogue"}
    real = json.loads(theatre.EXCLUSIONS.read_text(encoding="utf-8"))
    excl = theatre.load_exclusions()
    assert excl == real["documents"] and set(excl.values()) <= set(real["raisons"])
    for doc_id in ("wikisource_fr:livre:Aristophane - Lysistrata (trad. Raoul Vèze), 1928.djvu",
                   "wikisource_fr:livre:Jarry - Les Silènes (éd. Bibliophiles créoles), 1926.djvu",
                   "wikisource_fr:livre:Tableaux des mœurs du temps, suivi de l’Histoire de Zaïrette, 1911.djvu"):
        assert excl[doc_id] == "erotique_ou_obscene"
    assert not hasattr(theatre, "EXCLUSIONS_EXTRA")


def test_names_made_of_roman_letters_are_names():
    # « MIMI. » (La Vie de bohème), « LILI. » (Vitrac) étaient pris pour des chiffres romains : leurs
    # répliques se retrouvaient collées dans celle de RODOLPHE.
    assert classify("MIMI.")[0] == "speaker" and classify("LILI.")[0] == "speaker"
    assert classify("IV.")[0] == "heading" and classify("XII")[0] == "heading"     # numéros de scène
    (scene,) = parse_scenes("RODOLPHE.\nTu souffres ?\n\nMIMI.\nNon.\n\nRODOLPHE.\nMa pauvre Mimi !\n", strict=True)
    assert [t["speaker"] for t in scene] == ["RODOLPHE", "MIMI", "RODOLPHE"]


def test_name_dot_reply_recognized_if_already_a_speaker():
    # Plaute (Sommer), George Sand : « SOSIE. Mais quelle idée ! » sur une seule ligne. Reconnu si SOSIE
    # est déjà 3 fois locuteur ; pas dans la validation figée (cleanup=False).
    text = "\n".join(["SOSIE.", "Moi ?", "MERCURE.", "Toi.", "SOSIE.", "Non.", "MERCURE.", "Si.", "SOSIE.", "Bon.",
                      "MERCURE.", "Va-t'en.", "SOSIE. Mais quelle idée !", "MERCURE.", "Pars."])
    (scene,) = parse_scenes(text, strict=True)
    assert scene[-2] == {"speaker": "SOSIE", "text": "Mais quelle idée !"}
    (frozen,) = parse_scenes(text, strict=False, cleanup=False)
    assert frozen[-1]["speaker"] == "MERCURE" and "\nSOSIE. Mais quelle idée !\n" in frozen[-1]["text"]


def test_scene_numbers_notes_and_variants_close_the_block():
    # Shakespeare (Hugo) : « XVII. » seul sur sa ligne est un numéro de scène, pas la suite de la réplique.
    assert classify("XVII.")[0] == "heading" and classify("RECUEILS : 1652 in-12.")[0] == "heading"
    text = ("YORK.\nCinq contre vingt !\nWARWICK.\nEn avant.\n\nXVII.\n\nLE ROI.\nQui vive ?\nYORK.\nNous.\n"
            "\n[Note 24: Call hither to the stake.\nVoir l'édition de Johnson.\nLE ROI.\nBien.\n")
    scenes = parse_scenes(text, strict=True)
    assert [[t["speaker"] for t in s] for s in scenes] == [["YORK", "WARWICK"], ["LE ROI", "YORK"]]
    assert "Note" not in turns_text(scenes) and "XVII" not in turns_text(scenes)


def test_stage_position_numbers_after_name():
    # Feydeau : « LE GÉNÉRAL (3), devant la table. » n'était pas reconnu et se collait à la réplique précédente.
    text = "PETYPON.\nVoilà le général !\nLE GÉNÉRAL (3), devant la table.\nBonjour.\nPETYPON (2).\nMon général !\n"
    (scene,) = parse_scenes(text, strict=True)
    assert scene == [{"speaker": "PETYPON", "text": "Voilà le général !"},
                     {"speaker": "LE GÉNÉRAL", "text": "Bonjour."}, {"speaker": "PETYPON", "text": "Mon général !"}]


def test_scene_without_speech_marks_removed():
    # Distribution (« BLOUNT. Dailly. ») ou sommaire de revue : aucune question, exclamation ni « je/vous ».
    theatre._init_worker(_ref("texte sans rapport"))
    cast = [{"speaker": s, "text": a} for s, a in (("BLOUNT", "Dailly."), ("JOLLIVET", "Joumard."),
                                                   ("LE GRAND-DUC", "Bouyer."), ("OGAREFF", "Paulin Ménier."))]
    assert theatre.scene_reason(cast) == "scenes_retirees_sans_parole"
    talk = [{"speaker": "A" * (i % 2 + 2), "text": t} for i, t in
            enumerate(("Le roi est mort.", "Qui l'a tué ?", "Personne.", "Je ne te crois pas."))]
    assert theatre.scene_reason(talk) is None


# ---------------------------------------------------------------------------------------------
# Troisième tour de corrections (12/11, soir)
# ---------------------------------------------------------------------------------------------

def test_echo_kept_in_moliere_and_never_leaves_only_punctuation():
    # La Princesse d'Élide (kit) : MORON répète « Comment? » puis continue. Écho voulu par Molière : gardé.
    text = "LE PRINCE.\nComment?\nMORON.\nComment? C'est le plus orgueilleux petit vilain.\nLE PRINCE.\nSoit.\n"
    (scene,) = parse_scenes(text, strict=False)
    assert scene[1]["text"] == "Comment? C'est le plus orgueilleux petit vilain."
    # Mode strict (corpus) : jamais une réplique réduite à de la ponctuation (« Ha la ba… » → « . »).
    text = ("LE MUPHTI.\nHa la ba, ba la chou, ba la da.\nLES TURCS.\nHa la ba, ba la chou, ba la da.\n.\n"
            "LE MUPHTI.\nIoc !\n")
    (scene,) = parse_scenes(text, strict=True)
    assert scene[1]["text"] == "Ha la ba, ba la chou, ba la da.\n."
    assert all(any(c.isalpha() for c in t["text"]) for t in scene)


def test_strict_mode_refuses_novel_narration():
    # Ségur, romans dialogués : la narration après une ligne vide n'est plus collée au personnage précédent.
    text = ("LA MARCHANDE.\n\nC'est trente-deux francs.\n\n— Trente-deux francs ! s'écria Thérèse effrayée.\n\n"
            "THÉRÈSE.\n\nC'est trop cher.\n\nLA MARCHANDE.\n\nTant pis pour vous.\n\n"
            "En entendant ces petites voix chéries, la pauvre mère\nse leva vivement et courut à la porte.\n\n"
            "HÉLÈNE.\n\nMes enfants !\n\nTHÉRÈSE.\n\nMaman !\n")
    scenes = parse_scenes(text, strict=True)
    body = turns_text(scenes)
    assert "s'écria" not in body and "se leva" not in body          # incise sur la 2e ligne aussi
    # La scène est fermée sur le récit (la réplique isolée de LA MARCHANDE est perdue : c'est voulu).
    assert [[t["speaker"] for t in s] for s in scenes] == [["THÉRÈSE", "LA MARCHANDE"], ["HÉLÈNE", "THÉRÈSE"]]
    assert scenes[0][1]["text"] == "Tant pis pour vous."
    # Un long paragraphe de prose (> 300 caractères) sans incise est aussi refusé ; des vers courts restent.
    long_prose = "Le soir tombait sur la petite ville et les boutiques fermaient l'une après l'autre. " * 5
    scenes = parse_scenes(f"A.\n\nBonjour.\n\n{long_prose}\n\nB.\n\nBonsoir.\n\nA.\n\nAdieu.\n", strict=True)
    assert "boutiques" not in turns_text(scenes)
    assert theatre.narrative_paragraph(["Bonjour, Anaïk,"]) is False
    assert theatre.narrative_paragraph(["Tu ne dis rien ? demanda-t-il."]) is True


def test_cast_list_not_closed_by_a_name_seen_once():
    # Psyché : ZÉPHYRE (rôle muet, cité une fois) fermait la liste des personnages ; « Grâces. » devenait
    # une réplique de PHAÈNE.
    text = ("PERSONNAGES\nJUPITER\nZÉPHYRE\nPHAÈNE\nGrâces.\nCIDIPPE\nsœurs de Psyché.\nACTE I\n"
            "JUPITER\nBonjour, ma fille.\nPHAÈNE\nBonjour, mon père !\nJUPITER\nAdieu.\nPHAÈNE\nDéjà ?\n")
    for strict in (True, False):
        (scene,) = parse_scenes(text, strict=strict)
        assert [t["speaker"] for t in scene] == ["JUPITER", "PHAÈNE", "JUPITER", "PHAÈNE"]
        assert "Grâces" not in turns_text([scene]) and "sœurs" not in turns_text([scene])
    # « Prologue. » seul sur sa ligne ferme aussi la liste (sinon elle avalerait le prologue chanté).
    assert classify("Prologue.")[0] == "heading" and classify("Prologue de la pièce, en vers.")[0] == "text"


def test_merimee_stage_directions_and_cast():
    # Clara Gazul : « NOM. » puis « didascalie en minuscule. Réplique » sur la ligne suivante.
    text = ("UN DOMESTIQUE.\nentrant. Une dame demande à parler à monsieur.\nLE RÉSIDENT.\n"
            "à part. Peste ! c'est sans doute la femme d'un général.\nUN DOMESTIQUE.\nOui, monsieur.\n")
    (scene,) = parse_scenes(text, strict=True)
    assert [t["text"] for t in scene] == ["Une dame demande à parler à monsieur.",
                                          "Peste ! c'est sans doute la femme d'un général.", "Oui, monsieur."]
    (frozen,) = parse_scenes(text, strict=False, cleanup=False)      # validation figée : règles d'origine
    assert frozen[0]["text"].startswith("entrant.")
    # « PREMIÈRE JOURNÉE. » est un titre ; « PREMIÈRE. Un salon. » (SCÈNE manquant) aussi, en mode strict.
    assert classify("PREMIÈRE JOURNÉE.")[0] == "heading" and classify("COMÉDIE EN TROIS JOURNÉES")[0] == "heading"
    lines = []
    for k in range(3):
        lines += ["PREMIÈRE.", "Un salon.", "INÈS.", f"Que veux-tu, Esteban ? ({k})", "DON ESTEBAN.", "Rien !", ""]
    lines += ["PREMIÈRE. Une auberge.", "INÈS.", "Où suis-je ?", "DON ESTEBAN.", "À Elvas !"]
    scenes = parse_scenes("\n".join(lines), strict=True)
    assert "Une auberge" not in turns_text(scenes)
    assert scenes[-1] == [{"speaker": "INÈS", "text": "Où suis-je ?"}, {"speaker": "DON ESTEBAN", "text": "À Elvas !"}]
    # Distribution sans ligne PERSONNAGES : deux « répliques » sans aucune marque de parole, retirées.
    theatre._init_worker(_ref("texte sans rapport"))
    cast = [{"speaker": "JUAN MENDO", "text": "Don CÉSAR BELMONTE, cavalier portugais.\nLe corrégidor DE BADAJOZ."},
            {"speaker": "PÉDRO", "text": "Un Aubergiste portugais.\nDoña INÈS DE MENDOZA."}]
    assert theatre.scene_reason(cast) == "scenes_retirees_sans_parole"


def test_dedupe_sees_editions_with_old_spelling_and_verse_numbers():
    modern = ("Il trouvait que les yeux de sa maîtresse et le cœur de son ami avaient dit la même chose ce soir-là "
              "pendant que la foule passait sous les fenêtres du palais et que la nuit tombait doucement")
    old = modern.replace("trouvait", "trouvoit").replace("avaient", "avoient").replace(" et ", " & ")
    old = old.replace("yeux", "ïeux").replace("foule passait", "foule 1035 passoit")
    a = [{"speaker": "A", "text": modern}, {"speaker": "B", "text": "Vraiment ?"}]
    b = [{"speaker": "A", "text": old}, {"speaker": "B", "text": "Vraiment ?"}]
    assert shared(ngram_hashes(old), ngram_hashes(modern)) == 0            # sans repli : invisible
    kept, dropped = dedupe_across_documents([{"doc_id": "ws:1", "turns": a}, {"doc_id": "gut:2", "turns": b}])
    assert [r["doc_id"] for r in kept] == ["ws:1"] and dropped == {("ws:1", "gut:2"): 1}
    # Le repli ne sert qu'au dédoublonnage : les contrôles contre la validation restent sur le texte exact.
    assert theatre.fold_spelling("Je voyois & 1035").split() == theatre.fold_spelling("je voyais et").split()
