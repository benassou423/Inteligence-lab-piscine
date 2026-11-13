"""Tests rapides de chatbot/dialogues.py : petits extraits écrits à la main, aucun fichier du corpus."""

import numpy as np

from chatbot import dialogues as dl
from pipeline.io import read_docs, write_docs

EMPTY_REF = {"exclusions": {}, "val_moliere": np.empty(0, dtype=np.uint64)}


def roman(n_pairs=12, prefix=""):
    """Un faux roman : un paragraphe de récit, puis n_pairs questions/réponses au tiret."""
    paras = ["Le soir tombait sur la ville quand ils se retrouvèrent devant la gare."]
    for i in range(n_pairs):
        paras.append(f"— {prefix}Tu viens demain à la maison numéro {i} ? demanda Paul.")
        paras.append(f"— Oui, je viendrai vers midi avec mon frère {prefix}{i}.")
    return "\n\n".join(paras)


def test_les_trois_tirets_ouvrent_un_tour():
    text = "--Bonjour, mon ami.\n\n— Bonjour à vous.\n\n– Comment allez-vous ce matin?"
    exchanges, _ = dl.extract_exchanges(text)
    assert exchanges == [["Bonjour, mon ami.", "Bonjour à vous.", "Comment allez-vous ce matin?"]]


def test_incises_retirees():
    cases = {
        "— Tenez, dit-elle, voilà tout ce qui reste.": "Tenez, voilà tout ce qui reste.",
        "--Où est ta mère? demanda le laquais sans se déranger.": "Où est ta mère?",
        "--Bigre! fit-il en souriant, mes compliments.": "Bigre! Mes compliments.",
        "— Vous le connaissez ? s’écria Suzanne en saisissant la main de M. de Pomereux.": "Vous le connaissez?",
        "— Je ne sais pas, répondit-il d'une voix sourde.": "Je ne sais pas.",
        # récit à l'imparfait collé à l'incise : avalé avec elle ; une parole au futur reste
        "— Où allons-nous? demanda Henri, au moment où ils passaient sous l'allée.": "Où allons-nous?",
        "— Oui, dit-il, il viendra demain.": "Oui, il viendra demain.",
        # dialogue entre guillemets (Mérimée) : les guillemets ne gênent plus l'incise
        "— « Il m'a tué d'ennui, » dit le colonel. « Figurez-vous cela. »": "Il m'a tué d'ennui. Figurez-vous cela.",
        # sujets moins simples, tirets autour de l'incise
        "— Rien n'a été fait? demanda d’Harmental.": "Rien n'a été fait?",
        "— Oui, murmura sir John, nous sommes trop près.": "Oui, nous sommes trop près.",
        "— Nous quitter!… — s’écria Clémence.": "Nous quitter!…",
        "— C’est une maison — dit-elle — très ancienne.": "C'est une maison très ancienne.",
        "--Je viens seulement—expliqua Coryse—pour vous dire bonjour.": "Je viens seulement pour vous dire bonjour.",
        "— Et le vôtre aussi! réplique le pilote.": "Et le vôtre aussi!",
    }
    for paragraph, expected in cases.items():
        text, _, reason = dl.turn_from_paragraph(paragraph)
        assert (text, reason) == (expected, None), paragraph


def test_imperatif_n_est_pas_une_incise():
    text, _, reason = dl.turn_from_paragraph("— Écoute, demande la permission à ton père.")
    assert text == "Écoute, demande la permission à ton père." and reason is None


def test_recit_ferme_l_echange():
    text = ("— Tu pars déjà ?\n\n— Oui, il le faut.\n\n"
            "— Adieu donc, dit-il. Puis il se leva et sortit sans un regard.\n\n"
            "— Attends-moi !\n\n— Je t'attendrai.")
    exchanges, cuts = dl.extract_exchanges(text)
    # Le tour qui garde du récit au passé simple n'entre pas : il coupe l'échange en deux.
    assert exchanges == [["Tu pars déjà?", "Oui, il le faut."], ["Attends-moi!", "Je t'attendrai."]]
    assert cuts["passe_simple"] == 1
    # pluriel en -èrent, référence de catalogue : refusés aussi
    assert dl.turn_from_paragraph("— Son père! s’écrièrent, en même temps, Jean et Maurice.")[2] == "passe_simple"
    assert dl.turn_from_paragraph("— Pfeifferi Bernardi, p. 560.")[2] == "reference_biblio"


def test_paragraphe_de_recit_ferme_l_echange_et_trop_long_aussi():
    text = ("— Un.\n\n— Deux.\n\nIls marchèrent longtemps.\n\n— Trois.\n\n— Mot " + "long " * 100 +"\n\n— Quatre.")
    exchanges, cuts = dl.extract_exchanges(text)
    assert exchanges == [["Un.", "Deux."]]          # « Trois. » reste seul : pas un échange
    assert cuts["trop_long"] == 1


def test_alternance_a_b_et_meme_locuteur_nomme_deux_fois():
    exchanges, _ = dl.extract_exchanges("— Viens, dit Paul.\n\n— Non, répondit Marie.\n\n— Si, dit Paul.")
    assert exchanges == [["Viens.", "Non.", "Si."]]
    assert [t["speaker"] for t in dl.with_speakers(exchanges[0])] == ["A", "B", "A"]
    # Paul parle deux fois de suite : l'alternance serait fausse, on coupe.
    exchanges, cuts = dl.extract_exchanges("— Viens, dit Paul.\n\n— Oui.\n\n— Allons, reprit Marie.\n\n"
                                           "— Vite, ajouta Marie.\n\n— Bien.")
    # Marie au tour 2 (place de Paul) puis Marie encore au tour 3 : deux coupures ; « Allons. »
    # reste seul entre les deux et disparaît.
    assert cuts["alternance_fausse"] == 2
    assert exchanges == [["Viens.", "Oui."], ["Vite.", "Bien."]]


def test_decoupe_a_40_tours():
    pieces = dl.split_long([f"t{i}" for i in range(95)])
    assert [len(p) for p in pieces] == [32, 32, 31]
    assert sum(pieces, []) == [f"t{i}" for i in range(95)]
    text = "\n\n".join(f"— Réplique {i}." for i in range(81))
    exchanges, _ = dl.extract_exchanges(text)
    assert all(2 <= len(e) <= 40 for e in exchanges) and sum(len(e) for e in exchanges) == 81


def test_document_en_orthographe_ancienne_ecarte():
    old = {"id": "g:1", "text": roman() + "\n\nIl étoit tard, et l'on avoit faim. Elle disoit qu'il pouvoit venir."}
    assert dl.archaic_score(old["text"]) > dl.ARCHAIC_MAX
    assert dl.doc_reason(old, EMPTY_REF) == "orthographe_ancienne"
    assert dl.doc_reason({"id": "g:2", "text": roman() + "\n\nIl doit partir, on le voit bien."},
                         EMPTY_REF) is None
    assert dl.doc_reason({"id": "g:3", "text": "Cette œuvre est trop ancienne & ſans eſpoir."}, EMPTY_REF)


def test_liste_d_exclusion_mais_segur_gardee():
    ref = {"exclusions": {"a": "erotique_ou_obscene", "b": "pas_du_dialogue"}, "val_moliere": EMPTY_REF["val_moliere"]}
    assert dl.doc_reason({"id": "a", "text": roman()}, ref) == "liste_erotique_ou_obscene"
    assert dl.doc_reason({"id": "b", "text": roman()}, ref) is None


def test_validation_par_document_entier(tmp_path):
    ids = [f"gutenberg_fr:{i}" for i in range(40)]
    assert {dl.is_val(i, 5000) for i in ids} == {True, False}     # les deux côtés sont représentés
    assert [dl.is_val(i) for i in ids] == [dl.is_val(i) for i in ids]   # stable
    docs = [{"id": i, "source": "gutenberg_fr", "url": None, "license": None, "text": roman(prefix=f"{n} ")}
            for n, i in enumerate(ids)]
    corpus = tmp_path / "part-00000.jsonl.zst"
    write_docs(corpus, docs)
    stats = dl.build([str(corpus)], tmp_path / "out", theatre_path=None, workers=1, ref=EMPTY_REF,
                     val_per_10000=5000, min_doc_turns=2, log=lambda s: None)
    train = list(read_docs(tmp_path / "out" / "romans_train.jsonl.zst"))
    val = list(read_docs(tmp_path / "out" / "romans_val.jsonl.zst"))
    assert train and val
    assert not ({e["doc_id"] for e in train} & {e["doc_id"] for e in val})
    assert all(dl.is_val(e["doc_id"], 5000) for e in val)
    ex = train[0]
    assert set(ex) == {"id", "source", "doc_id", "url", "license", "style", "turns"}
    assert ex["style"] == "moderne" and [t["speaker"] for t in ex["turns"]][:2] == ["A", "B"]
    assert all("\n" not in t["text"] for e in train for t in e["turns"])
    assert stats["documents_romans"]["retenus"] == 40


def test_doublon_d_edition_retire():
    d = dl.Deduper()
    turns = dl.with_speakers(["Je vous assure que je n'ai jamais vu cet homme de ma vie, monsieur.", "Bien."])
    assert not d.seen(turns)
    other_edition = dl.with_speakers(["Je vous assure que je n’ai jamais vu cet homme de ma vie, Monsieur !",
                                      "Bien.", "Alors partons."])
    assert d.seen(other_edition)


def test_scene_de_theatre_en_prose_convertie():
    scene = {"turns": [
        {"speaker": "HENRI", "text": "Bonjour, madame. (Il salue.)\nVous allez bien ce matin?"},
        {"speaker": "MADAME", "text": "Très bien, merci."},
        {"speaker": "MADAME", "text": "Et vous?"},
        {"speaker": "LE NOTAIRE", "text": "Pardon, je vous dérange."},
    ]}
    assert dl.scene_to_tirets(scene) == [["Bonjour, madame. Vous allez bien ce matin?",
                                          "Très bien, merci. Et vous?", "Pardon, je vous dérange."]]
    prose = [{"turns": [{"speaker": "A", "text": "x" * 120}, {"speaker": "B", "text": "Oui."}]}]
    assert not dl.is_verse_play(prose)


def test_scene_en_vers_refusee():
    verse = {"turns": [
        {"speaker": "THÉRÈSE", "text": "Monsieur Octave, elle écrit, j'imagine,\nPour sa société de secours."},
        {"speaker": "OCTAVE", "text": "En effet.\nOh! c'est son grand souci!"},
        {"speaker": "THÉRÈSE", "text": "C'est de moi qu'elle a fait\nSon secrétaire, et je lui dois bien ça."},
    ]}
    assert dl.is_verse_turn(verse["turns"][0]["text"])
    assert dl.is_verse_play([verse])                 # aucune ligne longue : pièce en vers
    assert dl.scene_to_tirets(verse) == []           # les vers coupent : il ne reste rien


# --- Corrections après la revue indépendante du 13/11 ------------------------------------------

def test_oeuvres_erotiques_ecartees_par_auteur_titre_ou_euphemismes():
    assert dl.doc_reason({"id": "wikisource_fr:livre:Le Nismois - L’Hermaphrodite, tome 1, 1902.djvu",
                          "text": roman()}, EMPTY_REF) == "erotique_auteur_ou_titre"
    assert dl.doc_reason({"id": "gutenberg_fr:1", "text": roman(),
                          "meta": {"title": "Débauchées précoces", "authors": "Fuckwell"}},
                         EMPTY_REF) == "erotique_auteur_ou_titre"
    assert dl.doc_reason({"id": "gutenberg_fr:2", "text": roman(),
                          "meta": {"title": "X", "subjects": "Erotic literature"}}, EMPTY_REF) == "erotique_auteur_ou_titre"
    # Sans titre connu : la densité d'euphémismes suffit (aucun mot de la liste « crue »).
    euph = roman() + "\n\n" + "Elle tendit sa croupe et ses fesses, il caressa ses cuisses et ses tétons. " * 20
    assert dl.erotic_score(euph) > dl.EROTIC_MAX
    assert dl.doc_reason({"id": "g:3", "text": euph}, EMPTY_REF) == "euphemismes_erotiques"
    # Faux amis écartés de la liste : chats, culs-de-jatte, « Géorgie » (≠ « orgie »).
    chat = roman() + "\n\n" + "La chatte et sa minette dorment ; le cul-de-jatte roule. " * 20
    assert dl.doc_reason({"id": "g:4", "text": chat, "meta": {"title": "Voyage en Géorgie"}}, EMPTY_REF) is None


def test_dictionnaires_ecartes():
    assert dl.doc_reason({"id": "gutenberg_fr:48852", "text": roman(),
                          "meta": {"title": "Le parler populaire des Canadiens français"}}, EMPTY_REF) == "dictionnaire"
    assert dl.doc_reason({"id": "wikisource_fr:livre:Larousse - Grand dictionnaire universel du XIXe siècle - Tome 1.djvu",
                          "text": roman()}, EMPTY_REF) == "dictionnaire"


def test_incises_premiere_personne_et_recit_supplementaire():
    assert dl.turn_from_paragraph("— Ah! fît-elle avec un sourire, vous commencez.")[0::2] == ("Ah! Vous commencez.", None)
    assert dl.turn_from_paragraph("— Non, repartis-je en riant.")[0::2] == ("Non.", None)
    assert dl.turn_from_paragraph("— Viens donc! Frédéric arriva; il avait un œil enflé.")[2] == "passe_simple"
    assert dl.turn_from_paragraph("— Oui. Le savant se trouva mal.")[2] == "passe_simple"
    # narration au passé simple entre tirets : retirée, la parole reste
    assert dl.turn_from_paragraph("— Je viens — et il se leva — pour vous dire adieu.")[0::2] == \
        ("Je viens pour vous dire adieu.", None)
    # futur, présent, question : ce n'est pas du récit
    assert dl.turn_from_paragraph("— Il va se lever, demain il se lèvera tôt.")[2] is None


def test_tours_non_francais_et_tiret_restant():
    assert dl.turn_from_paragraph("— Παπαῖ!")[2] == "sans_lettre"
    assert dl.turn_from_paragraph("— Cornelius awaited the visit of the doctor, and he was not alone.")[2] == "anglais"
    assert dl.turn_from_paragraph("— -Il faut y compter, au contraire.")[0::2] == ("Il faut y compter, au contraire.", None)


def test_didascalies_collees_retirees_et_personnage_colle_separe():
    scene = {"turns": [
        {"speaker": "CASSAGNE", "text": "Je suis bien heureux de vous voir !"},
        {"speaker": "LÉONTINE", "text": "Ah ! ça, qu’est-ce que ça signifie ?\nElle est à droite un peu au-dessous de la table."},
        {"speaker": "L'ABBÉ", "text": "Merci. Jean le remercie d’un sourire. L’abbé regarde la porte. JEAN .—J’ai trop longtemps cru à cela."},
        {"speaker": "AGATHE", "text": ": Vous avez bien fait. À part. Ça tombe bien."},
    ]}
    assert dl.scene_to_tirets(scene) == [["Je suis bien heureux de vous voir!", "Ah! ça, qu'est-ce que ça signifie?",
                                          "Merci.", "J'ai trop longtemps cru à cela.",
                                          "Vous avez bien fait. Ça tombe bien."]]
    # Premier tour qui décrit le décor : retiré ; numéro de chapitre « XXII. — » : pas un personnage.
    decor = {"turns": [{"speaker": "X", "text": "Une salle d’audience. Au lever du rideau, le tribunal siège."},
                       {"speaker": "Y", "text": "Silence."}, {"speaker": "Z", "text": "Bien."}]}
    assert dl.scene_to_tirets(decor) == [["Silence.", "Bien."]]
    assert dl.split_glued_speakers("A", "XXII. — Ne faut-il pas?") == [("A", "XXII. — Ne faut-il pas?")]


def test_vers_recolles_couplets_et_pieces_en_partie_en_vers():
    assert dl.is_verse_flat("Dans l'antre de Makrâ, cet enfant vit le jour Et des bras maternels disparut sans retour Que")
    assert dl.is_verse_flat("AIR : Amour, amour, tu nous perdras !")
    assert not dl.is_verse_flat("Je vous dis que Paul et Marie viendront demain, et Jean aussi.")
    prose = "Voilà une réplique en prose assez longue pour dépasser largement les quatre-vingt-dix caractères."
    couplet = "Amour, amour, tu nous perdras\nQuand tu nous tiens, adieu prudence"
    scenes = [{"turns": [{"speaker": "A", "text": prose + "\nEt une deuxième ligne."},
                         {"speaker": "B", "text": couplet}, {"speaker": "A", "text": couplet}]}]
    share, n = dl.verse_turn_share(scenes)
    assert n == 2 and share > dl.VERSE_PLAY_SHARE


def test_fuites_vers_la_validation(tmp_path):
    # Échange court identique (autre édition) des deux côtés : retiré du train.
    val_whole = {dl.fingerprint("Oui.\nNon.")}
    assert dl.leaks(dl.with_speakers(["Oui.", "Non."]), set(), val_whole)
    assert not dl.leaks(dl.with_speakers(["Oui.", "Peut-être."]), set(), val_whole)
    # Un document de la validation des romans n'alimente pas le théâtre moderne.
    line = "Voilà une réplique en prose assez longue pour dépasser largement les quatre-vingt-dix caractères."
    scenes = [{"id": f"doc:{k}", "doc_id": "doc", "play": "P", "source": "wikisource_fr",
               "turns": [{"speaker": "A", "text": f"{line} {k}"}, {"speaker": "B", "text": f"Réponse {k}."}]}
              for k in range(3)]
    path = tmp_path / "theatre.jsonl.zst"
    write_docs(path, scenes)
    out, rejected, *_ = dl.theatre_moderne(path, EMPTY_REF, dl.Deduper(), val_docs={"doc"})
    assert out == [] and rejected["document_en_validation"] == 1
    out, *_ = dl.theatre_moderne(path, EMPTY_REF, dl.Deduper())
    assert len(out) == 3


def test_didascalies_en_minuscule_et_entrees_retirees():
    scene = {"turns": [
        {"speaker": "GNAFRON", "text": "Et t'as bien fait! ils sortent."},
        {"speaker": "GUIGNOL", "text": "Tiens renifle une miette, il se sauve. Rentre Titus Andronicus, accompagné."},
        {"speaker": "GNAFRON", "text": "Mon Dieu! il tombe!"},
    ]}
    # « il se sauve » n'est pas un verbe de jeu de la liste : la phrase reste (dans le doute, on garde
    # la parole ici ; la liste est courte exprès).
    assert dl.scene_to_tirets(scene) == [["Et t'as bien fait!", "Tiens renifle une miette, il se sauve.",
                                          "Mon Dieu! il tombe!"]]


def test_incises_supplementaires():
    assert dl.turn_from_paragraph("— Vite, dit de Melval, accélérant la descente.")[0] == "Vite."
    assert dl.turn_from_paragraph("— Vive la glace! clama le droguiste.")[0] == "Vive la glace!"
    assert dl.turn_from_paragraph("— Hi, hi, hi… geignit Pompon.")[0] == "Hi, hi, hi…"


def test_didascalies_longues_et_sujets_composes():
    long_dir = "(" + "Bondissant, il ouvre la porte et traverse l'antichambre. " * 5 + ")"
    scene = {"turns": [
        {"speaker": "A", "text": f"Hein?… {long_dir} Personne!… Qu'est-ce que ça veut dire?"},
        {"speaker": "B", "text": "Mais asseyez-vous d'abord. Godefroy et Césarine s'asseyent. Charolais père entre."},
        {"speaker": "A", "text": "Reste-là! Je vais couper le fil. Il le coupe. Boursin va pour sortir."},
    ]}
    assert dl.scene_to_tirets(scene) == [["Hein?… Personne!… Qu'est-ce que ça veut dire?",
                                          "Mais asseyez-vous d'abord.", "Reste-là! Je vais couper le fil."]]
