"""Nettoyage du théâtre d'affinage (tools/nettoyer_affinage.py) : on retire, on ne réécrit pas."""
import collections

import pytest

from tools import nettoyer_affinage as N


def propre(texte, crochets_scene=False):
    return N.nettoyer_replique(texte, collections.Counter(), crochets_scene)


@pytest.mark.parametrize("avant, apres", [
    ("(A George Dandin.) Allons, mettez-vous à genoux.", "Allons, mettez-vous à genoux."),
    ("Prenez qui vous voudrez. (Seul.) Voyez un peu!", "Prenez qui vous voudrez. Voyez un peu!"),
    ("Oui. (bas, à Cléante) Tais-toi.", "Oui. Tais-toi."),
    ("(Au médecin). Il n'était rien arrivé…", "Il n'était rien arrivé…"),
    ("Bien (lisant) « Cher ami »", "Bien « Cher ami »"),
    ("Oui (à M. Jourdain) monsieur.", "Oui monsieur."),
    ("Voilà. Ils sortent tous.", "Voilà."),
    ("tenez et prenez… (Il donne.) … ce sont les petits présents", "tenez et prenez… ce sont les petits présents"),
    ("rien de pareil à mon aise[1601-a].", "rien de pareil à mon aise."),
    ("Allons donc!... [Il s'arrête de nouveau.] Du diable", "Allons donc!... Du diable"),
    ("Ça les calmera. Ils sortent. <br/ >", "Ça les calmera."),
])
def test_les_indications_de_scene_sont_retirees(avant, apres):
    assert propre(avant) == apres


@pytest.mark.parametrize("dit", [
    "Je porte l'as de trèfle (admire mon malheur!)",
    "Donc avec Truffaldin (car je sors de chez lui) j'ai",
    "L'autre jour (pourroit-on se le persuader?) il",
    "Lélie (et l'action lui sera salutaire) va",
    "le respect que tu me conserves ? (Allons donc.) Tu as l'insolence",
    "Encore! Il va tout découvrir.",
    "Arrêté ? Il sort d'ici.",
    "Il s'en va maintenant que ma soeur n'a plus de dot.",
])
def test_ce_que_le_personnage_dit_est_garde(dit):
    assert propre(dit) == dit


def test_les_crochets_d_editeur_gardent_le_texte_dit():
    assert propre("un ignorant, [un] ignorantissime") == "un ignorant, un ignorantissime"


def test_les_crochets_sont_des_indications_dans_les_oeuvres_qui_les_emploient_ainsi():
    texte = "Fais-lui ce sacrifice. [Elle court à lui.] Jeanne, tu ne réponds pas?"
    assert propre(texte, crochets_scene=True) == "Fais-lui ce sacrifice. Jeanne, tu ne réponds pas?"


def test_une_scene_garde_ses_repliques_dites_et_perd_les_chantees():
    d = {"doc_id": "x", "turns": [{"speaker": "ARGAN", "text": "Bonjour, monsieur. (Il salue.)"},
                                  {"speaker": "UN MUSICIEN", "text": "La la la."},
                                  {"speaker": "TOINETTE", "text": "Votre servante."},
                                  {"speaker": "ARGAN", "text": "Que dites-vous ?"},
                                  {"speaker": "TOINETTE", "text": "Rien."}]}
    scene, raison, compte, _ = N.nettoyer_scene(d, 0.16, 4, False)
    assert raison is None and [t["speaker"] for t in scene["turns"]] == ["ARGAN", "TOINETTE", "ARGAN", "TOINETTE"]
    assert scene["turns"][0]["text"] == "Bonjour, monsieur." and compte["repliques_chantees_retirees"] == 1


def test_la_validation_ne_perd_aucune_scene():
    d = {"doc_id": "x", "turns": [{"speaker": "A", "text": "Oui. (Il sort.)"}]}
    scene, raison, _, _ = N.nettoyer_scene(d, 0.16, 4, False, validation=True)
    assert raison is None and scene["turns"][0]["text"] == "Oui."
