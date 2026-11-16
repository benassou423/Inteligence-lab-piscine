"""Écrit les recettes de l'essai « données nettoyées » à partir de chatbot/recettes_rtx.json (non modifié).

v4a : v1 + affinage Molière sur le théâtre nettoyé (base v1 gardée)             -> candidat D
v4  : v4a + base pré-entraînée sur les tokens nettoyés (dossier de sortie à part) -> candidat F
v4e : v1 + base pré-entraînée sur les tokens nettoyés (affinage v1)               -> E, BPB seulement
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
v1 = json.loads((ROOT / "chatbot/recettes_rtx.json").read_text(encoding="utf-8"))
PROPRE = "data/stages/chatbot_20261114/theatre_propre/"
PRE = "data/stages/chatbot_20261114/pretrain_propre"


def affinage(r):
    d = r["chatbots"]["moliere"]["etape1"]["donnees"]
    d["valeur"] = [PROPRE + "theatre_train.jsonl.zst", PROPRE + "moliere_train.jsonl.zst"]
    m = r["chatbots"]["moliere"]
    m["validation"]["valeur"] = PROPRE + "moliere_val.jsonl.zst"
    m["validation"]["pourquoi"] = ("les 3 mêmes pièces de Molière jamais vues, nettoyées comme l'entraînement (indications de scène retirées, "
                                   "aucune scène écartée) : la mesure est faite dans la forme apprise ; le BPB sur le fichier d'origine est "
                                   "donné à part")
    m["synthetique"]["avec"]["valeur"] = [PROPRE + "moliere_train.jsonl.zst"]
    d["pourquoi"] = ("théâtre FRANÇAIS d'origine seulement (traductions, livrets chantés, textes qui ne sont pas des pièces écartés : "
                     "chatbot/exclusions_theatre_v2.json), indications de scène retirées des répliques (tools/nettoyer_affinage.py) ; "
                     "puis Molière seul (hors validation), nettoyé de la même façon")


def base(r):
    r["base"]["donnees"]["valeur"] = PRE
    r["base"]["donnees"]["pourquoi"] = ("mêmes tokens que la v1, FineWeb trié (spam, charabia et listes écartés par tools/nettoyer_pretrain.py, "
                                        "règles dans chatbot/regles_pretrain_v4.json) ; les autres sources sont inchangées")


def ecrire(nom, version, quoi, *modifs):
    r = copy.deepcopy(v1)
    for m in modifs:
        m(r)
    r["version"]["valeur"] = version
    r["version"]["pourquoi"] = quoi + " ; seules différences avec recettes-rtx-v1 : cette étiquette et les chemins de données cités"
    (ROOT / "chatbot" / nom).write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(nom, "->", version)


ecrire("recettes_rtx_v4a.json", "recettes-rtx-v4a-affinage-propre", "v1 + affinage Molière sur le théâtre nettoyé", affinage)
ecrire("recettes_rtx_v4.json", "recettes-rtx-v4-donnees-propres", "v1 + pré-entraînement et affinage Molière sur données nettoyées", affinage, base)
ecrire("recettes_rtx_v4e.json", "recettes-rtx-v4e-base-propre", "v1 + pré-entraînement sur tokens nettoyés (affinage v1)", base)
