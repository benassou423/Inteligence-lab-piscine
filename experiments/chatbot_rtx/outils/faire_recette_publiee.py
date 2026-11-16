"""Écrit chatbot/recettes_rtx_publie.json : la recette exacte du modèle publié (S03).

C'est chatbot/recettes_rtx_v5_s03.json (l'essai du 15/11/2026, laissé tel quel comme pièce de l'essai) plus
deux déclarations qui ne changent pas l'entraînement :
- etape1.fichiers_synthetiques : le fichier de synthèse mélangé à l'étape 1, pour que le rapport du modèle
  (training_report.json) déclare sa part mesurée ;
- export : le décodage avec lequel le modèle a été jugé (--anchor-mu 1 --candidates 32), écrit dans son
  chat_config.json.

    python experiments/chatbot_rtx/outils/faire_recette_publiee.py
"""
import json
from pathlib import Path

src = json.loads(Path("chatbot/recettes_rtx_v5_s03.json").read_text(encoding="utf-8"))
m = src["chatbots"]["moliere"]
syn = [f for f in m["etape1"]["donnees"]["valeur"] if "synthetique" in f]
assert syn == ["data/stages/chatbot_20261114/synthetique_v4/entrainement/moliere.jsonl.zst"], syn
m["etape1"]["fichiers_synthetiques"] = {
    "valeur": syn,
    "pourquoi": "fichier de conversations de synthèse mélangé à l'étape 1 : déclaré pour que le rapport du "
                "modèle donne la part de tokens de synthèse réellement vue (mesurée, pas attendue)"}
m["export"] = {
    "valeur": ["--anchor-mu", "1", "--candidates", "32"],
    "pourquoi": "décodage avec lequel le modèle a été jugé à l'aveugle les 15 et 16/11/2026 : 32 réponses "
                "candidates, choix ancré sur le dernier message (chatbot/ancrage.py) ; écrit dans chat_config.json"}
src["version"] = {
    "valeur": "recettes-rtx-publie-s03",
    "pourquoi": "recette du modèle publié le 16/11/2026 : celle de l'essai v5-s03 du 15/11 (mêmes données, mêmes "
                "poids de tirage, même entraînement), plus la déclaration du fichier de synthèse et le décodage "
                "écrit à l'export"}
Path("chatbot/recettes_rtx_publie.json").write_text(json.dumps(src, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
print("chatbot/recettes_rtx_publie.json écrit")
