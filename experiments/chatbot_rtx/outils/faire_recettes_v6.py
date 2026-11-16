"""Recettes « v6 » : copies de recettes_rtx.json (v1) où l'étape 1 de Molière reçoit en 3e fichier la synthèse
v4 + v5 (gemma4), à des poids donnant la part voulue des tokens vus. Deux différences avec la v1, et deux seulement :
ce 3e fichier avec son poids, et l'export du modèle avec le décodage adopté (ancrage sur le dernier message,
32 candidates) pour que le chat_config.json publié donne le décodage qui a servi aux jugements.

Usage (racine du clone) : python experiments/chatbot_rtx/outils/faire_recettes_v6.py <fichier_synthetique> [part …]
"""
import json
import sys
from pathlib import Path

RACINE = Path.cwd()
sys.path.insert(0, str(RACINE))
N1, N2, TOKENS_ETAPE = 28_961_665, 844_977, 90_013_696  # journal moliere_etape1 du lancement v1


def main():
    fichier = sys.argv[1]
    from chatbot.finetune import open_tokenizer, read_scenes, scenes_to_tokens
    v1 = (RACINE / "chatbot/recettes_rtx.json").read_text(encoding="utf-8")
    tok = open_tokenizer(str(RACINE / json.loads(v1)["commun"]["tokenizer"]["valeur"]))
    scenes = list(read_scenes(fichier))
    n3 = len(scenes_to_tokens(scenes, tok))
    reste = N1 + 3 * N2
    bilan = {"fichier": Path(fichier).as_posix(), "conversations": len(scenes), "N3_tokens": n3, "recettes": []}
    for part in ([float(v) for v in sys.argv[2:]] or (0.03, 0.05, 0.10, 0.15)):
        w = round(part / (1 - part) * reste / n3, 2)
        reelle = w * n3 / (reste + w * n3)
        if reelle > 0.1505:
            sys.exit(f"REFUS : part {reelle:.4f} > 0,15")
        rec = json.loads(v1)
        moliere = rec["chatbots"]["moliere"]
        e1 = moliere["etape1"]
        e1["donnees"]["valeur"] = e1["donnees"]["valeur"] + [Path(fichier).as_posix()]
        e1["donnees"]["pourquoi"] += (" ; + conversations de synthèse écrites par gemma4:31b-it-q4_K_M (poids ouverts, Apache 2.0), "
                                      "consignes publiées (chatbot/prompts_synthetiques_v4.json et _v5.json, "
                                      "chatbot/axes_consignes_v4.json et _v5.json), mélangées à l'étape 1")
        e1["fichiers_synthetiques"] = {"valeur": [Path(fichier).as_posix()],
                                       "pourquoi": "fichiers de l'étape 1 écrits par un modèle : le rapport du modèle "
                                                   "(training_report.json, part_synthetique) déclare leur part mesurée"}
        e1["poids"]["valeur"] = [1, 3, w]
        e1["poids"]["pourquoi"] += (f" ; synthèse au poids {w} : part attendue des tokens vus {reelle:.4f} (= w·N3 / (N1 + 3·N2 + w·N3), "
                                    f"N3 = {n3} tokens, {len(scenes)} conversations), plafond 15 %")
        moliere["export"] = {"valeur": ["--anchor-mu", "1", "--candidates", "32"],
                             "pourquoi": "décodage adopté le 14/11 (chatbot/ancrage.py) : le choix de la réponse ne compte que ce que le "
                                         "dernier message apporte, parmi 32 candidates ; c'est le décodage de tous les jugements à l'aveugle"}
        nom = f"s{int(round(part * 100)):02d}"
        rec["version"]["valeur"] = f"recettes-rtx-v6-synthese-gemma4-{nom}"
        rec["version"]["pourquoi"] += (f" ; v6-{nom} = v1 + synthèse gemma4 (campagnes v4 et v5) dans l'étape 1 de moliere, part visée "
                                       f"{int(round(part * 100))} %, et décodage adopté écrit à l'export (rien d'autre ne change)")
        sortie = f"chatbot/recettes_rtx_v6_{nom}.json"
        (RACINE / sortie).write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bilan["recettes"].append({"sortie": sortie, "poids": w, "part_attendue": round(reelle, 4),
                                  "passages_par_conversation_au_pas_1000": round(w * 1000 * 32768 / (reste + w * n3), 2),
                                  "passages_sur_l_etape": round(w * TOKENS_ETAPE / (reste + w * n3), 2)})
    print(json.dumps(bilan, ensure_ascii=False, indent=1))


main()
