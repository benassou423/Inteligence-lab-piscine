"""Écrit chatbot/recettes_rtx_v2.json : copie de recettes_rtx.json, synthèse mélangée à l'étape 1 de Molière.

Usage (racine du clone) : python faire_recette_v2.py <fichier_synthetique> [--w W] [--sortie chatbot/recettes_rtx_v2.json] [--version ETIQUETTE]
Seuls changent : version, chatbots.moliere.etape1.donnees, chatbots.moliere.etape1.poids.
Le bloc « synthetique » reste tel quel : son fichier (data/stages/chatbot_20261113/synthetique/moliere.jsonl.zst)
n'existe pas, donc l'étape synthétique séparée ne fait pas partie du plan (pipeline_rtx.py:398-401).
"""
import argparse
import json
import sys
from pathlib import Path

RACINE = Path.cwd()
sys.path.insert(0, str(RACINE))

N1 = 28_961_665   # tokens rendus de theatre_train (journal moliere_etape1 du lancement v1)
N2 = 844_977      # tokens rendus de moliere_train, copies MOLIÈRE comprises (idem)
PART_MAX = 0.13   # plafond interne de la part de tokens VUS (règle de l'étudiant : 15 %)
W_MAX = 5.0       # au plus ~13 passages sur chaque conversation synthétique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fichier")
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--sortie", default="chatbot/recettes_rtx_v2.json")
    ap.add_argument("--version", default="recettes-rtx-v2-syn-etape1")
    a = ap.parse_args()
    from chatbot.finetune import open_tokenizer, read_scenes, scenes_to_tokens
    rec = json.loads((RACINE / "chatbot/recettes_rtx.json").read_text(encoding="utf-8"))
    tok = open_tokenizer(str(RACINE / rec["commun"]["tokenizer"]["valeur"]))
    scenes = list(read_scenes(a.fichier))
    n3 = len(scenes_to_tokens(scenes, tok))
    reste = N1 + 3 * N2
    w_plafond = PART_MAX / (1 - PART_MAX) * reste / n3
    w = a.w if a.w is not None else min(W_MAX, w_plafond)
    w = round(w, 2)
    part = w * n3 / (reste + w * n3)
    passages = w * 90_013_696 / (reste + w * n3)
    if part > 0.15:
        sys.exit(f"REFUS : part de tokens vus {part:.4f} > 0,15 avec w = {w}")
    e1 = rec["chatbots"]["moliere"]["etape1"]
    chemin = Path(a.fichier).as_posix()
    e1["donnees"]["valeur"] = e1["donnees"]["valeur"] + [chemin]
    e1["donnees"]["pourquoi"] += (" ; + conversations synthétiques (qwen3:8b, consignes publiées) mélangées à l'étape 1 : "
                                  "l'étape synthétique séparée choisit best.pt sur du vrai Molière, pas 0 compris, et peut n'avoir aucun effet")
    e1["poids"]["valeur"] = [1, 3, w]
    e1["poids"]["pourquoi"] += (f" ; synthétique au poids {w} : part attendue des tokens vus {part:.4f} "
                                f"(= w·N3 / (N1 + 3·N2 + w·N3), N3 = {n3} tokens, {len(scenes)} conversations), "
                                f"plafond interne {PART_MAX}, environ {passages:.1f} passages par conversation sur l'étape entière")
    rec["version"]["valeur"] = a.version
    rec["version"]["pourquoi"] += " ; v2 = v1 + synthétique dans l'étape 1 de moliere (rien d'autre ne change)"
    Path(a.sortie).write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sortie": a.sortie, "version": a.version, "conversations": len(scenes), "N3_tokens": n3,
                      "w": w, "w_pour_13_pour_cent": round(w_plafond, 2), "part_tokens_vus_attendue": round(part, 4),
                      "passages_sur_l_etape": round(passages, 1),
                      "passages_au_pas_1000": round(w * 1000 * 32768 / (reste + w * n3), 1),
                      "donnees": e1["donnees"]["valeur"], "poids": e1["poids"]["valeur"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
