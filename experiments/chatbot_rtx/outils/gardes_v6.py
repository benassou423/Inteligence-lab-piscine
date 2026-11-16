"""Gardes des essais v6 (règle du 15/11 16:00), lues dans runs/evals_manuelles/<nom>/bpb_campagne.json.

- BPB sur moliere_val <= BPB de A0 + 0,010 ;
- par-cœur : 1 - BPB(synthèse entraînée) / moyenne des BPB des 3 fichiers v5 tenus à l'écart, < 0,10.

    python experiments/chatbot_rtx/outils/gardes_v6.py
"""
import json
from pathlib import Path

BASE = Path("runs/evals_manuelles")
NOMS = ["v6_A0", "v6_S03t", "v6_s03", "v6_s05", "v6_s10", "v6_s15"]


def lire(nom):
    d = json.loads((BASE / nom / "bpb_campagne.json").read_text(encoding="utf-8"))
    val = ent = None
    ecart = []
    for f, x in d["fichiers"].items():  # {chemin: {val_bpb, val_loss, fenetres}}
        b = x["val_bpb"]
        if "moliere_val" in f:
            val = b
        elif "synthetique_v6" in f:
            ent = b
        elif "ecart" in f:
            ecart.append(b)
    return val, ent, ecart


res = {}
a0 = lire("v6_A0")[0]
print(f"limite BPB moliere_val : {a0:.5f} + 0,010 = {a0 + 0.010:.5f}")
for nom in NOMS:
    val, ent, ecart = lire(nom)
    moy = sum(ecart) / len(ecart)
    par_coeur = 1 - ent / moy
    mes = json.loads((BASE / nom / "neuf2" / "mesures.json").read_text(encoding="utf-8"))
    res[nom] = {"bpb_moliere_val": val, "bpb_synthese_entrainee": ent, "bpb_ecart": ecart, "bpb_ecart_moyen": moy,
                "par_coeur": par_coeur, "garde_bpb": val <= a0 + 0.010, "garde_par_coeur": par_coeur < 0.10,
                "mots_moyens": mes.get("mean_words"), "secondes_par_reponse": mes.get("seconds_per_reply")}
    print(f"{nom:8s} val {val:.5f} {'ok ' if val <= a0 + 0.010 else 'NON'} | entraînée {ent:.4f} | écart {moy:.4f} "
          f"| par-cœur {100 * par_coeur:5.2f} % {'ok ' if par_coeur < 0.10 else 'NON'} | mots {mes.get('mean_words')}")
(BASE / "campagne" / "gardes_v6.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
