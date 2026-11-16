"""Fabrique les deux paquets joints à la release du chatbot Molière, et leur liste d'empreintes.

    python experiments/chatbot_rtx/outils/paquets_release.py DOSSIER_DE_SORTIE

1. chatbot_moliere_s03.zip : le dossier du modèle publié (chatbot_moliere/ : ckpt.pt, tokenizer.json,
   chat_config.json, training_report.json).
2. synthese_moliere.zip : les conversations de synthèse, aux chemins que lisent les recettes ; à décompresser
   à la racine d'un clone du dépôt. Contient aussi les fichiers tenus à l'écart (garde de par-cœur), les
   traces des campagnes et un LISEZMOI.

Refuse d'écrire si un fichier texte à empaqueter contient un chemin personnel. Les horodatages des entrées
sont fixés : deux exécutions donnent les mêmes octets.
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import zstandard

RACINE = Path(__file__).resolve().parents[3]
SORTIE = Path(sys.argv[1])
MODELE = RACINE / "runs/chatbot_rtx/modeles/chatbot_moliere"
D = "data/stages/chatbot_20261114"
# Formes de chemins seulement : un nom de logiciel ou de service peut figurer dans une conversation.
PERSONNEL = re.compile(r"Users[\\/]|/c/Users|AppData[\\/]|[A-Za-z]:\\\\", re.I)
DATE = (2026, 11, 16, 0, 0, 0)

SYNTHESE = [
    f"{D}/synthetique_v4/entrainement/moliere.jsonl.zst", f"{D}/synthetique_v4/entrainement/moliere.garde.json",
    f"{D}/synthetique_v4/fusion/moliere.campagne.json", f"{D}/synthetique_v4/fusion/moliere.rapport.json",
    f"{D}/synthetique_v5/fusion/moliere.campagne.json", f"{D}/synthetique_v5/fusion/moliere.rapport.json",
    f"{D}/synthetique_v6/entrainement/moliere.jsonl.zst", f"{D}/synthetique_v6/entrainement/moliere.garde.json",
    f"{D}/synthetique_v6/union/moliere.union.json",
]
for v, graines in (("v4", (290, 291, 292)), ("v5", (390, 391, 392))):
    for g in graines:
        SYNTHESE += [f"{D}/synthetique_{v}/ecart/gemma_s{g}.jsonl.zst", f"{D}/synthetique_{v}/ecart/gemma_s{g}.campagne.json"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def conversations(chemin):
    """(nombre de conversations, texte décompressé) d'un .jsonl.zst."""
    brut = zstandard.ZstdDecompressor().stream_reader(open(chemin, "rb")).read()
    texte = brut.decode("utf-8")
    return sum(1 for l in texte.splitlines() if l.strip()), texte


def controle(nom, texte):
    m = PERSONNEL.search(texte)
    if m:
        raise SystemExit(f"REFUS : {nom} contient un chemin personnel ({m.group(0)!r})")


def ajouter(z, nom, octets, stocke=False):
    info = zipfile.ZipInfo(nom, DATE)
    info.compress_type = zipfile.ZIP_STORED if stocke else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    z.writestr(info, octets)


SORTIE.mkdir(parents=True, exist_ok=True)
liste = {}

# 1. le modèle
contenu = {}
for f in ("ckpt.pt", "tokenizer.json", "chat_config.json", "training_report.json"):
    b = (MODELE / f).read_bytes()
    if f != "ckpt.pt":
        controle(f, b.decode("utf-8"))
    contenu[f] = {"octets": len(b), "sha256": sha(b)}
with zipfile.ZipFile(SORTIE / "chatbot_moliere_s03.zip", "w") as z:
    for f in contenu:
        ajouter(z, f"chatbot_moliere/{f}", (MODELE / f).read_bytes(), stocke=(f == "ckpt.pt"))
liste["chatbot_moliere_s03.zip"] = {"contenu": {f"chatbot_moliere/{k}": v for k, v in contenu.items()}}

# 2. la synthèse
contenu, comptes = {}, {}
for rel in SYNTHESE:
    p = RACINE / rel
    b = p.read_bytes()
    if rel.endswith(".jsonl.zst"):
        n, texte = conversations(p)
        comptes[rel] = n
    else:
        texte = b.decode("utf-8")
    controle(rel, texte)
    contenu[rel] = {"octets": len(b), "sha256": sha(b), **({"conversations": comptes[rel]} if rel in comptes else {})}
lisezmoi = f"""Conversations de synthèse du chatbot Molière
==============================================

Dépôt : https://github.com/benassou423/Inteligence-lab-piscine
Licence de ces conversations : CC BY 4.0 (voir LICENCES.md du dépôt). Elles ont été écrites par un modèle à
poids ouverts, gemma4:31b-it-q4_K_M (licence Apache 2.0), à partir de consignes publiées dans le dépôt
(chatbot/prompts_synthetiques_v4.json, chatbot/prompts_synthetiques_v5.json, chatbot/axes_consignes_v4.json,
chatbot/axes_consignes_v5.json), puis triées par des règles automatiques. Aucune n'a été écrite ni corrigée
à la main.

Mode d'emploi : décompresser cette archive à la racine d'un clone du dépôt. Les fichiers se rangent aux
chemins que lisent les recettes de chatbot/.

Fichiers d'entraînement
- {D}/synthetique_v4/entrainement/moliere.jsonl.zst : {comptes[f"{D}/synthetique_v4/entrainement/moliere.jsonl.zst"]} conversations
  (campagne v4). C'est le fichier lu par la recette du modèle publié, chatbot/recettes_rtx_publie.json.
- {D}/synthetique_v6/entrainement/moliere.jsonl.zst : {comptes[f"{D}/synthetique_v6/entrainement/moliere.jsonl.zst"]} conversations
  (campagnes v4 et v5 réunies). Lu par les recettes des essais du 16/11/2026, chatbot/recettes_rtx_v6_*.json.
  Ces essais n'ont pas donné de modèle meilleur que le modèle publié.

Fichiers tenus à l'écart (jamais entraînés ; ils servent à mesurer le par-cœur)
- {D}/synthetique_v4/ecart/gemma_s290, s291, s292 et {D}/synthetique_v5/ecart/gemma_s390, s391, s392.

Traces
- *.campagne.json : consignes (empreinte sha256), générateur, graines et réglages de chaque fichier.
- *.rapport.json, *.garde.json, *.union.json : comptes des filtres (doublons, fuites de la validation,
  messages de test retirés).

Format : un objet JSON par ligne, compressé en zstandard. Champs : id, source, doc_id, license, style,
turns (liste de tours : locuteur et texte), generator (modèle, consigne, graine, température), sujet
(thème, registre), persona.

Le récit de ces campagnes est dans docs/chatbot/HISTORIQUE.md ; la fiche des données dans
docs/chatbot/FICHE_DONNEES.md.
"""
contenu["LISEZMOI_SYNTHESE.txt"] = {"octets": len(lisezmoi.encode("utf-8")), "sha256": sha(lisezmoi.encode("utf-8"))}
with zipfile.ZipFile(SORTIE / "synthese_moliere.zip", "w") as z:
    ajouter(z, "LISEZMOI_SYNTHESE.txt", lisezmoi.encode("utf-8"))
    for rel in SYNTHESE:
        ajouter(z, rel, (RACINE / rel).read_bytes(), stocke=rel.endswith(".zst"))
liste["synthese_moliere.zip"] = {"contenu": contenu}

for nom in liste:
    b = (SORTIE / nom).read_bytes()
    liste[nom] = {"octets": len(b), "sha256": sha(b), **liste[nom]}
(SORTIE / "EMPREINTES.json").write_text(json.dumps(liste, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for nom, v in liste.items():
    print(f"{nom} : {v['octets']} octets, sha256 {v['sha256']}, {len(v['contenu'])} fichiers")
for rel, n in comptes.items():
    print(f"   {n:6d} conversations  {rel}")
