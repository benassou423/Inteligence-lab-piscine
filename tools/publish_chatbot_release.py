"""Gros fichiers de l'entraînement RTX -> pièces jointes d'une release GitHub. Mac seulement.

    uv run python tools/publish_chatbot_release.py                                  # À BLANC
    uv run python tools/publish_chatbot_release.py --depot COMPTE/DEPOT --tag ETIQUETTE  # À BLANC
    uv run python tools/publish_chatbot_release.py --depot COMPTE/DEPOT --tag ETIQUETTE --publier

Pourquoi ? Git refuse tout fichier de plus de 100 Mo : les tokens de pré-entraînement (6 fichiers
.bin, 4,2 Go, le plus gros 1,2 Go) ne peuvent pas être commités. Une release GitHub accepte des
pièces jointes jusqu'à 2 Gio chacune : le PC les télécharge avec
`python -m chatbot.pipeline_rtx telecharger`, d'après la liste chatbot/donnees_rtx.json.

1. Régénère chatbot/donnees_rtx.json depuis les VRAIS fichiers : nom de pièce jointe (unique, sans
   accent ni espace), chemin de destination (le même que sur le Mac), taille, sha256. Rien de
   daté dans ce fichier : il ne change que si les données changent.
2. Par défaut, N'ENVOIE RIEN : affiche le plan exact (dépôt, étiquette, visibilité du dépôt lue
   par gh, fichiers, tailles, total, commandes gh qui seraient lancées).
3. --depot, --tag ET --publier : crée la release si besoin, envoie chaque fichier (reprise : un
   fichier déjà en ligne avec la même taille n'est pas renvoyé), contrôle les tailles en ligne,
   puis écrit le dépôt et l'étiquette dans chatbot/donnees_rtx.json (à commiter ensuite).
Le dépôt de la piscine est PUBLIC : l'étudiant choisit lui-même ce dépôt ou un dépôt privé à part.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot import pipeline_rtx as L  # noqa: E402

MAX_OCTETS = 2 * 1024 ** 3   # GitHub : une pièce jointe doit faire MOINS de 2 Gio
ENVOI = "data/stages/chatbot_20261113/envoi_release"  # liens physiques nommés comme les pièces jointes


def liste_fichiers(racine):
    """[(nom de pièce jointe, destination, rôle)] des fichiers présents sur ce Mac."""
    racine = Path(racine)
    meta = json.loads((racine / L.PRETRAIN / "meta.json").read_text(encoding="utf-8"))
    out = [("pretrain_meta.json", f"{L.PRETRAIN}/meta.json", "pre-entrainement")]
    out += [(f"pretrain_{i['file']}", f"{L.PRETRAIN}/{i['file']}", "pre-entrainement")
            for i in meta["sources"].values()]
    for d in L.DONNEES:
        groupe = "theatre" if "/theatre/" in d else "dialogues"
        out.append((f"{groupe}_{Path(d).name}", d, "affinage"))
    for d in L.SYNTHETIQUES:  # facultatifs : seulement s'ils existent
        if (racine / d).exists():
            out.append((f"synthetique_{Path(d).name}", d, "synthetique"))
    for style, d in L.ECHANTILLONS.items():  # minuscules : l'autotest teste l'étape synthétique
        if (racine / d).exists():
            out.append((f"essai_synthetique_{Path(d).name}", d, "essai"))
    return out


def regenerer(racine=ROOT, sortie=None, ancien=None):
    """Nouveau contenu de donnees_rtx.json (dépôt et étiquette repris de l'ancien fichier)."""
    racine = Path(racine)
    ancien = ancien or {}
    fichiers = []
    for nom, dest, role in liste_fichiers(racine):
        p = racine / dest
        if not p.is_file():
            sys.exit(f"ERREUR : fichier attendu absent : {dest}")
        octets = p.stat().st_size
        if octets >= MAX_OCTETS:
            sys.exit(f"ERREUR : {dest} fait {octets:,d} octets, 2 Gio ou plus : GitHub le refuserait")
        print(f"  sha256 de {dest} ({octets / 1e6:.1f} Mo)…", flush=True)
        fichiers.append({"piece_jointe": nom, "destination": dest, "role": role, "octets": octets,
                         "sha256": L.sha256_fichier(p)})
    noms = [f["piece_jointe"] for f in fichiers]
    if len(set(noms)) != len(noms) or not all(L.NOM_SUR.match(n) for n in noms):
        sys.exit(f"ERREUR : noms de pièces jointes non uniques ou avec accent/espace : {noms}")
    return {
        "note": "Gros fichiers de l'entraînement RTX (plus de 100 Mo : refusés par git), en pièces "
                "jointes d'une release GitHub. Régénéré par tools/publish_chatbot_release.py ; lu par "
                "`python -m chatbot.pipeline_rtx telecharger` et `verifier`.",
        "depot": ancien.get("depot") or f"{L.A_RENSEIGNER} (COMPTE/DEPOT choisi par l'étudiant)",
        "tag": ancien.get("tag") or f"{L.A_RENSEIGNER} (étiquette choisie par l'étudiant)",
        "total_octets": sum(f["octets"] for f in fichiers),
        "fichiers": fichiers,
    }


def preparer_envoi(racine, donnees):
    """Dossier d'envoi : un lien physique par fichier, nommé comme sa pièce jointe (gh envoie le
    fichier sous son propre nom). Aucun octet copié quand le disque le permet."""
    dossier = L.creer_dossier(Path(racine) / ENVOI)
    out = {}
    for f in donnees["fichiers"]:
        cible = dossier / f["piece_jointe"]
        if cible.exists():
            cible.unlink()
        try:
            os.link(Path(racine) / f["destination"], cible)
        except OSError:
            shutil.copy2(Path(racine) / f["destination"], cible)
        out[f["piece_jointe"]] = cible
    return dossier, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depot", default=None, help="COMPTE/DEPOT qui recevra la release")
    p.add_argument("--tag", default=None, help="étiquette de la release (ex. chatbot-donnees-20261113)")
    p.add_argument("--publier", action="store_true", help="envoyer pour de vrai (sinon : à blanc)")
    p.add_argument("--racine", default=str(ROOT), help="dossier qui contient data/ (tests)")
    p.add_argument("--json", default=str(L.DONNEES_RTX), help="fichier liste à régénérer")
    args = p.parse_args(argv)
    L.console_utf8()
    ancien = L.lire_json(args.json, {}) or {}
    print("1. Régénération de la liste des gros fichiers (taille, sha256)…", flush=True)
    donnees = regenerer(args.racine, ancien=ancien)
    L.ecrire_json(args.json, donnees)
    print(f"   {L.rel(args.json, args.racine)} : {len(donnees['fichiers'])} fichiers, {donnees['total_octets']:,d} octets")
    print("2. Plan d'envoi :", flush=True)
    exe = shutil.which("gh")
    vis = vide = None
    if exe and args.depot and L.gh(exe, "auth", "status", check=False).returncode == 0:
        vis, vide = L.visibilite(exe, args.depot), L.depot_vide(exe, args.depot)  # lecture seule
    L.afficher_plan(args.depot, args.tag, vis, {f["piece_jointe"]: f["octets"] for f in donnees["fichiers"]},
                    "Données d'entraînement des chatbots (RTX)", vide)
    if not args.publier:
        print("À BLANC : rien n'a été envoyé. Pour envoyer (décision de l'étudiant) : ajouter "
              "--depot COMPTE/DEPOT --tag ETIQUETTE --publier")
        return 0
    try:
        depot, tag = L.depot_tag(args.depot, args.tag)
        exe = L.gh_pret()
        dossier, fichiers = preparer_envoi(args.racine, donnees)
        L.envoyer_release(exe, depot, tag, fichiers, "Données d'entraînement des chatbots (RTX)",
                          "Tokens de pré-entraînement et dialogues des chatbots de démo (hors "
                          "concours). Liste, tailles et sha256 : chatbot/donnees_rtx.json.")
    except L.Refus as e:
        sys.exit(f"ERREUR : {e}")
    shutil.rmtree(dossier, ignore_errors=True)  # liens physiques : les données restent en place
    donnees.update(depot=depot, tag=tag)
    L.ecrire_json(args.json, donnees)
    print(f"PUBLICATION RÉUSSIE. {L.rel(args.json, args.racine)} contient maintenant depot={depot} et tag={tag} : le "
          f"commiter (avec l'accord de l'étudiant) pour que le PC les connaisse, ou passer "
          f"--depot/--tag à `telecharger`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
