"""Réécrit, dans les traces « .campagne.json » déjà sur le disque, le chemin absolu du fichier de consignes.

Pourquoi : avant le 15/11/2026, tools/campagne_synthese.py écrivait le chemin complet du poste dans la clé
« consignes ». Ces traces accompagnent les données publiées : elles ne doivent contenir que le chemin
relatif au dépôt. L'identité des consignes tient à « consignes_sha256 », que cet outil ne touche pas.

Seule la valeur de la clé « consignes » change ; tous les autres octets du fichier restent identiques.

    python experiments/chatbot_rtx/outils/traces_publiables.py DOSSIER [DOSSIER ...]            # à blanc
    python experiments/chatbot_rtx/outils/traces_publiables.py DOSSIER [DOSSIER ...] --ecrire
"""
import argparse
import json
import re
import sys
from pathlib import Path, PureWindowsPath

RACINE = Path(__file__).resolve().parents[3]
CLE = re.compile(r'("consignes"\s*:\s*)("(?:[^"\\]|\\.)*")')
ABSOLU = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")


def publiable(chemin, racine=RACINE):
    """Chemin relatif à la racine du dépôt, en barres obliques ; hors du dépôt, le nom du fichier seul.
    Un chemin déjà relatif est rendu tel quel."""
    if not ABSOLU.match(chemin):
        return chemin
    p = PureWindowsPath(chemin.replace("/", "\\"))
    r = PureWindowsPath(str(racine).replace("/", "\\"))
    parties, base = [x.lower() for x in p.parts], [x.lower() for x in r.parts]
    if parties[: len(base)] == base and len(parties) > len(base):
        return "/".join(p.parts[len(base):])
    return p.name


def corriger(texte):
    """(nouveau texte, liste des (ancien, nouveau)) pour un fichier de trace."""
    changes = []

    def remplace(m):
        ancien = json.loads(m.group(2))
        nouveau = publiable(ancien)
        if nouveau != ancien:
            changes.append((ancien, nouveau))
        return m.group(1) + json.dumps(nouveau, ensure_ascii=False)

    return CLE.sub(remplace, texte), changes


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dossiers", nargs="+")
    p.add_argument("--ecrire", action="store_true", help="sans cette option, rien n'est modifié")
    a = p.parse_args()
    vus = modifies = 0
    for dossier in a.dossiers:
        for f in sorted(Path(dossier).rglob("*.campagne.json")):
            vus += 1
            brut = f.read_bytes()
            texte = brut.decode("utf-8")
            nouveau, changes = corriger(texte)
            if not changes:
                continue
            json.loads(nouveau)  # reste un JSON valide, sinon on s'arrête avant d'écrire
            modifies += 1
            for _, n in changes:
                print(f"{'réécrit' if a.ecrire else 'à réécrire'} : {f.as_posix()} -> consignes = {n}")
            if a.ecrire:
                f.write_bytes(nouveau.encode("utf-8"))
    print(f"{vus} traces lues, {modifies} {'réécrites' if a.ecrire else 'à réécrire (relancer avec --ecrire)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
