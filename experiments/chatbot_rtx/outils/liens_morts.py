"""Liens relatifs des fichiers Markdown publiés : la cible existe-t-elle, et sera-t-elle dans git ?

    python experiments/chatbot_rtx/outils/liens_morts.py

Lit les .md suivis par git et ceux qui attendent un commit (non ignorés). Pour chaque lien « [texte](cible) »
relatif : la cible doit exister sur le disque et ne pas être ignorée par git. Les liens http(s), mailto et
les ancres seules sont laissés de côté. Sortie : une ligne par lien fautif, code de retour 1 s'il y en a.
"""
import re
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
LIEN = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def git(*args):
    return subprocess.run(["git", *args], cwd=RACINE, capture_output=True, text=True, encoding="utf-8").stdout


def main():
    suivis = set(git("ls-files").splitlines())
    attente = set(git("ls-files", "--others", "--exclude-standard").splitlines())
    publies = suivis | attente
    fautes = 0
    # un fichier suivi mais déplacé ou supprimé avant le commit n'est plus sur le disque : on le saute
    publies = {p for p in publies if (RACINE / p).exists()}
    mds = sorted(p for p in publies if p.endswith(".md"))
    for md in mds:
        dossier = (RACINE / md).parent
        hors_code = True
        for n, ligne in enumerate((RACINE / md).read_text(encoding="utf-8").splitlines(), 1):
            if ligne.lstrip().startswith("```"):
                hors_code = not hors_code
            if not hors_code:
                continue
            for cible in LIEN.findall(ligne):
                if re.match(r"^(https?:|mailto:|#)", cible):
                    continue
                chemin = cible.split("#")[0]
                if not chemin:
                    continue
                p = (dossier / chemin).resolve()
                try:
                    rel = p.relative_to(RACINE).as_posix()
                except ValueError:
                    print(f"{md}:{n} : sort du dépôt -> {cible}")
                    fautes += 1
                    continue
                if not p.exists():
                    print(f"{md}:{n} : cible absente -> {cible}")
                    fautes += 1
                elif p.is_file() and rel not in publies:
                    print(f"{md}:{n} : cible ignorée par git -> {cible}")
                    fautes += 1
                elif p.is_dir() and not any(x.startswith(rel + "/") for x in publies):
                    print(f"{md}:{n} : dossier sans fichier publié -> {cible}")
                    fautes += 1
    print(f"{len(mds)} fichiers Markdown lus, {fautes} lien(s) fautif(s)")
    return 1 if fautes else 0


if __name__ == "__main__":
    sys.exit(main())
