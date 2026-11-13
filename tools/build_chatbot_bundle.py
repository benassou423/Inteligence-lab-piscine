"""Paquet autonome pour entraîner les deux chatbots sur le PC Windows (RTX 5090). Mac seulement.

    uv run python tools/build_chatbot_bundle.py --remplacer                # paquet complet
    uv run python tools/build_chatbot_bundle.py --sans-donnees --remplacer # paquet léger d'essai

Sortie : data/stages/chatbot_20261113/paquet_rtx/FTB_CHATBOT_RTX_20261113/ (dossier) et
FTB_CHATBOT_RTX_20261113.zip (suffixe _LEGER avec --sans-donnees). Rejouable en une commande : si
les fichiers de dialogues changent, relancer la même commande avec --remplacer.

Contenu, aux MÊMES chemins relatifs que dans le dépôt (les commandes sont identiques partout) :
- le code strictement nécessaire : les modules chatbot/ et pipeline/ réellement importés (liste
  obtenue en les important, pas écrite à la main), tout ftb/ (le kit, en lecture seule), les recettes ;
- le tokenizer, les tokens de pré-entraînement (.bin + meta.json), les scènes, les dialogues, les
  validations et, s'ils existent, les fichiers synthétiques (sauf --sans-donnees) ;
- donnees_essai/ : de minuscules extraits des vraies données pour l'autotest (toujours) ;
- requirements_pc.txt (versions figées), PC_RTX.md (instructions de la session Claude Code du PC,
  les mêmes que pour un clone git), un CLAUDE.md court qui y renvoie, LISEZMOI_PC.md (pour
  l'étudiant) et TRANSFER_MANIFEST.json (sha256 et taille de chaque fichier).

Solution de SECOURS : la route normale est le clone git + `telecharger` (voir PC_RTX.md).

Les .bin et .zst sont rangés dans l'archive SANS compression (ils ne se compressent pas) ; l'archive
peut dépasser 4 Go (ZIP64). Les gros fichiers du dossier sont des liens physiques vers les données
(aucune copie de 4 Go en plus sur le disque).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chatbot import pipeline_rtx as L  # noqa: E402 - listes de données et extraits d'essai partagés

NOM = "FTB_CHATBOT_RTX_20261113"
SORTIE = ROOT / "data/stages/chatbot_20261113/paquet_rtx"
TOKENIZER = "submissions/2026-11-11_153839/tokenizer.json"
# Mêmes listes que la route git (chatbot/pipeline_rtx.py) : une seule source de vérité.
PRETRAIN, DONNEES, SYNTHETIQUES, ECHANTILLONS = L.PRETRAIN, L.DONNEES, L.SYNTHETIQUES, L.ECHANTILLONS
# Modules d'entrée : tout ce qu'ils importent du dépôt (chatbot.*, pipeline.*) part dans le paquet.
ENTREES = ["chatbot.pipeline_rtx", "chatbot.pretrain", "chatbot.finetune", "chatbot.generate",
           "chatbot.evaluate"]
PAQUETS = {"chatbot", "ftb", "pipeline"}
# Versions figées : celles de docs/remise/PASSAGE_PC_RTX5090.md ; on relit celles du .venv du Mac.
TORCH = "2.15.0"
INDEX_CUDA = "https://download.pytorch.org/whl/cu132"
AUTRES = ["numpy", "tokenizers", "regex", "zstandard", "xxhash"]
STOCKES = (".bin", ".zst", ".pt", ".npy", ".gz", ".zip")  # déjà compressés ou binaires : rangés tels quels


def sha256(path, bloc=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bloc):
            h.update(chunk)
    return h.hexdigest()


def modules_importes():
    """Fichiers du dépôt nécessaires à l'import des modules d'entrée (exécuté, pas deviné).

    Deux listes : les modules .py chargés (sys.modules) et les AUTRES fichiers du dépôt ouverts
    pendant l'import (crochet d'audit « open » de Python). Exemple trouvé ainsi : ftb/__init__.py
    lit config/official.json dès l'import ; sans lui, rien ne démarre.
    """
    code = ("import importlib, json, sys\n"
            "opened = set()\n"
            "sys.addaudithook(lambda ev, a: opened.add(str(a[0])) if ev == 'open' and a and "
            "isinstance(a[0], str) else None)\n"
            f"for m in {ENTREES!r}: importlib.import_module(m)\n"
            "mods = {m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)}\n"
            "print(json.dumps(sorted(mods | opened)))")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True)
    files = []
    for f in json.loads(out.stdout.splitlines()[-1]):
        try:
            r = Path(f).resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        if "__pycache__" in r.parts or r.as_posix() == "data/validation.jsonl":
            continue  # la validation du concours ne part jamais (et n'est pas lue à l'import)
        if (r.parts[0] in PAQUETS and r.suffix == ".py") or r.as_posix() == "config/official.json":
            files.append(r.as_posix())
    return sorted(set(files))


def code_du_paquet():
    imported = modules_importes()
    files = [f for f in imported if not f.startswith("ftb/")]
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "ftb").glob("*.py"))  # tout le kit
    files += ["chatbot/recettes_rtx.json", "chatbot/donnees_rtx.json", "PC_RTX.md"]
    if "config/official.json" not in files:
        sys.exit("ERREUR : config/official.json n'a pas été vu à l'import : vérifier modules_importes")
    for f in files:  # paquets Python : leur __init__.py
        init = (Path(f).parent / "__init__.py").as_posix()
        if (ROOT / init).exists() and init not in files:
            files.append(init)
    return sorted(set(files)), imported


def donnees_essai(dest):
    """Extraits minuscules des vraies données, dans donnees_essai/ (pour l'autotest)."""
    return L.preparer_donnees_essai(ROOT, dest)


def requirements():
    """requirements_pc.txt avec les versions réellement installées sur le Mac (+ écarts notés)."""
    lignes = ["# Python 3.12. Installer d'abord PyTorch CUDA depuis l'index officiel PyTorch :",
              f"#   uv pip install --python .venv/Scripts/python.exe torch=={TORCH} --index-url {INDEX_CUDA}",
              "# puis : uv pip install --python .venv/Scripts/python.exe -r requirements_pc.txt"]
    versions = {"torch": TORCH}
    for mod in AUTRES:
        v = metadata.version(mod)
        versions[mod] = v
        lignes.append(f"{mod}=={v}")
    mac_torch = metadata.version("torch")
    ecarts = []
    if mac_torch.split("+")[0] != TORCH:
        ecarts.append(f"torch du Mac {mac_torch} ≠ {TORCH}")
    ecarts.append("xxhash ajouté à la liste de PASSAGE_PC_RTX5090.md : chatbot/theatre.py l'importe "
                  "(chargé par chatbot.pretrain)")
    return "\n".join(lignes) + "\n", versions, ecarts


CLAUDE_MD = """# Mission de cette session Claude Code (PC Windows, RTX 5090)

Lis PC_RTX.md en entier et suis-le : ce sont tes instructions complètes (contexte, règles, ordre
des commandes, que faire en cas d'échec, ce qu'il faut rapporter).

Tu es dans le PAQUET ZIP de secours (TRANSFER_MANIFEST.json est présent), pas dans un clone git :
1. les données sont déjà là : saute l'étape `telecharger` ;
2. installe les modules avec `requirements_pc.txt` (mêmes versions que dans PC_RTX.md) ;
3. les sorties vont dans sorties_rtx/ et l'archive RESULTATS_CHATBOT_RTX.zip dans ce dossier ;
4. pas de git ici : l'étudiant rapporte l'archive sur le Mac à la main (clé, disque), sauf s'il
   te demande explicitement `publier` (à blanc d'abord, puis --publier avec son accord).
"""

LISEZMOI = """# Paquet d'entraînement des chatbots — PC Windows RTX 5090 (solution de secours)

Prochaine action : ouvrir ce dossier dans Claude Code sur le PC et lui coller le message de
docs/remise/PASSAGE_PC_RTX5090_CHATBOT.md (sur le Mac, partie « secours ») — 2 min.

1. Ce dossier contient tout : code, données déjà tokenisées, recettes, autotest. Rien à télécharger
   sauf Python et PyTorch (environ 3 Go, durée selon la connexion).
2. La session Claude Code du PC lit CLAUDE.md puis PC_RTX.md et enchaîne : installer, verifier,
   autotest, mesurer, lancer, resultats. Elle te demande ton budget d'heures après `mesurer`.
3. Une coupure (veille, Ctrl-C, redémarrage) n'est pas grave : relancer la même commande reprend
   là où on s'était arrêté.
4. À la fin : rapporter RESULTATS_CHATBOT_RTX.zip sur le Mac, puis
   `uv run python tools/import_chatbot_results.py RESULTATS_CHATBOT_RTX.zip` (vérifie, range,
   fait parler les deux chatbots) ; `--installer` pour les mettre dans la démo.
"""


def placer(src, dst, lien):
    """Lien physique (même disque, aucun octet copié) pour les gros fichiers, copie sinon."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if lien:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def construire(sans_donnees=False, sortie=SORTIE, remplacer=False):
    t0 = time.time()
    nom = NOM + ("_LEGER" if sans_donnees else "")
    dest = Path(sortie) / nom
    archive = Path(sortie) / f"{nom}.zip"
    if (dest.exists() or archive.exists()) and not remplacer:
        sys.exit(f"ERREUR : {dest} existe déjà : ajouter --remplacer pour le reconstruire")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    code, imported = code_du_paquet()
    print(f"code : {len(code)} fichiers (modules importés : {', '.join(imported)})")
    for f in code:
        placer(ROOT / f, dest / f, lien=False)
    data = [TOKENIZER]
    syn_reels = [s for s in SYNTHETIQUES if (ROOT / s).exists()]
    if not sans_donnees:
        meta = json.loads((ROOT / PRETRAIN / "meta.json").read_text(encoding="utf-8"))
        data += [f"{PRETRAIN}/meta.json"] + [f"{PRETRAIN}/{i['file']}" for i in meta["sources"].values()]
        data += DONNEES + syn_reels
    for f in data:
        placer(ROOT / f, dest / f, lien=(ROOT / f).stat().st_size > 50_000_000)
    syn_essai = donnees_essai(dest)
    req, versions, ecarts = requirements()
    (dest / "requirements_pc.txt").write_text(req, encoding="utf-8")
    (dest / "CLAUDE.md").write_text(CLAUDE_MD, encoding="utf-8")
    (dest / "LISEZMOI_PC.md").write_text(LISEZMOI, encoding="utf-8")

    fichiers = sorted(p for p in dest.rglob("*") if p.is_file())
    manifest = {"paquet": nom, "complet": not sans_donnees, "cree": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "construit_par": "tools/build_chatbot_bundle.py", "versions_figees": versions,
                "ecarts_versions": ecarts, "synthetiques_reels": syn_reels,
                "synthetiques_essai": syn_essai,
                "fichiers": {p.relative_to(dest).as_posix(): {"sha256": sha256(p), "octets": p.stat().st_size}
                             for p in fichiers}}
    (dest / "TRANSFER_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    fichiers.append(dest / "TRANSFER_MANIFEST.json")

    tmp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for p in fichiers:
            name = p.relative_to(dest).as_posix()
            z.write(p, f"{nom}/{name}",
                    compress_type=zipfile.ZIP_STORED if name.endswith(STOCKES) else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(tmp) as z:  # relecture complète : CRC de chaque fichier
        assert z.testzip() is None, "archive abîmée"
        assert len(z.namelist()) == len(fichiers)
        assert not any(n.startswith("/") or ".." in Path(n).parts for n in z.namelist())
    os.replace(tmp, archive)
    preuve = {"archive": archive.name, "octets": archive.stat().st_size, "sha256": sha256(archive),
              "fichiers": len(fichiers), "complet": not sans_donnees,
              "octets_donnees": sum(i["octets"] for k, i in manifest["fichiers"].items()
                                    if k.startswith("data/")),
              "secondes": round(time.time() - t0, 1), "versions_figees": versions,
              "ecarts_versions": ecarts, "synthetiques_reels": syn_reels,
              "synthetiques_essai": syn_essai, "cree": manifest["cree"]}
    (Path(sortie) / f"{nom}.verification.json").write_text(json.dumps(preuve, indent=2, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")
    print(json.dumps(preuve, indent=2, ensure_ascii=False))
    return preuve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sans-donnees", action="store_true",
                   help="archive légère : code + extraits d'essai, sans les vraies données")
    p.add_argument("--sortie", default=str(SORTIE))
    p.add_argument("--remplacer", action="store_true", help="reconstruire un paquet existant")
    args = p.parse_args(argv)
    construire(args.sans_donnees, args.sortie, args.remplacer)


if __name__ == "__main__":
    main()
