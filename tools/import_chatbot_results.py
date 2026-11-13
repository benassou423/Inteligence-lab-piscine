"""Retour du PC : vérifie RESULTATS_CHATBOT_RTX.zip, range les rapports, fait parler les chatbots.

    uv run python tools/import_chatbot_results.py RESULTATS_CHATBOT_RTX.zip
    uv run python tools/import_chatbot_results.py RESULTATS_CHATBOT_RTX.zip --installer
    uv run python tools/import_chatbot_results.py --depuis-release COMPTE/DEPOT ETIQUETTE

0. --depuis-release : télécharge par gh le manifeste et les morceaux de l'archive publiée par
   `python -m chatbot.pipeline_rtx publier` sur le PC, vérifie chaque sha256, reconstitue
   l'archive, puis suit le même chemin qu'une archive copiée à la main.

1. Contrôle de l'archive : chemins sûrs, aucun lien, chaque fichier conforme au manifeste (sha256,
   taille), aucun fichier en trop.
2. Rangement : tout le contenu sous data/stages/chatbot_20261113/resultats_rtx/<nom>/, et les
   rapports d'entraînement (report.json, log.jsonl) sous runs/<nom>/<étape>/.
3. Preuve de chargement : chaque chatbot répond à 3 messages, sur le PROCESSEUR (rien sur le GPU
   du Mac). Réponses écrites dans preuve_import.json.
4. --installer seulement : les deux dossiers modèles vont dans data/models/chatbot_moliere et
   data/models/chatbot_francais ; l'ancien modèle est gardé à côté, renommé
   chatbot_moliere.avant_<date>. Refusé pour une archive d'autotest ou de la recette d'essai
   (modèles minuscules), ou incomplète : seules les tailles S et M s'installent.
La démo n'est PAS redémarrée : c'est l'étudiant qui la relance s'il veut le nouveau modèle.
"""

import argparse
import hashlib
import json
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "data/stages/chatbot_20261113/resultats_rtx"
MAX_OCTETS = 5_000_000_000  # garde-fou : un modèle M exporté fait ~0,2 Go
DOSSIERS = {"chatbot_moliere": "repliques", "chatbot_francais": "tirets"}
TAILLES_INSTALLABLES = ("S", "M")  # « essai » = modèles minuscules de la preuve de bout en bout
MESSAGES = {
    "repliques": ["Bonjour, monsieur. Comment vous portez-vous ?", "Que pensez-vous des médecins ?",
                  "Je suis amoureux de votre fille."],
    "tirets": ["Salut, ça va ?", "Tu as passé un bon week-end ?", "J'ai peur de rater mon examen demain."],
}


def sha256(path, bloc=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bloc):
            h.update(chunk)
    return h.hexdigest()


def extraire(archive, dest):
    """Extrait après contrôle des chemins ; renvoie (dossier racine, manifeste)."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > MAX_OCTETS:
            raise ValueError("archive anormalement grande")
        tops = set()
        for i in infos:
            name = PurePosixPath(i.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in i.filename:
                raise ValueError(f"chemin dangereux : {i.filename}")
            if stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError(f"lien inattendu : {i.filename}")
            tops.add(name.parts[0])
        if len(tops) != 1:
            raise ValueError(f"un seul dossier racine attendu, trouvé {sorted(tops)}")
        top = tops.pop()
        if z.testzip() is not None:
            raise ValueError("archive abîmée (CRC)")
        z.extractall(dest)
    racine = Path(dest) / top
    man = json.loads((racine / "MANIFESTE_RESULTATS.json").read_text(encoding="utf-8"))
    vus = {p.relative_to(racine).as_posix() for p in racine.rglob("*") if p.is_file()}
    vus.discard("MANIFESTE_RESULTATS.json")
    if vus != set(man["fichiers"]):
        raise ValueError(f"fichiers hors manifeste : {sorted(vus ^ set(man['fichiers']))[:10]}")
    for name, info in man["fichiers"].items():
        p = racine / name
        if p.stat().st_size != info["octets"] or sha256(p) != info["sha256"]:
            raise ValueError(f"fichier modifié pendant le transfert : {name}")
    return racine, man


def faire_parler(dossier, fmt):
    """3 réponses du chatbot, sur processeur, graines fixes : preuve qu'il se charge et répond."""
    from chatbot.generate import ChatBot
    bot = ChatBot.load(dossier, device="cpu")
    if bot.config["format"] != fmt:
        raise ValueError(f"{dossier} : format {bot.config['format']}, attendu {fmt}")
    out = []
    for i, msg in enumerate(MESSAGES[fmt]):
        r = bot.reply([{"role": "user", "content": msg}], seed=100 + i)
        out.append({"message": msg, "reponse": r["reply"], "secondes": round(r["seconds"], 2)})
        print(f"  {r['interlocutor'] or 'VOUS'} : {msg}\n  {r['persona'] or 'BOT'} : {r['reply']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archive", nargs="?", help="archive copiée à la main (ou --depuis-release)")
    p.add_argument("--depuis-release", nargs=2, metavar=("DEPOT", "TAG"), default=None,
                   help="télécharger l'archive depuis une release GitHub (gh connecté)")
    p.add_argument("--nom-release", default=None,
                   help="si la release contient plusieurs archives : début de son nom")
    p.add_argument("--installer", action="store_true",
                   help="installer les deux modèles dans data/models (ancien modèle gardé, daté)")
    p.add_argument("--nom", default=None, help="nom du rangement (défaut : rtx_<date de l'archive>)")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--stages-dir", default=str(STAGES))
    p.add_argument("--models-dir", default=str(ROOT / "data/models"))
    args = p.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if bool(args.archive) == bool(args.depuis_release):
        p.error("donner SOIT une archive, SOIT --depuis-release DEPOT TAG")
    from chatbot.pipeline_rtx import Refus, creer_dossier, gh_pret, reconstituer_depuis_release
    stages = creer_dossier(Path(args.stages_dir))
    recu = None
    if args.depuis_release:
        recu = stages / f".release_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            args.archive = str(reconstituer_depuis_release(gh_pret(), *args.depuis_release, recu,
                                                           args.nom_release))
        except Refus as e:
            shutil.rmtree(recu, ignore_errors=True)
            sys.exit(f"ERREUR : {e}")
    try:
        importer(args, stages)
    finally:
        if recu is not None:
            shutil.rmtree(recu, ignore_errors=True)  # l'archive reconstituée : déjà rangée


def importer(args, stages):
    tmp = stages / f".import_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        racine, man = extraire(args.archive, tmp)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
        shutil.rmtree(tmp, ignore_errors=True)  # une archive refusée ne laisse aucun reste
        sys.exit(f"ERREUR : archive refusée : {e}")
    if args.installer and (man.get("autotest") or not man.get("complet")
                           or man.get("taille") not in TAILLES_INSTALLABLES):
        shutil.rmtree(tmp)  # refus AVANT de ranger quoi que ce soit
        sys.exit(f"ERREUR : --installer refusé : archive d'autotest, de la recette d'essai ou "
                 f"incomplète (taille {man.get('taille')}, autotest {man.get('autotest')}, complet "
                 f"{man.get('complet')}) ; seuls les lancements complets en taille S ou M s'installent")
    prefixe = ("autotest_" if man.get("autotest") else
               "rtx_" if man.get("taille") in TAILLES_INSTALLABLES else "essai_")
    nom = args.nom or (prefixe + man["date"].replace("-", "").replace(":", "").replace("T", "_")[:13])
    dest = stages / nom
    if dest.exists():
        shutil.rmtree(tmp)
        sys.exit(f"ERREUR : {dest} existe déjà (archive déjà importée ?) : choisir --nom")
    shutil.move(str(racine), str(dest))
    shutil.rmtree(tmp)
    print(f"archive vérifiée ({len(man['fichiers'])} fichiers, taille {man.get('taille')}, "
          f"autotest {man.get('autotest')}) et rangée dans {dest}")

    runs = Path(args.runs_dir) / nom
    for rep in sorted((dest / "runs").glob("*/report.json")):
        cible = runs / rep.parent.name
        cible.mkdir(parents=True, exist_ok=True)
        for f in ("report.json", "log.jsonl"):
            if (rep.parent / f).exists():
                shutil.copy2(rep.parent / f, cible / f)
    print(f"rapports d'entraînement copiés dans {runs}")

    preuve = {"archive": str(Path(args.archive).name), "archive_sha256": sha256(args.archive),
              "manifeste": {k: man.get(k) for k in ("date", "taille", "autotest", "complet", "recettes")},
              "rangement": str(dest), "rapports": str(runs), "chatbots": {}, "installe": False}
    for d, fmt in DOSSIERS.items():
        dossier = dest / "modeles" / d
        if not (dossier / "ckpt.pt").exists():
            print(f"ATTENTION : {d} absent de l'archive")
            continue
        print(f"\n{d} ({fmt}) répond, sur processeur :")
        preuve["chatbots"][d] = faire_parler(dossier, fmt)

    if args.installer:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for d in DOSSIERS:
            src, cible = dest / "modeles" / d, Path(args.models_dir) / d
            if not src.exists():
                continue
            if cible.exists():
                ancien = cible.with_name(f"{d}.avant_{stamp}")
                shutil.move(str(cible), str(ancien))
                print(f"ancien modèle gardé : {ancien}")
            shutil.copytree(src, cible)
            faire_parler(cible, DOSSIERS[d])  # le modèle installé se charge bien
            print(f"installé : {cible}")
        preuve["installe"] = True
        print("La démo n'a pas été redémarrée : la relancer pour servir les nouveaux modèles.")
    (dest / "preuve_import.json").write_text(json.dumps(preuve, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    print(f"\npreuve : {dest / 'preuve_import.json'}")


if __name__ == "__main__":
    main()
