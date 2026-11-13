"""Lanceur unique des entraînements du chatbot sur la RTX 5090 (identique sur Windows et sur Mac).

Pourquoi un seul lanceur en Python ? Sur le PC, une session Claude Code exécute les commandes : un
fichier .bat ou .sh se comporterait différemment selon la machine. Ici, tout est en Python pur,
chaque étape est une commande EXISTANTE (chatbot.pretrain, chatbot.finetune, chatbot.generate,
chatbot.evaluate) lancée comme sous-processus avec son journal, et toutes les valeurs viennent de
chatbot/recettes_rtx.json.

    python -m chatbot.pipeline_rtx telecharger --depot X --tag Y  # clone git : gros fichiers (release)
    python -m chatbot.pipeline_rtx verifier      # données intactes ? versions ? carte ? disque ?
    python -m chatbot.pipeline_rtx autotest      # recette minuscule de bout en bout (< 5 min)
    python -m chatbot.pipeline_rtx mesurer       # débit réel des tailles S et M -> durées estimées
    python -m chatbot.pipeline_rtx lancer --taille auto [--budget-heures 4] [--micro-lot 32]
    python -m chatbot.pipeline_rtx etat          # où en est-on ? (lisible pendant un lancement)
    python -m chatbot.pipeline_rtx resultats     # RESULTATS_CHATBOT_RTX.zip à rapporter au Mac
    python -m chatbot.pipeline_rtx publier [--depot X --tag Y --publier]  # À BLANC par défaut

Deux façons d'avoir ce dossier sur le PC :
- un CLONE du dépôt git : le code vient de git, les gros fichiers (tokens, dialogues : plus de
  100 Mo, refusés par git) viennent des pièces jointes d'une release GitHub (`telecharger`,
  liste dans chatbot/donnees_rtx.json). Sorties dans runs/chatbot_rtx/ (ignoré par git).
- le PAQUET zip de secours (TRANSFER_MANIFEST.json présent) : tout est déjà dedans. Sorties dans
  sorties_rtx/.

`lancer` enchaîne : base (pré-entraînement de zéro) -> chatbot A « moliere » (affinage répliques,
puis étape synthétique facultative) -> chatbot B « francais » (affinage tirets, idem) -> exports
-> évaluations -> bilan. L'état (etat_lanceur.json) est écrit après chaque étape : relancer la
MÊME commande ne refait jamais une étape finie et reprend une étape coupée par --resume.
`--seulement moliere` (ou francais) ré-entraîne un seul chatbot sans refaire la base.

Codes de sortie : 0 tout va bien, 1 échec (journal indiqué), 2 refus (réglage, données absentes),
3 interrompu (relancer la même commande pour reprendre).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # le dossier du paquet ou du clone du dépôt
RECETTES = ROOT / "chatbot" / "recettes_rtx.json"
DONNEES_RTX = ROOT / "chatbot" / "donnees_rtx.json"  # gros fichiers à télécharger (route git)
MANIFEST = "TRANSFER_MANIFEST.json"
RESULTATS = "RESULTATS_CHATBOT_RTX"
CHATBOTS = ("moliere", "francais")  # ordre du lancement : A puis B
TAILLES = ("M", "S")                # de la plus grande à la plus petite (choix « auto »)
VERSION = "pipeline-rtx-v2"
A_RENSEIGNER = "À RENSEIGNER"       # valeur par défaut de « depot » et « tag » dans donnees_rtx.json
LIMITE_PIECE = 2_000_000_000        # GitHub refuse une pièce jointe de 2 Gio (2 147 483 648 octets) ou plus

# Ce qu'on ne met JAMAIS dans l'archive de résultats : données, états de reprise, caches.
EXCLUS = (".bin", "state.pt", ".tmp", ".npy", ".pid")

# Les gros fichiers de l'entraînement (mêmes chemins sur le Mac et sur le PC).
PRETRAIN = "data/stages/chatbot_20261113/pretrain"
DONNEES = ["data/stages/chatbot_20261112/theatre/theatre_train.jsonl.zst",
           "data/stages/chatbot_20261112/theatre/moliere_train.jsonl.zst",
           "data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst",
           "data/stages/chatbot_20261113/dialogues/romans_train.jsonl.zst",
           "data/stages/chatbot_20261113/dialogues/romans_val.jsonl.zst",
           "data/stages/chatbot_20261113/dialogues/theatre_moderne_train.jsonl.zst"]
SYNTHETIQUES = ["data/stages/chatbot_20261113/synthetique/moliere.jsonl.zst",
                "data/stages/chatbot_20261113/synthetique/moderne.jsonl.zst"]
ECHANTILLONS = {"moliere": "data/stages/chatbot_20261113/synthetique_echantillon/moliere.jsonl.zst",
                "moderne": "data/stages/chatbot_20261113/synthetique_echantillon/moderne.jsonl.zst"}

# Extraits d'essai (autotest) : petits mais assez pour 60 pas de 16 × 128 tokens.
ESSAI_TOKENS = 60_000
ESSAI_SCENES = {"theatre_train": 150, "moliere_train": 80, "moliere_val": 20, "romans_train": 300,
                "romans_val": 30, "theatre_moderne_train": 100}
ESSAI_SYN = 60


def est_paquet(root=ROOT):
    """Paquet zip (manifeste présent) ou clone du dépôt git ?"""
    return (Path(root) / MANIFEST).exists()


def sortie_defaut(root=ROOT):
    """Paquet : sorties_rtx/ (habitude du paquet). Clone : runs/chatbot_rtx/, que .gitignore
    ignore déjà (« runs ») : un `git add` ne ramasse jamais des poids ni des journaux."""
    return Path(root) / ("sorties_rtx" if est_paquet(root) else "runs/chatbot_rtx")


# ---------------------------------------------------------------------------------------------
# Petits outils


def val(node):
    """Valeur d'une entrée de recettes_rtx.json : {"valeur": ..., "pourquoi": ...} -> ..."""
    return node["valeur"] if isinstance(node, dict) and "valeur" in node else node


def charger_recettes(path=RECETTES):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_fichier(path, bloc=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bloc):
            h.update(chunk)
    return h.hexdigest()


def ecrire_json(path, obj):
    """Écriture atomique : le fichier est soit l'ancien, soit le nouveau complet (coupure de courant)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for essai in range(20):  # Windows : refusé si le fichier est ouvert ailleurs à cet instant
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if essai == 19:
                raise
            time.sleep(0.25)


def lire_json(path, defaut=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else defaut


def maintenant():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def console_utf8():
    """Windows : sorties en UTF-8 même redirigées (sinon « ≈ » ferait planter print)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


def device_auto():
    """cuda si une carte NVIDIA répond, sinon cpu. JAMAIS mps : le Mac ne fait que des essais."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def rel(path, root):
    """Chemin relatif au paquet si possible : mêmes commandes sur le Mac et sur le PC."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


class Refus(Exception):
    """Refus expliqué (réglage manquant, outil absent, fichier faux) : code de sortie 2."""


def creer_dossier(path):
    """Crée un VRAI dossier s'il manque, sans jamais remplacer un lien existant.

    Sur le Mac, data/stages, data/models et runs sont des liens vers des dossiers *.nosync : mkdir
    les traverse. Dans un clone neuf (PC), ils n'existent pas : mkdir crée de vrais dossiers. Un
    lien cassé n'est jamais effacé ni remplacé : on s'arrête avec un message.
    """
    path = Path(path)
    for p in [path, *path.parents]:
        if p.is_symlink() and not p.exists():
            raise Refus(f"lien cassé : {p} (on ne le remplace pas : recrée sa cible, ou supprime-le toi-même)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def texte_commande(cmd):
    """Commande en texte, avec « python » au lieu du chemin de l'exécutable : pas de chemin
    personnel (C:\\Users\\<nom>\\...) dans l'état, les journaux et les fichiers publiés."""
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] == sys.executable:
        cmd[0] = "python"
    return subprocess.list2cmdline(cmd)


def sans_micro(cmd_txt):
    """La commande sans --micro-batch : changer le micro-lot ne change pas le calcul."""
    return re.sub(r" --micro-batch \d+", "", cmd_txt or "")


def processus_vivant(pid):
    """Le processus `pid` tourne-t-il encore, et est-ce bien une étape Python du lanceur ?

    Jamais os.kill(pid, 0) sous Windows : là-bas, ce n'est pas un test, cela TUE le processus.
    On interroge `tasklist` (Windows) ou `ps` (Mac, Linux) et on vérifie le nom du programme,
    pour ne pas confondre avec un autre programme qui aurait reçu le même numéro plus tard.
    """
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=20).stdout
            return f'"{int(pid)}"' in out and "python" in out.lower()
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True,
                             text=True, timeout=20)
        return out.returncode == 0 and "chatbot." in out.stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def etape_orpheline(racine):
    """Étape encore vivante dont le lanceur est mort (tué seul) : (pid, fichier) ou None.

    executer() écrit journaux/<étape>.pid au démarrage de chaque étape et l'efface à la fin. Si
    le lanceur est tué net (Gestionnaire des tâches, Stop-Process, kill -9), le fichier reste et
    l'étape peut continuer à calculer : on refuse alors de lancer un second calcul sur la carte.
    Un fichier dont le processus est mort est simplement effacé.
    """
    racine = Path(racine)
    for dossier in (racine / "journaux", racine / "autotest" / "journaux"):
        for f in sorted(dossier.glob("*.pid")) if dossier.exists() else []:
            try:
                pid = int(f.read_text().strip())
            except (OSError, ValueError):
                pid = None
            if pid and processus_vivant(pid):
                return pid, f
            f.unlink(missing_ok=True)
    return None


def refuser_si_orpheline(racine):
    o = etape_orpheline(racine)
    if o:
        arret = (f"taskkill /T /F /PID {o[0]}" if os.name == "nt" else f"kill {o[0]}")
        raise Refus(f"une étape tourne encore sans son lanceur (PID {o[0]}, {o[1].name}) : un seul "
                    f"calcul à la fois. Attends qu'elle s'arrête, ou arrête-la (`{arret}`), vérifie "
                    f"avec nvidia-smi qu'aucun chatbot.pretrain / chatbot.finetune ne tourne, puis "
                    f"relance la même commande (la reprise repart du dernier state.pt).")


class Verrou:
    """Un seul lanceur à la fois : verrou du système sur un fichier.

    Pourquoi pas un simple fichier « PID » ? Si le lanceur meurt (coupure, plantage), le système
    libère le verrou tout seul : pas de faux « déjà en cours » après un redémarrage. (Sous Windows,
    os.kill(pid, 0) TUERAIT le processus visé : on ne s'en sert surtout pas.)
    """

    def __init__(self, path):
        self.path = Path(path)
        self.f = None

    def __enter__(self):
        creer_dossier(self.path.parent)
        self.f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise SystemExit(f"ERREUR : un autre lanceur tourne déjà (verrou {self.path}). Un seul "
                             f"entraînement à la fois : attends sa fin, ou regarde `etat`.")
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()


# ---------------------------------------------------------------------------------------------
# Contexte d'un lancement


class Ctx:
    """Tout ce qui décide des commandes : recettes, taille, dossiers, matériel."""

    def __init__(self, recettes, taille, sortie, root=ROOT, device="cpu", python=sys.executable,
                 micro=None):
        self.rec, self.taille, self.root = recettes, taille, Path(root)
        self.sortie = Path(sortie)
        self.device, self.python = device, python
        self.micro = micro  # micro-lot mesuré pour cette taille (None = lot entier)
        commun = recettes["commun"]
        self.precision = val(commun["precision"])
        mode = val(commun["compile"])  # "cuda" = seulement sur carte NVIDIA
        self.compile = mode == "toujours" or (mode == "cuda" and str(device).startswith("cuda"))
        self.tokenizer = val(commun["tokenizer"])
        self.graine = val(commun["graine"])

    @property
    def etat_path(self):
        return self.sortie / "etat_lanceur.json"

    def p(self, path):
        """Chemin d'un fichier du paquet (relatif) ou d'une sortie, écrit dans une commande."""
        return rel(path, self.root)

    def recette(self):
        """Recette de la taille : modèle, budget, taux, base, chatbots (essai = recette minuscule)."""
        if self.taille == "essai":
            e = self.rec["essai"]
            return {"modele": val(e["modele"]), "budget": val(e["budget_tokens"]),
                    "lr": val(e["lr"]), "min_lr": val(e["min_lr"]), "base": e["base"],
                    "chatbots": e["chatbots"]}
        t = self.rec["tailles"][self.taille]
        return {"modele": val(t["modele"]), "budget": val(t["budget_tokens"]), "lr": val(t["lr"]),
                "min_lr": val(t["min_lr"]), "base": self.rec["base"], "chatbots": self.rec["chatbots"]}


def micro_pour(lot, micro):
    """Micro-lot pour un lot donné : celui mesuré s'il divise le lot et est plus petit, sinon None."""
    return micro if micro and micro < lot and lot % micro == 0 else None


def options_materiel(ctx, lot):
    opts = ["--device", ctx.device, "--precision", ctx.precision]
    if ctx.compile:
        opts.append("--compile")
    m = micro_pour(lot, ctx.micro)
    if m:
        opts += ["--micro-batch", str(m)]
    return opts


def commande_base(ctx, out, benchmark=None, micro=None):
    """chatbot.pretrain train : pré-entraînement de zéro (ou son benchmark)."""
    r = ctx.recette()
    b = r["base"]
    commun = ctx.rec["commun"]
    lot = val(b["lot"])
    model = dict(r["modele"], context_length=val(commun["contexte"]), vocab_size=val(commun["vocabulaire"]))
    cmd = [ctx.python, "-m", "chatbot.pretrain", "train",
           "--data", ctx.p(ctx.root / val(b["donnees"])),
           "--weights", *[f"{k}={v}" for k, v in val(b["parts"]).items()],
           "--model-config", json.dumps(model, separators=(",", ":")),
           "--tokenizer", ctx.p(ctx.root / ctx.tokenizer),
           "--tokens", str(r["budget"]), "--batch-size", str(lot),
           "--seq-len", str(val(b["seq_len"])), "--lr", str(r["lr"]), "--min-lr", str(r["min_lr"]),
           "--warmup-steps", str(val(b["echauffement"])), "--weight-decay", str(val(b["weight_decay"])),
           "--grad-clip", str(val(b["grad_clip"])), "--eval-every", str(val(b["eval_tous_les"])),
           "--seed", str(ctx.graine)]
    if benchmark:
        saved, ctx.micro = ctx.micro, micro
        cmd += ["--benchmark-steps", str(benchmark)] + options_materiel(ctx, lot)
        ctx.micro = saved
        return cmd
    return cmd + ["--val", ctx.p(ctx.root / val(b["validation"])), "--out", ctx.p(out),
                  "--note", f"lanceur RTX, taille {ctx.taille}"] + options_materiel(ctx, lot) + ["--resume"]


def commande_affinage(ctx, etape, fmt, validation, donnees, init, out, note):
    """chatbot.finetune : un affinage (étape 1 ou synthétique), toujours avec --resume."""
    lot = val(etape["lot"])
    seq = val(etape.get("seq_len", ctx.rec["commun"]["contexte"]))
    poids = val(etape["poids"])
    return [ctx.python, "-m", "chatbot.finetune", "--init", ctx.p(init),
            "--tokenizer", ctx.p(ctx.root / ctx.tokenizer),
            "--data", *[ctx.p(ctx.root / d) for d in donnees], "--weights", *map(str, poids),
            "--val", ctx.p(ctx.root / validation), "--out", ctx.p(out),
            "--tokens", str(val(etape["tokens"])), "--lr", str(val(etape["lr"])),
            "--min-lr", str(val(etape["min_lr"])), "--warmup-steps", str(val(etape["echauffement"])),
            "--batch-size", str(lot), "--seq-len", str(seq), "--eval-every", str(val(etape["eval_tous_les"])),
            "--persona-augment", str(val(etape.get("persona_augment", 0.0))), "--format", fmt,
            "--seed", str(ctx.graine), "--note", note] + options_materiel(ctx, lot) + ["--resume"]


def avec_synthetique(ctx, etat, cb, syn):
    """L'étape synthétique de ce chatbot fait-elle partie du plan ?

    La composition de chaque chatbot est FIGÉE dans etat["composition"] avant sa première étape
    (voir figer_composition). Pourquoi ? Si on la recalculait à chaque relance, une coupure
    pendant l'étape 1 (la plus longue) ferait, selon le cas, disparaître une étape synthétique
    prévue (export de l'étape 1, part synthétique absente du bilan, sans prévenir) ou apparaître
    une étape dont le fichier est arrivé en cours de route. Pas encore figée : le fichier décide.
    `--seulement ... --repartir` (archiver) efface la composition du chatbot refait.
    """
    compo = ((etat or {}).get("composition") or {}).get(cb)
    if compo is not None:
        return bool(syn) and f"{cb}_synthetique" in compo
    return bool(syn) and (ctx.root / val(syn["fichier"])).exists()


def figer_composition(etat, etapes):
    """Note, pour chaque chatbot pas encore noté, la liste de ses étapes d'entraînement."""
    compo = etat.setdefault("composition", {})
    for cb in CHATBOTS:
        noms = [e["nom"] for e in etapes if e["groupe"] == cb and e["type"] == "entrainement"]
        if noms and cb not in compo:
            compo[cb] = noms


def plan(ctx, etat=None):
    """Liste ordonnée des étapes. Chaque étape : nom, groupe, type, commande, sortie attendue.

    L'étape synthétique d'un chatbot n'existe que si son fichier est dans le paquet ; une fois le
    lancement commencé, la composition de chaque chatbot est figée (avec_synthetique).
    """
    r = ctx.recette()
    s = ctx.sortie
    base = s / "runs" / "base"
    etapes = [{"nom": "base", "groupe": "base", "type": "entrainement",
               "cmd": commande_base(ctx, base), "out": base, "attendu": base / "report.json"}]
    finals = {}
    for cb in CHATBOTS:
        c = r["chatbots"][cb]
        fmt, validation = val(c["format"]), val(c["validation"])
        out1 = s / "runs" / f"{cb}_etape1"
        e1 = c["etape1"]
        etapes.append({"nom": f"{cb}_etape1", "groupe": cb, "type": "entrainement",
                       "cmd": commande_affinage(ctx, e1, fmt, validation, val(e1["donnees"]),
                                                base / "best.pt", out1, f"lanceur RTX, {cb}, étape 1"),
                       "out": out1, "attendu": out1 / "report.json"})
        finals[cb] = out1
        syn = c.get("synthetique")
        nom_syn = f"{cb}_synthetique"
        if avec_synthetique(ctx, etat, cb, syn):
            out2 = s / "runs" / nom_syn
            donnees = [val(syn["fichier"])] + val(syn["avec"])  # le synthétique en premier
            etapes.append({"nom": nom_syn, "groupe": cb, "type": "entrainement",
                           "cmd": commande_affinage(ctx, syn, fmt, validation, donnees,
                                                    out1 / "best.pt", out2,
                                                    f"lanceur RTX, {cb}, étape synthétique"),
                           "out": out2, "attendu": out2 / "report.json"})
            finals[cb] = out2
    for cb in CHATBOTS:
        c = r["chatbots"][cb]
        dossier = s / "modeles" / val(c["dossier_modele"])
        etapes.append({"nom": f"{cb}_export", "groupe": cb, "type": "export",
                       "cmd": [ctx.python, "-m", "chatbot.generate", "export",
                               "--ckpt", ctx.p(finals[cb] / "best.pt"),
                               "--tokenizer", ctx.p(ctx.root / ctx.tokenizer), "--out", ctx.p(dossier),
                               "--format", val(c["format"])] + list(val(c.get("export", []))),
                       "out": dossier, "attendu": dossier / "chat_config.json"})
    for cb in CHATBOTS:
        c = r["chatbots"][cb]
        dossier = s / "modeles" / val(c["dossier_modele"])
        out = s / "evaluations" / cb
        etapes.append({"nom": f"{cb}_evaluation", "groupe": cb, "type": "evaluation",
                       "cmd": [ctx.python, "-m", "chatbot.evaluate", "--model", ctx.p(dossier),
                               "--out", ctx.p(out), "--device", ctx.device,
                               "--val", ctx.p(ctx.root / val(c["validation"]))]
                       + list(val(c.get("evaluation", []))),
                       "out": out, "attendu": out / "metrics.json"})
    return etapes


def donnees_manquantes(ctx):
    """Fichiers indispensables au plan absents du paquet (paquet léger, transfert incomplet)."""
    r = ctx.recette()
    need = [ctx.root / ctx.tokenizer, ctx.root / val(r["base"]["validation"])]
    data = ctx.root / val(r["base"]["donnees"])
    meta = lire_json(data / "meta.json")
    if meta is None:
        need.append(data / "meta.json")
    else:
        for name in val(r["base"]["parts"]):
            info = meta["sources"].get(name)
            need.append(data / (info["file"] if info else f"{name}.bin"))
    for cb in CHATBOTS:
        c = r["chatbots"][cb]
        need += [ctx.root / d for d in val(c["etape1"]["donnees"])] + [ctx.root / val(c["validation"])]
    return list(dict.fromkeys(ctx.p(p) for p in need if not Path(p).exists()))  # sans doublon


def _lire_docs(path, n):
    import zstandard
    out = []
    with zstandard.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(line)
            if len(out) >= n:
                break
    return out


def _ecrire_docs(path, lines):
    import zstandard
    creer_dossier(Path(path).parent)
    with zstandard.open(path, "wt", encoding="utf-8") as f:
        f.writelines(lines)


def preparer_donnees_essai(root, dest, ignorer_git=False):
    """Extraits minuscules des vraies données, dans <dest>/donnees_essai/ (pour l'autotest).

    Même fonction pour le paquet zip (tools/build_chatbot_bundle.py) et pour un clone (après
    `telecharger`) : l'autotest voit exactement les mêmes extraits. Renvoie {style: nb de lignes}
    des extraits synthétiques écrits (si les échantillons synthétiques sont présents).
    """
    import hashlib as _h
    import numpy as np
    root, dest = Path(root), Path(dest)
    meta = json.loads((root / PRETRAIN / "meta.json").read_text(encoding="utf-8"))
    out = {"tokenizer_sha256": meta["tokenizer_sha256"], "sources": {},
           "note": f"extraits d'essai : {ESSAI_TOKENS} premiers tokens de chaque source de {PRETRAIN}"}
    creer_dossier(dest / "donnees_essai/pretrain")
    if ignorer_git:  # clone : ces extraits ne doivent jamais apparaître dans `git status`
        (dest / "donnees_essai/.gitignore").write_text("*\n", encoding="utf-8")
    for name, info in meta["sources"].items():
        arr = np.asarray(np.memmap(root / PRETRAIN / info["file"], dtype=np.uint16, mode="r")[:ESSAI_TOKENS])
        (dest / f"donnees_essai/pretrain/{name}.bin").write_bytes(arr.tobytes())
        key = _h.sha256(json.dumps(info["key"], sort_keys=True).encode()).hexdigest()[:16]
        out["sources"][name] = {"file": f"{name}.bin", "tokens": int(len(arr)),
                                "key": {"extrait_essai_de": name, "premiers_tokens": int(len(arr)),
                                        "cle_source": key}}
    (dest / "donnees_essai/pretrain/meta.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for src in DONNEES:
        stem = Path(src).name.replace(".jsonl.zst", "")
        sub = "theatre" if "theatre/" in src else "dialogues"
        _ecrire_docs(dest / f"donnees_essai/{sub}/{Path(src).name}", _lire_docs(root / src, ESSAI_SCENES[stem]))
    syn = {}
    for style, src in ECHANTILLONS.items():
        if (root / src).exists():
            lines = _lire_docs(root / src, ESSAI_SYN)
            _ecrire_docs(dest / f"donnees_essai/synthetique/{style}.jsonl.zst", lines)
            syn[style] = len(lines)
    return syn


def assurer_donnees_essai(root=ROOT):
    """Clone : fabrique donnees_essai/ depuis les vraies données si elles sont là et lui absent."""
    root = Path(root)
    if est_paquet(root) or (root / "donnees_essai/pretrain/meta.json").exists():
        return
    if not (root / PRETRAIN / "meta.json").exists() or not all((root / d).exists() for d in DONNEES):
        return  # pas encore téléchargé : donnees_manquantes() le dira
    syn = preparer_donnees_essai(root, root, ignorer_git=True)
    print(f"extraits d'essai fabriqués dans donnees_essai/ (synthétiques : {syn or 'aucun'})", flush=True)


# ---------------------------------------------------------------------------------------------
# Exécution d'une commande et de l'état


ENFANT = None  # sous-processus en cours (pour lui relayer un SIGTERM venu d'ailleurs)


def arret_demande(signum, frame):
    """SIGTERM / Ctrl-Break reçu par le lanceur : même conduite que Ctrl-C.

    Sans ce gestionnaire, Ctrl-Break (Windows) ou SIGTERM (Mac, Linux) TUENT le lanceur aussitôt :
    l'étape en cours, qui a reçu le même signal et veut sauver state.pt, écrit alors son message
    dans le tube d'un lanceur mort et peut mourir sans sauver. Ici, on lève KeyboardInterrupt :
    executer() attend que l'étape ait fini son pas et sauvé son état, puis le lanceur rend le code 3.
    - Ctrl-Break (et Ctrl-C) dans la console Windows : TOUS les processus de la console le reçoivent,
      l'étape aussi : rien à relayer.
    - SIGTERM (Mac, Linux, par exemple `kill <pid du lanceur>`) : seul le lanceur le reçoit, on le
      relaie donc à l'étape. Jamais sous Windows : là, send_signal(SIGTERM) TUERAIT l'étape sans
      sauvegarde (TerminateProcess).
    """
    nom = signal.Signals(signum).name
    if nom == "SIGTERM" and os.name != "nt" and ENFANT is not None and ENFANT.poll() is None:
        ENFANT.send_signal(signal.SIGTERM)
    raise KeyboardInterrupt(nom)


def installer_arrets():
    for nom in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, nom):
            signal.signal(getattr(signal, nom), arret_demande)


def executer(cmd, journal, root=ROOT):
    """Lance une commande, recopie sa sortie à l'écran ET dans son journal ; renvoie son code.

    Ctrl-C : l'enfant le reçoit aussi (même console), finit son pas et sauve state.pt ; on attend
    donc sa fin au lieu de l'abandonner. Renvoie (code, interrompu_par_l_utilisateur).
    """
    global ENFANT
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    creer_dossier(Path(journal).parent)
    pidfile = Path(journal).with_suffix(".pid")  # lu par etape_orpheline() si le lanceur meurt
    stop = False
    with open(journal, "a", encoding="utf-8") as log:
        log.write(f"\n===== {maintenant()} : {texte_commande(cmd)}\n")
        log.flush()
        proc = ENFANT = subprocess.Popen([str(c) for c in cmd], cwd=root, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        pidfile.write_text(str(proc.pid))
        try:
            while True:
                try:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    proc.wait()
                    break
                except KeyboardInterrupt as e:
                    stop = True
                    print(f"\n{e or 'Ctrl-C'} : on attend que l'étape sauve son état (state.pt)…",
                          flush=True)
        finally:
            ENFANT = None
            if proc.poll() is not None:  # l'étape est finie : plus d'orpheline possible
                pidfile.unlink(missing_ok=True)
    return proc.returncode, stop


def lire_etat(ctx):
    return lire_json(ctx.etat_path, None)


def nouvel_etat(ctx):
    return {"lanceur": VERSION, "recettes": val(ctx.rec["version"]), "taille": ctx.taille,
            "device": ctx.device, "precision": ctx.precision, "compile": ctx.compile,
            "micro_lot": ctx.micro, "cree": maintenant(), "etapes": {}}


def dernieres_lignes(path, n=20):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


MESSAGES = {
    "divergence": "divergence (loss non finie) : baisse le « lr » de cette étape dans "
                  "chatbot/recettes_rtx.json, puis relance avec --seulement <chatbot> --repartir "
                  "(pour la base : un nouveau dossier, p. ex. `python -m chatbot.pipeline_rtx "
                  "--sortie runs/chatbot_rtx_2 lancer --taille <même taille>`, puis la même "
                  "--sortie pour etat, resultats et publier). Demande d'abord l'accord de l'étudiant",
    "erreur": "erreur du programme : lis les dernières lignes ci-dessous et le journal ; ne relance "
              "qu'après avoir compris (une relance reprendrait au dernier state.pt)",
    3: "interrompu proprement (état sauvé) : relance la MÊME commande pour reprendre",
}


def derniere_execution(journal):
    """Texte du journal depuis le dernier lancement de l'étape (les reprises s'y ajoutent)."""
    try:
        texte = Path(journal).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return texte.rsplit("\n===== ", 1)[-1]


def diagnostic(code, journal, cmd, taille):
    """(statut, message) d'une étape qui n'a pas fini normalement, d'après son journal.

    - « ARRÊT : loss ... non finis » : vraie divergence -> baisser le lr (et seulement là, --repartir) ;
    - « out of memory » : mémoire de la carte -> même commande avec un micro-lot deux fois plus petit ;
    - Traceback / ERREUR : erreur du programme -> lire le journal ;
    - rien de tout cela : processus tué de l'extérieur (Gestionnaire des tâches, Stop-Process,
      kill) -> relancer la MÊME commande reprend au dernier state.pt. Surtout pas --repartir.
    """
    texte = derniere_execution(journal)
    if "non finis" in texte:
        return "echec", MESSAGES["divergence"]
    if "out of memory" in texte.lower() or "OutOfMemoryError" in texte:
        cmd = [str(c) for c in cmd]
        flag = "--micro-batch" if "--micro-batch" in cmd else "--batch-size"
        actuel = int(cmd[cmd.index(flag) + 1]) if flag in cmd else 0
        conseil = max(actuel // 2, 1)
        return "echec", (f"mémoire de la carte insuffisante (lots de {actuel} séquences) : relance la "
                         f"même commande avec --micro-lot {conseil} (`lancer --taille {taille} "
                         f"--micro-lot {conseil}`). Même calcul (accumulation de gradient), moins de "
                         f"mémoire ; la reprise repart du dernier state.pt")
    if "Traceback" in texte or "ERREUR" in texte:
        return "echec", MESSAGES["erreur"]
    return "interrompu", (f"processus arrêté de l'extérieur (code {code}, ni erreur ni divergence "
                          f"dans le journal) : relance la MÊME commande pour reprendre au dernier "
                          f"state.pt. N'utilise PAS --repartir : il jetterait le travail déjà fait")


def lancer_etapes(ctx, etapes, runner=executer, interrompre=None, plan_complet=None):
    """Exécute les étapes dans l'ordre ; une étape finie n'est jamais refaite.

    interrompre = {"base": 30} : ajoute --stop-at-step 30 à cette étape (autotest : coupure simulée).
    plan_complet : noms de TOUTES les étapes du lancement (avec --seulement, on n'exécute qu'un
    chatbot, mais `etat` et `resultats` doivent toujours voir le plan entier : sinon un lancement
    inachevé passerait pour complet).
    Renvoie 0 (tout fini), 1 (échec) ou 3 (interrompu).
    """
    etat = lire_etat(ctx) or nouvel_etat(ctx)
    etat["plan"] = list(plan_complet or [e["nom"] for e in etapes])
    figer_composition(etat, etapes)  # écrit AVANT la première étape, jamais modifié ensuite
    ecrire_json(ctx.etat_path, etat)
    for e in etapes:
        st = etat["etapes"].get(e["nom"], {})
        cmd_txt = texte_commande(e["cmd"])
        if st.get("statut") == "fini":
            if sans_micro(st.get("commande")) != sans_micro(cmd_txt):
                print(f"[{e['nom']}] déjà fait (la recette a changé depuis : pas refait ; "
                      f"--seulement pour refaire un chatbot)", flush=True)
            else:
                print(f"[{e['nom']}] déjà fait, on passe", flush=True)
            continue
        # Fini mais pas noté (lanceur coupé juste après) : on le constate au lieu de le refaire.
        if e["type"] == "entrainement" and Path(e["attendu"]).exists() and not (Path(e["out"]) / "state.pt").exists():
            st.update(statut="fini", commande=cmd_txt, fin=maintenant(), note="constaté au redémarrage")
            etat["etapes"][e["nom"]] = dict(st, groupe=e["groupe"])
            ecrire_json(ctx.etat_path, etat)
            print(f"[{e['nom']}] déjà terminé (report.json présent)", flush=True)
            continue
        cmd = list(e["cmd"])
        if interrompre and e["nom"] in interrompre:
            cmd += ["--stop-at-step", str(interrompre[e["nom"]])]
        journal = ctx.sortie / "journaux" / f"{e['nom']}.log"
        reprise = (Path(e["out"]) / "state.pt").exists()
        st.update(groupe=e["groupe"], statut="en_cours", commande=cmd_txt, journal=ctx.p(journal),
                  debut=maintenant(), reprises=st.get("reprises", 0) + int(reprise))
        st.pop("message", None)
        etat["etapes"][e["nom"]] = st
        ecrire_json(ctx.etat_path, etat)
        print(f"\n[{e['nom']}] {'REPRISE' if reprise else 'début'} — journal {ctx.p(journal)}", flush=True)
        t0 = time.time()
        code, stop = runner(cmd, journal, ctx.root)
        st.update(code=code, fin=maintenant(),
                  secondes=round(st.get("secondes", 0) + time.time() - t0, 1))
        if code == 0 and Path(e["attendu"]).exists() and not stop:
            st["statut"] = "fini"
        elif code == 3 or stop:
            st["statut"] = "interrompu"
            st["message"] = MESSAGES[3]
        elif code == 0:
            st["statut"] = "echec"
            st["message"] = f"sortie attendue absente : {ctx.p(e['attendu'])}"
        else:
            st["statut"], st["message"] = diagnostic(code, journal, cmd, ctx.taille)
        ecrire_json(ctx.etat_path, etat)
        if st["statut"] != "fini":
            print(f"\nARRÊT à l'étape {e['nom']} ({st['statut']}, code {code}) : {st['message']}",
                  flush=True)
            print(f"Journal : {ctx.p(journal)}", flush=True)
            if st["statut"] == "echec":
                print("Dernières lignes :\n  " + "\n  ".join(dernieres_lignes(journal)), flush=True)
            return 3 if st["statut"] == "interrompu" else 1
        print(f"[{e['nom']}] fini en {st['secondes'] / 60:.1f} min", flush=True)
    bilan(ctx, etat)
    return 0


# ---------------------------------------------------------------------------------------------
# Bilan : rapport d'entraînement de chaque modèle exporté


def resume_run(report):
    if not report:
        return None
    best, first = report.get("best", {}), report.get("val_step0") or {}
    return {"tokens_vus": report.get("tokens_seen"), "pas": report.get("steps"),
            "bpb_validation_pas0": first.get("val_bpb"), "meilleur_bpb_validation": best.get("val_bpb"),
            "meilleur_pas": best.get("step"), "secondes": report.get("seconds"),
            "reprises_aux_pas": report.get("resumed_from_steps"), "precision": report.get("precision"),
            "micro_lot": report.get("micro_batch"), "memoire_gpu_max_go": report.get("peak_memory_gb"),
            "n_params": report.get("n_params")}


def part_synthetique(rep_syn, rep_1):
    """Part des tokens synthétiques : dans l'étape synthétique, et dans tout l'affinage."""
    if not rep_syn:
        return None
    data = rep_syn["data"]
    syn = data[0]["tokens_trained"]  # le fichier synthétique est le premier de --data
    total_etape = sum(d["tokens_trained"] for d in data)
    total = total_etape + (rep_1 or {}).get("tokens_seen", 0)
    return {"fichier": data[0]["path"], "tokens_synthetiques": syn,
            "part_dans_l_etape": round(syn / max(total_etape, 1), 4),
            "part_de_tout_l_affinage": round(syn / max(total, 1), 4)}


def bilan(ctx, etat):
    """Écrit training_report.json dans chaque dossier modèle et sorties/bilan.json."""
    s = ctx.sortie
    base = lire_json(s / "runs" / "base" / "report.json")
    resume = {"taille": ctx.taille, "recettes": val(ctx.rec["version"]), "lanceur": VERSION,
              "base": resume_run(base), "chatbots": {},
              "durees_minutes": {k: round(v.get("secondes", 0) / 60, 1) for k, v in etat["etapes"].items()},
              "duree_totale_heures": round(sum(v.get("secondes", 0) for v in etat["etapes"].values()) / 3600, 2)}
    for cb in CHATBOTS:
        c = ctx.recette()["chatbots"][cb]
        dossier = s / "modeles" / val(c["dossier_modele"])
        r1 = lire_json(s / "runs" / f"{cb}_etape1" / "report.json")
        rs = lire_json(s / "runs" / f"{cb}_synthetique" / "report.json")
        m = lire_json(s / "evaluations" / cb / "metrics.json")
        panel = (m or {}).get("panel", {})
        ev = None if m is None else {
            "bpb_tour_suivant": m.get("val_next_turn", {}).get("val_bpb"),
            "panel_commun": panel.get("commun_sans_harpagon") or panel.get("tout"),
            "discussion_6_tours": m.get("discussion_6_tours")}
        info = {"format": val(c["format"]), "taille": ctx.taille, "modele": ctx.recette()["modele"],
                "base": resume_run(base), "etape1": resume_run(r1), "synthetique": resume_run(rs),
                "part_synthetique": part_synthetique(rs, r1), "evaluation": ev,
                "environnement": (base or {}).get("environment"),
                "note": "modèle de démo hors concours, pré-entraîné de zéro sur nos données, aucun poids externe"}
        resume["chatbots"][cb] = info
        if (dossier / "chat_config.json").exists():
            ecrire_json(dossier / "training_report.json", info)
    ecrire_json(s / "bilan.json", resume)
    print("\nBILAN")
    if base:
        print(f"  base {ctx.taille} : BPB Molière {resume['base']['bpb_validation_pas0']} -> "
              f"{resume['base']['meilleur_bpb_validation']} ({resume['base']['tokens_vus']:,} tokens)")
    for cb, info in resume["chatbots"].items():
        last = info["synthetique"] or info["etape1"] or {}
        ev = info["evaluation"] or {}
        print(f"  {cb} : BPB validation {last.get('bpb_validation_pas0')} -> "
              f"{last.get('meilleur_bpb_validation')} ; évaluation BPB {ev.get('bpb_tour_suivant')}"
              + (f" ; part synthétique {info['part_synthetique']['part_de_tout_l_affinage']:.1%}"
                 if info["part_synthetique"] else ""))
    print(f"  durée totale : {resume['duree_totale_heures']} h")
    return resume


# ---------------------------------------------------------------------------------------------
# verifier


def versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "tokenizers", "regex", "zstandard", "xxhash"):
        try:
            out[mod] = metadata.version(mod)
        except metadata.PackageNotFoundError:
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
    except ImportError:
        out["cuda"] = None
    return out


def carte():
    """Nom, mémoire, bf16 et un VRAI calcul sur la carte (is_available ne suffit pas)."""
    import torch
    if not torch.cuda.is_available():
        return None
    from chatbot.accel import bf16_native
    p = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if bf16_native() else torch.float16
    x = torch.randn(256, 256, device="cuda", dtype=dtype, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(x.grad.float()).all())
    return {"nom": p.name, "memoire_go": round(p.total_memory / 1e9, 1),
            "capacite": f"{p.major}.{p.minor}", "bf16_natif": bf16_native(), "calcul_test_ok": ok}


def autres_processus_gpu():
    """Autres programmes sur la carte (nvidia-smi), pour respecter « un calcul à la fois »."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def verifier_manifeste(root=ROOT, progress=True):
    """(manifeste, erreurs) : chaque fichier listé doit exister avec la même taille et le même sha256."""
    path = Path(root) / MANIFEST
    if not path.exists():
        return None, [f"{MANIFEST} absent : ce dossier n'est pas un paquet construit par "
                      f"tools/build_chatbot_bundle.py"]
    man = json.loads(path.read_text(encoding="utf-8"))
    erreurs = []
    for name, info in sorted(man["fichiers"].items()):
        f = Path(root) / name
        if not f.is_file():
            erreurs.append(f"manquant : {name}")
            continue
        if f.stat().st_size != info["octets"]:
            erreurs.append(f"taille différente : {name}")
            continue
        if progress and info["octets"] > 200_000_000:
            print(f"  empreinte de {name} ({info['octets'] / 1e9:.2f} Go)…", flush=True)
        if sha256_fichier(f) != info["sha256"]:
            erreurs.append(f"contenu modifié (sha256) : {name}")
    return man, erreurs


def verifier_donnees(donnees, root=ROOT, progress=True):
    """Clone : (nb de fichiers, erreurs) des gros fichiers de donnees_rtx.json (taille, sha256)."""
    erreurs = []
    for f in donnees["fichiers"]:
        p = Path(root) / chemin_sur(f["destination"])
        if not p.is_file():
            erreurs.append(f"manquant : {f['destination']} (lancer `telecharger`)")
            continue
        if p.stat().st_size != f["octets"]:
            erreurs.append(f"taille différente : {f['destination']} (relancer `telecharger`)")
            continue
        if progress and f["octets"] > 200_000_000:
            print(f"  empreinte de {f['destination']} ({f['octets'] / 1e9:.2f} Go)…", flush=True)
        if sha256_fichier(p) != f["sha256"]:
            erreurs.append(f"contenu modifié (sha256) : {f['destination']} (relancer `telecharger`)")
    return len(donnees["fichiers"]), erreurs


def commit_git(root=ROOT):
    """(commit courant, nb de fichiers suivis modifiés) ou (None, None) hors dépôt git."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=20)
        if head.returncode:
            return None, None
        modif = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=60)
        return head.stdout.strip(), len([l for l in modif.stdout.splitlines() if l.strip()])
    except (OSError, subprocess.SubprocessError):
        return None, None


def cmd_verifier(args, rec, root=ROOT):
    racine = Path(args.sortie)
    erreurs, alertes = [], []
    man, commit = None, None
    if est_paquet(root):
        print("1/4 Intégrité du paquet (sha256 de chaque fichier)…", flush=True)
        man, err = verifier_manifeste(root)
        erreurs += err
        if man is not None:
            print(f"  {len(man['fichiers'])} fichiers contrôlés, {len(err)} problème(s)")
            if not man.get("complet"):
                alertes.append("paquet LÉGER (sans les vraies données) : autotest possible, lancer impossible")
    else:
        print("1/4 Clone du dépôt : commit, puis gros fichiers (taille et sha256)…", flush=True)
        commit, modifies = commit_git(root)
        print(f"  commit {commit or 'inconnu (pas de git ?)'}"
              + (f", {modifies} fichier(s) suivi(s) modifié(s) localement" if modifies else ""))
        if modifies:
            alertes.append(f"{modifies} fichier(s) suivi(s) modifié(s) depuis le commit {commit} "
                           f"(`git status` pour les voir)")
        donnees = lire_json(getattr(args, "donnees", None) or DONNEES_RTX)
        if donnees is None:
            erreurs.append("chatbot/donnees_rtx.json absent : impossible de contrôler les données")
        else:
            n, err = verifier_donnees(donnees, root)
            erreurs += err
            print(f"  {n} gros fichiers contrôlés, {len(err)} problème(s)")
        tok = Path(root) / val(rec["commun"]["tokenizer"])
        meta = lire_json(Path(root) / PRETRAIN / "meta.json")
        if not tok.exists():
            erreurs.append(f"tokenizer absent : {val(rec['commun']['tokenizer'])}")
        elif meta and meta.get("tokenizer_sha256") != sha256_fichier(tok):
            erreurs.append("le tokenizer ne correspond pas à celui des tokens de pré-entraînement "
                           "(cause fréquente sous Windows : fins de ligne converties en CRLF au clone ; "
                           "remède : recloner avec `git clone -c core.autocrlf=false ...`)")
    # Le code se charge-t-il avec les SEULS fichiers du dossier ? (un fichier oublié se voit ici,
    # pas au milieu de la nuit)
    test = subprocess.run([sys.executable, "-c", "import chatbot.pretrain, chatbot.finetune, "
                           "chatbot.generate, chatbot.evaluate"], cwd=root, capture_output=True,
                          text=True, env=dict(os.environ, PYTHONPATH=str(root)))
    if test.returncode:
        erreurs.append("le code ne se charge pas : " + (test.stderr.strip().splitlines() or ["?"])[-1])
    else:
        print("  le code se charge (pretrain, finetune, generate, evaluate)")
    print("2/4 Versions…", flush=True)
    v = versions()
    attendu = val(rec["commun"]["versions_attendues"])
    for k, want in attendu.items():
        got = v.get(k)
        if got is None and k != "cuda":
            erreurs.append(f"module absent : {k} (voir l'étape « installer » de PC_RTX.md)")
        elif got is not None and not str(got).split("+")[0].startswith(want):
            alertes.append(f"version {k} : {got} (attendu {want})")
    print("  " + ", ".join(f"{k} {x}" for k, x in v.items()))
    print("3/4 Carte graphique…", flush=True)
    gpu = None
    try:
        gpu = carte()
    except Exception as e:  # noqa: BLE001 - message clair plutôt qu'une trace
        erreurs.append(f"calcul de test sur la carte impossible : {type(e).__name__}: {e}")
    if gpu:
        print(f"  {gpu['nom']}, {gpu['memoire_go']} Go, bf16 natif {gpu['bf16_natif']}, "
              f"calcul de test {'OK' if gpu['calcul_test_ok'] else 'ÉCHEC'}")
        if not gpu["calcul_test_ok"]:
            erreurs.append("le calcul de test sur la carte donne des valeurs non finies")
    elif args.sans_gpu:
        alertes.append("pas de carte NVIDIA (--sans-gpu) : essais sur processeur seulement")
    elif v.get("torch"):
        erreurs.append("aucune carte NVIDIA visible par PyTorch (pilote ? PyTorch installé sans "
                       "CUDA ?). Sur le Mac d'essai : ajouter --sans-gpu")
    autres = autres_processus_gpu()
    if autres:
        alertes.append(f"autres programmes sur la carte : {autres} (un seul calcul à la fois)")
    print("4/4 Espace disque…", flush=True)
    libre = shutil.disk_usage(root).free / 1e9
    besoin = val(rec["commun"]["espace_disque_min_go"])
    print(f"  {libre:.1f} Go libres (minimum {besoin} Go)")
    if libre < besoin:
        erreurs.append(f"espace disque insuffisant : {libre:.1f} Go < {besoin} Go")
    rapport = {"date": maintenant(), "mode": "paquet" if est_paquet(root) else "clone",
               "paquet": (man or {}).get("paquet"), "commit": commit,
               "complet": (man or {}).get("complet") if man else not any("manquant" in e for e in erreurs),
               "versions": v, "carte": gpu, "disque_libre_go": round(libre, 1), "erreurs": erreurs,
               "alertes": alertes, "ok": not erreurs}
    ecrire_json(racine / "verification.json", rapport)
    for a in alertes:
        print(f"ATTENTION : {a}")
    for e in erreurs:
        print(f"ERREUR : {e}")
    print("VÉRIFICATION " + ("RÉUSSIE" if not erreurs else "ÉCHOUÉE") + f" ({racine / 'verification.json'})")
    return 0 if not erreurs else 1


# ---------------------------------------------------------------------------------------------
# mesurer


BENCH = re.compile(r"benchmark : (\d+) pas en ([\d.]+) s = ([\d,  ]+) tokens/s")


def lire_benchmark(texte):
    """Ce qu'affiche `chatbot.pretrain train --benchmark-steps` -> dict, ou None si absent."""
    m = None
    for m in BENCH.finditer(texte):
        pass  # on garde la dernière mesure du journal
    if m is None:
        return None
    tail = texte[m.start():].splitlines()[0]
    mem = re.search(r"mémoire GPU max ([\d.]+) Go", tail)
    prec = re.search(r"précision (\w+)", tail)
    comp = re.search(r"compile ([^|]+)", tail)
    return {"pas": int(m.group(1)), "secondes": float(m.group(2)),
            "tokens_par_s": float(re.sub(r"[^\d]", "", m.group(3))),
            "memoire_gpu_max_go": float(mem.group(1)) if mem else None,
            "precision": prec.group(1) if prec else None,
            "compile": comp.group(1).strip() if comp else None}


def estimer(ctx_taille, tps):
    """Durées estimées (heures) de chaque étape d'entraînement d'une taille, au débit mesuré."""
    rec = ctx_taille.rec
    r = ctx_taille.recette()
    h = {"base": r["budget"] / tps / 3600}
    for cb in CHATBOTS:
        c = r["chatbots"][cb]
        h[f"{cb}_etape1"] = val(c["etape1"]["tokens"]) / tps / 3600
        syn = c.get("synthetique")
        if syn and (ctx_taille.root / val(syn["fichier"])).exists():
            h[f"{cb}_synthetique"] = val(syn["tokens"]) / tps / 3600
    marge = val(rec["commun"]["marge_estimation"])
    forfait = val(rec["commun"]["forfait_evaluation_heures"])
    total = sum(h.values()) * marge + forfait * len(CHATBOTS)
    return {"etapes_heures": {k: round(v, 2) for k, v in h.items()},
            "marge": marge, "forfait_evaluations_heures": forfait * len(CHATBOTS),
            "total_heures": round(total, 2)}


def mesurer(rec, racine, device, tailles=("S", "M"), runner=executer, root=ROOT):
    """Benchmark de chaque taille ; essaie des micro-lots plus petits si la mémoire déborde."""
    res = {"date": maintenant(), "device": device, "tailles": {},
           "note": "durées = budget de tokens / débit mesuré, × marge, + forfait d'évaluation non "
                   "mesuré ; finetune supposé au même débit que le pré-entraînement (même modèle)"}
    for taille in tailles:
        ctx = Ctx(rec, taille, racine, root, device)
        lot = val(ctx.recette()["base"]["lot"])
        mesure = None
        # Lot entier d'abord, puis les micro-lots plus petits qui divisent le lot.
        essais = sorted({lot} | {m for m in val(rec["commun"]["micro_lots_essayes"])
                                 if m < lot and lot % m == 0}, reverse=True)
        for micro in essais:
            journal = racine / "journaux" / f"mesure_{taille}_micro{micro}.log"
            if journal.exists():
                journal.unlink()  # un journal par essai : on y relit le débit
            cmd = commande_base(ctx, None, benchmark=val(rec["commun"]["pas_benchmark"]),
                                micro=micro if micro < lot else None)
            print(f"\n[mesure {taille}] lot {lot}, micro-lot {micro}", flush=True)
            code, stop = runner(cmd, journal, root)
            if stop:
                return None
            mesure = lire_benchmark(journal.read_text(encoding="utf-8", errors="replace")) if journal.exists() else None
            if code == 0 and mesure:
                mesure["micro_lot"] = micro if micro < lot else None
                mesure["journal"] = rel(journal, root)
                break
            print(f"[mesure {taille}] échec avec micro-lot {micro} (mémoire ?) : on essaie plus petit",
                  flush=True)
            mesure = None
        if mesure is None:
            res["tailles"][taille] = {"erreur": "aucun micro-lot n'a fonctionné, voir journaux/mesure_*"}
            continue
        mesure.update(estimer(ctx, mesure["tokens_par_s"]))
        res["tailles"][taille] = mesure
        print(f"[mesure {taille}] {mesure['tokens_par_s']:,.0f} tokens/s -> total estimé "
              f"{mesure['total_heures']} h ({mesure['etapes_heures']})", flush=True)
    ecrire_json(racine / "mesures.json", res)
    return res


def choisir_taille(mesures, budget_heures):
    """La plus grande taille dont le total estimé tient dans le budget, ou None."""
    for t in TAILLES:
        m = (mesures or {}).get("tailles", {}).get(t)
        if m and "total_heures" in m and m["total_heures"] <= budget_heures:
            return t
    return None


# ---------------------------------------------------------------------------------------------
# lancer


def archiver(ctx, etat, groupe, etapes):
    """Range les sorties d'un chatbot dans archives/<date>/ et oublie ses étapes (--seulement)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for e in etapes:
        if e["groupe"] != groupe:
            continue
        out = Path(e["out"])
        if out.exists():
            dest = ctx.sortie / "archives" / stamp / out.relative_to(ctx.sortie)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(dest))
            print(f"  ancien {rel(out, ctx.sortie)} rangé dans {rel(dest, ctx.sortie)}")
        etat["etapes"].pop(e["nom"], None)
    etat.get("composition", {}).pop(groupe, None)  # recalculée : suit le paquet actuel
    ecrire_json(ctx.etat_path, etat)


def preparer_lancement(args, rec, device, runner=executer, root=ROOT):
    """(ctx, code) : taille choisie, mesures lues, données présentes, état compatible."""
    racine = Path(args.sortie)
    etat = lire_json(racine / "etat_lanceur.json")
    mesures = lire_json(racine / "mesures.json")
    taille = args.taille
    budget = args.budget_heures or val(rec["commun"]["budget_heures_defaut"])
    if taille == "essai" and etat and etat["taille"] != "essai":
        print("ERREUR : la recette d'essai se lance dans un dossier à part : "
              "`python -m chatbot.pipeline_rtx --sortie runs/essai lancer --taille essai`")
        return None, 2
    if etat and taille == "auto":
        taille = etat["taille"]
        print(f"Lancement déjà commencé en taille {taille} : on continue cette taille.")
    elif etat and taille != etat["taille"]:
        print(f"ERREUR : {racine} contient déjà un lancement en taille {etat['taille']} : continuer "
              f"avec --taille {etat['taille']}, ou repartir de zéro dans un autre dossier : "
              f"`python -m chatbot.pipeline_rtx --sortie runs/chatbot_rtx_2 lancer --taille {taille}` "
              f"(puis la même --sortie pour etat, resultats et publier)")
        return None, 2
    if taille == "auto":
        if not mesures:
            print("Pas encore de mesures/mesures.json : on mesure d'abord le débit (durée à mesurer).")
            mesures = mesurer(rec, racine, device, runner=runner, root=root)
            if mesures is None:
                return None, 3
        taille = choisir_taille(mesures, budget)
        if taille is None:
            est = {t: m.get("total_heures") for t, m in mesures["tailles"].items()}
            print(f"ERREUR : aucune taille ne tient dans {budget} h (estimations : {est}). "
                  f"Augmente --budget-heures ou force --taille S.")
            return None, 2
        print(f"Taille choisie : {taille} (estimation {mesures['tailles'][taille]['total_heures']} h "
              f"pour un budget de {budget} h)")
    m = (mesures or {}).get("tailles", {}).get(taille, {})
    demande = getattr(args, "micro_lot", None)
    if etat:
        # Reprise : on garde le micro-lot du début, SAUF s'il faut moins de mémoire (débordement) :
        # --micro-lot N, ou un mesures.json refait qui propose un micro-lot plus petit. Le calcul
        # ne change pas (accumulation de gradient) : les entraîneurs acceptent ce changement.
        micro = etat.get("micro_lot")
        if demande:
            micro = demande
        elif m.get("micro_lot") and (micro is None or m["micro_lot"] < micro):
            micro = m["micro_lot"]
            print(f"mesures.json propose un micro-lot plus petit ({micro}) : on l'utilise.")
    else:
        micro = demande or m.get("micro_lot")
        if m.get("total_heures"):
            print(f"Durée estimée pour la taille {taille} : {m['total_heures']} h (mesures.json)")
        elif taille != "essai":
            print("Aucune mesure pour cette taille : durée inconnue (lancer `mesurer` d'abord pour "
                  "l'estimer). Lot entier en une fois.")
    ctx = Ctx(rec, taille, racine, root, device, micro=micro)
    lot = val(ctx.recette()["base"]["lot"])
    if micro is not None and (micro < 1 or lot % micro):
        print(f"ERREUR : --micro-lot {micro} doit diviser le lot de la base ({lot}) : essaie "
              f"{', '.join(str(d) for d in (64, 32, 16, 8) if d < lot and lot % d == 0)}")
        return None, 2
    manque = donnees_manquantes(ctx)
    if manque:
        print("ERREUR : données absentes du paquet (paquet léger ? transfert incomplet ?) :\n  "
              + "\n  ".join(manque))
        return None, 2
    return ctx, 0


def cmd_lancer(args, rec, runner=executer, device=None, root=ROOT):
    device = device or device_auto()
    racine = Path(args.sortie)
    g = args.seulement
    if args.repartir and not g:
        print("ERREUR : --repartir s'utilise avec --seulement moliere|francais")
        return 2
    with Verrou(racine / "lanceur.verrou"):
        refuser_si_orpheline(racine)
        etat = lire_json(racine / "etat_lanceur.json")
        # Refus AVANT d'écrire quoi que ce soit : un refus ne doit pas figer la taille.
        if g and (etat or {}).get("etapes", {}).get("base", {}).get("statut") != "fini":
            print("ERREUR : --seulement suppose une base déjà finie : lancer d'abord sans --seulement")
            return 2
        if args.taille == "essai":
            assurer_donnees_essai(root)
        ctx, code = preparer_lancement(args, rec, device, runner, root)
        if ctx is None:
            return code
        etat = lire_etat(ctx)
        if etat is None:
            etat = nouvel_etat(ctx)
            ecrire_json(ctx.etat_path, etat)
        elif etat.get("micro_lot") != ctx.micro:
            print(f"micro-lot : {etat.get('micro_lot') or 'aucun'} -> {ctx.micro or 'aucun'} "
                  f"(même calcul, moins de mémoire ; les étapes finies ne sont pas refaites)")
            etat["micro_lot"] = ctx.micro
            ecrire_json(ctx.etat_path, etat)
        etapes = plan(ctx, etat)
        if g:
            mine = [e for e in etapes if e["groupe"] == g]
            if args.repartir or all(etat["etapes"].get(e["nom"], {}).get("statut") == "fini" for e in mine):
                print(f"On refait le chatbot {g} (base gardée).")
                archiver(ctx, etat, g, mine)
                etapes = plan(ctx, etat)  # recalcule : l'étape synthétique suit le paquet actuel
            else:
                print(f"Chatbot {g} commencé mais pas fini : on le reprend (--repartir pour le refaire).")
        complet = [e["nom"] for e in etapes]
        if g:
            etapes = [e for e in etapes if e["groupe"] == g]
        print(f"Plan : {' -> '.join(e['nom'] for e in etapes)} | taille {ctx.taille} | device "
              f"{ctx.device} | précision {ctx.precision} | compile {ctx.compile} | micro-lot "
              f"{ctx.micro or 'aucun'}", flush=True)
        return lancer_etapes(ctx, etapes, runner, plan_complet=complet)


# ---------------------------------------------------------------------------------------------
# autotest


def courbe(log_path):
    """(loss toutes finies ?, loss en baisse ?, première val_loss, dernière val_loss) d'un log.jsonl.

    « En baisse » : la validation finit plus bas qu'au pas 0, OU la loss d'entraînement finit plus
    bas qu'à la première mesure (une étape synthétique peut ne pas améliorer la validation Molière).
    """
    recs = [json.loads(l) for l in Path(log_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    vals = [r["val_loss"] for r in recs]
    trains = [r["train_loss"] for r in recs if r.get("train_loss") is not None]
    finies = all(math.isfinite(x) for x in vals + trains)
    baisse = min(vals[1:], default=vals[0]) < vals[0] or (len(trains) >= 2 and trains[-1] < trains[0])
    return finies, baisse, vals[0], vals[-1]


def cmd_autotest(args, rec, runner=executer, device=None, root=ROOT):
    """Recette « essai » de bout en bout, sur le device disponible, avec une coupure simulée."""
    t0 = time.time()
    device = device or device_auto()
    racine = Path(args.sortie)
    sortie = racine / "autotest"
    with Verrou(racine / "lanceur.verrou"):
        refuser_si_orpheline(racine)
        if sortie.exists():
            shutil.rmtree(sortie)
        assurer_donnees_essai(root)  # clone : extraits fabriqués depuis les données téléchargées
        ctx = Ctx(rec, "essai", sortie, root, device)
        manque = donnees_manquantes(ctx)
        if manque:
            print("ERREUR : données d'essai absentes (clone : lancer d'abord `telecharger`) :\n  "
                  + "\n  ".join(manque))
            return 2
        etapes = plan(ctx)
        print(f"Autotest : {' -> '.join(e['nom'] for e in etapes)} | device {device} | précision "
              f"{ctx.precision} (auto = bf16 sur CUDA, fp32 ailleurs)", flush=True)
        checks = []

        def check(nom, ok, detail=""):
            checks.append({"controle": nom, "ok": bool(ok), "detail": detail})
            print(f"  [{'OK' if ok else 'ÉCHEC'}] {nom} {detail}", flush=True)

        coupure = val(rec["essai"]["interruption_au_pas"])
        code1 = lancer_etapes(ctx, etapes, runner, interrompre={"base": coupure})
        etat = lire_etat(ctx)
        check("coupure simulée de la base", code1 == 3 and etat["etapes"]["base"]["statut"] == "interrompu",
              f"(code {code1}, attendu 3)")
        code2 = lancer_etapes(ctx, etapes, runner)
        check("reprise et fin de la chaîne", code2 == 0, f"(code {code2})")
        if code2 == 0:
            base = lire_json(sortie / "runs" / "base" / "report.json")
            check("reprise au bon pas", base["resumed_from_steps"] == [coupure],
                  f"(reprise aux pas {base['resumed_from_steps']})")
            for e in etapes:
                if e["type"] != "entrainement":
                    continue
                finies, baisse, first, last = courbe(Path(e["out"]) / "log.jsonl")
                best = lire_json(Path(e["out"]) / "report.json")["best"]["val_loss"]
                check(f"{e['nom']} : loss finies et en baisse", finies and baisse,
                      f"(val {first:.3f} -> {last:.3f}, meilleure {best:.3f})")
            for cb in CHATBOTS:
                c = ctx.recette()["chatbots"][cb]
                dossier = sortie / "modeles" / val(c["dossier_modele"])
                fmt = lire_json(dossier / "chat_config.json")["format"]
                check(f"export {cb} au bon format", fmt == val(c["format"]), f"({fmt})")
                cmd = [ctx.python, "-m", "chatbot.generate", "chat", "--model", ctx.p(dossier),
                       "--device", device, "--seed", "7", "--candidates", "2",
                       "Bonjour, comment allez-vous ?"]
                journal = sortie / "journaux" / f"{cb}_reponse.log"
                code, _ = runner(cmd, journal, root)
                lignes = [l for l in dernieres_lignes(journal, 40) if " : " in l and not l.startswith("   [")]
                reponse = lignes[-1].split(" : ", 1)[1] if lignes else ""
                check(f"réponse du chatbot {cb}", code == 0, f"« {reponse[:80]} »")
                m = lire_json(sortie / "evaluations" / cb / "metrics.json")
                bpb = m["val_next_turn"].get("val_bpb")
                check(f"évaluation réduite {cb}", bpb is not None and math.isfinite(bpb),
                      f"(BPB {bpb}, {m['n_replies']} réponses)")
            syn = [e["nom"] for e in etapes if e["nom"].endswith("_synthetique")]
            print(f"  étapes synthétiques testées : {syn or 'aucune (pas de fichier synthétique d essai)'}")
        secondes = round(time.time() - t0, 1)
        check("durée < 5 min", secondes < 300, f"({secondes} s)")
        ok = all(c["ok"] for c in checks)
        from chatbot.accel import resolve_precision
        effective = resolve_precision(ctx.precision, device)
        print(f"  précision effective : {effective}")
        ecrire_json(racine / "autotest.json", {"date": maintenant(), "device": device,
                                               "precision": effective, "secondes": secondes,
                                               "controles": checks, "ok": ok})
        print("AUTOTEST " + ("RÉUSSI" if ok else "ÉCHOUÉ") + f" en {secondes} s ({racine / 'autotest.json'})")
        return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# resultats


def environnement():
    """Machine et versions, sans nom de machine ni chemin personnel."""
    env = {"date": maintenant(), "os": f"{platform.system()} {platform.release()}",
           "machine": platform.machine(), "versions": versions()}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["carte"] = {"nom": p.name, "memoire_go": round(p.total_memory / 1e9, 1),
                            "capacite": f"{p.major}.{p.minor}"}
    except ImportError:
        pass
    return env


def fichiers_resultats(sortie, racine, avec_base=False):
    """{nom dans l'archive : fichier} — sans les données, les états de reprise ni les caches."""
    s = Path(sortie)
    pick = {}
    for pattern in ("modeles/*/*", "runs/*/report.json", "runs/*/log.jsonl", "evaluations/*/*",
                    "journaux/*.log", "*.json"):
        for f in s.glob(pattern):
            if f.is_file() and not f.name.endswith(EXCLUS):
                pick[f.relative_to(s).as_posix()] = f
    if avec_base and (s / "runs" / "base" / "best.pt").exists():
        pick["runs/base/best.pt"] = s / "runs" / "base" / "best.pt"
    for name in ("verification.json", "mesures.json", "autotest.json"):
        if (Path(racine) / name).exists() and Path(racine) != s:
            pick[f"lanceur/{name}"] = Path(racine) / name
    return pick


def cmd_resultats(args, rec, root=ROOT):
    racine = Path(args.sortie)
    sortie = racine / "autotest" if args.autotest else racine
    etat = lire_json(sortie / "etat_lanceur.json")
    if etat is None:
        print(f"ERREUR : aucun lancement dans {sortie}")
        return 2
    plan_noms = etat.get("plan", [])
    finis = [n for n in plan_noms if etat["etapes"].get(n, {}).get("statut") == "fini"]
    if len(finis) < len(plan_noms) and not args.partiel:
        print(f"ERREUR : lancement pas fini ({len(finis)}/{len(plan_noms)} étapes) : relancer "
              f"`lancer`, ou --partiel pour emporter quand même ce qui existe")
        return 2
    nom = RESULTATS + ("_autotest" if args.autotest else "")
    dest = Path(args.archive) if args.archive else archive_defaut(racine, args.autotest, root)
    pick = fichiers_resultats(sortie, racine, args.avec_base)
    ecrire_json(sortie / "environnement.json", environnement())
    pick["environnement.json"] = sortie / "environnement.json"
    manifeste = {"archive": nom, "date": maintenant(), "autotest": bool(args.autotest),
                 "complet": len(finis) == len(plan_noms), "taille": etat.get("taille"),
                 "lanceur": VERSION, "recettes": val(rec["version"]),
                 "fichiers": {k: {"sha256": sha256_fichier(f), "octets": f.stat().st_size}
                              for k, f in sorted(pick.items())}}
    creer_dossier(dest.parent)
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, f in sorted(pick.items()):
            z.write(f, f"{nom}/{k}")
        z.writestr(f"{nom}/MANIFESTE_RESULTATS.json", json.dumps(manifeste, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
    if bad:
        tmp.unlink(missing_ok=True)  # pas de reste à moitié écrit
        print(f"ERREUR : archive abîmée ({bad})")
        return 1
    os.replace(tmp, dest)
    print(f"Archive : {dest}\n  {len(pick)} fichiers, {dest.stat().st_size / 1e6:.1f} Mo, sha256 "
          f"{sha256_fichier(dest)}")
    if est_paquet(root):
        print(f"  à rapporter sur le Mac : python tools/import_chatbot_results.py {dest.name}")
    else:
        print("  suite : `publier` (à blanc d'abord) pour l'envoyer en pièce jointe d'une release, "
              "puis sur le Mac : tools/import_chatbot_results.py --depuis-release DEPOT TAG")
    return 0


def archive_defaut(racine, autotest=False, root=ROOT):
    """Paquet : à la racine du paquet (habitude). Clone : dans le dossier de sortie (ignoré par git)."""
    nom = RESULTATS + ("_autotest" if autotest else "") + ".zip"
    return Path(root) / nom if est_paquet(root) else Path(racine) / nom


# ---------------------------------------------------------------------------------------------
# Route git : release GitHub (gh), telecharger, publier
#
# Git refuse les fichiers de plus de 100 Mo : les tokens (4,2 Go) et les modèles passent en
# PIÈCES JOINTES d'une release GitHub (2 Gio au plus par fichier). `gh` (GitHub CLI) les envoie
# et les télécharge, pour un dépôt public comme privé, avec le compte connecté par `gh auth login`.


NOM_SUR = re.compile(r"^[A-Za-z0-9._-]+$")  # nom de pièce jointe : sans accent ni espace


def chemin_sur(rel_path):
    """Chemin relatif sûr (pas de « .. », pas de chemin absolu) : un JSON ne peut rien écrire ailleurs."""
    p = Path(str(rel_path).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise Refus(f"chemin refusé : {rel_path}")
    return p


def lire_donnees(path=None):
    path = Path(path or DONNEES_RTX)
    don = lire_json(path)
    if don is None:
        raise Refus(f"{path} absent")
    for f in don["fichiers"]:
        chemin_sur(f["destination"])
        if not NOM_SUR.match(f["piece_jointe"]):
            raise Refus(f"nom de pièce jointe refusé : {f['piece_jointe']}")
    return don


def depot_tag(depot, tag, donnees=None):
    """Dépôt et étiquette : ceux de la ligne de commande, sinon ceux de donnees_rtx.json."""
    depot = depot or (donnees or {}).get("depot")
    tag = tag or (donnees or {}).get("tag")
    if not depot or not tag or A_RENSEIGNER in depot or A_RENSEIGNER in tag:
        raise Refus("dépôt ou étiquette non renseignés : ajoute --depot COMPTE/DEPOT --tag ETIQUETTE "
                    "(ceux choisis par l'étudiant pour la release des données)")
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", depot):
        raise Refus(f"dépôt mal écrit : {depot} (attendu COMPTE/DEPOT)")
    return depot, tag


def gh_exe():
    exe = shutil.which("gh")
    if not exe:
        raise Refus("gh (GitHub CLI) introuvable. Windows : `winget install --id GitHub.cli`, puis "
                    "ouvrir un NOUVEAU terminal et `gh auth login` (c'est l'étudiant qui se connecte)")
    return exe


def gh(exe, *args, check=True):
    """Lance une commande gh ; en cas d'échec, Refus avec le message de gh."""
    r = subprocess.run([exe, *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if check and r.returncode:
        msg = (r.stderr or r.stdout).strip().splitlines()
        raise Refus(f"`gh {' '.join(map(str, args[:3]))} …` a échoué : {msg[-1] if msg else r.returncode}")
    return r


def gh_pret():
    """gh installé ET connecté ; sinon un message qui dit quoi faire."""
    exe = gh_exe()
    if gh(exe, "auth", "status", check=False).returncode:
        raise Refus("gh n'est pas connecté à GitHub : l'étudiant lance `gh auth login` lui-même "
                    "(Claude ne tape jamais de mot de passe ni de jeton)")
    return exe


def visibilite(exe, depot):
    """« PUBLIC », « PRIVATE »… lu par gh (lecture seule), ou None."""
    r = gh(exe, "repo", "view", depot, "--json", "visibility", check=False)
    try:
        return json.loads(r.stdout)["visibility"] if r.returncode == 0 else None
    except (ValueError, KeyError):
        return None


def depot_vide(exe, depot):
    """True si le dépôt n'a encore aucun commit (lu par gh, lecture seule), None si inconnu.
    GitHub refuse de créer une release dans un dépôt vide : il faut au moins un commit."""
    r = gh(exe, "repo", "view", depot, "--json", "isEmpty", check=False)
    try:
        return bool(json.loads(r.stdout)["isEmpty"]) if r.returncode == 0 else None
    except (ValueError, KeyError):
        return None


MESSAGE_VIDE = ("le dépôt {d} est VIDE (aucun commit) : GitHub refuse d'y créer une release. "
                "L'étudiant y met d'abord un premier commit (bouton « Add a README » sur la page du "
                "dépôt ; pour un nouveau dépôt : `gh repo create {d} --private --add-readme`), "
                "puis relance la même commande")


def pieces_existantes(exe, depot, tag):
    """{nom: taille} des pièces jointes de la release, ou None si la release n'existe pas."""
    r = gh(exe, "release", "view", tag, "--repo", depot, "--json", "assets", check=False)
    if r.returncode:
        return None
    return {a["name"]: a.get("size") for a in json.loads(r.stdout).get("assets", [])}


def telecharger_piece(exe, depot, tag, nom, octets, sha, dest):
    """Télécharge une pièce jointe dans un dossier temporaire, vérifie taille et sha256, PUIS la
    met à sa place : jamais de fichier à moitié téléchargé à la destination."""
    dest = Path(dest)
    tmp = creer_dossier(dest.parent / ".telechargement_rtx")
    try:
        gh(exe, "release", "download", tag, "--repo", depot, "--pattern", nom, "--dir", tmp, "--clobber")
        recu = tmp / nom
        if not recu.is_file():
            raise Refus(f"{nom} : rien reçu (pièce jointe absente de la release {tag} ?)")
        if recu.stat().st_size != octets or sha256_fichier(recu) != sha:
            recu.unlink()
            raise Refus(f"{nom} : taille ou sha256 différents de ceux attendus : fichier refusé "
                        f"(release incomplète ou modifiée ?)")
        os.replace(recu, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_telecharger(args, rec, root=ROOT):
    """Clone : télécharge les gros fichiers de donnees_rtx.json depuis la release, et les vérifie."""
    root = Path(root)
    don = lire_donnees(args.donnees)
    depot, tag = depot_tag(args.depot, args.tag, don)
    a_faire, deja = [], 0
    for f in don["fichiers"]:
        dest = root / chemin_sur(f["destination"])
        if dest.is_file() and dest.stat().st_size == f["octets"] and sha256_fichier(dest) == f["sha256"]:
            deja += 1
            continue
        a_faire.append(f)
    total = sum(f["octets"] for f in a_faire)
    print(f"{len(don['fichiers'])} fichiers attendus : {deja} déjà bons (sha256 vérifié), "
          f"{len(a_faire)} à télécharger ({total / 1e9:.2f} Go) depuis {depot}, release {tag}", flush=True)
    if not a_faire:
        assurer_donnees_essai(root)
        print("TÉLÉCHARGEMENT RÉUSSI (rien à faire)")
        return 0
    libre = shutil.disk_usage(next(p for p in [root, *root.parents] if p.exists())).free
    if libre < total + 1_000_000_000:
        raise Refus(f"espace disque insuffisant : {libre / 1e9:.1f} Go libres, il faut "
                    f"{total / 1e9:.1f} Go + 1 Go de marge")
    exe = gh_pret()
    t0 = time.time()
    for i, f in enumerate(sorted(a_faire, key=lambda x: x["octets"]), 1):
        dest = root / chemin_sur(f["destination"])
        creer_dossier(dest.parent)
        print(f"  [{i}/{len(a_faire)}] {f['piece_jointe']} ({f['octets'] / 1e6:.1f} Mo) -> "
              f"{f['destination']}", flush=True)
        t1 = time.time()
        telecharger_piece(exe, depot, tag, f["piece_jointe"], f["octets"], f["sha256"], dest)
        print(f"        reçu et vérifié en {time.time() - t1:.1f} s", flush=True)
    assurer_donnees_essai(root)
    print(f"TÉLÉCHARGEMENT RÉUSSI : {len(a_faire)} fichiers, {total / 1e9:.2f} Go en "
          f"{(time.time() - t0) / 60:.1f} min (mesuré)")
    return 0


def envoyer_release(exe, depot, tag, fichiers, titre, notes):
    """Envoie {nom de pièce jointe: fichier local} dans la release `tag` de `depot`.

    Crée la release si elle manque (sans la marquer « dernière »). Reprise : une pièce jointe
    déjà présente avec la même taille n'est pas renvoyée ; une de même nom mais d'une autre taille
    (envoi coupé) est remplacée. Contrôle final : chaque pièce jointe a la bonne taille.
    """
    for nom, path in fichiers.items():
        if not NOM_SUR.match(nom) or Path(path).name != nom:
            raise Refus(f"nom de pièce jointe refusé : {nom} ({path})")
        if Path(path).stat().st_size >= 2 * 1024 ** 3:
            raise Refus(f"{nom} fait 2 Gio ou plus : GitHub le refuserait")
    deja = pieces_existantes(exe, depot, tag)
    if deja is None and depot_vide(exe, depot):
        raise Refus(MESSAGE_VIDE.format(d=depot))  # avant tout envoi : rien à moitié fait
    if deja is None:
        print(f"  création de la release {tag} sur {depot}", flush=True)
        gh(exe, "release", "create", tag, "--repo", depot, "--title", titre, "--notes", notes,
           "--latest=false")
        deja = {}
    for i, (nom, path) in enumerate(fichiers.items(), 1):
        octets = Path(path).stat().st_size
        if deja.get(nom) == octets:
            print(f"  [{i}/{len(fichiers)}] {nom} déjà en ligne (même taille) : pas renvoyé", flush=True)
            continue
        print(f"  [{i}/{len(fichiers)}] envoi de {nom} ({octets / 1e6:.1f} Mo)…", flush=True)
        extra = ["--clobber"] if nom in deja else []
        gh(exe, "release", "upload", tag, str(path), "--repo", depot, *extra)
    final = pieces_existantes(exe, depot, tag) or {}
    faux = [n for n, p in fichiers.items() if final.get(n) != Path(p).stat().st_size]
    if faux:
        raise Refus(f"après envoi, taille différente ou pièce absente : {faux} (relancer reprend)")
    print(f"  contrôle : {len(fichiers)} pièces jointes en ligne, tailles conformes", flush=True)


def afficher_plan(depot, tag, vis, tailles, titre, vide=None):
    """Ce qui SERAIT envoyé ({nom: octets}), et les commandes gh exactes (mode à blanc)."""
    total = sum(tailles.values())
    print(f"Dépôt : {depot or A_RENSEIGNER} | étiquette : {tag or A_RENSEIGNER} | visibilité : "
          f"{vis or 'inconnue (dépôt non renseigné, ou gh absent / non connecté)'}")
    if vis == "PUBLIC":
        print("ATTENTION : dépôt PUBLIC : n'importe qui pourra télécharger ces fichiers.")
    if vide:
        print(f"ATTENTION : {MESSAGE_VIDE.format(d=depot)}.")
    for nom, octets in tailles.items():
        print(f"  {nom:48s} {octets:>15,d} octets")
    print(f"  total : {len(tailles)} fichiers, {total:,d} octets ({total / 1e9:.2f} Go)")
    d, t = depot or "COMPTE/DEPOT", tag or "ETIQUETTE"
    print("Commandes qui seraient lancées :")
    print(f"  gh release view {t} --repo {d} --json assets      (la release existe-t-elle ?)")
    print(f"  gh release create {t} --repo {d} --title \"{titre}\" --notes \"…\" --latest=false"
          f"      (si elle n'existe pas)")
    for nom in tailles:
        print(f"  gh release upload {t} <dossier d'envoi>/{nom} --repo {d}"
              f"      (sauté s'il est déjà en ligne, même taille)")
    print(f"  gh release view {t} --repo {d} --json assets      (contrôle des tailles)")
    return total


def decouper(archive, dossier, base, limite=None, ecrire=False):
    """Morceaux de moins de `limite` octets (base, ou base.001, .002…) + manifeste.

    ecrire=False (à blanc) : calcule seulement tailles et sha256, sans rien écrire.
    """
    limite = limite or LIMITE_PIECE
    archive = Path(archive)
    taille = archive.stat().st_size
    n = max(1, math.ceil(taille / limite))
    noms = [base] if n == 1 else [f"{base}.{i:03d}" for i in range(1, n + 1)]
    morceaux, entier = [], hashlib.sha256()
    if ecrire:
        creer_dossier(dossier)
    with open(archive, "rb") as src:
        for nom in noms:
            h, reste = hashlib.sha256(), min(limite, taille - src.tell())
            out = open(Path(dossier) / nom, "wb") if ecrire else None
            try:
                while reste > 0:
                    bloc = src.read(min(1 << 22, reste))
                    h.update(bloc)
                    entier.update(bloc)
                    reste -= len(bloc)
                    if out:
                        out.write(bloc)
            finally:
                if out:
                    out.close()
            morceaux.append({"piece_jointe": nom, "octets": (Path(dossier) / nom).stat().st_size if ecrire
                             else min(limite, taille - limite * len(morceaux)), "sha256": h.hexdigest()})
    return {"archive": base, "octets": taille, "sha256": entier.hexdigest(), "morceaux": morceaux}


def nom_lancement(etat, autotest=False):
    """Nom stable d'un lancement : date de création + taille (ex. 20261114_0312_S)."""
    cree = re.sub(r"[^0-9]", "", (etat or {}).get("cree", "")) or time.strftime("%Y%m%d%H%M%S")
    return f"{cree[:8]}_{cree[8:12]}_{(etat or {}).get('taille', 'x')}" + ("_autotest" if autotest else "")


def anonymiser(obj, remplacements):
    """Remplace chemins personnels et exécutable Python dans toutes les chaînes d'un JSON."""
    if isinstance(obj, dict):
        return {k: anonymiser(v, remplacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [anonymiser(v, remplacements) for v in obj]
    if isinstance(obj, str):
        for a, b in remplacements:
            obj = obj.replace(a, b)
    return obj


PETITS = ("bilan.json", "etat_lanceur.json", "environnement.json", "runs/*/report.json",
          "runs/*/log.jsonl", "evaluations/*/metrics.json", "modeles/*/training_report.json",
          "modeles/*/chat_config.json")


def copier_petits_fichiers(sortie, racine, dest, root=ROOT, max_octets=1_000_000):
    """Rapports à suivre par git (pas de poids, pas de journaux) dans experiments/chatbot_rtx/<nom>/.

    Les chemins personnels (dossier de l'utilisateur, exécutable Python) sont remplacés : le
    dépôt peut être public.
    """
    sortie, racine, dest = Path(sortie), Path(racine), Path(dest)
    rempl = sorted({(sys.executable, "python"), (str(Path(root).resolve()), "."),
                    (str(Path.home()), "~")}, key=lambda x: -len(x[0]))
    choix = {}
    for motif in PETITS:
        for f in sortie.glob(motif):
            k = f.relative_to(sortie).as_posix()
            # « runs/ » est ignoré par .gitignore (à tous les niveaux) : on range sous entrainements/
            choix["entrainements/" + k[len("runs/"):] if k.startswith("runs/") else k] = f
    for name in ("verification.json", "mesures.json", "autotest.json"):  # vérifs du lanceur
        if (racine / name).exists():
            choix[f"lanceur/{name}"] = racine / name
    copies = []
    for k, f in sorted(choix.items()):
        if f.stat().st_size > max_octets:
            print(f"  {k} trop gros pour git ({f.stat().st_size:,d} octets) : pas copié")
            continue
        cible = creer_dossier((dest / k).parent) / Path(k).name
        if f.suffix == ".jsonl":
            lignes = [json.dumps(anonymiser(json.loads(l), rempl), ensure_ascii=False)
                      for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            cible.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        else:
            ecrire_json(cible, anonymiser(json.loads(f.read_text(encoding="utf-8")), rempl))
        copies.append(k)
    return copies


def cmd_publier(args, rec, root=ROOT):
    """Envoie l'archive de résultats en pièce jointe d'une release (À BLANC par défaut) et copie
    les petits rapports dans experiments/chatbot_rtx/<nom>/ pour pouvoir les commiter."""
    racine = Path(args.sortie)
    sortie = racine / "autotest" if args.autotest else racine
    archive = Path(args.archive) if args.archive else archive_defaut(racine, args.autotest, root)
    etat = lire_json(sortie / "etat_lanceur.json")
    if not archive.exists() or etat is None:
        raise Refus(f"{archive} absent : lancer d'abord `resultats`" + (" --autotest" if args.autotest else ""))
    nom = nom_lancement(etat, args.autotest)
    dest = Path(root) / "experiments" / "chatbot_rtx" / nom
    copies = copier_petits_fichiers(sortie, racine, dest, root)
    print(f"{len(copies)} petits rapports copiés dans {rel(dest, root)}/ (à commiter avec l'accord "
          f"de l'étudiant : git add {rel(dest, root)})")
    base = f"{RESULTATS}_{nom}.zip"
    envoi = sortie / "envoi_release"
    if args.publier:
        depot, tag = depot_tag(args.depot, args.tag)
        exe = gh_pret()
        if envoi.exists():
            shutil.rmtree(envoi)
    else:
        depot, tag = args.depot, args.tag
        exe = shutil.which("gh")
    man = decouper(archive, envoi, base, ecrire=args.publier)
    man.update(lancement=nom, autotest=bool(args.autotest), cree=maintenant(), lanceur=VERSION,
               note="reconstituer : concaténer les morceaux dans l'ordre, vérifier le sha256 "
                    "(tools/import_chatbot_results.py --depuis-release le fait)")
    nom_man = f"{RESULTATS}_{nom}.manifeste.json"
    fichiers = {m["piece_jointe"]: envoi / m["piece_jointe"] for m in man["morceaux"]}
    fichiers[nom_man] = envoi / nom_man
    ecrire_json(creer_dossier(envoi) / nom_man, man)  # à blanc aussi : on montre sa vraie taille
    tailles = {m["piece_jointe"]: m["octets"] for m in man["morceaux"]}
    tailles[nom_man] = (envoi / nom_man).stat().st_size
    vis = vide = None
    if exe and depot and A_RENSEIGNER not in depot:
        try:
            if gh(exe, "auth", "status", check=False).returncode == 0:
                vis, vide = visibilite(exe, depot), depot_vide(exe, depot)  # lecture seule
        except OSError:
            vis = vide = None
    print(f"Archive {archive.name} : {man['octets']:,d} octets, sha256 {man['sha256']}, "
          f"{len(man['morceaux'])} morceau(x) de {LIMITE_PIECE:,d} octets au plus")
    afficher_plan(depot, tag, vis, tailles, f"Résultats chatbot RTX {nom}", vide)
    if not args.publier:
        print("À BLANC : rien n'a été envoyé. Pour envoyer (avec l'accord de l'étudiant) : "
              "`publier --depot COMPTE/DEPOT --tag ETIQUETTE --publier`")
        return 0
    try:
        envoyer_release(exe, depot, tag, fichiers, f"Résultats chatbot RTX {nom}",
                        f"Archive de résultats du lanceur RTX ({nom}) : modèles exportés, rapports, "
                        f"évaluations. Reconstituer avec tools/import_chatbot_results.py --depuis-release.")
    finally:
        shutil.rmtree(envoi, ignore_errors=True)  # copies des morceaux : refaites à chaque envoi
    print(f"PUBLICATION RÉUSSIE : {depot}, release {tag}. Sur le Mac : uv run python "
          f"tools/import_chatbot_results.py --depuis-release {depot} {tag}")
    return 0


def reconstituer_depuis_release(exe, depot, tag, dossier, nom_archive=None):
    """Télécharge le manifeste et les morceaux d'une archive de résultats, la reconstitue dans
    `dossier` et vérifie son sha256. Renvoie le chemin de l'archive reconstituée."""
    dossier = creer_dossier(dossier)
    pieces = pieces_existantes(exe, depot, tag)
    if pieces is None:
        raise Refus(f"release {tag} introuvable sur {depot} (ou pas d'accès avec ce compte gh)")
    mans = sorted(n for n in pieces if n.startswith(RESULTATS) and n.endswith(".manifeste.json"))
    if nom_archive:
        mans = [m for m in mans if m.startswith(nom_archive.removesuffix(".zip"))]
    if len(mans) != 1:
        raise Refus(f"{len(mans)} archive(s) de résultats dans la release {tag} : {mans or 'aucune'}"
                    + (" ; choisir avec --nom-release" if len(mans) > 1 else ""))
    recus = creer_dossier(dossier / "morceaux")  # à part : un morceau peut porter le nom de l'archive
    gh(exe, "release", "download", tag, "--repo", depot, "--pattern", mans[0], "--dir", recus, "--clobber")
    man = json.loads((recus / mans[0]).read_text(encoding="utf-8"))
    cible = dossier / chemin_sur(man["archive"]).name
    h = hashlib.sha256()
    with open(cible, "wb") as out:
        for m in man["morceaux"]:
            if not NOM_SUR.match(m["piece_jointe"]):
                raise Refus(f"nom de morceau refusé : {m['piece_jointe']}")
            print(f"  morceau {m['piece_jointe']} ({m['octets'] / 1e6:.1f} Mo)…", flush=True)
            gh(exe, "release", "download", tag, "--repo", depot, "--pattern", m["piece_jointe"],
               "--dir", recus, "--clobber")
            p = recus / m["piece_jointe"]
            if p.stat().st_size != m["octets"] or sha256_fichier(p) != m["sha256"]:
                raise Refus(f"morceau {m['piece_jointe']} : taille ou sha256 différents du manifeste")
            with open(p, "rb") as src:
                while bloc := src.read(1 << 22):
                    h.update(bloc)
                    out.write(bloc)
            p.unlink()
    shutil.rmtree(recus, ignore_errors=True)
    if h.hexdigest() != man["sha256"] or cible.stat().st_size != man["octets"]:
        raise Refus("archive reconstituée : sha256 différent du manifeste")
    print(f"archive reconstituée et vérifiée : {cible.name} ({man['octets']:,d} octets)")
    return cible


# ---------------------------------------------------------------------------------------------
# etat


def cmd_etat(args, rec):
    racine = Path(args.sortie)
    for sortie in (racine, racine / "autotest"):
        etat = lire_json(sortie / "etat_lanceur.json")
        if not etat:
            continue
        print(f"{sortie} : taille {etat['taille']}, device {etat['device']}, précision {etat['precision']}")
        for nom in etat.get("plan", []):
            st = etat["etapes"].get(nom, {})
            print(f"  {nom:24s} {st.get('statut', 'à faire'):11s} {st.get('secondes', 0) / 60:7.1f} min")
        courant = [n for n in etat.get("plan", []) if etat["etapes"].get(n, {}).get("statut") == "en_cours"]
        for n in courant:
            print(f"  dernières lignes de {n} :\n    " + "\n    ".join(
                dernieres_lignes(sortie / "journaux" / f"{n}.log", 5)))
    m = lire_json(racine / "mesures.json")
    if m:
        print("mesures : " + ", ".join(f"{t} {x.get('tokens_par_s', 0):,.0f} tokens/s ≈ {x.get('total_heures')} h"
                                       for t, x in m["tailles"].items()))
    return 0


# ---------------------------------------------------------------------------------------------
# Ligne de commande


def build_parser():
    p = argparse.ArgumentParser(prog="python -m chatbot.pipeline_rtx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sortie", default=None,
                   help="dossier des sorties (défaut : runs/chatbot_rtx/ dans un clone, sorties_rtx/ "
                        "dans le paquet zip)")
    p.add_argument("--recettes", default=str(RECETTES))
    p.add_argument("--donnees", default=str(DONNEES_RTX),
                   help="liste des gros fichiers (défaut : chatbot/donnees_rtx.json)")
    # --sortie est aussi acceptée APRÈS la sous-commande (`lancer --taille S --sortie D`) :
    # SUPPRESS = la sous-commande n'écrase pas la valeur donnée avant elle quand elle ne la répète pas.
    commun = argparse.ArgumentParser(add_help=False)
    commun.add_argument("--sortie", default=argparse.SUPPRESS,
                        help="dossier des sorties (même rôle que --sortie avant la sous-commande)")
    sub = p.add_subparsers(dest="commande", required=True)
    c = [commun]  # chaque sous-commande accepte aussi --sortie
    t = sub.add_parser("telecharger", parents=c, help="clone : gros fichiers depuis la release GitHub (gh)")
    t.add_argument("--depot", default=None, help="COMPTE/DEPOT de la release des données")
    t.add_argument("--tag", default=None, help="étiquette de la release des données")
    v = sub.add_parser("verifier", parents=c, help="données, versions, carte, disque")
    v.add_argument("--sans-gpu", action="store_true", help="Mac d'essai : pas de carte NVIDIA attendue")
    sub.add_parser("autotest", parents=c, help="recette minuscule de bout en bout, avec coupure et reprise")
    m = sub.add_parser("mesurer", parents=c, help="débit des tailles et durées estimées")
    m.add_argument("--tailles", nargs="+", default=["S", "M"], choices=["S", "M", "essai"])
    l = sub.add_parser("lancer", parents=c, help="base -> chatbot A -> chatbot B -> exports -> évaluations")
    l.add_argument("--taille", choices=["S", "M", "auto", "essai"], required=True,
                   help="essai = recette minuscule (preuve de bout en bout, modèles inutiles)")
    l.add_argument("--budget-heures", type=float, default=None)
    l.add_argument("--micro-lot", type=int, default=None,
                   help="couper chaque lot en morceaux de N séquences (mémoire insuffisante) : même "
                        "calcul, moins de mémoire ; accepté aussi en reprise")
    l.add_argument("--seulement", choices=CHATBOTS, default=None,
                   help="ré-entraîner un seul chatbot (base gardée)")
    l.add_argument("--repartir", action="store_true",
                   help="avec --seulement : refaire ce chatbot de zéro même s'il était en cours")
    r = sub.add_parser("resultats", parents=c, help="archive RESULTATS_CHATBOT_RTX.zip")
    r.add_argument("--autotest", action="store_true", help="emporter les sorties de l'autotest")
    r.add_argument("--partiel", action="store_true", help="accepter un lancement pas fini")
    r.add_argument("--avec-base", action="store_true", help="ajouter le modèle de base (runs/base/best.pt)")
    r.add_argument("--archive", default=None, help="chemin de l'archive")
    pu = sub.add_parser("publier", parents=c, help="archive de résultats -> release GitHub (À BLANC par défaut)")
    pu.add_argument("--depot", default=None, help="COMPTE/DEPOT qui recevra la release")
    pu.add_argument("--tag", default=None, help="étiquette de la release des résultats")
    pu.add_argument("--publier", action="store_true",
                    help="envoyer pour de vrai (sinon : plan à blanc, rien n'est envoyé)")
    pu.add_argument("--autotest", action="store_true", help="l'archive de l'autotest")
    pu.add_argument("--archive", default=None, help="chemin de l'archive (défaut : celle de `resultats`)")
    sub.add_parser("etat", parents=c, help="où en est le lancement")
    return p


def main(argv=None):
    console_utf8()
    args = build_parser().parse_args(argv)
    args.sortie = args.sortie or str(sortie_defaut(ROOT))
    rec = charger_recettes(args.recettes)
    print(f"{'paquet' if est_paquet(ROOT) else 'clone'} : {ROOT}", flush=True)
    installer_arrets()
    try:
        if args.commande == "mesurer":
            # Même verrou que lancer et autotest : une mesure en plein entraînement fausserait le
            # débit ET ralentirait l'entraînement (un seul calcul à la fois sur la carte).
            with Verrou(Path(args.sortie) / "lanceur.verrou"):
                refuser_si_orpheline(Path(args.sortie))
                res = mesurer(rec, Path(args.sortie), device_auto(), args.tailles)
            return 0 if res and all("erreur" not in x for x in res["tailles"].values()) else 1
        return {"telecharger": cmd_telecharger, "verifier": cmd_verifier, "autotest": cmd_autotest,
                "lancer": cmd_lancer, "resultats": cmd_resultats, "publier": cmd_publier,
                "etat": cmd_etat}[args.commande](args, rec)
    except Refus as e:
        print(f"ERREUR : {e}", flush=True)
        return 2
    except KeyboardInterrupt as e:  # arrêt demandé entre deux étapes (l'état est déjà écrit)
        print(f"\n{e or 'Ctrl-C'} : lanceur arrêté. Relancer la MÊME commande pour reprendre.",
              flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
