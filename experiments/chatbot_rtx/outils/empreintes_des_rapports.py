"""Empreintes de code consignées dans les rapports d'entraînement et d'évaluation, comparées aux fichiers.

Pour chaque rapport : la clé, l'empreinte consignée, et si elle correspond au fichier actuel, au fichier du
dernier commit (HEAD), ou à aucun des deux.

    python experiments/chatbot_rtx/outils/empreintes_des_rapports.py RAPPORT.json [RAPPORT.json ...]
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

NOMS = {"pretrain_sha256": "chatbot/pretrain.py", "format_sha256": "chatbot/format.py",
        "finetune_sha256": "chatbot/finetune.py", "accel_sha256": "chatbot/accel.py",
        "generate_sha256": "chatbot/generate.py", "evaluate_sha256": "chatbot/evaluate.py"}


def actuel(f):
    p = Path(f)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def head(f):
    r = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def paires(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and len(v) == 64 and "sha" in (pre + k).lower():
                yield pre + k, v
            else:
                yield from paires(v, pre + k + ".")


for rapport in sys.argv[1:]:
    print(rapport)
    d = json.loads(Path(rapport).read_text(encoding="utf-8"))
    for cle, h in paires(d):
        nom = cle.split(".")[-1]
        if cle.endswith(".py"):  # clé du type code_sha256.evaluate.py
            nom = ".".join(cle.split(".")[-2:])
        f = NOMS.get(nom) or ("chatbot/" + nom if nom.endswith(".py") else None)
        if not f:
            continue
        etat = "= fichier actuel" if actuel(f) == h else "= HEAD (fichier retouché depuis)" if head(f) == h else "ni l'un ni l'autre"
        print(f"   {cle:40s} {h[:12]}…  {f:24s} {etat}")
