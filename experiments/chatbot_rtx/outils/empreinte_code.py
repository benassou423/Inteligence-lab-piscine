"""Empreinte du CODE de chaque fichier Python du dépôt, sans commentaires ni docstrings.

Elle ne change pas si seuls les commentaires et les docstrings bougent : c'est la preuve qu'une retouche de
documentation n'a pas touché au comportement. L'empreinte brute (sha256 des octets) est notée aussi.

    python experiments/chatbot_rtx/outils/empreinte_code.py REFERENCE.json ecrire     # prend l'état de référence
    python experiments/chatbot_rtx/outils/empreinte_code.py REFERENCE.json            # compare à la référence
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sans_docstrings(arbre):
    for n in ast.walk(arbre):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            p = n.body[0]
            if isinstance(p, ast.Expr) and isinstance(p.value, ast.Constant) and isinstance(p.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return arbre


def empreintes(chemin):
    brut = Path(chemin).read_bytes()
    arbre = sans_docstrings(ast.parse(brut.decode("utf-8")))
    return {"code": hashlib.sha256(ast.dump(arbre).encode()).hexdigest(), "octets": hashlib.sha256(brut).hexdigest()}


def fichiers():
    sortie = ""
    for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        sortie += subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout
    return sorted({f for f in sortie.splitlines() if f.endswith(".py") and Path(f).exists()})


actuel = {f: empreintes(f) for f in fichiers()}
ref = Path(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "ecrire":
    ref.write_text(json.dumps(actuel, indent=1), encoding="utf-8")
    print(len(actuel), "empreintes écrites dans", ref.as_posix())
else:
    avant = json.loads(ref.read_text(encoding="utf-8"))
    code, doc = [], []
    for f in sorted(set(avant) | set(actuel)):
        a, b = avant.get(f), actuel.get(f)
        if a is None or b is None:
            print("AJOUTÉ OU RETIRÉ :", f)
        elif a["code"] != b["code"]:
            code.append(f)
        elif a["octets"] != b["octets"]:
            doc.append(f)
    for f in code:
        print("CODE CHANGÉ :", f)
    for f in doc:
        print("documentation seule :", f)
    print(f"comparé {len(actuel)} fichiers : {len(code)} avec du code changé, {len(doc)} avec la documentation seule")
