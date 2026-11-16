"""Découpe items.json (réponses mélangées, sans nom de modèle) en lots de 72 pour les juges.

    python experiments/chatbot_rtx/outils/decouper_lots.py DOSSIER_AVEUGLE DOSSIER_DES_LOTS

DOSSIER_DES_LOTS ne reçoit que lot1.json, lot2.json… : jamais cle.json. Les juges ne lisent que ce dossier.
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads((src / "items.json").read_text(encoding="utf-8"))
assert len(items) % 72 == 0, f"{len(items)} réponses : pas un multiple de 72"
dst.mkdir(parents=True, exist_ok=True)
assert not list(dst.glob("*")), f"{dst} n'est pas vide"
for i in range(0, len(items), 72):
    (dst / f"lot{i // 72 + 1}.json").write_text(json.dumps(items[i:i + 72], ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
print(f"{len(items)} réponses -> {len(items) // 72} lots de 72 dans {dst}")
