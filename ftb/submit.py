"""Prépare une soumission au leaderboard à partir d'un run officiel terminé.

    python -m ftb.submit --run runs/mon_run --participant "Prénom Nom"
    git add submissions/ && git commit -m "soumission" && git push

Copie dans submissions/<date>_<heure>/ : le checkpoint, le tokenizer, les logs d'entraînement,
l'empreinte du dataset (data_meta.json) et les 5 échantillons générés. Le mentor relève les
soumissions poussées sur ton dépôt, les évalue sur le set caché du leaderboard et met à jour
LEADERBOARD.md. 2 soumissions par jour (heure de Paris) au maximum.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from ftb import KIT_VERSION, load_config
from ftb.evaluate import check_model_config
from ftb.tokenizer import file_sha256

MAX_PER_DAY = 2
FILES = ["ckpt.pt", "tokenizer.json", "train_log.jsonl", "data_meta.json", "samples.txt"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.submit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="dossier de sortie de ftb.train")
    p.add_argument("--participant", required=True)
    p.add_argument("--dataset-url", default="",
                   help="où l'orga peut récupérer ton dataset (Hugging Face, release GitHub...)")
    p.add_argument("--note", default="", help="une ligne sur ce qui change dans cette soumission")
    p.add_argument("--dest", default="submissions")
    args = p.parse_args(argv)

    run = Path(args.run)
    missing = [f for f in FILES if not (run / f).exists()]
    if missing:
        sys.exit(f"ERREUR : fichiers manquants dans {run} : {missing}")
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    check_model_config(ckpt["model_config"])
    if not meta.get("official"):
        sys.exit("ERREUR : run de mise au point (--debug-budget) : non soumissible")
    if meta.get("kit_version") != KIT_VERSION:
        sys.exit(f"ERREUR : run fait avec le kit {meta.get('kit_version')}, kit actuel {KIT_VERSION}")
    cfg = load_config()
    if meta["tokens_seen"] != cfg["train"]["max_steps"] * cfg["train"]["tokens_per_step"]:
        sys.exit("ERREUR : le nombre de tokens vus ne correspond pas au budget officiel")
    for k in ("lr", "min_lr", "warmup_steps", "batch_size", "seq_len", "seed", "weight_decay"):
        if meta["train_config"].get(k) != cfg["train"][k]:
            sys.exit(f"ERREUR : hyperparamètre {k} modifié ({meta['train_config'].get(k)})")
    if file_sha256(run / "tokenizer.json") != meta["tokenizer_sha256"]:
        sys.exit("ERREUR : tokenizer.json ne correspond pas au checkpoint")

    now = datetime.now(ZoneInfo("Europe/Paris"))
    dest_root = Path(args.dest)
    today = now.strftime("%Y-%m-%d")
    already = [d for d in dest_root.glob(f"{today}_*") if d.is_dir()] if dest_root.exists() else []
    if len(already) >= MAX_PER_DAY:
        sys.exit(f"ERREUR : déjà {len(already)} soumissions aujourd'hui ({today}), maximum {MAX_PER_DAY}")
    dest = dest_root / now.strftime("%Y-%m-%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=False)
    for f in FILES:
        if f == "ckpt.pt":  # poids en float16 : 20 Mo au lieu de 40 par soumission
            half = {k: v.half() for k, v in ckpt["model"].items()}
            torch.save({**ckpt, "model": half}, dest / f)
        else:
            shutil.copyfile(run / f, dest / f)
    info = {
        "participant": args.participant,
        "submitted": now.isoformat(timespec="seconds"),
        "kit_version": KIT_VERSION,
        "val_bpb": meta["val_bpb"],
        "val_bpb_par_genre": meta["val_bpb_par_genre"],
        "dataset_sha256": meta["dataset_sha256"],
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "dataset_url": args.dataset_url,
        "note": args.note,
        "device": meta.get("device"),
        "train_time_s": meta.get("train_time_s"),
    }
    with open(dest / "submission.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    size = sum((dest / f).stat().st_size for f in FILES) / 1e6
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\nSoumission prête : {dest} ({size:.0f} Mo). Pousse-la :\n"
          f"  git add {dest} && git commit -m \"soumission {dest.name}\" && git push")


if __name__ == "__main__":
    main()
