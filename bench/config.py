"""Config d'un run du banc : valeurs par défaut + fichier YAML (`configs/bench/<run>.yaml`).

Dossiers (chemins relatifs à la racine du repo) :
- dataset ordonné, tokenizer, train.bin : <paths.data>/mixes/<run>/
- sorties de ftb.train et ftb.evaluate  : <paths.runs>/<run>/
- tableau des runs                      : <paths.experiments>/results.csv
"""
from __future__ import annotations

from pathlib import Path

from bench.common import REPO, deep_merge, load_yaml, resolve_path
from bench.mix import MIX_DEFAULTS, SOURCE_DEFAULTS

RUN_DEFAULTS = {
    "run": None,                       # nom du run (défaut : nom du fichier YAML)
    "seed": 1337,                      # graine du mix (mélange, tirages) ; le kit a la sienne
    "notes": "",
    "paths": {"data": "data", "runs": "runs", "experiments": "experiments"},
    "mix": MIX_DEFAULTS,
    "tokenizer": {
        "path": None,                  # tokenizer.json existant ; sinon entraîné sur l'échantillon du mix
        "pretokenizer": "gpt4",        # gpt4 (motif du kit) | fr (l'apostrophe reste avec « l' », « qu' »)
    },
    "train": {
        "mode": "proxy",               # proxy : --debug-budget (non soumissible) | full : 100 M tokens officiels
        "debug_budget": 25_000_000,    # tokens vus en mode proxy (~12 min sur le Mac, bench.throughput)
        "device": "auto",              # auto : mps en proxy sur le Mac, cuda sur Kaggle
        "micro_batch": None,           # accumulation de gradient (même résultat, moins de mémoire)
        "compile": False,
    },
    "eval": {"validation": None, "device": None},  # None : data/validation.jsonl du kit
}


def load_run_config(path=None, overrides: dict | None = None) -> dict:
    """Lit un YAML de run et le complète avec les valeurs par défaut."""
    user = load_yaml(path) if path else {}
    cfg = deep_merge(RUN_DEFAULTS, user)
    cfg = deep_merge(cfg, overrides or {})
    srcs = ((overrides or {}).get("mix") or {}).get("sources") or (user.get("mix") or {}).get("sources") or []
    cfg["mix"]["sources"] = [deep_merge(SOURCE_DEFAULTS, s) for s in srcs]
    if not cfg["run"]:
        cfg["run"] = Path(path).stem if path else "run"
    if cfg["train"]["mode"] not in ("proxy", "full"):
        raise SystemExit(f"train.mode inconnu : {cfg['train']['mode']!r} (proxy | full)")
    return cfg


def data_root(cfg) -> Path:
    return resolve_path(cfg["paths"]["data"], REPO)


def mix_dir(cfg) -> Path:
    return data_root(cfg) / "mixes" / cfg["run"]


def run_dir(cfg) -> Path:
    return resolve_path(cfg["paths"]["runs"], REPO) / cfg["run"]


def results_csv(cfg) -> Path:
    return resolve_path(cfg["paths"]["experiments"], REPO) / "results.csv"
