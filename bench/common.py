"""Outils communs du banc : chemins, configs YAML, hachage, journal, fichiers."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import xxhash
import yaml

REPO = Path(__file__).resolve().parent.parent
DOMAINS = ("encyclopedique", "litteraire", "presse", "dialogue")  # genres de la val du kit


# --------------------------------------------------------------------------- config

def deep_merge(base: dict, override: dict | None) -> dict:
    """Fusionne récursivement `override` dans une copie de `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path, root: Path = REPO) -> Path:
    """Chemin absolu : relatif à la racine du repo sauf s'il est déjà absolu."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p)


# --------------------------------------------------------------------------- hachage

def h64(*parts) -> int:
    """Hachage 64 bits stable (indépendant du processus, contrairement à hash())."""
    return xxhash.xxh64_intdigest(":".join(str(p) for p in parts).encode("utf-8"))


def unit_float(*parts) -> float:
    """Nombre pseudo-aléatoire déterministe dans [0, 1)."""
    return h64(*parts) / 2.0**64


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- fichiers

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    tmp.replace(path)


def append_csv(path, row: dict, fieldnames: list[str]) -> None:
    """Ajoute une ligne à un CSV (écrit l'en-tête si le fichier est neuf)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:  # un en-tête différent décalerait toutes les valeurs sans erreur visible
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != list(fieldnames):
            raise SystemExit(f"{path} : en-tête {header} ≠ colonnes attendues {list(fieldnames)} "
                             f"(renommer l'ancien fichier, une nouvelle table sera créée)")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def doc_files(directory) -> list[Path]:
    """Fichiers de documents d'un dossier, triés (parcours déterministe).

    Ignore tout chemin dont un élément commence par « _ » ou « . » (temporaires,
    échantillons de travail des collecteurs, fichiers `.part`)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.jsonl.zst")
                  if not any(part.startswith(("_", ".")) for part in p.relative_to(directory).parts))


_STAGE_RE = re.compile(r"^(\d+)_")


def final_stage_dir(stages_root, config_name: str) -> Path | None:
    """Dernière étape terminée d'une config pipeline : `data/stages/<config>/<NN>_<etape>/`.

    Une étape est terminée quand son `stats.json` existe. Un dossier `final/`
    terminé est prioritaire s'il existe.
    """
    root = Path(stages_root) / config_name
    if not root.is_dir():
        return None
    final = root / "final"
    if (final / "stats.json").exists():
        return final
    stages = []
    for d in root.iterdir():
        m = _STAGE_RE.match(d.name)
        if d.is_dir() and m and (d / "stats.json").exists():
            stages.append((int(m.group(1)), d.name, d))
    return max(stages)[2] if stages else None


def stage_limit(stage_dir) -> int | None:
    """`--limit` d'une étape de pipeline (run d'essai sur un échantillon), ou None."""
    stats = Path(stage_dir) / "stats.json"
    try:
        return read_json(stats).get("limit")
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- journal

def get_logger(name: str = "bench", log_file=None) -> logging.Logger:
    """Journal sur la console (et dans un fichier si demandé)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


def human_bytes(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if abs(n) < 1000 or unit == "Go":
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} Go"
