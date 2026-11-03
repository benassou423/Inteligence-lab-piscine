"""Kit officiel de la piscine I Lab « Feed the Beast »."""

import json
import math
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = KIT_ROOT / "config" / "official.json"
VALIDATION_PATH = KIT_ROOT / "data" / "validation.jsonl"


def load_config(path=CONFIG_PATH):
    """Charge la configuration officielle (modèle, entraînement, tokenizer, évaluation)."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tr = cfg["train"]
    tokens_per_step = tr["batch_size"] * tr["seq_len"]
    tr["tokens_per_step"] = tokens_per_step
    tr["max_steps"] = math.ceil(tr["budget_tokens"] / tokens_per_step)
    return cfg


KIT_VERSION = load_config()["kit_version"]
