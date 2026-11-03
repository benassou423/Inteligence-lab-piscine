"""Étape `exclude_heldout` : la val ne sert jamais à l'entraînement.

Retire tout document dont l'ID figure dans `data/val/*/heldout_ids.txt`.
Toujours en première position (le runner le vérifie).
"""
import glob

import xxhash

from pipeline.clean.base import Outcome, Stage

DEFAULT_GLOB = "data/val/*/heldout_ids.txt"


def heldout_files(pattern, root):
    """Fichiers d'IDs exclus, triés (chemins relatifs résolus depuis la racine du repo)."""
    pat = pattern if pattern.startswith("/") else str(root / pattern)
    return sorted(glob.glob(pat))


def load_ids(files):
    """Ensemble des IDs (une ligne par ID ; lignes vides et commentaires # ignorés)."""
    ids = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    ids.add(s)
    return ids


class ExcludeHeldoutStage(Stage):
    name = "exclude_heldout"
    filters = True

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.files = heldout_files(self.params.get("glob", DEFAULT_GLOB), ctx.root)
        self.ids = set()

    def setup(self):
        self.ids = load_ids(self.files)

    def fingerprint_extra(self, source=None):
        # Seuls les IDs de cette source comptent : ajouter des IDs d'une autre source
        # à la val ne force pas à recalculer celle-ci.
        if not hasattr(self, "_all_ids"):
            self._all_ids = sorted(load_ids(self.files))
        prefix = f"{source}:" if source else ""
        ids = [i for i in self._all_ids if i.startswith(prefix)]
        h = xxhash.xxh64()
        for i in ids:
            h.update(i.encode("utf-8") + b"\n")
        return {"n_ids": len(ids), "ids_digest": h.hexdigest()}

    def apply(self, doc):
        if doc["id"] in self.ids:
            return Outcome(doc, keep=False, reason="heldout")
        return Outcome(doc)

    def describe(self, st):
        n = st["removed_by_reason"].get("heldout", 0)
        if not self.files:
            return "aucune liste d'IDs exclus trouvée (data/val/*/heldout_ids.txt)"
        return f"{n} docs de la val retirés"
