"""Étape `exact_dedup` : doublons exacts, sur toutes les sources de la config.

Texte normalisé (minuscules, espaces compactés) -> hash 128 bits xxh3.
On garde la première occurrence dans l'ordre : sources dans l'ordre de la
config, fichiers triés, lignes dans l'ordre. Étape « globale » en deux passes :
1. chaque worker calcule les hashes d'un fichier ;
2. le processus principal décide (numpy), puis les workers réécrivent les fichiers.
"""
import bisect
from collections import Counter

import numpy as np
import xxhash

from pipeline.clean.base import Outcome, Stage


def normalize(text):
    """Minuscules et espaces compactés (y compris insécables)."""
    return " ".join(text.lower().split())


def text_key(text):
    """Empreinte 128 bits du texte normalisé (16 octets)."""
    return xxhash.xxh3_128_digest(normalize(text).encode("utf-8"))


class ExactDedupStage(Stage):
    name = "exact_dedup"
    kind = "global"
    filters = True

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.top_k = int(self.params.get("top_k", 20))
        self.units = []  # [(unit_key, source, n)] dans l'ordre global
        self.offsets = []  # index global du premier doc de chaque unité
        self.unit_index = {}
        self.keep = self.copies = self.first = None
        self.top_positions = {}  # index global -> rang dans le top des doublons
        self.top = []
        self.n_groups_dup = 0

    # ---- passe 1 (workers)
    def key(self, doc):
        return text_key(doc.get("text") or "")

    # ---- décision (processus principal)
    def plan(self, scanned):
        """scanned : [(unit_key, source, tableau (n, 2) uint64)] dans l'ordre global."""
        self.units = [(k, s, len(a)) for k, s, a in scanned]
        self.offsets, pos = [], 0
        for _, _, n in self.units:
            self.offsets.append(pos)
            pos += n
        self.unit_index = {k: i for i, (k, _, _) in enumerate(self.units)}
        n_total = pos
        if n_total == 0:
            self.keep = np.zeros(0, bool)
            self.copies = np.zeros(0, np.int32)
            self.first = np.zeros(0, np.int64)
            return xxhash.xxh64(b"").hexdigest()
        keys = np.concatenate([a for _, _, a in scanned]).astype(np.uint64, copy=False)
        hi, lo = keys[:, 0], keys[:, 1]
        idx = np.arange(n_total, dtype=np.int64)
        order = np.lexsort((idx, lo, hi))  # tri par hash, puis par position : la 1re occurrence d'abord
        hs, ls = hi[order], lo[order]
        new = np.ones(n_total, bool)
        new[1:] = (hs[1:] != hs[:-1]) | (ls[1:] != ls[:-1])
        gid_sorted = np.cumsum(new) - 1
        first_of_group = order[new]  # position de la première occurrence de chaque groupe
        counts = np.bincount(gid_sorted)
        group = np.empty(n_total, np.int64)
        group[order] = gid_sorted
        self.keep = np.zeros(n_total, bool)
        self.keep[first_of_group] = True
        self.copies = counts[group].astype(np.int32)
        self.first = first_of_group[group]
        # Textes les plus dupliqués
        dup_groups = np.flatnonzero(counts > 1)
        top = dup_groups[np.argsort(-counts[dup_groups], kind="stable")][: self.top_k]
        unit_sources = np.repeat(np.arange(len(self.units)), [n for _, _, n in self.units])
        self.top = []
        for rank, g in enumerate(top):
            members = np.flatnonzero(group == g)
            by_source = Counter(self.units[u][1] for u in unit_sources[members])
            self.top.append({"rank": rank + 1, "copies": int(counts[g]),
                             "first": self.position(int(first_of_group[g])),
                             "by_source": dict(by_source.most_common())})
        self.top_positions = {int(first_of_group[g]): r for r, g in enumerate(top)}
        self.n_groups_dup = int(len(dup_groups))
        h = xxhash.xxh64()
        h.update(keys.tobytes())
        return h.hexdigest()

    def position(self, g):
        """Index global -> 'source/part-xxxxx.jsonl.zst#ligne'."""
        u = bisect.bisect_right(self.offsets, g) - 1
        return f"{self.units[u][0]}#{g - self.offsets[u]}"

    # ---- passe 2 (workers)
    def apply_at(self, doc, unit_key, line):
        g = self.offsets[self.unit_index[unit_key]] + line
        meta = doc.setdefault("meta", {})
        if self.keep[g]:
            copies = int(self.copies[g])
            meta["exact_dedup_copies"] = copies
            capture = None
            if copies > 1:
                capture = {"pos": self.position(g), "id": doc["id"]}
                if g in self.top_positions:
                    capture["top"] = self.top_positions[g]
                    capture["text"] = (doc.get("text") or "")[:300]
            return Outcome(doc, counters={"kept_with_copies": int(copies > 1)}, capture=capture)
        meta["exact_dedup_first"] = self.position(int(self.first[g]))
        meta["exact_dedup_copies"] = int(self.copies[g])
        return Outcome(doc, keep=False, reason="exact_duplicate")

    # ---- fin d'étape (processus principal)
    def finalize(self, captures, samples):
        """Complète les échantillons (ID de la copie gardée) et construit le rapport."""
        pos_to_id = {c["pos"]: c["id"] for c in captures}
        for rec in samples.get("rejected", []):
            first = rec.get("meta", {}).get("exact_dedup_first")
            if first in pos_to_id:
                rec["meta"]["exact_dedup_first_id"] = pos_to_id[first]
        texts = {c["top"]: c for c in captures if "top" in c}
        for t in self.top:
            c = texts.get(t["rank"] - 1, {})
            t["id"] = c.get("id")
            t["text"] = c.get("text", "")
        report = {"duplicate_groups": self.n_groups_dup, "top_duplicates": self.top}
        lines = [f"Groupes de textes dupliqués : **{report['duplicate_groups']}**. Les plus fréquents :", ""]
        if self.top:
            lines.append("| Copies | Sources | Début du texte (gardé) |")
            lines.append("|--:|---|---|")
            for t in self.top[:10]:
                src = ", ".join(f"{k} {v}" for k, v in t["by_source"].items())
                snippet = " ".join(t["text"].split())[:120].replace("|", "\\|")
                lines.append(f"| {t['copies']} | {src} | {snippet} |")
        else:
            lines.append("_Aucun doublon exact._")
        return report, "\n".join(lines)

    def describe(self, st):
        n = st["removed_by_reason"].get("exact_duplicate", 0)
        return f"{n} doublons exacts retirés" if n else "aucun doublon"
