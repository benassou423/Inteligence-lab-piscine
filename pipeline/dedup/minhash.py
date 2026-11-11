"""Quasi-doublons : MinHash, bandes LSH, puis Jaccard vérifié avant retrait.

Les groupes de cinq mots sont encodés par hash 64 bits ; la signature MinHash
ne sert qu'à chercher des candidats. Le témoin de chaque retrait est un texte
effectivement conservé, jamais un voisin déjà retiré (pas de chaîne transitive).
Les signatures et ensembles de vérification restent sur disque, par fichier.
"""
import bisect
import logging
import re
import shutil
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xxhash
from datasketch import MinHash

from pipeline.clean.base import Outcome, Stage
from pipeline.io import read_docs


log = logging.getLogger('pipeline')
WORDS = re.compile(r'\w+', re.UNICODE)
MAX32 = np.iinfo(np.uint32).max


def shingles(text, size=5, min_words=50):
    """Groupes de mots uniques, sans ponctuation ; ne modifie pas le texte source."""
    words = WORDS.findall(text.lower())
    if len(words) < max(size, min_words):
        return np.empty(0, np.uint64)
    hashes = {xxhash.xxh3_64_intdigest(' '.join(words[i:i + size]).encode('utf-8'))
              for i in range(len(words) - size + 1)}
    return np.array(sorted(hashes), dtype=np.uint64)


def _hash32(value):
    return xxhash.xxh32_intdigest(value)


def jaccard(a, b):
    """Ensembles déjà triés et uniques ; le ratio des tailles borne la similarité."""
    if not len(a) or not len(b):
        return 0.
    common = len(np.intersect1d(a, b, assume_unique=True))
    return common / (len(a) + len(b) - common)


def band_hashes(signatures, band, rows):
    """Même hash de bande lors du tri initial et de la requête LSH."""
    h = np.full(len(signatures), 14695981039346656037, dtype=np.uint64)
    for col in range(band * rows, (band + 1) * rows):
        h ^= signatures[:, col].astype(np.uint64)
        h *= np.uint64(1099511628211)
    return h


class MinHashDedupStage(Stage):
    name = 'minhash_dedup'
    kind = 'global'
    filters = True
    key_dtype = np.uint32

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.num_perm = int(self.params.get('num_perm', 128))
        self.bands = int(self.params.get('bands', 16))
        self.rows = int(self.params.get('rows', 8))
        self.seed = int(self.params.get('seed', 1337))
        self.threshold = float(self.params.get('threshold', .8))
        self.shingle_size = int(self.params.get('shingle_size', 5))
        self.min_words = int(self.params.get('min_words', 50))
        self.reserve_bytes = int(self.params.get('reserve_bytes', 2_000_000_000))
        if (self.num_perm != self.bands * self.rows or self.num_perm < 1 or self.bands < 1
                or self.rows < 1 or not 0 < self.threshold <= 1 or self.shingle_size < 1
                or self.min_words < self.shingle_size):
            raise ValueError('Paramètres MinHash/LSH invalides')
        self.key_columns = self.num_perm + 1  # dernière colonne : taille de l'ensemble
        self.permutations = MinHash(num_perm=self.num_perm, seed=self.seed, scheme='legacy').permutations
        self.units, self.offsets, self.unit_index = [], [], {}
        self.keep = self.first = self.similarity = self.copies = None
        self.report = {}
        self._maps = OrderedDict()

    def key(self, doc):
        values = shingles(doc.get('text') or '', self.shingle_size, self.min_words)
        result = np.full(self.key_columns, MAX32, dtype=np.uint32)
        result[-1] = len(values)
        if len(values):
            m = MinHash(num_perm=self.num_perm, seed=self.seed, hashfunc=_hash32,
                        permutations=self.permutations, scheme='legacy')
            # Borner la matrice temporaire même pour un livre entier.
            for start in range(0, len(values), 2048):
                m.update_batch([int(v).to_bytes(8, 'little') for v in values[start:start + 2048]])
            result[:-1] = m.hashvalues.astype(np.uint32)
        return result.tobytes()

    def position(self, g):
        unit = bisect.bisect_right(self.offsets, g) - 1
        return f'{self.units[unit][0]}#{g - self.offsets[unit]}'

    def _cache_path(self, key, suffix):
        path = Path(self.work_dir) / '_verify' / (key + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _verification_sets(self, scanned, candidates):
        """Relire uniquement les fichiers contenant des candidats ; un ensemble/doc."""
        for u, (key, _, array) in enumerate(scanned):
            start = self.offsets[u]
            flags = candidates[start:start + len(array)]
            if not np.any(flags):
                continue
            path = self._cache_path(key, '.bin')
            index = np.zeros((len(array), 2), dtype=np.int64)
            count = 0
            with path.open('wb') as f:
                for row, doc in enumerate(read_docs(Path(self.input_dir) / key)):
                    if not flags[row]:
                        continue
                    values = shingles(doc.get('text') or '', self.shingle_size, self.min_words)
                    if len(values) != int(array[row, -1]):
                        raise ValueError(f'Entrée modifiée depuis le scan : {key}#{row}')
                    index[row] = (count, len(values))
                    values.tofile(f)
                    count += len(values)
                    if row % 128 == 0 and shutil.disk_usage(self.work_dir).free < self.reserve_bytes:
                        raise OSError('Réserve disque atteinte pendant MinHash ; reprise possible')
            np.save(self._cache_path(key, '.index.npy'), index)
            log.info('[minhash] ensembles vérifiés : fichier %d/%d, %d candidats', u + 1, len(scanned), int(flags.sum()))

    def _values(self, global_position):
        unit = bisect.bisect_right(self.offsets, global_position) - 1
        if unit not in self._maps:
            key = self.units[unit][0]
            self._maps[unit] = (np.load(self._cache_path(key, '.index.npy'), mmap_mode='r'),
                                np.memmap(self._cache_path(key, '.bin'), dtype=np.uint64, mode='r'))
            while len(self._maps) > 24:
                _, old = self._maps.popitem(last=False)
                for a in old:
                    a._mmap.close()
        self._maps.move_to_end(unit)
        index, data = self._maps[unit]
        start, length = index[global_position - self.offsets[unit]]
        return data[start:start + length]

    def plan(self, scanned):
        self.units = [(key, source, len(a)) for key, source, a in scanned]
        self.unit_index = {key: i for i, (key, _, _) in enumerate(self.units)}
        self.offsets = []
        n = 0
        h = xxhash.xxh3_128()
        for _, _, a in scanned:
            self.offsets.append(n)
            n += len(a)
            h.update(a.tobytes())
        self.keep = np.ones(n, dtype=bool)
        self.first = np.arange(n, dtype=np.int64)
        self.similarity = np.zeros(n, dtype=np.float64)
        self.copies = np.ones(n, dtype=np.int32)
        candidates = np.zeros(n, dtype=bool)
        eligible = np.concatenate([a[:, -1] > 0 for _, _, a in scanned]) if n else np.zeros(0, bool)
        band_path = Path(self.work_dir) / '_bands.npy'
        band_matrix = np.lib.format.open_memmap(band_path, mode='w+', dtype=np.uint64, shape=(n, self.bands))
        for band in range(self.bands):
            for unit, (_, _, a) in enumerate(scanned):
                start = self.offsets[unit]
                band_matrix[start:start + len(a), band] = band_hashes(a, band, self.rows)
            valid = np.flatnonzero(eligible)
            order = valid[np.argsort(band_matrix[valid, band], kind='stable')]
            hashes = band_matrix[order, band]
            equal = hashes[1:] == hashes[:-1]
            candidates[order[:-1][equal]] = True
            candidates[order[1:][equal]] = True
            log.info('[minhash] bande LSH %d/%d, %d candidats', band + 1, self.bands, int(candidates.sum()))
        self._verification_sets(scanned, candidates)
        db_path = Path(self.work_dir) / '_lsh.sqlite'
        db_path.unlink(missing_ok=True)
        db = sqlite3.connect(db_path)
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE lsh (band INTEGER, hash BLOB, doc INTEGER, PRIMARY KEY(band, hash, doc)) WITHOUT ROWID')
        comparisons = 0
        try:
            for done, g in enumerate(np.flatnonzero(candidates)):
                keys = [(band, int(value).to_bytes(8, 'little')) for band, value in enumerate(band_matrix[g])]
                neighbors = set()
                for band, value in keys:
                    neighbors.update(row[0] for row in db.execute('SELECT doc FROM lsh WHERE band=? AND hash=?', (band, value)))
                a = self._values(int(g)).copy()  # sa durée de vie dépasse les évictions du cache
                for other in sorted(neighbors):
                    b = self._values(other)
                    if min(len(a), len(b)) / max(len(a), len(b)) < self.threshold:
                        continue
                    comparisons += 1
                    similarity = jaccard(a, b)
                    if similarity >= self.threshold:
                        self.keep[g] = False
                        self.first[g] = other
                        self.similarity[g] = similarity
                        self.copies[other] += 1
                        break
                if self.keep[g]:
                    db.executemany('INSERT INTO lsh VALUES (?, ?, ?)', [(band, value, int(g)) for band, value in keys])
                if done % 10000 == 0:
                    db.commit()
                    log.info('[minhash] vérification %d/%d, %d comparaisons', done, int(candidates.sum()), comparisons)
            db.commit()
        finally:
            db.close()
            self.teardown()
            band_matrix._mmap.close()
        band_path.unlink(missing_ok=True)
        db_path.unlink(missing_ok=True)
        # Les décisions sont maintenant en mémoire ; ces ensembles temporaires
        # seront recalculés depuis l'entrée si le plan doit être rejoué.
        shutil.rmtree(Path(self.work_dir) / '_verify', ignore_errors=True)
        removed = np.flatnonzero(~self.keep)
        self.audit_positions = set(sorted(removed, key=lambda g: xxhash.xxh64_intdigest(str(int(g)).encode()))[:50])
        self.audit_witnesses = {int(self.first[g]) for g in self.audit_positions}
        self.report = {'eligible_documents': int(eligible.sum()), 'candidate_documents': int(candidates.sum()),
                       'verified_comparisons': comparisons, 'removed': len(removed),
                       'threshold': self.threshold, 'min_words': self.min_words,
                       'shingle_size': self.shingle_size, 'num_perm': self.num_perm,
                       'bands': self.bands, 'rows': self.rows, 'seed': self.seed,
                       'minimum_removed_similarity': float(self.similarity[removed].min()) if len(removed) else None,
                       'candidate_probability_at_threshold': 1 - (1 - self.threshold ** self.rows) ** self.bands,
                       'verification': 'Jaccard des groupes de mots uniques, hashes 64 bits',
                       'limitations': 'LSH probabiliste ; textes courts ignorés ; collisions de hash possibles'}
        h.update(self.keep.tobytes())
        h.update(self.first.tobytes())
        return h.hexdigest()

    def apply_at(self, doc, unit_key, line):
        g = self.offsets[self.unit_index[unit_key]] + line
        meta = dict(doc.get('meta') or {})
        if self.keep[g]:
            meta['near_dedup_copies'] = int(self.copies[g])
            capture = None
            if self.copies[g] > 1:
                capture = {'pos': self.position(g), 'id': doc['id']}
                if g in self.audit_witnesses:
                    capture['text'] = (doc.get('text') or '')[:2000]
            return Outcome({**doc, 'meta': meta}, capture=capture)
        meta.update(near_dedup_first=self.position(int(self.first[g])),
                    near_dedup_similarity=float(self.similarity[g]))
        capture = None
        if g in self.audit_positions:
            capture = {'removed': doc['id'], 'first': meta['near_dedup_first'],
                       'similarity': meta['near_dedup_similarity'], 'text': (doc.get('text') or '')[:2000]}
        return Outcome({**doc, 'meta': meta}, keep=False, reason='near_duplicate', capture=capture)

    def finalize(self, captures, samples):
        witnesses = {c['pos']: c for c in captures if 'pos' in c}
        for rec in samples.get('rejected', []):
            meta = rec.get('meta') or {}
            witness = witnesses.get(meta.get('near_dedup_first'))
            if witness:
                meta['near_dedup_first_id'] = witness['id']
        pairs = [{**c, 'kept': witnesses.get(c['first'])} for c in captures if 'removed' in c]
        report = {**self.report, 'audit_pairs': pairs}
        text = (f"{report['removed']} quasi-doublons retirés après vérification à Jaccard ≥ {self.threshold}. "
                f"{report['candidate_documents']} candidats ; {report['verified_comparisons']} comparaisons. "
                "Chaque témoin est conservé. Recherche LSH probabiliste, pas de garantie de rappel total.")
        return report, text

    def teardown(self):
        for arrays in self._maps.values():
            for a in arrays:
                a._mmap.close()
        self._maps.clear()

    def describe(self, st):
        return f"{st['removed_by_reason'].get('near_duplicate', 0)} quasi-doublons retirés"
