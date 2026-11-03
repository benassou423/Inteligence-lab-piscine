"""Préparation du dataset : tes textes -> flux de tokens `train.bin` + empreinte (hash).

    python -m ftb.prepare --inputs mes_donnees/*.jsonl --tokenizer tokenizer.json --out prepared/

Formats d'entrée acceptés (dans l'ordre où tu les donnes, puis dans l'ordre des lignes) :
  - .jsonl / .jsonl.gz : un document par ligne, champ "text" ;
  - .txt : un fichier = un document ;
  - .parquet : une ligne = un document, colonne "text" (nécessite pyarrow).

Chaque document est encodé puis suivi du token <|endoftext|>. L'ORDRE EST CONSERVÉ : c'est toi qui
décides du mélange, des répétitions et de l'ordre (par exemple garder le meilleur pour la fin).
L'entraînement lit ce flux dans l'ordre, à un mélange local près (voir README, « Ordre des données »).

Seuls les `budget_tokens` premiers tokens (100M) sont vus. Si ton flux est plus court, il est relu
depuis le début jusqu'à atteindre le budget (c'est alors une répétition non choisie : mieux vaut
décider toi-même quoi répéter).
"""

import argparse
import glob
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from ftb import KIT_VERSION, load_config

CFG = load_config()


def _expand(inputs):
    files = []
    for pattern in inputs:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            raise FileNotFoundError(f"aucun fichier pour « {pattern} »")
        files.extend(matches)
    return files


def iter_documents(inputs):
    """Itère sur les textes de tous les fichiers, dans l'ordre."""
    for path in _expand(inputs):
        if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        text = json.loads(line)["text"]
                        if text:
                            yield text
        elif path.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text:
                yield text
        elif path.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=["text"], batch_size=1000):
                for text in batch.column(0).to_pylist():
                    if text:
                        yield text
        else:
            raise ValueError(f"format non reconnu : {path} (.jsonl, .jsonl.gz, .txt, .parquet)")


def prepare(inputs, tokenizer_path, out_dir, batch_docs=2000):
    from ftb.tokenizer import load_tokenizer

    tok = load_tokenizer(tokenizer_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / "train.bin"
    files = _expand(inputs)
    h = hashlib.sha256()
    n_docs = n_tokens = n_bytes = 0
    t0 = time.time()
    eot = np.array([tok.eot_id], dtype=np.uint16)

    def flush(batch, f):
        nonlocal n_tokens
        for ids in tok.encode_batch(batch):
            arr = np.concatenate([np.asarray(ids, dtype=np.uint16), eot])
            buf = arr.tobytes()
            f.write(buf)
            h.update(buf)
            n_tokens += len(arr)

    with open(bin_path, "wb") as f:
        batch = []
        for text in iter_documents(files):
            batch.append(text)
            n_docs += 1
            n_bytes += len(text.encode("utf-8"))
            if len(batch) >= batch_docs:
                flush(batch, f)
                batch = []
                rate = n_bytes / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r{n_docs:,} docs, {n_tokens:,} tokens, {rate:.1f} Mo/s", end="",
                      file=sys.stderr)
        if batch:
            flush(batch, f)
    print(file=sys.stderr)

    budget = CFG["train"]["budget_tokens"]
    import tokenizers

    meta = {
        "kit_version": KIT_VERSION,
        "tokenizers_version": tokenizers.__version__,
        "dataset_sha256": h.hexdigest(),
        "tokenizer_sha256": tok.sha256,
        "n_documents": n_docs,
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
        "budget_tokens": budget,
        "passes_over_data": round(budget / max(n_tokens, 1), 3),
        "inputs": [{"path": p, "size": Path(p).stat().st_size} for p in files],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(out / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def dataset_sha256(bin_path):
    h = hashlib.sha256()
    with open(bin_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class TokenStream:
    """Lit train.bin dans l'ordre, découpé en séquences de seq_len + 1 tokens.

    Les séquences sont prises bout à bout : la séquence j couvre les tokens [j*T, j*T + T].
    À l'intérieur de chaque fenêtre de `shuffle_window_steps` pas, l'ordre des séquences est mélangé
    avec une graine fixe (des batchs moins corrélés), mais les fenêtres se suivent dans l'ordre
    du fichier : l'ordre global de tes données est respecté à l'échelle de ~1,6M tokens.
    """

    def __init__(self, bin_path, batch_size, seq_len, seed, shuffle_window_steps, total_steps):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.B, self.T = batch_size, seq_len
        self.n_seq = (len(self.data) - 1) // seq_len
        if self.n_seq < 1:
            raise ValueError("train.bin trop court")
        self.seed = seed
        self.window = shuffle_window_steps * batch_size  # séquences par fenêtre
        self.total = total_steps * batch_size  # séquences vues pendant tout l'entraînement
        self._perms = {}

    def _perm(self, w):
        # La dernière fenêtre est plus courte : on ne mélange que les séquences du budget,
        # pour que ce soient exactement les premiers tokens du flux qui soient vus.
        if w not in self._perms:
            size = min(self.window, self.total - w * self.window)
            order = list(range(size))
            random.Random(self.seed * 1_000_003 + w).shuffle(order)  # stable entre versions
            self._perms = {w: order}
        return self._perms[w]

    def batch(self, step):
        """Batch du pas `step` : tableau (B, T+1) d'int64."""
        out = np.empty((self.B, self.T + 1), dtype=np.int64)
        for b in range(self.B):
            k = step * self.B + b
            w, r = divmod(k, self.window)
            j = (w * self.window + self._perm(w)[r]) % self.n_seq
            out[b] = self.data[j * self.T: j * self.T + self.T + 1]
        return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.prepare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    meta = prepare(args.inputs, args.tokenizer, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "inputs"}, indent=2,
                     ensure_ascii=False))
    if meta["n_tokens"] < meta["budget_tokens"]:
        print(f"\nattention : {meta['n_tokens']:,} tokens < budget {meta['budget_tokens']:,}. "
              f"Le flux sera relu {meta['passes_over_data']} fois.", file=sys.stderr)


if __name__ == "__main__":
    main()
