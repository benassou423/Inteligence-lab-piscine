"""Comparer deux runs : l'écart de BPB est-il réel, ou dû au hasard des documents de la val ?

    uv run python -m bench.compare runs/A runs/B [--data data/validation.jsonl] [--device mps]

1. BPB de chaque document de la val pour les deux checkpoints (calcul du kit, `ftb.evaluate`) ;
2. bootstrap apparié, stratifié par genre : on retire au hasard, avec remise, autant de documents
   que la val en a dans chaque genre (le test caché : mêmes genres, autres documents), et on
   recalcule le BPB global des deux runs sur ce tirage. 10 000 tirages ;
3. sortie : écart B − A, intervalle à 95 %, et P(A meilleur) = part des tirages où A gagne.
Ce bruit-là (choix des documents du test) s'ajoute au bruit d'entraînement (graine du mix), mesuré
par deux runs de même recette et de graines différentes (voir docs/comprendre/banc_essai.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.common import REPO, sha256_file, write_json


def doc_bits(run: Path, data: Path, device: str) -> dict:
    """Bits de chaque document (`ftb.evaluate.evaluate_bpb`, un « genre » par document), en cache."""
    cache = run / "eval_docs.json"
    key = {"ckpt_sha256": sha256_file(run / "ckpt.pt"), "data_sha256": sha256_file(data)}
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            return c
    from ftb.evaluate import evaluate_bpb, read_jsonl
    from ftb.model import load_checkpoint
    from ftb.tokenizer import load_tokenizer

    model, _ = load_checkpoint(run / "ckpt.pt")
    tok = load_tokenizer(run / "tokenizer.json")
    docs = read_jsonl(data)
    res = evaluate_bpb(model, tok, [{"text": d["text"], "genre": str(i)} for i, d in enumerate(docs)], device=device)
    nbytes = [len(d["text"].encode("utf-8")) for d in docs]
    out = {"key": key, "genre": [d.get("genre", "tout") for d in docs], "bytes": nbytes,
           "bits": [res["bpb_par_genre"][str(i)] * b for i, b in enumerate(nbytes)]}
    write_json(cache, out)
    return out


def paired_bootstrap(bits_a, bits_b, nbytes, genres, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Écart de BPB global B − A sur des tirages de documents (avec remise, par genre)."""
    a, b, n = (np.asarray(x, np.float64) for x in (bits_a, bits_b, nbytes))
    g = np.asarray(genres)
    rng = np.random.default_rng(seed)
    sa = np.zeros(n_boot)
    sb = np.zeros(n_boot)
    sn = np.zeros(n_boot)
    for genre in np.unique(g):
        idx = np.flatnonzero(g == genre)
        pick = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
        sa += a[pick].sum(1)
        sb += b[pick].sum(1)
        sn += n[pick].sum(1)
    delta = (sb - sa) / sn
    by_genre = {str(x): round(float(b[g == x].sum() / n[g == x].sum() - a[g == x].sum() / n[g == x].sum()), 4)
                for x in np.unique(g)}
    return {"bpb_a": float(a.sum() / n.sum()), "bpb_b": float(b.sum() / n.sum()),
            "delta_b_minus_a": float((b.sum() - a.sum()) / n.sum()),
            "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "p_a_better": float((delta > 0).mean()), "delta_by_genre": by_genre, "n_boot": n_boot}


def main(argv=None):
    from ftb.evaluate import default_device

    ap = argparse.ArgumentParser(description="Écart de BPB entre deux runs, avec intervalle de confiance")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--data", default=str(REPO / "data" / "validation.jsonl"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    data, device = Path(args.data), args.device or default_device()
    da, db = (doc_bits(Path(r), data, device) for r in (args.run_a, args.run_b))
    res = paired_bootstrap(da["bits"], db["bits"], da["bytes"], da["genre"], args.n_boot)
    lo, hi = res["ci95"]
    print(json.dumps({"run_a": args.run_a, "run_b": args.run_b, **res}, indent=2, ensure_ascii=False))
    verdict = ("A meilleur" if lo > 0 else "B meilleur" if hi < 0 else "indécidable sur la val seule")
    print(f"BPB A {res['bpb_a']:.4f}, B {res['bpb_b']:.4f} ; B − A = {res['delta_b_minus_a']:+.4f} "
          f"[{lo:+.4f} ; {hi:+.4f}] ; P(A meilleur) = {res['p_a_better']:.2f} → {verdict}")


if __name__ == "__main__":
    main()
