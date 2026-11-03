"""Évaluation officielle : bits par octet (BPB) d'un checkpoint sur un jeu de textes.

    python -m ftb.evaluate --ckpt runs/mon_run/ckpt.pt --tokenizer tokenizer.json
    python -m ftb.evaluate ... --data data/validation.jsonl --out runs/mon_run/eval.json

BPB = (somme sur les tokens de -log2 p(token | contexte)) / (nombre d'octets UTF-8 du texte)

Chaque document est évalué séparément, précédé de <|endoftext|>. Les documents plus longs que le
contexte (512 tokens) sont lus par fenêtres glissantes au pas de 256 : chaque token est prédit avec
au moins 256 tokens de contexte (sauf au début du document). Le BPB global est la somme des bits
de tous les documents divisée par la somme de leurs octets ; le BPB par genre est donné à titre
indicatif.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from ftb import VALIDATION_PATH, load_config
from ftb.model import load_checkpoint
from ftb.tokenizer import TokenizerError, load_tokenizer

CFG = load_config()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_model_config(model_config):
    official = CFG["model"]
    if model_config != official:
        diff = {k: (model_config.get(k), v) for k, v in official.items()
                if model_config.get(k) != v}
        raise ValueError(f"configuration du modèle différente de la configuration officielle : {diff}")


def _windows(n, ctx, stride):
    """Fenêtres (début, fin, premier_index_noté) sur une suite de n tokens (EOT initial compris).

    On prédit les tokens 1..n-1. Une fenêtre [s, e) donne en entrée tokens[s:e] et note les
    cibles tokens[s+1:e+1] dont l'index est > dernier index déjà noté.
    """
    out = []
    last = 0
    while last < n - 1:
        e = min(ctx if last == 0 else last + stride, n - 1)
        s = max(0, e - ctx)
        out.append((s, e, last))
        last = e
    return out


@torch.no_grad()
def evaluate_bpb(model, tok, docs, device="cpu", batch_size=32, ctx=None, stride=None,
                 dtype=None, progress=False):
    ctx = ctx or CFG["eval"]["context_length"]
    stride = stride or CFG["eval"]["stride"]
    model.eval().to(device)
    if dtype is None:
        dtype = torch.float32
    texts = [d["text"] for d in docs]
    encoded = tok.encode_batch(texts)

    jobs = []  # (doc_idx, input_ids, targets, n_ignored)
    for i, (text, ids) in enumerate(zip(texts, encoded)):
        if tok.decode(ids) != text:
            raise TokenizerError(f"tokenizer non réversible sur le document {docs[i].get('id', i)}")
        toks = [tok.eot_id] + ids
        for s, e, last in _windows(len(toks), ctx, stride):
            inp = toks[s:e]
            tgt = toks[s + 1:e + 1]
            ignore = last - s  # cibles d'index <= last déjà notées
            jobs.append((i, inp, tgt, ignore))

    nats = [0.0] * len(docs)
    ntok = [0] * len(docs)
    t0 = time.time()
    # Tri par longueur : moins de padding
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
    for bstart in range(0, len(order), batch_size):
        chunk = [jobs[j] for j in order[bstart:bstart + batch_size]]
        L = max(len(c[1]) for c in chunk)
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        y = torch.full((len(chunk), L), -100, dtype=torch.long)
        for r, (_, inp, tgt, ignore) in enumerate(chunk):
            x[r, :len(inp)] = torch.tensor(inp)
            t = torch.tensor(tgt)
            t[:ignore] = -100
            y[r, :len(tgt)] = t
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype,
                            enabled=dtype != torch.float32):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="none").view(len(chunk), L)
        for r, (doc_i, _, _, _) in enumerate(chunk):
            nats[doc_i] += loss[r].sum().item()
            ntok[doc_i] += int((y[r] != -100).sum().item())
        if progress:
            print(f"\r{min(bstart + batch_size, len(order))}/{len(order)} fenêtres", end="",
                  file=sys.stderr)
    if progress:
        print(f" ({time.time() - t0:.0f} s)", file=sys.stderr)

    for i, text in enumerate(texts):
        assert ntok[i] == len(encoded[i]), "erreur interne : tous les tokens doivent être notés"
    nbytes = [len(t.encode("utf-8")) for t in texts]
    by_genre = defaultdict(lambda: [0.0, 0])
    for d, n, b in zip(docs, nats, nbytes):
        by_genre[d.get("genre", "tout")][0] += n
        by_genre[d.get("genre", "tout")][1] += b
    total_nats, total_bytes = sum(nats), sum(nbytes)
    return {
        "bpb": total_nats / (math.log(2) * total_bytes),
        "bpb_par_genre": {g: v[0] / (math.log(2) * v[1]) for g, v in sorted(by_genre.items())},
        "loss_par_token": total_nats / sum(ntok),
        "n_documents": len(docs),
        "n_tokens": sum(ntok),
        "n_octets": total_bytes,
        "octets_par_token": total_bytes / sum(ntok),
    }


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", default=str(VALIDATION_PATH))
    p.add_argument("--device", default=default_device())
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    model, ckpt = load_checkpoint(args.ckpt)
    check_model_config(ckpt["model_config"])
    meta = ckpt.get("meta", {})
    try:
        tok = load_tokenizer(args.tokenizer)
    except TokenizerError as e:
        sys.exit(f"ERREUR tokenizer : {e}")
    if meta.get("tokenizer_sha256") and meta["tokenizer_sha256"] != tok.sha256:
        sys.exit("ERREUR : ce tokenizer n'est pas celui utilisé pour entraîner ce checkpoint")
    docs = read_jsonl(args.data)
    res = evaluate_bpb(model, tok, docs, device=args.device, batch_size=args.batch_size,
                       progress=True)
    res.update({"ckpt": args.ckpt, "data": args.data, "tokenizer_sha256": tok.sha256})
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
