"""Tests du kit : python -m pytest tests/ -q"""

import json
import math
import random

import pytest
import torch

from ftb import load_config
from ftb.evaluate import _windows, evaluate_bpb
from ftb.model import build_model
from ftb.tokenizer import GPT4_SPLIT_PATTERN, FTBTokenizer, TokenizerError, from_merges

regex = pytest.importorskip("regex")

TEXT = open("corpus/moliere_val.txt", encoding="utf-8").read()[:60000]


# --- implémentation de référence minbpe (Karpathy), réduite à l'essentiel ----------------------
def _stats(ids, counts):
    for p in zip(ids, ids[1:]):
        counts[p] = counts.get(p, 0) + 1
    return counts


def _merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def minbpe_train(text, n_merges):
    chunks = [list(c.encode("utf-8")) for c in regex.findall(GPT4_SPLIT_PATTERN, text)]
    merges = []
    for k in range(n_merges):
        counts = {}
        for c in chunks:
            _stats(c, counts)
        pair = max(counts, key=counts.get)
        merges.append(pair)
        chunks = [_merge(c, pair, 256 + k) for c in chunks]
    return merges


def minbpe_encode(text, merges):
    rank = {p: i for i, p in enumerate(merges)}
    out = []
    for c in regex.findall(GPT4_SPLIT_PATTERN, text):
        ids = list(c.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(_stats(ids, {}), key=lambda p: rank.get(p, float("inf")))
            if pair not in rank:
                break
            ids = _merge(ids, pair, 256 + rank[pair])
        out.extend(ids)
    return out


def test_from_minbpe_matches_reference(tmp_path):
    merges = minbpe_train(TEXT[:20000], 300)
    tk = from_merges(merges)
    path = tmp_path / "tok.json"
    tk.save(str(path))
    tok = FTBTokenizer(path)
    sample = TEXT[20000:40000] + "\nÉté 2026 : 3,5 % — « ok » 🐉 中文\r\n  x"
    assert tok.encode(sample) == minbpe_encode(sample, merges)
    assert tok.decode(tok.encode(sample)) == sample
    assert tok.eot_id == 256 + 300
    with pytest.raises(TokenizerError):  # 557 tokens != 8192
        tok.validate()


def test_windows_cover_every_target_once():
    for n in [2, 3, 100, 512, 513, 514, 1000, 2049]:
        scored = []
        for s, e, last in _windows(n, 512, 256):
            assert e - s <= 512 and s >= 0 and e <= n - 1
            scored.extend(range(max(last, s) + 1, e + 1))
        assert scored == list(range(1, n)), n


def test_param_count_about_10M():
    cfg = load_config()
    n = build_model(cfg["model"]).num_params()
    assert 9.5e6 < n < 11e6, n


def test_budget():
    tr = load_config()["train"]
    assert tr["max_steps"] * tr["tokens_per_step"] >= tr["budget_tokens"]
    assert (tr["max_steps"] - 1) * tr["tokens_per_step"] < tr["budget_tokens"]


def test_bpb_matches_definition(tmp_path):
    """BPB du script = somme des -log2 p / octets, calculée à la main sur un petit modèle."""
    merges = minbpe_train(TEXT[:20000], 200)
    tk = from_merges(merges)
    tk.save(str(tmp_path / "tok.json"))
    tok = FTBTokenizer(tmp_path / "tok.json")
    torch.manual_seed(0)
    model = build_model(dict(vocab_size=tok.vocab_size, context_length=64, d_model=32, n_layer=2,
                             n_head=2, mlp_ratio=4, dropout=0.0, tie_embeddings=True)).eval()
    docs = [{"text": TEXT[i * 900:(i + 1) * 900], "genre": "g"} for i in range(4)]
    res = evaluate_bpb(model, tok, docs, ctx=64, stride=32, batch_size=3)
    # Calcul direct, fenêtre par fenêtre, sans batch ni padding
    total_bits, total_bytes = 0.0, 0
    with torch.no_grad():
        for d in docs:
            toks = [tok.eot_id] + tok.encode(d["text"])
            for s, e, last in _windows(len(toks), 64, 32):
                logits, _ = model(torch.tensor([toks[s:e]]))
                logp = torch.log_softmax(logits[0].double(), -1)
                for pos in range(s, e):
                    if pos + 1 > last:
                        total_bits -= logp[pos - s, toks[pos + 1]].item() / math.log(2)
            total_bytes += len(d["text"].encode("utf-8"))
    assert abs(res["bpb"] - total_bits / total_bytes) < 1e-4
