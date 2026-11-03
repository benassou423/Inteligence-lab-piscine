"""Tests du bigramme : comptes, probabilités, lissage, modèle uniforme, BPB, tirage."""
import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.bigram import BigramModel


def test_counts_are_exact_on_small_string():
    # "banana" contient 5 paires : ba, an, na, an, na.
    model = BigramModel(vocab_size=256).fit(list(b"banana"))
    a, b, n = ord("a"), ord("b"), ord("n")
    assert model.counts[b, a] == 1
    assert model.counts[a, n] == 2
    assert model.counts[n, a] == 2
    assert model.counts.sum() == 5  # donc toutes les autres cases valent 0


def test_each_row_of_probabilities_sums_to_one():
    model = BigramModel(vocab_size=256).fit(list("Le Misanthrope, comédie".encode("utf-8")))
    assert torch.allclose(model.probs.sum(dim=1), torch.ones(256, dtype=torch.float64))


def test_smoothing_gives_unseen_pairs_a_nonzero_probability():
    ids = list(b"banana")
    a, q = ord("a"), ord("q")  # la paire (a, q) n'apparaît jamais dans "banana"
    smoothed = BigramModel(vocab_size=256, alpha=1.0).fit(ids)
    # Ligne a : 2 paires vues (a→n deux fois). Laplace : (0 + 1) / (2 + 1 × 256).
    assert smoothed.probs[a, q] == pytest.approx(1 / 258)
    assert math.isfinite(smoothed.nll_bits([a, q]))
    # Sans lissage, la même paire a une probabilité nulle et coûte une infinité de bits.
    raw = BigramModel(vocab_size=256, alpha=0.0).fit(ids)
    assert raw.probs[a, q] == 0
    assert math.isinf(raw.nll_bits([a, q]))


@pytest.mark.parametrize("vocab_size", [27, 256])
def test_uniform_model_costs_log2_v_bits_per_token(vocab_size):
    # Aucune paire vue : il ne reste que le lissage, donc chaque ligne vaut 1/V partout.
    model = BigramModel(vocab_size).fit([])
    ids = torch.randint(vocab_size, (1000,), generator=torch.Generator().manual_seed(0))
    bits_per_token = model.nll_bits(ids) / (len(ids) - 1)  # le premier id n'est pas prédit
    assert bits_per_token == pytest.approx(math.log2(vocab_size))  # 8 bits pour 256 octets


def test_bpb_matches_the_transformer_formula():
    # Le transformer donne une loss moyenne en nats : BPB = loss × nb_tokens / (ln 2 × nb_octets).
    model = BigramModel(vocab_size=256).fit(list("Le Tartuffe est une comédie de Molière.".encode("utf-8")))
    text = "Tartuffe, ou l'Imposteur".encode("utf-8")
    ids = torch.tensor(list(text))
    # Log-probabilités du bigramme, une ligne par position. cross_entropy refait un softmax,
    # qui ne les change pas : chaque ligne de probabilités somme déjà à 1.
    logits = torch.log(model.probs[ids[:-1]])
    loss = F.cross_entropy(logits, ids[1:])  # moyenne en nats sur les len(ids) - 1 prédictions
    expected = loss.item() * (len(ids) - 1) / (math.log(2) * len(text))
    assert model.bpb(ids, n_bytes=len(text)) == pytest.approx(expected)


def test_sample_is_reproducible_with_the_same_seed():
    model = BigramModel(vocab_size=256).fit(list("Le Bourgeois gentilhomme".encode("utf-8")))
    first = model.sample(start_id=ord("L"), n=50, seed=42)
    assert len(first) == 50
    assert first == model.sample(start_id=ord("L"), n=50, seed=42)  # même seed → même suite
    assert first != model.sample(start_id=ord("L"), n=50, seed=7)  # autre seed → autre suite


def test_sample_follows_the_probability_table():
    # Sans lissage, "abcabc" donne a→b, b→c, c→a avec probabilité 1 : chaque tirage est forcé.
    model = BigramModel(vocab_size=256, alpha=0.0).fit(list(b"abcabc"))
    assert bytes(model.sample(start_id=ord("a"), n=6, seed=0)) == b"bcabca"
