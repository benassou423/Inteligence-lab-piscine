"""Tests du mini-GPT (minigpt/gpt.py) : briques maison comparées à PyTorch, puis comportement."""
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from minigpt import gpt
from minigpt.gpt import GPT, CausalSelfAttention, GPTConfig, LayerNorm

# Modèle minuscule : tests en quelques secondes. Sans dropout, les sorties sont déterministes.
CONFIG = GPTConfig(vocab_size=16, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


@pytest.fixture(autouse=True)
def four_cpu_threads():
    """Le bloc 1 tourne sur CPU avec 4 threads au plus ; on remet la valeur d'avant après chaque test."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


def test_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(CONFIG)
    idx = torch.randint(0, CONFIG.vocab_size, (8, 10))  # 10 tokens < block_size : le masque est découpé
    logits, loss = model(idx)
    assert logits.shape == (8, 10, CONFIG.vocab_size)
    assert loss is None
    targets = torch.randint(0, CONFIG.vocab_size, (8, 10))  # cibles tirées au hasard
    _, loss = model(idx, targets)
    assert loss.shape == ()  # un seul nombre : la moyenne
    # Petits poids à l'init : prédiction presque uniforme, donc loss ≈ ln(vocab_size).
    assert abs(loss.item() - math.log(CONFIG.vocab_size)) < 0.05


def test_num_params_matches_formula():
    V, T, C, L = CONFIG.vocab_size, CONFIG.block_size, CONFIG.n_embd, CONFIG.n_layer
    # Embeddings (la matrice partagée compte une fois) + LayerNorm finale + L blocs de 12·C² + 13·C.
    expected = V * C + T * C + 2 * C + L * (12 * C * C + 13 * C)
    assert GPT(CONFIG).num_params() == expected


def test_layernorm_matches_pytorch():
    torch.manual_seed(0)
    ln = LayerNorm(32)
    with torch.no_grad():  # gain et biais quelconques, pour tester aussi la partie apprise
        ln.gamma.normal_()
        ln.beta.normal_()
    x = 3 * torch.randn(2, 5, 32) + 1  # moyenne et variance loin de 0 et 1
    expected = F.layer_norm(x, (32,), ln.gamma, ln.beta, eps=1e-5)
    assert torch.allclose(ln(x), expected, atol=1e-5)


def test_gelu_matches_pytorch():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(gpt.gelu(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_attention_matches_scaled_dot_product_attention():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CONFIG).eval()
    B, T, C, H = 2, 10, CONFIG.n_embd, CONFIG.n_head
    x = torch.randn(B, T, C)

    def split_heads(t):  # (B, T, C) -> (B, H, T, C // H), comme dans le module
        return t.reshape(B, T, H, C // H).transpose(1, 2)

    # Mêmes poids q, k, v et proj que le module ; seul le cœur de l'attention est calculé par PyTorch.
    q, k, v = split_heads(attn.query(x)), split_heads(attn.key(x)), split_heads(attn.value(x))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = attn.proj(y.transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(attn(x), expected, atol=1e-5)


def test_full_model_matches_pytorch_reference():
    # Tout le forward refait avec les fonctions de PyTorch et les mêmes poids : un oubli dans
    # l'assemblage (positions, résiduelle, ordre pre-LN, GELU, LayerNorm finale) se verrait ici.
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    B, T, C, H = 2, 10, CONFIG.n_embd, CONFIG.n_head
    idx = torch.randint(0, CONFIG.vocab_size, (B, T))

    def ln(x, norm):  # LayerNorm de PyTorch, avec le gain et le biais du module
        return F.layer_norm(x, (C,), norm.gamma, norm.beta, eps=1e-5)

    def split_heads(t):  # (B, T, C) -> (B, H, T, C // H)
        return t.reshape(B, T, H, C // H).transpose(1, 2)

    x = model.tok_emb(idx) + model.pos_emb.weight[:T]
    for block in model.blocks:
        h = ln(x, block.ln1)
        q, k, v = split_heads(block.attn.query(h)), split_heads(block.attn.key(h)), split_heads(block.attn.value(h))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + block.attn.proj(y)
        x = x + block.mlp.proj(F.gelu(block.mlp.fc(ln(x, block.ln2)), approximate="tanh"))
    expected = ln(x, model.ln_f) @ model.tok_emb.weight.T  # tête de sortie = embeddings (weight tying)
    assert torch.allclose(model(idx)[0], expected, atol=1e-5)


def test_future_token_does_not_change_past_logits():
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    idx = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.block_size))
    logits, _ = model(idx)
    for t in range(CONFIG.block_size):
        changed = idx.clone()
        changed[0, t] = (idx[0, t] + 1) % CONFIG.vocab_size  # un autre token à la position t
        new_logits, _ = model(changed)
        assert torch.equal(new_logits[:, :t], logits[:, :t])  # positions < t : rien ne bouge
        assert not torch.allclose(new_logits[:, t], logits[:, t])  # position t : le changement se voit


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    model = GPT(CONFIG)
    data = torch.randint(0, CONFIG.vocab_size, (4, CONFIG.block_size + 1))
    # Débuts tous différents : à la position 0, le modèle ne voit qu'un token ; deux séquences qui
    # commencent pareil puis divergent rendraient la cible impossible à deviner.
    data[:, 0] = torch.arange(4)
    x, y = data[:, :-1], data[:, 1:]  # la cible est le token suivant
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_generate_length_and_block_size():
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    prompt = torch.randint(0, CONFIG.vocab_size, (2, 3))
    out = model.generate(prompt, max_new_tokens=30)  # 3 + 30 > block_size : le contexte doit être coupé
    assert out.shape == (2, 33)
    assert torch.equal(out[:, :3], prompt)  # la sortie commence par le prompt
    assert 0 <= out.min() and out.max() < CONFIG.vocab_size
    # top_k = 1 : on prend toujours le token le plus probable, donc deux générations sont identiques.
    greedy = model.generate(prompt, 10, top_k=1)
    assert torch.equal(greedy, model.generate(prompt, 10, top_k=1))
    # Température minuscule : logits ÷ 1e-4, le softmax met tout sur le meilleur token = glouton.
    assert torch.equal(model.generate(prompt, 10, temperature=1e-4), greedy)
    # top_k plus grand que le vocabulaire : on garde tous les tokens, comme sans top_k.
    torch.manual_seed(1)
    big_k = model.generate(prompt, 10, top_k=CONFIG.vocab_size + 5)
    torch.manual_seed(1)
    assert torch.equal(big_k, model.generate(prompt, 10))
    # Prompt plus long que block_size : seuls les block_size DERNIERS tokens comptent.
    long_prompt = torch.randint(0, CONFIG.vocab_size, (2, 40))
    tail = long_prompt[:, -CONFIG.block_size :]
    assert torch.equal(model.generate(long_prompt, 5, top_k=1)[:, -5:], model.generate(tail, 5, top_k=1)[:, -5:])
    with pytest.raises(AssertionError):  # forward refuse une séquence plus longue que block_size
        model(torch.zeros(1, CONFIG.block_size + 1, dtype=torch.long))


def test_no_black_box_in_model_code():
    source = Path(gpt.__file__).read_text(encoding="utf-8")
    forbidden = ["nn.Transformer", "MultiheadAttention", "scaled_dot_product_attention", "nn.LayerNorm",
                 "layer_norm", "nn.GELU", "F.gelu"]
    for name in forbidden:
        assert name not in source, f"{name} est une boîte noire interdite dans minigpt/gpt.py"
