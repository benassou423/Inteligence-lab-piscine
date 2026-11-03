"""Modèle officiel de la compétition : transformer décodeur de type GPT-2 (~10M paramètres).

Ce fichier est la référence utilisée par l'orga pour entraîner et évaluer tous les participants.
Il n'est pas modifiable. Ton mini-GPT du bloc 1 doit être le tien : tu dois pouvoir expliquer
chaque ligne de ton code, pas de celui-ci.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        self.fc = nn.Linear(d_model, mlp_ratio * d_model, bias=False)
        self.proj = nn.Linear(mlp_ratio * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-LN + connexion résiduelle
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_layer, n_head, mlp_ratio=4,
                 dropout=0.0, tie_embeddings=True):
        super().__init__()
        self.context_length = context_length
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, mlp_ratio, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Projections de sortie des blocs : variance réduite (GPT-2)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_length, f"séquence de {T} tokens > contexte {self.context_length}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eot_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eot_id is not None and (nxt == eot_id).all():
                break
        return idx


def build_model(model_cfg):
    return GPT(**model_cfg)


def load_checkpoint(path, map_location="cpu"):
    """Charge un checkpoint produit par ftb.train. Retourne (modèle, checkpoint complet).

    checkpoint["model_config"] : configuration du modèle ; checkpoint["meta"] : métadonnées du run.
    """
    # weights_only=True : un checkpoint ne peut pas exécuter de code au chargement
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ckpt["model_config"])
    # Les soumissions stockent les poids en float16 (taille / 2) : on les remet en float32
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    return model, ckpt
