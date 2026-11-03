"""Mini-GPT : un transformer décodeur écrit à la main, sans boîte noire.

Le trajet d'un batch de B séquences de T tokens :
  1. embeddings : chaque token devient un vecteur de taille n_embd, auquel on ajoute
     le vecteur de sa position ;
  2. n_layer blocs : l'attention causale (chaque token va chercher de l'information
     dans les tokens passés, lui compris), puis le MLP (chaque token traite seul ce qu'il a reçu) ;
  3. LayerNorm finale puis tête de sortie : un score (logit) pour chaque token du
     vocabulaire, qui donne, après softmax, la probabilité du token suivant.

On ne prend à PyTorch que des briques simples : nn.Linear, nn.Embedding, nn.Dropout,
les opérations sur tenseurs (produit matriciel, softmax...), la cross-entropy et le
calcul automatique des gradients.
Inspiré de « Let's build GPT » (A. Karpathy, 2023) et de nanoGPT.
Démo : uv run python -m minigpt.gpt
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass  # Python écrit le constructeur à partir des champs : GPTConfig(vocab_size=..., n_layer=...)
class GPTConfig:
    """Hyperparamètres du modèle. Par défaut : un petit modèle qui s'entraîne sur CPU."""

    vocab_size: int  # nombre de tokens différents (taille du vocabulaire du tokenizer)
    block_size: int = 128  # longueur maximale du contexte, en tokens
    n_layer: int = 4  # nombre de blocs empilés
    n_head: int = 4  # nombre de têtes d'attention par bloc
    n_embd: int = 128  # taille du vecteur qui représente un token
    # Dropout : à l'entraînement, chaque activation est mise à 0 avec la probabilité p = dropout, et
    # les autres sont multipliées par 1/(1-p) pour garder la même moyenne. En mode eval, il ne fait rien.
    dropout: float = 0.1


class LayerNorm(nn.Module):
    """Normalise le vecteur de chaque token (moyenne 0, variance 1), puis gain et biais appris."""

    def __init__(self, n_embd):
        super().__init__()
        self.eps = 1e-5  # évite de diviser par 0 si la variance est nulle (même valeur que PyTorch)
        self.gamma = nn.Parameter(torch.ones(n_embd))  # gain : vaut 1 au départ
        self.beta = nn.Parameter(torch.zeros(n_embd))  # biais : vaut 0 au départ

    def forward(self, x):
        # Statistiques sur la dernière dimension : les n_embd nombres d'un même token, pas le batch.
        # keepdim=True garde la forme (B, T, 1), pour retirer la moyenne à chacun des n_embd nombres.
        mean = x.mean(dim=-1, keepdim=True)
        # Variance = moyenne des carrés des écarts (on divise par n, comme PyTorch).
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # x̂ (« x chapeau ») : moyenne 0, variance 1
        # gamma et beta laissent le réseau choisir l'échelle et le décalage qui l'arrangent.
        return self.gamma * x_hat + self.beta


def gelu(x):
    """GELU(x) = x · Φ(x), avec Φ(x) = P(N(0, 1) ≤ x), ici approchée avec tanh comme dans GPT-2.

    Les constantes √(2/π) et 0,044715 viennent de l'article GELU (Hendrycks et Gimpel, 2016).
    Proche de ReLU (≈ 0 pour x très négatif, ≈ x pour x très positif), mais lisse autour de 0.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class CausalSelfAttention(nn.Module):
    """Attention multi-têtes masquée : chaque token combine les informations des tokens passés."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd doit être divisible par n_head"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        # Pour chaque token : une requête (ce que je cherche), une clé (ce que je contiens)
        # et une valeur (ce que je transmets). Une seule couche calcule toutes les têtes à la fois.
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # mélange les résultats des têtes
        self.attn_dropout = nn.Dropout(config.dropout)  # éteint au hasard des poids d'attention
        self.resid_dropout = nn.Dropout(config.dropout)  # sur la sortie, avant l'addition résiduelle
        # tril garde le triangle inférieur : mask[i, j] = 1 si j <= i (le token i peut regarder le token j).
        # Buffer et pas paramètre : l'optimiseur n'y touche pas, mais il suit le modèle
        # (sauvegarde, changement de device).
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape  # batch, nombre de tokens, n_embd
        # (B, T, C) -> (B, T, n_head, head_size) : on coupe q (de même k et v) en n_head tranches, une
        # par tête. Chaque tranche est calculée à partir de tout x : une tête voit tout le token.
        # transpose -> (B, n_head, T, head_size) : @ agit sur les 2 dernières dimensions, donc
        # chaque (séquence, tête) fait son propre calcul sur ses T vecteurs, en parallèle.
        q = self.query(x).reshape(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).reshape(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).reshape(B, T, self.n_head, self.head_size).transpose(1, 2)
        # Score de chaque paire (i, j) = q_i · k_j. Si les composantes de q et k sont indépendantes,
        # de moyenne 0 et de variance 1, q·k a une variance de head_size : diviser par √head_size
        # la ramène à 1. Sinon le softmax sature (presque tout le poids sur un seul token) et les
        # gradients s'éteignent.
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # Q·Kᵀ / √d : (B, n_head, T, T)
        # Pas le droit de regarder le futur : -inf devient un poids exactement nul après le softmax.
        # [:T, :T] car la séquence peut être plus courte que block_size.
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # chaque ligne : des poids ≥ 0 (0 pour le futur), de somme 1
        att = self.attn_dropout(att)
        y = att @ v  # moyenne des valeurs, pondérée par l'attention : (B, n_head, T, head_size)
        # Recolle les têtes côte à côte : (B, n_head, T, head_size) -> (B, T, C).
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Réseau à 2 couches appliqué à chaque token séparément."""

    def __init__(self, config):
        super().__init__()
        # fc (« fully connected ») élargit ×4, comme le Transformer de 2017.
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)  # revient à n_embd pour l'addition
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(gelu(self.fc(x))))


class Block(nn.Module):
    """Un bloc du transformer : communication entre tokens (attention), puis calcul (MLP)."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN (comme GPT-2) : on normalise l'entrée de chaque sous-couche, et on AJOUTE sa sortie
        # à x. L'article de 2017 normalisait après l'addition (post-LN), ce qui est moins stable.
        # Ce « x + ... » (connexion résiduelle) laisse le gradient remonter tout droit jusqu'aux
        # premiers blocs : c'est ce qui rend un réseau profond plus facile à entraîner.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Le modèle complet : indices de tokens (B, T) -> logits du token suivant (B, T, vocab_size)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # un vecteur appris par token
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # un vecteur appris par position
        # ModuleList et pas une liste Python : sinon PyTorch ne verrait pas les paramètres des blocs.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)  # en pre-LN, la sortie du dernier bloc n'est pas normalisée
        # Tête de sortie (lm = language model) : un score par token du vocabulaire, sans biais comme GPT-2.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying : la tête de sortie réutilise la matrice des embeddings (même forme,
        # vocab_size x n_embd). Le score du token v est le produit scalaire entre le vecteur
        # final et l'embedding de v. On économise vocab_size x n_embd paramètres.
        self.lm_head.weight = self.tok_emb.weight
        # Init de nanoGPT (le 0,02 vient de GPT-2) : poids tirés selon une loi normale d'écart-type
        # 0,02, biais à 0. Petits poids -> logits proches de 0 -> prédiction presque uniforme au départ.
        # Simplification : nanoGPT divise en plus l'écart-type de attn.proj et mlp.proj par
        # √(2·n_layer) (article GPT-2) ; on ne le fait pas ici.
        # modules() parcourt tout le modèle, couches des blocs comprises. La matrice partagée est
        # tirée deux fois (comme Embedding, puis comme Linear) avec la même loi : sans effet.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """idx, targets : (B, T) indices de tokens. Renvoie (logits, loss) ; loss vaut None sans targets."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"{T} tokens > block_size = {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)  # positions 0, 1, ..., T-1, sur le même appareil que idx
        # (B, T, C) + (T, C) : les mêmes vecteurs de position s'ajoutent à chaque séquence du batch.
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size) : un score par token possible
        if targets is None:
            return logits, None
        # Cross-entropy : moyenne sur les B*T positions de -ln p(bon token suivant), en nats (ln),
        # avec p = softmax(logits), calculé à l'intérieur de F.cross_entropy.
        # F.cross_entropy attend une ligne de logits par prédiction : on aplatit batch et positions.
        loss = F.cross_entropy(logits.reshape(B * T, self.config.vocab_size), targets.reshape(B * T))
        return logits, loss

    def num_params(self):
        """Nombre de paramètres (parameters() ne compte qu'une fois la matrice partagée)."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()  # pas d'entraînement ici : inutile de garder de quoi calculer les gradients
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Prolonge idx (B, T) de max_new_tokens tokens, tirés un par un. Appeler model.eval() avant."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]  # le modèle ne connaît que block_size positions
            logits, _ = self(idx_cond)  # self(...) appelle forward
            # Seule la dernière position prédit un token encore inconnu. temperature < 1 rend le tirage
            # plus prudent (les tokens probables le deviennent encore plus), > 1 plus varié.
            # temperature = 0 diviserait par 0 : pour toujours prendre le meilleur token, top_k=1.
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # On ne garde que les k meilleurs tokens : les autres passent à -inf (probabilité 0).
                k = min(top_k, self.config.vocab_size)  # pas plus que le vocabulaire (topk planterait)
                # topk renvoie les k meilleurs scores, triés : la dernière colonne est le k-ième.
                # [:, -1:] garde la forme (B, 1) : chaque ligne est comparée à son propre seuil.
                kth_best = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)  # logits -> probabilités de somme 1
            next_id = torch.multinomial(probs, num_samples=1)  # tirage au sort selon probs : (B, 1)
            idx = torch.cat([idx, next_id], dim=1)  # on l'ajoute au bout, puis on recommence
        return idx


if __name__ == "__main__":
    # Démo : un tout petit GPT apprend par cœur trois répliques de Molière, caractère par caractère.
    torch.manual_seed(0)
    torch.set_num_threads(4)  # le bloc 1 tourne sur CPU, 4 threads au plus
    text = (
        "Il faut manger pour vivre, et non pas vivre pour manger.\n"
        "Que diable allait-il faire dans cette galère ?\n"
        "Couvrez ce sein que je ne saurais voir.\n"
    )
    chars = sorted(set(text))  # ici, un token = un caractère
    stoi = {c: i for i, c in enumerate(chars)}  # stoi (« string to index ») : caractère -> numéro
    data = torch.tensor([stoi[c] for c in text])
    config = GPTConfig(vocab_size=len(chars), block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = GPT(config)
    print(f"{model.num_params():,} paramètres, {len(chars)} caractères différents")
    print(f"loss attendue avant entraînement : ln({len(chars)}) = {math.log(len(chars)):.3f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(301):
        ix = torch.randint(len(data) - config.block_size, (16,))  # 16 débuts de fenêtre au hasard
        x = torch.stack([data[i : i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + config.block_size] for i in ix])  # cible : décalée d'un cran
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"pas {step:3d} | loss {loss.item():.3f}")
    model.eval()  # coupe le dropout pour générer
    prompt = torch.tensor([[stoi[c] for c in "Que diable"]])
    out = model.generate(prompt, max_new_tokens=100)
    print("".join(chars[i] for i in out[0].tolist()))
