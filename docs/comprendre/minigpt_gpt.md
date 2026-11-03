# Mini-GPT : le transformer décodeur (`minigpt/gpt.py`)

Démo : `uv run python -m minigpt.gpt` · tests : `tests/test_minigpt_gpt.py`

## 1. À quoi ça sert

Prédire le **token suivant** (token = morceau de texte) à partir des précédents : le cœur de GPT.
**Attention, LayerNorm et GELU écrites à la main**, sans boîte noire, vérifiées contre PyTorch.

## 2. Comment ça marche

- **Embeddings** (vecteurs appris) : un vecteur par token + un vecteur par **position**.
- **Attention causale** : chaque token a une **requête** q (ce qu'il cherche), une **clé** k (ce qu'il offre) et une **valeur** v (ce qu'il transmet). Il reçoit la moyenne des v du **passé** (lui compris), pondérée par softmax(q·k / √d). **Multi-têtes** : `n_head` attentions en parallèle.
- **MLP** : Linear (×4) → **GELU** (un ReLU lisse) → Linear, token par token.
- **Bloc pre-LN** (comme GPT-2) : `x = x + attn(ln1(x))` puis `x = x + mlp(ln2(x))`. Le « x + » est la **résiduelle** : le gradient remonte par là **sans obstacle**.
- **Sortie** : LayerNorm → `lm_head` → **logits** (scores) → **softmax** (probabilités du token suivant).

## 3. Les équations et chiffres clés

- **Attention** (Vaswani, 2017) = softmax(Q·Kᵀ / √d + M) · V ; d = taille d'une tête ; M = −∞ (futur).
- **LayerNorm** (Ba, 2016) = γ · (x − μ) / √(σ² + ε) + β, sur chaque token seul ; ε = 1e-5.
- **GELU**(x) ≈ 0,5 · x · (1 + tanh(√(2/π) · (x + 0,044715 · x³))) (Hendrycks, 2016).
- **Loss** = moyenne de −ln p(bon token), en **nats**. Au départ ≈ **ln(vocab)** : poids ~ N(0 ; 0,02²) petits, donc softmax presque uniforme (démo : 3,46 ; ln 31 = 3,43).
- **Paramètres** = V·C + T·C + 2C + L·(12C² + 13C) (T = `block_size`, C = `n_embd`) : démo **104 128** ; config officielle **10 183 680** (`ftb/model.py`, sans biais : 10 166 400).

## 4. « Explique cette ligne »

1. `q = self.query(x).reshape(B, T, self.n_head, self.head_size).transpose(1, 2)` → **multi-têtes** : q (calculé sur tout x) est coupé en `n_head` tranches ; `transpose` met les têtes devant, car `@` agit sur les 2 dernières dimensions.
2. `att = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)` → scores Q·Kᵀ. Si q et k sont de variance 1, q·k a une variance d : sans ÷ √d, le softmax **sature**.
3. `att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))` → **masque causal** : exp(−∞) = 0, le futur a un poids nul (sinon le modèle lit la réponse).
4. `self.lm_head.weight = self.tok_emb.weight` → **weight tying** : une seule matrice pour lire et prédire les tokens, V·C paramètres en moins.
5. `idx_cond = idx[:, -self.config.block_size :]` → on garde les `block_size` derniers tokens : au-delà, pas de vecteur de position.

## 5. Questions probables du jury

1. **Pourquoi `n_head` têtes ?** Chacune **peut** suivre sa relation (accord…), à paramètres égaux.
2. **Pourquoi des positions ?** L'attention fait une **moyenne**, qui ignore l'ordre : sans positions, une couche seule prédit la même suite après « le chien mord Paul » et « chien le mord Paul ».
3. **LayerNorm plutôt que BatchNorm ?** Elle normalise chaque token **seul** : rien ne dépend du batch ni du futur ; même calcul en entraînement et en génération.
4. **Comment savez-vous que c'est juste ?** Mêmes logits (à 1e-5) qu'un calcul fait avec les fonctions de PyTorch ; un token ne change aucun logit avant lui ; mini-batch appris par cœur.
5. **temperature et top_k ?** Logits ÷ temperature (< 1 : plus sûr, > 1 : plus varié) ; top_k garde les k meilleurs tokens (top_k = 1 : toujours le meilleur).
