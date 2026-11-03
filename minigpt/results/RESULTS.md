# Bloc 1 : le mini-GPT sur Molière

Validation : les 3 pièces tenues à l'écart par le kit (260 488 octets UTF-8, 84 108 tokens BPE). BPB = bits de **tout** ce texte ÷ ses octets.

| Modèle | BPB validation (bits par octet, plus bas = mieux) |
|---|---|
| Bigramme d'octets (table 256 × 256, lissage de Laplace) | 3,354 |
| Bigramme de tokens BPE (table 2048 × 2048, Laplace) | 2,494 |
| **Mini-GPT** (1 071 872 paramètres) | **1,738** |

| Réglage | Valeur |
|---|---|
| Tokenizer | notre BPE, 2048 tokens, 3,10 octets par token en validation |
| Modèle | 4 blocs, 4 têtes, n_embd 128, contexte 128 tokens, dropout 0,1 |
| Entraînement | 3 401 pas de 32 × 128 tokens, AdamW, warmup puis cosinus (lr 0,001 → 0,0001), 11,6 min sur 4 threads CPU |
| Loss finale (nats par token) | 3,040 en train, 3,716 en validation |
| Graine | 1337 |

Interprétation :

1. Sur 3 pièces jamais vues, le mini-GPT dépense **1,74 bits par octet** : 48 % de moins que le bigramme d'octets et 30 % de moins que le bigramme de tokens.
2. Le bigramme de tokens bat déjà celui d'octets : un token couvre 3,1 octets, donc « le précédent » en dit plus. Le reste du gain vient du contexte : le mini-GPT lit jusqu'à 128 tokens (≈ 396 octets).
3. La loss de validation passe de 7,65 (≈ ln 2048 = 7,62 : le hasard) à 3,72. Son minimum, 3,710, est mesuré au pas 3 350 ; elle finit 0,006 au-dessus : elle plafonne.
4. Écart validation − train en fin d'entraînement : 0,676 nat par token. Plus il grandit, plus le modèle apprend le train par cœur au lieu de généraliser (surapprentissage).
5. Limite : 13,9 M de tokens vus, soit ≈ 29 passages sur les 475 419 tokens du train. Le modèle de la compétition a 10 M de paramètres et voit 100 M de tokens.
6. Prudence : n_embd (128 plutôt que 192) et le learning rate (0,001 plutôt que 0,002) ont été choisis sur le BPB de ces 3 pièces (essais de 3 min). La validation a donc un peu servi à régler le modèle : ce BPB est **légèrement optimiste**. Une mesure sans biais demanderait un 3e jeu, jamais regardé.

Rejouer : `uv run python -m minigpt.train_moliere --steps 3401` (graine 1337, 4 threads) redonne les mêmes poids au bit près, sur la même machine avec les mêmes versions de Python et de torch. Puis `… --bridge` pour le pont vers la compétition.

Fichiers : `loss_curve.png` (courbe), `samples.md` (5 échantillons), `metrics.json` (chiffres), `tokenizer.json` (notre BPE), `model.pt` (poids, ignoré par git).

## Pont vers la compétition : notre BPE en 8 192 tokens

| Mesure | Valeur |
|---|---|
| Entraînement sur les 25 pièces du train (7 935 fusions) | 13 s |
| Octets par token, validation du kit (`data/validation.jsonl`, 230 textes) | 2,9751 |

Conversion et vérification par le kit (il encode exactement comme `minigpt/bpe.py`) :

```bash
uv run python -m ftb.tokenizer from-minbpe --model minigpt/results/bpe8192.model --out minigpt/results/tokenizer_8192.json
uv run python -m ftb.tokenizer check --tokenizer minigpt/results/tokenizer_8192.json
```
