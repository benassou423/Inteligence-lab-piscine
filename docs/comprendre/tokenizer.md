# Tokenizer BPE 8 192

## À quoi ça sert

Couper le texte en **tokens** (morceaux de mots) que le modèle sait lire. Le tokenizer est **libre**, mais le kit impose ses règles ; et comme le budget est compté **en tokens**, c'est un vrai levier.

## Comment ça marche

- **BPE** (*Byte Pair Encoding*) : on part des **256 octets** possibles et on fusionne la paire voisine la plus fréquente, encore et encore, jusqu'à 8 192 tokens.
- **Byte-level** : aucun caractère n'est inconnu (accents, « ’ », emojis) et decode(encode(texte)) redonne **exactement** le texte.
- **Pré-découpage** : une regex coupe d'abord en mots, nombres, ponctuation ; le BPE ne fusionne jamais à travers deux morceaux. `gpt4` (motif du kit) : « l » + « 'homme » ; `fr` (option du banc) : « l' » + « homme », « qu' » + « il ».
- Le banc l'entraîne avec la fonction du kit (`ftb.tokenizer.train_hf`) sur un **échantillon déterministe** du mix (`tokenizer_sample.jsonl.gz`), puis le fait vérifier par `ftb.tokenizer check`.
- `<|endoftext|>` est ajouté **après chaque document** par `ftb.prepare` : le modèle apprend où un texte finit.

## Les chiffres clés

| Chiffre | Valeur | D'où ça vient |
|---|---|---|
| Vocabulaire | **exactement 8 192** = 256 octets + 7 935 fusions + `<|endoftext|>` | règle du kit |
| Autres règles | sans perte, sans normalisation, déterministe, format `tokenizer.json` Hugging Face | `docs/KIT_README.md` |
| Octets par token, mix Wikipédia (tokenizer `gpt4` entraîné dessus) | **3,56** | `data/mixes/rank_wiki/prepared/data_meta.json` |
| Octets par token sur la val | colonne `val_bytes_per_token` de `experiments/results.csv` | `ftb.evaluate` |
| BPE pour le texte ; BPE sur les octets | Sennrich et al. 2016 ; Radford et al. 2019 (GPT-2) | articles fondateurs |

Plus d'octets par token = plus de texte vu pour les mêmes 100 M tokens. Mesure : `uv run python -m ftb.tokenizer check --tokenizer <fichier>`.

## 3 questions probables du jury

1. **Pourquoi byte-level ?** Rien n'est inconnu, la tokenisation est sans perte (règle du kit), et le BPB se calcule exactement sur les octets du texte.
2. **Pourquoi une variante française ?** En français, « l' », « d' », « qu' » sont des mots à part entière. Les garder entiers évite de gaspiller des fusions sur « 'homme », « 'eau »… C'est une hypothèse : on la tranche par une ablation (même mix, `pretokenizer: fr` contre `gpt4`).
3. **Sur quoi entraîner le tokenizer ?** Sur un texte qui ressemble au test : il doit bien compresser les 4 genres. Appris sur Molière seul, il ne fait que **2,98 octets/token** sur la val, contre **3,66** appris sur Wikipédia (`experiments/results.csv`) : le modèle voit alors moins de texte pour le même budget.
