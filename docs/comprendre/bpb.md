# BPB — bits par octet (calcul officiel du kit)

## À quoi ça sert

Mesurer à quel point le modèle **prédit bien** un texte qu'il n'a jamais vu. C'est la note de la compétition : **plus le BPB est bas, mieux c'est**. Calcul : `ftb/evaluate.py` (kit, non modifiable).

## Comment ça marche

- Pour chaque token, le modèle donne une probabilité **p** au bon token. Coût : **−log2 p bits** (p = 1/2 → 1 bit ; p = 1/8 → 3 bits).
- On additionne les bits, puis on divise par les **octets UTF-8** du texte d'origine (« a » = 1 octet, « é » = 2, « ’ » = 3) : la note ne dépend **pas du tokenizer**.
- Chaque document est évalué **seul**, précédé de `<|endoftext|>` (le modèle sait qu'un texte commence).
- Document plus long que 512 tokens : **fenêtres glissantes au pas de 256**. Chaque token est prédit une seule fois, avec **au moins 256 tokens de contexte** (sauf au début du document).
- **BPB global** = bits de tous les documents ÷ octets de tous les documents. Les 4 genres pèsent ~200 Ko chacun : le global est presque la **moyenne des 4 BPB par genre**.

## Les chiffres clés

| Repère | Valeur | Source |
|---|---|---|
| Formule avec la perte moyenne (nats/token) | BPB = perte × nb tokens ÷ (ln 2 × nb octets) = perte ÷ ln 2 ÷ octets par token | `docs/KIT_README.md` |
| Exemple : perte 3,0 et 3,6 octets/token | 3,0 ÷ 0,693 ÷ 3,6 ≈ **1,20** | calcul |
| Val publique | 230 documents, 813 Ko, 4 genres à parts égales en octets | `data/validation.jsonl` |
| Contexte et pas des fenêtres | 512 tokens, pas de 256 | `config/official.json` |
| Texte tiré au hasard parmi 256 octets | 8 bits/octet | définition |
| Métrique « BPB » pour comparer des tokenizers | — | Gao et al. 2020, *The Pile* |

## 3 questions probables du jury

1. **Pourquoi des bits par octet et pas la perplexité ?** La perplexité dépend du tokenizer (un token peut être une lettre ou un mot). Le BPB ramène tout au même diviseur, les octets du texte : deux tokenizers sont comparables.
2. **Pourquoi des fenêtres glissantes ?** Avec des fenêtres collées bout à bout, le début de chaque fenêtre serait prédit presque sans contexte, ce qui gonfle la note. Au pas de 256, chaque token a au moins 256 tokens de contexte : la mesure est plus juste.
3. **Un meilleur tokenizer baisse-t-il le BPB ?** Pas directement : la note est en octets. Mais le budget est compté en **tokens** (100 M) : un tokenizer qui fait plus d'octets par token fait voir plus de texte au modèle, et c'est ce qui peut baisser le BPB.
