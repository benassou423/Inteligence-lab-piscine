# BPE : le tokenizer (`minigpt/bpe.py`)

Démo : `uv run python -m minigpt.bpe` · tests : `uv run pytest -q tests/test_minigpt_bpe.py`

## 1. À quoi ça sert

Un modèle lit des **nombres**. Le **tokenizer** coupe le texte en **tokens** (morceaux fréquents : « qu », « pour »), un **id** chacun. Le **BPE** (*Byte Pair Encoding*) apprend ces morceaux sur le corpus.

## 2. Comment ça marche

- **Base = 256 octets** : en **UTF-8**, « é » = 2 octets, 😀 = 4. **Aucun caractère inconnu**.
- **Pré-découpage** par une **regex** (motif de recherche de texte) : celle de **GPT-4**, **identique au kit**. Une fusion reste dans un mot : pas de token « e, » ni « t d ».
- **Entraînement** : on fusionne la **paire voisine la plus fréquente** en un nouveau token (256, 257…). À **égalité**, la plus petite paire d'ids : une **règle fixe**.
- **Rapide** : on compte les **morceaux uniques** (× fréquence) ; on ne recompte que les touchés.
- **Encoder** = rejouer les fusions dans l'**ordre appris** ; **décoder** = recoller les octets. `<|endoftext|>` (fin de document) prend le **dernier id**.

## 3. Équations et chiffres clés

- **Fusions = vocab_size − 256 − tokens spéciaux** : 2048 → 1 791 ; 8192 → **7 935** (kit).
- **À la main** : « aaabdaaabac » → aa = Z ; « Za » et « ab » à égalité : la plus petite paire, ab = Y ; ZY = X → « XdXac » : **5 tokens au lieu de 11**.
- **Octets par token** (Molière du kit, train / **validation**) : 512 → 2,09 / 2,10 · 2048 → 3,23 / **3,10** · 8192 → 4,11 / **3,78**. Entraînement : 3 à 12 s (2048), 13 à 90 s (8192), selon la charge du CPU.
- **Pont vers la compétition** : `save_minbpe()` écrit le **.model de minbpe** (« minbpe v1 », motif, tokens spéciaux, une fusion « a b » par ligne). BPE 8192 → `ftb.tokenizer from-minbpe` → `check` **passe** (sans perte) : **2,98 octets par token** sur la validation du kit.

## 4. Explique cette ligne

1. `r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|…` (SPLIT_PATTERN) → **contraction anglaise** ('s, 't…), **mot** (1 caractère devant au plus), **nombre**, **ponctuation**, **blancs**. Prix : « p'tite » → « p », « 't », « ite » (6 morceaux sur 341 416).
2. `pair = min(pair_counts, key=lambda p: (-pair_counts[p], p))` → le **min du compte négatif** = la paire la plus fréquente ; à égalité, la plus petite.
3. `pair_counts[p] -= counts[j]` puis `+= counts[j]` → mot touché : on **retire ses anciennes paires**, on **ajoute les nouvelles**, × sa fréquence.
4. `min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))` → parmi les **paires voisines**, celle **apprise le plus tôt** ; jamais apprise = **infini**.
5. `data.decode("utf-8", errors="replace")` → « é » coupé → « � », sans **planter**.

## 5. Questions probables du jury

1. **Pourquoi des octets ?** → 256 symboles couvrent tout Unicode (œ, emoji) : rien d'« inconnu ».
2. **Pourquoi l'ordre des fusions ?** → Une fusion réutilise les anciennes : « abc » → [a, bc].
3. **Comment savez-vous que c'est juste ?** → **19 tests** : français piège, exemple à la main, **mêmes fusions qu'un BPE naïf**, **mêmes ids que le kit**.
4. **Pourquoi le motif de GPT-4 ?** → Celui du kit : converti, notre BPE donne **les mêmes ids**.
5. **Quelle taille de vocabulaire ?** → Plus de tokens = séquences plus courtes, mais plus d'**embeddings** à apprendre et des tokens rares peu vus.
