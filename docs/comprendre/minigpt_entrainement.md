# Entraînement sur Molière (`minigpt/train_moliere.py`)

`uv run python -m minigpt.train_moliere --steps 3401` (8 à 12 min), puis `… --bridge`

## 1. À quoi ça sert

Assembler les briques (**BPE**, **bigramme**, **GPT**) et **mesurer** : le mini-GPT doit battre les deux bigrammes en **BPB** (bits par octet) sur **3 pièces jamais vues**.

## 2. Comment ça marche

- **Découpage du kit** : 25 pièces pour apprendre, 3 à l'écart (Le Misanthrope, Le Sicilien, L'Avare). Le **BPE n'apprend que sur le train**.
- **Un pas** : 32 fenêtres de 128 tokens au hasard ; **cible** = la même fenêtre **décalée d'un cran** ; loss → gradients → **AdamW** (Adam + **weight decay** : poids tirés vers 0).
- **Learning rate** : **warmup** (5 % des pas) puis **cosinus** (0,001 → 0,0001), sur **3 401 pas fixés**.
- **Courbe** : tous les 50 pas, loss sur 10 batchs **fixes**, **dropout** coupé (`model.eval()`).
- **BPB final** : **toute** la validation, fenêtres **sans recouvrement** : chaque token prédit **une fois**.

## 3. Équations et chiffres clés

- **BPB = Σ nats ÷ (ln 2 × 260 488 octets)** : ÷ ln 2 change les nats (ln) en bits (log2).
- **BPB de validation** : bigramme d'octets **3,35** · bigramme BPE **2,49** · **mini-GPT 1,74**.
- **Vocabulaire 2048** : **3,10 octets par token** (8192 : 3,78), **232 occurrences par token** dans le train (8192 : ~46) : petit corpus → petit vocabulaire.
- **Modèle** : 4 blocs, 4 têtes, n_embd 128, contexte 128, **1 071 872 paramètres** ; 13,9 M de tokens vus (≈ 29 passages).
- **Réglé sur la validation** (essais de 3 min) : n_embd 128 bat 192 (2,16 / 2,25) et lr 0,002 (2,32). Le BPB de 1,74 est donc **un peu optimiste**.

## 4. « Explique cette ligne »

1. `y = torch.stack([ids[s + 1 : s + 1 + BLOCK_SIZE] for s in starts])` → **cible décalée d'un cran** : chaque position apprend **le suivant**, caché par le **masque causal**.
2. `progress = step / steps if steps else (time.perf_counter() - start) / (60 * minutes)` → **avancement 0 → 1** (en pas, ou en temps avec `--minutes`) : règle le lr et l'arrêt.
3. `return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))` → **cosinus** : t = 0 → LR ; t = 1 → cos = −1 → MIN_LR.
4. `nats += loss.item() * len(y)` → **moyenne** × nb de tokens = **somme**.
5. `bpb = total_nats(model, val_ids) / (math.log(2) * n_val_bytes)` → nats → bits (÷ ln 2), ÷ **tous les octets**.

## 5. Questions probables du jury

1. **Pourquoi le BPB et pas la loss ?** → La loss (**par token**) dépend du tokenizer, pas le BPB (**par octet**).
2. **Pourquoi toute la validation ?** → Au hasard, des tokens comptent deux fois, d'autres jamais. Le kit fait des fenêtres **glissantes** (plus de contexte) : notre BPB est **prudent**.
3. **Bigramme BPE trop faible ?** → Laplace (+1 dans 4 M de cases) lisse fort. Le meilleur α sur la validation (0,02) : **2,09** ; le mini-GPT fait encore **17 % de mieux**.
4. **Surapprentissage ?** → En partie : val − train = **0,68 nat**. La loss de validation **plafonne** : minimum 3,710 au pas 3 350, 3,716 à la fin (**+0,006**).
5. **Le run se rejoue-t-il ?** → Oui : pas fixés, graine 1337, 4 threads → **mêmes poids au bit près** (vérifié : 2 relances, même machine). Avec `--minutes`, non.
