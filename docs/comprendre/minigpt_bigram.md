# Bigramme par comptage — `minigpt/bigram.py`

D'après Karpathy, *makemore* 1 · démo : `uv run python -m minigpt.bigram` (1 s)

## 1. À quoi ça sert
Un des **modèles de langue les plus simples** : il devine le symbole suivant en ne regardant **que le précédent**. C'est le **point de comparaison** : le transformer doit obtenir un **BPB plus bas**.

## 2. Comment ça marche
- Le texte devient une suite d'**ids** : ici les **octets** UTF-8, de 0 à 255 (« é » = 2 octets).
- **fit** : on **compte** chaque paire (précédent → suivant) dans une table **256 × 256**, puis **+α = 1** dans chaque case (**lissage de Laplace** : aucune probabilité nulle).
- **Normaliser** : chaque ligne divisée par sa somme ; `probs[a, b]` = probabilité que b suive a.
- **sample** : on part de `"\n"`, on **tire au sort** le suivant selon sa ligne, et on recommence.

## 3. Équations et chiffres clés
- **P(b | a) = (N[a, b] + α) / (N[a] + α·V)** ; N[a, b] = nb de paires, N[a] = total de la ligne a, V = 256.
- **NLL** (log-vraisemblance négative) = coût en **bits** = Σ −log2 P(suivant | précédent) ; **BPB** = NLL ÷ nb d'octets. Depuis une loss en **nats** (ln) : **BPB = loss × nb_tokens ÷ (ln 2 × nb_octets)**.
- Repères : **8** = hasard pur (log2 256). Molière du **kit** : **3,35** en validation, 3,36 en train.
- **Comparable entre tokenizers** : on divise par les **octets**, indépendants du découpage : bigramme d'octets et transformer BPE se comparent **directement**.

## 4. Explique cette ligne
1. `self.counts.index_put_((prev_ids, next_ids), torch.ones_like(prev_ids), accumulate=True)` → **compter** : +1 dans la case de chaque paire, en une opération. **accumulate=True** additionne les paires répétées (sinon une case vue vaudrait 1) ; « _ » = **sur place**.
2. `smoothed = self.counts.double() + self.alpha` → **lisser** : jamais P = 0, donc jamais −log2(0) = ∞. **double** = float64, par précaution (en float32, le BPB bouge de 3·10⁻⁷).
3. `self.probs = smoothed / smoothed.sum(dim=1, keepdim=True)` → **normaliser** : ligne ÷ **sa** somme. **keepdim** : forme (256, 1) ; sinon [a, b] serait divisé par la somme de la ligne b.
4. `p = self.probs[ids[:-1], ids[1:]]` → **mesurer** : un **vecteur**, p[i] = probs[ids[i], ids[i+1]] = la probabilité donnée au **vrai** suivant.
5. `current = torch.multinomial(self.probs[current], 1, generator=generator).item()` → **générer** : tirage pondéré par la ligne du symbole courant, `.item()` → entier. **Seed fixe** : même texte.

## 5. Questions probables du jury
1. **Pourquoi lisser ?** Une paire jamais vue aurait P = 0 : BPB **infini** ; un octet absent du train : 0/0 = **NaN** (« pas un nombre »). Cas réel : « Ï », 3 fois en validation, jamais en train.
2. **Pourquoi des octets ?** 256 ids couvrent **tout** texte (aucun caractère inconnu), comme la base du BPE. Et 1 id = 1 octet : le BPB se lit directement.
3. **Pourquoi le transformer fait mieux ?** Le bigramme n'a **qu'un octet de contexte** ; le mini-GPT voit jusqu'à **128 tokens** (≈ 400 octets) : mots, syntaxe, répliques.
4. **D'où viennent les « � » ?** Du **lissage** : il laisse une chance aux octets qui coupent un caractère. α = 1 : **≈ 1,7 %** de « � » (200 000 octets tirés) ; sans lissage : quasi aucun.
5. **Et la validation ?** 3 pièces **entières** du kit, jamais vues, comme pour le transformer. Val 3,35 ≈ train 3,36 : 65 536 cases n'apprennent pas 1,5 Mo **par cœur** (pas de **surapprentissage**).
