# Bloc 1 — mini-GPT from scratch

Codé avec Claude, ce que l'orga a autorisé. **Condition : Ben explique chaque ligne à la soutenance.** Chaque fichier est donc court et commenté en français, et chaque composant a sa fiche dans `docs/comprendre/minigpt_*.md`.

## Attendus

- [x] `micrograd.py` : classe `Value` avec `backward()`, vérifiée contre PyTorch
- [x] `bigram.py` : modèle bigramme par comptage (point de comparaison)
- [x] `bpe.py` : tokenizer BPE from scratch (`train`, `encode`, `decode`)
- [x] `gpt.py` : transformer décodeur (attention multi-têtes avec masque causal, LayerNorm, MLP, connexions résiduelles)
- [x] `train_moliere.py` : entraînement sur Molière avec `bpe.py` → courbe de loss, BPB sur la validation, 5 échantillons

Corpus officiel du kit : `corpus/moliere_train.txt` (25 pièces) et `corpus/moliere_val.txt` (Le Misanthrope, L'Avare, Le Sicilien). Résultats : `minigpt/results/`.
