# Bloc 1 — mini-GPT from scratch

**Zone de Ben.** Règle de la piscine : le mini-GPT doit être codé par Ben, qui explique chaque ligne en soutenance. Aucun assistant IA n'écrit de code dans ce dossier. Claude explique, relit et pose des questions.

## Attendus

- [ ] `micrograd.py` : classe `Value` avec `backward()`
- [ ] `test_micrograd.py` : mêmes gradients que PyTorch
- [ ] `bigram.py` : modèle bigramme par comptage (point de comparaison)
- [ ] `bpe.py` : tokenizer BPE from scratch (`train`, `encode`, `decode`)
- [ ] `gpt.py` : transformer décodeur (attention multi-têtes avec masque causal, LayerNorm, MLP, connexions résiduelles)
- [ ] `train_moliere.py` : entraînement sur Molière avec `bpe.py` → courbe de loss, BPB sur la validation, 5 échantillons

Corpus Molière provisoire (en attendant le kit) : `data/moliere/moliere.txt`.
