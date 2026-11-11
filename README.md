# Feed the Beast — nourrir un GPT avec le meilleur français possible

Piscine I Lab (ECE Paris), 16/10 → 16/11/2026. Le modèle et le budget sont les mêmes pour tous : seule la qualité des données fait la différence.

## Les 3 blocs

| Bloc | Dossier | Contenu |
|---|---|---|
| 1. mini-GPT from scratch | `minigpt/` | autograd, bigramme, BPE, transformer, entraînement sur Molière |
| 2. Pipeline de données | `pipeline/` | collecte → nettoyage → déduplication → filtrage qualité |
| 3. Compétition | `bench/` | tokenizer 8 192, entraînement, BPB par domaine |

## Démarrage

Depuis le dossier `Inteligence-lab-piscine`, sur macOS/Linux :

```bash
uv run --no-project python tools/reproduce.py --recipe final
```

Cette commande prépare les liens vers les dossiers `.nosync` avant l'environnement
Python, collecte les sources, exécute le nettoyage de référence, recrée le tokenizer,
sélectionne le mélange et masque les coordonnées. Elle contrôle la contamination
et les hashes attendus. Le calcul tourne avec `nohup` et son journal est indiqué dans
`data/stages/logs/`. Prévoir 100 Go libres pour une reconstruction intégrale.

La variante expérimentale `--recipe complete` ajoute les filtres qualité web et
MinHash/LSH. Elle a été évaluée séparément et n'est pas retenue pour le jeu final :
elle dégrade le BPB sur les deux tirages et retire parfois des fiches d'entités différentes.

Après préparation de l'environnement :

```bash
uv run pytest -q
uv run python -m minigpt.train_moliere --steps 3401
```

Le `Makefile` historique conserve les commandes de la référence de base. Les
instructions de reproduction et leurs limites sont dans
[REPRODUCTION.md](docs/remise/REPRODUCTION.md). Le passage Windows pour entraîner
sur RTX 5090 est décrit dans [PASSAGE_PC_RTX5090.md](docs/remise/PASSAGE_PC_RTX5090.md).

Les conventions communes sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Les explications de chaque étape sont dans `docs/comprendre/`.
