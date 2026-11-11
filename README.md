# Feed the Beast — nourrir un GPT avec le meilleur français possible

Piscine I Lab (ECE Paris), 16/10 → 16/11/2026. Le modèle et le budget sont les mêmes pour tous : seule la qualité des données fait la différence.

## Les 3 blocs

| Bloc | Dossier | Contenu |
|---|---|---|
| 1. mini-GPT from scratch | `minigpt/` | autograd, bigramme, BPE, transformer, entraînement sur Molière |
| 2. Pipeline de données | `pipeline/` | collecte → nettoyage → déduplication → filtrage qualité |
| 3. Compétition | `bench/` | tokenizer 8 192, entraînement, BPB par domaine |

## Première soumission officielle — 11/11/2026

Le paquet est dans [submissions/2026-11-11_153839](submissions/2026-11-11_153839) : modèle, tokenizer,
journaux, métadonnées et cinq générations. Il a été créé par `ftb.submit` sans
modifier le kit. Entraînement Mac MPS float32 : **100 007 936 tokens vus**, score
public de validation **1,288292 BPB**. Le score du test caché et le contrôle T4
relèvent de l'organisateur et ne sont pas encore attestés.

Le [dataset final](https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/ftb-final-20261110)
contient 72 612 documents et 114 701 016 tokens disponibles. Ses dix fichiers ont
été téléchargés sans authentification et vérifiés ; la préparation des fichiers
retéléchargés reproduit la même empreinte `b88033ad2fba…`.

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
