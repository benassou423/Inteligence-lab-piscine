# Feed the Beast — nourrir un GPT avec le meilleur français possible

Piscine I Lab (ECE Paris), 16/10 → 16/11/2026. Le modèle et le budget sont les mêmes pour tous : seule la qualité des données fait la différence.

## Les 3 blocs

| Bloc | Dossier | Contenu |
|---|---|---|
| 1. mini-GPT from scratch | `minigpt/` | autograd, bigramme, BPE, transformer, entraînement sur Molière |
| 2. Pipeline de données | `pipeline/` | collecte → nettoyage → déduplication → filtrage qualité |
| 3. Compétition | `bench/` | tokenizer 8 192, entraînement, BPB par domaine |

## Démarrage

```bash
uv sync          # installe l'environnement (Python 3.12)
make test        # lance les tests
make all         # regénère tout : collecte, nettoyage, val, run proxy
```

Les conventions communes sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Les explications de chaque étape sont dans `docs/comprendre/`.
