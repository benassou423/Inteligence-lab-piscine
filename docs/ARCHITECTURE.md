# Architecture et contrat commun

Ce fichier fixe les conventions partagées par tous les chantiers. On le lit avant d'écrire du code.

## Le projet en 5 lignes

- Piscine I Lab « Feed the Beast » (ECE Paris), du 16/10 au 16/11/2026.
- Le modèle (~10 M de paramètres), les hyperparamètres et le budget de tokens sont figés pour tous. Le vocabulaire du tokenizer est fixé à **8 192**.
- Seule la qualité du dataset fait la différence. Métrique : **BPB** (bits par octet) sur un test caché de textes français publiés après le 16/10/2026, en 4 domaines : encyclopédique, littéraire, presse, dialogue.
- Le kit officiel (config du modèle, script d'évaluation, set de validation, corpus Molière) n'est pas encore arrivé : `bench/` fournit un banc d'essai **provisoire**.
- Rendus : pipeline reproductible (`make all`), rapport d'ablation, datasheet, soutenance.

## Règles

1. **`minigpt/` = code écrit par Ben uniquement** (règle de la piscine). Aucun agent n'y écrit.
2. **Gel des données au 2026-10-16** : aucun contenu daté d'après.
3. **La val ne sert jamais à l'entraînement** : les IDs listés dans `data/val/*/heldout_ids.txt` sont exclus des données d'entraînement.
4. **Pas de commit ni de push** par les agents : la session principale s'en charge.
5. **Ne pas modifier** `pyproject.toml`, `uv.lock`, `Makefile`, `pipeline/io.py` ni ce fichier (plusieurs chantiers tournent en parallèle). Voir « Dépendances ».
6. **Licence notée pour chaque source** avant de l'utiliser. Common Crawl : on respecte les opt-out (robots.txt, réserve TDM).

## Arborescence des données (`data/`, ignoré par git)

```
data/raw/<source>/part-00000.jsonl.zst      sortie de la collecte
data/raw/<source>/_collect.log              log de collecte
data/stages/<config>/<NN>_<etape>/<source>/part-*.jsonl.zst
data/stages/<config>/<NN>_<etape>/stats.json
data/stages/<config>/funnel.md              entonnoir lisible (docs et octets par étape)
data/val/provisional/<domaine>.jsonl.zst    val provisoire
data/val/provisional/heldout_ids.txt        IDs exclus de l'entraînement
data/tok/<run>/                             tokenizer.json, train.bin, val_<domaine>.bin, meta.json
data/moliere/moliere.txt                    corpus Molière provisoire (bloc 1)
data/annotations/                           annotations manuelles de Ben
```

## Format d'un document

JSONL compressé zstd (`.jsonl.zst`), un objet JSON par ligne, UTF-8. Lecture et écriture : `pipeline/io.py` (`read_docs(path)`, `write_docs(path, docs)`). Fichiers découpés en `part-00000.jsonl.zst`, `part-00001…` (~100 Mo de texte au plus par fichier).

| Champ | Type | Contenu |
|---|---|---|
| `id` | str | `<source>:<identifiant stable>`, identique d'une exécution à l'autre |
| `source` | str | `wikipedia_fr`, `wikisource_fr`, `gutenberg_fr`, `commoncrawl`, `assemblee_nationale`… |
| `domain` | str | `encyclopedique`, `litteraire`, `presse`, `dialogue` ou `web` (web non classé) |
| `text` | str | texte brut à la collecte, puis texte transformé par chaque étape |
| `url` | str ou null | URL d'origine |
| `date` | str ou null | date de publication ou de capture, `AAAA-MM-JJ` |
| `license` | str | licence du document |
| `meta` | dict | libre. Chaque étape ajoute ses clés préfixées par son nom (`langid_score`, `langid_label`…) |
| `html` | str, optionnel | HTML brut, seulement pour `commoncrawl` au stade `raw` |

## Statistiques d'étape

Chaque étape du pipeline écrit `stats.json` : par source, `docs_in`, `docs_out`, `bytes_in`, `bytes_out` (octets UTF-8 de `text`) et `removed_by_reason` (raison → nombre). C'est l'entonnoir qui prouve que chaque étape sert.

## Commandes (interfaces à respecter)

| Commande | Chantier |
|---|---|
| `uv run python -m pipeline.collect <source\|all> [--limit N]` | Collecte |
| `uv run python -m pipeline.run --config configs/pipeline/<nom>.yaml` | Nettoyage |
| `uv run python -m bench.val` | Banc d'essai |
| `uv run python -m bench.run --config configs/bench/<run>.yaml [--stage prepare\|train\|eval\|all]` | Banc d'essai |
| `uv run pytest -q` | Tous |

## Propriété des dossiers

| Dossier ou fichier | Chantier |
|---|---|
| `pipeline/collect/`, `configs/collect.yaml`, `docs/sources/` | Collecte |
| `pipeline/run.py`, `pipeline/stats.py`, `pipeline/clean/`, `pipeline/dedup/`, `configs/pipeline/`, `tools/` | Nettoyage |
| `bench/`, `configs/bench/`, `experiments/` | Banc d'essai |
| `pipeline/io.py`, `docs/ARCHITECTURE.md`, `Makefile`, `pyproject.toml`, `README.md` | Session principale |
| `minigpt/` | Ben |

Tests : `tests/test_<chantier>_*.py`, chacun écrit les siens (petits fichiers d'exemple dans `tests/fixtures/<chantier>/`).

## Fiches « comprendre » (`docs/comprendre/<sujet>.md`)

Chaque composant dispose d’une fiche courte pour préparer la soutenance, selon la structure suivante :

1. **À quoi ça sert** (2 lignes)
2. **Comment ça marche** (5 puces maximum)
3. **Les chiffres clés** (seuils, volumes) et d'où ils viennent (article, référence)
4. **3 questions probables du jury**, avec une réponse courte

Phrases courtes, mots-clés en gras, aucun jargon sans définition.

## Code

- Python 3.12, lancé avec `uv run`. Identifiants en anglais, docstrings et commentaires en français, courts.
- Déterministe : seeds fixes, parcours de fichiers triés.
- Parallélisme : `multiprocessing`, **8 workers par défaut** (3 chantiers partagent 15 cœurs et 24 Go de RAM).
- Traitements de plus de 5 minutes : lancés détachés (`nohup … > log 2>&1 &`), log dans le dossier de sortie, reprise possible après interruption.
- Disque : ~115 Go libres. Traitement en flux, fichiers temporaires supprimés après usage.

## Dépendances

Déjà installées : numpy, pandas, pyyaml, tqdm, regex, zstandard, requests, warcio, lxml, datasets, huggingface_hub, ftfy, trafilatura, datasketch, xxhash, tokenizers, torch (MPS disponible), matplotlib, fasttext (paquet `fasttext-numpy2`, `import fasttext`), pytest.

Besoin d'un autre paquet : vérifier qu'aucune commande uv ne tourne (`pgrep -fl "uv (add|sync|lock)"`), lancer `uv add <paquet>`, et le signaler dans le rapport final.
