# Architecture et contrat commun

Ce fichier fixe les conventions partagées par tous les chantiers. On le lit avant d'écrire du code.

## Le projet en 5 lignes

- Piscine I Lab « Feed the Beast » (ECE Paris), du 16/10 au 16/11/2026.
- Le modèle (10 166 400 paramètres), les hyperparamètres et le budget (**100 M tokens vus**) sont figés pour tous. Le tokenizer est libre mais fait **exactement 8 192 tokens**, `<|endoftext|>` compris, et doit être **sans perte**.
- Seule la qualité du dataset fait la différence. Métrique : **BPB** (bits par octet) sur un test caché en 4 genres à parts égales : encyclopédique, littéraire, presse, dialogue.
- **Le kit officiel est copié à la racine, sans modification** (Meuleur/Piscine, commit 3709c85) : `ftb/`, `config/official.json`, `corpus/` (Molière), `data/validation.jsonl`, `tests/test_kit.py`, `examples/`, `notebooks/`. Sa doc : [KIT_README.md](KIT_README.md).
- Rendus : pipeline reproductible (`make all`), rapport d'ablation, datasheet, soutenance, soumissions dans `submissions/`.

## Règles

1. **`minigpt/` (bloc 1) est codé avec Claude**, ce que l'orga a autorisé. Condition : Ben doit **expliquer chaque ligne** à la soutenance → code court, commenté en français, une fiche par composant. Seuls les agents du bloc 1 y écrivent.
2. **Gel des données au 2026-10-16** : aucun contenu daté d'après.
3. **La val ne sert jamais à l'entraînement** : l'étape `decontaminate` retire tout document qui recouvre `data/validation.jsonl` (13-grammes, comme `ftb/contamination.py`).
4. **Pas de commit ni de push** par les agents : la session principale s'en charge.
5. **Ne pas modifier** les fichiers du kit, `pyproject.toml`, `uv.lock`, `Makefile`, `pipeline/io.py` ni ce fichier. Voir « Dépendances ».
6. **Licence notée pour chaque source** avant de l'utiliser. Common Crawl : on respecte les opt-out (robots.txt, réserve TDM).

## Ce que la val nous apprend (mesuré le 03/11)

| Genre | Source de la val | Format |
|---|---|---|
| Encyclopédique | articles Wikipédia FR créés fin octobre 2026 | 1er paragraphe, puis titres de sections seuls sur leur ligne |
| Littéraire | livres FR du Projet Gutenberg mis en ligne en 2026 (extraits) | dialogues en « -- », apostrophe ’ |
| Presse | Wikinews FR 2024-2026 (le test : d'autres sources de presse récentes) | 1re ligne = la date |
| Dialogue | comptes rendus de l'Assemblée nationale, octobre 2026 (extraits) | « M. Nom : propos », ligne vide entre interventions |

Typographie de la val : ’ 4 383, ' 2 587, « 357, – 195, — 26, **aucune espace insécable** (U+00A0, U+202F). Le pipeline remplace donc les espaces insécables par des espaces normales.

## Arborescence

```
data/raw/<source>/part-00000.jsonl.zst      sortie de la collecte (ignoré par git)
data/stages/<config>/<NN>_<etape>/<source>/part-*.jsonl.zst   sorties du pipeline
data/stages/<config>/<NN>_<etape>/stats.json
data/stages/<config>/funnel.md              entonnoir lisible (docs et octets par étape)
data/mixes/<run>/train.jsonl.gz             dataset ordonné, prêt pour ftb.prepare
data/validation.jsonl                       val officielle du kit (suivie par git)
data/annotations/                           annotations manuelles de Ben
runs/<run>/                                 sorties de ftb.train (ignoré par git)
submissions/<date>_<heure>/                 soumissions officielles (suivies par git)
experiments/results.csv                     une ligne par run
```

## Format d'un document (pipeline)

JSONL compressé zstd (`.jsonl.zst`), un objet JSON par ligne, UTF-8. Lecture et écriture : `pipeline/io.py` (`read_docs(path)`, `write_docs(path, docs)`). Fichiers découpés en `part-00000.jsonl.zst`, `part-00001…` (~100 Mo de texte au plus par fichier). Le kit, lui, lit `.jsonl`, `.jsonl.gz`, `.txt` ou `.parquet` (champ `text`) : `bench/mix.py` fait la conversion.

| Champ | Type | Contenu |
|---|---|---|
| `id` | str | `<source>:<identifiant stable>`, identique d'une exécution à l'autre |
| `source` | str | `wikipedia_fr`, `wikisource_fr`, `gutenberg_fr`, `commoncrawl`, `assemblee_nationale`, `wikinews_fr`, `fineweb2_fra`… |
| `domain` | str | `encyclopedique`, `litteraire`, `presse`, `dialogue` ou `web` (web non classé) |
| `text` | str | texte brut à la collecte, puis texte transformé par chaque étape |
| `url` | str ou null | URL d'origine |
| `date` | str ou null | date de publication ou de capture, `AAAA-MM-JJ` |
| `license` | str | licence du document |
| `meta` | dict | libre. Chaque étape ajoute ses clés préfixées par son nom (`langid_score`…) |
| `html` | str, optionnel | HTML brut, seulement pour `commoncrawl` au stade `raw` |

## Statistiques d'étape

Chaque étape du pipeline écrit `stats.json` : par source, `docs_in`, `docs_out`, `bytes_in`, `bytes_out` (octets UTF-8 de `text`) et `removed_by_reason` (raison → nombre). C'est l'entonnoir qui prouve que chaque étape sert.

## Commandes

| Commande | Rôle |
|---|---|
| `uv run python -m pipeline.collect <source\|all> [--limit N]` | Collecte |
| `uv run python -m pipeline.run --config configs/pipeline/<nom>.yaml` | Nettoyage, filtres, décontamination |
| `uv run python -m bench.run --config configs/bench/<run>.yaml` | Mix → tokenizer → `ftb.prepare` → `ftb.train` → `ftb.evaluate` → `results.csv` |
| `uv run python -m ftb.contamination --inputs <fichiers>` | Recouvrement avec la val (kit) |
| `uv run python -m ftb.submit --run runs/<run> --participant "Benjamin Assouline"` | Soumission (2 par jour) |
| `uv run pytest -q` | Tous les tests, kit compris |

## Propriété des dossiers

| Dossier ou fichier | Chantier |
|---|---|
| `pipeline/collect/`, `configs/collect.yaml`, `docs/sources/` | Collecte |
| `pipeline/run.py`, `pipeline/stats.py`, `pipeline/clean/`, `pipeline/dedup/`, `configs/pipeline/`, `tools/` | Nettoyage |
| `bench/`, `configs/bench/`, `experiments/`, `data/mixes/` | Banc d'essai |
| `minigpt/`, `tests/test_minigpt_*.py`, `docs/comprendre/minigpt_*.md` | Bloc 1 |
| `ftb/`, `config/`, `corpus/`, `data/validation.jsonl`, `tests/test_kit.py`, `examples/`, `notebooks/` | Kit officiel : lecture seule |
| `pipeline/io.py`, `docs/ARCHITECTURE.md`, `Makefile`, `pyproject.toml`, `README.md` | Session principale |

Tests : `tests/test_<chantier>_*.py`, chacun écrit les siens (petits fichiers d'exemple dans `tests/fixtures/<chantier>/`).

## Fiches « comprendre » (`docs/comprendre/<sujet>.md`)

Ben a un TDAH et doit défendre chaque choix à la soutenance. Chaque module livré s'accompagne d'une fiche d'**une page maximum**, en français :

1. **À quoi ça sert** (2 lignes)
2. **Comment ça marche** (5 puces maximum)
3. **Les chiffres clés** (seuils, volumes) et d'où ils viennent (article, référence)
4. **3 questions probables du jury**, avec une réponse courte

Phrases courtes, mots-clés en gras, aucun jargon sans définition.

## Code

- Python 3.12, lancé avec `uv run`. Identifiants en anglais, docstrings et commentaires en français, courts.
- Déterministe : seeds fixes, parcours de fichiers triés.
- Parallélisme : `multiprocessing`, **8 workers par défaut** (plusieurs chantiers partagent 15 cœurs et 24 Go de RAM).
- Traitements de plus de 5 minutes : lancés détachés (`nohup … > log 2>&1 &`), log dans le dossier de sortie, reprise possible après interruption.
- Disque : ~100 Go libres. Traitement en flux, fichiers temporaires supprimés après usage.

## Dépendances

Installées : numpy, pandas, pyyaml, tqdm, regex, zstandard, requests, warcio, lxml, datasets, huggingface_hub, ftfy, trafilatura, datasketch, xxhash, tokenizers, torch (MPS disponible), matplotlib, fasttext (paquet `fasttext-numpy2`, `import fasttext`), mwparserfromhell, pytest.

Besoin d'un autre paquet : vérifier qu'aucune commande uv ne tourne (`pgrep -fl "uv (add|sync|lock)"`), lancer `uv add <paquet>`, et le signaler dans le rapport final.
