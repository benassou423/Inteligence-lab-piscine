# Architecture et contrat commun

Ce fichier fixe les conventions partagées par tous les chantiers. On le lit avant d'écrire du code.

État au 16/11/2026 : le rendu du concours a été fait le 11/11/2026 (soumission officielle, la seule du dépôt : voir le [README](../README.md)). Le score du test caché relève de l'organisateur. Les formats, les commandes et les règles sur les données valent toujours. Les règles d'organisation du travail (qui écrit où, qui commite) ne valent plus : elles sont gardées comme trace et signalées comme telles. La table des matières de la documentation est [README.md](README.md), l'index des outils [tools/README.md](../tools/README.md), le récit daté du chatbot [chatbot/HISTORIQUE.md](chatbot/HISTORIQUE.md).

## Le projet en 5 lignes

- Piscine I Lab « Feed the Beast » (ECE Paris), du 16/10 au 16/11/2026.
- Le modèle (10 166 400 paramètres), les hyperparamètres et le budget (**100 M tokens vus**) sont figés pour tous. Le tokenizer est libre mais fait **exactement 8 192 tokens**, `<|endoftext|>` compris, et doit être **sans perte**.
- Seule la qualité du dataset fait la différence. Métrique : **BPB** (bits par octet) sur un test caché en 4 genres à parts égales : encyclopédique, littéraire, presse, dialogue.
- **Le kit officiel est copié à la racine, sans modification** (Meuleur/Piscine, commit 3709c85) : `ftb/`, `config/official.json`, `corpus/` (Molière), `data/validation.jsonl`, `tests/test_kit.py`, `examples/`, `notebooks/`. Sa doc : [KIT_README.md](KIT_README.md).
- Rendus : pipeline reproductible (`uv run --no-project python tools/reproduce.py --recipe final`, voir [remise/REPRODUCTION.md](remise/REPRODUCTION.md) ; `make all` ne rejoue que la référence de base : collecte, nettoyage `nettoye`, run proxy), rapport d'ablation, datasheet, soutenance, soumissions dans `submissions/`.

Hors concours, le dépôt contient aussi un chatbot « à la manière de Molière » : un GPT de 54 749 440 paramètres, entraîné de zéro puis affiné sur du théâtre. Son code est dans `chatbot/`. Il réutilise la classe `GPT` du kit (`ftb/model.py`, non modifié). Le modèle est décrit dans [chatbot/FICHE_MODELE.md](chatbot/FICHE_MODELE.md), ses données dans [chatbot/FICHE_DONNEES.md](chatbot/FICHE_DONNEES.md). Le guide d'entraînement sur carte NVIDIA est [PC_RTX.md](../PC_RTX.md). Ce qui est sous quelle licence est écrit dans [LICENCES.md](../LICENCES.md).

## Règles

Les règles 2, 3 et 6 portent sur les données : elles valent toujours. La première partie de la règle 5 (ne pas modifier le kit) vaut aussi toujours. Les règles 1 et 4, et le reste de la règle 5, sont des règles de travail du temps du concours : plusieurs agents IA écrivaient en parallèle dans le dépôt, et une session principale se chargeait des commits. Les six règles gardent leur numéro, parce que d'autres fichiers les citent (`configs/pipeline/brut.yaml` cite la règle 3).

1. **`minigpt/` (bloc 1) est codé avec Claude**, ce que l'orga a autorisé selon l'étudiant (pas de confirmation écrite dans ce dépôt, voir [remise/CONSIGNES_TRANSMISES.md](remise/CONSIGNES_TRANSMISES.md)). Condition : l'étudiant doit **expliquer chaque ligne** à la soutenance → code court, commenté en français, une fiche par composant. Règle de travail passée : seuls les agents du bloc 1 y écrivaient.
2. **Gel des données au 2026-10-16** : aucun contenu daté d'après.
3. **La val ne sert jamais à l'entraînement** : l'étape `decontaminate` retire tout document qui recouvre `data/validation.jsonl` (13-grammes, comme `ftb/contamination.py`).
4. Règle de travail passée : **pas de commit ni de push** par les agents ; la session principale s'en chargeait.
5. **Ne pas modifier** les fichiers du kit : cette partie vaut toujours. Pendant le projet, la règle couvrait aussi `pyproject.toml`, `uv.lock`, `Makefile`, `pipeline/io.py` et ce fichier, réservés à la session principale. Voir « Dépendances ».
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
data/stages/<config>/<NN>_<etape>/<source>/part-*.jsonl.zst   sorties du pipeline (ignoré par git)
data/stages/<config>/<NN>_<etape>/stats.json
data/stages/<config>/funnel.md              entonnoir lisible (docs et octets par étape)
data/mixes/<run>/train.jsonl.gz             dataset ordonné, prêt pour ftb.prepare (ignoré par git)
data/validation.jsonl                       val officielle du kit (suivie par git)
data/annotations/controle_langue_80/        contrôle du filtre de langue : 80 extraits classés automatiquement, sans annotation manuelle (suivi par git)
runs/<run>/                                 sorties de ftb.train (ignoré par git)
submissions/<date>_<heure>/                 soumissions officielles (suivies par git)
experiments/results.csv                     une ligne par run
minigpt/results/                            résultats du bloc 1 : RESULTS.md, metrics.json, courbe de loss, générations, tokenizers
chatbot/                                    chatbot Molière (hors concours) : code, recettes (recettes_rtx*.json), consignes de synthèse
runs/chatbot_rtx/                           sorties de chatbot.pipeline_rtx : journaux, états de reprise, modèles (ignoré par git)
experiments/chatbot_rtx/<nom>/              petits rapports d'un entraînement du chatbot, sans poids
docs/README.md                              table des matières de la documentation
docs/chatbot/                               chatbot : fiche du modèle, fiche des données, récit daté (HISTORIQUE.md), guide de transfert vers le PC
docs/comprendre/                            une fiche d'explication par composant
docs/sources/                               une fiche par source collectée, et le tableau des volumes (volumes.md)
docs/remise/                                rendus du concours : rapport, fiche et carte du jeu de données, exigences, reproduction
output/pdf/                                 rapport d'ablation et fiche du jeu de données en PDF
tools/README.md                             index des outils : une ligne par script
PC_RTX.md                                   guide d'entraînement du chatbot sur le PC à carte NVIDIA
LICENSE                                     texte de la licence Apache 2.0 (code écrit pour ce dépôt)
LICENCES.md                                 ce qui est sous quelle licence (code, poids, données, kit)
```

`data/raw/`, `data/stages/`, `data/mixes/`, `data/moliere/` et `runs/` sont ignorés par git : ils restent sur le poste de travail. Une preuve qu'une fiche cite dans ces dossiers n'est donc pas dans le dépôt ; les fiches l'écrivent « (non publié) ». Les jeux de données publiés sont des pièces jointes des releases GitHub `ftb-final-20261110` (jeu final du concours), `chatbot-donnees-20261113` (données du chatbot) et `chatbot-moliere-20261116` (conversations de synthèse du chatbot, `synthese_moliere.zip`, à côté du modèle publié).

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
| `uv run python -m ftb.submit --run runs/<run> --participant "NOM Prénom"` | Soumission (2 par jour) |
| `uv run pytest -q` | Tous les tests, kit compris |
| `python -m chatbot.pipeline_rtx <telecharger\|verifier\|autotest\|mesurer\|lancer\|etat\|resultats\|publier>` | Chatbot : données, contrôles, entraînement, archive des résultats (guide : [PC_RTX.md](../PC_RTX.md)) |
| `python -m chatbot.generate chat --model <dossier du modèle> --anchor-mu 1 --candidates 32 "message"` | Parler au chatbot (sans message : chat interactif) |
| `python tools/campagne_synthese.py <apercu\|generer\|bilan\|fusion\|refiltrer> --consignes <fichier> …` | Chatbot : conversations de synthèse, écrites par un modèle à poids ouverts servi par Ollama |

Les commandes en `uv run` valent pour macOS et Linux. Sur le PC Windows à carte NVIDIA, on n'utilise jamais `uv run` ni `uv sync` : ils installeraient un PyTorch sans CUDA. Seule exception : les commandes `uv run --no-project` du guide du concours, [remise/PASSAGE_PC_RTX5090.md](remise/PASSAGE_PC_RTX5090.md), qui ne synchronisent pas l'environnement (ce guide est une pièce historique : la route qu'il décrit, entraîner le modèle du concours sur ce PC, n'a jamais été exécutée). Sur ce PC, on écrit `.venv\Scripts\python.exe` à la place de `python`, avec `PYTHONUTF8=1`. Cet environnement est celui du chatbot (les trois dernières lignes du tableau) ; son installation est décrite dans [PC_RTX.md](../PC_RTX.md).

Réponse du chatbot : le choix ancré sur le dernier message (`chatbot/ancrage.py`) se règle par la clé `anchor_mu` du `chat_config.json` du modèle ou par l'option `--anchor-mu`. Réglage retenu : `anchor_mu` = 1, avec 32 candidates. Détail dans [chatbot/FICHE_MODELE.md](chatbot/FICHE_MODELE.md), § 5.

## Propriété des dossiers (règle de travail pendant le projet)

Pendant le projet, chaque chantier n'écrivait que dans ses dossiers. Cette règle ne vaut plus. Le tableau reste utile pour savoir à quel sujet se rattache un dossier.

| Dossier ou fichier | Chantier |
|---|---|
| `pipeline/collect/`, `configs/collect.yaml`, `docs/sources/` | Collecte |
| `pipeline/run.py`, `pipeline/stats.py`, `pipeline/clean/`, `pipeline/dedup/`, `configs/pipeline/`, `tools/` (outils du concours) | Nettoyage |
| `bench/`, `configs/bench/`, `experiments/`, `data/mixes/` | Banc d'essai |
| `minigpt/`, `tests/test_minigpt_*.py`, `docs/comprendre/minigpt_*.md` | Bloc 1 |
| `ftb/`, `config/`, `corpus/`, `data/validation.jsonl`, `data/VALIDATION_SOURCES.md`, `tests/test_kit.py`, `examples/`, `notebooks/`, `requirements.txt`, `docs/KIT_README.md` | Kit officiel : lecture seule |
| `chatbot/`, `tests/test_chatbot_*.py`, `tests/test_nettoyer_affinage.py`, les outils du chatbot dans `tools/` (`campagne_synthese.py`, `composer_consignes_v4.py`, `nettoyer_affinage.py`, `nettoyer_pretrain.py`, `nuit_moliere.py`, `trier_fineweb.py`, `build_chatbot_bundle.py`, `import_chatbot_results.py`, `publish_chatbot_release.py`), `docs/chatbot/`, `experiments/chatbot_rtx/`, `PC_RTX.md` | Chatbot (hors concours) |
| `pipeline/io.py`, `docs/ARCHITECTURE.md`, `Makefile`, `pyproject.toml`, `README.md`, `LICENSE`, `LICENCES.md` | Session principale |

Tests : `tests/test_<chantier>_*.py`, chaque chantier écrivait les siens (petits fichiers d'exemple dans `tests/fixtures/<chantier>/`). `tests/test_nettoyer_affinage.py`, test d'un outil du chatbot, ne suit pas ce nom.

## Fiches « comprendre » (`docs/comprendre/<sujet>.md`)

Chaque composant dispose d’une fiche courte pour préparer la soutenance, selon la structure suivante (quatre fiches, `audit_langue_complete.md`, `langue.md`, `nettoyage_cible.md` et `wikipedia_modeles.md`, ont des points numérotés et une première ligne qui donne l'état et le temps de lecture) :

1. **À quoi ça sert** (2 lignes)
2. **Comment ça marche** (5 puces maximum)
3. **Les chiffres clés** (seuils, volumes) et d'où ils viennent (article, référence)
4. **3 questions probables du jury**, avec une réponse courte

Phrases courtes, mots-clés en gras, aucun jargon sans définition.

## Code

- Python 3.12. Sur macOS et Linux : lancé avec `uv run`. Sur le PC Windows à carte NVIDIA : jamais `uv run` ni `uv sync` (ils installeraient un PyTorch sans CUDA) ; on lance `.venv\Scripts\python.exe` avec `PYTHONUTF8=1`, comme l'explique [PC_RTX.md](../PC_RTX.md).
- Identifiants en anglais, docstrings et commentaires en français, courts.
- Déterministe : seeds fixes, parcours de fichiers triés.
- Parallélisme : `multiprocessing`, **8 workers par défaut** (pendant le projet, plusieurs chantiers partageaient 15 cœurs et 24 Go de RAM).
- Traitements de plus de 5 minutes : lancés détachés (`nohup … > log 2>&1 &`), log dans le dossier de sortie, reprise possible après interruption.
- Disque : ~100 Go libres. Traitement en flux, fichiers temporaires supprimés après usage.

## Dépendances

Déclarées dans `pyproject.toml` : numpy, pandas, pyyaml, tqdm, regex, zstandard, requests, warcio, lxml, datasets, huggingface_hub, ftfy, trafilatura, datasketch, xxhash, tokenizers, torch (MPS disponible sur le Mac), matplotlib, fasttext (paquet `fasttext-numpy2`, `import fasttext`), mwparserfromhell ; pytest est dans le groupe `dev`.

Deux écarts connus au 16/11/2026 :

- `pandas` et `tqdm` sont déclarés, mais aucun fichier Python du dépôt ne les importe ;
- `tools/build_delivery_documents.py`, qui fabrique les deux PDF de `output/pdf/`, importe `reportlab` et `pypdf`, qui ne sont pas déclarés : il faut les installer à part pour relancer cet outil.

Le chatbot, sur le PC Windows, n'utilise pas cet environnement : [PC_RTX.md](../PC_RTX.md) installe ses propres paquets.

Règle de travail passée, pour ajouter un paquet pendant le projet : vérifier qu'aucune commande uv ne tourne (`pgrep -fl "uv (add|sync|lock)"`), lancer `uv add <paquet>`, et le signaler dans le rapport final.
