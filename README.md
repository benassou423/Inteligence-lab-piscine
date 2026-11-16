# Feed the Beast — nourrir un GPT avec le meilleur français possible

Piscine I Lab (ECE Paris), 16/10 → 16/11/2026. Ce dépôt contient deux choses :

1. **Le rendu d'un concours**, « Feed the Beast ». Le modèle et le budget sont les mêmes pour tous : seule la qualité des données fait la différence. La note est le BPB (bits par octet) : plus bas = mieux.
2. **Hors concours, un chatbot « à la manière de Molière »** : un GPT de 54 749 440 paramètres entraîné de zéro, dont les poids sont publiés (voir plus bas).

## Carte du dépôt

| Dossier | Contenu | Écrit par |
|---|---|---|
| `minigpt/` | Bloc 1 du concours, mini-GPT from scratch : autograd, bigramme, BPE, transformer, entraînement sur Molière | l'auteur |
| `pipeline/` | Bloc 2, pipeline de données : collecte → nettoyage → déduplication → filtrage qualité | l'auteur |
| `bench/` | Bloc 3, compétition : tokenizer 8 192, entraînement, BPB par domaine | l'auteur |
| `chatbot/` | Code, recettes et consignes de synthèse des chatbots ; table des essais dans [chatbot/README.md](chatbot/README.md) | l'auteur |
| `tools/` | Scripts d'audit, de préparation et de publication ; index dans [tools/README.md](tools/README.md) | l'auteur |
| `tests/` | Tests du projet (`uv run pytest -q`) | l'auteur, sauf `tests/test_kit.py` (kit) |
| `configs/` | Réglages YAML de l'auteur : collecte, nettoyage, runs du banc d'essai | l'auteur |
| `config/` | `official.json` : modèle et budget officiels du concours | **kit du concours** |
| `corpus/` | Corpus Molière du concours | **kit du concours** |
| `ftb/` | Code du concours : modèle, tokenizer, entraînement, évaluation, soumission | **kit du concours** |
| `data/` | `validation.jsonl` et `VALIDATION_SOURCES.md` (kit) ; `annotations/` (contrôle du filtre de langue, auteur). Le reste est ignoré par git | kit et auteur |
| `docs/` | Documentation ; table des matières dans [docs/README.md](docs/README.md) | l'auteur, sauf `docs/KIT_README.md` (kit) |
| `experiments/` | `results.csv` (runs du banc d'essai), rapports d'expérience, rapports du chatbot (`chatbot_rtx/`) | l'auteur |
| `submissions/` | Paquet de la soumission officielle du 11/11/2026, créé par `ftb.submit` | l'auteur |
| `output/` | `pdf/` : rapport d'ablation et fiche du jeu de données en PDF | l'auteur |
| `examples/` | Exemple du concours (`make_example_dataset.py`) | **kit du concours** |
| `notebooks/` | Carnet de démarrage du concours (`quickstart_colab.ipynb`) | **kit du concours** |

Le « kit du concours » est le code et les fichiers fournis par l'organisateur à tous les participants ; il est copié ici sans modification. À la racine, `requirements.txt` est le fichier du kit (Python ≥ 3.10) ; le projet, lui, s'installe avec `pyproject.toml` et `uv.lock` (Python 3.12). Ne sont pas dans git : `runs/` (sorties des calculs), les données lourdes de `data/` (publiées en pièces jointes des releases GitHub) et les poids du chatbot.

## Installer

Toutes les commandes se lancent depuis la racine du dépôt.

- **macOS ou Linux** : installer `uv`, puis `uv sync` (Python 3.12 et les dépendances de `uv.lock`) et `uv run pytest -q` (les tests). Pour refaire aussi le jeu du concours, ne pas lancer `uv sync` en premier : lancer d'abord la commande de la section « Refaire le jeu du concours ». Elle crée `.venv` sous forme de lien et refuse un dossier `.venv` ordinaire.
- **Windows avec une carte NVIDIA** : ne pas utiliser `uv run` ni `uv sync`. Ils installeraient un PyTorch sans CUDA. Suivre [PC_RTX.md](PC_RTX.md) (étapes 2 à 6) : il crée `.venv`, et on lance ensuite les commandes avec `.venv\Scripts\python.exe` et `PYTHONUTF8=1`. Ce guide n'installe pas `pytest`.
- **Seulement parler au chatbot** : il suffit d'un environnement Python 3.12 avec PyTorch, tokenizers et numpy (versions du projet : 2.15.0, 0.23.3 et 2.5.4). Sur macOS ou Linux, `uv sync` les installe.
- **Windows sans carte NVIDIA, Linux avec une carte NVIDIA** : non essayé dans ce projet.
- **Tests** : le 16/11/2026, `python -m pytest tests` sur le PC Windows, dans l'environnement réduit de [PC_RTX.md](PC_RTX.md), a donné 454 réussis, 10 ignorés, 9 échecs et 17 erreurs (105 s). Tous les échecs et erreurs viennent de cet environnement : sept modules du concours n'y sont pas installés (`ftfy`, `requests`, `mwparserfromhell`, `datasketch`, `warcio`, `pyarrow`, `matplotlib`) et un test (`tests/test_bench_kaggle.py`) utilise `os.killpg`, qui n'existe que sous Unix. Les tests du chatbot passent tous. Non relancé sur macOS le 16/11 : le dernier compte connu sur Mac est 360 tests au 11/11, avant les tests du chatbot.

## Le concours

### Soumission officielle — 11/11/2026

Le paquet est dans [submissions/2026-11-11_153839](submissions/2026-11-11_153839) : modèle, tokenizer, journaux, métadonnées et cinq générations. Il a été créé par `ftb.submit` sans modifier le kit. Entraînement Mac MPS float32 : **100 007 936 tokens vus**, score public de validation **1,288292 BPB**. Au 16/11/2026, c'est la seule soumission du dépôt. Le score du test caché et le contrôle T4 relèvent de l'organisateur : le dépôt n'en contient aucun résultat.

Le [dataset final](https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/ftb-final-20261110) contient 72 612 documents et 114 701 016 tokens disponibles. Ses dix fichiers ont été téléchargés sans authentification et vérifiés ; la préparation des fichiers retéléchargés reproduit la même empreinte `b88033ad2fba…`.

### Rendu du concours

- Rapport d'ablation : [docs/remise/RAPPORT.md](docs/remise/RAPPORT.md) (PDF : [output/pdf/rapport_ablation.pdf](output/pdf/rapport_ablation.pdf))
- Fiche du jeu de données : [docs/remise/DATASHEET.md](docs/remise/DATASHEET.md) (PDF : [output/pdf/datasheet.pdf](output/pdf/datasheet.pdf)) ; carte du jeu : [docs/remise/DATASET_CARD.md](docs/remise/DATASET_CARD.md)
- Audit des exigences : [docs/remise/EXIGENCES.md](docs/remise/EXIGENCES.md)
- Reproduction et ses limites : [docs/remise/REPRODUCTION.md](docs/remise/REPRODUCTION.md)
- Résultats des runs : [experiments/results.csv](experiments/results.csv) ; mini-GPT du bloc 1 : [minigpt/results/RESULTS.md](minigpt/results/RESULTS.md)

### Vérifier soi-même

Sur macOS ou Linux, après `uv sync` :

```bash
uv run python -m ftb.evaluate --ckpt submissions/2026-11-11_153839/ckpt.pt --tokenizer submissions/2026-11-11_153839/tokenizer.json
uv run python -m ftb.tokenizer check --tokenizer submissions/2026-11-11_153839/tokenizer.json
uv run --no-project python tools/reproduce.py --check-only
```

1. La première recalcule le BPB public (clé `bpb`). Le score déposé, 1,288292, a été mesuré à la fin de l'entraînement sur Mac (MPS, poids en float32) ; le fichier déposé stocke les poids en float16. Le résultat exact de ce recalcul n'est pas noté dans le dépôt.
2. La deuxième affiche l'empreinte du tokenizer (clé `sha256`) : `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`.
3. La troisième compare les 18 fichiers du kit à [docs/remise/KIT_SHA256.json](docs/remise/KIT_SHA256.json) et doit afficher `"kit_files_identical": 18`. Elle ne modifie rien.

### Refaire le jeu du concours

Depuis le dossier `Inteligence-lab-piscine`, sur macOS/Linux :

```bash
uv run --no-project python tools/reproduce.py --recipe final
```

Cette commande prépare les liens vers les dossiers `.nosync` avant l'environnement Python, collecte les sources, exécute le nettoyage de référence, recrée le tokenizer, sélectionne le mélange et masque les coordonnées. Elle contrôle la contamination et les hashes attendus. Le calcul tourne avec `nohup` et son journal est indiqué dans `data/stages/logs/`. Prévoir 100 Go libres pour une reconstruction intégrale, et 2 à 4 heures depuis zéro selon la connexion ([REPRODUCTION.md](docs/remise/REPRODUCTION.md)).

La variante expérimentale `--recipe complete` ajoute les filtres qualité web et MinHash/LSH. Elle a été évaluée séparément et n'est pas retenue pour le jeu final : elle dégrade le BPB sur les deux tirages et retire parfois des fiches d'entités différentes.

Après préparation de l'environnement, les tests et le mini-GPT du bloc 1 :

```bash
uv run pytest -q
uv run python -m minigpt.train_moliere --steps 3401
```

Le `Makefile` historique conserve les commandes de la référence de base. Les instructions de reproduction et leurs limites sont dans [REPRODUCTION.md](docs/remise/REPRODUCTION.md).

**Routes préparées et jamais exécutées.** Une route Windows pour entraîner le modèle du concours sur RTX 5090 a été préparée le 10/11/2026 ; la soumission a finalement été entraînée sur le Mac. Son guide, [PASSAGE_PC_RTX5090.md](docs/remise/PASSAGE_PC_RTX5090.md), est gardé comme pièce historique, avec ses outils (`tools/build_windows_bundle.py`, `tools/windows_gpu.py`, `tools/import_windows_results.py`). Il dépend d'une archive absente du dépôt, et ses commandes passent par `uv run`, que [PC_RTX.md](PC_RTX.md) interdit sur le PC Windows du projet (il installerait un PyTorch sans CUDA). La route Kaggle (`bench/kaggle.py`, `configs/bench/full.yaml`) n'a jamais été lancée non plus. Pour le chatbot, le guide est [PC_RTX.md](PC_RTX.md).

## Chatbot Molière (hors concours)

Un petit modèle de langue qui répond à des messages courts à la manière de Molière. Il est entraîné de zéro, sans aucun poids venu d'ailleurs, puis affiné sur du théâtre. C'est une démonstration : il imite un style et ne sait rien de fiable.

- **Taille** : 54 749 440 paramètres (640 dimensions, 10 couches, 10 têtes), fenêtre de 512 tokens, vocabulaire de 8 192 tokens.
- **Entraînement** : pré-entraînement sur 2 000 027 648 tokens de français écrit (6 sources), puis affinage sur 90 013 696 tokens (théâtre français et Molière, plus des conversations de synthèse selon le modèle), sur une carte NVIDIA GeForce RTX 5090.
- **Modèle publié : « S03 »** (3,02 % des tokens de l'affinage viennent de conversations de synthèse : 2 720 768 sur 90 013 696). Recette : [chatbot/recettes_rtx_publie.json](chatbot/recettes_rtx_publie.json). BPB sur les 3 pièces de validation de Molière : 1,12985 → 1,05585.
- **Résultat final (16/11/2026)** : jugement à l'aveugle sur un jeu de messages neuf (« neuf2 »), 144 réponses par modèle. **S03 reste le modèle retenu : aucun candidat n'atteint l'écart exigé** (+12 réponses sans défaut sur S03, fixé le 15/11 avant la génération). Les juges sont des agents IA, pas des personnes ; ceux du 16/11 (Claude Opus 5.5) sont un autre modèle d'IA que ceux des 14 et 15/11.

  | Modèle | Part de synthèse | Réponses sans défaut sur 144 | Écart avec S03 |
  |---|---|---|---|
  | A0 (sans synthèse) | 0 % | 73 | −23 |
  | **S03 (publié)** | 3,02 % | **96** | — |
  | meilleur candidat (synthèse v4 + v5 à 15 %) | 14,98 % | 105 | +9 |

  Les trois autres candidats (3, 5 et 10 %) sont à −3, +5 et −8. Le candidat à 15 % passe la garde de par-cœur de justesse (9,81 % pour une limite de 10 %). Un tiers des réponses de S03 (48 sur 144) gardent un défaut. Qualité jugée par des personnes : jamais mesurée. Détail : [experiments/chatbot_rtx/jugements/README.md](experiments/chatbot_rtx/jugements/README.md).
- **Poids** : release GitHub **`chatbot-moliere-20261116`** de ce dépôt ([adresse](https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116)). Pièce jointe `chatbot_moliere_s03.zip` (240 087 338 octets, sha256 `6d62aa50f4917fbb23d24e24a8d64807bde55f273c9cb4203c6970c32c886417`), qui contient le dossier `chatbot_moliere/`. Empreinte sha256 de `ckpt.pt` (239 991 123 octets) : `01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577`.
- **Conversations de synthèse** : pièce jointe `synthese_moliere.zip` de la même release (4 442 456 octets, 22 fichiers, CC BY 4.0). À décompresser à la RACINE du clone : les fichiers se rangent aux chemins que lisent les recettes. Avec les données de la release `chatbot-donnees-20261113` (`python -m chatbot.pipeline_rtx telecharger`, voir [PC_RTX.md](PC_RTX.md)), la recette du modèle publié se relance depuis un clone ; un réentraînement ne redonne pas forcément les mêmes octets, ce qui n'a pas été essayé.

Pour lui parler : télécharger `chatbot_moliere_s03.zip` dans la release, le décompresser à la racine du dépôt, puis lancer (environnement : section « Installer ») :

```bash
python -m chatbot.generate chat --model chatbot_moliere --persona TOINETTE "Je me sens mourir."
```

Sur macOS ou Linux, écrire `uv run python` à la place de `python` ; sous Windows, `.venv\Scripts\python.exe` avec `PYTHONUTF8=1` (section « Installer »).

`chatbot_moliere` est le dossier du modèle (`ckpt.pt`, `tokenizer.json`, `chat_config.json`, `training_report.json`). Sans message, le chat est interactif. Sans `--persona`, le personnage est MOLIÈRE (11 personnages proposés). Décodage retenu : 32 réponses candidates, et on garde celle qui dépend le plus du dernier message (`chatbot/ancrage.py`, `anchor_mu` = 1). Avec ce modèle, ces options sont déjà dans son `chat_config.json` (`"candidates": 32`, `"anchor_mu": 1.0`) : la commande n'a pas besoin de `--anchor-mu 1 --candidates 32` (elles restent permises). Un modèle exporté par le lanceur avec la recette par défaut demande 16 candidates, sans ancrage.

- Fiche du modèle (architecture, mesures, limites) : [docs/chatbot/FICHE_MODELE.md](docs/chatbot/FICHE_MODELE.md)
- Fiche des données (sources, préparation, licences) : [docs/chatbot/FICHE_DONNEES.md](docs/chatbot/FICHE_DONNEES.md)
- Comment il marche, étape par étape : [docs/comprendre/chatbot_moliere.md](docs/comprendre/chatbot_moliere.md)
- Ce qui a été essayé, jour par jour, échecs compris : [docs/chatbot/HISTORIQUE.md](docs/chatbot/HISTORIQUE.md)
- Preuves des jugements à l'aveugle : [experiments/chatbot_rtx/jugements/README.md](experiments/chatbot_rtx/jugements/README.md)
- Guide d'entraînement sur un PC Windows à carte NVIDIA : [PC_RTX.md](PC_RTX.md)

**Un second chatbot, « français courant ».** Le lancement du 13/11/2026 a aussi affiné le même modèle de base sur des dialogues tirés de romans (Project Gutenberg, Wikisource) et sur du théâtre en prose, au format « — réplique » : 150 011 904 tokens vus, BPB de validation 1,15956 → 1,04751. Son code est `chatbot/dialogues.py`, ses rapports sont dans `experiments/chatbot_rtx/20261113_1956_M/`, et ses trois fichiers de dialogues sont dans la release `chatbot-donnees-20261113`. Il n'est pas le sujet de la publication et n'est pas dans le paquet du modèle (`chatbot_moliere_s03.zip`). Ses poids sont seulement dans l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`, les deux chatbots et leurs rapports), une autre pièce jointe de la release `chatbot-moliere-20261116`.

## Ce qui s'est passé

- **03/11/2026** : squelette du projet et kit du concours intégré sans modification ; premiers runs de classement (Wikipédia seule 1,7467 BPB, Molière seul 6,3848).
- **04/11** : Wikipédia reconstruite depuis le dump `frwiki` 20261001 ; un nettoyage ciblé supplémentaire, mesuré moins bon (+0,017365 BPB), est rejeté.
- **10/11** : fin de l'ablation à six runs (brut 1,657747, nettoyé 1,622266, complet 1,632570 BPB en moyenne). La variante complète est rejetée (+0,010304). Entraînement final au budget officiel.
- **11/11** : soumission officielle à 1,288292 BPB ; le jeu de données est public (release `ftb-final-20261110`).
- **13/11** : chatbot. Modèle de base entraîné de zéro sur la RTX 5090, puis affinages, dont « A0 » (Molière, sans synthèse) : 1 h 57 min en tout.
- **14/11** : des essais négatifs (première synthèse, données nettoyées) et un gain : le choix de la réponse par ancrage (rapport au message 96 → 114 sur 144).
- **15/11** : « S03 » (3,02 % de synthèse) donne 100 réponses sans défaut sur 144 contre 82 pour A0. Les parts de 5, 10 et 15 % sont écartées : elles dépassent la limite de par-cœur fixée d'avance (le modèle récite la synthèse).
- **16/11** : licences, fiches, arrêt de la génération de synthèse à 61 % des appels prévus (29 468 sur 48 000 ; le fichier d'entraînement compte 27 624 conversations, v4 comprise) et essais finaux : quatre candidats avec 3, 5, 10 et 15 % de synthèse, jugés à l'aveugle. Aucun n'atteint +12 sur S03 (au mieux +9) : S03 reste le modèle retenu et il est publié dans la release `chatbot-moliere-20261116`.

Le détail est dans [docs/remise/RAPPORT.md](docs/remise/RAPPORT.md) pour le concours et dans [docs/chatbot/HISTORIQUE.md](docs/chatbot/HISTORIQUE.md) pour le chatbot.

## Documentation

La table des matières de tous les documents est dans [docs/README.md](docs/README.md). Les conventions communes sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Les explications de chaque étape sont dans `docs/comprendre/`, les fiches des sources dans `docs/sources/`.

## Licences

Le code écrit pour ce dépôt est sous Apache 2.0 ([LICENSE](LICENSE)), les poids des modèles sous CC BY-SA 4.0, les conversations de synthèse sous CC BY 4.0. Le kit du concours (`ftb/`, `config/`, `corpus/`, `examples/`, `notebooks/` et les fichiers du kit listés dans LICENCES.md) n'est pas couvert, et chaque texte d'entraînement garde la licence de sa source. Ce qui s'applique à quoi est écrit dans [LICENCES.md](LICENCES.md).
