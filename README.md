# Feed the Beast — nourrir un GPT avec le meilleur français possible

Piscine I Lab (ECE Paris), 16/10 → 16/11/2026. Ce dépôt contient trois choses :

1. **Le rendu d'un concours**, « Feed the Beast ». Le modèle et le budget sont les mêmes pour tous : seule la qualité des données fait la différence. La note est le BPB (bits par octet) : plus bas = mieux.
2. **Hors concours, un chatbot « à la manière de Molière »** : un GPT de 54 749 440 paramètres entraîné de zéro, dont les poids sont publiés.
3. **Deux façons de lui parler hors du terminal** : une application macOS à télécharger, sans aucune commande, et une démo web locale, lancée par trois commandes puis utilisée dans le navigateur.

Pour essayer tout de suite : section [Lancer l'application et la démo](#lancer-lapplication-et-la-démo). Pour savoir ce qui était demandé et ce qui a été fait en plus : section suivante.

## Ce qui était prévu, ce qui a été demandé ensuite, ce qui va au-delà

Le dépôt est plus gros que le sujet de départ. Cette section dit d'où vient chaque partie, pour qu'on ne prenne pas un travail hors concours pour une exigence, ni l'inverse.

### 1. Prévu à l'origine : le sujet du concours

Le sujet « Feed the Beast » demande trois blocs et un rendu. C'est tout ce qui est noté.

| Demandé par le sujet | Où c'est dans le dépôt | État |
|---|---|---|
| **Bloc 1** : un mini-GPT « from scratch » (autograd vérifié contre PyTorch, bigramme, BPE, transformer), entraîné sur Molière, avec courbe, BPB et cinq générations. Le sujet réserve ce bloc à l'étudiant, qui doit pouvoir en expliquer chaque ligne | `minigpt/` | Fait : 1 071 872 paramètres, 1,738 BPB. Il a été codé avec un assistant IA, avec l'accord de l'organisateur d'après l'étudiant (pas de confirmation écrite dans le dépôt : [ARCHITECTURE.md](docs/ARCHITECTURE.md), règle 1). L'explication ligne par ligne se vérifie à la soutenance |
| **Bloc 2** : un pipeline de données reproductible en une commande (Common Crawl brut et au moins deux sources propres, encodage, langue, texte parasite des pages, doublons exacts et quasi-doublons MinHash + LSH, qualité, masquage des données personnelles) et une ablation brut / nettoyé / complet | `pipeline/`, `tools/reproduce.py`, [RAPPORT.md](docs/remise/RAPPORT.md) | Fait, avec quatre écarts : (1) la précision du filtre de langue devait être mesurée sur un échantillon annoté par une personne ; elle l'a été sur un classement automatique de 80 extraits ; (2) Common Crawl : 10 fichiers WARC d'un même segment, pas le segment entier ; (3) la commande unique a été rejouée avec les données brutes déjà collectées, pas avec un nouveau téléchargement complet ; (4) MinHash + LSH et le filtre de qualité sont codés et mesurés dans la variante « complet », qui n'est pas retenue pour le jeu soumis parce qu'elle dégrade le BPB |
| **Bloc 3** : la compétition. Modèle figé de 10 166 400 paramètres, 100 007 936 tokens vus, tokenizer de 8 192 tokens, données documentées, soumission par `ftb.submit` | `bench/`, `submissions/` | Fait : une soumission, le 11/11/2026, 1,288292 BPB sur la validation publique. Le score du test caché et le réentraînement de contrôle par l'organisateur ne sont pas dans le dépôt |
| **Rendu** : rapport d'ablation de deux pages, fiche du jeu de données, soutenance | `docs/remise/`, `output/pdf/` | Rapport et fiche faits (le PDF de la fiche date du 10/11 ; le texte à jour est dans DATASHEET.md) ; la soutenance n'a pas encore eu lieu |

Le sujet propose aussi des pistes et des options. **Elles n'ont pas été explorées** : proportions du mélange, répétition des données, ordre des données, choix du tokenizer (aucune comparaison de tokenizers n'est rapportée), classifieur de qualité (seules des règles simples sont utilisées), données de synthèse dans le jeu du concours (aucune n'a été ajoutée au jeu de la soumission). La publication sur Hugging Face, qui relevait d'un prix bonus, n'a pas été faite : le jeu est publié en pièces jointes d'une release GitHub. Le suivi ligne par ligne est dans [EXIGENCES.md](docs/remise/EXIGENCES.md), les consignes dans [CONSIGNES_TRANSMISES.md](docs/remise/CONSIGNES_TRANSMISES.md).

### 2. Demandé ensuite par le référent du projet (11/11/2026), hors concours

Le 11/11/2026, le référent a demandé un modèle avec lequel on discute : « tu lui parles, et il te répond à la façon de Molière » ([chatbot_moliere.md](docs/comprendre/chatbot_moliere.md)). Rien de cela ne compte dans le BPB du concours.

- **Deux premiers chatbots ont été faits sur le Mac** : le modèle du concours affiné sur du théâtre, puis un modèle de 29,6 M de paramètres entraîné de zéro. Ils sont remplacés ; leur histoire est dans [chatbot_moliere.md](docs/comprendre/chatbot_moliere.md).
- **La démo web** (`demo/`) a été faite pour montrer ces modèles. Elle a été mise à niveau le 16/11 pour le modèle publié.

### 3. Au-delà de ces demandes : choix de l'auteur

Tout ce qui suit n'était pas demandé par le sujet, et va plus loin que la demande du référent telle que le dépôt la rapporte.

| Ce qui a été ajouté | Pourquoi | Où |
|---|---|---|
| Un modèle dédié de 54 749 440 paramètres, entraîné de zéro sur une carte RTX 5090 (2 000 027 648 tokens) | Le modèle du concours sature quand on l'affine (`chatbot/pretrain.py`) ; un premier modèle de 29,6 M avait été entraîné sur le Mac. Sur la RTX 5090, on a pris la plus grande taille qui tenait en 3 heures de calcul | [FICHE_MODELE.md](docs/chatbot/FICHE_MODELE.md), [PC_RTX.md](PC_RTX.md) |
| Des conversations de synthèse écrites par des modèles à poids ouverts, avec leurs consignes publiées. Trois campagnes : un premier essai (1 305 conversations, écarté), la v4 (9 055, celles du modèle publié) et la v5 (20 394, pour les essais du 16/11 seulement) | Mieux répondre au message reçu : avec 3,02 % de synthèse dans l'affinage, le modèle publié donne 96 réponses sans défaut sur 144, contre 73 sans synthèse (jugement du 16/11) | [FICHE_DONNEES.md](docs/chatbot/FICHE_DONNEES.md), `chatbot/synthetic.py` |
| Des jugements à l'aveugle par des agents IA, avec une règle de décision écrite avant les notes (les exceptions sont déclarées dans HISTORIQUE.md, « Méthode et limites »), et les essais ratés gardés | Savoir si un changement améliore vraiment les réponses, et pouvoir le prouver | [experiments/chatbot_rtx/jugements/README.md](experiments/chatbot_rtx/jugements/README.md), [HISTORIQUE.md](docs/chatbot/HISTORIQUE.md) |
| Un second chatbot, « français courant » (nommé « Français moderne » dans la démo) | Un chatbot qui répond en français d'aujourd'hui, entraîné dans le même lancement ; il n'a pas été jugé à l'aveugle | [chatbot/README.md](chatbot/README.md), § 3 |
| La publication : poids du modèle, données d'entraînement (15 fichiers, 4 344 702 490 octets), fiche du modèle, fiche des données, licences | Permettre de vérifier les résultats et de relancer la recette du modèle publié (ce réentraînement n'a pas été essayé) | releases `chatbot-moliere-20261116` et `chatbot-donnees-20261113`, [LICENCES.md](LICENCES.md) |
| L'application macOS « Molière », publiée en `.dmg` | Parler au chatbot sans installer Python ni taper de commande | `app_moliere/`, [APPLICATION.md](docs/chatbot/APPLICATION.md) |
| Côté concours : Wikipédia reconstruite depuis le dump officiel du 01/10/2026, un audit du filtre de langue sur des documents complets, un nettoyage ciblé essayé puis rejeté | Améliorer le texte (dates et nombres perdus dans l'ancienne version de Wikipédia), vérifier le filtre de langue sur des textes entiers, et garder la trace de ce qui n'a pas marché | [wikipedia_fr.md](docs/sources/wikipedia_fr.md), [audit_langue_complete.md](docs/comprendre/audit_langue_complete.md), [nettoyage_cible.md](docs/comprendre/nettoyage_cible.md) |

Ce que ce travail en plus **ne change pas** : la soumission du concours est celle du 11/11/2026, entraînée avec le kit non modifié. Le chatbot ne peut pas être soumis (le kit refuse tout modèle hors de sa configuration) et son BPB, mesuré sur d'autres textes, ne se compare pas à celui du concours.

## Carte du dépôt

| Dossier | Contenu | Écrit par |
|---|---|---|
| `minigpt/` | Bloc 1 du concours, mini-GPT from scratch : autograd, bigramme, BPE, transformer, entraînement sur Molière | l'auteur |
| `pipeline/` | Bloc 2, pipeline de données : collecte → nettoyage → déduplication → filtrage qualité | l'auteur |
| `bench/` | Bloc 3, compétition : tokenizer 8 192, entraînement, BPB par domaine | l'auteur |
| `chatbot/` | Code, recettes et consignes de synthèse des chatbots ; table des essais dans [chatbot/README.md](chatbot/README.md) | l'auteur |
| `app_moliere/` | Application macOS « Molière » : fenêtre de discussion, serveur local, téléchargement vérifié du modèle ; guide dans [docs/chatbot/APPLICATION.md](docs/chatbot/APPLICATION.md) | l'auteur |
| `demo/` | Démo web locale : chat avec les chatbots et les modèles du concours, ateliers pédagogiques | l'auteur |
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
| `output/` | `pdf/` : rapport d'ablation et fiche du jeu de données en PDF ; `icone-moliere/` : image d'origine de l'icône de l'application et sa consigne | l'auteur |
| `examples/` | Exemple du concours (`make_example_dataset.py`) | **kit du concours** |
| `notebooks/` | Carnet de démarrage du concours (`quickstart_colab.ipynb`) | **kit du concours** |

Le « kit du concours » est le code et les fichiers fournis par l'organisateur à tous les participants ; il est copié ici sans modification. « L'auteur » désigne le reste, c'est-à-dire le projet : son code a été écrit avec des assistants IA (Claude Code, Codex), ce que disent [ARCHITECTURE.md](docs/ARCHITECTURE.md) et [HISTORIQUE.md](docs/chatbot/HISTORIQUE.md). À la racine, `requirements.txt` est le fichier du kit (Python ≥ 3.10) ; le projet, lui, s'installe avec `pyproject.toml` et `uv.lock` (Python 3.12). Ne sont pas dans git : `runs/` (sorties des calculs), les données lourdes de `data/` (publiées en pièces jointes des releases GitHub) et les poids du chatbot.

## Installer

Toutes les commandes se lancent depuis la racine du dépôt, c'est-à-dire le dossier `Inteligence-lab-piscine` obtenu par :

```bash
git clone https://github.com/benassou423/Inteligence-lab-piscine.git
cd Inteligence-lab-piscine
```

`uv` (le gestionnaire qui installe Python et les dépendances) s'installe depuis https://docs.astral.sh/uv/.

- **macOS ou Linux** : installer `uv`, puis `uv sync` (Python 3.12 et les dépendances de `uv.lock`) et `uv run pytest -q` (les tests). Pour refaire aussi le jeu du concours, ne pas lancer `uv sync` en premier : lancer d'abord la commande de la section « Refaire le jeu du concours ». Elle crée `.venv` sous forme de lien et refuse un dossier `.venv` ordinaire.
- **Windows avec une carte NVIDIA** : ne pas utiliser `uv run` ni `uv sync`. Ils installeraient un PyTorch sans CUDA. Suivre [PC_RTX.md](PC_RTX.md) (étapes 2 à 6) : il crée `.venv`, et on lance ensuite les commandes avec `.venv\Scripts\python.exe` et `PYTHONUTF8=1`. Ce guide n'installe pas `pytest`.
- **Seulement parler au chatbot** : il suffit d'un environnement Python 3.12 avec PyTorch, tokenizers et numpy (versions du projet : 2.15.0, 0.23.3 et 2.5.4). Sur macOS ou Linux, `uv sync` les installe.
- **Windows sans carte NVIDIA, Linux avec une carte NVIDIA** : non essayé dans ce projet.
- **Tests** : le 16/11/2026, `python -m pytest tests` sur le PC Windows, dans l'environnement réduit de [PC_RTX.md](PC_RTX.md), a donné 454 réussis, 10 ignorés, 9 échecs et 17 erreurs (105 s). Tous les échecs et erreurs viennent de cet environnement : sept modules du concours n'y sont pas installés (`ftfy`, `requests`, `mwparserfromhell`, `datasketch`, `warcio`, `pyarrow`, `matplotlib`) et un test (`tests/test_bench_kaggle.py`) utilise `os.killpg`, qui n'existe que sous Unix. Les tests du chatbot passent tous. Sur macOS, le 16/11/2026, `uv run --group app pytest -q` a donné 907 réussis, 0 échec (146 s), tests de la démo web et de l'application compris.

## Lancer l'application et la démo

Quatre façons de parler au chatbot, de la plus simple à la plus technique. Les trois dernières demandent un clone du dépôt et `uv` (section « Installer »). Les commandes des parties B et C ont été rejouées le 16/11/2026 dans un clone neuf du dépôt public, sur macOS.

### A. L'application macOS (aucune installation de Python)

Il faut : un Mac à puce Apple (M1 ou plus récent ; pas de Mac Intel), macOS 14 ou plus, environ 1,5 Go d'espace libre pendant l'installation, et Internet au premier lancement seulement.

1. Télécharger [`Moliere-1.0.0-macos-arm64.dmg`](https://github.com/benassou423/Inteligence-lab-piscine/releases/download/chatbot-moliere-20261116/Moliere-1.0.0-macos-arm64.dmg) (224 746 507 octets, sha256 `228aad290b9ab0a119891952b5ac66d47772801af17c50530a1ce2d67e4a7a35`), pièce jointe de la release `chatbot-moliere-20261116`.
2. Ouvrir le `.dmg` et glisser Moliere dans Applications.
3. Ouvrir Moliere. L'application n'est pas notarisée par Apple : macOS la bloque la première fois. Cliquer sur **Terminé**, puis, dans l'heure qui suit, **Réglages Système > Confidentialité et sécurité > Ouvrir quand même**, et confirmer (il faut un compte administrateur). Sur macOS 14, un clic droit sur l'application puis **Ouvrir** suffit.
4. Dans l'application, cliquer sur **Télécharger le modèle** (240 Mo, une seule fois, vérifié par empreinte sha256). Ensuite elle fonctionne sans Internet.

Le pas-à-pas complet, la désinstallation et les limites sont dans [docs/chatbot/APPLICATION.md](docs/chatbot/APPLICATION.md). L'application a été essayée sur un seul Mac (macOS 26.6).

### B. La démo web locale (essayée sur macOS seulement)

```bash
uv sync
uv run python -m app_moliere.modele --dest data/models/chatbot_moliere
uv run python -m demo --port 8766
```

1. `uv sync` installe Python 3.12 et les dépendances.
2. La deuxième commande télécharge le modèle publié (240 Mo) depuis la release, vérifie ses deux empreintes et le range là où la démo le cherche. Elle refuse d'écraser un dossier existant sans `--remplacer`. Si elle répond « Impossible de joindre le serveur de téléchargement », la relancer : rien n'est installé tant que le fichier n'est pas complet et vérifié.
3. La troisième lance la démo. Ouvrir ensuite http://127.0.0.1:8766 dans un navigateur ; `Ctrl+C` l'arrête. Si le port est pris, changer le numéro après `--port`.

La démo ne répond qu'à l'ordinateur qui la lance (adresse `127.0.0.1`) et calcule sur le processeur : 2 à 5 secondes par réponse du chatbot sur le Mac de l'auteur. Son menu propose quatre modèles ; un modèle absent est grisé.

| Entrée du menu | Ce que c'est | Ce qu'il faut pour l'avoir |
|---|---|---|
| Molière · chatbot | Le modèle publié S03, 11 personnages | La deuxième commande ci-dessus |
| Français moderne · chatbot | Le second chatbot (« français courant ») | Ses poids ne sont que dans l'archive des résultats (445 Mo) : `uv run python tools/import_chatbot_results.py --depuis-release benassou423/Inteligence-lab-piscine chatbot-moliere-20261116 --installer`. Cette commande demande l'outil `gh` connecté à GitHub. Elle installe les deux chatbots dans `data/models/` : pour avoir les deux, elle suffit, et la deuxième commande ci-dessus devient inutile. Un dossier `chatbot_moliere` déjà présent est gardé à côté, renommé `chatbot_moliere.avant_<date_heure>`. Relancer ensuite la démo |
| Modèle du concours · complétion | Le modèle de la soumission : il prolonge un texte, il ne discute pas | Rien : il est dans `submissions/` |
| Mini GPT Molière · complétion | Le mini-GPT du bloc 1 | `uv run python -m minigpt.train_moliere --steps 3401` (11,6 minutes sur le Mac de l'auteur). Attention : cette commande réécrit aussi les résultats suivis par git dans `minigpt/results/` (`RESULTS.md`, `metrics.json`, `samples.md`, `loss_curve.png`, `tokenizer.json`) ; sur une autre machine, les chiffres peuvent changer |

La page http://127.0.0.1:8766/atelier propose cinq ateliers pédagogiques (gradient, tokens, nettoyage, mesures, génération). `tools/start_demo.py` lance la même démo en arrière-plan, mais il est fait pour le Mac de l'auteur : il recrée des liens vers des dossiers hors iCloud.

Après ces commandes, `.venv` et `data/models/` sont des dossiers ordinaires : `tools/reproduce.py` et `tools/start_demo.py` refusent alors de tourner. Pour refaire aussi le jeu du concours, utiliser un autre clone (section « Refaire le jeu du concours »).

### C. L'application depuis les sources, et sa reconstruction (macOS)

```bash
uv run --group app python -m app_moliere                # la fenêtre de l'application
uv run --group app python -m app_moliere --navigateur   # la même page, dans le navigateur
uv run --group app python tools/build_app_macos.py      # reconstruire l'application et le .dmg
```

Le groupe `app` ajoute pywebview, PyInstaller et certifi. Lancée depuis les sources, l'application cherche le modèle dans le dossier donné par la variable `MOLIERE_MODELE`, puis dans celui de l'application installée (`~/Library/Application Support/Moliere/modele/`), puis dans `data/models/chatbot_moliere/` ; sinon elle propose de le télécharger. La construction prend environ 2 minutes et écrit le `.dmg`, l'application et `rapport_construction.json` dans `data/stages/app_moliere_20261116/` ; elle teste le programme emballé avec le modèle de `data/models/chatbot_moliere/`, qui doit donc être présent. Une application reconstruite n'a pas la même empreinte que celle de la release. Détail : [docs/chatbot/APPLICATION.md](docs/chatbot/APPLICATION.md), § 2.

### D. En ligne de commande

Voir « Pour lui parler » dans la section [Chatbot Molière](#chatbot-molière-hors-concours).

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
- **Application macOS** : pièce jointe `Moliere-1.0.0-macos-arm64.dmg` de la même release (224 746 507 octets, sha256 `228aad290b9ab0a119891952b5ac66d47772801af17c50530a1ce2d67e4a7a35`), ajoutée le 16/11/2026.
- **Conversations de synthèse** : pièce jointe `synthese_moliere.zip` de la même release (4 442 456 octets, 22 fichiers, CC BY 4.0). À décompresser à la RACINE du clone : les fichiers se rangent aux chemins que lisent les recettes. Avec les données de la release `chatbot-donnees-20261113` (`python -m chatbot.pipeline_rtx telecharger`, voir [PC_RTX.md](PC_RTX.md)), la recette du modèle publié se relance depuis un clone ; un réentraînement ne redonne pas forcément les mêmes octets, ce qui n'a pas été essayé.

Pour lui parler : télécharger `chatbot_moliere_s03.zip` dans la release, le décompresser à la racine du dépôt, puis lancer (environnement : section « Installer ») :

```bash
python -m chatbot.generate chat --model chatbot_moliere --persona TOINETTE "Je me sens mourir."
```

Sur macOS ou Linux, écrire `uv run python` à la place de `python` ; sous Windows, `.venv\Scripts\python.exe` avec `PYTHONUTF8=1` (section « Installer »).

`chatbot_moliere` est le dossier du modèle (`ckpt.pt`, `tokenizer.json`, `chat_config.json`, `training_report.json`). Sans message, le chat est interactif. Sans `--persona`, le personnage est MOLIÈRE (11 personnages proposés). Décodage retenu : 32 réponses candidates, et on garde celle qui dépend le plus du dernier message (`chatbot/ancrage.py`, `anchor_mu` = 1). Avec ce modèle, ces options sont déjà dans son `chat_config.json` (`"candidates": 32`, `"anchor_mu": 1.0`) : la commande n'a pas besoin de `--anchor-mu 1 --candidates 32` (elles restent permises). Un modèle exporté par le lanceur avec la recette par défaut demande 16 candidates, sans ancrage.

**Sans écrire de commande à chaque message** : l'application macOS et la démo web, section [Lancer l'application et la démo](#lancer-lapplication-et-la-démo). Explication courte de l'application : [docs/comprendre/application_moliere.md](docs/comprendre/application_moliere.md).

- Fiche du modèle (architecture, mesures, limites) : [docs/chatbot/FICHE_MODELE.md](docs/chatbot/FICHE_MODELE.md)
- Fiche des données (sources, préparation, licences) : [docs/chatbot/FICHE_DONNEES.md](docs/chatbot/FICHE_DONNEES.md)
- Comment il marche, étape par étape : [docs/comprendre/chatbot_moliere.md](docs/comprendre/chatbot_moliere.md)
- Ce qui a été essayé, jour par jour, échecs compris : [docs/chatbot/HISTORIQUE.md](docs/chatbot/HISTORIQUE.md)
- Preuves des jugements à l'aveugle : [experiments/chatbot_rtx/jugements/README.md](experiments/chatbot_rtx/jugements/README.md)
- L'application macOS (installer, reconstruire, limites) : [docs/chatbot/APPLICATION.md](docs/chatbot/APPLICATION.md)
- Guide d'entraînement sur un PC Windows à carte NVIDIA : [PC_RTX.md](PC_RTX.md)

**Un second chatbot, « français courant »** (la démo web l'appelle « Français moderne »). Le lancement du 13/11/2026 a aussi affiné le même modèle de base sur des dialogues tirés de romans (Project Gutenberg, Wikisource) et sur du théâtre en prose, au format « — réplique » : 150 011 904 tokens vus, BPB de validation 1,15956 → 1,04751. Son code est `chatbot/dialogues.py`, ses rapports sont dans `experiments/chatbot_rtx/20261113_1956_M/`, et ses trois fichiers de dialogues sont dans la release `chatbot-donnees-20261113`. Il n'est pas le sujet de la publication et n'est pas dans le paquet du modèle (`chatbot_moliere_s03.zip`). Ses poids sont seulement dans l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`, les deux chatbots et leurs rapports), une autre pièce jointe de la release `chatbot-moliere-20261116`.

## Ce qui s'est passé

- **03/11/2026** : squelette du projet et kit du concours intégré sans modification ; premiers runs de classement (Wikipédia seule 1,7467 BPB, Molière seul 6,3848).
- **04/11** : Wikipédia reconstruite depuis le dump `frwiki` 20261001 ; un nettoyage ciblé supplémentaire, mesuré moins bon (+0,017365 BPB), est rejeté.
- **10/11** : fin de l'ablation à six runs (brut 1,657747, nettoyé 1,622266, complet 1,632570 BPB en moyenne). La variante complète est rejetée (+0,010304). Entraînement final au budget officiel.
- **11/11** : soumission officielle à 1,288292 BPB ; le jeu de données est public (release `ftb-final-20261110`).
- **11/11 → 13/11** : le référent demande un modèle avec lequel on discute. Démo web locale, puis deux premiers chatbots entraînés sur le Mac (le modèle du concours affiné, puis un modèle de 29,6 M de paramètres) ; ils sont remplacés depuis.
- **13/11** : chatbot sur le PC. Modèle de base entraîné de zéro sur la RTX 5090, puis affinages, dont « A0 » (Molière, sans synthèse) : 1 h 57 min en tout.
- **14/11** : des essais négatifs (première synthèse, données nettoyées) et un gain : le choix de la réponse par ancrage (rapport au message 96 → 114 sur 144).
- **15/11** : « S03 » (3,02 % de synthèse) donne 100 réponses sans défaut sur 144 contre 82 pour A0. Les parts de 5, 10 et 15 % sont écartées : elles dépassent la limite de par-cœur fixée d'avance (le modèle récite la synthèse).
- **16/11** : licences, fiches, arrêt de la génération de synthèse à 61 % des appels prévus (29 468 sur 48 000 ; le fichier des essais du 16/11 compte 27 624 conversations, v4 comprise ; le modèle publié n'a vu que les 9 055 de la v4) et essais finaux : quatre candidats avec 3, 5, 10 et 15 % de synthèse, jugés à l'aveugle. Aucun n'atteint +12 sur S03 (au mieux +9) : S03 reste le modèle retenu et il est publié dans la release `chatbot-moliere-20261116`. Le même jour : démo web mise à niveau pour ce modèle, et application macOS « Molière » construite puis publiée dans la même release.

Le détail est dans [docs/remise/RAPPORT.md](docs/remise/RAPPORT.md) pour le concours et dans [docs/chatbot/HISTORIQUE.md](docs/chatbot/HISTORIQUE.md) pour le chatbot.

## Documentation

La table des matières de tous les documents est dans [docs/README.md](docs/README.md). Les conventions communes sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Les explications de chaque étape sont dans `docs/comprendre/`, les fiches des sources dans `docs/sources/`.

## Licences

Le code écrit pour ce dépôt, application et démo comprises, est sous Apache 2.0 ([LICENSE](LICENSE)), les poids des modèles sous CC BY-SA 4.0, les conversations de synthèse sous CC BY 4.0. Le kit du concours (`ftb/`, `config/`, `corpus/`, `examples/`, `notebooks/` et les fichiers du kit listés dans LICENCES.md) n'est pas couvert, et chaque texte d'entraînement garde la licence de sa source. L'application macOS embarque ce kit et des logiciels tiers qui gardent leur licence : comme le kit n'a pas de licence, la redistribuer en dehors de ce dépôt demande l'accord de l'auteur du kit ([APPLICATION.md](docs/chatbot/APPLICATION.md), § 3) ; aucune licence n'est revendiquée sur son icône, produite par un outil de génération d'images. Ce qui s'applique à quoi est écrit dans [LICENCES.md](LICENCES.md).
