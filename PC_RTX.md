# PC_RTX.md — instructions de la session Claude Code du PC (Windows, RTX 5090)

> **Pour un lecteur humain.** Ce guide a été écrit pour l'assistant IA (Claude Code) qui a mené
> l'entraînement du chatbot sur le PC le 13/11/2026. C'est pourquoi il tutoie son lecteur et mêle aux
> commandes des règles de consentement (« seulement avec l'accord de l'étudiant »). Vous pouvez le suivre
> tel quel : là où il est écrit « l'étudiant », lisez « vous ». La commande `telecharger` passe par `gh`
> et demande un compte GitHub connecté (`gh auth login`), même pour une release publique. La version
> courte est dans [docs/chatbot/FICHE_MODELE.md](docs/chatbot/FICHE_MODELE.md), § 8 ; le récit daté de
> ce qui a été fait est dans [docs/chatbot/HISTORIQUE.md](docs/chatbot/HISTORIQUE.md).

Ce fichier est suivi par git : il est dans chaque clone du dépôt. Il suffit à lui seul (les
fichiers de mémoire du Mac, AGENTS.md et CLAUDE.md, ne sont pas dans le dépôt).

## 1. Contexte (10 lignes)

1. Projet étudiant (piscine « Feed the Beast », ECE) : deux petits chatbots de démonstration, HORS concours.
2. Chatbot A « moliere » : répond à la manière de Molière (format « NOM. / réplique »).
3. Chatbot B « francais » : répond en français courant (format « — réplique »).
4. Les deux partent d'un modèle pré-entraîné DE ZÉRO sur ce PC (aucun poids externe), puis affiné.
5. Le code vient de git ; les gros fichiers (tokens déjà préparés, dialogues : 4,34 Go, refusés par git) viennent des pièces jointes d'une release GitHub, liste et sha256 dans `chatbot/donnees_rtx.json`.
6. Un seul lanceur fait tout : `python -m chatbot.pipeline_rtx` (valeurs dans `chatbot/recettes_rtx.json` par défaut ; une autre recette se choisit avec l'option `--recettes`, placée AVANT la sous-commande : voir « Autre recette » plus bas).
7. Tailles : S (≈ 29,6 M de paramètres) ou M (54 749 440 paramètres) ; « auto » choisit la plus grande qui tient dans le budget d'heures.
8. Repère mesuré le 13/11/2026 sur une RTX 5090 (taille M, recette par défaut) : base 2 000 027 648 tokens en 5 988,8 s, affinage Molière 297,0 s, affinage français 508,6 s (détail dans « Repère mesuré » plus bas). Sur une autre carte ou avec une autre recette, toute durée vient de la commande `mesurer`.
9. Sorties dans `runs/chatbot_rtx/` (ignoré par git) ; archive finale `runs/chatbot_rtx/RESULTATS_CHATBOT_RTX.zip`.
10. Retour vers le Mac : l'archive part en pièce jointe d'une release (`publier`), les petits rapports dans `experiments/chatbot_rtx/` (à commiter) — seulement avec l'accord de l'étudiant.

L'étudiant te donne dans son message : le dépôt et l'étiquette (tag) de la release des données
(DEPOT = COMPTE/DEPOT, TAG = ETIQUETTE). Si `chatbot/donnees_rtx.json` les contient déjà (pas
« À RENSEIGNER »), `--depot` et `--tag` sont facultatifs.

**Repère mesuré** (lancement du 13/11/2026, RTX 5090, taille M, recette `chatbot/recettes_rtx.json`,
7 étapes finies sans reprise ; rapports dans `experiments/chatbot_rtx/20261113_1956_M/`) :

| Étape | Tokens vus | Calcul | Étape entière | Micro-lot | Mémoire max de la carte |
|-------|-----------:|-------:|--------------:|----------:|------------------------:|
| base | 2 000 027 648 | 5 988,8 s | 6 001,1 s | 128 | 23,67 Go |
| affinage Molière | 90 013 696 | 297,0 s | 300,1 s | 64 | 12,23 Go |
| affinage français | 150 011 904 | 508,6 s | 511,6 s | 64 | 12,23 Go |

- Calcul en bf16, sans compilation (`torch.compile` a échoué, repli automatique).
- Débit de la base : 333 977 tokens/s à la dernière mesure du journal (entre 322 465 et 338 123 selon les mesures).
- Exports : 2,8 s chacun. Évaluations : 127,3 s (Molière) et 106,8 s (français).
- Lancement entier : 7 052,5 s, soit 1 h 57 min (de 19:56:34 à 21:54:07).
- Avant ce lancement, `mesurer` avait donné pour M 340 354 tokens/s (30 pas, 23,68 Go) et une durée estimée de 2,6 heures ; pour S, 545 588 tokens/s (17,83 Go) et 1,52 heure. La taille S n'a pas été entraînée en entier sur cette carte : sa durée réelle reste « à mesurer ».

**Autre recette** : l'option `--recettes` s'écrit avant la sous-commande, par exemple
`& $py -m chatbot.pipeline_rtx --recettes chatbot\recettes_rtx_v5_s03.json lancer --taille M`.
`chatbot/recettes_rtx_v5_s03.json` est la recette par défaut avec, en plus, des conversations de
synthèse dans l'affinage Molière. Elle lit un fichier de plus,
`data/stages/chatbot_20261114/synthetique_v4/entrainement/moliere.jsonl.zst`, qui n'est pas dans les
15 fichiers de la release `chatbot-donnees-20261113`.
Les conversations de synthèse sont dans la pièce jointe `synthese_moliere.zip` de la release
`chatbot-moliere-20261116` : à décompresser à la racine du clone, les fichiers se rangent aux chemins que
lisent les recettes. La recette du modèle publié (S03) est `chatbot/recettes_rtx_publie.json` : c'est
`chatbot/recettes_rtx_v5_s03.json` avec en plus la déclaration du fichier de synthèse et du décodage écrit à
l'export.
Sans le fichier de synthèse (`moliere.jsonl.zst`), garder la recette par défaut.
Attention : une étape finie n'est jamais refaite, même si la recette change. Dans un dossier de
sortie qui contient déjà un lancement, la nouvelle recette ne sert donc qu'avec `--seulement moliere`
(ce chatbot est refait, la base est gardée) ou avec une autre `--sortie`.

## 2. Règles

1. **Un seul entraînement à la fois** sur la carte. Le lanceur a un verrou et refuse de démarrer si une étape tourne encore ; ne lance aucun autre programme GPU pendant un lancement.
2. **Ne modifie pas le code ni les recettes sans le dire à l'étudiant AVANT** (quoi, pourquoi). Ne modifie JAMAIS `ftb/`, `config/`, `corpus/`, `data/validation.jsonl`, `tests/test_kit.py` (kit officiel du concours).
3. **Aucun commit, aucun push, aucun envoi** (`publier --publier`, `gh release ...`) sans l'accord explicite de l'étudiant, demandé à chaque fois. `publier` sans `--publier` est à blanc : il n'envoie rien.
4. **Chiffres réels** : rapporte débits, durées, BPB, tailles tels qu'ils sortent, jamais arrondis à la hausse ni embellis. Une durée non mesurée s'écrit « à mesurer ».
5. **Explique chaque étape simplement** : d'abord la prochaine action, puis sa durée (mesurée, ou « à mesurer »), puis le résultat en une ou deux phrases.
6. Ne tape jamais de mot de passe ni de jeton : `gh auth login`, l'installation du pilote NVIDIA et le réglage de la mise en veille sont faits par l'étudiant lui-même.
7. N'efface jamais `runs/chatbot_rtx/` (états de reprise, journaux) et n'utilise `--repartir` qu'avec l'accord de l'étudiant.
8. N'utilise ni `uv sync` ni `uv run` : ils liraient `uv.lock` et installeraient un PyTorch SANS CUDA. On utilise toujours `.venv\Scripts\python.exe`.

## 3. Ordre exact des commandes (PowerShell, depuis la racine du clone)

Dans l'outil Bash de Claude Code (Git Bash), les mêmes commandes s'écrivent avec `/` :
`.venv/Scripts/python.exe -m chatbot.pipeline_rtx ...` et `PYTHONUTF8=1` devant.

| # | Commande | Doit afficher | Durée |
|---|----------|---------------|-------|
| 1 | `nvidia-smi` | « NVIDIA GeForce RTX 5090 », environ 32 Go, une version de pilote | quelques secondes |
| 2 | `uv --version` ; s'il manque : `winget install --id astral-sh.uv -e`, puis ouvrir un NOUVEAU terminal | un numéro de version | à mesurer |
| 3 | `uv venv --python 3.12 .venv` | `.venv` créé (uv télécharge Python 3.12 s'il manque) | à mesurer |
| 4 | `uv pip install --python .venv\Scripts\python.exe torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132` | torch 2.15.0+cu132 installé (plusieurs Go) | à mesurer |
| 5 | `uv pip install --python .venv\Scripts\python.exe numpy==2.5.4 tokenizers==0.23.3 regex==2026.9.29 zstandard==0.25.0 xxhash==4.0.2` | 5 modules installés | à mesurer |
| 6 | `$env:PYTHONUTF8 = "1"` puis `$py = ".venv\Scripts\python.exe"` | rien (variables de cette fenêtre) | immédiat |
| 7 | `gh auth status` ; si gh manque : `winget install --id GitHub.cli`, nouveau terminal ; si non connecté : l'étudiant lance `gh auth login` | « Logged in to github.com » | à mesurer |
| 8 | `& $py -m chatbot.pipeline_rtx telecharger --depot DEPOT --tag TAG` | « TÉLÉCHARGEMENT RÉUSSI » (15 fichiers, 4,34 Go, chaque sha256 vérifié) | à mesurer (connexion) |
| 9 | `& $py -m chatbot.pipeline_rtx verifier` | « VÉRIFICATION RÉUSSIE » (commit, données, versions, carte, disque ≥ 15 Go) | à mesurer |
| 10 | `& $py -m chatbot.pipeline_rtx autotest` | « AUTOTEST RÉUSSI » (chaîne entière en miniature, coupure simulée et reprise) | 93,5 s mesurées sur la RTX 5090 le 13/11/2026 (prévu < 5 min) ; 22 à 30 s sur le processeur du Mac |
| 11 | `& $py -m chatbot.pipeline_rtx mesurer` | débit (tokens/s) et durée estimée de S et M, dans `runs\chatbot_rtx\mesures.json` | à mesurer |
| 12 | Montrer à l'étudiant les chiffres de `mesurer` SANS les arrondir, et lui demander son budget d'heures (H). Lui demander aussi de désactiver lui-même la mise en veille du PC pendant le calcul. | — | — |
| 13 | Lancer en arrière-plan (voir ci-dessous) : `lancer --taille auto --budget-heures H` | « Plan : base -> ... » puis les étapes | durée donnée par `mesurer` |
| 14 | Suivre : `& $py -m chatbot.pipeline_rtx etat` (toutes les 15 à 30 min) et prévenir l'étudiant à chaque étape finie | statut et minutes de chaque étape | immédiat |
| 15 | `& $py -m chatbot.pipeline_rtx resultats` | `runs\chatbot_rtx\RESULTATS_CHATBOT_RTX.zip`, sa taille et son sha256 | à mesurer |
| 16 | `& $py -m chatbot.pipeline_rtx publier --depot DEPOT_RESULTATS --tag TAG_RESULTATS` (À BLANC) | le plan : visibilité du dépôt, pièces jointes, tailles, commandes gh | quelques secondes |
| 17 | Seulement si l'étudiant dit oui : la même commande avec `--publier` | « PUBLICATION RÉUSSIE » et la commande d'import pour le Mac | à mesurer (connexion) |
| 18 | Seulement si l'étudiant dit oui : commiter `experiments/chatbot_rtx/<nom>/` (petits rapports, aucun poids) et pousser | — | — |

**Lancement en arrière-plan (étape 13)**, au choix :
- outil Bash de Claude Code avec `run_in_background` :
  `PYTHONUTF8=1 .venv/Scripts/python.exe -m chatbot.pipeline_rtx lancer --taille auto --budget-heures H > runs/chatbot_rtx/lanceur.log 2>&1`
- PowerShell :
  ```powershell
  New-Item -ItemType Directory -Force runs\chatbot_rtx | Out-Null
  $p = Start-Process -FilePath $py -ArgumentList "-m","chatbot.pipeline_rtx","lancer","--taille","auto","--budget-heures","H" -RedirectStandardOutput runs\chatbot_rtx\lanceur.log -RedirectStandardError runs\chatbot_rtx\lanceur.err.log -NoNewWindow -PassThru
  $p.Id   # noter ce numéro : c'est le PID du lanceur
  ```

**Reprise après une coupure** (veille, redémarrage, Ctrl-C, lanceur arrêté) : relancer EXACTEMENT
la même commande. Une étape finie n'est jamais refaite ; une étape coupée reprend à son dernier
point de sauvegarde (state.pt, toutes les 1 000 pas pour la base).

## 4. Que faire si une étape échoue

| Étape | Symptôme | Que faire |
|-------|----------|-----------|
| 1 `nvidia-smi` | introuvable ou erreur | Pilote NVIDIA absent : l'étudiant l'installe lui-même (site NVIDIA ou GeForce Experience), redémarre, puis on recommence. Ne rien installer d'autre. |
| 2-5 installation | erreur réseau, version introuvable | Montrer le message. Ne pas changer de version sans l'accord de l'étudiant. Ne jamais utiliser `uv sync` / `uv run`. |
| 7 `gh auth status` | « not logged in » | L'étudiant lance `gh auth login` lui-même. |
| 8 `telecharger` | code 2 « dépôt ou étiquette non renseignés » | Demander DEPOT et TAG à l'étudiant. |
| 8 `telecharger` | « release ... introuvable », « no assets match » | La release n'existe pas ou ce compte gh n'y a pas accès (dépôt privé) : le dire à l'étudiant. |
| 8 `telecharger` | « taille ou sha256 différents » | Le fichier est refusé et n'est pas mis en place. Relancer `telecharger` une fois ; si cela recommence, la release ne correspond pas à `chatbot/donnees_rtx.json` (mauvaise étiquette ou fichiers changés sur le Mac) : arrêter et le dire. |
| 8 `telecharger` | « espace disque insuffisant » | Libérer de la place (c'est l'étudiant qui choisit quoi effacer). |
| 9 `verifier` | « VÉRIFICATION ÉCHOUÉE » | Ne rien lancer. Lire les lignes ERREUR : fichier manquant ou modifié → `telecharger` ; module absent → étape 5 ; « aucune carte NVIDIA » → PyTorch installé sans CUDA (refaire l'étape 4) ou pilote. Une ligne ATTENTION (autre version) n'est pas bloquante : la signaler. |
| 9 `verifier` | « le tokenizer ne correspond pas à celui des tokens de pré-entraînement » | Presque toujours les fins de ligne : Git for Windows a converti `submissions/2026-11-11_153839/tokenizer.json` en CRLF au clone. Le dépôt contient un `.gitattributes` (`* -text`) qui l'empêche ; si le clone est plus ancien, recloner avec `git clone -c core.autocrlf=false ...`. Ne jamais « réparer » le fichier à la main. |
| 10 `autotest` | « AUTOTEST ÉCHOUÉ » | Lire le contrôle en ÉCHEC et son journal (`runs\chatbot_rtx\autotest\journaux\`). Ne pas lancer l'entraînement ; expliquer à l'étudiant. |
| 11 `mesurer` | « aucun micro-lot n'a fonctionné » | Mémoire insuffisante même en lots de 32 : montrer les journaux `runs\chatbot_rtx\journaux\mesure_*.log`, vérifier avec `nvidia-smi` qu'aucun autre programme n'occupe la carte. |
| 13 `lancer` | code 3 « interrompu » | État sauvé : relancer la même commande. |
| 13 `lancer` | « processus arrêté de l'extérieur » | Relancer la même commande (reprise au dernier state.pt). **Pas de `--repartir`.** |
| 13 `lancer` | « mémoire de la carte insuffisante » (débordement mémoire, « out of memory ») | Relancer la même commande avec le micro-lot conseillé dans le message, par exemple `lancer --taille auto --budget-heures H --micro-lot 64` (puis 32 si cela recommence). Même calcul, moins de mémoire, reprise au dernier state.pt, étapes finies gardées. |
| 13 `lancer` | « divergence (loss non finie) » | Montrer le journal. Proposer à l'étudiant de baisser le « lr » de cette étape dans `chatbot/recettes_rtx.json` (modification de code : son accord d'abord), puis `lancer --taille <même taille> --seulement <chatbot> --repartir` ; pour la base : un nouveau dossier de sortie, `& $py -m chatbot.pipeline_rtx --sortie runs\chatbot_rtx_2 lancer --taille <même taille>`, puis la MÊME `--sortie` pour `etat`, `resultats` et `publier`. |
| 13 `lancer` | « erreur du programme » | Lire les dernières lignes et le journal `runs\chatbot_rtx\journaux\<étape>.log`, expliquer à l'étudiant avant toute relance. |
| 13 `lancer` | « une étape tourne encore sans son lanceur (PID N) » | Le lanceur a été tué seul et l'étape calcule encore : attendre, ou l'arrêter avec `taskkill /T /F /PID N`, vérifier avec `nvidia-smi` qu'aucun `python` ne reste sur la carte, puis relancer la même commande. |
| 13 `lancer` | « un autre lanceur tourne déjà » | Un lancement est en cours : le suivre avec `etat`, ne pas en démarrer un second. |
| 13 `lancer` | code 2 « contient déjà un lancement en taille X » | Continuer avec `--taille X` (ou `auto`), ou, si l'étudiant veut repartir de zéro, un autre dossier : `& $py -m chatbot.pipeline_rtx --sortie runs\chatbot_rtx_2 lancer --taille ...`, puis la MÊME `--sortie` pour `etat`, `resultats` et `publier` (elle s'écrit avant ou après la sous-commande). |
| 15 `resultats` | « lancement pas fini » | Finir `lancer`, ou `resultats --partiel` si l'étudiant veut emporter ce qui existe. |
| 16-17 `publier` | « dépôt ou étiquette non renseignés », « gh n'est pas connecté », taille différente après envoi | Demander DEPOT/TAG ; `gh auth login` par l'étudiant ; relancer la même commande (reprise : ce qui est déjà en ligne n'est pas renvoyé). |
| 16-17 `publier` | « dépôt … est VIDE (aucun commit) » | GitHub refuse une release dans un dépôt sans commit. L'étudiant y met un premier commit (par exemple un README depuis la page du dépôt), puis relancer la même commande. |

**Arrêter exprès un lancement** : dans la console du lanceur, Ctrl-C ou Ctrl-Break (l'étape finit
son pas, sauve son état, le lanceur rend le code 3). Lancé en arrière-plan : `taskkill /T /F /PID
<PID du lanceur>` arrête le lanceur ET son étape (sans `/T`, l'étape continuerait seule) ; on
perd au plus le travail depuis le dernier point de sauvegarde, et la même commande reprend.

**Commandes lancées à la main** (`python -m chatbot.generate chat ...`, `python -m chatbot.evaluate
...`) : elles passent déjà leur sortie en UTF-8 ; garder quand même `PYTHONUTF8=1`.

**Parler au modèle Molière avec le décodage retenu** (choix ancré sur le dernier message,
`chatbot/ancrage.py`) : il faut deux options, `--anchor-mu 1` et `--candidates 32`.

```powershell
& $py -m chatbot.generate chat --model runs\chatbot_rtx\modeles\chatbot_moliere --anchor-mu 1 --candidates 32 "Bonjour, comment allez-vous ?"
```

Sans ces deux options, la commande applique le `chat_config.json` du dossier du modèle. Celui que
ce lanceur a exporté le 13/11/2026 demande 16 candidates et n'a pas la clé `anchor_mu` ; un export
refait avec le code actuel et la recette par défaut écrit 16 candidates et `"anchor_mu": 0.0`. Dans
les deux cas c'est l'ancien choix, sans ancrage. Le modèle publié (S03, pièce jointe `chatbot_moliere_s03.zip` de la release `chatbot-moliere-20261116`) fait exception : son `chat_config.json` porte déjà `"candidates": 32` et `"anchor_mu": 1.0`, et les deux options n'y changent rien. Les clés `"anchor_mu"` et `"candidates"` de ce fichier ont le même
effet que les deux options ; ne modifie pas ce fichier sans le dire à l'étudiant avant (règle 2).
`--anchor-mu 0` rend l'ancien choix. Sans message, le chat est interactif ; `--device cuda` fait
répondre la carte (défaut : le processeur) et suit la règle 1 : jamais pendant un lancement.

## 5. Ce qu'il faut rapporter à la fin (à l'étudiant, chiffres réels)

1. Le commit du clone (`git log -1 --oneline`) et `runs\chatbot_rtx\verification.json` : versions, carte, mémoire.
2. Les débits mesurés (`runs\chatbot_rtx\mesures.json`) : tokens/s, micro-lot retenu, mémoire GPU max, durées estimées de S et M.
3. La taille choisie et le budget d'heures donné.
4. Les durées RÉELLES de chaque étape (`etat`, ou `runs\chatbot_rtx\bilan.json` → `durees_minutes`) et la durée totale.
5. Les BPB : base (BPB Molière au pas 0 → meilleur), chaque chatbot (validation et évaluation `bpb_tour_suivant`), part synthétique s'il y en a une (`runs\chatbot_rtx\bilan.json`).
6. Les tailles et sha256 : `RESULTATS_CHATBOT_RTX.zip`, dossiers `runs\chatbot_rtx\modeles\chatbot_moliere` et `chatbot_francais`.
7. Les chemins : archive, journaux, `experiments/chatbot_rtx/<nom>/` si `publier` a été lancé.
8. Les reprises et incidents (coupures, micro-lot changé, erreurs) avec leur cause.
9. Ce qui a été envoyé ou commité (avec l'accord de l'étudiant) et ce qui ne l'a pas été.
10. Sur le Mac, l'étudiant importe avec :
    `uv run python tools/import_chatbot_results.py --depuis-release DEPOT_RESULTATS TAG_RESULTATS`
    (ou, archive copiée à la main : `uv run python tools/import_chatbot_results.py RESULTATS_CHATBOT_RTX.zip`).
