Prochaine action : sur le PC, cloner le dépôt, ouvrir Claude Code et coller le prompt de la partie B — les données sont déjà en ligne (release `chatbot-donnees-20261113`, envoyée et vérifiée le 13/11).

# Entraîner les deux chatbots sur le PC Windows (RTX 5090) — route « git clone »

**Ce que le clone change, et ce qu'il ne change pas.**
1. Il ne rend PAS l'entraînement plus rapide : c'est la même carte, le même lanceur, les mêmes recettes. La durée vient toujours de la commande `mesurer`.
2. Il change le TRANSPORT : plus de clé USB de 4,35 Go. Le code arrive par `git clone`, les gros fichiers par les pièces jointes d'une release GitHub, les résultats repartent de la même façon. On peut donc tout suivre et tout récupérer de n'importe où.
3. Limite de GitHub : un fichier de plus de 100 Mo ne peut pas être commité. Les tokens (6 fichiers .bin, 4,2 Go, le plus gros 1,2 Go) et les modèles (0,13 à 0,25 Go) passent donc en pièces jointes de release (2 Gio au plus chacune), comme le dataset officiel du 11/11.

**Ce qui n'a jamais tourné** : rien n'a encore été exécuté sur la RTX 5090. Aucun débit ni aucune durée sur cette carte n'est connu. Ce qui est prouvé sur le Mac : la route complète sur un clone simulé (téléchargement par un faux `gh`, vérification, autotest, lancement de la recette d'essai, résultats, publication à blanc).

## A. Sur le Mac (une fois) : envoyer les gros fichiers

1. **Décider du dépôt** (décision de l'étudiant) :
   - dépôt de la piscine (`benassou423/Inteligence-lab-piscine`) : il est **PUBLIC**, tout le monde pourra télécharger les tokens ;
   - ou un dépôt **privé** à part, créé par l'étudiant : `gh repo create COMPTE/ftb-chatbot-donnees --private --add-readme`. Le `--add-readme` est indispensable : il fait un premier commit, et GitHub refuse de créer une release dans un dépôt vide (l'outil le détecte et s'arrête avant tout envoi). `gh release download` marche aussi pour un dépôt privé, avec le même compte `gh` sur le PC.
2. **Envoi à blanc** (rien ne part ; régénère `chatbot/donnees_rtx.json` avec les vrais sha256, 15 fichiers, 4 344 702 490 octets) — 2,9 s mesurées sur le Mac :
   ```bash
   uv run python tools/publish_chatbot_release.py --depot benassou423/Inteligence-lab-piscine --tag chatbot-donnees-20261113
   ```
   Il affiche la visibilité du dépôt, chaque pièce jointe, sa taille, le total et les commandes `gh` exactes.
3. **Envoi réel** (seulement si l'étudiant le décide) — durée à mesurer (4,34 Go à envoyer) :
   ```bash
   uv run python tools/publish_chatbot_release.py --depot benassou423/Inteligence-lab-piscine --tag chatbot-donnees-20261113 --publier
   ```
   Reprise : relancer la même commande ne renvoie pas ce qui est déjà en ligne. À la fin, `chatbot/donnees_rtx.json` contient le dépôt et l'étiquette.
4. **Commit et push** (avec l'accord de l'étudiant) des fichiers listés dans `data/stages/chatbot_20261113/fichiers_a_committer.txt` (partie « INDISPENSABLES » au minimum). Sans eux, le clone du PC n'a pas le lanceur.

## B. Sur le PC : cloner, ouvrir Claude Code, coller le prompt

1. Installer, si besoin (l'étudiant, 10 à 20 min, durée à mesurer) : Git for Windows, GitHub CLI (`winget install --id GitHub.cli`), Claude Code. Puis `gh auth login` (l'étudiant se connecte lui-même).
2. Cloner dans un chemin court (évite les chemins trop longs de Windows), sans conversion des fins de ligne (sinon l'empreinte du tokenizer change et `verifier` refuse) :
   ```powershell
   git clone -c core.autocrlf=false https://github.com/benassou423/Inteligence-lab-piscine.git C:\FTB
   ```
3. Ouvrir Claude Code dans `C:\FTB` et coller **exactement** ce message (remplacer les deux dépôts et étiquettes) :

```text
Lis PC_RTX.md à la racine de ce dépôt, en entier, et suis-le dans l'ordre sans sauter d'étape.
Release des données : --depot benassou423/Inteligence-lab-piscine --tag chatbot-donnees-20261113
Release des résultats (pour `publier`, à la fin) : --depot benassou423/Inteligence-lab-piscine --tag chatbot-resultats-20261114
Ce PC : Windows, RTX 5090 (32 Go). Aucun commit, push ou envoi sans mon accord explicite, demandé à chaque fois.
Avant chaque commande, dis-moi ce qu'elle fait et sa durée (mesurée, ou « à mesurer »).
```

4. **Ce qui va se passer** (détail et cas d'échec dans `PC_RTX.md`) :
   1. installation : Python 3.12 et PyTorch 2.15.0 CUDA 13.2 depuis l'index officiel PyTorch (jamais `uv sync` : il installerait un PyTorch sans CUDA) — à mesurer ;
   2. `telecharger` : 15 fichiers, 4,34 Go, chaque sha256 vérifié ; un fichier déjà bon n'est pas retéléchargé — à mesurer ;
   3. `verifier` : commit, données, versions, carte (vrai calcul de test), disque ≥ 15 Go ;
   4. `autotest` : toute la chaîne en miniature, coupure simulée et reprise — 21,6 à 22,2 s sur le processeur du Mac (clone simulé), prévu < 5 min ;
   5. `mesurer` : débit réel des tailles S et M → durée estimée. **C'est là qu'on apprend combien de temps prendra l'entraînement.** La session te montre les chiffres et te demande ton budget d'heures ;
   6. `lancer` en arrière-plan : base (1,5 milliard de tokens en S, 2 milliards en M) → Molière (90 M) → français (150 M) → exports → évaluations. Une coupure n'est pas grave : la même commande reprend ;
   7. `resultats` puis `publier` à blanc ; l'envoi réel et le commit des petits rapports (`experiments/chatbot_rtx/<nom>/`) attendent ton accord.
5. Pendant le calcul : désactiver toi-même la mise en veille du PC (Paramètres Windows), sinon le calcul s'arrête (il reprend quand on relance).

## C. Récupérer les résultats sur le Mac (de n'importe où)

1. Archive depuis la release (vérifie chaque morceau et l'archive entière, range tout, fait répondre chaque chatbot à 3 messages sur le processeur) — 1 à 2 min plus le téléchargement :
   ```bash
   uv run python tools/import_chatbot_results.py --depuis-release benassou423/Inteligence-lab-piscine chatbot-resultats-20261114
   ```
2. Pour mettre les nouveaux modèles dans la démo (l'ancien est gardé à côté, daté) : ajouter `--installer --nom rtx_installe`, puis redémarrer la démo toi-même.
3. Petits rapports suivis par git (si commités depuis le PC) : `git pull`, ils sont dans `experiments/chatbot_rtx/<nom>/`.

## D. Solution de secours : le paquet zip (sans GitHub)

Si GitHub n'est pas possible (pas de réseau, pas d'accord pour une release) :
1. Fichier : `data/stages/chatbot_20261113/paquet_rtx/FTB_CHATBOT_RTX_20261113.zip` ; taille et sha256 dans `FTB_CHATBOT_RTX_20261113.verification.json` à côté.
2. Copier sur une clé ou un disque en **exFAT ou NTFS** (FAT32 refuse les fichiers de plus de 4 Go). Sur le PC, vérifier : `Get-FileHash .\FTB_CHATBOT_RTX_20261113.zip -Algorithm SHA256` (doit être égal au sha256 du fichier de vérification).
3. Extraire : `tar -xf .\FTB_CHATBOT_RTX_20261113.zip -C C:\FTB-chatbot`, ouvrir Claude Code dans `C:\FTB-chatbot\FTB_CHATBOT_RTX_20261113` et coller :

```text
Lis CLAUDE.md puis PC_RTX.md, en entier, et suis-les dans l'ordre. Tu es dans le paquet zip : pas d'étape `telecharger`.
Ce PC : Windows, RTX 5090 (32 Go). Aucun commit, push ou envoi sans mon accord explicite, demandé à chaque fois.
Avant chaque commande, dis-moi ce qu'elle fait et sa durée (mesurée, ou « à mesurer »).
```
4. Retour : copier `RESULTATS_CHATBOT_RTX.zip` sur le Mac, puis `uv run python tools/import_chatbot_results.py RESULTATS_CHATBOT_RTX.zip`.

## E. Plus tard : ré-entraîner un seul chatbot

1. Si les dialogues changent sur le Mac : relancer l'envoi (partie A, étapes 2 et 3, avec une NOUVELLE étiquette), commiter `chatbot/donnees_rtx.json`, puis sur le PC `git pull` et `telecharger` (seuls les fichiers changés sont retéléchargés).
2. Sur le PC : `lancer --taille <même taille> --seulement francais` (ou `moliere`) refait ce chatbot **sans refaire la base** ; l'ancien est rangé dans `runs\chatbot_rtx\archives\`.
