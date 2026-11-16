# Reproduire le jeu de données du concours

État au 16/11/2026 : commande à lancer depuis le dossier du projet. Elle a été rejouée
sur le Mac à partir des sources déjà collectées ; la reconstruction intégrale depuis un
dépôt neuf n'est pas attestée (détail au point 5).

```bash
uv run --no-project python tools/reproduce.py --recipe final
```

1. **Préparation — environ 2–5 minutes sur une machine macOS/Linux neuve.** Installer `uv`
   au préalable et placer le dépôt hors d'un dossier synchronisé, ou conserver
   les liens vers les dossiers `.nosync`. La commande vérifie les 18 fichiers
   officiels du kit, refuse de remplacer un dossier ordinaire, crée les liens
   manquants et prépare Python 3.12 dans `.venv.nosync`. Le lancement initial
   utilise seulement Python standard pour ne pas créer une `.venv` ordinaire.
   Sur Windows natif, la commande refuse la reconstruction intégrale (macOS ou Linux
   requis). Le guide de transfert RTX 5090 prévu pour l'entraînement du concours n'a
   jamais été exécuté.
2. **Calcul — prévoir 2–4 heures depuis zéro, à mesurer selon la connexion.**
   La commande lance un processus détaché par `nohup`. Son PID et le chemin de
   son journal `data/stages/logs/reproduction-*.log` sont affichés. Elle collecte
   les sept sources actives, exécute `nettoye`, reconstruit le tokenizer de référence,
   sélectionne le mélange puis masque ses coordonnées. Les hashes du tokenizer,
   du mélange parent et du mélange final sont contrôlés. Aucune publication automatique.
3. **Résultat — contrôle en 1 minute.** Lire
   `data/stages/reproduction/final/status.json`. `complete: true` exige le
   succès de toutes les commandes et un contrôle de contamination à zéro.
   Les données finales sont dans `data/mixes/full_reference_masked_20261110/` ;
   la transformation et le contrôle sont dans
   `data/stages/finalisation_20261110/masked_reference/`. Le corpus `nettoye` est conservé.
4. **Place et limites.** Prévoir au moins **100 Go libres** pour une reconstruction
   intégrale avec les caches du nettoyage de base ; environ 30 Go de budget
   réservé aux données brutes. La variante complète libère ses propres copies
   intermédiaires progressivement. L'option `--skip-collect` suppose des sources
   déjà présentes ; `--recipe reference` s'arrête à la référence nettoyée.
   `--recipe complete` rejoue la variante qualité/MinHash expérimentale, mesurée
   moins bonne. Elle est distincte du mélange final choisi.
5. **Niveau de preuve (état au 11/11/2026).** Les garde-fous et la conservation du
   kit ont été testés ; `--check-only` a été exécuté sur le Mac. D'après
   [EXIGENCES.md](EXIGENCES.md), `--recipe final --skip-collect` a été exécuté sur le
   Mac avec les sources brutes locales et le cache : même hash final. Son fichier de
   statut est sous `data/`, ignoré par git : il n'est pas publié. Une reconstruction
   intégrale depuis un dépôt neuf, avec un nouveau téléchargement, n'est pas attestée.
   L'entraînement et les livrables restent des étapes distinctes. Ne pas lancer
   cette commande en même temps qu'un autre pipeline écrivant les mêmes dossiers.

Les sources Hugging Face peuvent évoluer ou devenir indisponibles. Les fiches de
source, dates gelées, manifests et hashes produits servent à détecter ces écarts ;
un cache local réutilisé n'est pas une preuve de téléchargement neuf identique.

L'étape `exclude_heldout` lit des listes `data/val/*/heldout_ids.txt`. Elles ne sont
pas dans le dépôt et aucun outil suivi par git ne les crée. Sans elles, l'étape ne
retire rien et l'écrit dans son rapport ; l'étape `decontaminate` reste appliquée.
Le dépôt ne dit pas si ces listes existaient sur le Mac lors du run de référence.

Pour entraîner au budget officiel après préparation du mélange final :

```bash
nohup uv run python -m bench.run --config configs/bench/full_reference_masked_20261110.yaml > data/stages/logs/reproduction-final-100m.log 2>&1 &
```

Le run existant ne doit pas être écrasé. Un nouveau run doit vérifier le SHA du
flux attendu, le budget, le score et son matériel. Le passage Windows/RTX 5090 du
concours a été préparé le 10/11/2026 et jamais exécuté ; son guide est gardé comme
pièce historique dans [PASSAGE_PC_RTX5090.md](PASSAGE_PC_RTX5090.md).

Les deux PDF de `output/pdf/` sont produits sur le Mac par
`tools/build_delivery_documents.py` : modules reportlab et pypdf (absents de
`pyproject.toml`), polices Arial de macOS, fichiers sous `data/` et `runs/` non
publiés. Ils ne se refont pas depuis un clone. Ne pas relancer cet outil tel quel :
il réécrit aussi `docs/remise/DATASHEET.md` et `docs/remise/RAPPORT.md` avec son texte
du 10/11/2026, ce qui effacerait les corrections faites depuis dans `DATASHEET.md`.
`tools/prepare_dataset_release.py` assemble le dossier des dix fichiers de la release du jeu (neuf fichiers et
`SHA256SUMS.json`). Les empreintes de la soumission du 11/11/2026 sont dans
`experiments/submission_20261111.json`.

## Chatbot Molière (hors concours)

Cette page ne reproduit que le jeu du concours. Le chatbot « à la manière de Molière »
se refait à part, sur un PC Windows à carte NVIDIA, sans `uv run` ni `uv sync` (ils y
installeraient un PyTorch sans CUDA). Le guide pas à pas est
[PC_RTX.md](../../PC_RTX.md), à la racine du dépôt : `python -m chatbot.pipeline_rtx`
avec `telecharger`, `verifier`, `autotest`, `mesurer` puis `lancer`. Les gros fichiers
d'entraînement de la recette par défaut (`chatbot/recettes_rtx.json`) viennent de la release
GitHub `chatbot-donnees-20261113` (15 fichiers) ; leur liste et leurs sha256 sont dans
`chatbot/donnees_rtx.json`. Les conversations de synthèse du modèle publié et des essais du 16/11 n'y
sont pas (cette release ne contient que deux fichiers d'essai écrits par `qwen3:8b`, qui ne servent pas
au modèle publié) : elles sont dans la pièce jointe `synthese_moliere.zip` de la release `chatbot-moliere-20261116`
(même dépôt). À décompresser à la racine du clone : les fichiers se rangent aux chemins que lisent
les recettes. La recette du modèle publié est `chatbot/recettes_rtx_publie.json`
(à passer avec `--recettes`, placé avant la sous-commande). Un réentraînement ne redonne pas
forcément les mêmes octets : cela n'a pas été essayé.

Le modèle et la façon de le refaire sont décrits dans
[docs/chatbot/FICHE_MODELE.md](../chatbot/FICHE_MODELE.md) (§ 8), ses données dans
[docs/chatbot/FICHE_DONNEES.md](../chatbot/FICHE_DONNEES.md). Les licences sont dans
[LICENCES.md](../../LICENCES.md).
