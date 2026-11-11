# Prochaine action : lancer la reproduction depuis le dossier du projet

```bash
uv run --no-project python tools/reproduce.py --recipe final
```

1. **Préparation — environ 2–5 minutes sur une machine macOS/Linux neuve.** Installer `uv`
   au préalable et placer le dépôt hors d'un dossier synchronisé, ou conserver
   les liens vers les dossiers `.nosync`. La commande vérifie les 18 fichiers
   officiels du kit, refuse de remplacer un dossier ordinaire, crée les liens
   manquants et prépare Python 3.12 dans `.venv.nosync`. Le lancement initial
   utilise seulement Python standard pour ne pas créer une `.venv` ordinaire.
   Le PC Windows natif utilise le guide de transfert RTX 5090 pour l'entraînement.
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
5. **Niveau de preuve actuel.** Les garde-fous et la conservation du kit ont été
   testés ; `--check-only` a été exécuté sur le Mac. Une reconstruction intégrale
   depuis un dépôt neuf et un nouveau téléchargement n'est pas encore attestée.
   L'entraînement et les livrables restent des étapes distinctes. Ne pas lancer
   cette commande en même temps qu'un autre pipeline écrivant les mêmes dossiers.

Les sources Hugging Face peuvent évoluer ou devenir indisponibles. Les fiches de
source, dates gelées, manifests et hashes produits servent à détecter ces écarts ;
un cache local réutilisé n'est pas une preuve de téléchargement neuf identique.

Pour entraîner au budget officiel après préparation du mélange final :

```bash
nohup uv run python -m bench.run --config configs/bench/full_reference_masked_20261110.yaml > data/stages/logs/reproduction-final-100m.log 2>&1 &
```

Le run existant ne doit pas être écrasé. Un nouveau run doit vérifier le SHA du
flux attendu, le budget, le score et son matériel ; le passage Windows/RTX 5090
est décrit séparément dans `PASSAGE_PC_RTX5090.md`.
