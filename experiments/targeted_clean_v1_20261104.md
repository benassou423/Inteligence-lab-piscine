Prochaine action : conserver le corpus actuel et utiliser ce résultat comme ablation — lecture 4 min.

# Nettoyage ciblé v1 — 4 novembre 2026

**Décision : conserver le corpus actuel. La variante est moins bonne sur les deux tirages et dans les quatre genres.** Le corpus actif `nettoye` reste la référence. Les sorties expérimentales sont séparées dans `data/stages/nettoye_cible_v1/03_exact_dedup/`.

## 1. Changement évalué

Cette expérience ajoute un nettoyage structurel au corpus final actuel, déjà filtré avec `fr >= 0,65`. Elle ne récupère aucun document rejeté par ce filtre. Le code retire des tableaux bruts suffisamment reconnaissables sur Wikipédia/Wikinews, des cadres `Infobox`, des identifiants Wikidata isolés, des marqueurs de mise en page et certaines longues répétitions.

Les textes littéraires imposent une limite : leurs tableaux peuvent contenir de la prose ou des traductions. Leur suppression est désactivée sur Wikisource. Les cellules de tableau ambiguës restent également intactes. Un `|` isolé devant une biographie est retiré seul ; les formules mathématiques et les opérateurs logiques sont protégés. Ces protections viennent de l’inspection des données d’entraînement, sans utiliser les textes de validation pour choisir les règles. Aucune liste manuelle d’IDs n’est appliquée au corpus.

L’étape garde la provenance des anciens scores de langue lorsqu’elle change le texte : ils ne sont pas présentés comme des scores recalculés après nettoyage. La déduplication exacte est rejouée, puis le corpus complet passe le contrôle de contamination officiel équivalent.

## 2. Effet mesuré sur les données

| Mesure | Corpus actuel | Variante finale |
|---|---:|---:|
| Documents | 3 119 274 | 3 119 273 |
| Octets UTF-8 de texte | 17 352 211 188 | 17 340 524 573 |
| Documents modifiés par le nettoyage | — | 36 475 |
| Nouveaux doublons exacts retirés | — | 0 |
| Documents de validation signalés pour contamination | 0 / 230 | 0 / 230 |

Le nettoyage retire **11 686 615 octets**, soit **11,69 Mo / environ 0,067 %** du texte. Le seul document retiré est une table brute de racines grecques (`wikipedia_fr:14384878`, 77 677 octets), qui ne laisse plus de texte après application des règles.

| Source | Documents modifiés | Octets de texte retirés |
|---|---:|---:|
| Wikipédia | 36 243 | 11 648 642 |
| Wikinews | 81 | 26 528 |
| Wikisource | 151 | 11 445 |
| Gutenberg, Assemblée, FineWeb, Common Crawl | 0 | 0 |

Les **535 818 documents** des quatre dernières sources ont été comparés intégralement : textes, IDs, dates, licences et autres champs identiques, sauf le compteur de copies exactes recalculé sur 126 documents. Ce changement de métadonnée est attendu après une nouvelle déduplication ; il ne change pas le texte d’apprentissage.

## 3. Comparaison d’apprentissage

Deux graines de mélange, **1337 et 2027**, fixent le tirage et l’ordre des documents. L’initialisation du modèle reste fixée à 1337 par le kit. Les quatre essais utilisent le même tokenizer, c’est-à-dire le même outil de découpage du texte, provenant de la référence 1337. Son empreinte SHA-256 est `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`.

Les essais utilisent 100 Mo de texte tiré, le même modèle de 10 166 400 paramètres, MPS en float32, un batch effectif de 64 et des lots de 16. Budget nominal de 25 M tokens, soit **25 001 984 tokens réellement vus** après arrondi au nombre entier de pas. Le flux doit être assez long pour éviter de relire son début. La référence 1337 déjà exécutée est réutilisée ; trois entraînements nouveaux complètent les deux paires.

Le critère pratique est fixé avant les essais dans `comparison_spec.json` : amélioration du BPB global sur les deux tirages, amélioration moyenne d’au moins 0,01 BPB et aucune régression moyenne par genre supérieure à 0,02. Ce seuil de décision n’est pas un test de significativité statistique. **BPB signifie bits par octet ; plus bas est meilleur.**

| Graine de mélange | Référence | Variante | Différence variante − référence |
|---|---:|---:|---:|
| 1337 | 1,615490 | 1,630004 | +0,014514 |
| 2027 | 1,629042 | 1,649257 | +0,020216 |
| **Moyenne** | **1,622266** | **1,639631** | **+0,017365** |

| Genre — moyenne des deux tirages | Référence | Variante | Différence |
|---|---:|---:|---:|
| Dialogue | 1,381516 | 1,394490 | +0,012973 |
| Encyclopédique | 1,776886 | 1,793648 | +0,016762 |
| Littéraire | 1,726435 | 1,736942 | +0,010507 |
| Presse | 1,609074 | 1,638543 | +0,029469 |

La moyenne augmente de **0,017365 BPB** (environ **1,07 %**, moins bon). Aucun critère d’adoption n’est satisfait : les deux différences globales sont défavorables, et la presse régresse de 0,029469 BPB en moyenne. La recette n’est donc pas promue. Le code, le corpus expérimental et les résultats sont conservés pour l’ablation ; ce constat à 25 M tokens ne prouve pas qu’aucun autre nettoyage ne pourrait aider.

`tools/compare_targeted_results.py` a vérifié sur les quatre résultats : même budget réellement vu, même tokenizer, modèle, configuration d’entraînement, appareil, précision et versions logicielles. Aucun flux n’a dû être relu. Valeurs complètes : `experiments/targeted_clean_v1_20261104.json`.

Durées des passes finales : traitement **5 min 24 s**, contamination **6 min 14 s**, nouveaux essais **12 min 12 s**, **9 min 38 s** et **9 min 37 s**. La référence 1337 déjà disponible évite un quatrième nouvel entraînement.

## 4. Contrôles et limites

**306 tests verts en 195,80 s**, avec 44 avertissements existants liés au lancement de processus Python. Le test de reprise a été exécuté en lots de 16 via une nouvelle option relayée par `bench/kaggle.py` : même budget, mêmes assertions, accumulation prévue par le kit. Aucun fichier du kit n’est modifié.

L’essai porte sur 2 000 documents frais et les 888 cas du précédent audit. Sur les 2 000 : 18 modifiés, aucun retiré, 12 témoins positifs identiques. Les différences des 18 documents ont été relues par IA. Une inspection supplémentaire d’extraits de 11 documents après traitement a permis de protéger les formules ; cinq formules du document réel concerné sont conservées à l’identique. Ces relectures ciblées ne constituent pas une mesure représentative de taux d’erreur ni une validation humaine exhaustive.

Le contrôle complet de contamination de la variante a pris 374,3 s : **0 document signalé**, recouvrement maximal **8,4 %**, sous le seuil officiel de 10 %. Cela ne signifie pas zéro phrase commune. Les sept liens `.nosync`, les fichiers protégés et le corpus de référence sont conservés. Les fichiers de la variante sont également identiques à leur état au moment du contrôle de contamination. Preuve finale : `experiments/targeted_clean_v1_20261104_verification.json`.

Ce nettoyage ne règle pas les éditions bilingues, les catalogues de noms, tous les sommaires ou les tableaux littéraires. Les essais locaux utilisent un quart du budget officiel ; une recette retenue doit être confirmée avec le protocole complet sur T4. Les durées ne servent pas à mesurer un gain de vitesse, car la charge du Mac varie entre essais.

## 5. Reproduction et preuves

À lancer depuis la racine du projet, en attendant la fin de chaque étape avant la suivante ; détacher chaque tâche longue avec `nohup` et garder son journal sous `data/stages/logs/`. Exemple de lancement long : `nohup uv run python -m pipeline.run --config configs/pipeline/nettoye_cible_v1.yaml > data/stages/logs/targeted-clean-pipeline.log 2>&1 &`. Commandes des étapes :

```bash
uv run python -m tools.targeted_clean_pilot
uv run pytest -q
uv run python -m pipeline.run --config configs/pipeline/nettoye_cible_v1.yaml --workers 8
uv run python -m tools.check_contamination --config nettoye_cible_v1 --fast
uv run python -m bench.run --config configs/bench/proxy_cible_base_s2027.yaml
uv run python -m bench.run --config configs/bench/proxy_cible_v1_s1337.yaml
uv run python -m bench.run --config configs/bench/proxy_cible_v1_s2027.yaml
uv run python -m tools.compare_targeted_results
```

Code : `pipeline/clean/targeted.py`, enregistrement de l’étape dans `pipeline/run.py`, tests `tests/test_clean_targeted.py`. Protocole et preuves : `data/stages/targeted_clean_20261104_audit/` ; résultats d’entraînement : `runs/proxy_cible_*/` et `experiments/results.csv`. Les grosses copies intermédiaires de cette expérience sont supprimées après leur dernière utilisation ; statistiques, reçus, échantillons et corpus final sont conservés. Le pipeline les reconstruit si nécessaire. Aucun nouveau commit ni push.
