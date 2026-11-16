# Index des outils (`tools/`)

Une ligne par script : à quoi il sert, qui l'appelle, et ce qu'il vaut aujourd'hui.
Les scripts sont rangés par usage : d'abord le concours, puis le chatbot Molière (hors concours).

Statuts :

- **utile** : sert encore ; se lance depuis un clone, une fois ses données d'entrée présentes ;
- **usage unique, résultat commité** : a tourné une fois ; son résultat est dans le dépôt ; le relancer n'apporte rien ou écraserait une pièce ;
- **Mac seulement** : ne tourne que sur le Mac du projet, où sont ses données d'entrée (`data/raw`, `data/stages`, `data/mixes` ne sont pas dans git) ;
- **route jamais exécutée** : préparé pour entraîner le modèle du concours sur le PC à carte RTX 5090 ; ce contrôle, facultatif, n'a pas été fait ([`docs/remise/EXIGENCES.md`](../docs/remise/EXIGENCES.md)) ;
- **essai négatif** : outil d'un essai dont le résultat n'a pas été retenu ; gardé pour que l'essai puisse être refait.

Dans la colonne « Qui l'appelle », « aucun script ni test » veut dire que l'outil se lance seulement à la main.
Sur le Mac et sous Linux, les commandes se lancent avec `uv run`. Sur le PC Windows à carte NVIDIA, jamais :
voir [`PC_RTX.md`](../PC_RTX.md).

## Concours : collecte

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `archive_wikipedia_hf2023.py` | Range l'ancienne Wikipédia (jeu Hugging Face de 2023) sous la source d'archive `wikipedia_fr_hf2023` : seuls la source et l'identifiant changent, chaque document est vérifié. | [`docs/sources/wikipedia_fr_hf2023.md`](../docs/sources/wikipedia_fr_hf2023.md), `tests/test_collect_wikipedia.py` | Mac seulement |
| `check_wikipedia_quality.py` | Mesure la collecte de Wikipédia et prépare 20 articles à relire. Il n'approuve rien tout seul. | [`docs/sources/wikipedia_fr.md`](../docs/sources/wikipedia_fr.md), `tests/test_collect_wikipedia.py` | utile |

## Concours : nettoyage et contrôles

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `check_contamination.py` | Vérifie une sortie du pipeline avec le contrôle officiel du kit (`ftb.contamination`) : aucun document de validation ne doit être signalé. | `reproduce.py`, `prepare_raw_ablation.py`, `run_complete_pipeline.py` ; [`docs/comprendre/decontamination.md`](../docs/comprendre/decontamination.md) | utile |
| `annotate_langid.py` | Annotation manuelle du filtre de langue, à l'aveugle, sur un tirage stratifié. | [`docs/comprendre/langue.md`](../docs/comprendre/langue.md), `tests/test_clean_annotate.py` ; importé par `prepare_human_langid.py` et `finalize_langid_ai_80.py` | utile |
| `langid_report.py` | Précision et rappel du filtre de langue à partir d'annotations. Il écrit `langid_report.md` dans le dossier donné : sur `data/annotations/controle_langue_80`, il remplace le résumé publié (voir le README de ce dossier). | [`docs/comprendre/langue.md`](../docs/comprendre/langue.md), [`data/annotations/controle_langue_80/README.md`](../data/annotations/controle_langue_80/README.md), `tests/test_clean_annotate.py` ; importé par `finalize_langid_ai_80.py` | utile |
| `prepare_human_langid.py` | Tire 80 extraits à l'aveugle sur les sept sources, sans créer d'étiquette. L'annotation humaine prévue n'a pas eu lieu : les 80 extraits ont été classés par une IA (voir `finalize_langid_ai_80.py`). | Aucun script ni test | Mac seulement |
| `finalize_langid_ai_80.py` | Joint les 80 décisions prises par une IA aux scores, et calcule des bornes en faisant varier les 17 cas limites. Il travaille dans un dossier local non publié ; la copie publiée, sans les extraits, est `data/annotations/controle_langue_80/`. | `tests/test_langid_sensitivity.py` | usage unique, résultat commité |
| `audit_langid_full.py` | Audit en lecture seule du filtre de langue sur les textes complets, avec deux détecteurs. | [`docs/comprendre/audit_langue_complete.md`](../docs/comprendre/audit_langue_complete.md), [`experiments/langid_full_audit_20261104.md`](../experiments/langid_full_audit_20261104.md), `tests/test_clean_langid_full_audit.py` | usage unique, résultat commité |
| `targeted_clean_pilot.py` | Essai du nettoyage ciblé sur 2 000 documents et sur les cas de l'audit de langue. | [`experiments/targeted_clean_v1_20261104.md`](../experiments/targeted_clean_v1_20261104.md) | essai négatif |
| `compare_targeted_results.py` | Compare les quatre runs appariés du nettoyage ciblé et écrit `experiments/targeted_clean_v1_20261104.json`. Résultat : +0,017365 BPB en moyenne, variante non retenue. | [`experiments/targeted_clean_v1_20261104.md`](../experiments/targeted_clean_v1_20261104.md) ; importé par `compare_complete_results.py` | essai négatif |
| `complete_pipeline_pilot.py` | Essai de la variante complète (masquage, qualité du web, quasi-doublons) sur le tirage de 2 000 documents écrit par `targeted_clean_pilot.py`. | [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | essai négatif |

## Concours : ablation brut / nettoyé / complet (10/11/2026)

Ces cinq scripts ont servi à l'ablation dont le résultat est dans [`experiments/ablation_complete_20261110.md`](../experiments/ablation_complete_20261110.md) :
la variante complète est moins bonne que la référence de 0,010304 BPB en moyenne, le corpus actif n'a pas changé.
Pour rejouer seulement le pipeline de cette variante : `tools/reproduce.py --recipe complete`.

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `prepare_raw_ablation.py` | Prépare les témoins bruts : deux tirages de 100 Mo, protégés contre la validation, sans autre nettoyage. | Aucun script ni test ; cité par [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | Mac seulement |
| `run_complete_pipeline.py` | Lance la variante complète sur tout le corpus, puis contrôle la contamination. Il exige une copie du kit dans un dossier voisin du dépôt et une relecture locale de l'essai. | Aucun script ni test ; cité par [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | Mac seulement |
| `audit_near_pairs.py` | Recalcule la ressemblance des 50 paires de quasi-doublons tirées pour la relecture. | Aucun script ni test ; cité par [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | Mac seulement |
| `audit_complete_privacy.py` | Contrôle des masquages réels, sur au plus 50 documents masqués par source. Les extraits restent locaux. | Aucun script ni test ; cité par [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | Mac seulement |
| `compare_complete_results.py` | Compare brut, nettoyé et complet à contrat identique ; écrit `experiments/ablation_complete_20261110.json` et `.md`. | Aucun script ni test ; cité par [`docs/comprendre/pipeline_complet.md`](../docs/comprendre/pipeline_complet.md) | usage unique, résultat commité |

## Concours : remise

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `reproduce.py` | Une commande pour refaire la collecte et le pipeline du concours (`--recipe reference`, `complete` ou `final`). Il vérifie d'abord les fichiers du kit : cette vérification (`--check-only`) marche aussi sous Windows depuis le 16/11/2026 (avant, elle s'arrêtait sur « Fichier ajouté au kit » alors que le kit était intact). La reproduction intégrale demande macOS ou Linux. Il ne publie rien. | `Makefile` (en commentaire), [`README.md`](../README.md), [`docs/remise/REPRODUCTION.md`](../docs/remise/REPRODUCTION.md), `tests/test_reproduce.py` | utile |
| `export_mix_provenance.py` | Rattache chaque ligne du mélange aux métadonnées de son document d'origine. | `reproduce.py` | utile |
| `prepare_masked_reference.py` | Masque les coordonnées dans le mélange retenu et retire les nouveaux doublons exacts, puis contrôle le flux final. | `reproduce.py`, [`docs/remise/DATASET_CARD.md`](../docs/remise/DATASET_CARD.md), `tests/test_masked_reference.py` | utile |
| `prepare_dataset_release.py` | Rassemble en local les fichiers à diffuser avec le jeu (9 fichiers et leurs empreintes). Il ne publie rien. Son état écrit en dur (« non publié ») est celui du 10/11/2026. | Aucun script ni test ; cité par [`docs/remise/REPRODUCTION.md`](../docs/remise/REPRODUCTION.md) | Mac seulement |
| `build_delivery_documents.py` | Écrit les deux PDF du rendu (`output/pdf/rapport_ablation.pdf`, `output/pdf/datasheet.pdf`) et les fichiers `docs/remise/RAPPORT.md` et `DATASHEET.md`. Ne pas le relancer : il écraserait ces deux fichiers, retouchés depuis. Il importe `reportlab` et `pypdf`, absents de `pyproject.toml`. | Aucun script ni test ; cité par [`docs/remise/REPRODUCTION.md`](../docs/remise/REPRODUCTION.md) et [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | usage unique, résultat commité |
| `build_windows_bundle.py` | Construit l'archive de transfert du concours vers le PC. | Aucun script ni test ; cité par [`docs/remise/PASSAGE_PC_RTX5090.md`](../docs/remise/PASSAGE_PC_RTX5090.md) et [`README.md`](../README.md) | route jamais exécutée |
| `windows_gpu.py` | Fait passer le kit officiel sur le PC Windows à carte RTX 5090, avec contrôles. | [`docs/remise/PASSAGE_PC_RTX5090.md`](../docs/remise/PASSAGE_PC_RTX5090.md), [`README.md`](../README.md), `tests/test_windows_gpu_bundle.py` ; cité par `build_windows_bundle.py` | route jamais exécutée |
| `import_windows_results.py` | Importe l'archive de résultats du PC après contrôle et prépare un dossier local de soumission. | `tests/test_gpu_result_import.py` ; cité par [`README.md`](../README.md) | route jamais exécutée |

## Chatbot : données

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `publish_chatbot_release.py` | Envoie les gros fichiers d'entraînement en pièces jointes d'une release GitHub et régénère `chatbot/donnees_rtx.json`. Sans `--publier`, il n'envoie rien. | [`docs/chatbot/PASSAGE_PC_RTX5090_CHATBOT.md`](../docs/chatbot/PASSAGE_PC_RTX5090_CHATBOT.md), `tests/test_chatbot_pipeline_rtx.py` | Mac seulement |
| `nettoyer_affinage.py` | Nettoie le théâtre d'affinage (indications de scène, œuvres écartées par `chatbot/exclusions_theatre_v2.json`) et écrit de nouveaux fichiers. Essais D et F du 14/11/2026 : 34 et 37 réponses sans défaut sur 72, contre 44 pour le témoin. | `tests/test_nettoyer_affinage.py`, [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md) ; recettes `recettes_rtx_v4.json` et `_v4a.json` | essai négatif |
| `nettoyer_pretrain.py` | Filtre les tokens de pré-entraînement selon `chatbot/regles_pretrain_v4.json` et écrit de nouveaux fichiers. Essais E et F du 14/11/2026 : 38 et 37 réponses sans défaut sur 72, contre 44 pour le témoin. | [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md) ; recettes `recettes_rtx_v4.json` et `_v4e.json` ; `trier_fineweb.py` | essai négatif |
| `trier_fineweb.py` | Tri appris de FineWeb : un modèle à poids ouverts note des documents réels, puis une sonde linéaire apprend ces notes (`chatbot/sonde_fineweb.npz`). Aucun texte n'est généré. | [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md) ; `nettoyer_pretrain.py` | essai négatif |

## Chatbot : conversations de synthèse

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `composer_consignes_v4.py` | Compose les fichiers de consignes des campagnes v4 et v5 à partir des axes (`chatbot/axes_consignes_v4.json`, `_v5.json`). Il n'écrit aucune conversation. Recomposer redonne les deux fichiers publiés, octet pour octet (vérifié le 16/11/2026). | `campagne_synthese.py` ; [`docs/chatbot/FICHE_DONNEES.md`](../docs/chatbot/FICHE_DONNEES.md) | utile |
| `campagne_synthese.py` | Lanceur des campagnes de synthèse : `apercu`, `generer` (par un modèle à poids ouverts servi par Ollama), `bilan`, `fusion`, `refiltrer`. | [`docs/chatbot/FICHE_DONNEES.md`](../docs/chatbot/FICHE_DONNEES.md), [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | utile |

## Chatbot : entraînement (transport entre le Mac et le PC)

L'entraînement lui-même est lancé par `python -m chatbot.pipeline_rtx` : voir [`chatbot/README.md`](../chatbot/README.md).

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `build_chatbot_bundle.py` | Paquet zip autonome pour entraîner les deux chatbots sur le PC. Solution de secours : la route normale est le clone git et la commande `telecharger`. | [`docs/chatbot/PASSAGE_PC_RTX5090_CHATBOT.md`](../docs/chatbot/PASSAGE_PC_RTX5090_CHATBOT.md) ; cité par `chatbot/pipeline_rtx.py` | Mac seulement |
| `import_chatbot_results.py` | Retour du PC : vérifie l'archive de résultats, range les rapports et fait répondre chaque chatbot à 3 messages sur le processeur. Avec `--depuis-release`, il lit l'archive des résultats publiée dans une release (par exemple `chatbot-moliere-20261116`). | [`PC_RTX.md`](../PC_RTX.md), `tests/test_chatbot_pipeline_rtx.py` ; cité par `chatbot/pipeline_rtx.py` | Mac seulement |

## Chatbot : évaluation

| Script | À quoi il sert | Qui l'appelle | Statut |
| --- | --- | --- | --- |
| `nuit_moliere.py` | Contient les quatre jeux de messages de test (« reglage », « test », « neuf », « neuf2 ») et les commandes `protocole` (joue un jeu) et `bpb`. Sert aussi à la synthèse : `generer`, `refiltrer`, `fusion`, `plafonner`, `controle`. | `campagne_synthese.py` ; cité par `experiments/chatbot_rtx/jugements/aveugle.py`, [`docs/chatbot/FICHE_MODELE.md`](../docs/chatbot/FICHE_MODELE.md), [`docs/chatbot/FICHE_DONNEES.md`](../docs/chatbot/FICHE_DONNEES.md) | utile |

Les preuves des jugements à l'aveugle sont décrites dans [`experiments/chatbot_rtx/jugements/README.md`](../experiments/chatbot_rtx/jugements/README.md).
Les outils de dépouillement (`aveugle.py`, `recompter_jugements.py`, `apparie.py`) sont dans `experiments/chatbot_rtx/jugements/`, pas dans `tools/`.

## À savoir

- `composer_consignes_v4.py`, `campagne_synthese.py` et `nuit_moliere.py` ne doivent être ni renommés ni déplacés : leur chemin est écrit dans les fichiers de consignes publiés.
- `tools/__init__.py` est vide : il permet les commandes `python -m tools.<nom>`.
- Le récit des essais du chatbot est dans [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md) ; la table des recettes est dans [`chatbot/README.md`](../chatbot/README.md).
