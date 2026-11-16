# Table des matières de `docs/`

Ce dossier contient les documents de deux travaux :

- le **concours** « Feed the Beast » (jeu de données français, mesuré en BPB), rendu le 11/11/2026 ;
- le **chatbot « à la manière de Molière »**, hors concours, du 13 au 16/11/2026.

Une ligne par fichier. La colonne « Statut » dit ce que vaut le document aujourd'hui :

- **à jour** : décrit l'état du dépôt au 16/11/2026 ;
- **rendu du concours du 11/11/2026** : document écrit pour le rendu du concours ; sauf mention contraire, il ne décrit pas le chatbot ;
- **pièce historique** : gardé comme trace, ne décrit plus une chose à faire ;
- **kit du concours** : fourni par l'organisateur, non modifié.

Hors de `docs/` : la présentation du dépôt est dans [`README.md`](../README.md), les licences dans
[`LICENCES.md`](../LICENCES.md), le guide d'entraînement du chatbot dans [`PC_RTX.md`](../PC_RTX.md),
l'index des outils dans [`tools/README.md`](../tools/README.md) et la table des essais du chatbot (avec la recette
du modèle publié) dans [`chatbot/README.md`](../chatbot/README.md).

## À la racine de `docs/`

| Fichier | Ce qu'il contient | Statut |
| --- | --- | --- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Conventions communes du projet : arborescence, format d'un document du pipeline, commandes, dossiers. | à jour |
| [`KIT_README.md`](KIT_README.md) | Documentation du kit officiel : les trois blocs du sujet, la configuration du modèle, les scripts fournis. Elle cite un fichier `LEADERBOARD.md` qui n'est pas dans ce dépôt : le document vient du kit et n'est pas modifié. | kit du concours |

## `docs/chatbot/` : le chatbot Molière

| Fichier | Ce qu'il contient | Statut |
| --- | --- | --- |
| [`FICHE_MODELE.md`](chatbot/FICHE_MODELE.md) | Fiche du modèle : architecture, entraînement, choix de la réponse, évaluation, utilisation, licences, limites. Le modèle publié est S03. | à jour |
| [`FICHE_DONNEES.md`](chatbot/FICHE_DONNEES.md) | Fiche des données : les six sources du pré-entraînement, le théâtre d'affinage, les conversations de synthèse, les licences, où sont les fichiers. | à jour |
| [`HISTORIQUE.md`](chatbot/HISTORIQUE.md) | Récit daté du 13 au 16/11/2026 : ce qui a été essayé, les résultats, les décisions, les erreurs et les incidents. | à jour |
| [`PASSAGE_PC_RTX5090_CHATBOT.md`](chatbot/PASSAGE_PC_RTX5090_CHATBOT.md) | Transport du code, des gros fichiers et des résultats entre le Mac et le PC à carte RTX 5090 (route « git clone »), suivi le 13/11/2026. Plusieurs parties demandent le Mac de l'auteur. Pour refaire l'entraînement, suivre [`PC_RTX.md`](../PC_RTX.md). | pièce historique |

## `docs/comprendre/` : une fiche par sujet

Fiches courtes, écrites pour expliquer chaque étape. Toutes portent sur le concours, sauf `chatbot_moliere.md`.

| Fichier | Ce qu'il contient | Statut |
| --- | --- | --- |
| [`audit_langue_complete.md`](comprendre/audit_langue_complete.md) | Audit de la langue sur les textes complets (04/11/2026) : ce qu'un filtre plus large ajouterait, et pourquoi le filtre actuel est gardé. | rendu du concours du 11/11/2026 |
| [`banc_essai.md`](comprendre/banc_essai.md) | Le banc d'essai : comparer des jeux de données avec le kit officiel et lire le BPB par genre. | rendu du concours du 11/11/2026 |
| [`boilerplate.md`](comprendre/boilerplate.md) | Retirer ce qui entoure le contenu : menus et pieds de page du web, en-têtes et licence de Gutenberg. | rendu du concours du 11/11/2026 |
| [`bpb.md`](comprendre/bpb.md) | Le BPB (bits par octet), la note du concours : comment le kit le calcule. | rendu du concours du 11/11/2026 |
| [`chatbot_moliere.md`](comprendre/chatbot_moliere.md) | Le chatbot Molière expliqué : format des conversations, choix de la réponse, entraînement, essais, limites. | à jour |
| [`collecte.md`](comprendre/collecte.md) | La collecte : une commande par source, un document par ligne JSON, avec sa date et sa licence. | rendu du concours du 11/11/2026 |
| [`decontamination.md`](comprendre/decontamination.md) | Retirer des données les textes qui recouvrent la validation du kit (suites de 13 mots, même mesure que le kit). | rendu du concours du 11/11/2026 |
| [`dedup_exacte.md`](comprendre/dedup_exacte.md) | Doublons exacts : garder une seule copie de chaque texte, par empreinte du texte normalisé. | rendu du concours du 11/11/2026 |
| [`encodage.md`](comprendre/encodage.md) | Réparer le texte mal décodé (« Ã©tÃ© » au lieu de « été ») et le mettre dans une seule forme Unicode. | rendu du concours du 11/11/2026 |
| [`langue.md`](comprendre/langue.md) | Le filtre de langue : le détecteur GlotLID, le seuil de 0,65 sur le score du français, les mesures du 04/11/2026. | rendu du concours du 11/11/2026 |
| [`minigpt_bigram.md`](comprendre/minigpt_bigram.md) | Bloc 1 : le bigramme par comptage (`minigpt/bigram.py`), point de comparaison du mini-GPT. | rendu du concours du 11/11/2026 |
| [`minigpt_bpe.md`](comprendre/minigpt_bpe.md) | Bloc 1 : le tokenizer BPE écrit à la main (`minigpt/bpe.py`). | rendu du concours du 11/11/2026 |
| [`minigpt_entrainement.md`](comprendre/minigpt_entrainement.md) | Bloc 1 : l'entraînement du mini-GPT sur Molière (`minigpt/train_moliere.py`) et sa comparaison aux bigrammes. | rendu du concours du 11/11/2026 |
| [`minigpt_gpt.md`](comprendre/minigpt_gpt.md) | Bloc 1 : le transformer décodeur écrit à la main (`minigpt/gpt.py`), vérifié contre PyTorch. | rendu du concours du 11/11/2026 |
| [`minigpt_micrograd.md`](comprendre/minigpt_micrograd.md) | Bloc 1 : le calcul des gradients en Python pur (`minigpt/micrograd.py`). | rendu du concours du 11/11/2026 |
| [`nettoyage_cible.md`](comprendre/nettoyage_cible.md) | Le nettoyage ciblé (tableaux bruts, cadres, répétitions). Variante non retenue : mesures dans [`experiments/targeted_clean_v1_20261104.md`](../experiments/targeted_clean_v1_20261104.md). | rendu du concours du 11/11/2026 |
| [`pipeline.md`](comprendre/pipeline.md) | Le pipeline de nettoyage : les étapes dans l'ordre, la configuration YAML, les statistiques de chaque étape. | rendu du concours du 11/11/2026 |
| [`pipeline_complet.md`](comprendre/pipeline_complet.md) | La variante `complet_20261110` : coordonnées masquées, qualité du web, quasi-doublons. Variante non retenue : mesures dans [`experiments/ablation_complete_20261110.md`](../experiments/ablation_complete_20261110.md). | rendu du concours du 11/11/2026 |
| [`reflow.md`](comprendre/reflow.md) | Mise en forme des livres de Gutenberg : recoller les lignes coupées, pour retrouver la forme de la validation. | rendu du concours du 11/11/2026 |
| [`tokenizer.md`](comprendre/tokenizer.md) | Le tokenizer BPE de 8 192 tokens du concours : les règles du kit et le pré-découpage. | rendu du concours du 11/11/2026 |
| [`wikipedia_modeles.md`](comprendre/wikipedia_modeles.md) | Pourquoi et comment les modèles du wikitexte de Wikipédia sont rendus en texte lisible. | rendu du concours du 11/11/2026 |

## `docs/sources/` : une fiche par source de données

Chaque fiche donne l'origine, la commande de collecte, la licence déclarée et les limites de la source.

| Fichier | Ce qu'il contient | Statut |
| --- | --- | --- |
| [`assemblee_nationale.md`](sources/assemblee_nationale.md) | Débats en séance publique de l'Assemblée nationale (données ouvertes, format XML Syceron). | rendu du concours du 11/11/2026 |
| [`commoncrawl.md`](sources/commoncrawl.md) | Pages françaises de Common Crawl (crawl CC-MAIN-2026-39). | rendu du concours du 11/11/2026 |
| [`fineweb2_fra.md`](sources/fineweb2_fra.md) | FineWeb-2 en français (jeu Hugging Face, configuration `fra_Latn`), sous-ensemble tiré de façon déterministe. | rendu du concours du 11/11/2026 |
| [`gutenberg_fr.md`](sources/gutenberg_fr.md) | Livres en français du Project Gutenberg, pris sur un miroir officiel. | rendu du concours du 11/11/2026 |
| [`moliere.md`](sources/moliere.md) | Corpus Molière de Wikisource (édition Louandre). Il a servi au bloc 1 en attendant le kit, qui l'a remplacé pour ce bloc. Il sert encore au théâtre du chatbot (`chatbot/theatre.py`) et au pré-entraînement (`chatbot/pretrain.py`). | à jour |
| [`volumes.md`](sources/volumes.md) | Tableau des volumes collectés par source, généré le 04/11/2026. | rendu du concours du 11/11/2026 |
| [`wikinews_fr.md`](sources/wikinews_fr.md) | Wikinews en français (dump officiel du 01/10/2026). | rendu du concours du 11/11/2026 |
| [`wikipedia_fr.md`](sources/wikipedia_fr.md) | Wikipédia en français (dump officiel du 01/10/2026) : méthode, conversion en texte, licence, volumes, limites. | rendu du concours du 11/11/2026 |
| [`wikipedia_fr_hf2023.md`](sources/wikipedia_fr_hf2023.md) | Ancienne version de Wikipédia (jeu Hugging Face de 2023), gardée comme archive et désactivée dans la collecte. | pièce historique |
| [`wikisource_fr.md`](sources/wikisource_fr.md) | Wikisource en français (dump officiel du 01/10/2026) : un document par livre scanné et relu, ou par page qui porte son propre texte. | rendu du concours du 11/11/2026 |

## `docs/remise/` : le rendu du concours

| Fichier | Ce qu'il contient | Statut |
| --- | --- | --- |
| [`RAPPORT.md`](remise/RAPPORT.md) | Rapport d'expériences : corpus, recette de référence, contrôle de contamination, ablation brut / nettoyé / complet, limites. | rendu du concours du 11/11/2026 |
| [`DATASHEET.md`](remise/DATASHEET.md) | Fiche du jeu de données du concours : composition, origine et licences, prétraitement, données personnelles, diffusion. | rendu du concours du 11/11/2026 |
| [`DATASET_CARD.md`](remise/DATASET_CARD.md) | Carte du jeu de données : usage, fichiers à diffuser ensemble, sources et licences déclarées, limites, reproduction. | rendu du concours du 11/11/2026 |
| [`EXIGENCES.md`](remise/EXIGENCES.md) | Audit de remise du 11/11/2026 : chaque exigence, son état constaté, sa preuve ou sa limite. | rendu du concours du 11/11/2026 |
| [`CONSIGNES_TRANSMISES.md`](remise/CONSIGNES_TRANSMISES.md) | Les exigences du sujet, bloc par bloc, telles qu'elles ont été transmises le 10/11/2026. | rendu du concours du 11/11/2026 |
| [`PROTOCOLE_ABLATION.md`](remise/PROTOCOLE_ABLATION.md) | Protocole de l'ablation brut / nettoyé / complet : conditions fixes, sélection, contrôles, lecture des résultats. | rendu du concours du 11/11/2026 |
| [`REPRODUCTION.md`](remise/REPRODUCTION.md) | Reconstruire le jeu du concours en une commande (`tools/reproduce.py --recipe final`). Une section finale renvoie au chatbot. | rendu du concours du 11/11/2026 |
| [`KIT_SHA256.json`](remise/KIT_SHA256.json) | Empreintes sha256 des 18 fichiers du kit officiel (commit `3709c85`), pour vérifier que le kit n'a pas été modifié. | rendu du concours du 11/11/2026 |
| [`PASSAGE_PC_RTX5090.md`](remise/PASSAGE_PC_RTX5090.md) | Guide préparé pour refaire l'entraînement du concours sur le PC à carte RTX 5090. Cette route n'a jamais été exécutée : le modèle remis a été entraîné sur le Mac. | pièce historique |

## Ce qui n'est pas dans `docs/`

- Les mesures du concours et du chatbot : [`experiments/README.md`](../experiments/README.md) (pour le concours, une ligne par run dans `results.csv` et un fichier par expérience).
- Les rapports du chatbot : `experiments/chatbot_rtx/20261113_1956_M/` (lancement du 13/11/2026), `experiments/chatbot_rtx/20261116_modele_publie/` (modèle publié, 16/11/2026), `experiments/chatbot_rtx/jugements/` (jugements à l'aveugle) et `experiments/chatbot_rtx/outils/` (scripts des essais, décrits dans son [`README.md`](../experiments/chatbot_rtx/outils/README.md)).
- Les preuves des jugements à l'aveugle du chatbot, dont le jugement final du 16/11/2026 (`final_neuf2/`) : dossier [`experiments/chatbot_rtx/jugements/`](../experiments/chatbot_rtx/jugements/), décrit dans son [`README.md`](../experiments/chatbot_rtx/jugements/README.md).
- Les licences des données et du modèle : [`LICENCES.md`](../LICENCES.md).
- Les deux PDF du rendu : `output/pdf/rapport_ablation.pdf` et `output/pdf/datasheet.pdf`.
- Les résultats du bloc 1 : `minigpt/results/RESULTS.md`.
