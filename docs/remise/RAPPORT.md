# Feed the Beast

Rapport expérimental - version de travail, résultats vérifiés au 10 novembre 2026

## 1. Objectif et protocole

Nous étudions la qualité des données pour un modèle identique à celui des autres participants : 10 166 400 paramètres, contexte de 512 tokens et vocabulaire de 8 192 unités. Le BPB mesure les bits nécessaires par octet UTF-8 de validation. Une baisse indique une meilleure prédiction. Le kit, sa configuration et les 230 documents de validation restent intacts.

## 2. Corpus et recette de référence

Le corpus actif contient 3 119 274 documents et 17 352 Mo de texte issus de sept sources. Les données sont gelées avant le 16 octobre 2026. Le mélange complet est un tirage déterministe de 72 612 documents (419,30 Mo), mélangés avec la graine 1337. Sa composition est la suivante :

| Source | Documents sélectionnés | Mo de texte | Part du texte |
| --- | --- | --- | --- |
| Wikipédia | 59 289 | 193,90 | 46,24 % |
| Wikisource | 527 | 134,33 | 32,04 % |
| Gutenberg | 101 | 43,64 | 10,41 % |
| FineWeb-2 | 12 111 | 40,34 | 9,62 % |
| Assemblée nationale | 32 | 6,03 | 1,44 % |
| Wikinews | 397 | 0,59 | 0,14 % |
| Common Crawl | 155 | 0,47 | 0,11 % |

La littérature et Wikipédia dominent le mélange. Wikinews représente 0,14 % du texte ; la quantité de presse présente dans le web général n’est pas mesurée. La validation comporte environ 25 % de presse. Les proportions naturelles constituent la référence ; leur optimisation reste une piste supplémentaire.

## 3. Préparation et contrôle de contamination

Le nettoyage de référence exclut les IDs réservés, répare l’encodage, extrait le contenu principal, filtre le français à 0,65, remet les livres en paragraphes, décontamine et retire les doublons exacts. Le mélange final masque ensuite les coordonnées structurées : 548 textes modifiés, 72 612 documents conservés dans leur ordre initial. Le contrôle officiel par 13-grammes sur les 419 Mo sélectionnés signale 0 document sur 230. Zéro signalement ne signifie pas zéro phrase commune.

Sources : configs/pipeline/nettoye.yaml ; manifest et prepared/data_meta du mélange full_reference_masked_20261110 ; masked_reference/preparation.json et contamination.json.

# Expériences et décision

## 4. Réparer les données sans inventer les valeurs

La version Wikipédia HF 2023 perdait de nombreuses dates et valeurs issues des modèles wiki. Nous l’avons archivée et remplacée par le dump officiel frwiki-20261001, rendu localement. Sur l’essai de 2 000 articles : 15 trous détectés par regex (0,75 %), aucun premier mot collé et 20 relectures tracées. Sur 2 641 265 articles collectés : 0,804 % de trous regex. Les valeurs dynamiques absentes ne sont pas inventées. Ces mesures de qualité ne constituent pas une ablation d’entraînement HF 2023 contre dump 2026.

## 5. Ablation : brut, nettoyé et complet

Six runs comparent le même modèle, le même tokenizer et les mêmes hyperparamètres : 25 001 984 tokens vus, MPS float32, initialisation 1337. Les nombres 1337 et 2027 ci-dessous sont les graines de tirage des données. Les trois recettes utilisent des proportions naturelles et un plafond de 100 Mo. Le brut exclut les IDs réservés avant sélection, puis est décontaminé ; il ne reçoit aucun autre nettoyage. Le complet ajoute masquage structuré, qualité web et MinHash/LSH avec Jaccard vérifié à 0,80. Tous les flux sont assez longs pour éviter une répétition involontaire.

| Données | Tirage 1337 | Tirage 2027 | Moyenne BPB |
| --- | --- | --- | --- |
| Brutes protégées | 1,644003 | 1,671490 | 1,657747 |
| Nettoyées | 1,615490 | 1,629042 | 1,622266 |
| Complètes | 1,622668 | 1,642472 | 1,632570 |

Nettoyé moins brut : -0,035481 BPB en moyenne. Complet moins nettoyé : 0,010304. Une différence négative est meilleure. Le complet ne satisfait pas le critère conservateur de confirmation (deux tirages meilleurs et aucun genre dégradé en moyenne). La référence reste conservée.

Le contrôle de 50 paires confirme le seuil numérique, mais 38 décrivent des entités différentes au moyen d’un même gabarit : la déduplication peut réduire la variété et les faits disponibles. L’ordre des sources peut aussi conserver une édition moins bien présentée. Dans une expérience séparée, un nettoyage structurel supplémentaire dégradait déjà le score de +0,017365 BPB : son apparence plus propre ne suffisait pas à l’adopter.

## 6. Limites et passage au budget officiel

Le contrôle automatique de langue couvre 300 premiers extraits puis 80 extraits des sept sources, dont 46 tronqués et 17 cas limites. Il reste distinct d’une validation manuelle et ne caractérise pas tous les livres entiers. Le seuil fr 0,65 est conservé. Deux tirages et une initialisation commune ne prouvent pas une significativité statistique ; les différences incluent les proportions de sources après filtrage.

Le modèle final avec coordonnées masquées a vu 100 007 936 tokens : BPB 1,2883, MPS float32. La vérification CUDA float16 reste distincte. Le flux contient 114 701 016 tokens, assez pour le budget sans retour au début. Les versions et le hash sont conservés. Le passage de 25M à 100M change aussi le budget et l’échantillon : il ne constitue pas une ablation isolée du nettoyage.

Preuves : experiments/ablation_complete_20261110.json ; audit des paires ; contrôles de contamination ; expériences de nettoyage ciblé. Les mélanges, répétitions et ordres de données restent des pistes supplémentaires. Résultats issus des exécutions du projet.
