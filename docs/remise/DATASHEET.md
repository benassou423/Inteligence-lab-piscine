# Fiche du jeu de données - Feed the Beast

État du 10/11/2026. La publication et la vérification GPU restent à finaliser.

## 1. Motivation et responsabilité

Ce jeu sert au projet étudiant Feed the Beast à ECE Paris. Il permet d’entraîner depuis une initialisation aléatoire le modèle du kit et de comparer des recettes de données à budget fixe. Responsable du projet : Benjamin Assouline. Le code et la rédaction bénéficient de l’assistance de Codex ; cette assistance ne vaut pas validation humaine des annotations.

La recette de référence combine des textes encyclopédiques, littéraires, de presse, de dialogue parlementaire et de web général. Elle ne vise pas une représentation équilibrée de toute la langue française. Le test caché et le classement final sont gérés par l’organisateur.

## 2. Composition et unités

Trois niveaux sont distingués : 3 223 847 documents bruts actifs ; 3 119 274 documents après le nettoyage de référence ; 72 612 documents dans le mélange sélectionné. Une ligne du fichier d’entraînement est un objet JSON avec le champ text. Chaque document reçoit ensuite un marqueur de fin pour la tokenisation.

Le mélange contient 419 295 632 octets UTF-8 de texte et 114 701 016 tokens disponibles. Il est assez long pour le budget de 100 007 936 tokens vus. Une partie de sa fin ne sera pas vue. Les parts ci-dessous sont des parts de texte, pas des parts garanties de tokens ni des parts du préfixe effectivement entraîné.

| Source | Documents | Mo de texte | Part du texte |
|---|---:|---:|---:|
| Wikipédia | 59 289 | 193,90 | 46,24 % |
| Wikisource | 527 | 134,33 | 32,04 % |
| Gutenberg | 101 | 43,64 | 10,41 % |
| FineWeb-2 | 12 111 | 40,34 | 9,62 % |
| Assemblée nationale | 32 | 6,03 | 1,44 % |
| Wikinews | 397 | 0,59 | 0,14 % |
| Common Crawl | 155 | 0,47 | 0,11 % |

## 3. Origine, dates et licences déclarées

Wikipédia provient du dump frwiki-20261001, sous CC BY-SA 4.0 déclarée. Wikinews fournit des articles sous CC BY 2.5. Wikisource regroupe des transcriptions et œuvres avec statuts de droits variables, décrits dans les fiches de source. Gutenberg déclare les textes de domaine public aux États-Unis ; les situations dans d’autres pays restent à vérifier. L’Assemblée nationale utilise la Licence Ouverte.

FineWeb-2 français est un sous-ensemble déterministe d’un dépôt figé (af9c13333eb981300149d5ca60a8e9d659b276b9), déclaré ODC-By 1.0. Common Crawl utilise CC-MAIN-2026-39, avec droits des pages détenus par leurs auteurs. Les déclarations des collecteurs ne constituent pas une vérification juridique indépendante de chaque œuvre ou page. Aucune licence uniforme n’est attribuée à l’ensemble.

Le gel porte sur le 16/10/2026. La date d’une capture web, celle d’une révision wiki et celle de publication d’une œuvre ont des sens différents. Les dates des sources sont conservées et leur sémantique est décrite par source. La fiche des sources et les URLs de provenance doivent accompagner une diffusion autorisée.

## 4. Collecte et sélection

Les collecteurs lisent les dumps ou fichiers en flux et conservent les documents sous JSONL compressé zstd. Ils enregistrent des identifiants stables, une source, un domaine, une URL, une date et une licence déclarée. Les archives, manifestes et dossiers de travail sont conservés sous des liens .nosync sur le Mac.

L’ancienne Wikipédia HF 2023 reste archivée pour ablation. Elle n’entre pas dans le mélange officiel préparé. Les autres limites de collecte comprennent le faible échantillon de Common Crawl, les livres très longs de Wikisource et les dates anciennes du web général. Aucune campagne de consentement individuel n’a été menée auprès des personnes mentionnées dans ces sources publiques.

## 5. Prétraitement, langue et qualité

La configuration nettoye applique huit étapes : exclusion des IDs de référence, réparation d’encodage, extraction du contenu principal, score de langue, filtre français, remise en paragraphes des livres, décontamination puis déduplication exacte. Le seuil actif est fr >= 0,65. GlotLID donne un diagnostic probabiliste ; les langues proches ne sont pas toutes assimilées au français.

Le contrôle automatique comprend 300 premiers extraits, puis 80 extraits couvrant les sept sources : 63 français, 10 sans français rédigé exploitable, 7 mélangés ; 17 cas limites. Pour ces 80 extraits, la précision pondérée du filtre est de 99,69 % relativement aux classes automatiques ; la sensibilité aux cas limites donne 99,56–99,80 %. Les 46 extraits tronqués ne permettent pas de juger leurs livres entiers. L’audit complémentaire utilise deux détecteurs sur 888 documents entiers, avec 75 relectures automatisées (64 intégrales, 11 par passages). Ce protocole reste distinct d’une validation manuelle indépendante.

La variante complète ajoute qualité web, masquage et MinHash/LSH avec Jaccard vérifié à 0,80. Elle a été exécutée, contrôlée à zéro contamination et évaluée sur deux tirages : 1,632570 BPB moyen, contre 1,622266 pour la référence nettoyée. Les paires proches peuvent décrire des entités différentes ; cette variante n’est pas retenue. Le mélange final conserve la référence avec masquage structuré de ses coordonnées, dans le même ordre. Le contrôle final trouve zéro document de validation signalé.

## 6. Validation, fuite et reproductibilité

La validation publique contient 230 documents et reste réservée à la mesure. Le contrôle officiel recherche les 13-grammes de mots partagés, signale les documents dépassant 10 % de recouvrement et a signalé 0 document sur le mélange complet. Zéro signalement n’implique pas l’absence de toute phrase commune. Le contrôle du test caché relève de l’organisateur.

Le tokenizer a 8 192 tokens, sans normalisation destructive, padding, troncature ou BPE-dropout. Il est repris de la référence des essais pour isoler les effets du nettoyage. Tokenizers 0.23.3 est figé ; PyTorch 2.15.0 est la version du projet. Le PC utilisera le binaire CUDA adapté, avec une vraie vérification de calcul GPU. Le kit et sa configuration restent intacts.

Hash du flux préparé : b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee. Hash du tokenizer : 35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584. Le hash du gzip de texte vaut 910ae65f995bc3bc4685532b551422c0e3694f3eec6e474907f8fea96823dd7d. Les scripts de transfert refusent les écarts sur les données et les fichiers critiques. Une reconstruction complète depuis une copie neuve et la comparaison T4 à ±0,02 BPB ne sont pas encore attestées.

## 7. Données personnelles, biais et usages

Le mélange final masque 423 occurrences d’emails, 486 de téléphones et 4 d’adresses IP, dans 548 textes. Aucun article n’est retiré par cette transformation. Les numéros explicitement contextuels sont distingués des tableaux et décimales. Ce masquage ne constitue pas une anonymisation exhaustive : noms, adresses postales, formats obfusqués et de nombreux téléphones internationaux ne sont pas couverts. Les métadonnées d’attribution restent également à considérer avant diffusion.

Les textes anciens, les traductions, les opinions politiques, le commerce, les stéréotypes et des erreurs de numérisation peuvent influencer le modèle. La provenance publique ne garantit pas la factualité. Le modèle reste petit et son entraînement expérimental ; il ne doit pas servir à des décisions médicales, juridiques, financières ou individuelles importantes.

Usages prévus : recherche pédagogique, comparaison de données, reproduction du protocole et analyse de générations. Aucun effet sur des populations particulières n’a été mesuré. Les scores locaux ne prouvent pas une performance générale sur toute la langue française.

## 8. Diffusion et maintenance

Le mélange et les paquets restent locaux ; aucune URL publique définitive n’est encore validée. L’index provenance.jsonl.gz relie les 72 612 lignes au document d’origine et au texte transformé, grâce à leurs deux SHA-256 et aux masques appliqués. Le texte et l’ordre sont vérifiés après préparation du banc. Les URLs et licences déclarées d’origine sont conservées ; cet index et les fiches des sources doivent accompagner une diffusion autorisée.

Une diffusion doit garder les textes, tokenizer, manifeste, versions, empreintes et informations de provenance nécessaires. Les conditions de chaque source s’appliquent ; les auteurs et liens d’origine doivent rester consultables. Les demandes de correction ou de retrait devront être traitées par le responsable du projet et donner lieu à une nouvelle version identifiée, jamais au remplacement silencieux du flux soumis.

Cette fiche suit les thèmes de motivation, composition, collecte, préparation, usages, distribution et maintenance proposés par Gebru et al., Datasheets for Datasets, arXiv:1803.09010 (2018, version publiée 2021). Elle décrit l’état constaté du projet et sera mise à jour après les vérifications CUDA et la préparation de la publication.
