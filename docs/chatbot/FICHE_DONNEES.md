# Fiche des données — chatbot « Molière »

Cette fiche dit d'où viennent les textes qui ont entraîné le chatbot, comment ils ont été préparés, sous
quelles licences ils sont déclarés, et ce qu'on ne sait pas. Elle ne décrit pas le jeu de données du
concours (voir [`docs/remise/DATASET_CARD.md`](../remise/DATASET_CARD.md)), même si ce jeu fait partie
du pré-entraînement sous le nom « melange ».

Le modèle lui-même est décrit dans [`FICHE_MODELE.md`](FICHE_MODELE.md).

## 1. En bref

| Étape | Données | Tokens disponibles | Tokens vus par le modèle |
| --- | --- | ---: | ---: |
| Pré-entraînement (de zéro) | 6 sources de français écrit | 2 114 055 586 | 2 000 027 648 |
| Affinage | théâtre français + Molière (+ synthèse, voir § 4) | 29 806 642 de théâtre | 90 013 696 |
| Validation (jamais entraînée) | 3 pièces de Molière | 71 358 | — |

Tous les nombres de tokens sont comptés avec le tokenizer du projet (BPE au niveau des octets, 8 192
entrées, `submissions/2026-11-11_153839/tokenizer.json`, sha256 `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`).
Ce tokenizer a été appris pour le concours, sur un échantillon du corpus nettoyé, pas sur les données du chatbot.

## 2. Pré-entraînement

### 2.1 Les six sources

| Source | Origine | Licence déclarée par le dépôt | Tokens dans le fichier | Part du tirage | Tokens vus | Passages |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `wikipedia` | dump officiel frwiki du 01/10/2026 | CC BY-SA 4.0 | 600 000 000 (plafond) | 22 % | 440 682 496 | 0,734 |
| `wikisource` | dump officiel frwikisource du 01/10/2026, pages relues (niveaux 3 et 4) | œuvres : statut variable ; transcription Wikisource CC BY-SA 4.0 | 500 000 000 (plafond) | 18 % | 360 245 248 | 0,72 |
| `fineweb` | Hugging Face `HuggingFaceFW/fineweb-2`, configuration `fra_Latn`, commit `af9c13333eb981300149d5ca60a8e9d659b276b9` (pages Common Crawl de 2013 à avril 2024) | ODC-By 1.0, plus les conditions de Common Crawl ; le droit d'auteur des pages reste à leurs auteurs | 453 689 490 | 20 % | 399 759 872 | 0,881 |
| `gutenberg` | Project Gutenberg, catalogue du 01/11/2026, 4 228 livres en français | Project Gutenberg License ; domaine public aux États-Unis | 440 796 409 | 27 % | 539 434 496 | 1,224 |
| `melange` | jeu de données du concours : 72 612 documents de 7 sources | celle de chaque source (aucune licence unique) | 112 808 606 | 12 % | 240 072 192 | 2,128 |
| `wikinews` | dump officiel frwikinews du 01/10/2026 | CC BY 2.5 | 6 761 081 | 1 % | 19 833 344 | 2,933 |
| **Total** | | | **2 114 055 586** | 100 % | **2 000 027 648** | |

« Part du tirage » : à l'entraînement, chaque fenêtre de 512 tokens est tirée dans une source avec cette
probabilité, quelle que soit la taille de la source. C'est pourquoi le modèle repasse plus de deux fois sur
les petites sources et moins d'une fois sur les grandes. « Tokens vus » et « passages » sont mesurés dans le
rapport du pré-entraînement (`experiments/chatbot_rtx/20261113_1956_M/entrainements/base/report.json`).

Le « melange » contient aussi deux sources qui n'ont pas de fichier à elles : 32 séances de l'Assemblée
nationale (open data, Licence Ouverte 2.0) et 155 pages Common Crawl (crawl CC-MAIN-2026-39).

Les fiches détaillées de chaque source sont dans [`docs/sources/`](../sources/).

### 2.2 Préparation

1. **Nettoyage commun** (pipeline « nettoye », 8 étapes) : réparation d'encodage et normalisation NFC,
   retrait des en-têtes et pieds de page, détection de la langue et filtre « français », remise en
   paragraphes des livres, retrait des passages communs avec la validation du concours (suites de 13 mots),
   retrait des doublons exacts. Détail : [`docs/comprendre/pipeline_complet.md`](../comprendre/pipeline_complet.md).
2. **Filtres propres au chatbot** (`chatbot/pretrain.py`, version « pretrain-filtres-v2 »), dans cet ordre :
   liste d'exclusion [`chatbot/exclusions_theatre.json`](../../chatbot/exclusions_theatre.json) (108 documents
   concernés : orthographe trop ancienne, contenu érotique ou obscène, texte illisible), texte de moins de
   200 caractères, vocabulaire sexuel cru, puis recouvrement avec la validation Molière.
3. **Mise à l'écart de la validation Molière** : tout document qui partage au moins 3 suites de 13 mots avec
   les 3 pièces de validation est écarté. 100 documents l'ont été (gutenberg 43, wikisource 39, fineweb 8,
   wikipedia 7, melange 3).
4. **Fichiers de tokens** : un fichier `.bin` par source (entiers de 16 bits), chaque document suivi du token
   `<|endoftext|>`. Le bilan par source (documents lus, gardés, écartés par raison) est dans
   `data/stages/chatbot_20261113/pretrain/meta.json`, publié sous le nom `pretrain_meta.json`.

Au total 865 671 documents gardés sur 882 584 lus, soit 7 726 016 390 octets de texte.

### 2.3 Limites à connaître

- **`wikipedia` et `wikisource` ne sont pas des tirages uniformes.** Les deux fichiers sont coupés à leur
  plafond après lecture dans l'ordre trié des fichiers : 258 347 articles lus sur 2 546 229 pour Wikipédia,
  8 510 documents lus pour Wikisource. C'est le début du corpus nettoyé, pas un échantillon représentatif.
- **Doublons possibles entre `melange` et les autres sources.** Le mélange est tiré du même corpus nettoyé
  que les cinq autres sources, et il n'y a pas de dédoublonnage entre sources : un même document peut se
  trouver deux fois. Leur nombre n'a pas été mesuré.
- **Seuls les doublons exacts sont retirés** par le pipeline du projet. Une variante avec quasi-doublons a
  été évaluée pour le concours et non retenue.
- **FineWeb contient du texte de mauvaise qualité** (pages commerciales, texte généré, listes). Un essai de
  tri a été fait et n'a pas amélioré le modèle : voir § 6.

## 3. Affinage : le théâtre

### 3.1 Les trois fichiers

| Fichier | Contenu | Scènes | Répliques | Tokens |
| --- | --- | ---: | ---: | ---: |
| `theatre_train.jsonl.zst` | théâtre français hors Molière : 969 livres (800 de Wikisource, 169 de Gutenberg) | 33 396 | 711 078 | 28 961 665 |
| `moliere_train.jsonl.zst` | 30 pièces de Molière (24 du kit du concours, 6 de Wikisource) | 790 | 13 583 | 564 910 (844 977 avec les copies, voir 3.3) |
| `moliere_val.jsonl.zst` | 3 pièces de Molière **jamais entraînées** : *L'Avare*, *Le Misanthrope*, *Le Sicilien* | 96 | 1 686 | 71 358 |

Une ligne = une scène : `id`, `source`, `play`, `doc_id`, `url`, `license`, `is_moliere`, `turns` (liste de
`{speaker, text}`). Il n'y a pas de champ « auteur » : on peut compter les livres, pas les auteurs ni les pièces
(un livre peut être un recueil).

Origine de Molière : *Œuvres complètes*, édition Chasles de 1888 sur Project Gutenberg (ebooks 40086, 43535,
50173 et 57270), préparée par le kit du concours (`corpus/build_moliere.py`) ; six pièces absentes du kit
viennent de Wikisource (*Le Bourgeois gentilhomme*, *Psyché*, *Les Fourberies de Scapin*, *La Comtesse
d'Escarbagnas*, *Les Femmes savantes*, *Le Malade imaginaire*).

Licences écrites dans les fichiers : « Domaine public » pour le kit ; « Domaine public (œuvres) ; transcription
Wikisource CC BY-SA 4.0 » pour Wikisource (28 871 scènes de `theatre_train`, 235 de `moliere_train`) ;
« Project Gutenberg License (domaine public aux États-Unis) » pour Gutenberg (4 525 scènes).

### 3.2 Comment une pièce devient des scènes

Le code est [`chatbot/theatre.py`](../../chatbot/theatre.py). En résumé :

- chaque ligne reçoit un rôle : nom de personnage, réplique, titre, liste des personnages, didascalie ;
- les titres, listes de personnages, didascalies, numéros de vers et appels de note sont retirés ;
- une scène est la suite des répliques entre deux titres ; elle est jetée si elle a moins de 2 répliques ou
  moins de 2 personnages, et coupée en morceaux de 60 répliques au plus ;
- un livre n'est gardé que s'il ressemble à du théâtre (au moins 10 répliques, des noms qui reviennent, etc.) ;
- 142 livres sont écartés par la liste [`chatbot/exclusions_theatre.json`](../../chatbot/exclusions_theatre.json) :
  68 pour orthographe trop ancienne, 34 pour contenu érotique ou obscène, 34 parce que ce n'est pas du
  dialogue, 6 illisibles ou pas en français ;
- les scènes en double entre éditions sont retirées (80 % de suites de 13 mots communes).

**Protection de la validation.** Aucune édition des trois pièces de validation ne doit entrer dans
l'entraînement. Le contrôle se fait par suites de 13 mots (livres) et de 12 mots (scène par scène). Vérifié de
façon indépendante le 15/11/2026 : aucune des 34 186 scènes d'entraînement ne partage une suite de 12 mots
avec `moliere_val`.

### 3.3 Format d'entraînement et copies « MOLIÈRE »

Une réplique s'écrit « `NOM.` », retour à la ligne, le texte, puis une ligne vide. Au chat, le message de
l'utilisateur devient la réplique d'un interlocuteur (LA GRANGE pour MOLIÈRE) et le modèle écrit la réplique
du personnage.

Le personnage MOLIÈRE n'a que 99 vraies répliques (*L'Impromptu de Versailles*). Pour qu'il devienne « la voix
moyenne des personnages de Molière », une scène de Molière sur deux est ajoutée une seconde fois avec son
personnage principal renommé MOLIÈRE et le deuxième renommé LA GRANGE (option `persona_augment` 0,5 : 387
copies, le texte des répliques n'est pas modifié).

À l'affinage, Molière est tiré 3 fois plus souvent par token que le reste du théâtre (poids 1 et 3).

### 3.4 Limites à connaître

- **HARPAGON et ALCESTE** sont proposés au chat, mais *L'Avare* et *Le Misanthrope* sont des pièces de
  validation : le modèle n'a jamais vu ces rôles chez Molière. Ce sont des tests de généralisation.
- `theatre_train` contient 36 répliques d'un personnage nommé HARPAGON dans un livre de théâtre de Guignol
  (1874) et 41 répliques d'un personnage nommé MOLIÈRE dans trois livres qui ne sont pas de Molière.
- `theatre_train` contient des traductions et des livrets chantés (voir § 6 : les retirer n'a pas amélioré le
  modèle).
- Le statut de chaque œuvre en droit français n'a pas été vérifié une à une (voir § 7).

## 4. Données de synthèse

Le théâtre apprend le style, mais aucun texte réel ne montre « un message d'aujourd'hui → une réponse à la
manière de Molière ». Une petite part de conversations de ce genre a donc été **écrite par un modèle de
langue** et mélangée à l'affinage.

### 4.1 Règles suivies

- Le générateur est un modèle **à poids ouverts** : `gemma4:31b-it-q4_K_M` (licence Apache 2.0), lancé en
  local avec Ollama. Aucune conversation n'a été écrite à la main ni par un modèle fermé.
- Les **consignes sont publiées** : les axes ([`chatbot/axes_consignes_v4.json`](../../chatbot/axes_consignes_v4.json),
  [`chatbot/axes_consignes_v5.json`](../../chatbot/axes_consignes_v5.json)), le programme qui les croise
  ([`tools/composer_consignes_v4.py`](../../tools/composer_consignes_v4.py)) et les consignes complètes
  (`chatbot/prompts_synthetiques_v4.json`, `chatbot/prompts_synthetiques_v5.json`).
- La **part de synthèse est bornée** (15 % au plus) **et déclarée** dans le rapport du modèle
  (`training_report.json`, clé `part_synthetique`), mesurée sur les tokens réellement vus.
- Chaque fichier de données porte une trace (`.campagne.json`) : empreinte des consignes, modèle, graine,
  options envoyées.

### 4.2 Fabrication

Un appel = une « fiche » = une conversation. Une fiche croise un thème (651 thèmes), un type de message et le
type de réponse qui lui va (51 paires), une voix d'utilisateur (20), un personnage (MOLIÈRE dans environ une
fiche sur deux, 8 autres personnages de Molière sinon) et une longueur (2 à 8 répliques). Environ un quart des
fiches portent sur la conversation de base (saluer, demander le nom, l'âge, dire qu'on est fatigué, remercier,
prendre congé).

Chaque conversation passe ensuite par des filtres automatiques ([`chatbot/synthetic.py`](../../chatbot/synthetic.py),
[`tools/campagne_synthese.py`](../../tools/campagne_synthese.py)) :

- forme (alternance des deux voix, longueur, pas de tutoiement du personnage, pas de vocabulaire cru, pas
  d'écho du message) ;
- relecture du fond par le générateur lui-même (trois questions fermées, consigne publiée) ;
- anti-doublons : conversation identique, réplique longue déjà vue, formule de 5 mots trop répandue, premier
  mot de réponse trop fréquent ;
- retrait de toute conversation qui partage 6 mots de suite avec la validation Molière ou qui cite un de ses
  personnages ;
- retrait de toute conversation dont un message d'utilisateur est un des messages de test (§ 5).

### 4.3 Chiffres

| Campagne | Consignes | Appels | Gardées à la génération | Après fusion et refiltrage | Après retrait des messages de test |
| --- | --- | ---: | ---: | ---: | ---: |
| v4 (14 et 15/11/2026) | `prompts_synthetiques_v4.json`, 9 000 fiches | 13 660 | 10 170 | 9 538 | 9 055 (897 109 tokens) |
| v5 (15 et 16/11/2026) | `prompts_synthetiques_v5.json`, 26 000 fiches | 29 468 (sur 48 000 prévus, soit 61 % ; arrêtée volontairement le 16/11 à 1 h 03) | 21 830 | 20 394 (21 766 après fusion, puis refiltrage : tour répété 1 315, doublon exact 51, quasi-doublon 3, ouverture répétée 3) | non mesuré pour la v5 seule |
| v4 + v5 réunies (fichier des essais du 16/11) | | | | 28 971 (9 538 + 19 433 ; 936 tours répétés et 25 doublons exacts écartés) | **27 624** (1 147 010 mots, 2 694 831 tokens) |

Sur la v5, la fusion a retiré 64 conversations en tout : 62 fuites de la validation sur 6 mots et 2 personnages
de validation. Dans le fichier v4 + v5, le retrait des messages de test a écarté les conversations qui
contenaient un message du jeu « reglage » (817), « test » (340), « neuf » (142) ou « neuf2 » (48).

**Contrôle de qualité de la campagne v4** : 120 conversations tirées au hasard, lues par 3 lecteurs
indépendants avec la même grille (français correct, réponse en rapport avec le message, vouvoiement, style) :
113 sans défaut à la majorité (seuil fixé d'avance : 96).

**Contrôle de qualité du fichier v4 + v5** (27 624 conversations) : même méthode, 120 conversations tirées au
hasard, 3 lecteurs (agents IA) : **109 sans défaut** à la majorité (seuil 96). Par lecteur : 110, 109 et 107.
Par critère : français 117, rapport au message 115, vouvoiement 120, style 116. Les notes sont dans
[`experiments/chatbot_rtx/jugements/controle_synthese_v4_v5/`](../../experiments/chatbot_rtx/jugements/controle_synthese_v4_v5/).
Ce contrôle porte sur le fichier préparé pour les essais du 16/11, pas sur celui du modèle publié.

**Synthèse du modèle publié (S03).** Il a été affiné avec le fichier de la campagne v4 seule :
`data/stages/chatbot_20261114/synthetique_v4/entrainement/moliere.jsonl.zst`, 9 055 conversations. Son rapport
d'entraînement (`training_report.json`) déclare 2 720 768 tokens de synthèse sur 90 013 696 tokens d'affinage,
soit une part de 0,0302 (3,02 % des tokens vus), « melangee_a_l_etape1 ». Le fichier fait 897 109 tokens. Les
2 720 768 tokens de synthèse sont comptés sur tout l'affinage (2 747 pas) ; les poids gardés sont ceux du pas
1 000, où chaque conversation a été vue environ 1,1 fois. La campagne v5 n'a pas servi à entraîner le modèle
publié : elle n'a servi qu'aux quatre candidats du 16/11, dont aucun n'a atteint l'écart exigé sur S03 (voir
[`FICHE_MODELE.md`](FICHE_MODELE.md), § 6.2).

**Où sont les fichiers de synthèse.** Dans la pièce jointe `synthese_moliere.zip` de la release
`chatbot-moliere-20261116` (voir § 8) : le fichier du modèle publié (`synthetique_v4/...`), le fichier v4 + v5
des essais (`data/stages/chatbot_20261114/synthetique_v6/entrainement/moliere.jsonl.zst`, 27 624 conversations),
les 6 fichiers tenus à l'écart (v4 : 151, 158 et 141 conversations ; v5 : 157, 160 et 132), les traces des
campagnes (`.campagne.json`) et un `LISEZMOI_SYNTHESE.txt`. Le zip fait 4 442 456 octets, sha256
`e0b144a75acafb5187160d37bcdf1099d60218846cc4b081565819e891c4d3bf`, 22 fichiers, licence CC BY 4.0.

### 4.4 Limites à connaître

- Les lecteurs du contrôle de qualité sont des **agents IA** (Claude, modèle fermé d'Anthropic). Ils ont
  seulement noté ; ils n'ont écrit ni corrigé aucune conversation.
- Le générateur a ses **tics**. Dans la campagne v4 : réponses qui commencent par « Certes » ou « Hélas »,
  conseil « consultez un médecin » qui revient chez tous les personnages. Les consignes v5 les corrigent ; il
  en reste d'autres (la maxime « la patience est une vertu », des messages d'utilisateur télégraphiques).
- **Au-delà d'un passage environ par conversation, le modèle apprend la synthèse par cœur.** Mesuré le
  15/11/2026 avec les 9 055 conversations de la v4 : à 3 % de synthèse l'écart entre conversations vues et
  jamais vues est de 4,6 % ; à 5 % il est de 10,2 % ; à 10 % de 23,4 % ; à 15 % de 30,5 %. C'est pour cela que
  la campagne v5 a produit beaucoup plus de conversations. Avec le fichier v4 + v5 (27 624 conversations), le
  16/11/2026, à part égale, le par-cœur a presque disparu : 2,69 % à 5 %, 6,23 % à 10 %, 9,81 % à 15 %. La
  qualité jugée n'a pas progressé de façon démontrée.
- La campagne v5 a été arrêtée à 61 % de ses appels prévus (29 468 sur 48 000) : le reste n'a pas été produit.
- Ces conversations imitent Molière ; ce ne sont pas des textes de Molière. Elles peuvent contenir des
  tournures fautives ou anachroniques.

Une première campagne avec un générateur plus petit (`qwen3:8b`, consignes
`chatbot/prompts_synthetiques.json`) n'a pas été retenue : voir § 6.

## 5. Validation et messages de test

- **Validation chiffrée** : `moliere_val` (3 pièces jamais entraînées). On y mesure le BPB (bits par octet :
  plus c'est bas, mieux le modèle prédit le texte). Le fichier est figé par son empreinte sha256
  `6e7d26d14598666ea6c298104cd19c63fb566a13b9fb3a064ebc7ed06b2c5a6b`.
- **Messages de test** : quatre jeux de messages d'utilisateur écrits à la main (« reglage », « test »,
  « neuf », « neuf2 »), dans [`tools/nuit_moliere.py`](../../tools/nuit_moliere.py). Ils servent aux jugements
  à l'aveugle décrits dans la fiche du modèle. Aucune conversation d'entraînement ne contient un de ces
  messages (comparaison sur les lettres seules, sans tenir compte de la casse).

## 6. Ce qui a été essayé et n'a pas servi

Ces essais sont gardés dans le dépôt pour pouvoir être refaits. **Le modèle publié ne les utilise pas.**

| Essai (14/11/2026) | Résultat, jugé à l'aveugle sur 72 réponses (référence : 44 sans défaut) |
| --- | --- |
| Théâtre nettoyé (traductions, livrets chantés et didascalies retirés ; `tools/nettoyer_affinage.py`) | 34 sans défaut ; BPB de validation moins bon (1,0628 contre 1,0560) |
| Pré-entraînement filtré (tri appris de FineWeb, règles par source ; `tools/nettoyer_pretrain.py`, `tools/trier_fineweb.py`) + théâtre nettoyé | 37 sans défaut |
| Pré-entraînement filtré + théâtre d'origine | 38 sans défaut ; BPB de validation inchangé (1,0572) |
| Synthèse `qwen3:8b` à 2,87 % (essai B, `chatbot/recettes_rtx_v2.json`) | éliminé : le modèle apprend la synthèse par cœur (écart 0,1834, limite 0,10) ; style 61 contre 69 |
| Synthèse `qwen3:8b` à 1,16 % | 45 sans défaut contre 44 : gain non démontré |

Leçon retenue : des données « plus propres » n'ont pas donné de meilleures réponses à cette échelle.

## 7. Licences et droits : ce qui est établi et ce qui ne l'est pas

| Donnée | Ce que le dépôt déclare | Réserve |
| --- | --- | --- |
| Wikipédia | CC BY-SA 4.0 | attribution et partage à l'identique |
| Wikisource | œuvres du domaine public ; transcription CC BY-SA 4.0 | quelques traductions ou auteurs morts depuis moins de 70 ans sont possibles ; non vérifié œuvre par œuvre |
| Project Gutenberg | domaine public **aux États-Unis** | le domaine public en France n'est pas garanti pour toutes les œuvres |
| FineWeb-2, Common Crawl | ODC-By 1.0 pour le jeu de données | le droit d'auteur de chaque page reste à son auteur |
| Wikinews | CC BY 2.5 | attribution |
| Assemblée nationale | Licence Ouverte 2.0 | attribution |
| Molière (kit) | domaine public | — |
| Conversations de synthèse | écrites par un modèle sous Apache 2.0 ; publiées sous CC BY 4.0 (voir [`LICENCES.md`](../../LICENCES.md)) | un texte écrit par une machine n'est pas forcément protégé par le droit d'auteur |

Ce qui n'est **pas** établi :

- la licence sous laquelle les fichiers de tokens et les fichiers de théâtre sont redistribués (aucune n'est
  écrite : seules les licences des sources le sont) ;
- le statut en France de chaque livre de Gutenberg et de Wikisource. Au moins un livre encore protégé en
  France a été repéré dans les données tirées de Gutenberg (*L'Atlantide* de Pierre Benoit, mort en 1962) ;
- l'effet des licences « partage à l'identique » sur un modèle entraîné : la question n'est pas tranchée.

Qui réutilise ces données doit donc vérifier lui-même ce qui s'applique dans son pays.

## 8. Où sont les fichiers

- **Release `chatbot-donnees-20261113`** du dépôt `benassou423/Inteligence-lab-piscine` : 15 pièces jointes,
  4 344 702 490 octets. La liste, avec tailles et empreintes sha256, est dans
  [`chatbot/donnees_rtx.json`](../../chatbot/donnees_rtx.json) ; la commande
  `python -m chatbot.pipeline_rtx telecharger` les télécharge et vérifie leurs empreintes. Par rôle :
  - pré-entraînement (7) : `pretrain_meta.json` (le bilan par source), `pretrain_melange.bin`,
    `pretrain_gutenberg.bin`, `pretrain_wikisource.bin`, `pretrain_wikipedia.bin`, `pretrain_fineweb.bin`,
    `pretrain_wikinews.bin` ;
  - théâtre (3) : `theatre_theatre_train.jsonl.zst`, `theatre_moliere_train.jsonl.zst`,
    `theatre_moliere_val.jsonl.zst` ;
  - dialogues (3), pour le second chatbot (« français courant ») et non pour le modèle Molière :
    `dialogues_romans_train.jsonl.zst`, `dialogues_romans_val.jsonl.zst`,
    `dialogues_theatre_moderne_train.jsonl.zst`. Ils sont tirés par `chatbot/dialogues.py` des romans du corpus
    (Gutenberg et Wikisource) et des scènes en prose de `theatre_train` ;
  - essai (2) : `essai_synthetique_moliere.jsonl.zst` et `essai_synthetique_moderne.jsonl.zst`, deux petits
    échantillons de synthèse écrits par `qwen3:8b`, pour la recette d'essai du lanceur. Ils ne servent pas au
    modèle publié. Le fichier de synthèse de l'essai du § 6 n'est dans aucune release.
- **Release `chatbot-moliere-20261116`** du même dépôt (https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116) :
  `chatbot_moliere_s03.zip` (le modèle publié, 240 087 338 octets), `synthese_moliere.zip` (les conversations de
  synthèse, § 4.3) et l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip` : les deux
  chatbots et leurs rapports). Le zip de synthèse se décompresse à la racine d'un clone.
- **Jeu du concours (« melange »)** : release `ftb-final-20261110` du même dépôt.
- **Conversations de synthèse** : pièce jointe `synthese_moliere.zip` de la release `chatbot-moliere-20261116`.

Les fichiers intermédiaires (corpus nettoyé avant découpage, texte Wikisource de Molière) ne sont pas dans
ces releases : le découpage du théâtre ne peut pas être refait à partir d'un simple clone.

## 9. Ce qu'on ne sait pas

- le nombre d'auteurs et de pièces distincts dans `theatre_train` (pas de champ auteur) ;
- la date exacte de téléchargement des ebooks de Molière et de FineWeb-2 (seuls le commit et le catalogue
  sont notés) ;
- le nombre de documents présents à la fois dans `melange` et dans leur source d'origine ;
- la composition par source du « melange » après les filtres du chatbot (67 494 documents gardés sur 72 612) ;
- l'édition Wikisource exacte de 3 des 6 pièces de Molière ajoutées ;
- pourquoi *Le Ballet du Mariage forcé*, présent dans le kit, n'a donné aucune scène.
