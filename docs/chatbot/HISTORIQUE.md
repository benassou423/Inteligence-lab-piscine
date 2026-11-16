# Ce qui s'est passé : journal du chatbot Molière (13 au 16/11/2026)

Ce document raconte quatre jours de travail sur un chatbot « à la manière de Molière » : ce qui a été essayé, ce que cela a donné, ce qui a été décidé et pourquoi. Les essais qui ont échoué y figurent comme les autres.

- **But** : une conversation basique, avec des réponses à la manière de Molière.
- **Point de départ** : un modèle de 54 749 440 paramètres, entraîné de zéro le 13/11/2026 sur une carte RTX 5090, sans aucun poids venu d'ailleurs.
- **Résultat au 16/11** : le modèle publié est S03, affiné avec 3,02 % de tokens de synthèse (des conversations écrites par un modèle de langue à poids ouverts). Le 15/11, il obtient 100 réponses sans défaut sur 144, contre 82 pour le modèle sans synthèse, dans un jugement à l'aveugle fait par des agents IA (p = 0,044). Le 16/11, sur un jeu de messages neuf, le résultat se confirme : 96 contre 73. Quatre essais à 3, 5, 10 et 15 % de synthèse (conversations des campagnes v4 et v5) ont été jugés le 16/11 : aucun n'atteint l'écart exigé, **S03 reste le modèle retenu**. Ces deux jugements sont des gains probables, pas des certitudes.
- **Comment** : le travail a été mené par l'étudiant avec un assistant IA (Claude Code), qui a écrit le code, lancé les calculs et tenu le journal de travail. Les réponses du modèle ont été jugées par des agents IA, à l'aveugle, jamais par des personnes.
- **Fiches** : le modèle dans [`FICHE_MODELE.md`](FICHE_MODELE.md), les données dans [`FICHE_DONNEES.md`](FICHE_DONNEES.md), l'explication dans [`docs/comprendre/chatbot_moliere.md`](../comprendre/chatbot_moliere.md), les licences dans [`LICENCES.md`](../../LICENCES.md).

Ce récit est tiré du journal de travail brut, qui n'est pas publié. Plusieurs heures de ce journal sont des estimations : on donne ici l'ordre des faits, rarement l'heure.

Mots utilisés.

- **Pré-entraînement, affinage** : le modèle apprend d'abord le français sur un grand volume de textes, puis il est affiné sur du théâtre.
- **Tailles S et M** : deux tailles de modèle mesurées par le lanceur ; M, la plus grande des deux, a été retenue (54 749 440 paramètres).
- **BPB** : bits par octet sur 3 pièces de Molière jamais entraînées ; plus bas = le modèle prédit mieux le texte.
- **A0** : le modèle affiné sans synthèse le 13/11, témoin de tous les essais.
- **Synthèse** : conversations écrites par un modèle de langue à poids ouverts, ajoutées à l'affinage.
- **B, C, D, E, F, S03, S05, S10, S15** : noms des modèles essayés contre A0 ; le nombre après S est la part de synthèse visée, en pour cent.
- **v6 à 3, 5, 10 et 15 %** : les quatre essais du 16/11. Même affinage que S03, mais avec la synthèse des campagnes v4 et v5 réunies (27 624 conversations). Leurs recettes sont `chatbot/recettes_rtx_v6_s03.json`, `_s05`, `_s10` et `_s15`.
- **« melange »** : nom d'une des six sources du pré-entraînement, le jeu de données du concours.
- **Jeux de messages** (« reglage », « test », « neuf », « neuf2 ») : listes de messages posés aux modèles. Pour un jugement, chaque jeu est joué 3 fois avec un tirage au hasard différent (une « graine »), soit 72 réponses par modèle pour « reglage » et « test », 144 pour « neuf » et « neuf2 ».
- **Français, rapport au message, style, sans défaut** : notes des juges sur une réponse, parfois abrégées F, R, S, T. Une réponse est « sans défaut » si chacun des trois critères est validé par la majorité des 3 juges. Attention : la lettre F désigne aussi un essai du 14/11, et la lettre S une taille de modèle.
- **Par-cœur** : le modèle récite les conversations de synthèse au lieu d'apprendre à converser. On le mesure par l'écart de BPB entre un fichier de synthèse vu à l'entraînement et un fichier de même fabrication jamais vu.
- **Garde** : limite fixée d'avance qu'un candidat ne doit pas franchir.
- **Candidates, choix de la réponse (ou décodage)** : pour chaque message, le modèle écrit plusieurs réponses candidates et n'en garde qu'une. Le choix de la réponse, appelé aussi décodage, est la façon de retenir cette réponse.
- **Ancrage** : façon de choisir qui garde la candidate qui dépend le plus du dernier message, et non de ce que le modèle a déjà dit.
- **Radotage** : le modèle répète ses propres mots d'un tour à l'autre. La mesure exacte est donnée avec le tableau du 14/11.
- **p** : probabilité qu'un écart au moins aussi grand vienne du hasard (test des signes, réponse par réponse).

## En bref

| Date | Ce qui a été fait | Résultat | Décision |
| --- | --- | --- | --- |
| 13/11 | Mesure du débit des tailles S et M | S : 545 588 tok/s ; M : 340 354 tok/s ; budget de 3 h | taille M |
| 13/11 | Pré-entraînement de zéro | 100,0 min ; BPB 3,87695 → 1,12985 | base de tous les essais |
| 13/11 | Affinage Molière sans synthèse (A0) | BPB 1,05608 au pas 1 000 sur 2 747, puis il remonte | A0 devient le témoin |
| 13 et 14/11 | Première synthèse, petit générateur (`qwen3:8b`) | 7 120 appels, 1 305 conversations après filtres ; contrôle de qualité 38 sur 60 (seuil 48) | poursuivie à titre d'essai |
| 14/11 | Essais B (2,87 % de synthèse) et C (1,16 %) | B : par-cœur et style −8 ; C : rapport au message +8 sur 144 (seuil 14) | A0 reste |
| 14/11 | Choix de la réponse : ancrage sur le dernier message, 32 candidates | rapport au message 96 → 114 sur 144 ; français 125 → 120 ; style 137 → 133 | adopté |
| 14/11 | Audit des données réelles par lecture | 0 fuite de la validation dans le théâtre ; FineWeb : 10 extraits graves sur 40 | tenter un nettoyage |
| 14/11 | Données nettoyées : essais D, F, E | 34, 37 et 38 réponses sans défaut sur 72, contre 44 pour A0 | A0 reste |
| 14/11 | C rejoué avec le nouveau choix de réponse | 45 sans défaut contre 44 ; français −7 | non retenu |
| 14/11 | Nouvelles consignes (v4) et générateur plus gros | pilote : 205 conversations gardées sur 300 ; contrôle 48 sur 60 (seuil 48) | campagne lancée |
| 15/11 | Fichier de synthèse v4 | 13 660 appels, 9 055 conversations ; contrôle 113 sur 120 (seuil 96) | accepté |
| 15/11 | Essais à 5, 10 et 15 % de synthèse | écart de par-cœur 10,2 %, 23,4 %, 30,5 % (limite 10 %) | écartés ; un tour de réglage à 3 % |
| 15/11 | S03 (3,02 %) jugé à l'aveugle | par-cœur 4,6 % ; T 100 contre 82 sur 144 (p = 0,044) | S03 déclaré « meilleur » ; publication laissée à l'étudiant |
| 15/11 | Campagne de synthèse v5 | v5a arrêtée après 3 minutes ; v5b : contrôle sur les premières conversations, 52 sur 60 (seuil 48) | génération v5b poursuivie |
| 15 et 16/11 | Préparation de l'ouverture du modèle | audit, deux fiches, documents mis à jour | Apache 2.0 (code), CC BY-SA 4.0 (poids), CC BY 4.0 (synthèse) |
| 16/11 | Arrêt volontaire de la génération v5b à 61 % | 29 468 appels faits sur 48 000 ; 21 830 conversations v5 gardées à la génération (avant filtres) ; 27 624 après filtres, v4 comprise | passer aux essais finaux |
| 16/11 | Contrôle de qualité du fichier v4 + v5 | 109 conversations sans défaut sur 120 (seuil 96) | fichier accepté |
| 16/11 | Quatre essais (v6 à 3, 5, 10 et 15 %) jugés à l'aveugle sur le jeu « neuf2 » | T (réponses sans défaut sur 144) : 93, 101, 88 et 105 contre 96 pour S03 ; les quatre passent les gardes | aucun n'atteint +12 : S03 reste le modèle retenu |
| 16/11 | S03 contre A0 sur un jeu neuf | 96 contre 73 sur 144 (p = 0,0095) | gain confirmé |
| 16/11 | Installation de S03 comme modèle publié | poids identiques octet pour octet ; export et évaluation refaits, aucun réentraînement | modèle prêt à publier |
| 16/11 | Relecture du dépôt, preuves des jugements publiées, trois bogues corrigés | audit de 303 fichiers, 5 fichiers retirés | dépôt nettoyé |
| 16/11 | Publication de la release `chatbot-moliere-20261116` | modèle, conversations de synthèse, archive des résultats | modèle publié |

## 13/11 : le modèle de base et le témoin

**Cadre fixé avant tout calcul.** La synthèse est autorisée, 15 % au plus. Le générateur doit être à poids ouverts, les consignes publiées, la part bornée et déclarée ([`FICHE_DONNEES.md`](FICHE_DONNEES.md), § 4.1). Rien n'est enregistré dans git ni publié sans l'accord de l'étudiant.

**Vérifications.** L'autotest fait tourner toute la chaîne sur un modèle minuscule (coupure simulée, reprise, affinages, exports) : réussi en 93,5 s, 15 contrôles sur 15 (`experiments/chatbot_rtx/20261113_1956_M/lanceur/autotest.json`).

**Choix de la taille.** On cherchait la plus grande taille qui tient dans le budget de 3 heures donné par l'étudiant.
- Mesure : taille S 545 588 tok/s (1,52 h estimée), taille M 340 354 tok/s (2,6 h estimées) (`experiments/chatbot_rtx/20261113_1956_M/lanceur/mesures.json`). Décision : taille M.
- `torch.compile` a échoué (Triton absent). Le lanceur s'est replié sans compilation : même calcul.

**Pré-entraînement.** 100,0 minutes. Le BPB passe de 3,87695 à 1,12985. Le meilleur point est le dernier pas (30 518).

**Affinage sans synthèse (A0).** On a laissé finir le lancement en cours, sans synthèse, pour avoir un témoin avant de changer quoi que ce soit. Durée totale du lancement : 1,96 h (`experiments/chatbot_rtx/20261113_1956_M/bilan.json`).
- Le BPB passe de 1,12985 à 1,05608 au pas 1 000 sur 2 747, puis remonte : l'affinage sur le théâtre sur-apprend vite. C'est pourquoi le lanceur garde le meilleur point mesuré, pas le dernier : ce choix était déjà dans la recette.
- Second affinage du même lancement, « français moderne » : BPB 1,15956 → 1,04751.

**Première lecture de A0.** A0 passe 9 critères de forme sur 9. Mais une lecture rapide de 48 réponses trouve environ 29 réponses en rapport avec le message. Cette lecture a été faite par l'assistant, non à l'aveugle : elle ne vaut pas un jugement. Ce qu'elle a appris : les mesures automatiques ne disent pas si une réponse est bonne ; il faut juger le contenu, et à l'aveugle.

**Préparer la synthèse.**
- Défaut repéré en relisant le code : dans la recette d'origine, l'étape de synthèse séparée garde son meilleur point mesuré sur du vrai Molière, point de départ compris. Elle peut donc n'avoir aucun effet. Correction : un fichier de recettes nouveau, `chatbot/recettes_rtx_v2.json`, mêle la synthèse à l'étape 1. La recette d'origine n'est pas modifiée.
- Deux bornes. En documents : 15 % donnent 6 034 conversations, d'où la cible de 5 000 au plus. En tokens réellement vus : le poids de tirage multiplie la part, il est donc limité pour que la synthèse reste sous 13 % des tokens vus.
- Un fichier de 300 conversations est tenu à l'écart, jamais entraîné, pour mesurer plus tard si le modèle apprend la synthèse par cœur.
- Générateur : `qwen3:8b`, lancé en local. À l'arrêt du soir : 2 309 conversations gardées pour 4 855 appels. La cible de 5 000 n'est pas atteinte.

**Deux constats notés ce soir-là.** (1) `theatre_train` contient 5 scènes d'un pastiche de Guignol où parle HARPAGON, personnage d'une pièce de validation. Le fichier est figé par son empreinte : la limite est déclarée plutôt que corrigée ([`FICHE_DONNEES.md`](FICHE_DONNEES.md), § 3.4, où elle est comptée en répliques : 36). (2) Le rapport de vérification du lanceur contenait des informations sur le poste et entrait dans l'archive des résultats (suite dans « Erreurs et incidents »).

## 14/11 : trois essais négatifs, un gain

### Matin : la première synthèse (négatif)

**Ce qu'on cherchait.** Savoir si des conversations écrites par un petit générateur ouvert améliorent le rapport au message.

**Les données.** Après fusion et filtres anti-doublons, 3 594 conversations brutes (7 120 appels, 20 426 candidates) n'en laissent que 1 305, soit un taux de rejet global de 0,9361. Les compteurs tenus graine par graine surestimaient d'un facteur d'environ 2,7 : ils ne voient pas les doublons entre graines. Seul le compte après fusion est vrai. Le plafond de 5 000 conversations est déclaré hors d'atteinte.

**Contrôle de qualité : raté.** 60 conversations tirées au hasard, 3 lecteurs IA indépendants, seuil fixé d'avance à 48 sans défaut. Résultat : 38. Le défaut principal est le français (44 conversations correctes sur 60). L'étudiant ayant demandé d'avancer sans lui, l'assistant décide de continuer à titre d'essai et l'écrit, avec une règle de plus sur le français et le modèle sans synthèse livré par défaut.

**Essai B : synthèse au poids 5** (`chatbot/recettes_rtx_v2.json`), soit 2,87 % des tokens vus et 13,9 passages par conversation sur l'étape entière (5,1 au pas 1 000, le point gardé).
- BPB sur Molière : 1,05712 contre 1,05602 pour A0 (dans la garde de 0,010).
- BPB sur la synthèse entraînée 0,56491, sur la synthèse jamais vue 0,74834 : écart de 0,1834, au-dessus de la limite de 0,10 fixée d'avance. Le modèle apprend ces conversations par cœur.
- Jugement à l'aveugle (72 réponses) : rapport au message 48, comme A0 ; style 61 contre 69, soit −8 pour une tolérance de 4. B est éliminé.

**Essai C : synthèse au poids 2** (`chatbot/recettes_rtx_v3.json`), soit 1,16 % des tokens vus. C'est le seul tour de réglage prévu par la règle (« par-cœur : baisser le poids »).
- BPB sur Molière 1,0558 ; écart de par-cœur 0,0659, sous la limite.
- Jeu de réglage (72 réponses) : A0 F 63, R 48, S 69 ; C F 65, R 53, S 70. Sans défaut : 44 pour A0, 49 pour C. Le rapport au message gagne 5 pour un seuil de 7 : selon la règle de la veille, la synthèse n'est pas déclarée utile sur ce jeu.
- Le jeu de test est quand même joué une fois, sous une règle de verdict écrite après les notes du réglage et avant ce jeu. Sur 144 réponses (réglage + test) : rapport au message 94 → 102, soit +8 pour un seuil de 14 ; style 135 → 142 ; français 126 et 126 ; sans défaut 79 → 92.
- Décision : le gain sur le rapport au message n'est pas démontré. Le modèle livré reste A0, sans synthèse.

**Ce que l'essai a appris.** Avec 1 305 conversations seulement, 2,87 % de synthèse suffisent à faire réciter le modèle. Le petit générateur écrit un français fautif dans plus d'une conversation sur quatre. Écrire la règle avant de regarder les notes oblige à accepter un résultat négatif.

### Midi : le choix de la réponse (le gain du jour)

**Ce qu'on cherchait.** Améliorer le rapport au message sans réentraîner, en changeant la façon de choisir une réponse parmi les candidates. La règle est écrite avant toute sortie, avec les mêmes seuils que le matin (+7 sur 72 pour être candidat, +14 sur 144 pour être adopté, français et style à −4 au plus).

**Trois variantes sur A0.** V1 : ancrage sur le dernier message, 16 candidates. V2 : ancrage, 32 candidates. V3 : 32 candidates sans ancrage.

| Jeu de réglage, 72 réponses | Français | Rapport au message | Style | Radotage (sur 45 réponses) |
| --- | ---: | ---: | ---: | ---: |
| A0, ancien choix | 62 | 47 | 70 | 28 |
| V1 | 61 | 49 | 68 | 9 |
| V2 | 61 | 55 | 68 | 11 |
| V3 | 60 | 44 | 68 | 20 |

Radotage : parmi 45 réponses des tours 2 et suivants, nombre de celles qui reprennent un mot de 5 lettres ou plus déjà dit par le bot et absent du message courant ; plus bas = mieux. Seule V2 est candidate (+8 sur le rapport au message). Sur les 144 réponses (réglage + test) : rapport au message 96 → 114 (33 réponses meilleures, 15 moins bonnes, p = 0,0133) ; français 125 → 120 ; style 137 → 133. Aux tours 2 et suivants d'une conversation : 29 → 43 sur 54. Sur le jeu de test, le français recule de 4, exactement la limite fixée. Coût : 0,81 s par réponse contre 0,47 s. **Décision** : V2 est adoptée. Le code est dans `chatbot/ancrage.py`, ses tests dans `tests/test_chatbot_ancrage.py`. Sur le jeu de réglage, ni l'ancrage seul (+2) ni les 32 candidates seules (−3) n'atteignent le seuil de +7 ; seule leur combinaison l'atteint.

### Après-midi : les données nettoyées (négatif)

**Audit des données réelles.**
- Fuites : aucun passage de 12 mots de la validation dans les 33 396 + 790 scènes d'entraînement. Dans le pré-entraînement : 22 zones de citation isolée (603 tokens), aucune copie de pièce.
- Lecture de 40 extraits par source par des agents IA (propre / défaut léger / défaut grave) : Wikisource 31 / 9 / 0 ; Gutenberg 31 / 6 / 3 ; « melange » 26 / 11 / 3 ; Wikipédia 21 / 17 / 2 ; FineWeb 17 / 13 / 10 ; Wikinews 3 / 32 / 5.
- Le modèle relit plus de deux fois les petites sources (Wikinews : 2,933 passages ; « melange » : 2,128) et moins d'une fois Wikipédia, Wikisource et FineWeb (Wikisource : 0,72). Gutenberg, tiré plus souvent, est relu 1,2 fois.

**Règle écrite avant tout entraînement.** Deux candidats, entraînés à part sans toucher au modèle en place. D : théâtre d'affinage nettoyé (`chatbot/recettes_rtx_v4a.json`). F : pré-entraînement filtré + théâtre nettoyé (`chatbot/recettes_rtx_v4.json`). Un candidat passe au jeu de test s'il gagne au moins 6 réponses sans défaut sur 72, sans reculer de plus de 4 sur un critère.

**Un troisième modèle, E, hors de cette règle.** E (pré-entraînement filtré + théâtre d'origine, `chatbot/recettes_rtx_v4e.json`) n'était prévu que pour sa mesure de BPB. Son jugement à l'aveugle est un essai complémentaire, annoncé après le verdict négatif de D et F et avant toute réponse de E. La règle est la même, avec une issue de plus (« non inférieur » : au plus 4 réponses sans défaut de moins que A0, et au plus 4 de moins sur chaque critère). C'est une comparaison de plus contre A0 : à lire avec cette réserve.

**Le tri de FineWeb.**
- Lecture de 320 documents par des agents IA : 122 propres, 111 défauts légers, 87 défauts graves (27,2 %).
- Des règles simples ne les repèrent pas. Un classifieur de mots non plus (51 graves retrouvés sur 87).
- Méthode retenue : `qwen3:8b` note 12 000 documents sans rien écrire, puis une sonde linéaire apprend ces notes sur les activations du modèle de base. Au seuil retenu, elle écarte 71 documents graves sur 87 et 4 propres sur 122 (`chatbot/sonde_fineweb.npz.rapport.json`, `tools/trier_fineweb.py`).
- Filtrage complet : FineWeb perd 31,54 % de ses tokens ; l'ensemble des 6 sources passe de 2 114 055 586 à 1 917 228 299 tokens (9,31 % écartés) (`tools/nettoyer_pretrain.py`, `chatbot/regles_pretrain_v4.json`).
- Contrôle sur un tirage neuf, seuils écrits avant lecture : parmi 80 morceaux gardés de FineWeb, 3 graves (27 % avant filtrage) ; parmi 80 écartés, 7 propres (seuil : 8 au plus). Accepté.

**Le nettoyage du théâtre** (`tools/nettoyer_affinage.py`).
- Classement des 969 œuvres par des lecteurs IA, sur une fiche par œuvre (titres, personnages, 3 extraits), avec deux lecteurs indépendants pour toute mise à l'écart : 606 œuvres de théâtre français (59,7 % des mots), 170 traductions (29,6 %), 92 qui ne sont pas du théâtre, 88 livrets chantés, 12 en langue non standard, 1 illisible. 363 œuvres écartées (`chatbot/exclusions_theatre_v2.json`).
- Contrôle n° 1 : raté pour Molière (16,5 répliques en défaut sur 150, en moyenne de deux lecteurs ; seuil 7). Toutes les parenthèses étaient retirées, or Molière fait dire des incises entre parenthèses à ses personnages.
- Contrôle n° 2 : Molière passe (1 et 1) ; le reste du théâtre rate d'un demi-point (7,5).
- Contrôle n° 3 : les deux fichiers passent (2 et 2 sur 150). Ce troisième tour est un écart à la règle, qui disait « corriger, refaire une fois » ; l'écart est écrit dans le journal avec son motif.
- Résultat : `theatre_train` passe de 33 396 à 19 153 scènes, `moliere_train` de 790 à 778.

| Essai | BPB sur la validation d'origine | Sans défaut sur 72 | A0 dans le même lot |
| --- | ---: | ---: | ---: |
| D : théâtre nettoyé | 1,0628 | 34 | 44 |
| F : pré-entraînement filtré + théâtre nettoyé | 1,0628 | 37 | 44 |
| E : pré-entraînement filtré + théâtre d'origine | 1,0572 | 38 | 44 |

Le BPB de A0 est 1,0560. Avec le théâtre nettoyé, le meilleur point arrive au pas 500 au lieu de 1 000 : le modèle sur-apprend deux fois plus tôt. Le rapport au message recule dans les trois essais (47 ou 48 contre 56). Aucun écart pris seul n'est significatif (p de 0,09 à 0,36), sauf le style de D (p = 0,039).

**Décision et leçon.** Ni D ni F ne passe la première porte. E n'est ni supérieur ni « non inférieur » (38 pour un minimum de 40) : il est écarté. Le modèle livré reste A0. Les outils et les listes restent dans le dépôt, avec la mention « essai négatif ». Ce que l'essai a appris, avec prudence : retirer 43 % du théâtre d'affinage coûte plus que ne rapporte le retrait des traductions et des indications de scène ; retirer 31,5 % de FineWeb enlève du spam mais aussi du français d'aujourd'hui, celui des messages de l'utilisateur. « Plus propre » n'a pas donné « meilleures réponses ».

**C rejoué avec le nouveau choix de réponse.** 45 réponses sans défaut contre 44 pour A0 ; français 55 contre 62, soit −7 pour une tolérance de 4. Non retenu. Le gain du matin ne se retrouve pas.

### Soir : préparer une vraie campagne de synthèse

**Un jeu de messages neuf.** Les jeux « reglage » et « test » ont servi à régler le choix de la réponse et à cinq comparaisons : ils ne sont plus neutres. Un jeu « neuf » est écrit avant toute réponse d'un modèle de la campagne (`tools/nuit_moliere.py`).

**Pourquoi 94 % de rejets le matin.** La cause est trouvée dans le code, pas dans le générateur : un filtre de quasi-doublon vite saturé, 176 sujets seulement, des exemples recopiés par le générateur, trois conversations par appel dans le même moule. Changer de générateur seul n'aurait pas suffi.

**Ce qui change.**
- Générateur : `gemma4:31b-it-q4_K_M` (poids ouverts, Apache 2.0), téléchargé après accord explicite de l'étudiant. Le téléchargement échoue 6 fois ; le fichier est récupéré à la même source par un autre moyen et son empreinte est vérifiée contre celle du registre.
- Consignes v4 : une conversation par appel, pilotée par une fiche (651 notes de thème, 51 types de message, 20 voix d'utilisateur ; 9 000 fiches) (`chatbot/axes_consignes_v4.json`, `chatbot/prompts_synthetiques_v4.json`, `tools/composer_consignes_v4.py`). Deux relectures contradictoires : 17 constats, tous corrigés.
- Trois seuils de filtre assouplis, et c'est déclaré : formules répétées 3 → 6 pour mille ; seuils d'écho 0,6 → 0,7 et 0,4 → 0,5.

**Deux pilotes.** Pilote n° 1 : 202 conversations gardées sur 300, les seuils automatiques sont tenus. Mais en lisant des exemples, un tic apparaît : le mot « étrange » dans 23 % des réponses, dans 103 conversations sur 202. Cause : une phrase fixe de la consigne. Ces 202 conversations sont écartées. Leçon : des seuils tenus ne dispensent pas de lire. Pilote n° 2, après une correction unique des consignes : 205 gardées sur 300. Contrôle par 3 lecteurs IA : 48 conversations sans défaut sur 60 pour un seuil de 48. Il passe, à la limite (38 le matin avec le petit générateur).

**Règle de la campagne, écrite avant la génération de nuit.** Candidats à 5, 10 et 15 % de synthèse. Un candidat est « meilleur » s'il gagne au moins 12 réponses sans défaut sur 144 sans reculer de plus de 6 sur un critère, avec deux gardes : BPB sur Molière à +0,010 au plus, écart de par-cœur sous 10 %. Le modèle en place n'est pas remplacé d'office : publier un modèle entraîné avec de la synthèse reste la décision de l'étudiant. La génération de nuit est lancée (3 flux de 7 000 appels).

## 15/11 : un gain avec 3 % de synthèse, pas au-delà

**Le fichier de synthèse v4.** La génération de nuit s'est arrêtée à 1 h 14 (redémarrage brutal du poste). Elle n'est pas relancée : la fin des travaux était alors fixée au 16/11.
- 13 660 appels, 10 170 conversations gardées (74 %) ; 9 538 après retrait des fuites vers la validation et des doublons ; 9 055 (897 109 tokens) après retrait de toute conversation dont un message est un des 96 messages de test.
- Contrôle de qualité : 113 conversations sans défaut sur 120 (seuil 96). Les lecteurs relèvent des tics hors grille : « Certes » et « Hélas » en tête de réponse, le conseil « consultez un médecin ».

**Essais S05, S10, S15 : trop de synthèse (négatif).** Trois recettes identiques à celle de A0, à une différence près : le poids du fichier de synthèse (`chatbot/recettes_rtx_v5_s05.json`, `_s10.json`, `_s15.json`).

| Essai | Part visée | BPB Molière | BPB synthèse vue | BPB synthèse jamais vue | Écart |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 (référence) | 0 % | 1,0560 | 1,1971 | 1,1906 | — |
| S03 | 3 % | 1,0557 | 0,7376 | 0,7733 | 4,6 % |
| S05 | 5 % | 1,0565 | 0,6508 | 0,7245 | 10,2 % |
| S10 | 10 % | 1,0582 | 0,5300 | 0,6922 | 23,4 % |
| S15 | 15 % | 1,0591 | 0,4777 | 0,6878 | 30,5 % |

S05, S10 et S15 dépassent la garde de par-cœur ; S05 de 0,2 point. Aucun des trois ne peut être déclaré « meilleur ». La règle de la campagne, écrite la veille, ne prévoyait pas de tour de réglage : un candidat qui dépasse la garde est écarté. L'assistant a repris la clause de la première synthèse (« par-cœur : baisser le poids »), déjà utilisée une fois pour l'essai C. C'est S03 (`chatbot/recettes_rtx_v5_s03.json`), ajouté après avoir vu les gardes de S05, S10 et S15, mais avant tout jugement à l'aveugle. Il est aussi écrit, avant de voir une seule note, que S05, S10 et S15 seront jugés pour information seulement.

**Ce que l'essai suggère.** Lecture de l'assistant, que la campagne v5 doit vérifier : le par-cœur suivrait le nombre de passages sur chaque conversation. Au pas 1 000, S03 a fait 1,1 passage et généralise encore (écart 4,6 %) ; S05 en a fait 1,8 et commence à réciter (10,2 %). Mais les quatre essais utilisent le même fichier de synthèse : la part et le nombre de passages montent ensemble, l'essai ne les sépare pas. La campagne v5 vise plus de conversations, donc moins de passages à part égale.

**Jugement à l'aveugle sur le jeu « neuf ».** 720 réponses mélangées (144 par modèle), 3 juges IA par lot, aucune note manquante.

| Modèle | Français | Rapport au message | Style | Sans défaut |
| --- | ---: | ---: | ---: | ---: |
| A0 | 119 | 110 | 124 | 82 |
| **S03** | **124** | **114** | **138** | **100** |
| S05 (pour information) | 128 | 116 | 137 | 97 |
| S10 (pour information) | 121 | 108 | 137 | 88 |
| S15 (pour information) | 123 | 121 | 141 | 100 |

- Réponse par réponse, S03 est meilleur 45 fois et moins bon 27 fois (test des signes, p = 0,044). Les trois juges vont dans le même sens (82 → 100 à la majorité ; 83 → 102, 83 → 101, 80 → 98 par juge).
- Le gain vient des conversations suivies (38 → 57 sans défaut sur 72), pas des messages isolés (44 → 43).
- Part de synthèse réellement vue par S03 : 2 720 768 tokens sur 90 013 696, soit 3,02 %.

**Verdict.** S03 passe les seuils de la règle écrite la veille : +18 sans défaut (seuil 12), aucun critère en recul, BPB 1,0557, par-cœur 4,6 %. Le journal le déclare « meilleur ». S03 lui-même n'était pas un candidat prévu par cette règle : c'est le tour de réglage décrit plus haut. A0 reste installé ; la décision de publier revient à l'étudiant. Les réserves sont écrites avec le résultat ([`FICHE_MODELE.md`](FICHE_MODELE.md), § 6.3) : un seul jeu de test, des thèmes proches de ceux des consignes, des juges IA, un bruit de l'ordre de 10 réponses (S10 à 88 entre 97 et 100), des tics pris au générateur (« Certes » en tête de 28 réponses sur 240, « Hélas » de 24, contre 0 et 1 pour A0), et des non-sens qui restent (« Je m'embarte avec plaisir »).

**Nouvelle consigne de l'étudiant.** L'échéance du 16/11 est levée. La synthèse continue si elle améliore le modèle. À la fin, tout doit être poussé sur le dépôt et le modèle rendu ouvert et inspectable (fait le 16/11 : voir la dernière section).

**Campagne v5 : plus de conversations plutôt que plus de passages.** Pour 10 % de synthèse à environ 1,2 passage par conversation, il faut environ 27 600 conversations ; pour 15 %, environ 41 500.
- Un jeu de test « neuf2 » et la règle de la suite sont écrits avant de regarder la moindre donnée v5 : le témoin devient S03 ; un candidat le remplace s'il gagne au moins 12 réponses sans défaut sur 144 sans reculer de plus de 6 sur un critère ; mêmes gardes ; plafond de 15 %.
- Consignes v5a : la v4 plus des retouches contre les tics (plus de « Certes », « Hélas », « Certainement » en tête ; plus de renvoi au médecin). Le pilote passe le contrôle de justesse (49 sur 60), mais les trois lecteurs relèvent un défaut nouveau : des adverbes plaqués en tête de réponse (« Prudemment ! », « Probablement, »). Interdire trois adverbes poussait vers leurs synonymes. La génération v5a est arrêtée après 3 minutes et jamais utilisée.
- Consignes v5b (`chatbot/axes_consignes_v5.json`, `chatbot/prompts_synthetiques_v5.json`, 26 000 fiches) : une règle positive, « une réponse commence par ce qu'elle a à dire ». La génération est relancée (3 flux de 16 000 appels) et le contrôle porte sur ses premières conversations : 52 sur 60 (seuil 48). Les adverbes plaqués ont disparu ; la génération continue et les consignes ne sont plus retouchées.
- Le poste redémarre brutalement à 19 h 55 : 14 585 appels faits sur 48 000, 11 103 conversations gardées, rien de perdu. La génération est relancée pour le reste.

**Préparer l'ouverture du modèle** (sur le processeur, la carte restant à la génération).
- Audit par des agents IA : le dépôt était public mais sans licence ; les poids n'étaient publiés nulle part ; le choix de réponse utilisé pour les jugements n'était pas dans le modèle exporté ; il n'y avait ni fiche de modèle ni fiche des données.
- Corrections : l'ancrage devient une clé du `chat_config.json` du modèle (`anchor_mu`, dans `chatbot/generate.py`), pour que chacun obtienne les mêmes réponses ; le rapport d'un modèle déclare sa part de synthèse mesurée sur les tokens vus ; les outils n'écrivent plus de chemin absolu.
- Une relecture indépendante (34 agents IA) confirme 5 défauts, corrigés aussitôt. Elle relève aussi 27 passages de documentation périmés, traités ensuite, et des points sur des fichiers déjà publiés, non corrigés et laissés à la décision de l'étudiant.
- Les deux fiches sont écrites fait par fait : 208 faits avec leur preuve, dont 13 corrigés et 46 ajoutés par une contre-lecture.

## 16/11 (nuit) : licences, documents, arrêt de la génération

- **Licences**, sur décision de l'étudiant : Apache 2.0 pour le code, CC BY-SA 4.0 pour les poids, CC BY 4.0 pour les conversations de synthèse. `LICENSE` et [`LICENCES.md`](../../LICENCES.md) disent aussi ce que ces licences ne couvrent pas : le kit du concours et les textes d'entraînement.
- **Documents périmés** : 8 fichiers mis à jour par des rédacteurs IA, puis relus par des contre-lecteurs chargés de réfuter. Environ 260 affirmations vérifiées, 26 défauts signalés, 24 corrigés, 2 laissés à la décision de l'étudiant.
- **Poids** : publiés dans une release du dépôt, choix de l'étudiant (voir la dernière section : release `chatbot-moliere-20261116`).
- **Génération v5b** : arrêtée volontairement à 61 % (29 468 appels faits sur 48 000). Au moment de la décision, à 1 h 01, le compteur indiquait 21 768 conversations v5 gardées : c'est un compte avant filtres. Le compte net, fait après l'arrêt : 21 830 conversations gardées à la génération ; 20 394 après fusion et refiltrage ; 19 433 gardées à l'union avec les 9 538 conversations de la campagne v4 (total 28 971) ; 27 624 après retrait de celles qui contenaient un message de test. Le fichier d'entraînement compte donc 27 624 conversations, v4 comprise, et non 21 768 + 9 055. Les raisons de l'arrêt sont dans la section suivante.

## Erreurs et incidents

**Erreurs de méthode ou de manipulation, et leur correction.**
- *Étape de synthèse sans effet possible* (recette d'origine). Corrigé par une recette nouvelle qui mêle la synthèse à l'étape 1.
- *Compteurs trop optimistes* : les conversations « gardées » étaient comptées graine par graine, sans voir les doublons entre graines (facteur d'environ 2,7). Correction : compter après fusion.
- *Contrôle de qualité raté puis essai poursuivi* (38 sur 60, seuil 48). Ce n'est pas caché : la décision est écrite, présentée comme un essai, avec le modèle sans synthèse livré par défaut.
- *Un message de test laissé dans la première synthèse.* Depuis la campagne v4, toute conversation contenant un message de test est retirée (483 pour la v4).
- *Parenthèses de Molière supprimées* par une règle de nettoyage trop large. Vu au contrôle par lecture, avant tout entraînement ; règle corrigée.
- *Écart à une règle écrite d'avance* : un troisième tour de corrections du nettoyage au lieu de deux. Déclaré avec son motif.
- *Tic créé par une consigne* (« étrange »), puis *défaut créé par la correction d'un tic* (adverbes plaqués de la v5a). Dans les deux cas, les données touchées ont été écartées.
- *Trois erreurs de manipulation rattrapées* : une relance avec les anciennes consignes, la recomposition des consignes v5 ayant échoué sans être vue (arrêtée une minute plus tard) ; un contrôle lancé sur le mauvais fichier (arrêté avant tout résultat) ; une option mal comprise, le plafond d'appels du générateur valant par exécution et non au total (vu avant la relance du 15/11, faite avec le reste exact).
- *Réglages de génération et de calcul* : 7 processus sur un seul serveur de génération ne débitaient pas plus qu'un seul (0,56 appel/s), il a fallu 3 serveurs (environ 1,08 appel/s) ; un cache laissé à sa valeur par défaut a saturé la mémoire vive ; un second serveur aux réglages par défaut a fait tomber le débit du nouveau générateur à 5,9 tokens/s (95 à 105 après correction) ; le premier filtrage du pré-entraînement saturait la mémoire de la carte, les lots ayant tous des longueurs différentes. Tous corrigés avant de continuer.
- *Informations sur le poste dans un rapport* : le rapport de vérification du lanceur en contenait, et il entrait dans l'archive des résultats. La première correction ne protégeait que les rapports futurs ; la relecture indépendante l'a vu. Le lanceur assainit maintenant tout rapport avant publication (`chatbot/pipeline_rtx.py`, fonctions `rapport_publiable` et `assainir_verification`) et il est décidé de ne jamais publier les anciennes archives.
- *Part de synthèse non déclarée* : le rapport de S03 disait `part_synthetique : null` alors que 3,02 % des tokens vus étaient de la synthèse. Corrigé dans le lanceur.
- *Heures estimées* : plusieurs heures du journal ont été écrites par estimation, puis relues dans les journaux de calcul. L'ordre exact de quelques entrées n'est pas sûr à quelques minutes près.
- *Deux défauts de code notés et non corrigés quand le journal s'arrête* dans `chatbot/pipeline_rtx.py` : `processus_vivant` peut lever une erreur non attrapée quand il reste un fichier de suivi d'une étape ; `versions` et `environnement` ne prévoient que l'absence de PyTorch, pas l'échec de son chargement. Le 16/11, un autre défaut de ce fichier a été corrigé (voir la dernière section) ; on ne dit pas ici que ces deux-là le sont.

**Incidents du poste.**
- Le 14/11 au matin, PyTorch ne se chargeait plus sur le poste, pour une cause extérieure au projet. L'affinage et les mesures ont été suspendus ; la génération de texte a continué sans PyTorch (fidélité vérifiée ensuite : 0 différence sur 18 979 paires de répliques). L'étudiant a réglé le problème de son côté. Un premier essai de reprise a encore échoué ; le travail n'a repris qu'après un import réussi.
- Le 14/11 vers 8 h 45, le poste a redémarré et a coupé la première génération ; elle a été relancée vers 10 h 10.
- Le poste a redémarré brutalement trois fois le 15/11. Deux de ces arrêts ont coupé une génération : dans la nuit du 14 au 15/11 (campagne v4), puis à 19 h 55 (campagne v5b). Le poste est instable sous charge ; la cause n'est pas connue. L'assistant n'a rien modifié dans le système du poste ; le constat a été transmis à l'étudiant.
- Aucune donnée n'a été perdue : chaque fichier est écrit de façon atomique, avec son rapport, et les générateurs reprennent seuls. L'arrêt de la nuit du 14 au 15/11 a d'abord été attribué à une mise en veille ; c'était un plantage.

**Arrêt volontaire de la génération v5b, le 16/11 vers 1 h, à 61 %.** L'étudiant demande de finir et de finaliser le rendu ; l'assistant arrête la génération et écrit trois raisons :

1. Le compteur indique 21 768 conversations v5 gardées (compte avant filtres), plus les 9 055 de la v4 (compte après filtres) : assez pour des candidats à 3, 5 et 10 % sous environ 1,2 passage par conversation. Seul le candidat à 15 % sera limite, et la garde de par-cœur tranchera. Le compte net, fait après l'arrêt, est de 27 624 conversations, v4 comprise.
2. Le poste a planté trois fois le 15/11. Cinq heures de plus sous charge, avant même d'entraîner, faisaient courir le risque de ne rien livrer.
3. L'arrêt est réversible : la campagne reprend où elle s'est arrêtée.

Aucun résultat des données v5 ni des candidats suivants n'avait été regardé avant cette décision. La règle de jugement reste celle écrite le 15/11 ; seule la quantité de données diffère du plan.

## Méthode et limites

- **La règle de décision est écrite avant de voir les résultats** : seuils, gardes, nombre de tours de réglage permis. Elle a fait rejeter cinq modèles (B, C, D, E, F) et empêché de retenir S05, qui dépassait une limite de 0,2 point. Les cas où la règle a été écrite ou complétée en cours de route sont dans les limites ci-dessous.
- **Le jugement est à l'aveugle.** Les réponses de plusieurs modèles sont mélangées, sans leur nom. Trois juges indépendants notent chaque réponse et on garde la majorité. Le témoin est rejugé dans les mêmes lots que les candidats.
- **Les messages de test ne sont pas dans les données.** Toute conversation de synthèse qui contient un message de test est retirée. Un jeu de test qui a servi à un choix est remplacé par un jeu neuf pour le choix suivant (« neuf » le 14/11, « neuf2 » le 15/11).
- **Une garde contre le par-cœur.** Un fichier de synthèse de la même fabrique, jamais entraîné, permet de comparer le BPB sur des conversations vues et jamais vues.
- **La validation Molière est tenue à l'écart**, et cela a été recontrôlé de façon indépendante : aucune des 34 186 scènes d'entraînement ne partage 12 mots de suite avec elle.
- **Les données et les consignes sont lues**, pas seulement comptées : chaque fichier de synthèse est soumis à un contrôle par 3 lecteurs avec un seuil fixé d'avance. Le premier fichier a raté ce contrôle (38 sur 60, seuil 48) et a quand même servi à deux essais, B et C, déclarés comme tels.
- **Les documents sont contre-vérifiés** : chaque fait des fiches est rattaché à un fichier, puis relu par des contre-lecteurs chargés de réfuter.

**Limites.**
- Tous les juges, lecteurs et relecteurs sont des agents IA. Aucune personne n'a noté les réponses. Les juges du 16/11 sont des agents Claude Opus 5.5, un autre modèle d'IA que celui des jugements des 14 et 15/11.
- Les jeux de test sont petits : 24 tours de parole pour « reglage » et « test », 48 pour « neuf » et « neuf2 », chacun joué avec 3 graines. Une notation peut bouger d'une dizaine de réponses sur 144. Le résultat de S03 (p = 0,044 le 15/11, 96 contre 73 le 16/11) est un gain probable, pas une certitude.
- Les thèmes des messages de test recoupent ceux des consignes de synthèse. Les messages identiques sont retirés, pas les thèmes : cela favorise la mesure.
- Les jeux « reglage » et « test » ont servi plusieurs fois le 14/11. Le choix de réponse a été réglé sur A0 : un léger avantage pour A0 dans les comparaisons de l'après-midi est possible.
- La règle a changé de critère entre le matin du 14/11 (rapport au message) et la suite (réponses sans défaut). Le matin du 14/11, la règle du verdict de C a été écrite après les notes du jeu de réglage, avant le jeu de test. Pour C rejoué le soir, elle a été écrite en connaissant les chiffres du matin ; c'est déclaré. Le jugement de E a été décidé après le verdict de D et F. S03 est un tour de réglage que la règle de la campagne ne prévoyait pas, décidé après avoir vu les gardes de S05, S10 et S15, mais avant tout jugement à l'aveugle.
- Les réponses jugées, les clés de mélange et les notes de chaque juge sont publiées dans `experiments/chatbot_rtx/jugements/`, avec les contrôles de qualité des fichiers de synthèse. Le dépôt contient aussi les recettes, les consignes, les outils et les rapports du lancement du 13/11 (`experiments/chatbot_rtx/20261113_1956_M/`). Les textes de synthèse ne peuvent pas être régénérés à l'identique : seules les consignes sont rejouables. Les conversations de synthèse elles-mêmes sont dans la release (dernière section).
- Les quatre modèles candidats du 16/11 ne sont pas publiés : seuls leurs recettes et leurs réponses jugées le sont.

**Ce qu'on ne sait pas.**
- Si le gain de S03 se retrouve sur un second jeu de test : oui, sur le jeu « neuf2 », 96 contre 73 le 16/11. Ce n'est plus une inconnue, mais cela reste un jeu de 48 messages jugé par des agents IA.
- Si le par-cœur tient au nombre de passages par conversation plutôt qu'à la part de synthèse : à part égale, avec plus de conversations, il a presque disparu le 16/11 (à 5 % : 2,69 % contre 10,2 %). Cela va dans ce sens sans le prouver : entre les deux essais, les données ont aussi changé (consignes v5 ajoutées), et le par-cœur n'est pas mesuré sur les mêmes fichiers tenus à l'écart.
- Si plus de synthèse peut améliorer le modèle : les essais du 16/11 n'ont rien montré de tel. La génération v5 est arrêtée à 61 % ; le reste (environ 18 500 appels) n'a pas été produit, et on ne sait pas ce qu'il aurait changé.
- Pourquoi le contrôle de qualité passe de 48 sur 60 au pilote à 113 sur 120 sur le fichier v4 complet, avec la même consigne de lecture.
- Ce que des personnes penseraient des réponses : aucune n'en a noté.
- Le comportement du modèle sur des messages longs, agressifs ou hors des thèmes testés.
- Le temps de réponse sur un ordinateur sans carte graphique : non mesuré.
- Si S03 se reproduit à l'identique par un réentraînement : non essayé. Un réentraînement ne redonne pas forcément les mêmes octets.
- La cause des redémarrages du poste.

Les deux fiches ont chacune leur liste : [`FICHE_MODELE.md`](FICHE_MODELE.md), § 10, et [`FICHE_DONNEES.md`](FICHE_DONNEES.md), § 9.

## 16/11 : essais finaux et publication

**La génération s'arrête.** La génération v5b est arrêtée à 61 % (29 468 appels sur 48 000). Les raisons et les comptes nets sont donnés plus haut, dans « 16/11 (nuit) » et dans « Erreurs et incidents » : on ne les répète pas ici.

**Les données v4 + v5.** Le fichier d'entraînement des essais compte 27 624 conversations (1 147 010 mots, 2 694 831 tokens). Contrôle de qualité : 120 conversations tirées au hasard, 3 lecteurs IA, seuil fixé d'avance à 96 sans défaut. Résultat : **109 sans défaut** à la majorité ; par lecteur 110, 109 et 107. Par critère : français 117, rapport au message 115, vouvoiement 120, style 116. Le fichier est accepté. Les notes sont dans `experiments/chatbot_rtx/jugements/controle_synthese_v4_v5/`.

**La règle, écrite le 15/11 à 16 h, avant la génération.** Le témoin est S03. Les candidats sont le même affinage avec la synthèse v4 + v5 à 3, 5, 10 et 15 %. Deux gardes : BPB sur Molière au plus égal à celui de A0 plus 0,010 (soit 1,06602), et par-cœur sous 10 %. Jugement à l'aveugle sur le jeu « neuf2 », jamais utilisé avant : 144 réponses par modèle. Un candidat remplace S03 seulement s'il gagne au moins 12 réponses sans défaut (T) et si le français (F), le rapport au message (R) et le style (S) ne reculent pas de plus de 6.

**Les quatre essais et leurs gardes.** Les recettes sont `chatbot/recettes_rtx_v6_s03.json`, `_s05`, `_s10` et `_s15` (poids de tirage de la synthèse 0,36 ; 0,62 ; 1,3 ; 2,06). Les quatre passent les deux gardes. Celui à 15 % passe de justesse : par-cœur 9,81 % pour une limite de 10 %.

**Le jugement final.** 6 modèles, 864 réponses (144 par modèle), 36 agents IA : 3 juges par lot de 72 réponses, aucune note manquante, vote à la majorité. La consigne est identique mot pour mot à celle des jugements précédents. Nombre de réponses sur 144 :

| Modèle | Part de synthèse | Meilleur pas | BPB Molière | Par-cœur | F | R | S | T (sans défaut) | T − T(S03) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 (sans synthèse) | 0 % | 1 000 | 1,05602 | −0,65 % | 113 | 94 | 127 | 73 | −23 |
| **S03 (témoin, modèle publié)** | 3,02 % (v4) | 1 000 | 1,05570 | 1,92 % | 124 | 111 | 142 | **96** | — |
| v6 à 3 % | 2,99 % | 1 000 | 1,05568 | 0,80 % | 130 | 105 | 139 | 93 | −3 |
| v6 à 5 % | 5,04 % | 1 250 | 1,05568 | 2,69 % | 133 | 109 | 144 | 101 | +5 |
| v6 à 10 % | 10,01 % | 1 250 | 1,05746 | 6,23 % | 132 | 104 | 134 | 88 | −8 |
| v6 à 15 % | 14,98 % | 1 250 | 1,05864 | 9,81 % | 128 | 118 | 141 | 105 | +9 |

Par-cœur = 1 − BPB de la synthèse entraînée / moyenne des BPB des 3 fichiers v5 tenus à l'écart. Pour S03, la mesure est faite sur le fichier v4 + v5, dont il n'a vu que la partie v4 ; ce n'est donc pas la mesure du 15/11 (4,6 %). Les parts des candidats v6 sont les parts attendues ; celle de S03 est mesurée.

**Le verdict : aucun candidat n'atteint +12. S03 reste le modèle retenu.** Comparaison appariée contre S03 sur T (même message, même graine ; test des signes bilatéral) : A0 meilleur 25 fois, moins bon 48 fois, p = 0,0095 ; v6 à 3 % : 28 / 31, p = 0,79 ; à 5 % : 36 / 31, p = 0,63 ; à 10 % : 25 / 33, p = 0,36 ; à 15 % : 31 / 22, p = 0,27. Aucun candidat ne se distingue de S03. Détail de T sur 72 : messages isolés, A0 39, S03 48, v6 à 3 % 48, à 5 % 50, à 10 % 48, à 15 % 57 ; conversations de plusieurs tours, A0 34, S03 48, v6 à 3 % 45, à 5 % 51, à 10 % 40, à 15 % 48. Mots par réponse (5 graines) : A0 11,8 ; S03 16,3 ; v6 : 14,6, 14,0, 14,9 et 15,1.

**Ce que la campagne v5 a appris.**
- Le par-cœur a presque disparu à part égale : à 5 %, 2,69 % contre 10,2 % le 15/11 ; à 10 %, 6,23 % contre 23,4 % ; à 15 %, 9,81 % contre 30,5 %.
- La qualité jugée n'a pas progressé de façon démontrée. Un peu de synthèse (3 %) aide ; en mettre plus, même sans par-cœur, n'a pas donné mieux.
- S03 contre A0 est confirmé sur un jeu de messages neuf : 96 contre 73 le 16/11, après 100 contre 82 le 15/11 (jeu « neuf », p = 0,044).

**Le changement de juges.** Les juges du 16/11 sont des agents Claude Opus 5.5 : un autre modèle d'IA que celui des jugements des 14 et 15/11. Ce sont toujours des agents IA, jamais des personnes. Les notes du 16/11 ne se comparent donc pas directement à celles du 15/11. La décision n'en dépend pas, car S03 est rejugé dans le même lot, par les mêmes juges, que les candidats. Accord unanime des 3 juges : F 829 sur 864, R 793 sur 864, S 839 sur 864.

**S03 devient le modèle publié, sans réentraînement.** L'entraînement de S03 a été remis en place ; le lanceur a repris seulement l'export et l'évaluation (`python -m chatbot.pipeline_rtx --recettes chatbot/recettes_rtx_publie.json lancer --taille M --seulement moliere`, 2 min 27 s). Le fichier `ckpt.pt` du modèle publié est donc, octet pour octet, celui qui a été jugé à l'aveugle les 15 et 16/11. La recette `chatbot/recettes_rtx_publie.json` est celle de l'essai (`chatbot/recettes_rtx_v5_s03.json`) plus deux déclarations qui ne changent pas l'entraînement : le fichier de synthèse et le décodage écrit à l'export (`--anchor-mu 1 --candidates 32`). Le `chat_config.json` du modèle porte ce décodage : la commande de chat n'a plus besoin de ces options. L'évaluation automatique, refaite avec ce décodage sur le panel de 120 réponses, donne 15,9 mots par réponse, une reprise de mots du message de 27,27 %, 0,505 seconde par réponse sur la RTX 5090, aucune réponse vide et aucun nom de rôle dans les réponses. Le second chatbot (« français courant ») n'a pas changé depuis le 13/11 et n'est pas dans le paquet du modèle.

**La relecture et le nettoyage du dépôt.**
- Audit de 303 fichiers ; 5 fichiers retirés du dépôt.
- Les preuves des jugements sont publiées dans `experiments/chatbot_rtx/jugements/` : pour le jugement final (`final_neuf2/`), ce que les juges ont vu, la clé qui dit quel modèle a écrit quoi, les notes des 3 juges, les notes finales, les gardes et les réponses des 6 modèles. Les outils de recalcul sont `aveugle.py`, `recompter_jugements.py` et `apparie.py`.
- Trois bogues corrigés : `python tools/reproduce.py --check-only` s'arrêtait sous Windows sur « Fichier ajouté au kit » alors que le kit était intact (chemins comparés avec des barres inverses) ; dans `chatbot/pipeline_rtx.py`, la fonction qui teste si une étape tourne encore plantait sous Windows quand la réponse de la commande système contenait des accents ; dans `chatbot/ancrage.py`, dans une conversation assez longue pour être coupée, le contexte de comparaison perdait une réplique de plus que le vrai contexte (sans effet sur les jugements, dont les messages sont courts et jamais coupés).
- Un commentaire de `chatbot/format.py` affirmait à tort que le modèle de démonstration était ré-entraîné sur les pièces de validation ; il est corrigé, sans aucune ligne de code changée. Les rapports d'entraînement consignent l'empreinte sha256 des fichiers de code : pour retrouver celle de `format.py` consignée dans le rapport de la base, `git show 4def4b5:chatbot/format.py | sha256sum`. L'évaluation du modèle publié, refaite le 16/11, consigne les empreintes de `evaluate.py`, `generate.py`, `format.py` et `finetune.py` tels qu'ils sont dans le dépôt.
- Tests : suite lancée le 16/11 sur le PC Windows, dans l'environnement réduit que crée [`PC_RTX.md`](../../PC_RTX.md) : 454 réussis, 10 ignorés, 9 échecs, 17 erreurs, en 105 s. Tous les échecs et erreurs viennent de cet environnement (sept modules du concours déclarés dans `pyproject.toml` n'y sont pas installés, et un test utilise `os.killpg`, qui n'existe que sous Unix), pas du code. Les tests du chatbot passent tous. La suite n'a pas été relancée sur macOS le 16/11.

**La publication.** Release `chatbot-moliere-20261116` du dépôt `benassou423/Inteligence-lab-piscine` (<https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116>). Pièces jointes :
- `chatbot_moliere_s03.zip` (240 087 338 octets, sha256 `6d62aa50f4917fbb23d24e24a8d64807bde55f273c9cb4203c6970c32c886417`) : le dossier `chatbot_moliere/` du modèle, 4 fichiers. `ckpt.pt` : 239 991 123 octets, sha256 `01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577`.
- `synthese_moliere.zip` (4 442 456 octets, sha256 `e0b144a75acafb5187160d37bcdf1099d60218846cc4b081565819e891c4d3bf`, 22 fichiers, licence CC BY 4.0) : les conversations de synthèse des deux fichiers (v4 : 9 055 conversations, celui du modèle publié ; v4 + v5 : 27 624 conversations, celui des essais du 16/11), les six fichiers tenus à l'écart, les traces des campagnes et un `LISEZMOI_SYNTHESE.txt`. À décompresser à la racine d'un clone.
- l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`, 74 fichiers, 445 528 215 octets, sha256 `9028641e6d5c0aece190dd7a7930545e8693a790927e04b69e3de28a32b1ea5a`, avec son manifeste `RESULTATS_CHATBOT_RTX_20261113_1956_M.manifeste.json` : les deux chatbots, leurs rapports et leurs évaluations), refaite le 16/11. Le nom porte la date de création du lancement (13/11, 19 h 56, taille M) ; le contenu est l'état du 16/11.

Avec `synthese_moliere.zip` décompressé à la racine et les données de la release `chatbot-donnees-20261113` (`python -m chatbot.pipeline_rtx telecharger`), la recette du modèle publié est relançable depuis un clone. Un réentraînement ne redonne pas forcément les mêmes octets : cela n'a pas été essayé.
