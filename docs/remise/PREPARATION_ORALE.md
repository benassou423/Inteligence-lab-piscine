Prochaine action : répondre aux trois questions du premier groupe sans lire les réponses — 5 min.

# Préparation orale de Benjamin

Ce document prépare la soutenance. Il ne prouve pas que Ben maîtrise déjà le code. Les démonstrations et réponses devront être faites par Ben. Les fiches `docs/comprendre/minigpt_*` donnent le parcours de lecture du code.

## 1. Calcul des gradients — 10 min

**Question 1. À quoi sert micrograd ?**

Il construit un graphe des opérations sur des nombres et calcule comment une sortie dépend de chaque entrée. Un gradient indique la variation locale de la sortie quand une entrée varie. Exemple : pour `y = x*x`, la dérivée vaut `2*x`. L’entraînement utilise ces dérivées pour ajuster les paramètres.

**Question 2. Pourquoi parcourir le graphe à l’envers ?**

La sortie dépend d’opérations intermédiaires. On connaît d’abord la dérivée de la sortie par rapport à elle-même, égale à 1. La règle de la chaîne propage ensuite les contributions vers les entrées. Si une variable intervient dans deux branches, on additionne les contributions au lieu d’en écraser une.

**Question 3. Pourquoi remettre les gradients à zéro ?**

Le calcul les accumule. Sans remise à zéro entre deux mises à jour, on additionnerait aussi les gradients d’anciennes étapes. La remise à zéro évite cette accumulation involontaire.

**Démonstration :** ouvrir `minigpt/micrograd.py`, montrer l’opération de multiplication et la méthode qui propage son gradient. Calculer à la main une petite expression, puis comparer au code.

## 2. Modèle bigramme — 10 min

**Question 1. Que prédit ce modèle ?**

Il prédit le prochain symbole à partir du symbole actuel. Il utilise donc un contexte d’un seul symbole. Des comptages de paires peuvent estimer ces probabilités ; une version entraînable apprend une table de paramètres.

**Question 2. Pourquoi le score d’entraînement seul ne suffit-il pas ?**

Le modèle peut apprendre des particularités des textes déjà vus. Il faut mesurer ses prédictions sur des textes distincts. Pour le bloc 1, les pièces de validation Molière restent tenues à l’écart.

**Question 3. Que mesure le BPB ?**

Les bits par octet représentent le coût des prédictions, ramené à la taille du texte UTF-8. Une baisse est meilleure. Cela rend comparables des tokenizers qui découpent le même texte en nombres de tokens différents.

**Démonstration :** montrer comment une ligne de la table devient des probabilités et expliquer pourquoi elles totalisent 1.

## 3. BPE et tokenisation — 10 min

**Question 1. Qu’est-ce qu’un token ?**

C’est une unité du découpage du texte. Elle peut représenter un octet, un morceau de mot, un mot ou de la ponctuation. Le modèle prédit ces unités, pas directement des phrases entières.

**Question 2. Comment fonctionne BPE ?**

Il part d’unités élémentaires et fusionne progressivement des paires fréquentes. Les règles de fusion créent un vocabulaire. Le mini-GPT Molière mesuré utilise 2 048 tokens ; le vocabulaire de la compétition comporte exactement 8 192 tokens, y compris le marqueur de fin de document. Les deux résultats ne doivent pas être confondus.

**Question 3. Pourquoi exiger une reconstruction exacte ?**

Si le décodage change les accents, espaces ou caractères, le modèle n’a pas appris le texte annoncé et le score devient difficile à comparer. Il faut que décoder l’encodage restitue exactement les mêmes octets. La normalisation des données appartient à une étape explicite du pipeline, pas à une transformation cachée du tokenizer.

**Démonstration :** encoder puis décoder une phrase avec accents, apostrophe, espaces et retour à la ligne. Expliquer le rôle du marqueur de fin de document.

## 4. Transformer et entraînement — 15 min

**Question 1. Pourquoi un masque causal ?**

Pendant la prédiction d’un symbole, le modèle doit utiliser seulement les symboles précédents. Le masque bloque l’attention vers les symboles futurs. Sans ce blocage, le modèle pourrait lire la réponse pendant l’entraînement.

**Question 2. À quoi servent les embeddings et les positions ?**

Les embeddings transforment les identifiants de tokens en vecteurs de nombres. Les informations de position permettent de distinguer leur ordre. L’attention combine ensuite les informations pertinentes du contexte.

**Question 3. Pourquoi découper un batch de 64 en petits lots de 16 ?**

Les activations nécessaires au calcul occupent moins de mémoire. On additionne les gradients de quatre petits lots avant la mise à jour. Le batch effectif reste 64. Cette option est prévue par le kit officiel.

**Démonstration :** montrer le masque dans le code du mini-GPT, puis le calcul de la loss et l’appel à la propagation des gradients. Distinguer le modèle personnel du bloc 1 du kit figé de la compétition.

## 5. Données et expériences — 15 min

**Question 1. Pourquoi refaire Wikipédia ?**

La source HF 2023 supprimait des valeurs contenues dans les modèles wiki. Le nouveau collecteur rend les modèles courants du dump 20261001. Les petits extraits de test vérifient dates, nombres et apostrophes. Des modèles dynamiques restent non résolus, et leurs valeurs ne sont pas inventées.

**Question 2. Pourquoi avoir conservé une recette qui contient encore des résidus ?**

Les six runs à 25M donnent des moyennes de 1,657747 sur le brut, 1,622266 après nettoyage et 1,632570 avec qualité/MinHash supplémentaires. Le nettoyage de base aide ; la variante complète dégrade les deux tirages. Les 50 paires proches atteignent le seuil de mots communs, mais 38 décrivent des entités différentes. Nous conservons donc la référence, puis masquons seulement les coordonnées du mélange retenu. Les 72 612 articles restent présents ; le modèle final à 100M obtient 1,288292 BPB. Ce score à 100M ne doit pas être comparé aux petits runs comme si seul le nettoyage avait changé.

**Question 3. Que prouvent zéro signalement et le hash du dataset ?**

Zéro signalement signifie qu’aucun document de validation ne dépasse le seuil de recouvrement du contrôle, pas qu’aucune phrase n’est commune. Le hash identifie le flux de tokens préparé. Il aide à vérifier que deux machines utilisent exactement le même flux. Ni le hash seul ni un fichier de statut ne prouvent que le calcul est terminé : il faut examiner le modèle, les logs, le budget réellement vu et l’évaluation.

**Démonstration :** ouvrir le manifeste du mélange et retrouver son nombre de documents, sa répartition par source, son tokenizer et son empreinte. Expliquer que les 72 612 documents préparés ne sont pas tous nécessairement vus entièrement avec le budget de 100 M tokens.

## Répétition finale — 20 min

Présenter les douze diapositives principales en 8 à 10 min ; garder les deux annexes Molière pour les questions. Répondre ensuite à cinq questions sans lire les notes. Montrer une exécution du bloc 1 et le hash retrouvé par la commande de reproduction sur les bruts locaux. Relever les réponses incomplètes et reprendre la fiche correspondante. Ne cocher la maîtrise qu’après cette vérification réelle.
