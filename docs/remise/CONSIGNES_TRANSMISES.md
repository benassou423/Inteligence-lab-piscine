Prochaine action : utiliser ces exigences comme référence de remise — lecture 3 min.

# Consignes officielles transmises par Ben le 10/11/2026

Source : copie du sujet « Feed the Beast : nourris ton propre GPT » fournie par Ben dans ce chat. Ce document en extrait les exigences pour éviter de confondre obligations et pistes de recherche. La valeur du budget laissée vide sur la page est précisée par le kit officiel : **100 007 936 tokens vus**.

## Bloc 1 obligatoire

1. Mini-autograd, classe `Value` et `backward()`, vérifié contre PyTorch.
2. Bigramme par comptage.
3. BPE implémenté from scratch : apprentissage, encodage et décodage.
4. Transformer décodeur : attention multi-têtes, masque causal, LayerNorm, MLP et connexions résiduelles.
5. Modèle entraîné sur Molière avec ce tokenizer, courbe de loss, BPB de validation et cinq générations.

Le sujet exige que Ben puisse expliquer chaque ligne. Il réserve la réalisation personnelle du mini-GPT et autorise librement l’assistance IA sur les autres blocs. AGENTS.md rapporte une autorisation particulière de l’organisateur, selon Ben, pour coder le bloc 1 avec assistance ; une confirmation écrite n’est pas encore enregistrée. L’assistance reste indiquée, et l’appropriation orale n’est pas déclarée vérifiée.

## Bloc 2 obligatoire

1. Pipeline reproductible avec une commande, collecte depuis Common Crawl brut et au moins deux sources propres françaises. Fiche de chaque source : origine, licence, volume, période, risques. Le mot « segment » du sujet doit être rapproché des unités réellement collectées : la collecte actuelle traite dix WARC du premier segment, pas une preuve de traitement du segment entier.
2. Réparation d’encodage, langue, suppression du boilerplate. Précision du filtre de langue sur un échantillon annoté humainement. Les annotations IA précédentes ne seront pas rebaptisées humaines.
3. Déduplication exacte par hash et quasi-doublons par **MinHash + LSH**, seuil justifié.
4. Qualité par **heuristiques et/ou classifieur** et masquage des données personnelles. Le classifieur supplémentaire est une option, pas une obligation distincte.
5. Rapport d’ablation de deux pages maximum : même modèle sur données **brutes, nettoyées et complètement traitées**, avec analyse de l’impact des étapes. La seule expérience de nettoyage ciblé ne remplit pas encore cette comparaison à trois niveaux.

## Bloc 3 obligatoire

Modèle, hyperparamètres, seed, batch et contexte figés par le kit. Tokenizer libre entraîné sur les données, vocabulaire fixé à 8 192. Sources légalement utilisables et documentées. Maximum deux soumissions par jour, avec modèle, tokenizer, journaux et hash du dataset. Réentraînement de vérification par l’organisateur, sans écart inexpliqué.

Interdictions : poids pré-entraînés, distillation des logits et tentative de reconstitution du test. Les données synthétiques sont facultatives, limitées à 20 %, issues d’un modèle open weights avec prompts publiés. La recette actuelle n’en utilise pas.

## Rendu et évaluation

Classement sur le test caché, vote à l’aveugle sur les générations, soutenance couvrant données, ablations, échecs et code du mini-GPT, datasheet inspirée de Gebru et al. Le dépôt GitHub lié à la plateforme porte les rendus. La publication Hugging Face avec dataset card relève d’un prix bonus.

Barème : BPB 40 %, rapport/datasheet/soutenance 30 %, mini-GPT et pipeline 15 %, générations à l’aveugle 15 %.

Les proportions de mélange, la répétition, le tokenizer, l’ordre et les données synthétiques sont des pistes à explorer. Le plan interne détaille d’autres ablations utiles ; leur absence ne remplace pas les obligations explicites ci-dessus. Le suivi vérifié de chaque obligation se trouve dans EXIGENCES.md.
