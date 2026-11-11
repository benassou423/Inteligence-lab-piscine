Prochaine action : lire la comparaison avec le corpus actuel — lecture 2 min.

# Nettoyage ciblé : retirer les défauts sans perdre le texte

1. **À quoi ça sert.** Certains articles contiennent encore des tableaux bruts, des cadres techniques ou un même mot répété des dizaines de fois. Ces caractères prennent de la place dans le temps d’apprentissage du modèle.

2. **Comment ça marche.** `pipeline/clean/targeted.py` applique des règles explicites. Sur Wikipédia et Wikinews, il retire les tableaux clairement délimités, les cadres `Infobox` et les longues répétitions. Il retire aussi des lignes de mise en page comme `center|450px` ou un identifiant `Q12345` isolé. Un simple `|` devant une vraie phrase est retiré seul. Les petits textes, la poésie, les dates dans les phrases et les noms étrangers ne sont pas des motifs de rejet.

3. **Ce qui protège les bons textes.** Les tableaux de Wikisource peuvent contenir des traductions ou des pages de prose : leur suppression est désactivée. Les tableaux dont les cellules débordent sur des lignes ordinaires sont également conservés. Le premier essai a servi à repérer ces cas avant la comparaison. Sur le tirage frais de 2 000 documents : **18 modifiés, aucun retiré**, 12 témoins positifs identiques. Le contrôle du corpus complet a ensuite permis de corriger une courte biographie précédée d’un `|`.

4. **Comment comparer.** La variante se trouve dans `data/stages/nettoye_cible_v1/` ; le corpus actuel reste dans `data/stages/nettoye/`. Même seuil de français, même modèle, même outil de découpage du texte, même budget de 25 millions de morceaux de texte. Deux graines de mélange, **1337 et 2027**, fixent les tirages et l’ordre des documents. L’initialisation du modèle reste celle du kit. Le score **BPB**, bits par octet, mesure la qualité des prédictions : plus bas est meilleur. **Résultat : la moyenne passe de 1,6223 à 1,6396 BPB, donc elle se dégrade. Le corpus actuel est conservé.** Le bilan chiffré se trouve dans `experiments/targeted_clean_v1_20261104.md`.

5. **Trois questions du jury.**

   **Retirer plus de texte améliore-t-il forcément le modèle ?** Non. Il faut vérifier ce qui a été retiré, puis comparer les résultats d’apprentissage.

   **Avez-vous aussi élargi le filtre de langue ?** Non. Cela permet d’évaluer le nettoyage seul. Les livres bilingues demandent un traitement distinct.

   **Ces essais peuvent-ils être soumis ?** Non. Ils utilisent un quart du budget officiel sur le Mac. Une recette retenue doit encore être confirmée avec le protocole complet.
