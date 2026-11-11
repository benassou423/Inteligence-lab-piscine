# Prochaine action : entraîner les témoins bruts et complets une fois leurs contrôles verts

1. **Question mesurée.** Quel effet le pipeline a-t-il sur le même modèle à budget
   constant ? Trois niveaux : brut protégé contre la validation ; nettoyé actuel ;
   complet avec masquage, qualité web et quasi-doublons. Les deux premiers ne sont
   pas présentés comme des jeux publiables : ils peuvent conserver des coordonnées.
2. **Conditions fixes.** Modèle du kit, initialisation 1337, mêmes hyperparamètres,
   25 001 984 tokens effectivement vus, tokenizer de référence figé à 8 192 tokens,
   micro-lots de 16, MPS float32 et compilation désactivée. Deux graines de tirage
   du mélange : 1337 et 2027. Elles ne sont pas des graines d'initialisation du modèle.
3. **Sélection.** Proportions naturelles, plafond de 100 millions d'octets de texte,
   tirage et ordre par hash de l'ID. Pour le brut, les IDs réservés sont exclus avant
   toute sélection. Une pré-sélection de 100 Mo conservant IDs et provenance passe
   seulement par exclusion des IDs et décontamination ; aucun filtre de langue,
   qualité ou doublon. Cette protection est appliquée aussi aux autres variantes.
   Le plafond évite une copie complète de 18 Go ; il ne choisit pas les textes
   selon leur score de validation. La décontamination peut réduire ce témoin sous
   100 Mo, limite à déclarer avec les volumes effectivement mesurés.
4. **Contrôles avant entraînement.** Kit intact, 0/230 document de validation signalé,
   flux assez long pour ne pas répéter son début. Réutiliser les deux runs nettoyés
   existants uniquement après vérification de ces conditions et des signatures.
   Les comparaisons de corpus incluent leur changement naturel de proportions de
   sources ; elles n'isolent pas l'effet d'un changement de proportion particulier.
5. **Lecture des résultats.** Moyenne des deux tirages et résultats de chacun des
   quatre genres. Deux tirages mesurent une partie de la variabilité ; ils ne
   suffisent pas à une estimation précise d'incertitude. Un score à 25M n'est pas
   une soumission au budget officiel de 100M. Ne pas remplacer le corpus actif
   avant les résultats, les relectures des retraits et les contrôles finaux.

Durée estimée : préparation brute 3–8 minutes ; quatre entraînements nouveaux
environ 40–60 minutes au total sur le Mac. Le traitement complet doit d'abord
terminer ; sa durée et la charge de la machine peuvent rallonger cette estimation.

Le rapport de deux pages devra remplacer son état de travail par le tableau des
six runs réellement obtenus, avec budgets, scores, volumes, limites et décision.
La comparaison de nettoyage ciblé déjà négative peut rester une expérience annexe.
