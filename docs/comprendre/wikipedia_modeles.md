Prochaine action : lire les exemples, puis répondre aux 3 questions — 4 min.

# Pourquoi rendre les modèles Wikipédia ?

1. **À quoi ça sert.** Le *wikitexte* est la syntaxe utilisée pour écrire Wikipédia. Un *modèle* est une petite instruction, comme `{{nombre|4200}}`. Si on l’efface, la phrase perd son nombre. On transforme cette instruction en texte lisible.

2. **Comment ça marche — environ 1 min.**
   - On lit le *dump*, une copie officielle datée du 01/10/2026, morceau par morceau.
   - On garde les articles et leur date de dernière révision ; on retire redirections et homonymies.
   - `WikipediaRenderer` s’appuie sur le moteur `wikitext.py`. Il traduit les modèles connus, y compris ceux imbriqués.
   - On retire les infobox, références, tableaux et fichiers. On compte les modèles inconnus pour améliorer les règles.
   - On mesure d’abord 2 000 articles et on en lit 20. La collecte complète vient après les seuils atteints.

   | Avant conversion | Texte conservé |
   |---|---|
   | `né le {{date|1|2|2000}}` | né le 1er février 2000 |
   | `{{nombre|4200}} habitants` | 4 200 habitants |
   | `{{unité|12.5|km|2}}` | 12,5 km² |
   | `L{{'}}'''algorithmique'''` | L'algorithmique |

3. **Chiffres observés le 04/11/2026 — 1 min.** Sur 2 000 articles : **0,75 %** de trous selon la regex du brief (limite 5 %), **0 %** de premiers mots collés (limite 0,2 %), **20 lectures valides**. Sur les **2 641 265 articles bruts** : **0,804 %** de trous regex. Après nettoyage : **2 546 229 articles Wikipédia**. Contrôle de contamination sur tout le corpus : **0 document signalé**. Les mesures et textes relus sont conservés dans `data/stages/wikipedia_quality_2000/` et `data/stages/logs/`.

4. **Trois questions du jury — 1 min.**

   | Question | Réponse courte |
   |---|---|
   | Pourquoi garder HF 2023 ? | Pour mesurer l’effet de la nouvelle source par une *ablation*, une comparaison entre versions. Son texte d’origine est conservé. |
   | Pourquoi ne pas chercher les nombres manquants en ligne ? | Une valeur actuelle pourrait dépasser le gel des données. Les modèles dynamiques non résolus restent une limite déclarée. |
   | Comment prouver que cela fonctionne ? | Tests sur de petits exemples, essai mesuré, 20 lectures, audit de tous les articles, kit inchangé et contrôle complet de contamination à 0. |

La regex ne voit pas toutes les erreurs. Les anciens chiffres du diagnostic HF et les chiffres du nouvel essai ne suffisent pas à prédire un gain de score : l’ablation d’entraînement reste à faire.
