Prochaine action : lire la méthode et les limites avant d’utiliser la source — 5 min.

# Wikipédia en français — `wikipedia_fr`

1. **Origine et date.** [Dump officiel frwiki du 1er octobre 2026](https://dumps.wikimedia.org/frwiki/20261001/), en 15 parties `pages-articles-multistream` : 7 284 764 433 octets compressés. Le manifeste officiel fournit les tailles et empreintes SHA-1, contrôlées avant lecture. La source est figée avant le gel du 16/10/2026. Le code est dans `pipeline/collect/wikipedia.py`.

2. **Ce qui devient un document.** Un article de l’espace principal (espace 0), sans redirection ni modèle d’homonymie reconnu. Lecture progressive du XML, 4 processus, au plus 2 téléchargements simultanés ; chaque bz2 est supprimé après traitement réussi. `date` est le jour UTC de la dernière révision, au format commun AAAA-MM-JJ ; l’horodatage exact reste dans `meta.last_revision`. Les révisions après le 01/10/2026 sont exclues. L’ID de page, le titre, l’URL et le nom de la partie du dump sont conservés.

3. **Conversion en texte.** Le wikitexte est la syntaxe d’édition de Wikipédia. `WikipediaRenderer` (`pipeline/collect/wikipedia_render.py`) hérite de `pipeline/collect/wikitext.py`, dont le comportement partagé reste intact. Il restitue notamment dates, nombres, unités, langues, noms japonais, siècles, ordinaux, heures, citations et pays. Il conserve les apostrophes de `L'''algorithmique'''` et `L{{'}}'''algorithmique'''`. Infobox, références, tableaux, fichiers et catégories sont retirés. Les modèles inconnus sont comptés dans `meta.unknown_templates` et `_stats.json`. Les formules mathématiques restent en notation textuelle TeX.

4. **Licence et attribution.** Textes sous **CC BY-SA 4.0**, d’après les [conditions Wikimedia, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content). L’URL de chaque article donne accès à son historique et à ses contributeurs. La transformation du texte et la licence doivent accompagner une redistribution de ce corpus. Aucune image n’est collectée.

5. **Volumes réellement vérifiés le 04/11/2026.**

   | Mesure | Résultat |
   |---|---:|
   | Articles bruts | 2 641 265 |
   | Texte brut UTF-8 | 8 909 572 729 octets |
   | Fichiers compressés | 97, environ 3,24 Go |
   | Dates de dernière révision | 08/05/2007 → 01/10/2026 |
   | Redirections retirées | 1 844 540 |
   | Homonymies retirées | 140 985 |
   | Textes vides retirés | 230 |
   | Articles après `nettoye` et doublons exacts | 2 546 229 |
   | Texte nettoyé | 8 225 740 068 octets |

6. **Essai obligatoire avant collecte complète.** Les 2 000 articles sont répartis sur les 15 parties, pris au début de chaque partie après les exclusions structurelles. Cet échantillon est déterministe, sans prétendre être un tirage uniforme. Les 20 lectures sont sélectionnées par empreinte d’ID, sans choisir les textes les plus courts. Après corrections sur les mêmes articles : **15/2 000 trous regex (0,75 %)**, **0/2 000 premiers mots avec apostrophe perdue**, **20/20 lectures sans trou visible dans la prose conservée**. Les tableaux et références retirés ne font pas partie de la prose vérifiée. La perte d’apostrophe est vérifiée par comparaison au wikitexte original.

   Regex du brief : ` (le|du|de|en|à|par|était de|s'élevait à) [.,]`.

   L’audit de **tous les articles bruts** retrouve **21 238 documents avec cette regex (0,804 %)**, aucune date/source/ID/document vide invalide, et des textes identiques pour les 20 articles relus. La regex comporte des faux positifs (`.NET`, `.am`) et ne mesure pas toutes les erreurs possibles.

7. **Limites et décontamination.** Ce convertisseur n’exécute pas tout MediaWiki, Lua ni Wikidata. Les modèles rares peuvent rester partiellement rendus. Une population absente du wikitexte n’est pas inventée ni récupérée depuis une valeur actuelle. Les biais de thèmes, les ébauches, les listes et les biographies de personnes vivantes restent ceux de Wikipédia. Le premier contrôle de contamination signalait 1 document à 15,3 % : des phrases fréquentes échappaient au masquage. Le pipeline plafonne désormais le cumul de ces exemptions au seuil du kit. Après reprise : **0 document signalé sur 230**, recouvrement maximal **8,4 %**, sur les 7 sources nettoyées. Voir `docs/comprendre/decontamination.md`.

8. **Rejouer et retrouver les preuves.** Toutes les commandes partent de la racine `Inteligence-lab-piscine/`.

```bash
# Avant une collecte complète : produire l’essai.
uv run python -m pipeline.collect wikipedia_fr --limit 2000
# Collecte complète : lancer avec nohup et journal dans data/stages/logs/.
uv run python -m pipeline.collect wikipedia_fr
# Relire les mesures de l’essai déjà archivé, sans remplacer la collecte complète.
uv run python -m tools.check_wikipedia_quality --source data/stages/wikipedia_quality_2000/sample
uv run python -m tools.check_contamination --config nettoye --fast
uv run python -m pipeline.collect stats
```

Une collecte complète présente est protégée : `--limit` est ignoré sans `--force`. L’essai, les 20 lectures, le classement des inconnus et les empreintes du code sont dans `data/stages/wikipedia_quality_2000/`. Le manifeste du dump est dans `data/raw/wikipedia_fr/_dump_manifest.json`. Les journaux, l’audit intégral et les contrôles avant/après sont dans `data/stages/logs/wikipedia-*.log` et `wikipedia-*.json`.

L’ancienne source est conservée sous [`wikipedia_fr_hf2023`](wikipedia_fr_hf2023.md), exclue du corpus par défaut. La comparaison d’entraînement reste à faire : elle fera varier à la fois la date du dump et son convertisseur, pas uniquement le rendu des modèles.

## Les 30 modèles encore inconnus dans l’essai

Le compteur porte sur les modèles rencontrés par le rendu, hors blocs explicitement retirés. Classement observé après les corrections :

| Modèle | Occurrences |
|---|---:|
| `réfsérie titre` | 63 |
| `réfsérie` | 49 |
| `fc` | 49 |
| `ntsh` | 48 |
| `album` | 39 |
| `lire en ligne` | 39 |
| `souverain2` | 38 |
| `2` | 38 |
| `volley classement ggpp` | 38 |
| `historique tournoi tennis/gc base` | 36 |
| `wsc non-famille` | 35 |
| `tri` | 34 |
| `catégorie principale` | 33 |
| `écrit` | 33 |
| `refsou` | 33 |
| `homon` | 33 |
| `pistes` | 33 |
| `geogroup` | 33 |
| `sport couleur` | 33 |
| `formule chimique` | 31 |
| `fchim` | 31 |
| `maillot` | 31 |
| `mul` | 30 |
| `coll` | 29 |
| `tableau coupe 8 (3 sets)` | 29 |
| `références nombreuses` | 28 |
| `uk` | 28 |
| `bra` | 27 |
| `arbre généalogique` | 26 |
| `%` | 26 |
