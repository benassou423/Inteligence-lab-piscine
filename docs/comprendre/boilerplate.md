# Boilerplate : ne garder que le contenu

## À quoi ça sert
Le **boilerplate**, c'est ce qui entoure le contenu : menus, pieds de page, bandeaux cookies, licence Gutenberg, listes de références.
C'est répétitif et sans langue naturelle : le modèle gaspillerait son budget de tokens dessus.

## Comment ça marche
- **Web (Common Crawl)** : **trafilatura** lit le HTML et garde le texte principal. Sans commentaires, sans tableaux, en mode **précision**. Rien d'extrait → page retirée.
- **Gutenberg** : on coupe avant le marqueur `*** START OF …` (ou `*END*THE SMALL PRINT`), les paragraphes de crédits (« Produced by… », « Credits: », ancien en-tête recopié) parmi les 4 premiers, et tout ce qui suit `*** END OF …` (la licence).
- **Wikipédia** : on coupe au premier titre final (« Notes et références », « Voir aussi », « Liens externes »…) **s'il n'y a plus de prose après**. Puis on retire les restes : titres vides en fin d'article, wikitexte brut (`{|`, `|-`, `{{Infobox…`).
- **Autres sources** (Assemblée, Wikisource) : déjà propres, inchangées.
- Le texte extrait du HTML passe par la **même réparation d'encodage** que l'étape précédente.

## Les chiffres clés
| Source | Effet mesuré | D'où |
|---|---|---|
| Wikipédia | sections finales coupées dans **90 %** des articles, résidus dans 19,5 %, wikitexte dans 2,1 % ; **8,1 % des octets** retirés | 40 000 articles réels |
| Gutenberg | en-tête **et** licence trouvés dans **4 228 / 4 228** livres ; 10 livres gardaient un 2e paragraphe de crédits (corrigé) | corpus complet, 03/11 |
| Wikipédia, 30 articles relus | 29 sans aucune ligne de prose retirée ; 1 ligne de bibliographie | tirage par hachage |
| Common Crawl | 77 % des pages extraites ; texte brut 13,8 Mo → **4,6 Mo** (les deux tiers étaient du menu) | `--limit 2000` |
| Mode trafilatura | « secours » activé : +18 % de pages mais +5 % de texte seulement, surtout du rebut, et 1,6x plus lent | 300 pages réelles |

Références : trafilatura (Barbaresi, ACL 2021), utilisé par **FineWeb** (Penedo et al., 2024) avec la même option de précision.

## 3 questions probables du jury
1. **Pourquoi ne pas couper toutes les sections « Bibliographie » ?**
   Parfois une section de fin contient de la vraie prose (une note rédigée). La règle « plus de prose après » la protège : on n'enlève que des listes.
2. **Comment savez-vous que ça marche ?**
   Chaque étape écrit un **échantillon avant/après** (`_changed_sample/apercu.md`) et les pages retirées (`_rejected_sample/`). Sur 40 000 articles, les longues lignes retirées étaient presque toutes des références bibliographiques.
3. **Pourquoi `deduplicate=False` dans trafilatura ?**
   Cette option garde un cache **partagé** entre pages : le résultat dépendrait de l'ordre de traitement. Notre pipeline doit donner **le même résultat** à chaque exécution.
