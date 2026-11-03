# Comprendre : la collecte

## À quoi ça sert
Rassembler des **textes français bruts** de plusieurs sources, **au même format**, avec leur **date** et leur **licence**.
Tout le reste (nettoyage, déduplication, entraînement) part de là : si la collecte est floue, rien n'est prouvable.

## Comment ça marche
- **Une commande par source** : `uv run python -m pipeline.collect <source> [--limit N]` (`all` = toutes, `stats` = tableau des volumes `docs/sources/volumes.md`). `--limit 200` fait un **échantillon** rapide.
- **Un document = une ligne JSON** (`id` stable, `source`, `domain`, `text`, `url`, `date`, `license`, `meta`) dans `data/raw/<source>/part-NNNNN.jsonl.zst` (**≤ 100 Mo de texte** par fichier, compressé **zstd**).
- **Gel au 16/10/2026** : tout document daté d'après est **écarté** et compté (22 séances, 2 livres). Le crawl Common Crawl est le dernier **fini avant** le gel. Rien d'autre n'est filtré (la décontamination contre la validation est une étape du nettoyage).
- **Texte brut** : on garde menus, pieds de page, en-têtes Gutenberg. Les retirer = étapes de **nettoyage**, dont on mesure l'effet (**ablation**).
- **Reprise et politesse** : travail découpé en **unités** (fichier, lot de livres) notées dans `_work/done.jsonl` ; **miroir officiel** Gutenberg, **1 requête/s** sur l'API Wikisource, **backoff** sur les erreurs 503, **opt-out TDM** respecté.

## Les chiffres clés
| Source | Domaine | Documents | Texte | D'où vient le chiffre |
|---|---|---:|---:|---|
| Assemblée nationale | dialogue | 1 550 séances | 318 Mo | open data XML, 15e (partielle) à 17e législature |
| Wikipédia | encyclopédique | 2,56 M articles | 7,8 Go | dump 2023-11-01 (HF `wikimedia/wikipedia`) |
| Gutenberg | littéraire | 4 228 livres | 1,8 Go | catalogue du 2026-11-01, langue `fr` |
| Wikisource | littéraire | 20 554 livres + 1 046 pages | 5,3 Go | dump 2026-10-01, pages **relues** (niveau ≥ 3) |
| Wikinews | presse | 20 428 articles | 32 Mo | dump 2026-10-01 (site fermé en mai 2026) |
| Common Crawl | web | 11 865 pages | 86 Mo | 10 WARC de CC-MAIN-2026-39 ; **5,5 %** des pages HTML sont françaises |
| FineWeb-2 | web | 522 923 pages | 1,8 Go | 8 fichiers sur 135, à pas régulier |

- La **validation du kit** vient de nos sources (Assemblée, Wikipédia, Gutenberg, Wikinews) : l'Assemblée est écrite **au même format** (« M. Nom : propos », ligne vide entre interventions) et chaque document garde sa **date** pour la **décontamination** (faite au nettoyage).
- **cld2** : détecteur de langue de Google, déjà lancé par Common Crawl (champ `languages-cld2` du WARC).
- **TDMRep** : protocole du W3C (2024) pour dire « pas de fouille de textes » ; base légale : **art. L122-5-3 du CPI** (directive UE 2019/790, art. 4). **33 pages** écartées.

## 3 questions probables du jury
1. **« Pourquoi garder du texte sale (menus, licence Gutenberg) ? »**
   Pour **prouver** que le nettoyage sert : on compare le BPB avec et sans chaque étape. Sans version brute, pas d'ablation.
2. **« Comment garantissez-vous le gel du 16/10 et les licences ? »**
   Chaque document a une `date` et une `license`. La collecte **rejette** ce qui est daté d'après le gel et le **compte** (`_stats.json`). Les sources sont figées (commit HF, dump daté, crawl fini avant le gel). Vérifié le 03/11 : **0 document** daté après le 16/10 sur 3,1 M. La `date` = séance, publication ou mise en ligne (Assemblée, Wikinews, Gutenberg), **capture** (web), **dump** ou **dernière révision** (Wikipédia, Wikisource) : chaque fiche le dit.
3. **« Avez-vous le droit d'utiliser des pages web et des débats nommant des personnes ? »**
   Web : **exception de fouille de textes** + respect des **réserves TDM** et de robots.txt (via Common Crawl). Assemblée : **Licence Ouverte**, orateurs = **élus dans leurs fonctions**, données publiées pour la transparence. Les e-mails et téléphones du web seront masqués au nettoyage (RGPD).
