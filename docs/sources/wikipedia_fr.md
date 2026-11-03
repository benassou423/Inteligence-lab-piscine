# Wikipédia en français — `wikipedia_fr`

**Domaine** : encyclopédique · **Commande** : `uv run python -m pipeline.collect wikipedia_fr` · **Code** : `pipeline/collect/wikipedia.py`

## Origine
- Dataset Hugging Face **`wikimedia/wikipedia`**, config **`20231101.fr`** = dump de Wikipédia du **1er novembre 2023**, déjà converti en texte par Wikimedia (wikitexte → texte).
- C'est la **seule config française** du dépôt (liste vérifiée le 2026-11-03 : uniquement `20231101.*`). Commit figé : `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- 17 fichiers parquet (4,6 Go) téléchargés un par un, convertis, puis **supprimés**.
- Un document = un article. On garde **titre** (`meta.title`) et **URL**. `date` = date du dump (2023-11-01).

## Licence
- **CC BY-SA 4.0** (textes de Wikipédia). Obligations : **citer** Wikipédia et ses auteurs, **partager à l'identique** (un dataset publié qui contient ces textes doit rester sous CC BY-SA).

## Volume
- **2 564 645 articles**, **7 814 Mo** de texte (UTF-8), 87 fichiers, 2,75 Go sur disque (zstd).
- 1 article vide écarté.

## Période couverte
- Instantané du **2023-11-01**. Rien après : aucun événement de 2024-2026.

## Risques
- **Biais temporel** : 3 ans de retard sur le test caché (textes publiés après le 16/10/2026). Les sujets récents manquent. Piste : dump officiel `frwiki-20261001` (dumps.wikimedia.org), mais il faut alors convertir le wikitexte soi-même (cf. `wikisource_fr`).
- **Qualité** : la conversion de Wikimedia **supprime les modèles** → trous dans le texte : « né le  à Moulins », « au  siècle » (dates, siècles, unités disparus). Beaucoup d'**ébauches** et d'articles générés en série (communes, espèces) : texte répétitif.
- **Doublons** : listes, homonymies, articles quasi identiques (communes) → déduplication.
- **RGPD** : biographies de personnes vivantes (surtout publiques). Données déjà publiques, intérêt légitime ; ne pas extraire d'informations personnelles ciblées.
- **Biais** : sur-représentation de certains thèmes (sport, géographie, Europe) et d'un registre neutre et descriptif.
