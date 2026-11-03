# FineWeb-2, français — `fineweb2_fra` (optionnel)

**Domaine** : web (non classé) · **Commande** : `uv run python -m pipeline.collect fineweb2_fra` · **Code** : `pipeline/collect/fineweb2.py`

## Origine
- Dataset Hugging Face **`HuggingFaceFW/fineweb-2`**, config **`fra_Latn`** (commit figé `af9c13333eb981300149d5ca60a8e9d659b276b9`, dépôt mis à jour en octobre 2025) : pages Common Crawl de 2013 à avril 2024, **déjà** extraites (trafilatura), filtrées (langue, qualité) et dédupliquées (MinHash, par langue, globalement) par Hugging Face.
- 135 fichiers parquet (540 Go) rangés par dump Common Crawl. Sous-ensemble **déterministe et réparti** :
  - **8 fichiers à pas régulier** (n° 0, 19, 38, 57, 77, 96, 115, 134 dans l'ordre trié) → toutes les périodes ;
  - dans chaque fichier, des **groupes de 1 000 lignes à pas régulier** (~85 sur ~3 300), jusqu'à **250 Mo de texte** par fichier ;
  - lecture **en flux** (requêtes Range), rien n'est téléchargé en entier.
- Document : `id = fineweb2_fra:<WARC-Record-ID>`, `url`, `date` = date de capture, `meta` : dump, score de langue, taille du groupe MinHash, fichier, groupe de lignes.

## Licence
- **ODC-By 1.0** (attribution à FineWeb-2) + conditions d'utilisation de Common Crawl ; le droit d'auteur des pages reste à leurs auteurs (exception de fouille de textes, art. L122-5-3 CPI).

## Volume
- Voir `docs/sources/volumes.md` (cible : **~2 Go de texte**, ~600 000 pages).

## Période couverte
- Captures de **l'été 2013 à avril 2024** (96 dumps Common Crawl).

## Risques
- **Comparaison avec notre Common Crawl** : FineWeb-2 est le **témoin « déjà nettoyé »** (extraction + filtres + dédup faits par d'autres) ; utile pour l'ablation « notre pipeline vs un pipeline de référence ».
- **Juridique** : pas de vérification des réserves TDM par page (le HTML n'est pas fourni) ; opt-out = robots.txt au moment du crawl.
- **RGPD** : e-mails et IP **déjà anonymisés** par FineWeb-2 (`email@example.com`, `22.214.171.124`), mais noms et téléphones restent possibles.
- **Qualité** : filtres heuristiques imparfaits (spam, listes, contenu adulte) ; `minhash_cluster_size` signale les textes très dupliqués avant dédup.
- **Biais** : pages anciennes (2013-2024), loin des textes publiés après le 16/10/2026 ; sur-représentation du commerce, des forums et du SEO.
- **Recouvrement** avec `commoncrawl` : faible (dumps différents), mais même web → déduplication croisée utile.
