# Wikinews en français — `wikinews_fr`

**Domaine** : presse · **Commande** : `uv run python -m pipeline.collect wikinews_fr` · **Code** : `pipeline/collect/wikinews.py`

## Origine
- **Dump officiel** `frwikinews-20261001-pages-articles.xml.bz2` (dumps.wikimedia.org, 25 Mo). Wikinews FR est **fermé depuis mai 2026** : ce dump contient tous les articles.
- Un document = un **article de l'espace principal** (hors redirections, page d'accueil, textes de moins de 200 caractères), y compris les pages « Brèves : <date> ».
- Texte au **format de la validation** du kit : **date en première ligne** (« 1er janvier 2024 »), **paragraphes séparés par une ligne vide**, intertitres gardés (« Sources »), liste des sources, tableaux, catégories et mise en forme retirés. Conversion wikitexte → texte : celle de Wikisource (`wikitext.py`), avec quelques règles Wikinews (`{{w|cible|libellé}}` → libellé, `{{Date|…}}` sur sa ligne).
- `date` = **date de publication** : modèle `{{Date|…}}`, sinon `{{Publication|jour|mois|année}}`, sinon titre des brèves ; à défaut, dernière révision. `meta.date_source` dit d'où vient la date ; `meta.statut` = statut de publication (article, brève…).

## Licence
- **CC BY 2.5** : réutilisation libre avec **attribution** (Wikinews et ses contributeurs, URL de chaque article dans `url`).

## Volume
- **20 428 articles**, **32,2 Mo** de texte, 1 fichier (11 Mo zstd). Écartés : 2 478 redirections, 3 791 textes de moins de 200 caractères, la page d'accueil.
- Origine de la date (03/11) : `{{Date}}` 19 384, titre 941, `{{Publication}}` 64, dernière révision 39. Les 941 « titre » sont 42 pages « Brèves : <date> » et **898 pages « Évènements du <date> »** (listes de titres d'articles du jour, 0,3 Mo) : gardées, ce ne sont pas des articles (à filtrer au nettoyage si besoin).
- Format : **89 %** des textes commencent par la date sur sa ligne, comme la validation (sinon : intertitre ou encadré « Sur le même sujet » avant la date).

## Période couverte
- Articles du **2005-01-28 au 2026-04-08** (fermeture en mai 2026), surtout 2007-2016 ; 485 articles de 2024-2026.

## Risques
- **Contamination** : la validation publique contient **104 articles Wikinews de 2024 à mars 2026** : ils sont dans cette collecte. Rien n'est filtré ici ; la **date de publication** est renseignée pour la décontamination (étape du nettoyage). Le test « presse » viendra d'**autres sources récentes**.
- **Qualité** : articles courts, parfois des brouillons jamais publiés (statut absent), pages de résultats sportifs (tableaux aplatis), dates manquantes pour quelques pages (date de dernière révision : 39 articles, parfois fausse, ex. JO 2010 daté 2012). **Restes de balisage** : 709 articles (3,5 %) gardent des attributs `style=`, 318 des lignes de tableau `{| |- |}` (tableaux mal fermés que mwparserfromhell ne lit pas). Une date impossible (« 31 février 2012 », page parodique).
- **Corrigé dans le code le 03/11** (effet à la prochaine collecte ; essai complet dans un dossier temporaire) : balises HTML orphelines retirées (352 articles → 0), date impossible refusée (repli sur la dernière révision), 9 ébauches passent sous 200 caractères une fois le HTML retiré (20 419 articles). Les tableaux mal fermés restent (nettoyage).
- **Biais** : presse **citoyenne et amateur**, sujets ancrés dans 2005-2016 ; style plus simple que la presse professionnelle du test.
- **RGPD** : personnes publiques citées dans l'actualité ; faits divers avec des noms de particuliers possibles.
