# Common Crawl, pages françaises — `commoncrawl`

**Domaine** : web (non classé) · **Commande** : `uv run python -m pipeline.collect commoncrawl` · **Code** : `pipeline/collect/commoncrawl.py`, `pipeline/collect/htmltext.py`

## Origine
- Crawl **CC-MAIN-2026-39** (du 2026-09-04 au 2026-09-17) : le **dernier crawl terminé avant le gel**, choisi dans `https://index.commoncrawl.org/collinfo.json` (CC-MAIN-2026-43 finit le 18/10, après le gel). Le choix est revérifié à chaque lancement.
- **Fichiers WARC** du **premier segment** (`1788492699475.64`), dans l'**ordre trié** de `warc.paths.gz` (déterministe) : les **10 premiers** (`n_warc_files` dans `configs/collect.yaml`).
- Lecture **en flux HTTP** avec `warcio`, **rien n'est stocké sur disque** sauf les pages gardées.
- Une page est **gardée** si les langues **cld2** de son enregistrement `metadata` contiennent le **français** (à n'importe quel rang) **ou** si son domaine finit par **`.fr`**.
- Une page est **écartée** si elle **réserve la fouille de textes** (protocole TDMRep : `<meta name="tdm-reservation" content="1">` ou en-tête HTTP `tdm-reservation: 1`).
- Document : `id = commoncrawl:<WARC-Record-ID>`, `url`, `date` = WARC-Date, `html` brut décodé avec le bon charset (BOM, puis UTF-8 s'il est valide, puis charset HTTP ou `<meta>`, puis charset détecté par Common Crawl, puis windows-1252), `text` = **tout le texte visible**, menus et pieds de page compris (version « brute »). `meta` : fichier WARC, langues cld2, content-type, charset, raison du choix (`langue`, `tld`, `langue+tld`), titre.
- Robustesse : backoff exponentiel sur les **503 « Slow Down »** et 5xx, **3 téléchargements en parallèle**, **reprise** dans un fichier (requête `Range` depuis le dernier enregistrement « request » sûr) et **entre fichiers** (`_work/done.jsonl`).

## Licence
- **Conditions d'utilisation de Common Crawl** ; le **droit d'auteur** des pages reste à leurs auteurs. Base légale en France : **exception de fouille de textes et de données** (art. L122-5-3 du CPI, directive 2019/790 art. 4), valable **sauf opposition** de l'ayant droit → on respecte les **opt-out** (TDMRep). Common Crawl respecte **robots.txt** au moment du crawl (CCBot).

## Volume
- **11 865 pages**, **85,6 Mo** de texte (le HTML brut en plus : 349 Mo zstd sur disque, 10 fichiers).
- Par fichier WARC (~930 Mo, ~22 000 réponses) : **~1 190 pages gardées** (5,5 % des pages HTML).
- Écartées : 204 351 pages non françaises, 3 448 réponses non HTML, **33 réserves TDM** (~3 par fichier), 115 pages sans texte.
- **Débit mesuré** : 9,3 Go de WARC lus en 285 s, soit **~12,7 Mo/s par téléchargement** (7 à 29 Mo/s selon le fichier) et **~33 Mo/s au total** avec 3 téléchargements parallèles. Aucune coupure ni 503 sur ce run.

## Période couverte
- Pages capturées le **2026-09-04** (date de capture, pas de publication).

## Risques
- **Juridique** : l'opt-out n'est vérifié que dans la page (TDMRep) ; on ne lit ni `/.well-known/tdmrep.json` ni les règles robots.txt visant les robots d'IA (GPTBot, CCBot ajoutés après le crawl). Risque résiduel faible sur 12 000 pages, à documenter dans la datasheet.
- **RGPD** : les pages contiennent des **données personnelles** (noms, e-mails, téléphones, avis signés) → l'étape de nettoyage doit masquer e-mails et numéros.
- **Qualité** : texte brut = **menus, boutons, pieds de page, cookies**, listes de liens, spam SEO, contenu adulte possible → extraction du contenu principal (trafilatura) et filtres qualité au nettoyage.
- **Biais** : 10 fichiers d'**un seul segment** = 0,01 % du crawl, pas représentatif ; le critère `.fr` seul ajoute **181 pages** (1,5 % des pages gardées), souvent en anglais ; 7 853 pages sont gardées par la langue seule et 3 831 par les deux critères ; sur-représentation du commerce et des annuaires.
- **Doublons** : pages quasi identiques (même gabarit, pagination) → déduplication.
