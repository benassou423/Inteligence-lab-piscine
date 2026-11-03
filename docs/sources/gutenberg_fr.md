# Project Gutenberg, livres en français — `gutenberg_fr`

**Domaine** : littéraire · **Commande** : `uv run python -m pipeline.collect gutenberg_fr` · **Code** : `pipeline/collect/gutenberg.py`

## Origine
- **Catalogue hors ligne** `pg_catalog.csv.gz` (version du 2026-11-01, copie dans `data/raw/gutenberg_fr/_pg_catalog.csv.gz`) : on garde les livres de type **Text** dont une des langues est **`fr`** (4 252 livres, dont 45 bilingues).
- Textes téléchargés sur un **miroir officiel** de la liste `MIRRORS.ALL` : **mirror.cs.odu.edu** (Old Dominion University). Fichier `pg<n>.txt` (UTF-8), sinon `<n>-0.txt`, `<n>-8.txt`, `<n>.txt`.
- **Pas d'aspiration de www.gutenberg.org** : le site l'interdit aux robots ; l'ancien point d'accès « robot harvest » répond 404 en 2026. Le miroir officiel `gutenberg.pglaf.org` marche aussi mais ~60 Ko/s depuis Paris.
- Pourquoi pas le dataset HF `manu/project_gutenberg` (split `fr`) ? Instantané de **2023** (614 livres français sortis depuis manquent) et **lignes en double** (même livre plusieurs fois).
- **Texte brut gardé tel quel**, **avec l'en-tête et le pied de page Gutenberg** (« *** START OF THE PROJECT GUTENBERG EBOOK … *** », licence). Les retirer est une étape de nettoyage : c'est notre démonstration du **boilerplate**. Seules retouches : BOM retiré, fins de ligne `\r\n` → `\n`.
- `date` = date de sortie dans Gutenberg (`Issued`), pas la date d'écriture de l'œuvre. `url` = `https://www.gutenberg.org/ebooks/<n>`.

## Licence
- **Project Gutenberg License** : œuvres du **domaine public aux États-Unis**. Réutilisation libre ; la marque « Project Gutenberg » doit être **retirée** si on redistribue des textes modifiés (raison de plus pour enlever l'en-tête au nettoyage).

## Volume
- **4 228 livres**, **1 786 Mo** de texte, 43 fichiers, 624 Mo sur disque.
- Écartés : 22 livres sans fichier texte sur le miroir (seulement HTML), 2 livres sortis **après le gel** (22/10/2026).

## Période couverte
- Dates de sortie Gutenberg : **1997-01-01 → 2026-10-15**. Les œuvres elles-mêmes datent surtout du XVIe au début du XXe siècle.

## Risques
- **Juridique** : domaine public **aux États-Unis**, pas forcément **en France** (70 ans après la mort de l'auteur ou du traducteur). Quelques auteurs morts après 1956 peuvent être concernés → à vérifier avant de **publier** le dataset.
- **Gel** : les fichiers sont régénérés par Gutenberg (date de modification 10-11/2026) ; le contenu (l'œuvre) est ancien, seules des corrections typographiques peuvent dater d'après le 16/10.
- **Qualité** : boilerplate (en-tête, licence en anglais ~18 000 caractères par livre), notes du transcripteur, tables des matières, illustrations « [Illustration] », mots coupés.
- **Format ≠ validation** : les fichiers `.txt` sont **coupés à ~70 caractères par ligne** ; les extraits Gutenberg de la validation du kit ont des **paragraphes sur une seule ligne** (séparés par une ligne vide). Recoller les lignes d'un paragraphe est une étape de **nettoyage** à prévoir.
- **Contamination** : la validation contient des extraits de **13 livres mis en ligne de juillet à octobre 2026** (ebooks 79119 à 79796), présents ici (date `Issued` ≤ 15/10). Rien n'est filtré à la collecte : la date de chaque livre est renseignée pour la **décontamination**.
- **Langue** (mesuré le 03/11, part des mots outils fr/en/de dans le corps) : **13 livres** ne sont pas majoritairement en français (bilingues `en`+`fr` dominés par l'anglais, latin, tables de la classification décimale), soit 8,5 M caractères (**0,5 %**). Gardés à la collecte : le filtre de langue du nettoyage décide.
- **Doublons** (mesuré le 03/11) : **2 paires de corps identiques** sous deux numéros, dont le fichier du miroir contient **le texte d'un autre livre** (n° 47756 = texte du 47632, n° 52443 = texte du 50356 ; le titre de `meta` est alors faux) ; **6 œuvres** en double (même titre et mêmes auteurs, 12 livres : *Le Tour du monde en 80 jours* 800/3456, *La Tulipe noire* 1910/26504…). Plus les éditions et volumes d'une même œuvre, et le recouvrement avec **Wikisource** → déduplication (après retrait de l'en-tête, qui diffère d'un numéro à l'autre).
- **Biais** : **français ancien** (orthographe « j'avois », « foible »), registre littéraire du XIXe siècle, canon masculin et occidental ; loin du français de 2026.
- **RGPD** : négligeable (auteurs morts depuis longtemps).
