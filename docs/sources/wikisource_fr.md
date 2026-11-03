# Wikisource en français — `wikisource_fr`

**Domaine** : littéraire · **Commande** : `uv run python -m pipeline.collect wikisource_fr` · **Code** : `pipeline/collect/wikisource.py`, `pipeline/collect/wikitext.py`

## Origine
- **Dump officiel** `frwikisource-20261001-pages-articles` (dumps.wikimedia.org, **1er octobre 2026**, 4 fichiers bz2, 2,6 Go), téléchargé (2 connexions à la fois), lu en flux, puis **supprimé**.
- Pourquoi pas le dataset HF **`wikimedia/wikisource`** (config `20231201.fr`, la plus récente) ? Il ne contient que l'espace principal, dont la plupart des pages ne font que **transclure** l'espace « Page: » : **23 201 pages, texte médian de 146 caractères** (liens interlangues, catégories). Le vrai texte relu est dans l'espace **Page:** du dump.
- Deux sortes de documents :
  - **un document par livre scanné** (`id = wikisource_fr:livre:<fichier>`) : pages « Page:<livre>/<n> » **relues** (niveau **3 = corrigé** ou **4 = validé**), triées par numéro et recollées (mots coupés `{{tiret}}` recollés, phrase qui continue d'une page à l'autre) ; métadonnées de la page « Livre: » (titre, auteur, année, type, volume) ;
  - **un document par page de l'espace principal** qui porte **son propre texte** (`id = wikisource_fr:page:<id>`), hors redirections, listes d'éditions et pages qui transcluent.
- Conversion **wikitexte → texte** maison (mwparserfromhell) : texte des modèles de mise en forme gardé (`{{c|…}}`, `{{sc|…}}`), personnages de théâtre en capitales (`{{Personnage|Alceste}}` → `ALCESTE.`), notes `<ref>`, catégories, navigation et en-têtes/pieds de page (`<noinclude>`) retirés ; un saut de ligne simple = espace, comme dans MediaWiki, sauf dans `<poem>`.
- `date` = **dernière révision** des pages du document (≤ 2026-10-01). `meta.is_theatre` : heuristique « au moins 20 lignes « NOM DU PERSONNAGE » en capitales (≥ 4 % des lignes) et au moins une scène ».

## Licence
- Œuvres du **domaine public** ; transcription et mise en forme Wikisource sous **CC BY-SA 4.0** (citer Wikisource, partage à l'identique).

## Volume
- **21 600 documents**, **5 344 Mo** de texte, 73 fichiers (1,8 Go zstd) : **20 554 livres** (5 329 Mo) et **1 046 pages** de l'espace principal (15 Mo) ; 907 documents marqués théâtre.
- 4 659 968 pages lues ; pages `Page:` écartées : 273 734 de niveau 0, 682 122 de niveau 1, 628 de niveau 2. Détail : `data/raw/wikisource_fr/_stats.json` (dont les modèles inconnus).

## Qualité de la conversion (mesurée le 03/11 sur 200 livres, tirage déterministe par empreinte de l'id)
- **Restes de balisage** dans 60 livres sur 200, mais **0,4 % des caractères** : surtout des tableaux wiki mal fermés (`{| … |- … |}`, 0,29 %), du HTML ouvert sur une page et fermé sur la suivante (`<div class="titre">`, `</div>`), des valeurs CSS seules (« 0.1em »).
- **Modèles mal rendus** : `{{style|css|texte}}` donnait le CSS, `{{lp}}`, `{{lia}}`, `{{hwp}}` la cible du lien au lieu du texte affiché, `{{StdT|s|texte}}` la lettre « s », `{{ligat|f|f}}` une seule lettre (« afaire »), `{{Pom|…}}` (poème) perdait ses retours à la ligne. Plus de 150 000 appels. **Corrigé dans `wikitext.py` le 03/11** (avec les restes HTML et CSS) : effet à la **prochaine collecte** (les fichiers actuels de `data/raw/` sont d'avant). Contrôle sur 121 pages relues tirées au hasard sur fr.wikisource : 2 pages changent, toutes deux corrigées (vers rétablis, « fig. 21 » au lieu de « ch-5 »), aucune régression.
- **Mots coupés en fin de page** sans `{{tiret}}` (« principa- lement ») : 873 dans les 200 livres ; corrigé de même (recollés sans tiret).
- **Pages de garde** (éditeur, « LIBRAIRIE HACHETTE », « Droits réservés ») dans 48 livres sur 200, **tables des matières** dans 35 : gardées (boilerplate, nettoyage).

## Période couverte
- Dernières révisions du **2007-09-11 au 2026-10-01** (date de transcription, **pas** de publication de l'œuvre). Les œuvres datent surtout du XVIe au début du XXe siècle (champ `meta.year` pour les livres).

## Risques
- **Qualité** : pages **non relues** (niveaux 0-2 : OCR brut) **écartées** dès la collecte ; conversion des modèles **imparfaite** (modèles rares rendus par leur premier paramètre, tableaux aplatis) ; titres courants et numéros oubliés hors `<noinclude>`.
- **Domaine mal étiqueté** : Wikisource contient aussi des **revues** (*Revue des Deux Mondes* → presse), des lois, des thèses, l'*Encyclopédie* de Diderot → `meta.type` (book, journal, collection…) aide à reclasser.
- **Doublons** : une même œuvre en plusieurs éditions, et recouvrement avec **Gutenberg** → déduplication.
- **Biais** : **français ancien** (« foible », « connoissois »), canon classique ; loin de la langue de 2026.
- **Juridique** : quelques traductions ou textes d'auteurs morts il y a moins de 70 ans peuvent être hébergés sous d'autres régimes ; à vérifier avant publication. **RGPD** : négligeable (noms des contributeurs non collectés).
