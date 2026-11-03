# Corpus Molière provisoire (bloc 1) — `moliere`

**Usage** : bloc 1 (mini-GPT from scratch) · **Commande** : `uv run python -m pipeline.collect moliere` · **Code** : `pipeline/collect/moliere.py` · **Sortie** : `data/moliere/moliere.txt` + `data/moliere/README.md` (liste des pièces et URLs)

**Remplacé par le kit officiel** (`corpus/moliere_train.txt`, `corpus/moliere_val.txt`, éd. Chasles 1888 via Gutenberg, pièces séparées par `=====`). Ce corpus provisoire a servi en attendant le kit ; la source est retirée de `all` (on peut encore la lancer à la main).

## Origine
- **Wikisource**, *Œuvres complètes de Molière*, éd. **Charles Louandre**, Charpentier, **1910** (3 tomes, textes validés pour la plupart).
- Ordre des pièces = **tables des matières** des 3 tomes (pages `Page:Molière - Édition Louandre, 1910, tome N.djvu/…`).
- Pour chaque pièce : page principale + sous-pages (actes, préface, placets de Molière) rendues en HTML par l'**API MediaWiki** (`action=parse`, **1 requête/s**, 138 requêtes), puis converties en texte.
- Si la table pointe vers une page « éditions » (ex. `{{molière|…}}`), on suit l'**édition Louandre**, sinon la première édition transcrite (*La Princesse d'Élide* : éd. Garnier 1904).
- Gardé : **noms des personnages** (en capitales), **vers** (un par ligne), **didascalies**, préfaces de Molière. Retiré : **notices de l'éditeur** (Louandre), **lettres d'autres auteurs**, notes, numéros de vers et de page, navigation.

## Format
- UTF-8, **pièces à la suite, séparées par une ligne vide** ; **aucune ligne vide à l'intérieur d'une pièce** ; chaque pièce commence par son titre en capitales.

## Licence
- Textes de Molière : **domaine public**. Transcription Wikisource : **CC BY-SA 4.0** (citer Wikisource).

## Volume
- **33 pièces**, **53 105 lignes**, **2,27 millions de caractères** (de *La Jalousie du Barbouillé* au *Malade imaginaire*).

## Période couverte
- Pièces de **1645 à 1673** ; édition de 1910 (orthographe en partie d'époque : « j'allois », « monnoie »).

## Risques
- **Qualité** : quelques didascalies collées au texte (« Seul.Et vous »), titres et listes de personnages de mise en forme variable (« Personnages » / « PERSONNAGES »), une pièce d'une autre édition.
- **Écart avec le corpus officiel** : édition, découpage et format peuvent différer → ne pas comparer des pertes entre les deux corpus.
- **Juridique / RGPD** : aucun risque (domaine public, auteur mort en 1673).
