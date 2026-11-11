---
pretty_name: Feed the Beast - référence française 20261110
language:
  - fr
task_categories:
  - text-generation
size_categories:
  - 10K<n<100K
license: other
---

# Feed the Beast - recette de référence française

**Version finale du 11/11/2026.** Mélange préparé le 10/11 et retenu pour la première soumission du modèle Mac. Fichiers de diffusion : [GitHub Release ftb-final-20261110](https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/ftb-final-20261110). Les conditions d’utilisation restent propres à chaque source ; aucune licence uniforme n’est revendiquée.

## Usage et contenu

Jeu pédagogique de 72 612 documents, 419 295 632 octets de texte UTF-8, préparé pour le projet ECE Paris « Feed the Beast ». Le modèle officiel a 10 166 400 paramètres et doit voir 100 007 936 tokens. Le flux préparé en contient 114 701 016, ce qui évite de relire son début.

Le mélange reprend les proportions naturelles du corpus nettoyé : Wikipédia 46,24 %, Wikisource 32,04 %, Gutenberg 10,41 %, FineWeb-2 9,62 %, Assemblée nationale 1,44 %, Wikinews 0,14 %, Common Crawl 0,11 % des octets. Ces parts ne décrivent ni les tokens exactement vus ni une représentation équilibrée du français.

## Fichiers à diffuser ensemble

| Fichier | Rôle |
|---|---|
| `train.jsonl.gz` | Une ligne JSON avec `text`, dans l’ordre d’entraînement |
| `tokenizer.json` | Tokenizer sans perte, vocabulaire 8 192 |
| `manifest.json` | Sélection, quantités, sources et empreintes |
| `expected_data_meta.json` | Empreinte et dimensions du flux attendu |
| `provenance.jsonl.gz` | Pour chaque ligne : ID, URL, date, licence déclarée, source, SHA-256 avant/après et masques appliqués |
| `provenance_report.json` | Couverture de provenance, contrôles et hash des fichiers |
| `DATASHEET.md` | Motivation, composition, traitement, usages, limites et maintenance |

Les 72 612 lignes sont reliées aux documents originaux par leurs SHA-256 avant et après transformation. Le masquage structuré modifie 548 textes : 423 occurrences d’emails, 486 de téléphones et 4 d’adresses IP. Aucun article n’est supprimé et l’ordre est conservé. Le contrôle trouve zéro URL et zéro licence déclarée manquantes. Les informations de licence viennent des collecteurs ; leur présence ne vaut pas une validation juridique indépendante.

## Sources et licences déclarées

| Source | Déclaration du collecteur |
|---|---|
| Wikipédia FR, dump 20261001 | CC BY-SA 4.0 |
| Wikisource FR | Statut variable selon l’œuvre ; licence des transcriptions et conditions de Wikisource |
| Projet Gutenberg | Project Gutenberg License, domaine public aux États-Unis |
| Wikinews FR | CC BY 2.5 |
| Assemblée nationale | Licence Ouverte 2.0, Etalab |
| FineWeb-2 | ODC-By 1.0 et conditions de Common Crawl, droits des pages conservés par leurs auteurs |
| Common Crawl | Conditions de Common Crawl, droits des pages conservés par leurs auteurs |

Le champ `license: other` évite d’attribuer une licence unique au mélange. Consulter les conditions de chaque source et les informations de provenance avant toute réutilisation ou redistribution. Les auteurs et historiques des pages d’origine doivent rester consultables.

## Traitement

Les données sont gelées au 16/10/2026. Le pipeline exclut les IDs de référence, répare l’encodage, extrait le contenu principal, calcule un score de langue, garde le français à `fr >= 0,65`, remet les livres en paragraphes, décontamine puis retire les doublons exacts. La configuration de référence est `configs/pipeline/nettoye.yaml` ; le mélange est ensuite transformé par `tools/prepare_masked_reference.py`.

Le contrôle officiel sur le mélange complet signale zéro document de validation sur 230. Il utilise des 13-grammes et un seuil de recouvrement de 10 %. Il ne prouve pas l’absence de toute phrase commune. La variante qualité/MinHash a été évaluée séparément et non retenue : BPB moyen 1,632570 contre 1,622266 pour la référence. Le mélange final conserve cette référence avec masquage structuré. Les noms, adresses postales, formats obfusqués et de nombreux téléphones internationaux ne sont pas couverts ; aucune anonymisation exhaustive n’est revendiquée.

## Limites

Les textes comportent des biais de période et de domaine, des erreurs de numérisation, des catalogues, du texte bilingue et des résidus de conversion. Les documents publics peuvent contenir des données personnelles. Les filtres de langue ont des erreurs, et le contrôle des extraits est un classement automatique, distinct d’une relecture manuelle indépendante. Le jeu sert à la recherche pédagogique et ne constitue pas une source de vérité pour des décisions importantes.

## Reproduction

```python
from datasets import load_dataset
data = load_dataset('json', data_files='train.jsonl.gz', split='train')
```

Préparer le flux avec `ftb.prepare`, le tokenizer fourni et la même version de `tokenizers` (**0.23.3**). Comparer les empreintes avant l’apprentissage. Le modèle est initialisé aléatoirement par le kit. Le run GPU doit utiliser la précision float16 prévue par ce kit.

- Flux : `b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee`
- Tokenizer : `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`
- Texte gzip : `910ae65f995bc3bc4685532b551422c0e3694f3eec6e474907f8fea96823dd7d`
- Index de provenance : `e129e72468ef6ae205838460afbcecfceb7c34b3f8245ee9b70f9290ceb3de4e`

## Responsabilité et maintenance

Responsable du projet : Benjamin Assouline, ECE Paris. La préparation et la documentation bénéficient d’une assistance IA, indiquée dans le dépôt. Les corrections et retraits doivent produire une nouvelle version identifiée ; conserver l’ancienne empreinte pour l’audit du run soumis. Le modèle soumis a été entraîné sur Mac en MPS float32. Aucune comparaison CUDA préalable n’est revendiquée ; le contrôle T4 de l’organisateur reste distinct de cette première soumission.
