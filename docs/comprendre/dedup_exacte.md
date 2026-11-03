# Doublons exacts : chaque texte une seule fois

## À quoi ça sert
Un texte présent 10 fois est appris 10 fois : il **vole du budget** aux autres et pousse le modèle à **réciter**.
On garde **une seule copie** de chaque texte, toutes sources confondues.

## Comment ça marche
- On **normalise** le texte : minuscules, espaces compactés. « Le  CHAT » = « le chat ».
- On calcule une **empreinte** de 128 bits (xxh3) : deux textes identiques ont la même empreinte.
- Étape **globale** en deux passes : 1) les workers calculent les empreintes ; 2) on garde la **première occurrence**.
- L'ordre est fixé : sources dans l'ordre de la config (**Wikipédia d'abord, web en dernier**), fichiers triés, lignes dans l'ordre. Même résultat à chaque exécution.
- Chaque document garde `exact_dedup_copies` (nombre de copies) ; chaque retrait note **quelle copie** a été gardée.

## Les chiffres clés
| Mesure | Valeur | D'où |
|---|---|---|
| Collision d'empreinte | ~1 chance sur 10²⁵ pour 10 millions de textes (paradoxe des anniversaires : n² / 2¹²⁹) | calcul |
| Échantillon `--limit 2000` | 7 doublons, tous dans Common Crawl (pages « mentions », formulaires) | `funnel.md` |
| Vitesse | ~20 à 50 Mo/s par cœur | `stats.json` |

Références : Lee et al. 2022, *Deduplicating Training Data Makes Language Models Better* (moins de récitation, même qualité avec moins de données). CCNet (Wenzek et al. 2019) déduplique aussi par empreinte.

## 3 questions probables du jury
1. **Pourquoi des doublons exacts seulement ?**
   C'est la v1 : simple, sûre, sans faux positif. Les **quasi-doublons** (même texte avec une date changée) demandent MinHash : c'est l'étape suivante.
2. **Pourquoi Wikipédia garde la copie, et pas le site web ?**
   À texte égal, la source soignée a de meilleures métadonnées (licence, date, domaine). Le web passe en dernier dans `source_order`.
3. **Pourquoi normaliser la casse et les espaces ?**
   Une page qui recopie un article en changeant juste les retours à la ligne est la même chose pour le modèle. Le document gardé, lui, n'est **pas** modifié.
