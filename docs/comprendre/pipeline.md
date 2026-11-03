# Le pipeline : une suite d'étapes qui prouvent qu'elles servent

## À quoi ça sert
Passer du texte **brut** de la collecte à un texte **propre**, en étapes simples, chacune mesurée.
Une commande : `uv run python -m pipeline.run --config configs/pipeline/nettoye.yaml`.

## Comment ça marche
- Une **config YAML** liste les étapes dans l'ordre : `exclude_heldout` (toujours 1re), `encoding`, `boilerplate`, `langid_score` (note), `langid` (seuil), `reflow`, `decontaminate`, `exact_dedup`.
- Chaque étape lit la sortie de la précédente et écrit `data/stages/<config>/<NN>_<étape>/` : les documents gardés, **`stats.json`** (documents et octets entrés/sortis, raisons de retrait, débit) et un **échantillon des documents retirés** (2 000 max par source, tirage déterministe).
- **`funnel.md`** résume tout : l'**entonnoir** étape × source. C'est la preuve que chaque étape sert.
- **Parallèle** : 8 workers, un fichier par worker. **Déterministe** : mêmes octets avec 1 ou 4 workers et d'un run à l'autre (vérifié sur 2 024 documents réels, 7 sources).
- **Reprise** : chaque fichier traité laisse un reçu avec une **empreinte** (paramètres, code **pris au lancement**, versions des bibliothèques, entrées, liste des IDs de la val). Relancer ne refait que ce qui a changé. Une étape sans `stats.json` n'est **pas finie**.

## Les chiffres clés
| Mesure | Valeur | D'où |
|---|---|---|
| Corpus complet (hors Wikisource) | 2,58 M documents, **9,9 Go** de texte | `01_exclude_heldout/stats.json` |
| Temps estimé sur 8 cœurs libres | ~35 min (dont ~25 pour le filtre de langue) | débits par cœur mesurés |
| Débit par cœur | encodage ~5 Mo/s · boilerplate 10 à 35 Mo/s · langue ~1,1 Mo/s · doublons ~35 Mo/s | `stats.json` |
| Échantillon tiré pour chaque raison de retrait | jusqu'à 2 000 par source | `_rejected_sample/` |
| Le champ `html` | gardé seulement jusqu'à l'étape qui l'utilise (boilerplate) | runner |

Configs : **`brut`** (texte de la collecte, seulement **décontaminé** pour que son BPB de val reste juste) sert de référence pour l'ablation ; **`nettoye`** applique tout.

## 3 questions probables du jury
1. **Comment savez-vous qu'une étape sert à quelque chose ?**
   `funnel.md` montre ce qu'elle retire (documents, Mo, raisons) et `_rejected_sample/apercu.md` montre **des exemples réels**. L'ablation `brut` contre `nettoye` mesure l'effet sur le BPB.
2. **Pourquoi tout est déterministe ?**
   Pour que l'orga **refasse exactement** notre dataset (même empreinte, BPB à ±0,02). Tirages par hachage, fichiers triés, ordre des sources fixé, aucun choix qui dépende de l'ordre d'un `set` Python (bug trouvé et corrigé dans la décontamination).
3. **Pourquoi des fichiers entre chaque étape ?**
   On peut regarder chaque étape, la relancer seule, et reprendre après une coupure sans tout refaire.
