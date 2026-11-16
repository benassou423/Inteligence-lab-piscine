# Le pipeline : une suite d'étapes qui prouvent qu'elles servent

## À quoi ça sert
Passer du texte **brut** de la collecte à un texte **propre**, en étapes simples, chacune mesurée.
Une commande : `uv run python -m pipeline.run --config configs/pipeline/nettoye.yaml`.

## Comment ça marche
- Une **config YAML** liste les étapes dans l'ordre : `exclude_heldout` (toujours 1re), `encoding`, `boilerplate`, `langid_score` (note), `langid` (seuil), `reflow`, `decontaminate`, `exact_dedup`. `exclude_heldout` lit les listes `data/val/*/heldout_ids.txt` : elles ne sont pas dans le dépôt et aucun outil du dépôt ne les crée. Sans elles, l'étape ne retire rien et l'écrit dans son `stats.json` (« aucune liste d'IDs exclus trouvée ») ; c'est alors `decontaminate` qui protège la val.
- Chaque étape lit la sortie de la précédente et écrit `data/stages/<config>/<NN>_<étape>/` : les documents gardés, **`stats.json`** (documents et octets entrés/sortis, raisons de retrait, débit) et un **échantillon des documents retirés** (2 000 max par source, tirage déterministe).
- **`funnel.md`** résume tout : l'**entonnoir** étape × source. C'est la preuve que chaque étape sert.
- **Parallèle** : 8 workers, un fichier par worker. **Déterministe** : mêmes octets avec 1 ou 4 workers et d'un run à l'autre (vérifié sur 2 024 documents réels, 7 sources).
- **Reprise** : chaque fichier traité laisse un reçu avec une **empreinte** (paramètres, code **pris au lancement**, versions des bibliothèques, entrées, liste des IDs de la val). Relancer ne refait que ce qui a changé. Une étape sans `stats.json` n'est **pas finie**.

## Les chiffres clés
| Mesure | Valeur | D'où |
|---|---|---|
| Corpus après `nettoye` (7 sources) | **3 119 274 documents**, 17 352 Mo de texte | `docs/remise/RAPPORT.md`, § 2 |
| Corpus brut actif (7 sources) | 3 223 847 documents, 18 256,3 Mo de texte | `docs/sources/volumes.md` (04/11) |
| Ancienne mesure, d'avant le 04/11 (hors Wikisource, ancienne Wikipédia HF 2023) | 2,58 M documents, **9,9 Go** de texte | `01_exclude_heldout/stats.json` (non publié) |
| Temps estimé sur 8 cœurs libres, pour ce corpus d'avant le 04/11 | ~35 min (dont ~25 pour le filtre de langue) | débits par cœur mesurés |
| Débit par cœur | encodage ~5 Mo/s · boilerplate 10 à 35 Mo/s · langue ~1,1 Mo/s · doublons ~35 Mo/s | `stats.json` |
| Échantillon tiré pour chaque raison de retrait | jusqu'à 2 000 par source | `_rejected_sample/` |
| Le champ `html` | gardé seulement jusqu'à l'étape qui l'utilise (boilerplate) | runner |

Configs : **`nettoye`** applique tout : c'est la référence. **`brut`** (texte de la collecte, seulement **décontaminé** pour que son BPB de val reste juste) décrit le point de comparaison de l'ablation ; les deux runs bruts mesurés le 10/11 ont été préparés par `tools/prepare_raw_ablation.py`, qui applique les deux mêmes étapes (`exclude_heldout`, `decontaminate`) à un tirage de 100 Mo par graine.

Variantes essayées et non retenues : `nettoye_cible_v1` ([nettoyage_cible.md](nettoyage_cible.md)) et `complet_20261110` ([pipeline_complet.md](pipeline_complet.md)). `nettoye_hf2023` (ancienne Wikipédia) n'a pas été lancée ([fiche de la source](../sources/wikipedia_fr_hf2023.md)).

## 3 questions probables du jury
1. **Comment savez-vous qu'une étape sert à quelque chose ?**
   `funnel.md` montre ce qu'elle retire (documents, Mo, raisons) et `_rejected_sample/apercu.md` montre **des exemples réels**. L'ablation brut contre nettoyé mesure l'effet sur le BPB : le 10/11, **1,657747** (brut) contre **1,622266** (nettoyé), moyenne de deux tirages à 25 M tokens (`experiments/ablation_complete_20261110.md`).
2. **Pourquoi tout est déterministe ?**
   Pour que l'orga **refasse exactement** notre dataset (même empreinte, BPB à ±0,02). Tirages par hachage, fichiers triés, ordre des sources fixé, aucun choix qui dépende de l'ordre d'un `set` Python (bug trouvé et corrigé dans la décontamination).
3. **Pourquoi des fichiers entre chaque étape ?**
   On peut regarder chaque étape, la relancer seule, et reprendre après une coupure sans tout refaire.

## Après le pipeline : le jeu soumis
Le pipeline s'arrête au corpus nettoyé. Le jeu soumis en est tiré par le banc d'essai ([banc_essai.md](banc_essai.md)) : un tirage déterministe de **72 612 documents** (419,30 Mo), à proportions naturelles, mélangés avec la graine 1337. `tools/prepare_masked_reference.py` y masque ensuite e-mails, téléphones et adresses IP : 548 textes modifiés, aucun document retiré. Résultat à 100 007 936 tokens : **1,288292 BPB** sur la val. Détail : [RAPPORT.md](../remise/RAPPORT.md) et [DATASHEET.md](../remise/DATASHEET.md).
