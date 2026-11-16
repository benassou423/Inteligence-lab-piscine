# Banc d'essai : comparer des datasets avec le kit officiel

## À quoi ça sert

Tester une idée de dataset **en 4 à 10 min sur le Mac** avant de la payer en GPU : même modèle, même code que l'orga (le kit `ftb/`, jamais modifié), on change seulement les données et on lit le **BPB par genre**.

## Comment ça marche

- **Mix** (`bench/mix.py`) : prend les sorties du pipeline, applique **poids** (part tenue en octets, au demi-document près), répétitions, plafonds et **l'ordre** (mélange, ou blocs : « le meilleur pour la fin ») → `data/mixes/<run>/train.jsonl.gz`. Le kit garde cet ordre.
- **Kit** : tokenizer entraîné par la fonction du kit sur un échantillon du mix, puis `ftb.prepare`, `ftb.train`, `ftb.evaluate` lancés **tels quels**.
- **Garde-fous** : le kit voit exactement 100 007 936 tokens. Le banc vérifie que ce budget ne **coupe pas le dernier bloc** et ne **relit pas** le début du flux ; un run de mise au point n'est jamais pris pour un run complet.
- **Proxy** : `--debug-budget` (10 à 25 M tokens) sur le GPU du Mac, non soumissible. `configs/bench/smoke.yaml` essaie toute la chaîne en ~2 min, sans rien mesurer. **Complet** : 100 M tokens, soumissible. Le banc prévoyait de le lancer sur Kaggle (`bench/kaggle.py`, reprise si la session coupe) : cette route n'a jamais été exécutée. Les deux runs complets du 10/11/2026 ont tourné sur le Mac (MPS, float32) ; le second est celui qui a été soumis le 11/11.
- Chaque run ajoute **une ligne** à `experiments/results.csv` ; `bench.compare` mesure si l'écart entre deux runs dépasse le hasard des documents de la val.

## Les chiffres clés

| Chiffre | Valeur | D'où ça vient |
|---|---|---|
| Débit sur le Mac (M5 Pro, float32) | **40 000 à 44 000 tok/s** : proxy de 10 M ≈ 4 min, de 25 M ≈ 10,5 min | runs du 03/11 |
| Datasets opposés | Wikipédia seule **1,747** (dump HF 2023) contre Molière seul **6,385** (25 M tokens) | `rank_wiki`, `rank_moliere` |
| **Bruit d'entraînement** : même recette, 2 graines de mix, 10 M tokens | **2,076 contre 1,953** : écart **0,12**, même sens sur 229 documents sur 230 | revue du 03/11 (`data/raw` : presse en partie dans la val, même écart sur les 3 autres genres) ; ces deux runs de 10 M tokens ne sont pas dans `experiments/results.csv` |
| Bruit dû au choix des documents de la val | ± 0,01 (intervalle à 95 %) | bootstrap par genre, `bench.compare` |
| Run complet, prévu | 100 M tokens, < 1 h sur T4 (float16, comme le ré-entraînement de l'orga) ; non mesuré ici | `docs/KIT_README.md` |
| Run complet, fait le 10/11 sur le Mac (MPS, float32) | 100 007 936 tokens : **1,2874** BPB en 4 423,5 s (`full_reference_20261110`), puis **1,2883** en 2 039,9 s avec les coordonnées masquées (`full_reference_masked_20261110`, le run soumis) | `experiments/results.csv` |

## 3 questions probables du jury

1. **Le proxy prédit-il le run complet ?** Pour des datasets très différents, oui : Wikipédia bat Molière à chaque évaluation, dès 6 M tokens. Pour deux mélanges **proches**, un seul run ne suffit pas : deux runs de même recette diffèrent de **0,12** à 10 M tokens. Règle : un écart ne compte que s'il dépasse ce bruit (2 graines par mélange, même tokenizer), puis on confirme à 100 M tokens. C'était prévu sur Kaggle ; cela a été fait sur le Mac le 10/11/2026, pour la seule recette de référence.
2. **D'où vient ce bruit ?** Pas de la val (± 0,01) ni des documents tirés : le modèle « chanceux » est aussi meilleur sur les données d'entraînement de l'autre run (1,93 contre 2,03). C'est l'**entraînement** lui-même. Hypothèse : un run court (306 pas, warmup réduit à 30 pas par le kit en mise au point) dépend beaucoup des premiers lots. Elle n'a pas été testée à part. Mesuré depuis à 25 M tokens : deux graines de mélange d'une même recette diffèrent de 0,013 à 0,027 BPB (`experiments/ablation_complete_20261110.md`), bien moins que 0,12.
3. **Pourquoi l'ordre des documents compte-t-il ?** Le kit lit le flux dans l'ordre (mélangé seulement par paquets de ~1,6 M tokens) : la fin, vue quand le learning rate baisse, marque davantage le modèle. Mais un flux trop long **coupe** le dernier bloc, et un flux trop court est **relu** depuis le début à la fin de l'entraînement. Le banc avertit dans les deux cas.
