# Décontamination : ne jamais apprendre la val

## À quoi ça sert
La val du kit contient des **séances d'octobre 2026**, des **livres Gutenberg de 2026** et des articles récents. Certains sont **déjà dans nos données**.
Si le modèle les apprend, notre BPB de validation devient **faux** : on perd notre seul repère.

## Comment ça marche
- **Même mesure que le kit** (`ftb/contamination.py`) : on découpe en **13-grammes** (suites de 13 mots, en minuscules) et on importe sa fonction de hachage.
- **Copie** : un document qui recouvre à lui seul **≥ 20 %** d'un document de val est **retiré en entier** (la séance, le livre ou l'article source). On prend le document de val dont la **part** recouverte est la plus grande (pas le plus grand nombre de 13-grammes) : même décision d'un run à l'autre.
- **Formule** : un 13-gramme présent dans **≥ 20 documents** d'entraînement est une formule de séance (« le scrutin est annoncé dans l'enceinte de l'Assemblée nationale »). Ce n'est pas une fuite : on le **garde**.
- **Le reste** : on retire **seulement les lignes** qui portent un 13-gramme partagé, pas toute la séance.
- **Vérification** : `uv run python -m tools.check_contamination` lance le contrôle officiel sur notre sortie. Attendu : **0 document signalé**.

## Les chiffres clés (mesurés sur toute l'Assemblée et tout Gutenberg)
| Mesure | Valeur |
|---|---|
| Vraies copies trouvées | **11 séances** (1er au 15 octobre 2026), **13 livres** (mis en ligne de juillet à octobre 2026), **2 articles** Wikipédia (dont la source de « Transports à Tours ») |
| Recouvrement d'une vraie copie | **25 % à 100 %** d'un document de val |
| Recouvrement d'une séance sans rapport | **8 % au plus** (formules, noms des orateurs, numéros d'amendements) |
| Séances qui partagent au moins un 13-gramme avec la val | **1 189 sur 1 235** : d'où le besoin d'un seuil |
| Si on retirait chaque document au moindre 13-gramme | **96 %** de l'Assemblée perdue |
| Seuil document seul, même à 2 % | 585 séances perdues, et le kit signale **encore 2** documents (formules cumulées) |
| Part d'un doc de val couverte par les seules formules gardées | **8,4 %** au plus sur la sortie finale du 03/11 (6,8 % avec 1 235 séances) : le kit signale au-delà de 10 % |

Le seuil de **20 %** tombe dans le **trou** entre 8 % (bruit) et 25 % (copie). La règle des 13-grammes vient de **GPT-3** (Brown et al., 2020) ; le kit l'applique aussi au test caché.

## 3 questions probables du jury
1. **Pourquoi ne pas retirer tout document qui partage un 13-gramme ?**
   Parce que **1 189 séances sur 1 235** en partagent : les formules de procédure sont dans toutes les séances. On perdrait presque tout le genre « dialogue » pour rien.
2. **Garder les formules, ce n'est pas tricher ?**
   Non : le test caché aura **les mêmes formules**, publiées après la piscine. Les apprendre, c'est apprendre le genre, pas recopier un texte. Elles couvrent au plus 8,4 % d'un document de val (marge à surveiller : le seuil de 20 documents est absolu, plus on a de séances, plus il y a de « formules »).
3. **Comment prouver que c'est propre ?**
   Avec l'outil **officiel** du kit sur notre sortie finale : 0 document signalé. Chaque retrait note quel document de val il recouvrait (`decontaminate_val_id`).
