# Chatbot « à la manière de Molière » — `chatbot/`

Demande du référent (réunion du 11/11) : « tu lui parles, et il te répond à la façon de Molière ». Essai : http://127.0.0.1:8766 (modèle « Molière · chatbot ») ou `uv run python -m chatbot.generate chat --model data/models/chatbot_moliere --persona TOINETTE "Je me sens mourir !"`

## 1. À quoi ça sert
L'ancien chat **prolongeait un texte** : il recevait « Utilisateur : … Assistant : », une forme **jamais vue** à l'entraînement, et répondait hors sujet (« Salut, ça va ? » → un texte sur la Californie). Le chatbot, lui, **répond** : l'utilisateur joue un personnage, le modèle écrit la **réplique suivante**.

## 2. L'idée en une phrase
**Le théâtre est déjà du dialogue** : dans une pièce, chaque réplique répond à la précédente. On apprend donc au modèle des scènes au format `NOM.` / réplique / ligne vide, puis on lui parle **dans ce même format**.

```
LA GRANGE.                      ← l'utilisateur (nom de l'interlocuteur du personnage choisi)
Que pensez-vous des médecins?

MOLIÈRE.                        ← le modèle écrit la suite et s'arrête au prochain « NOM. »
```

## 3. Les 5 étapes (un fichier chacune)
1. **Données — `theatre.py`.** Extrait des scènes en répliques : Molière (kit + Wikisource, 30 pièces, 790 scènes) et le théâtre de Wikisource/Gutenberg déjà dans notre corpus nettoyé (969 documents, 33 396 scènes, 711 078 répliques, 83 M de caractères). Trois gardes : (a) **validation tenue à l'écart** — Le Misanthrope, Le Sicilien, L'Avare n'apparaissent nulle part à l'entraînement (0 passage de 12 ou 13 mots en commun, contrôle indépendant) ; (b) **liste d'exclusion auditée** `exclusions_theatre.json` — 142 documents (34 œuvres obscènes, 68 en orthographe trop ancienne, 34 qui ne sont pas du dialogue, 6 illisibles) ; (c) copies quasi exactes d'éditions retirées (scènes à ≥ 80 % de passages communs : 16 % → 0 % ; environ 3 % de scènes à 50–80 % restent).
2. **Format — `format.py`.** Une seule fonction de rendu pour l'entraînement **et** pour le chat (même typographie, mêmes sauts de ligne). Un message ne peut pas fabriquer une fausse réplique : `clean_text` retire les lignes vides et `<|endoftext|>`, `neutralize_names` passe en minuscules une ligne « NOM. ».
3. **Affinage — `finetune.py`.** On **continue l'entraînement** d'un modèle qui sait déjà le français, avec un petit taux d'apprentissage (montée courte puis descente en cosinus), sur les scènes. Mesure : **BPB de la réplique suivante** sur les 3 pièces jamais vues, au pas 0 puis régulièrement ; on garde le meilleur point.
4. **Génération — `generate.py`.** On tire **16 réponses candidates** (température 0,8 ; top-p 0,9 ; pas de répétition de 4 tokens), on s'arrête dès que le modèle commence une autre réplique, puis on **choisit** celle qui dépend le plus du message (§ 5).
5. **Mesure — `evaluate.py`.** Un **panel fixe de 40 messages** × 3 graines, les mêmes pour chaque version.

## 4. Deux modèles
- **v1 — le modèle du concours, affiné.** 10,2 M de paramètres, celui de la soumission, puis 60 M de tokens de théâtre.
- **v2 — un modèle de démo plus gros, hors concours (c'est lui dans la démo).** Le v1 **sature** : son erreur d'entraînement égale son erreur de validation. On a donc entraîné **à partir de zéro, sans aucun poids externe**, la **même architecture** en plus large et plus profond (512 dimensions, 8 couches, 8 têtes = **29,6 M de paramètres**) : 400 M de tokens de notre corpus (mélange officiel 35 %, Gutenberg 40 %, Wikisource 25 %) en 6,1 h sur le Mac (`pretrain.py`), puis le même affinage théâtre (60 M de tokens, 52 min). Les pièces de validation, la liste d'exclusion et les pages au vocabulaire cru sont filtrées du pré-entraînement.

## 5. Comment la réponse est choisie (le point le plus fin)
- **Score** d'une candidate = Σ sur ses tokens de [log P(token | conversation) − log P(token | personnage seul)]. C'est l'**information mutuelle** : ce que la conversation apporte pour prédire la réponse. « Oui. » est probable partout → score faible ; « Je n'ai pas de téléphone portable. » après « As-tu un téléphone portable ? » → score fort.
- **Filtres avant le score**, chacun avec repli (s'il ne reste rien, on garde tout) : pas de nom de rôle, pas d'**écho** (≥ 60 % des mots du message **ou d'une réponse précédente du bot** recopiés), pas de réponse qui commence par une minuscule (indication de scène), **4 à 30 mots**. Les didascalies entre parenthèses sont retirées.
- **Ce que la mesure montre, sur les mêmes poids (v2)** : une seule réponse tirée, sans choix → reprise du sujet **2,0 %** ; choix par score **moyen** par token → 22,2 % mais **3,7 mots** (« Oui. », « Non. ») ; choix par score **total** sur 4–30 mots → **31,3 %** et 11,7 mots. L'affinage apporte la langue et le style ; **c'est le choix parmi 16 candidates qui apporte le rapport au message**.
- **Limite connue** : le score total favorise aussi la reprise *partielle* des mots du message (sous le seuil de 60 %) ; c'est souvent une vraie réponse au sujet, parfois du perroquet.

## 6. Chiffres (tous mesurés)
Panel de 40 messages × 3 graines = 120 réponses ; BPB sur les 3 pièces de Molière jamais vues. Les trois dernières lignes utilisent **exactement le même décodage**.

| Version | BPB Molière jamais vu | Reprise du sujet (hors écho) | Arrêt naturel | Mots par réponse |
|---|---|---|---|---|
| Ancien chat (prompt « Utilisateur / Assistant ») | 1,4746 | 2,0 % | 43,3 % | 47,5 (texte hors sujet) |
| Modèle du concours **non affiné**, nouveau format | 1,4746 | 18,2 % | 94,2 % | 13,6 |
| v1 : modèle du concours **affiné** | 1,2273 | 29,3 % | 100 % | 11,8 |
| **v2 : gros modèle affiné (dans la démo)** | **1,1044** | **31,3 %** | 100 % | 11,7 |

Repères du v2 : 3,8709 (initialisation aléatoire) → 1,2155 (après pré-entraînement) → 1,1044 (après affinage théâtre). « Reprise du sujet » = part des réponses qui reprennent un mot plein du message sans le recopier ; « arrêt naturel » = le modèle a fini sa réplique de lui-même.

**Jury à l'aveugle** (12 jurés IA indépendants — spectateur, professeur de français, comédien —, 80 messages, réponses anonymisées et mélangées, 240 verdicts) :

| | Rang moyen (1 = meilleur) | Classé 1er | Rapport au message | Langue | Style |
|---|---|---|---|---|---|
| v2 | **1,30** | **70 %** | 3,04 / 5 | 3,70 / 5 | 3,65 / 5 |
| v1 | 1,70 | 30 % | 2,58 | 3,16 | 3,32 |
| Ancien chat | 3,00 | 0 % | 1,07 | 1,40 | 1,04 |

Ce sont des jurés **IA**, pas des humains : à présenter comme tel.

## 7. Explique cette ligne
1. `return f"{normalize_speaker(speaker)}.\n{clean_text(text)}\n\n"` (`render_turn`, format.py) → **le format unique** : nom, point, réplique, ligne vide. Le même rendu sert à l'entraînement et au chat.
2. `pool = [t for t in pool if lo <= len(WORD.findall(t[0])) <= hi] or pool` (generate.py) → **filtre avec repli** : on ne garde que les réponses de 4 à 30 mots, mais s'il n'en reste aucune on garde tout (jamais de bulle vide). Les autres filtres ont la même forme.
3. `scores = [s * len(r) for s, r in zip(scores, replies_ids)]` → **moyenne × nombre de tokens = somme** : l'information totale tirée de la conversation.
4. `if args.warmup_steps >= max_steps: sys.exit(...)` (finetune.py) → sans pas restants après la montée du taux d'apprentissage, il n'y a pas de descente en cosinus : on refuse plutôt que d'entraîner de travers.
5. `"format_sha256": file_sha256(fmt.__file__)` (clé du cache, finetune.py) → le cache des tokens dépend du **code du format** : on ne peut pas entraîner sur un format et discuter dans un autre.

## 8. Questions probables du jury
1. **Est-ce un « vrai » chatbot ?** Non : c'est un modèle de langue de 29,6 M de paramètres entraîné de zéro. Il produit des répliques **dans le style du théâtre classique**, souvent en rapport avec le message, sans garantie de sens ni de vérité. Les assistants du commerce ont des milliers de fois plus de paramètres et de données.
2. **Avez-vous utilisé un modèle pré-entraîné ?** Non. Le v1 part de notre modèle du concours ; le v2 d'une initialisation aléatoire. Tout vient de nos données.
3. **Le chatbot change-t-il la soumission ?** Non. La soumission du 11/11 (BPB 1,288292) est intacte, le kit n'est pas modifié (`diff -r ../Piscine-kit/ftb ftb` vide), et le v2 n'est pas soumis.
4. **Pourquoi « MOLIÈRE » comme personnage ?** Il n'a que 99 répliques (L'Impromptu de Versailles). Pendant l'affinage, on ajoute des **copies de scènes de Molière où le personnage principal est renommé MOLIÈRE** : il devient la voix moyenne de ses personnages.
5. **Comment savez-vous que c'est mieux qu'avant ?** Mêmes 40 messages, mêmes graines, même décodage, une validation jamais vue, et un jury à l'aveugle. Les réponses sont lisibles dans `data/stages/chatbot_20261112/eval/final_*/transcript.md`.
6. **Harpagon et Alceste viennent des pièces de validation : comment les jouez-vous ?** Le modèle de la démo ne les a pas vus dans Molière ; il les joue par généralisation. Une variante finie sur tout Molière existe (`runs/chatbot_v2_30m_final`), mais un second jury à l'aveugle ne l'a pas préférée (rang moyen 1,95 contre 1,90 : mieux en style, moins en rapport au message) et son BPB (1,0462) n'est plus tenu à l'écart : on ne le cite pas.
7. **Qu'est-ce qui n'a pas marché ?** (a) Le premier reclassement choisissait les réponses les plus courtes (3,7 mots) ; (b) deux entraînements en même temps sur le GPU du Mac : le petit a été affamé (600 pas en 5 h) ; (c) l'extracteur coupait les répliques à chaque paragraphe (27 % de scènes de 3 répliques ou moins → 12 %) ; (d) en conversation longue, le bot recopiait ses propres réponses (42 % → 0 % après correction) ; (e) finir sur Molière seul améliore le BPB (1,1044 → 1,0974) mais pas l'avis du jury. Trois revues adverses ont confirmé 70 défauts ; la plupart sont corrigés, les autres sont écrits (voir § 9).

## 9. Limites connues
- Réponses parfois **à côté** ou absurdes ; orthographe ancienne possible (« connoître », « très-bien »).
- 0,06 % des répliques d'entraînement gardent une ligne « NOM + action » attribuée au mauvais personnage ; 65 caractères invisibles restent dans deux pièces de `moliere_train` (retirés au rendu).
- La « reprise du sujet » compte des mots repris : elle est optimisée par le choix de la réponse. Le jury à l'aveugle est la mesure indépendante, mais ses jurés sont des IA.
- Résidus détaillés : `data/stages/chatbot_20261112/reprise_finale.json` et `corrections_finales.json`.
