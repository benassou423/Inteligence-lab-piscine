# Chatbot « à la manière de Molière » — `chatbot/`

Demande du référent (réunion du 11/11) : « tu lui parles, et il te répond à la façon de Molière ».

**État au 16/11/2026.** Le modèle actuel est le « modèle M » : 54 749 440 paramètres, entraîné de zéro sur une carte NVIDIA GeForce RTX 5090 (§ 4). Les modèles v1 et v2, plus anciens, sont gardés ici comme **historique**. Le modèle publié est **S03** : le modèle M affiné avec 3,02 % de conversations de synthèse (§ 4). Il est resté le modèle retenu après les essais du 16/11 (§ 6). Empreinte de son `ckpt.pt` (sha256) : `01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577`.

Essai, depuis la racine du dépôt :

```bash
python -m chatbot.generate chat --model DOSSIER_DU_MODELE --persona TOINETTE "Je me sens mourir !"
```

- `DOSSIER_DU_MODELE` : le dossier du modèle (`ckpt.pt`, `tokenizer.json`, `chat_config.json`). Les poids du modèle publié sont la pièce jointe `chatbot_moliere_s03.zip` (240 087 338 octets, sha256 `6d62aa50f4917fbb23d24e24a8d64807bde55f273c9cb4203c6970c32c886417`) de la release `chatbot-moliere-20261116` du dépôt `benassou423/Inteligence-lab-piscine` : <https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116>. Elle contient le dossier `chatbot_moliere/` (quatre fichiers, dont `training_report.json`). Après décompression : `python -m chatbot.generate chat --model chatbot_moliere "Bonjour, comment allez-vous ?"`.
- Avec le modèle publié, la commande n'a pas besoin des options de décodage `--anchor-mu 1 --candidates 32` (le choix de réponse retenu le 14/11, § 5). Sans ces options, le chat suit le `chat_config.json` du modèle, et celui du modèle publié porte déjà `"candidates": 32` et `"anchor_mu": 1.0`. Les options restent permises et ne changent rien. Avec un modèle exporté avec la recette par défaut (`chatbot/recettes_rtx.json`), qui garde 16 candidates et `"anchor_mu": 0.0`, il faut les écrire pour obtenir le même choix de réponse.
- Il faut l'environnement Python du projet (Python 3.12, PyTorch, tokenizers).
- Sans message, le chat est interactif. Autres options : `--persona` (11 personnages, MOLIÈRE par défaut), `--seed`, `--device` (`cpu` par défaut).
- Sur le PC Windows à carte NVIDIA : `.venv\Scripts\python.exe` à la place de `python`, avec `PYTHONUTF8=1` ; jamais `uv run` ni `uv sync` (ils installeraient un PyTorch sans CUDA). Le guide est [`PC_RTX.md`](../../PC_RTX.md).

À lire aussi : la fiche du modèle ([`docs/chatbot/FICHE_MODELE.md`](../chatbot/FICHE_MODELE.md)), la fiche des données ([`docs/chatbot/FICHE_DONNEES.md`](../chatbot/FICHE_DONNEES.md)) et les licences ([`LICENCES.md`](../../LICENCES.md)).

## 1. À quoi ça sert
L'ancien chat **prolongeait un texte** : il recevait « Utilisateur : … Assistant : », une forme **jamais vue** à l'entraînement, et répondait hors sujet (« Salut, ça va ? » → un texte sur la Californie). Le chatbot, lui, **répond** : l'utilisateur joue un personnage, le modèle écrit la **réplique suivante**.

## 2. L'idée en une phrase
**Le théâtre est déjà du dialogue** : dans une pièce, chaque réplique répond à la précédente. On apprend donc au modèle des scènes au format `NOM.` / réplique / ligne vide, puis on lui parle **dans ce même format**.

```
LA GRANGE.                      ← l'utilisateur (nom de l'interlocuteur du personnage choisi)
Que pensez-vous des médecins?

MOLIÈRE.                        ← le modèle écrit la suite et s'arrête au prochain « NOM. »
```

## 3. Les 6 étapes (fichiers de `chatbot/`, sauf mention)
1. **Données — `theatre.py`.** Extrait des scènes en répliques : Molière (kit + Wikisource, 30 pièces, 790 scènes) et le théâtre de Wikisource/Gutenberg déjà dans notre corpus nettoyé (969 documents, 33 396 scènes, 711 078 répliques, 83 M de caractères). Trois gardes : (a) **validation tenue à l'écart** — Le Misanthrope, Le Sicilien, L'Avare n'apparaissent nulle part à l'entraînement (0 passage de 12 ou 13 mots en commun, contrôle indépendant) ; (b) **liste d'exclusion auditée** `exclusions_theatre.json` — 142 documents (34 œuvres obscènes, 68 en orthographe trop ancienne, 34 qui ne sont pas du dialogue, 6 illisibles) ; (c) copies quasi exactes d'éditions retirées (scènes à ≥ 80 % de passages communs : 16 % → 0 % ; environ 3 % de scènes à 50–80 % restent).
2. **Format — `format.py`.** Une seule fonction de rendu pour l'entraînement **et** pour le chat (même typographie, mêmes sauts de ligne). Un message ne peut pas fabriquer une fausse réplique : `clean_text` retire les lignes vides et `<|endoftext|>`, `neutralize_names` passe en minuscules une ligne « NOM. ».
3. **Affinage — `finetune.py`.** On **continue l'entraînement** d'un modèle qui sait déjà le français, avec un petit taux d'apprentissage (montée courte puis descente en cosinus), sur les scènes. Mesure : **BPB de la réplique suivante** sur les 3 pièces jamais vues, au pas 0 puis régulièrement ; on garde le meilleur point.
4. **Génération — `generate.py` et `ancrage.py`.** On tire **32 réponses candidates** (température 0,8 ; top-p 0,9 ; pas de répétition de 4 tokens), on s'arrête dès que le modèle commence une autre réplique, puis on **choisit** celle qui dépend le plus du **dernier message** (§ 5). Jusqu'au 14/11 : 16 candidates, et un choix sur toute la conversation. Ces anciennes valeurs (16 candidates, sans ancrage) sont encore les valeurs par défaut du code ; le réglage retenu se met dans le `chat_config.json` du modèle (clés `candidates` et `anchor_mu`) ou sur la ligne de commande (`--candidates 32 --anchor-mu 1`).
5. **Mesure — `evaluate.py`.** Un **panel fixe de 40 messages** × 3 graines, les mêmes pour chaque version. Pour le modèle M s'ajoute un **jugement à l'aveugle** des réponses par 3 juges IA (§ 6) ; ses messages de test sont dans `tools/nuit_moliere.py`.
6. **Synthèse (option) — `synthetic.py` et `tools/campagne_synthese.py`.** Une petite part de conversations écrites par un modèle à poids ouverts, mélangée à l'affinage (§ 7).

## 4. Trois modèles
Les deux premiers sont **historiques**. Le troisième est le modèle actuel.

- **v1 (historique) — le modèle du concours, affiné.** 10 166 400 paramètres, celui de la soumission, puis 60 M de tokens de théâtre.
- **v2 (historique) — l'ancienne démo du Mac, plus grosse, hors concours.** Le v1 **sature** : son erreur d'entraînement égale son erreur de validation. On a donc entraîné **à partir de zéro, sans aucun poids externe**, la **même architecture** en plus large et plus profond (512 dimensions, 8 couches, 8 têtes = **29,6 M de paramètres**) : 400 M de tokens de notre corpus (mélange officiel 35 %, Gutenberg 40 %, Wikisource 25 %) en 6,1 h sur le Mac (`pretrain.py`), puis le même affinage théâtre (60 M de tokens, 52 min). Les pièces de validation, la liste d'exclusion et les pages au vocabulaire cru sont filtrées du pré-entraînement. Ce modèle est remplacé par le modèle M.
- **Modèle M (actuel), hors concours.** Toujours la même architecture (classe `GPT` de `ftb/model.py`, non modifiée), encore plus grande : 640 dimensions, 10 couches, 10 têtes = **54 749 440 paramètres**, fenêtre de 512 tokens, vocabulaire de 8 192 tokens. Entraîné **à partir de zéro, sans aucun poids externe**, sur **2 000 027 648 tokens** de notre corpus (30 518 pas ; tirage : Gutenberg 27 %, Wikipédia 22 %, FineWeb-2 20 %, Wikisource 18 %, jeu du concours 12 %, Wikinews 1 %) en 5 988,8 secondes sur une carte NVIDIA GeForce RTX 5090 (`pretrain.py`, lancé par `pipeline_rtx.py`). Filtres du pré-entraînement : liste d'exclusion, textes de moins de 200 caractères, vocabulaire sexuel cru, recouvrement avec les pièces de validation. Puis affinage théâtre : 90 013 696 tokens (2 747 pas) ; on garde le meilleur point, au pas 1 000. Les rapports de ce lancement sont dans `experiments/chatbot_rtx/20261113_1956_M/`. Trois affinages sont à distinguer :
  - **A0** (13/11) : affinage **sans synthèse**.
  - **S03** (15/11) : le même affinage avec **3,02 %** de conversations de synthèse (§ 7), recette de l'essai `chatbot/recettes_rtx_v5_s03.json`. Au jugement du 15/11 : 100 réponses sans défaut sur 144, contre 82 pour A0 (§ 6).
  - **Modèle publié : S03.** Recette exacte : `chatbot/recettes_rtx_publie.json` (étiquette « recettes-rtx-publie-s03 »). C'est la recette de l'essai plus deux déclarations qui ne changent pas l'entraînement : le fichier de synthèse et le décodage écrit à l'export (`--anchor-mu 1 --candidates 32`). Son `training_report.json` déclare 2 720 768 tokens de synthèse sur 90 013 696 tokens d'affinage, soit 0,0302. Le `ckpt.pt` (239 991 123 octets, sha256 `01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577`) est, octet pour octet, celui qui a été jugé à l'aveugle les 15 et 16/11. Le `tokenizer.json` fait 555 175 octets (sha256 `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`).
  - **Essais du 16/11 (« v6 »)** : le même affinage avec les conversations des campagnes v4 et v5 réunies, à 3, 5, 10 et 15 % de synthèse. Aucun n'a remplacé S03 (§ 6). Ces quatre modèles ne sont pas publiés ; leurs recettes (`chatbot/recettes_rtx_v6_s03.json`, `_s05`, `_s10`, `_s15`) et leurs réponses jugées le sont.

Pour refaire le modèle M : `python -m chatbot.pipeline_rtx` (sous-commandes `telecharger`, `verifier`, `lancer --taille M`, `resultats`). Une autre recette se choisit avec `--recettes`, placé avant la sous-commande. Le guide est [`PC_RTX.md`](../../PC_RTX.md) ; le détail de l'entraînement est dans la fiche du modèle.

## 5. Comment la réponse est choisie (le point le plus fin)
- **Score** d'une candidate = Σ sur ses tokens de [log P(token | conversation) − log P(token | personnage seul)]. C'est l'**information mutuelle** : ce que la conversation apporte pour prédire la réponse. « Oui. » est probable partout → score faible ; « Je n'ai pas de téléphone portable. » après « As-tu un téléphone portable ? » → score fort.
- **Filtres avant le score**, chacun avec repli (s'il ne reste rien, on garde tout) : pas de nom de rôle, pas d'**écho** (≥ 60 % des mots du message **ou d'une réponse précédente du bot** recopiés), pas de réponse qui commence par une minuscule (indication de scène), **4 à 30 mots**. Les didascalies entre parenthèses sont retirées.
- **Ce que la mesure montre, sur les mêmes poids (mesure historique : v2, 16 candidates, sans ancrage)** : une seule réponse tirée, sans choix → reprise du sujet **2,0 %** ; choix par score **moyen** par token → 22,2 % mais **3,7 mots** (« Oui. », « Non. ») ; choix par score **total** sur 4–30 mots → **31,3 %** et 11,7 mots. L'affinage apporte la langue et le style ; **c'est le choix parmi les candidates qui apporte le rapport au message**.
- **Limite connue** : le score total favorise aussi la reprise *partielle* des mots du message (sous le seuil de 60 %) ; c'est souvent une vraie réponse au sujet, parfois du perroquet.
- **Ancrage sur le dernier message (`ancrage.py`, retenu le 14/11).** Le score ci-dessus récompense aussi un mot que le bot a déjà dit deux tours plus tôt : le bot se met à tourner autour des mêmes mots. On retire donc du score ce qui était prévisible **sans lire le dernier message**. Pour cela, on note la réponse une fois de plus, après la même conversation où le dernier message est remplacé par « … ». Avec le réglage retenu (`anchor_mu` = 1, **32 candidates**), le score devient log P(réponse | conversation) − log P(réponse | conversation sans le dernier message) : ce que le dernier message, et lui seul, apporte.
- **Ce que l'ancrage change, mesuré le 14/11 sur le modèle M** (144 réponses, 3 juges IA à l'aveugle, règle fixée avant ; ancrage à 32 candidates contre l'ancien choix à 16 candidates) : rapport au message **96 → 114** ; français correct 125 → 120 ; style 137 → 133. Aux tours 2 et suivants d'une conversation : rapport 29 → 43 sur 54. Pris seuls, l'ancrage à 16 candidates (rapport au message : +2 sur 72) et les 32 candidates sans ancrage (−3 sur 72) ne suffisent pas. Coût, mesuré le 14/11 pendant ce choix : 0,81 seconde par réponse contre 0,47 sur la RTX 5090. L'évaluation du modèle publié, refaite le 16/11 avec le réglage retenu, mesure 0,505 seconde par réponse (§ 6).
- **Où se règle l'ancrage** : clé `anchor_mu` du `chat_config.json` du modèle (0 ou absente : ancien choix), ou option `--anchor-mu` de `python -m chatbot.generate chat` (de 0 à 2). Un modèle exporté avant le 15/11 n'a pas cette clé et garde 16 candidates : il faut alors `--anchor-mu 1 --candidates 32`.

## 6. Chiffres (tous mesurés)
Panel de 40 messages × 3 graines = 120 réponses ; BPB sur les 3 pièces de Molière jamais vues. Les quatre dernières lignes utilisent **les mêmes réglages de décodage**, ceux d'origine (16 candidates, sans ancrage), pour rester comparables.

| Version | BPB Molière jamais vu | Reprise du sujet (hors écho) | Arrêt naturel | Mots par réponse |
|---|---|---|---|---|
| Ancien chat (prompt « Utilisateur / Assistant ») | 1,4746 | 2,0 % | 43,3 % | 47,5 (texte hors sujet) |
| Modèle du concours **non affiné**, nouveau format | 1,4746 | 18,2 % | 94,2 % | 13,6 |
| v1 (historique) : modèle du concours **affiné** | 1,2273 | 29,3 % | 100 % | 11,8 |
| v2 (historique) : gros modèle affiné, ancienne démo du Mac | 1,1044 | 31,3 % | 100 % | 11,7 |
| **Modèle M affiné sans synthèse (A0)** | **1,0560** | **32,3 %** | 100 % | 13,0 |

Repères du v2 (historique) : 3,8709 (initialisation aléatoire) → 1,2155 (après pré-entraînement) → 1,1044 (après affinage théâtre). Repères du modèle M : 3,87695 (initialisation aléatoire) → 1,12985 (après pré-entraînement) → 1,0560 (A0, après affinage théâtre) ; S03, avec synthèse (modèle publié) : 1,05585 en fin d'affinage, 1,05570 à l'évaluation. « Reprise du sujet » = part des réponses qui reprennent un mot plein du message sans le recopier ; « arrêt naturel » = le modèle a fini sa réplique de lui-même.

Les mesures du modèle M sont dans `experiments/chatbot_rtx/20261113_1956_M/bilan.json` et `experiments/chatbot_rtx/20261113_1956_M/evaluations/moliere/metrics.json`.

**Jury à l'aveugle des anciens modèles (historique : il ne porte pas sur le modèle M)** (12 jurés IA indépendants — spectateur, professeur de français, comédien —, 80 messages, réponses anonymisées et mélangées, 240 verdicts) :

| | Rang moyen (1 = meilleur) | Classé 1er | Rapport au message | Langue | Style |
|---|---|---|---|---|---|
| v2 | **1,30** | **70 %** | 3,04 / 5 | 3,70 / 5 | 3,65 / 5 |
| v1 | 1,70 | 30 % | 2,58 | 3,16 | 3,32 |
| Ancien chat | 3,00 | 0 % | 1,07 | 1,40 | 1,04 |

**Jugement à l'aveugle du modèle M** (15/11 ; jeu de messages « neuf » ; 144 réponses par modèle ; 3 juges IA indépendants ; réponses mélangées, sans le nom du modèle ; décodage ancré du § 5 pour les deux modèles). Une réponse est « sans défaut » si la majorité des juges valide les trois critères. Nombre de réponses sur 144 :

| | Français correct | Rapport au message | Style | Sans défaut |
|---|---|---|---|---|
| A0 (sans synthèse) | 119 | 110 | 124 | 82 |
| **S03 (3,02 % de synthèse)** | **124** | **114** | **138** | **100** |

Réponse par réponse, S03 est meilleur 45 fois et moins bon 27 fois (test des signes : p = 0,044). Le gain vient surtout des conversations de plusieurs tours (57 sans défaut sur 72, contre 38). Réserves : peu de messages de test (48 par jeu, joués 3 fois) ; une notation refaite sur les mêmes réponses peut bouger d'une dizaine de réponses sur 144. Le détail et les règles de décision, écrites avant de voir les notes, sont dans la fiche du modèle (§ 6).

**Mesure automatique du modèle publié (S03), refaite le 16/11 avec le décodage retenu** (panel de 120 réponses) : 15,9 mots par réponse en moyenne ; reprise de mots du message 27,27 % ; 0,505 seconde par réponse sur la carte RTX 5090 ; aucune réponse vide ; aucun nom de rôle dans les réponses. BPB de l'évaluation : 1,05570.

**Jugement final du 16/11 : S03 reste le modèle retenu.** Les essais du 16/11 ont cherché un modèle meilleur que S03 avec les 27 624 conversations des campagnes v4 et v5 réunies (§ 7). La règle a été écrite le 15/11 à 16 h, avant la génération. Le témoin est S03. Les candidats sont le même affinage avec 3, 5, 10 et 15 % de synthèse. Deux gardes : BPB sur Molière au plus égal à celui de A0 + 0,010 (soit 1,06602), et par-cœur inférieur à 10 %. Jugement à l'aveugle sur le jeu « neuf2 », jamais utilisé avant : 144 réponses par modèle. Un candidat remplace S03 seulement si son nombre de réponses sans défaut (T) dépasse celui de S03 d'au moins 12, et si F, R et S ne reculent pas de plus de 6.

| Modèle | Part de synthèse | Meilleur pas | BPB Molière | Par-cœur | F | R | S | T (sans défaut, sur 144) | T − T(S03) |
|---|---|---|---|---|---|---|---|---|---|
| A0 (sans synthèse) | 0 % | 1 000 | 1,05602 | −0,65 % | 113 | 94 | 127 | 73 | −23 |
| **S03 (témoin, modèle publié)** | 3,02 % (v4) | 1 000 | 1,05570 | 1,92 % | 124 | 111 | 142 | **96** | n/a |
| v6 à 3 % | 2,99 % | 1 000 | 1,05568 | 0,80 % | 130 | 105 | 139 | 93 | −3 |
| v6 à 5 % | 5,04 % | 1 250 | 1,05568 | 2,69 % | 133 | 109 | 144 | 101 | +5 |
| v6 à 10 % | 10,01 % | 1 250 | 1,05746 | 6,23 % | 132 | 104 | 134 | 88 | −8 |
| v6 à 15 % | 14,98 % | 1 250 | 1,05864 | 9,81 % | 128 | 118 | 141 | 105 | +9 |

F = français correct, R = rapport au message, S = style, T = sans défaut. « Par-cœur » = 1 − (BPB de la synthèse entraînée ÷ moyenne des BPB des 3 fichiers v5 tenus à l'écart). Pour S03, la mesure porte sur le fichier v4 + v5, dont il n'a vu que la partie v4. Les parts des candidats sont les parts attendues ; celle de S03 est mesurée.

- **Verdict : aucun candidat n'atteint +12. S03 reste le modèle retenu.** Les quatre candidats passent les deux gardes ; celui à 15 % de justesse (9,81 %).
- Comparaison appariée contre S03 (même message, même graine ; test des signes bilatéral) sur T. A0 est meilleur 25 fois et moins bon 48 fois (p = 0,0095). Pour les candidats : 3 %, 28 contre 31 (p = 0,79) ; 5 %, 36 contre 31 (p = 0,63) ; 10 %, 25 contre 33 (p = 0,36) ; 15 %, 31 contre 22 (p = 0,27). Aucune de ces différences n'est démontrée.
- T sur 72 : messages isolés A0 39, S03 48, v6 à 3 % 48, à 5 % 50, à 10 % 48, à 15 % 57 ; conversations de plusieurs tours A0 34, S03 48, v6 à 3 % 45, à 5 % 51, à 10 % 40, à 15 % 48.
- Mots par réponse (5 graines) : A0 11,8 ; S03 16,3 ; candidats v6 14,6, 14,0, 14,9 et 15,1.
- Ce que la campagne v5 a apporté : à part égale, le par-cœur a presque disparu (5 % : 2,69 % contre 10,2 % le 15/11 ; 10 % : 6,23 % contre 23,4 % ; 15 % : 9,81 % contre 30,5 %). La qualité jugée n'a pas progressé de façon démontrée. Un peu de synthèse (3 %) aide ; en mettre plus, même sans par-cœur, n'a pas donné mieux.
- S03 contre A0 est confirmé sur un jeu de messages neuf : 96 contre 73 le 16/11, après 100 contre 82 le 15/11 (jeu « neuf », p = 0,044).
- Les juges : 36 agents IA, 3 par lot de 72 réponses, 864 réponses, aucune note manquante, vote à la majorité. La consigne est identique, mot pour mot, à celle des jugements précédents. Les 3 juges sont d'accord pour F dans 829 cas sur 864, pour R dans 793 et pour S dans 839. **Les juges du 16/11 sont des agents Claude Opus 5.5, un autre modèle d'IA que celui des jugements des 14 et 15/11.** Les notes du 16/11 ne se comparent donc pas directement à celles du 15/11. La décision n'en dépend pas : S03 est rejugé dans le même lot, par les mêmes juges, que les candidats.
- Où lire les réponses jugées et les notes : `experiments/chatbot_rtx/jugements/final_neuf2/` (`items.json`, `cle.json`, `notes.json`, `notes_finales.json`, `gardes.json`, `reponses_5_graines_<modèle>.jsonl`). Dans la clé, le témoin S03 s'appelle `S03t` et les candidats `s03`, `s05`, `s10`, `s15`. Les recalculs : `python experiments/chatbot_rtx/jugements/recompter_jugements.py` et `python experiments/chatbot_rtx/jugements/apparie.py experiments/chatbot_rtx/jugements/final_neuf2 S03t`.

Dans tous ces jugements, ce sont des jurés **IA**, pas des humains : à présenter comme tel. La qualité jugée par des personnes n'a jamais été mesurée.

## 7. Conversations de synthèse et données nettoyées
**Pourquoi de la synthèse.** Le théâtre apprend le style. Mais aucun texte réel ne montre « un message d'aujourd'hui → une réponse à la manière de Molière ». On fait donc écrire une petite part de conversations de ce genre par un modèle de langue, et on les mélange à l'affinage.

- **Qui écrit** : `gemma4:31b-it-q4_K_M`, un modèle **à poids ouverts** (licence Apache 2.0), lancé en local avec Ollama. Il écrit des données ; il ne donne aucun poids à notre modèle. Aucune conversation n'a été écrite à la main ni par un modèle fermé.
- **Consignes publiées** : les axes (`chatbot/axes_consignes_v4.json`, `chatbot/axes_consignes_v5.json`), le programme qui les croise (`tools/composer_consignes_v4.py`) et les consignes complètes (`chatbot/prompts_synthetiques_v4.json`, `chatbot/prompts_synthetiques_v5.json`). Le lanceur est `tools/campagne_synthese.py` ; les filtres sont dans `chatbot/synthetic.py`.
- **Part bornée et déclarée** : 15 % au plus des tokens vus à l'affinage. Pour S03 : 9 055 conversations (897 109 tokens) de la campagne v4, qui font **3,02 %** des tokens vus. Le modèle A0 n'en a vu aucune.
- **Pourquoi si peu** : avec ces 9 055 conversations, une part plus forte fait apprendre la synthèse **par cœur**. Écart entre des conversations vues et jamais vues : 4,6 % à 3 % de synthèse ; 10,2 % à 5 % ; 23,4 % à 10 % ; 30,5 % à 15 % (limite fixée d'avance : moins de 10 %). La campagne v5 (26 000 fiches) devait produire plus de conversations pour pouvoir monter la part. Elle a été arrêtée volontairement le 16/11 à 1 h 03 : 29 468 appels faits sur 48 000 prévus (61 %). Réunie à la v4, elle donne 27 624 conversations (2 694 831 tokens). Un contrôle de 120 conversations tirées au hasard, par 3 lecteurs (agents IA), en trouve 109 sans défaut (seuil 96). Avec ces conversations, aucun des quatre essais (3, 5, 10 et 15 %) n'a atteint l'écart exigé sur S03 (§ 6) : le modèle publié reste S03, à 3,02 % de synthèse. Les conversations sont publiées dans la pièce jointe `synthese_moliere.zip` (22 fichiers, 4 442 456 octets, licence CC BY 4.0, à décompresser à la racine d'un clone) de la release `chatbot-moliere-20261116`.
- **Défaut connu** : la synthèse transmet les tics du générateur. S03 commence 28 réponses sur 240 par « Certes » et 24 par « Hélas » (0 et 1 pour A0).
- Une première campagne avec un générateur plus petit (`qwen3:8b`, consignes `chatbot/prompts_synthetiques.json`, 1,16 % de synthèse) n'a pas été retenue : 45 réponses sans défaut sur 72 contre 44, gain non démontré.

**Données nettoyées : résultat négatif (essais du 14/11).** On a aussi essayé de nettoyer davantage les données. Jugement à l'aveugle sur 72 réponses ; la référence en a 44 sans défaut.

| Essai | Résultat |
|---|---|
| Théâtre nettoyé (traductions, livrets chantés et didascalies retirés ; `tools/nettoyer_affinage.py`) | 34 sans défaut ; BPB de validation moins bon (1,0628 contre 1,0560) |
| Pré-entraînement filtré (tri appris de FineWeb, règles par source ; `tools/nettoyer_pretrain.py`, `tools/trier_fineweb.py`) + théâtre nettoyé | 37 sans défaut |
| Pré-entraînement filtré + théâtre d'origine | 38 sans défaut ; BPB de validation presque inchangé (1,0572) |

Aucun essai ne bat la référence. Leçon retenue : à cette échelle, des données « plus propres » n'ont pas donné de meilleures réponses. Ces outils restent dans le dépôt pour pouvoir refaire l'essai ; ni A0 ni S03 ne les utilisent.

Le détail est dans la fiche des données (§ 4 pour la synthèse, § 6 pour les essais).

## 8. Explique cette ligne
1. `return f"{normalize_speaker(speaker)}.\n{clean_text(text)}\n\n"` (`render_turn`, format.py) → **le format unique** : nom, point, réplique, ligne vide. Le même rendu sert à l'entraînement et au chat.
2. `pool = [t for t in pool if lo <= len(WORD.findall(t[0])) <= hi] or pool` (generate.py) → **filtre avec repli** : on ne garde que les réponses de 4 à 30 mots, mais s'il n'en reste aucune on garde tout (jamais de bulle vide). Les autres filtres ont la même forme.
3. `scores = [s * len(r) for s, r in zip(scores, replies_ids)]` → **moyenne × nombre de tokens = somme** : l'information totale tirée de la conversation.
4. `if args.warmup_steps >= max_steps: sys.exit(...)` (finetune.py) → sans pas restants après la montée du taux d'apprentissage, il n'y a pas de descente en cosinus : on refuse plutôt que d'entraîner de travers.
5. `"format_sha256": file_sha256(fmt.__file__)` (clé du cache, finetune.py) → le cache des tokens dépend du **code du format** : on ne peut pas entraîner sur un format et discuter dans un autre.

## 9. Questions probables du jury
1. **Est-ce un « vrai » chatbot ?** Non : c'est un modèle de langue de 54 749 440 paramètres, entraîné de zéro sur 2 000 027 648 tokens. Il produit des répliques **dans le style du théâtre classique**, souvent en rapport avec le message, sans garantie de sens ni de vérité. Les assistants du commerce ont des milliers de fois plus de paramètres et de données.
2. **Avez-vous utilisé un modèle pré-entraîné ?** Non : aucun poids ne vient d'ailleurs. Le v1 part de notre modèle du concours ; le v2 et le modèle M d'une initialisation aléatoire. Une précision pour S03 : 3,02 % des tokens vus à l'affinage sont des conversations écrites par un modèle à poids ouverts, `gemma4:31b-it-q4_K_M` (§ 7). Ce modèle a écrit des données ; il n'a donné aucun poids. A0 n'a vu aucune donnée de synthèse.
3. **Le chatbot change-t-il la soumission ?** Non. La soumission du 11/11 (BPB 1,288292) est intacte, le kit n'est pas modifié (empreintes de référence dans `docs/remise/KIT_SHA256.json` ; la comparaison `diff -r` du dossier `ftb` avec le kit d'origine était vide), et ni le v2 ni le modèle M ne sont soumis.
4. **Pourquoi « MOLIÈRE » comme personnage ?** Il n'a que 99 répliques (L'Impromptu de Versailles). Pendant l'affinage, on ajoute des **copies de scènes de Molière où le personnage principal est renommé MOLIÈRE** : il devient la voix moyenne de ses personnages.
5. **Comment savez-vous que c'est mieux qu'avant ?** Mêmes 40 messages, mêmes graines, mêmes réglages de décodage, une validation jamais vue, et un jugement à l'aveugle. Les mesures du modèle M sont dans `experiments/chatbot_rtx/20261113_1956_M/evaluations/moliere/metrics.json`. Les transcriptions des essais du Mac (v1, v2) ne sont pas publiées. Les réponses jugées et les notes des juges du modèle publié et des candidats du 16/11 sont dans `experiments/chatbot_rtx/jugements/final_neuf2/` (§ 6).
6. **Harpagon et Alceste viennent des pièces de validation : comment les jouez-vous ?** Ni le v2 ni le modèle M ne les ont vus dans Molière ; ils les jouent par généralisation. Une variante du v2 finie sur tout Molière a existé (non publiée), mais un second jury à l'aveugle ne l'a pas préférée (rang moyen 1,95 contre 1,90 : mieux en style, moins en rapport au message) et son BPB (1,0462) n'est plus tenu à l'écart : on ne le cite pas.
7. **Qu'est-ce qui n'a pas marché ?** (a) Le premier reclassement choisissait les réponses les plus courtes (3,7 mots) ; (b) deux entraînements en même temps sur le GPU du Mac : le petit a été affamé (600 pas en 5 h) ; (c) l'extracteur coupait les répliques à chaque paragraphe (27 % de scènes de 3 répliques ou moins → 12 %) ; (d) en conversation longue, le bot recopiait ses propres réponses (42 % → 0 % après correction) ; (e) finir sur Molière seul améliore le BPB (1,1044 → 1,0974) mais pas l'avis du jury. Trois revues adverses ont confirmé 70 défauts ; la plupart sont corrigés, les autres sont écrits (voir § 10). Les points (a) à (e) datent des modèles v1 et v2. Avec le modèle M : (f) nettoyer davantage les données n'a pas donné de meilleures réponses (§ 7) ; (g) la première campagne de synthèse, avec `qwen3:8b`, n'a pas montré de gain (§ 7) ; (h) 5 % de synthèse ou plus avec les mêmes 9 055 conversations : le modèle les apprend par cœur (§ 7) ; (i) `torch.compile` a échoué sur le PC (Triton absent) : le lanceur a continué sans compilation, même calcul ; (j) avec les 27 624 conversations des campagnes v4 et v5, aucun des essais à 3, 5, 10 et 15 % de synthèse n'a atteint l'écart exigé sur S03 (§ 6).

## 10. Limites connues
- Réponses parfois **à côté** ou absurdes ; orthographe ancienne possible (« connoître », « très-bien »). Pour le modèle M, au jugement à l'aveugle du 15/11 (§ 6) : 44 réponses sur 144 ont un défaut pour S03, le modèle publié, soit environ trois sur dix ; 62 sur 144 pour A0 (phrase bancale, mot inventé, réponse à côté du message) ; il répond en français classique, avec le vouvoiement, même à un message familier ; au-delà de 512 tokens, il oublie les échanges les plus anciens. La liste complète est dans la fiche du modèle (§ 2).
- 0,06 % des répliques d'entraînement gardent une ligne « NOM + action » attribuée au mauvais personnage ; 65 caractères invisibles restent dans deux pièces de `moliere_train` (retirés au rendu).
- La « reprise du sujet » compte des mots repris : elle est optimisée par le choix de la réponse. Le jury à l'aveugle est la mesure indépendante, mais ses jurés sont des IA.
- Résidus détaillés des modèles v1 et v2 : fichiers de travail du Mac, non publiés.
- Tous les jugements du modèle M sont faits par des agents IA ; aucune personne n'a noté les réponses.
