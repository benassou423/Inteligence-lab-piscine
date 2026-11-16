# Fiche du modèle — chatbot « Molière »

Un petit modèle de langue français, entraîné **de zéro** (aucun poids venu d'ailleurs), puis affiné sur du
théâtre pour répondre à des messages courts **à la manière de Molière**. C'est une démonstration
pédagogique, hors concours : tout est fait pour qu'on puisse l'utiliser, le refaire et l'inspecter.

Les données sont décrites dans [`FICHE_DONNEES.md`](FICHE_DONNEES.md).

## 1. En bref

| | |
| --- | --- |
| Type | transformer décodeur de type GPT-2 (classe `GPT` de `ftb/model.py`) |
| Taille | 54 749 440 paramètres : 640 dimensions, 10 couches, 10 têtes |
| Fenêtre | 512 tokens (environ 11 échanges courts) |
| Vocabulaire | 8 192 tokens (BPE au niveau des octets) |
| Langue | français seulement |
| Pré-entraînement | 2 000 027 648 tokens de français écrit, 6 sources |
| Affinage | 90 013 696 tokens : théâtre français, Molière, et 3,02 % de conversations de synthèse (2 720 768 tokens) |
| Matériel | une carte NVIDIA GeForce RTX 5090 |
| Durée | pré-entraînement 1 h 40 ; affinage 5 minutes |
| Version publiée | S03, release `chatbot-moliere-20261116` du 16/11/2026 ; `ckpt.pt` : 239 991 123 octets, sha256 `01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577` |

## 2. À quoi il sert, et ses limites

**Fait pour** : une conversation courte et légère avec un personnage de Molière ; montrer ce qu'un petit
modèle entraîné de zéro sait faire ; servir d'exemple complet et vérifiable (données, consignes, code, mesures).

**Pas fait pour** : donner une information, un conseil, une traduction, un résumé. Le modèle ne sait rien de
fiable. Il imite un style.

Limites mesurées ou observées :

- **Environ une réponse sur trois a un défaut.** Jugé à l'aveugle sur 144 réponses, le modèle publié en a
  100 sans défaut au jeu « neuf » (15/11/2026) et 96 au jeu « neuf2 » (16/11/2026), soit 44 et 48 réponses
  avec un défaut (§ 6). Les défauts : phrase bancale, mot inventé (« Je m'embarte avec
  plaisir »), réponse à côté du message.
- Il répond en français classique, avec le vouvoiement, même à un message familier.
- Il perd le fil d'une longue conversation : au-delà de 512 tokens, les échanges les plus anciens sont oubliés.
- Il ne comprend que le français.
- Les textes d'entraînement vont surtout du XVIIe au XIXe siècle : ils portent les idées et les préjugés de
  leur époque. Des filtres retirent le vocabulaire sexuel cru et les œuvres obscènes, rien de plus.
- HARPAGON et ALCESTE sont proposés au chat, mais le modèle n'a jamais lu *L'Avare* ni *Le Misanthrope*
  (pièces tenues à l'écart pour la mesure) : il improvise ces rôles.

## 3. Architecture

Le modèle est la classe `GPT` du kit du concours « Feed the Beast » (`ftb/model.py`, non modifié), en plus
grand que le modèle officiel du concours (10 166 400 paramètres).

- plongement de tokens + plongement de positions **apprises** (positions absolues 0 à 511) ;
- 10 blocs à pré-normalisation : `x = x + attention(LayerNorm(x))`, puis `x = x + MLP(LayerNorm(x))` ;
- attention causale à 10 têtes de 64 dimensions, couches linéaires sans biais ;
- MLP de largeur 4 × 640 avec activation GELU ; pas de dropout ;
- LayerNorm finale, puis tête linéaire **liée** au plongement de tokens.

Compte des paramètres : plongement de tokens 8 192 × 640 = 5 242 880 ; positions 512 × 640 = 327 680 ;
10 blocs de 4 917 760 = 49 177 600 ; LayerNorm finale 1 280 ; total 54 749 440.

## 4. Entraînement

Tout est décrit, valeur par valeur et avec sa raison, dans les fichiers de recette
(`chatbot/recettes_rtx*.json`). Le lanceur est `chatbot/pipeline_rtx.py`.

### 4.1 Pré-entraînement (13/11/2026)

- départ : poids tirés au hasard (initialisation du kit), graine 1337 ;
- 30 518 pas de 128 fenêtres de 512 tokens, soit 2 000 027 648 tokens ;
- optimiseur AdamW (bêtas 0,9 et 0,95, décroissance des poids 0,1), norme du gradient coupée à 1,0 ;
- taux d'apprentissage : montée linéaire sur 1 000 pas jusqu'à 0,0008, puis descente en cosinus jusqu'à 0,00008 ;
- calcul en bfloat16, poids en float32 ; sans compilation (`torch.compile` a échoué, repli automatique) ;
- 5 988,8 secondes, 23,67 Go de mémoire de la carte, 333 977 tokens par seconde à la dernière mesure du journal (moyenne : 2 000 027 648 / 5 988,8 = 333 961).

Mesure sur la validation Molière (3 pièces jamais vues) : le BPB (bits par octet, plus bas = mieux) passe de
3,87695 au départ à **1,12985** à la fin.

### 4.2 Affinage

- départ : le modèle pré-entraîné ;
- 2 747 pas de 64 fenêtres de 512 tokens, soit 90 013 696 tokens ;
- données : théâtre français (poids 1), Molière (poids 3, avec les copies « MOLIÈRE »), conversations de
  synthèse de la campagne v4 (poids 1,09 ; 9 055 conversations, fichier
  `data/stages/chatbot_20261114/synthetique_v4/entrainement/moliere.jsonl.zst`) ;
- taux d'apprentissage 0,0002 descendant à 0,00002, 100 pas de montée ;
- la validation est mesurée tous les 250 pas et **on garde le meilleur point**, pas le dernier : le BPB baisse
  jusqu'au pas 1 000 puis remonte (le modèle commence à apprendre le théâtre par cœur).

| Modèle | Synthèse vue | Meilleur pas | BPB sur la validation Molière |
| --- | ---: | ---: | ---: |
| Pré-entraîné seul | — | — | 1,12985 |
| A0 : affinage sans synthèse (13/11/2026) | 0 % | 1 000 | 1,0560 |
| S03 : affinage avec la synthèse v4 (15/11/2026) | 3,02 % | 1 000 | 1,0557 |
| **S03, modèle publié** (release du 16/11/2026 ; même modèle que la ligne précédente, évalué à nouveau le 16/11) | 3,02 % | 1 000 | 1,05570 |

Le meilleur point de l'affinage du modèle publié est le pas 1 000 sur 2 747. Son rapport
d'entraînement (`training_report.json`) donne un BPB de 1,05585 sur les 3 pièces de validation ; l'évaluation
refaite le 16/11/2026 donne 1,05570 (c'est le chiffre du tableau). Le fichier de synthèse fait 897 109 tokens.
Ces 2 720 768 tokens de synthèse sont comptés sur les 2 747 pas de l'affinage ; les poids gardés sont ceux du
pas 1 000. Au pas 1 000, chaque conversation de synthèse a été vue environ 1,1 fois (voir
[`HISTORIQUE.md`](HISTORIQUE.md)).

Versions des logiciels : Python 3.12.11, PyTorch 2.15.0 (CUDA 13.2), numpy 2.5.4, tokenizers 0.23.3.

## 5. Comment une réponse est choisie

Le modèle ne produit pas une réponse : il en tire plusieurs et on en choisit une
([`chatbot/generate.py`](../../chatbot/generate.py)).

1. La conversation est mise en scène comme du théâtre : l'utilisateur dit la réplique de l'interlocuteur, le
   modèle écrit celle du personnage.
2. **32 réponses candidates** sont tirées en parallèle (température 0,8, top-p 0,9, sans répéter une suite de
   4 tokens), chacune arrêtée dès qu'un autre personnage prendrait la parole.
3. On écarte les candidates vides, celles qui écrivent un nom de rôle, celles qui recopient un message de la
   conversation, celles qui commencent par une minuscule, et on ne compare que des réponses de 4 à 30 mots.
4. On garde celle qui **dépend le plus du dernier message** : on mesure combien la réponse est plus probable
   après la conversation qu'après la même conversation où le dernier message est remplacé par « … »
   ([`chatbot/ancrage.py`](../../chatbot/ancrage.py), réglage `anchor_mu` = 1).

Le point 4 a été adopté le 14/11/2026 après un jugement à l'aveugle sur 144 réponses : le nombre de réponses
en rapport avec le message passe de 96 à 114, pour un léger recul du français correct (125 à 120) et du style
(137 à 133). Il coûte du temps : mesuré le 14/11/2026 sur A0, 0,81 seconde par réponse contre 0,47 sur la RTX 5090. Le 16/11, le modèle publié répond en 0,505 seconde en moyenne (§ 7).

Ces réglages sont dans le fichier `chat_config.json` du modèle. Celui du modèle publié porte
`"candidates": 32` et `"anchor_mu": 1.0` : la commande de chat n'a pas besoin des options
`--anchor-mu 1 --candidates 32` (elles restent permises et ne changent rien). Un modèle exporté avec la recette
par défaut (`chatbot/recettes_rtx.json`) garde 16 candidates et `"anchor_mu": 0.0`. Les modèles exportés avant
le 15/11/2026 n'ont pas la clé `anchor_mu` : ils gardent l'ancien choix, avec 16 candidates.

## 6. Évaluation

### 6.1 Deux mesures

- **Le BPB** sur la validation Molière : il dit si le modèle prédit bien du vrai Molière. Il ne dit pas si le
  modèle répond bien à un message d'aujourd'hui.
- **Un jugement à l'aveugle des réponses.** Le modèle répond à des messages de test avec 3 graines
  différentes. Les réponses de plusieurs modèles sont mélangées, sans leur nom, et notées par 3 juges
  indépendants sur trois critères : français correct (F), rapport avec le message (R), style (S). Une réponse
  est « sans défaut » (T) si la majorité des juges valide les trois critères.

Les règles de décision sont **écrites avant** de voir les notes : un candidat ne remplace le modèle en place
que s'il gagne au moins 12 réponses sans défaut sur 144, sans reculer de plus de 6 sur un critère, et s'il
respecte deux gardes : le BPB de validation ne doit pas monter de plus de 0,010, et le modèle ne doit pas
apprendre la synthèse par cœur (écart de moins de 10 % entre des conversations de synthèse vues et jamais vues).

La règle du 16/11/2026 a été écrite le 15/11 à 16 h, avant la génération des conversations de synthèse : le
témoin est S03 ; les candidats sont le même affinage avec la synthèse v4 + v5 à 3, 5, 10 et 15 % ; gardes :
BPB sur Molière au plus égal à celui de A0 + 0,010 (soit 1,06602) et par-cœur sous 10 % ; jugement à l'aveugle
sur le jeu « neuf2 » (jamais utilisé avant), 144 réponses par modèle ; un candidat remplace S03 seulement s'il
gagne au moins 12 réponses sans défaut sur 144 et si F, R et S ne reculent pas de plus de 6.

Les messages de test sont dans [`tools/nuit_moliere.py`](../../tools/nuit_moliere.py) (jeux « reglage »,
« test », « neuf », « neuf2 » : messages isolés et conversations de 4 tours). Aucun n'est dans les données
d'entraînement.

### 6.2 Résultats

Jeu « neuf », 144 réponses par modèle, décodage du § 5 pour tous, 15/11/2026 :

| Modèle | Français correct | Rapport au message | Style | Sans défaut |
| --- | ---: | ---: | ---: | ---: |
| A0 (sans synthèse) | 119 | 110 | 124 | 82 |
| **S03 (3,02 % de synthèse)** | **124** | **114** | **138** | **100** |

Réponse par réponse, S03 est meilleur 45 fois et moins bon 27 fois (test des signes : p = 0,044). Les trois
juges vont dans le même sens. Le gain vient surtout des conversations de plusieurs tours (57 sans défaut sur
72 contre 38).

Des parts de synthèse plus fortes avec les mêmes 9 055 conversations (5, 10 et 15 %) ont été écartées : le
modèle les apprenait par cœur (écarts de 10,2 %, 23,4 % et 30,5 %).

Les réponses jugées, la clé, les notes des trois juges et les lignes pour recompter les jugements des 14 et
15/11 sont dans [`experiments/chatbot_rtx/jugements/`](../../experiments/chatbot_rtx/jugements/README.md).

**Essais du 16/11/2026 (jeu « neuf2 »).** La campagne v5 a donné plus de conversations de synthèse (voir
[`FICHE_DONNEES.md`](FICHE_DONNEES.md), § 4). Quatre candidats ont été entraînés avec la synthèse v4 + v5
(27 624 conversations) à 3, 5, 10 et 15 %, puis jugés à l'aveugle avec A0 et S03 sur le jeu « neuf2 », 144
réponses par modèle :

| Modèle | Part de synthèse | Meilleur pas | BPB Molière | Par-cœur | F | R | S | Sans défaut (T) | T − T(S03) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 (sans synthèse) | 0 % | 1 000 | 1,05602 | −0,65 % | 113 | 94 | 127 | 73 | −23 |
| **S03 (témoin, modèle publié)** | 3,02 % (v4) | 1 000 | 1,05570 | 1,92 % | 124 | 111 | 142 | **96** | — |
| v6 à 3 % | 2,99 % | 1 000 | 1,05568 | 0,80 % | 130 | 105 | 139 | 93 | −3 |
| v6 à 5 % | 5,04 % | 1 250 | 1,05568 | 2,69 % | 133 | 109 | 144 | 101 | +5 |
| v6 à 10 % | 10,01 % | 1 250 | 1,05746 | 6,23 % | 132 | 104 | 134 | 88 | −8 |
| v6 à 15 % | 14,98 % | 1 250 | 1,05864 | 9,81 % | 128 | 118 | 141 | 105 | +9 |

F = français correct, R = rapport au message, S = style. « Par-cœur » = 1 moins le BPB de la synthèse
entraînée divisé par la moyenne des BPB des 3 fichiers v5 tenus à l'écart. Pour S03, la mesure est faite sur le
fichier v4 + v5, dont il n'a vu que la partie v4. Parts « attendues » pour les candidats v6, part mesurée pour
S03.

- **Verdict : aucun candidat n'atteint +12. S03 reste le modèle retenu.** Les quatre candidats passent les
  deux gardes (celui à 15 % de justesse : 9,81 % de par-cœur). Les quatre candidats ne sont pas publiés : seules
  leurs recettes et leurs réponses jugées le sont.
- Comparaison appariée contre S03 (même message, même graine ; test des signes bilatéral) sur « sans défaut » :
  A0 meilleur 25 fois et moins bon 48 fois (p = 0,0095) ; v6 à 3 % : 28 et 31 (p = 0,79) ; à 5 % : 36 et 31
  (p = 0,63) ; à 10 % : 25 et 33 (p = 0,36) ; à 15 % : 31 et 22 (p = 0,27).
- « Sans défaut » sur 72 réponses, messages isolés : A0 39, S03 48, v6 à 3 % 48, à 5 % 50, à 10 % 48, à 15 % 57.
  Conversations de plusieurs tours : A0 34, S03 48, v6 à 3 % 45, à 5 % 51, à 10 % 40, à 15 % 48.
- Mots par réponse (5 graines) : A0 11,8 ; S03 16,3 ; candidats v6 : 14,6, 14,0, 14,9 et 15,1.
- Ce que la campagne v5 a apporté : à part égale, le par-cœur a presque disparu (5 % : 2,69 % contre 10,2 % le
  15/11 ; 10 % : 6,23 % contre 23,4 % ; 15 % : 9,81 % contre 30,5 %). La qualité jugée n'a pas progressé de
  façon démontrée. Un peu de synthèse (3 %) aide ; en mettre plus, même sans par-cœur, n'a pas donné mieux.
- S03 contre A0 est confirmé sur un jeu de messages neuf : 96 contre 73 le 16/11 (jeu « neuf2 »), après 100
  contre 82 le 15/11 (jeu « neuf »).

**Deux jeux, deux séries de juges.** Le tableau « A0 contre S03 » du 15/11 (jeu « neuf », 82 et 100) et celui
du 16/11 (jeu « neuf2 », 73 et 96) viennent de deux jeux de messages différents et de deux séries de juges
différentes. Les juges du 16/11 sont 36 agents Claude Opus 5.5, un autre modèle d'IA que celui des jugements
des 14 et 15/11. La consigne est identique mot pour mot. Les notes du 16/11 ne se comparent donc pas
directement à celles du 15/11. La décision n'en dépend pas : S03 est rejugé dans le même lot, par les mêmes
juges, que les candidats. Il y avait 3 juges par lot de 72 réponses, 864 réponses au total, aucune note
manquante, vote à la majorité. Les 3 juges sont d'accord pour F sur 829 réponses sur 864, pour R sur 793 et
pour S sur 839.

Les pièces du jugement du 16/11 sont dans
[`experiments/chatbot_rtx/jugements/final_neuf2/`](../../experiments/chatbot_rtx/jugements/final_neuf2/) :
`items.json` (ce que les juges ont vu), `cle.json` (identifiant vers modèle ; le témoin S03 s'y appelle `S03t`
et les candidats `s03`, `s05`, `s10`, `s15`), `notes.json` (les 3 juges), `notes_finales.json` (majorité),
`gardes.json` (BPB et par-cœur des six modèles) et `reponses_5_graines_<modèle>.jsonl` (240 réponses par
modèle ; les graines 11, 22 et 33 sont jugées). Pour recompter :

```bash
python experiments/chatbot_rtx/jugements/recompter_jugements.py
python experiments/chatbot_rtx/jugements/apparie.py experiments/chatbot_rtx/jugements/final_neuf2 S03t
```

### 6.3 Réserves

- Les juges sont des **agents IA** (Claude, modèle fermé d'Anthropic), pas des personnes. Ils notent ; ils
  n'ont écrit aucune donnée d'entraînement. Ceux du 16/11/2026 sont un autre modèle que ceux des 14 et 15/11 :
  les notes de deux jours différents ne se comparent pas directement (§ 6.2).
- Peu de messages de test : 48 messages par jeu, joués 3 fois. Une notation refaite sur les mêmes réponses
  peut bouger d'une dizaine de réponses sur 144.
- Les thèmes des messages de test (saluer, se présenter, la fatigue, le temps qu'il fait) recoupent ceux des
  conversations de synthèse. Les messages identiques ont été retirés des données, pas les thèmes : c'est le
  but de la synthèse, mais cela favorise la mesure.
- La synthèse transmet les tics du générateur. S03 commence 28 réponses sur 240 par « Certes » et 24 par
  « Hélas » (0 et 1 pour A0).

## 7. Utilisation

**Télécharger.** Le modèle publié est la pièce jointe `chatbot_moliere_s03.zip` (240 087 338 octets, sha256
`6d62aa50f4917fbb23d24e24a8d64807bde55f273c9cb4203c6970c32c886417`) de la release `chatbot-moliere-20261116` du
dépôt `benassou423/Inteligence-lab-piscine` :
<https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116>. Elle contient le
dossier `chatbot_moliere/`. Le second chatbot (« français courant ») n'a pas changé depuis le 13/11 et n'est pas
dans ce paquet.

Un modèle est un dossier de quatre fichiers : `ckpt.pt` (poids), `tokenizer.json`, `chat_config.json`
(réglages du § 5), `training_report.json` (ce qui l'a entraîné, part de synthèse comprise).

Après décompression du zip à la racine d'un clone du dépôt, dans l'environnement Python du projet (Python
3.12, PyTorch, tokenizers) :

```bash
python -m chatbot.generate chat --model chatbot_moliere "Bonjour, comment allez-vous ?"
```

Sans message, le chat est interactif. Options utiles : `--persona SCAPIN` (11 personnages, MOLIÈRE par
défaut), `--seed 7` (réponse rejouable), `--device cuda`, `--anchor-mu 0` (ancien choix de réponse).

En Python :

```python
from chatbot.generate import ChatBot
bot = ChatBot.load("chatbot_moliere")
r = bot.reply([{"role": "user", "content": "Quel temps fait-il ?"}], seed=7)
print(r["reply"])
```

Sur la carte RTX 5090, avec le décodage retenu, une réponse prend 0,505 s en moyenne (évaluation du
16/11/2026, panel de 120 réponses : 15,9 mots par réponse, 27,27 % de mots du message repris, aucune réponse
vide, aucun nom de rôle). Sur un Mac à puce Apple, processeur seul (4 fils), la mesure du 16/11/2026 donne 1,7 à 2,3 s par réponse sur 3 messages, et 1,8 à 5,1 s dans l'application macOS sur 4 essais. C'est une mesure sur une seule machine.

## 8. Refaire le modèle

```bash
python -m chatbot.pipeline_rtx telecharger     # données : release chatbot-donnees-20261113
python -m chatbot.pipeline_rtx verifier        # empreintes, versions, carte, disque
python -m chatbot.pipeline_rtx --recettes RECETTE.json lancer --taille M
python -m chatbot.pipeline_rtx resultats       # archive des modèles et des rapports
```

Le guide pas à pas est [`PC_RTX.md`](../../PC_RTX.md). Le transport du code, des données et des résultats
entre le Mac et le PC, suivi le 13/11/2026, est décrit dans
[`PASSAGE_PC_RTX5090_CHATBOT.md`](PASSAGE_PC_RTX5090_CHATBOT.md) (pièce historique). Le lanceur reprend là où
il s'est arrêté, n'écrase jamais une étape finie, et enregistre pour chaque étape la commande exacte, la durée
et les mesures.
Les rapports du modèle publié sont dans `experiments/chatbot_rtx/20261116_modele_publie/` et ceux du
lancement du 13/11/2026 dans `experiments/chatbot_rtx/20261113_1956_M/`.

**Recette exacte du modèle publié** : [`chatbot/recettes_rtx_publie.json`](../../chatbot/recettes_rtx_publie.json)
(étiquette « recettes-rtx-publie-s03 »). C'est
[`chatbot/recettes_rtx_v5_s03.json`](../../chatbot/recettes_rtx_v5_s03.json) (la recette de l'essai du 15/11,
gardée telle quelle) plus deux déclarations qui ne changent pas l'entraînement : le fichier de synthèse
(`etape1.fichiers_synthetiques`) et le décodage écrit à l'export (`--anchor-mu 1 --candidates 32`). Les recettes
des candidats du 16/11 sont `chatbot/recettes_rtx_v6_s03.json`, `recettes_rtx_v6_s05.json`,
`recettes_rtx_v6_s10.json` et `recettes_rtx_v6_s15.json` (poids de tirage de la synthèse 0,36, 0,62, 1,3 et
2,06).

**Les conversations de synthèse.** La pièce jointe `synthese_moliere.zip` de la release
`chatbot-moliere-20261116` (4 442 456 octets, sha256
`e0b144a75acafb5187160d37bcdf1099d60218846cc4b081565819e891c4d3bf`, 22 fichiers, licence CC BY 4.0) est à
décompresser à la RACINE d'un clone : les fichiers se rangent aux chemins que lisent les recettes. Elle contient
le fichier du modèle publié (v4, 9 055 conversations), le fichier v4 + v5 des essais du 16/11 (27 624
conversations), les 6 fichiers tenus à l'écart, les traces des campagnes et un `LISEZMOI_SYNTHESE.txt`. Avec ce
zip et les données de la release `chatbot-donnees-20261113` (`python -m chatbot.pipeline_rtx telecharger`), la
recette du modèle publié se relance depuis un clone :

```bash
python -m chatbot.pipeline_rtx --recettes chatbot/recettes_rtx_publie.json lancer --taille M
```

Le 16/11/2026, l'entraînement de S03 était déjà fait : le lanceur n'a repris que l'export et l'évaluation
(`... lancer --taille M --seulement moliere`, 2 min 27 s). Rien n'a été réentraîné. **Un réentraînement ne
redonne pas forcément les mêmes octets : cela n'a pas été essayé.**

**Rapports et notes des juges.** Le paquet du modèle contient son `training_report.json` (part de synthèse
comprise). Une autre pièce jointe de la release `chatbot-moliere-20261116` est l'archive des résultats du
lanceur (les deux chatbots, leurs rapports et leurs évaluations ; `RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`,
74 fichiers, 445 528 215 octets, sha256 `9028641e6d5c0aece190dd7a7930545e8693a790927e04b69e3de28a32b1ea5a`,
avec son manifeste `RESULTATS_CHATBOT_RTX_20261113_1956_M.manifeste.json`). Le nom porte la date de création
du lancement ; le contenu est l'état du 16/11 (chatbot Molière = S03). Elle se lit avec
`tools/import_chatbot_results.py --depuis-release benassou423/Inteligence-lab-piscine chatbot-moliere-20261116`.
Les notes des juges et les réponses jugées sont dans
[`experiments/chatbot_rtx/jugements/`](../../experiments/chatbot_rtx/jugements/README.md) (§ 6.2).

**Traçabilité du code.** Les rapports d'entraînement consignent l'empreinte sha256 des fichiers de code. Le
rapport de la base (`experiments/chatbot_rtx/20261113_1956_M/`) consigne celle de `chatbot/pretrain.py` (inchangé
depuis) et de `chatbot/format.py` dans sa version du commit `4def4b5`. Depuis, seul un commentaire de
`chatbot/format.py` a été corrigé le 16/11 (il affirmait à tort que le modèle de démonstration était ré-entraîné
sur les pièces de validation) : aucune ligne de code n'a changé. Pour retrouver l'empreinte consignée :
`git show 4def4b5:chatbot/format.py | sha256sum`. L'évaluation du modèle publié a été refaite le 16/11 : son
`metrics.json` consigne les empreintes des fichiers `evaluate.py`, `generate.py`, `format.py` et `finetune.py`
tels qu'ils sont dans le dépôt.

## 9. Licences et crédits

Ce qui s'applique à quoi est écrit dans [`LICENCES.md`](../../LICENCES.md). En bref :

- **Poids du modèle** : CC BY-SA 4.0. Le partage à l'identique est choisi parce qu'une partie des textes
  d'entraînement (Wikipédia, Wikisource) est sous cette licence.
- **Code du dépôt** (`chatbot/`, `tools/`, etc.) : Apache 2.0.
- **Conversations de synthèse et consignes** : CC BY 4.0.
- **Code du modèle** (`ftb/`) : kit du concours « Feed the Beast », Intelligence Lab, ECE Paris, 2026. Il
  n'est pas couvert par les licences ci-dessus : aucune licence n'accompagne ce code.
- **Données** : Wikipédia et Wikisource (CC BY-SA 4.0), Project Gutenberg, FineWeb-2 et Common Crawl
  (ODC-By 1.0), Wikinews (CC BY 2.5), Assemblée nationale (Licence Ouverte 2.0). Détail et réserves dans
  [`FICHE_DONNEES.md`](FICHE_DONNEES.md), § 7.
- **Générateur de la synthèse** : `gemma4:31b-it-q4_K_M`, licence Apache 2.0.

## 10. Ce qu'on ne sait pas

- la qualité jugée par des personnes : jamais mesurée (tous les juges et lecteurs sont des agents IA) ;
- la génération de synthèse v5 a été arrêtée à 61 % (29 468 appels sur 48 000 prévus) : le reste, environ
  18 500 appels, n'a pas été produit ;
- aucun réentraînement de contrôle de S03 n'a été fait : la reproductibilité à l'octet n'a pas été essayée ;
- le poste d'entraînement a redémarré brutalement plusieurs fois les 14 et 15/11 ; la cause n'est pas connue ;
- le comportement sur des messages longs, agressifs ou hors des thèmes testés ;
- le temps de réponse sur un ordinateur sans carte graphique, sauf la mesure du 16/11/2026 sur un seul Mac à puce Apple (§ 7) ;
- la consommation électrique de l'entraînement (non mesurée).
