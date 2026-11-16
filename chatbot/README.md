# Le dossier `chatbot/`

Ce dossier contient le code du chatbot « à la manière de Molière » (hors concours), ses recettes d'entraînement
et les fichiers de consignes de ses essais. Le modèle est un GPT de 54 749 440 paramètres, entraîné de zéro.

- Le modèle : [`docs/chatbot/FICHE_MODELE.md`](../docs/chatbot/FICHE_MODELE.md). Les données : [`docs/chatbot/FICHE_DONNEES.md`](../docs/chatbot/FICHE_DONNEES.md).
- Le récit daté des essais : [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md).
- Le guide d'entraînement sur carte NVIDIA : [`PC_RTX.md`](../PC_RTX.md). Les outils : [`tools/README.md`](../tools/README.md).

## 1. Les modules

| Module | Ce qu'il fait |
| --- | --- |
| `__init__.py` | Marque le dossier comme paquet Python. |
| `accel.py` | Tout ce qui dépend du matériel : choix de la carte ou du processeur, précision du calcul, compilation, soucis propres à Windows. |
| `ancrage.py` | Choix de la réponse « ancré sur le dernier message » : une option de décodage, sans réentraînement, adoptée le 14/11/2026. |
| `dialogues.py` | Extrait de vrais dialogues en français moderne des romans du corpus (format « tirets »), pour le second chatbot. |
| `evaluate.py` | Banc d'évaluation : BPB sur les pièces jamais vues, et un panel fixe de 40 messages avec des mesures automatiques. |
| `finetune.py` | Affinage du modèle sur des scènes de théâtre mises en répliques. |
| `format.py` | Les deux formats de conversation, « repliques » et « tirets » : les mêmes pour l'affinage et pour le chat. |
| `generate.py` | Génération des réponses : commandes `export` (fabrique le dossier d'un modèle) et `chat`. |
| `pipeline_rtx.py` | Lanceur unique des entraînements : `telecharger`, `verifier`, `autotest`, `mesurer`, `lancer`, `etat`, `resultats`, `publier`. |
| `pretrain.py` | Pré-entraînement à partir de zéro, avec la classe `GPT` du kit (non modifié). |
| `synthetic.py` | Générateur et filtres des conversations de synthèse, écrites par un modèle à poids ouverts servi par Ollama. |
| `theatre.py` | Fabrique le jeu « théâtre » : des scènes découpées en répliques (`moliere_train`, `moliere_val`, `theatre_train`). |

Les tests sont dans `tests/test_chatbot_*.py`.

## 2. La table des essais

Une recette (`recettes_rtx*.json`) donne toutes les valeurs d'un entraînement, chacune avec sa raison. On la choisit ainsi,
l'option `--recettes` avant la sous-commande :

```bash
python -m chatbot.pipeline_rtx --recettes RECETTE.json lancer --taille M
```

Mots du tableau : **BPB** = bits par octet sur 3 pièces de Molière jamais entraînées (plus bas = mieux) ;
**sans défaut** = réponses validées sur les trois critères par la majorité de 3 juges IA, à l'aveugle ;
**par-cœur** = écart de BPB entre des conversations de synthèse vues à l'entraînement et des conversations jamais vues. Pour B et C, il est donné en BPB (limite 0,10). Pour A0, S03 et les suivants, il est donné en part du BPB des conversations jamais vues (limite 10 %). Les deux limites ont été fixées d'avance.
« La release » = `chatbot-donnees-20261113` (15 fichiers, liste dans `donnees_rtx.json`). « La release du modèle » = `chatbot-moliere-20261116`, qui porte le modèle publié et `synthese_moliere.zip` (voir § 2 bis). Les chemins de données cités
sont sous `data/stages/`. Les chiffres viennent de [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md), des recettes et des fichiers eux-mêmes.

| Fichier | Essai et date | Ce qui change | Résultat | Retenu ? | Relançable depuis un clone ? |
| --- | --- | --- | --- | --- | --- |
| `recettes_rtx.json` | A0, 13/11/2026 | Recette de départ : pré-entraînement de zéro (2 000 027 648 tokens), puis affinage sur le théâtre (90 013 696 tokens), sans synthèse. | BPB 1,12985 après le pré-entraînement, 1,05608 après l'affinage. 82 sans défaut sur 144 (jeu « neuf », 15/11). | Témoin du 13 au 15/11. Rejugé le 16/11 sur le jeu « neuf2 » : 73 sans défaut sur 144, contre 96 pour S03. | Oui : ses données sont dans la release. L'étape de synthèse séparée est sautée quand son fichier est absent. |
| `recettes_rtx_v2.json` | B, 14/11/2026 | A0 + première synthèse (`qwen3:8b`, 1 305 conversations) mêlée à l'affinage, poids 5 : 2,87 % des tokens vus. | BPB 1,05712. Par-cœur : écart 0,1834 (limite 0,10). Style 61 contre 69 sur 72. | Non. | Non : `chatbot_20261113/synthetique_fusion/moliere.jsonl.zst` n'est pas dans la release. |
| `recettes_rtx_v3.json` | C, 14/11/2026 | Comme B, poids 2 : 1,16 % des tokens vus (1,17 % attendu). | BPB 1,0558, écart 0,0659. Rapport au message 94 → 102 sur 144 (seuil : +14). Rejoué avec le nouveau choix de réponse : 45 sans défaut contre 44 sur 72, français −7. | Non. | Non : même fichier absent. |
| `recettes_rtx_v4a.json` | D, 14/11/2026 | A0 avec le théâtre d'affinage nettoyé (363 œuvres écartées, indications de scène retirées). | BPB 1,0628. 34 sans défaut sur 72, contre 44 pour A0. | Non : essai négatif. | Non tel quel : `chatbot_20261114/theatre_propre/` n'est pas dans la release. À refaire d'abord avec `tools/nettoyer_affinage.py`, qui lit le théâtre de la release. |
| `recettes_rtx_v4.json` | F, 14/11/2026 | Pré-entraînement filtré + théâtre nettoyé. | BPB 1,0628. 37 sans défaut sur 72, contre 44. | Non : essai négatif. | Non : `chatbot_20261114/pretrain_propre/` n'est pas dans la release. Le refaire demande le modèle de base du 13/11, dont la sonde vérifie l'empreinte et qui n'est pas publié. |
| `recettes_rtx_v4e.json` | E, 14/11/2026 | Pré-entraînement filtré (9,31 % des tokens écartés sur les 6 sources), théâtre d'origine. | BPB 1,0572. 38 sans défaut sur 72, contre 44. | Non : essai négatif. | Non : même raison que F. |
| `recettes_rtx_v5_s03.json` | S03, 15/11/2026 | A0 + synthèse de la campagne v4 (`gemma4:31b-it-q4_K_M`, 9 055 conversations, 897 109 tokens) mêlée à l'affinage, poids 1,09 : 3,02 % des tokens vus. | BPB 1,0557. Par-cœur 4,6 %. 100 sans défaut sur 144, contre 82 pour A0 (p = 0,044). Rejugé le 16/11 sur le jeu « neuf2 » : 96 sans défaut sur 144, par-cœur 1,92 % (mesuré sur le fichier v4 + v5, dont S03 n'a vu que la partie v4). | Oui : **c'est le modèle publié.** Témoin des essais du 16/11, il est resté le modèle retenu. | Oui : avec `synthese_moliere.zip` décompressé à la racine du clone, qui apporte `chatbot_20261114/synthetique_v4/entrainement/moliere.jsonl.zst`. |
| `recettes_rtx_v5_s05.json` | S05, 15/11/2026 | Comme S03, poids 1,85 : part visée 5 %. | BPB 1,0565. Par-cœur 10,2 % : limite de 10 % dépassée. 97 sans défaut sur 144, pour information. | Non : écarté par la garde de par-cœur. | Oui : même fichier v4 que S03, fourni par `synthese_moliere.zip`. |
| `recettes_rtx_v5_s10.json` | S10, 15/11/2026 | Comme S03, poids 3,9 : part visée 10 %. | BPB 1,0582. Par-cœur 23,4 %. 88 sans défaut sur 144, pour information. | Non : écarté. | Oui : même fichier v4 que S03, fourni par `synthese_moliere.zip`. |
| `recettes_rtx_v5_s15.json` | S15, 15/11/2026 | Comme S03, poids 6,2 : part visée 15 %. | BPB 1,0591. Par-cœur 30,5 %. 100 sans défaut sur 144, pour information. | Non : écarté. | Oui : même fichier v4 que S03, fourni par `synthese_moliere.zip`. |
| `recettes_rtx_publie.json` | Modèle publié, 16/11/2026 | La recette `recettes_rtx_v5_s03.json`, gardée telle quelle, plus deux déclarations qui ne changent pas l'entraînement : le fichier de synthèse (`etape1.fichiers_synthetiques`) et le décodage écrit à l'export (`--anchor-mu 1 --candidates 32`). | Même modèle que S03 : BPB 1,05570, 96 sans défaut sur 144 (jeu « neuf2 », 16/11). Étiquette « recettes-rtx-publie-s03 ». | Oui : c'est le modèle publié. | Oui : avec `synthese_moliere.zip` à la racine du clone et les données de la release `chatbot-donnees-20261113` (`python -m chatbot.pipeline_rtx telecharger`). Un réentraînement ne redonne pas forcément les mêmes octets : cela n'a pas été essayé. |
| `recettes_rtx_v6_s03.json`, `recettes_rtx_v6_s05.json`, `recettes_rtx_v6_s10.json`, `recettes_rtx_v6_s15.json` | Essais du 16/11/2026 | A0 + synthèse des campagnes v4 et v5 (27 624 conversations, 2 694 831 tokens), parts visées 3, 5, 10 et 15 % (poids 0,36 ; 0,62 ; 1,3 ; 2,06). La recette déclare ses fichiers de synthèse et écrit le décodage adopté dans le modèle exporté (ancrage, 32 candidates). | Jugés à l'aveugle sur le jeu « neuf2 » (144 réponses par modèle) ; BPB, par-cœur, sans défaut sur 144, écart avec S03 (96) : 3 % : 1,05568, 0,80 %, 93, −3 ; 5 % (part attendue 5,04 %) : 1,05568, 2,69 %, 101, +5 ; 10 % : 1,05746, 6,23 %, 88, −8 ; 15 % : 1,05864, 9,81 % (la limite de 10 % tient de justesse), 105, +9. | Non : aucun candidat n'atteint l'écart exigé de +12 sur « sans défaut ». S03 reste le modèle retenu. Les quatre modèles ne sont pas publiés. | Oui : `chatbot_20261114/synthetique_v6/entrainement/moliere.jsonl.zst` est dans `synthese_moliere.zip`. |
| `prompts_synthetiques.json` | Première synthèse, 13 et 14/11/2026 | Consignes du générateur `qwen3:8b` (176 sujets), lues par `synthetic.py`. | 7 120 appels, 1 305 conversations après fusion et filtres. Contrôle de qualité : 38 sur 60 (seuil 48). | Non pour ses données (essais B et C). Le fichier reste l'origine dont héritent les consignes v4 et v5. | Les consignes sont rejouables avec Ollama ; les textes ne se régénèrent pas à l'identique. |
| `axes_consignes_v4.json` | Campagne de synthèse v4, 14 et 15/11/2026 | Les axes des fiches : 651 notes de thème, 51 types de message, 20 voix d'utilisateur. Aucun texte de conversation. | Voir la ligne suivante. | Oui : données de S03. | Entrée de `tools/composer_consignes_v4.py`. |
| `prompts_synthetiques_v4.json` | Campagne v4 | Consignes complètes : 9 000 fiches, une conversation par appel, générateur `gemma4:31b-it-q4_K_M`. | 13 660 appels, 9 055 conversations après tous les filtres. Contrôle : 113 sur 120 (seuil 96). | Oui : données de S03. | Fichier recomposable à l'identique (vérifié le 16/11/2026 : mêmes octets). Génération rejouable, textes différents. |
| `axes_consignes_v5.json` | Campagne de synthèse v5, 15 et 16/11/2026 | Copie des axes v4, retouchée contre les tics relevés (« Certes », « Hélas », renvoi au médecin, adverbes en tête de réponse). | Voir la ligne suivante. | Données des essais du 16/11, avec celles de la v4. | Entrée de `tools/composer_consignes_v4.py`. |
| `prompts_synthetiques_v5.json` | Campagne v5 | Consignes complètes : 26 000 fiches. | 29 468 appels faits sur 48 000 (arrêt volontaire à 61 %), 20 394 conversations après fusion et refiltrage. Contrôle sur les premières conversations : 52 sur 60 (seuil 48). | Données des essais du 16/11, avec celles de la v4. | Fichier recomposable à l'identique (vérifié le 16/11/2026). Génération rejouable, textes différents. |
| `prompts_tri_fineweb.json` | Tri de FineWeb, 14/11/2026 | Consigne de notation : `qwen3:8b` donne une note à des documents réels (propre, défaut léger, défaut grave). Aucun texte n'est écrit. | 12 000 documents notés, dont 3 425 « défaut grave ». | Non : essais E et F négatifs. | Notation rejouable avec Ollama ; les 12 000 notes ne sont pas publiées. |
| `sonde_fineweb.npz` et `sonde_fineweb.npz.rapport.json` | Tri de FineWeb, 14/11/2026 | Sonde linéaire apprise sur les activations du modèle de base (`tools/trier_fineweb.py`), et le rapport de ses mesures. | Au seuil retenu, sur 320 documents de contrôle : 71 graves écartés sur 87, 4 propres écartés sur 122. | Non : essais E et F négatifs. | Non : la sonde est liée à l'empreinte du modèle de base du 13/11, qui n'est pas publié. |
| `regles_pretrain_v4.json` | Essais E et F, 14/11/2026 | Règles de filtrage du pré-entraînement, source par source (6 sources), lues par `tools/nettoyer_pretrain.py`. FineWeb y ajoute le score de la sonde. | 2 114 055 586 → 1 917 228 299 tokens (9,31 % écartés) ; FineWeb perd 31,54 % de ses tokens. | Non. | Non : dépend de la sonde. |
| `exclusions_theatre.json` | Préparation du théâtre, 12/11/2026 | 142 documents écartés du théâtre après audit ; appliqué par `theatre.py`. | Sert au théâtre de tous les modèles. | Oui. | Non : `theatre.py` lit des fichiers intermédiaires qui ne sont dans aucune release. |
| `exclusions_theatre_v2.json` | Essais D et F, 14/11/2026 | 363 œuvres écartées de l'affinage : 170 traductions, 92 textes qui ne sont pas du théâtre, 88 livrets chantés, 12 en langue non standard, 1 illisible. Appliqué par `tools/nettoyer_affinage.py`. | `theatre_train` passe de 33 396 à 19 153 scènes. | Non. | Liste utilisable depuis un clone, avec le théâtre de la release. |
| `donnees_rtx.json` | Toutes les recettes | Liste des 15 gros fichiers de la release (4 344 702 490 octets), avec tailles et sha256 ; lue par `telecharger` et `verifier`. | — | Oui. | Oui. |

À savoir sur les noms :

- « v4 » et « v5 » ne désignent pas la même chose partout. Les recettes `v4`, `v4a`, `v4e` et `regles_pretrain_v4.json` sont l'essai de nettoyage. Les recettes `v5_s03` à `v5_s15` entraînent sur les conversations de la campagne de synthèse **v4**. Les recettes `v6` entraînent sur les campagnes de synthèse v4 et v5 réunies.
- Dans `recettes_rtx_v3.json`, le texte d'explication de l'étiquette dit encore « v2 = v1 + … » : c'est bien la recette de l'essai C (poids 2).
- `prompts_synthetiques_v5.json` et `axes_consignes_v5.json` portent encore des étiquettes internes « v4 ». Ces fichiers ne sont pas corrigés : les consignes v4 et v5 enregistrent l'empreinte de leurs axes et celle de `prompts_synthetiques.json`, et chaque fichier de données porte l'empreinte de ses consignes.
- Les rapports du lancement A0 sont dans `experiments/chatbot_rtx/20261113_1956_M/`. Les preuves des jugements à l'aveugle sont décrites dans [`experiments/chatbot_rtx/jugements/README.md`](../experiments/chatbot_rtx/jugements/README.md).
- La release `chatbot-donnees-20261113` ne contient que deux petits échantillons de synthèse, pour la recette d'essai (`chatbot_20261113/synthetique_echantillon/`). Les conversations de synthèse des essais sont dans la pièce jointe `synthese_moliere.zip` de la release `chatbot-moliere-20261116` (voir § 2 bis). Le fichier de synthèse des essais B et C (`chatbot_20261113/synthetique_fusion/moliere.jsonl.zst`) n'y est pas : c'est pour cela que ces deux recettes ne se relancent pas.

## 2 bis. Le modèle publié et la release du 16/11/2026

Le modèle publié est **S03**. Le 16/11/2026, quatre candidats ont été comparés à lui sur un jeu de messages neuf (« neuf2 »). La règle était écrite d'avance, le 15/11 à 16 h : un candidat remplace S03 seulement s'il gagne au moins 12 réponses « sans défaut » sur 144, sans que le français, le rapport au message ou le style ne reculent de plus de 6, et s'il passe les deux gardes (BPB sur Molière au plus 1,06602 ; par-cœur sous 10 %). **Verdict : S03 reste le modèle retenu, aucun candidat n'atteint l'écart exigé.** Le meilleur écart est +9 (v6 à 15 %).

Les juges sont des agents IA, pas des personnes : 36 agents, trois par lot de 72 réponses, vote à la majorité. Ceux du 16/11 sont des agents Claude Opus 5.5, un autre modèle d'IA que celui des jugements des 14 et 15/11. Les notes du 16/11 ne se comparent donc pas directement à celles du 15/11. La décision n'en dépend pas, car S03 est rejugé dans le même lot, par les mêmes juges, que les candidats. La qualité jugée par des personnes n'a jamais été mesurée.

- Release : `chatbot-moliere-20261116` du dépôt `benassou423/Inteligence-lab-piscine`. Pièces jointes : `chatbot_moliere_s03.zip` (le dossier `chatbot_moliere/` : `ckpt.pt`, `tokenizer.json`, `chat_config.json`, `training_report.json`), `synthese_moliere.zip` (22 fichiers, 4 442 456 octets, licence CC BY 4.0) et l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`, 445 528 215 octets, avec son manifeste `RESULTATS_CHATBOT_RTX_20261113_1956_M.manifeste.json` ; le nom porte la date de création du lancement, le contenu est l'état du 16/11).
- `synthese_moliere.zip` se décompresse à la RACINE d'un clone : les fichiers se rangent aux chemins que lisent les recettes. Il contient les conversations de la campagne v4 (9 055, fichier du modèle publié), celles des campagnes v4 et v5 réunies (27 624, essais du 16/11), les six fichiers tenus à l'écart et les traces des campagnes.
- Le décodage retenu (32 candidates, ancrage 1) est écrit dans le `chat_config.json` du modèle publié : la commande de chat n'a plus besoin des options `--anchor-mu 1 --candidates 32`.
- Un modèle exporté avec la recette par défaut (`recettes_rtx.json`) garde 16 candidates et `"anchor_mu": 0.0`.
- Les preuves du jugement du 16/11 sont dans `experiments/chatbot_rtx/jugements/final_neuf2/`.

## 3. Le second chatbot : « français courant »

Le même lanceur entraîne un second chatbot, nommé `francais`, qui répond en français d'aujourd'hui et non à la manière de Molière.
Son format est « tirets » : chaque tour de parole est une ligne « — texte », sans nom de personnage (`format.py`).
Ses données viennent de `dialogues.py`, qui extrait des dialogues des romans du corpus (Gutenberg et Wikisource) et convertit les scènes en prose de `theatre_train` : trois fichiers de la release (`romans_train`, `romans_val`, `theatre_moderne_train`).
Il a été affiné une fois, le 13/11/2026, dans le même lancement que A0 : 150 011 904 tokens vus, BPB de 1,15956 à 1,04751 sur `romans_val`. Les rapports sont dans `experiments/chatbot_rtx/20261113_1956_M/modeles/chatbot_francais/`.
Les essais du 14 au 16/11 portent sur le chatbot Molière : les recettes d'essai gardent la partie `francais` sans la changer, et [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md) ne rapporte aucun jugement à l'aveugle de ce second chatbot.
Au 16/11/2026, il n'est pas dans le paquet du modèle (`chatbot_moliere_s03.zip`). Ses poids sont seulement dans l'archive des résultats du lanceur, jointe à la release `chatbot-moliere-20261116`.
