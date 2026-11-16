# Consigne des juges et des lecteurs

Ce document dit ce que les juges ont vu, ce qu'on leur a demandé, et comment leurs notes sont comptées.
Les juges et les lecteurs sont des **agents IA** (Claude, modèle fermé d'Anthropic), jamais des personnes.
Les juges du jugement final du 16/11/2026 sont d'un autre modèle de cette famille que ceux des jugements
précédents (§ 2).

Il y a deux consignes :

- celle des **juges**, qui notent des réponses du chatbot (§ 1 à § 4) ;
- celle des **lecteurs**, qui contrôlent la qualité des conversations de synthèse (§ 5).

Les fichiers cités sont dans ce dossier ; le tableau des chiffres est dans [`README.md`](README.md).

## 1. Ce que les juges voient

Le programme est [`aveugle.py`](aveugle.py), commande `prepare`. Il fait ceci :

1. Il lit les réponses de chaque modèle. Une réponse est repérée par son message (`job`), sa graine et son
   tour de parole.
2. Il garde les graines 11, 22 et 33.
3. Pour une conversation de plusieurs tours, il joint à chaque réponse les échanges précédents de la même
   conversation, avec les réponses du même modèle (champ `avant`).
4. Il mélange toutes les réponses de tous les modèles, avec une graine de mélange fixe (`--melange`).
5. Il donne à chaque réponse un identifiant dans l'ordre du mélange : `r001`, `r002`, etc.
6. Il écrit deux fichiers. `items.json` est ce que les juges voient : identifiant, échanges précédents,
   message, réponse. `cle.json` dit quel modèle a écrit quelle réponse. Les juges ne voient jamais `cle.json`.

Les réponses mélangées sont ensuite coupées en **lots de 72**, dans l'ordre : le lot 1 contient `r001` à
`r072`, le lot 2 `r073` à `r144`, etc. Chaque lot est un fichier écrit dans un dossier temporaire, sans la
clé. Vérifié le 16/11/2026 pour les 8 premiers jugements (pas pour `final_neuf2`) : les lots encore présents sur le poste, mis bout à bout,
redonnent `items.json` à l'identique.

Chaque lot est noté par **3 juges**. Un juge est un agent IA lancé avec la consigne du § 2 et un seul lot.
Il ne voit ni les autres lots ni les notes des autres juges. Un jugement de 720 réponses compte donc 10 lots
et 30 agents. Le jugement du 16/11/2026 (`final_neuf2`, 864 réponses) compte 12 lots et 36 agents.

Graine de mélange de chacun des 8 premiers jugements, retrouvée le 16/11/2026 en relançant `prepare` sur les
réponses d'origine (les deux fichiers obtenus sont identiques à ceux de ce dossier) :

| Sous-dossier | Modèles mélangés | Graine de mélange | Réponses | Lots |
| --- | --- | ---: | ---: | ---: |
| `synthese_qwen_reglage` | A0, B, C | 20261113 | 216 | 3 |
| `synthese_qwen_test` | A0, C | 20261114 | 144 | 2 |
| `decodage_reglage` | A0, V1, V2, V3 | 20261114 | 288 | 4 |
| `decodage_test` | A0, V2 | 20261115 | 144 | 2 |
| `donnees_nettoyees_reglage` | A0, D, F | 20261117 | 216 | 3 |
| `pretrain_filtre_reglage` | A0, E | 20261118 | 144 | 2 |
| `synthese_qwen_ancre_reglage` | A0, C | 20261119 | 144 | 2 |
| `campagne_v4_neuf` | A0, S03, S05, S10, S15 | 20261120 | 720 | 10 |
| `final_neuf2` | A0, S03t, s03, s05, s10, s15 | non retrouvée | 864 | 12 |

Pour `final_neuf2`, la graine de mélange n'a pas été retrouvée : `prepare` ne peut pas être rejoué pour ce
jugement. `items.json` et `cle.json` sont publiés tels que les juges et le dépouillement les ont utilisés.

## 2. La consigne des juges

Texte exact envoyé à chaque juge. Deux remplacements seulement : le chemin du fichier à lire est remplacé
par « LOT.json », et le numéro du juge (1, 2 ou 3) par « N ».

```text
Tu es juge dans une évaluation À L'AVEUGLE de petits chatbots français qui doivent tenir une conversation
basique en répondant « à la manière de Molière » (théâtre classique du XVIIe siècle).

Lis EN ENTIER ce seul fichier, et AUCUN autre fichier (ne cherche pas d'où viennent les réponses, n'ouvre aucun
autre dossier, ne lance aucune commande) :
LOT.json

Il contient 72 éléments. Chaque élément a :
- "id" : identifiant ;
- "avant" : les échanges précédents de la même conversation (liste de {message, reponse}, souvent vide) ;
- "message" : ce que l'utilisateur d'aujourd'hui vient d'écrire ;
- "reponse" : la réponse du chatbot à CE message. C'est elle, et elle seule, que tu notes.
Les réponses viennent de plusieurs chatbots différents, mélangés ; tu ne sais pas lesquels et tu n'as pas à le deviner.

Pour chaque élément, donne trois notes, 1 ou 0 :
- F (français) : 1 si la réponse est faite de phrases françaises compréhensibles et FINIES. 0 si une phrase est
  coupée, si un passage est du charabia, s'il y a un mot inventé ou déformé, ou une faute de grammaire grossière qui
  gêne la compréhension. Les tournures classiques correctes (« ne … point », « parbleu », « je vous en conjure »)
  valent 1. Un mélange de « tu » et de « vous » pour la même personne n'est PAS une faute de F (c'est jugé dans S).
- R (rapport) : 1 si la réponse a un rapport CLAIR avec CE message : le salut est rendu, la question reçoit une
  réponse (même brève, même par une pirouette qui porte sur la question), le propos de l'utilisateur est commenté.
  0 si la réponse parle d'autre chose, ignore la question, ou pourrait suivre n'importe quel message. Tiens compte de
  "avant" seulement pour comprendre le message ; une réponse qui se rattache à un échange précédent mais pas au
  message courant vaut 0.
- S (style) : 1 si la réponse vouvoie l'utilisateur ou emploie une adresse classique (monsieur, madame), a une
  tournure de théâtre classique, et ne contient aucun mot ni aucune idée d'aujourd'hui venant du chatbot. 0 s'il
  tutoie l'utilisateur, s'il parle comme aujourd'hui, ou s'il mélange « tu » et « vous » pour la même personne.
  Une réponse courte et neutre mais d'allure classique (« Fort bien, monsieur. ») vaut 1.
Les trois notes sont INDÉPENDANTES : une réponse en beau français classique mais hors sujet vaut F=1, R=0, S=1.
Sois exigeant et constant : applique la même sévérité du premier au dernier élément. En cas de doute réel sur R,
demande-toi : « si je cachais le message, pourrais-je deviner de quoi il parlait ? » Si non, R=0.

Rends exactement 72 notes, une par "id" du fichier, sans en oublier ni en inventer.
(Tu es le juge n° N de ce lot ; d'autres juges notent les mêmes éléments sans se concerter.)
```

Le juge rend une liste de notes dans un format imposé : pour chaque réponse, son `id` et trois entiers
`F`, `R`, `S` qui valent 0 ou 1. Il ne rend aucun commentaire.

Trois scripts de notation ont servi aux jugements des 14 et 15/11/2026 (deux du matin du 14/11, un de 13 h 33
le même jour). Leur consigne est la même, mot pour mot. Le troisième prend le dossier des lots et le nombre de
lots en paramètres. D'après le journal de travail, il a resservi tel quel pour les jugements suivants des 14 et 15/11.

Un quatrième script, écrit le 16/11/2026 à 1 h 58 pour le jugement `final_neuf2`, a la même consigne, mot pour
mot (comparée par différence de texte avec les trois autres). **Ce jugement a utilisé la consigne ci-dessus
sans changement, mais avec des juges d'un autre modèle d'IA : des agents Claude Opus 5.5, au lieu du modèle
des jugements précédents.** Il y a eu 36 juges pour 12 lots de 72 réponses (3 juges par lot). Les notes du
16/11 ne se comparent donc pas directement à celles du 15/11.

## 3. La grille F, R, S

| Note | Question | Vaut 0 si |
| --- | --- | --- |
| **F** (français) | Les phrases sont-elles françaises, compréhensibles et finies ? | phrase coupée, charabia, mot inventé ou déformé, faute de grammaire grossière |
| **R** (rapport) | La réponse a-t-elle un rapport clair avec ce message ? | elle parle d'autre chose, ignore la question, ou pourrait suivre n'importe quel message |
| **S** (style) | La réponse a-t-elle le vouvoiement et la tournure du théâtre classique ? | tutoiement, façon de parler d'aujourd'hui, mélange de « tu » et de « vous » |

Les trois notes sont indépendantes. Le tableau résume la consigne ; c'est le texte du § 2 qui fait foi.

## 4. Le vote à la majorité

Le programme est [`aveugle.py`](aveugle.py), commande `agrege`.

- Pour chaque réponse et chaque critère, on garde la note donnée par **la majorité des 3 juges** : 1 si au
  moins 2 juges ont mis 1, sinon 0.
- Une réponse à laquelle il manque toutes les notes d'un critère est listée dans `manquants`. Il n'y en a
  aucune dans les 9 jugements de ce dossier.
- Une réponse est **« sans défaut »** (lettre T dans les documents) si ses trois notes à la majorité valent 1.
  `agrege` ne calcule pas T : les lignes de Python du [`README.md`](README.md) le font, ainsi que
  [`recompter_jugements.py`](recompter_jugements.py) et [`apparie.py`](apparie.py).
- `accord_unanime` compte, par critère, les réponses où les 3 juges ont mis la même note.

Dans `notes.json`, le « juge 1 » réunit les juges n° 1 de tous les lots. Ce sont des agents différents d'un
lot à l'autre : les totaux « par juge » sont des totaux par place, pas par personne.

## 5. La consigne des lecteurs du contrôle de qualité

Pour contrôler un fichier de conversations de synthèse, on en tire au hasard 60 ou 120 (commande `controle`
de [`tools/nuit_moliere.py`](../../../tools/nuit_moliere.py), tirage à graine fixe).
Le fichier obtenu commence par 5 lignes de statistiques, puis donne les conversations numérotées. Trois
lecteurs le lisent en entier, chacun de son côté.

Texte exact envoyé à chaque lecteur, pour le contrôle de 120 conversations du 15/11/2026. Deux
remplacements seulement : le chemin du fichier à lire est remplacé par « CONTROLE.txt », et le numéro du
lecteur (1, 2 ou 3) par « N ».

```text
Tu es un relecteur exigeant de français. Lis EN ENTIER le fichier
CONTROLE.txt
Il contient 5 lignes de statistiques puis 120 conversations numérotées « ### 1. … » à « ### 120. … », tirées au hasard
d'un jeu de données d'entraînement pour un petit chatbot qui doit « répondre à la manière de Molière ».
Format de chaque conversation : des répliques « NOM. » suivi du texte. Le PREMIER nom qui parle est A (l'utilisateur
d'aujourd'hui, il parle en français courant, il peut tutoyer ou vouvoyer, il peut citer des objets modernes). Le SECOND
nom est B (le personnage : MOLIÈRE, SGANARELLE, TOINETTE…), c'est LUI que le chatbot devra imiter.

Pour CHACUNE des 120 conversations, note quatre critères, 1 (bon) ou 0 (défaut) :
- F : toutes les répliques sont du français correct, compréhensible et FINI (pas de phrase coupée, pas de mot inventé ou
  déformé, pas de faute de grammaire grossière, pas de caractères parasites). Les tournures classiques (« ne … point »,
  « je vous en conjure ») sont correctes.
- R : CHAQUE réplique de B a un rapport clair avec ce que A vient de dire (elle répond à la question, rend le salut, ou
  réagit au propos). Une réponse hors sujet, ou qui contredit sans raison ce que A a dit, ou qui confond les rôles, vaut 0.
- V : B vouvoie A dans toutes ses répliques (aucun « tu, te, toi, ton, ta, tes » adressé à A). Une réplique de B sans
  pronom d'adresse ne compte pas comme défaut.
- S : les répliques de B ont une tournure de théâtre classique (adresse « monsieur / madame », vocabulaire et syntaxe
  d'époque ou neutres) et B n'emploie pas DE LUI-MÊME un mot ou une idée d'aujourd'hui (téléphone, internet, stress,
  « ok », anglicisme…). B peut reprendre avec étonnement un objet moderne que A vient de nommer : ce n'est pas un défaut.
« sans_defaut » vaut 1 seulement si F, R, V et S valent tous 1.
Sois strict mais juste : ne mets 0 que pour un défaut que tu peux citer ; dans « defaut » recopie alors les quelques
mots fautifs et dis pourquoi (20 mots au plus). Si tout est bon, « defaut » = "".
Ne lis aucun autre fichier, n'écris rien, ne lance aucune commande autre que la lecture de ce fichier.
Rends exactement 120 entrées, numéros 1 à 120, dans l'ordre.
(Tu es le lecteur n° N ; d'autres lecteurs notent les mêmes conversations sans se concerter.)
```

Les cinq contrôles de ce dossier ont la même consigne. Seuls changent le fichier lu et le nombre de
conversations (60 ou 120).

Le lecteur rend, pour chaque conversation, son numéro, quatre notes `F`, `R`, `V`, `S` (0 ou 1), la note
`sans_defaut` et le texte du défaut. Il rend aussi une impression générale.

Dépouillement :

- par critère, on garde la note de la majorité des 3 lecteurs ;
- une conversation est **sans défaut** si au moins 2 lecteurs sur 3 lui ont mis 1 aux quatre critères. Ce
  n'est pas le même calcul que pour les juges : une conversation peut avoir ses quatre critères à 1 à la
  majorité et ne pas être sans défaut, si deux lecteurs lui trouvent chacun un défaut différent. Le cas se
  présente 1 fois, dans le contrôle du 14/11/2026 au matin ;
- le fichier de notes garde les textes des défauts relevés et les impressions des trois lecteurs.

Le seuil à atteindre est fixé avant la lecture : 48 sans défaut sur 60, ou 96 sur 120.

## 6. Ce que ces consignes ne garantissent pas

- La consigne interdit au juge d'ouvrir un autre fichier. Ce dossier ne contient pas de trace qui permette
  de vérifier, après coup, que chaque agent l'a respecté.
- Les juges notent ce qu'ils lisent. Ils ne savent pas si une réponse est recopiée des données
  d'entraînement.
- Les limites des jugements eux-mêmes (petits jeux, juges IA, jeux réutilisés) sont dans le
  [`README.md`](README.md).
