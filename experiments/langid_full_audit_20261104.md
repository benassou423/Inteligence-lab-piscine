Prochaine action : nettoyer les défauts de structure avant de réessayer un filtre plus large — lecture 4 min.

# Audit de langue sur documents complets — 4 novembre 2026

**Décision : conserver `fr >= 0,65`.** La règle `family >= 0,65` additionne les scores du français, du normand, du francoprovençal, du picard et du wallon. Elle récupère du français rejeté à tort, mais aussi des documents bilingues, des sommaires et des catalogues. L’accord de deux détecteurs ne suffit pas à valider ces ajouts. Aucun réglage ni document du corpus actif n’a changé.

## 1. Ce qui a été contrôlé

Le recensement couvre les **3 221 115 documents**, soit **17 681 692 192 octets de texte UTF-8**, des 251 fichiers précédant le filtre de langue. Les décisions des deux règles sont calculées sur les scores déjà enregistrés, avec le même seuil. Le total concorde avec les statistiques de l’étape.

L’analyse automatique détaillée couvre **888 documents complets**, soit **26 658 801 octets** : les 300 documents du premier tirage, les 513 ajouts hors Wikipédia, 60 nouveaux ajouts Wikipédia répartis en trois classes de longueur, et 15 témoins déjà acceptés des trois sources auparavant absentes. Les 60 Wikipédia sont choisis par empreinte déterministe, 20 par classe ; ce n’est pas un échantillon uniforme de tous les ajouts. Les témoins comprennent quatre choix par empreinte et le plus long document de chaque source.

GlotLID et fastText `lid.176` analysent le document entier puis des passages successifs de 3 000 caractères au maximum. **9 016 passages couvrent exactement les 25 453 374 caractères**, sans trou. Les scores GlotLID du corpus sont reproduits à une tolérance de 0,00001. Les textes initiaux de 1 500 caractères et les tailles annoncées concordent avec les 300 documents retrouvés.

« Complet » signifie ici **tout le texte collecté dans le fichier** : certains fichiers Wikisource ne contiennent que la couverture ou le sommaire du livre annoncé. Cela ne prouve pas la collecte intégrale de l’ouvrage imprimé.

## 2. Effet exact du changement proposé

Au seul stade du filtre de langue : **3 120 231 → 3 167 218 documents** ; **17 369 722 288 → 17 499 732 054 octets**. Le candidat ajouterait **46 987 documents / 130 009 766 octets**, soit environ **+0,75 % de texte**. Ces nombres précèdent la remise en forme, la décontamination et la déduplication ; ils ne remplacent pas les volumes du corpus final.

| Source | Documents ajoutés | Octets ajoutés | Ajouts analysés par les deux détecteurs |
|---|---:|---:|---:|
| Wikipédia | 46 474 | 120 881 067 | 90 |
| Wikinews | 329 | 495 131 | 329 sur 329 |
| Wikisource | 121 | 7 528 138 | 121 sur 121 |
| Gutenberg | 7 | 1 004 835 | 7 sur 7 |
| Common Crawl | 56 | 100 595 | 56 sur 56 |
| Assemblée nationale | 0 | 0 | 5 témoins |
| FineWeb-2 | 0 | 0 | 12 documents initiaux |

Les trois sources absentes des premières annotations représentent 41,6 % du texte avant filtre, mais seulement **128 ajouts / 8 532 973 octets** dans ce changement. L’Assemblée ne changerait pas. Le défaut de couverture du premier tirage ne doit donc pas être confondu avec le volume réellement touché.

Une recherche de signes de structure sur **tous les 46 987 ajouts** trouve 1 262 documents avec au moins un marqueur tel que `{{`, `|}`, `rowspan=` ou `colspan=` : 1 229 Wikipédia, 18 Wikinews et 15 Wikisource. Elle repère aussi 192 documents contenant une suite d’au moins dix mots identiques. Ce sont des **signaux descriptifs**, pas 1 262 ou 192 erreurs validées ; ils peuvent se recouper et ne couvrent pas tous les défauts.

## 3. Relectures IA et exemples

**75 documents relus par Codex : 64 intégralement et 11 par cinq passages de 1 000 caractères répartis du début à la fin.** Les bornes, empreintes, motifs et décisions sont conservés. Les 30 ajouts du premier échantillon ont tous été relus intégralement. Les sept ajouts Gutenberg ont été examinés, dont cinq par passages. Au total, 258 422 caractères ont été présentés pour ces relectures.

Cette sélection vise surtout les risques : désaccord des détecteurs, changement de langue après le début, tableaux, livres longs. Elle est complétée par des choix déterministes sans présélection positive et des témoins. **Aucun taux de précision global ne doit être calculé à partir de ces 75 cas.** Il s’agit d’annotations IA, pas d’une validation humaine ; les onze contrôles par passages ne valident pas toutes les phrases des documents concernés.

| Observation | Exemple contrôlé | Conséquence |
|---|---|---|
| Vrai français rejeté | Wikinews `57629`, décès de Marco Gal ; Common Crawl, annonce du congrès d’Arles | Il existe des textes à récupérer. Le sujet « langues minoritaires » ne détermine pas la langue du texte. |
| Deux détecteurs peuvent manquer le bilinguisme | Gutenberg `37854`, *Le Poème du Rhône*, contient du provençal et du français ; `47919`, chansons de Bresse, mêle patois et traductions | `lid.176` donne pourtant un score français global de 0,916729 et 0,931620. Un score n’est pas une garantie. |
| Le début ne représente pas la suite | Dictionnaire français-italien de Ronna : préface française, puis conjugaisons, listes bilingues et préface italienne | Les 1 500 premiers caractères ne suffisent pas. |
| Titre de livre sans livre | Wikisource `2908304`, *L’Utopie* : seulement une liste d’éditions ; Gutenberg `75297`, atlas : légendes et crédits | Reconnaître des mots français ne suffit pas à valider la matière d’apprentissage. |
| Bon corps français, structure parasite ensuite | Wikipédia `2737000`, Tomáš Pospíšil ; `3122258`, Human Powered Health : biographies puis tableaux bruts | Ces cas sont déjà acceptés : le filtre actuel ne résout pas la qualité de structure. |
| Le début peut aussi sous-estimer le français | Wikipédia `5886867`, coffret Miles Davis : commentaires français après une liste de titres | Ne pas remplacer l’analyse complète par le seul début. |

Les catalogues de noms propres et les titres étrangers ne sont pas automatiquement de la prose étrangère. Langue, qualité de conversion et intérêt pour l’entraînement restent trois questions distinctes. Les verdicts diagnostiques se répartissent en 10 ajouts jugés acceptables dans les cas lus, 20 documents à ne pas ajouter en bloc, 19 à nettoyer, 24 à comparer séparément et 2 témoins cohérents. Ces catégories ne sont appliquées à aucune donnée de production.

## 4. Décision et travail suivant

Conserver **`fr >= 0,65`**. L’audit confirme des faux rejets mais réfute l’idée qu’une simple addition des langues proches récupère uniquement du français utile. Il ne mesure pas le gain d’entraînement du candidat.

La prochaine comparaison doit d’abord traiter les restes de tableaux et de modèles, les répétitions, les couvertures/sommaires et les documents réellement bilingues. Il faut conserver la prose française et tester les règles sur des exemples ordinaires également, sans supprimer tous les noms étrangers. Préparer un corpus expérimental distinct, mesurer les pertes et gains par source, puis comparer deux graines identiques au corpus de référence. « Graine » désigne le réglage qui fixe les tirages aléatoires ; deux essais limitent les conclusions dues au hasard.

La référence existante `proxy_wiki20261001_fr065` reste à **1,6155 BPB**, une seule graine. BPB signifie bits par octet : plus bas est meilleur. Aucun nouvel entraînement ni soumission n’a été lancé pour cet audit.

## 5. Preuves et reproduction

**277 tests verts en 111,21 s** ; 44 avertissements existants sur le lancement de processus Python. Kit identique à la référence, sept liens `.nosync` préservés. Configuration, manifeste et signatures des 251 fichiers du corpus final inchangés. Le contrôle de contamination précédent reste applicable : **0 document signalé sur 230**, recouvrement maximal de 8,4 % sous le seuil de signalement de 10 %. Ce résultat ne signifie pas l’absence de toute phrase commune. Aucun nouveau contrôle de contamination n’était nécessaire pour cet audit en lecture seule.

Les tableaux et les relectures exportés sont dans `data/annotations/ia-20261104/full_document_audit/`. Les textes complets, scores, bornes des passages, signatures des entrées et preuves détaillées restent dans `data/stages/langid_full_audit_20261104/`. `final_verification.json` rassemble les contrôles finaux. Aucun nouveau commit ni push.

À lancer depuis la racine du projet ; les tâches longues sont détachées avec `nohup`, avec leurs journaux dans `data/stages/logs/` :

```bash
nohup uv run python -m tools.audit_langid_full extract --workers 8 > data/stages/logs/langid-full-extract.log 2>&1 &
# Attendre le succès de l'extraction avant l'analyse.
nohup uv run python -m tools.audit_langid_full analyse --workers 2 > data/stages/logs/langid-full-analyse.log 2>&1 &
uv run python data/stages/logs/langid-full-summarize.py
uv run pytest -q
```

Le cache de l’audit dépend des signatures des entrées, du code, des modèles et des textes. Les empreintes des modèles figurent dans `summary.json`. Le second modèle provient de la [distribution officielle fastText](https://fasttext.cc/docs/en/language-identification.html), sous CC-BY-SA 3.0 ; les deux détecteurs servent de diagnostic et leurs erreurs peuvent être corrélées.
