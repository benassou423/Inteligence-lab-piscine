# Assemblée nationale, débats en séance publique — `assemblee_nationale`

**Domaine** : dialogue · **Priorité n° 1** (le genre « dialogue » de la validation et du test vient de cette source) · **Commande** : `uv run python -m pipeline.collect assemblee_nationale` · **Code** : `pipeline/collect/assemblee.py`

## Origine
- Open data **data.assemblee-nationale.fr**, jeu « Débats » : comptes rendus de la séance publique au format **XML Syceron**, une archive zip par législature :
  - **17e législature** (`/17/vp/syceronbrut/syseron.xml.zip`, 652 séances, mise à jour du 2026-11-03) ;
  - **16e législature** (archive, 605 séances) ;
  - **15e législature** (archive de 149 Mo, 2017-2022) : le serveur **coupe l'archive après ~30 Mo** (connexions interrompues, requêtes `Range` vides au-delà, même avec curl). On lit les fichiers complets du début de l'archive tronquée (en-têtes zip locaux) : **315 séances**, soit ~20 % de l'archive (30 Mo sur 149), marquées `meta.archive_partielle`. Pour tout avoir plus tard : relancer avec `--force` (2 minutes si le serveur répond).
  - **14e législature : pas de XML** en open data (404). Seul le HTML existe, et robots.txt impose 30 s entre deux requêtes (~11 h pour 1 300 séances) → non collecté.
- **Un document = une séance**, au **format de la validation du kit** (`data/validation.jsonl`, genre dialogue) :
  - **un paragraphe par intervention**, « **M. Nom : propos** » (nom imprimé, **sans** la qualité : « Mme la présidente », « M. Thomas Ménagé », « Un député du groupe RN ») ;
  - **une ligne vide** entre deux paragraphes ; les sauts de ligne d'une même intervention deviennent des espaces ;
  - mentions sans orateur (« (Il est procédé au scrutin.) », « (L'amendement no 1024 n'est pas adopté.) ») = paragraphes à part ; didascalies en ligne (« (Applaudissements…) ») gardées dans le propos ;
  - **comme dans la validation** : pas de titres des points de l'ordre du jour, pas de signature finale (« Le directeur des comptes rendus… »).
- `date` = **date de la séance** (indispensable pour la décontamination) ; `url` = page du compte rendu sur assemblee-nationale.fr. `meta` : législature, session, numéro, présidence, état, version, nombre d'interventions, d'orateurs et de points.
- Téléchargement robuste : reprise par `Range` tant que des octets arrivent ; abandon après 4 essais sans progrès.

## Licence
- **Licence Ouverte 2.0 (Etalab)** : réutilisation libre, y compris commerciale, avec **mention de la source** (« Assemblée nationale ») et de la date de mise à jour.

## Volume
- **1 550 séances**, **318 Mo** de texte, 5 fichiers : 17e = 630, 16e = 605, 15e = 315 (partielle).
- Contrôle de format (revérifié le 03/11) : les **22 extraits** de la validation antérieurs au gel (11 séances, du 01/10 au 15/10) sont des **sous-chaînes exactes** de nos documents : **421 paragraphes sur 421** identiques, à l'octet près. Les 14 extraits postérieurs (19/10 → 30/10) n'ont pas de séance chez nous (gel).
- **22 séances écartées** car postérieures au gel (après le 16/10/2026).

## Période couverte
- **2017-07-06 → 2026-10-16** (15e à 17e législatures ; une séance de février 2021 est rangée dans l'archive de la 16e).

## Risques
- **Contamination** : la validation publique contient des séances du **1er au 30 octobre 2026** ; celles du 1er au 16 octobre sont dans notre collecte (même séance, extraits). Rien n'est filtré ici : la **date** de chaque séance est renseignée, l'étape de **décontamination** du nettoyage s'en charge.
- **RGPD** : les **noms des orateurs restent** (députés, ministres). Ce sont des **élus dans l'exercice de leurs fonctions**, publiés par l'Assemblée au nom de la transparence : base légale = intérêt légitime, données déjà publiques. Les débats citent aussi des **personnes privées** (victimes, témoins) → ne pas en faire une base de profils.
- **Reproductibilité** : la 15e législature dépend de l'endroit où le serveur coupe l'archive ; une nouvelle collecte peut donner un autre nombre de séances (les 16e et 17e sont complètes et déterministes : mêmes ids, même ordre).
- **Qualité** : version « **avant JO** » pour les séances récentes (non définitive) ; compte rendu intégral = **oral réécrit** (style soutenu) ; formules répétées (« La parole est à… », « (Il est procédé au scrutin.) ») → quasi-doublons de lignes.
- **Biais** : vocabulaire **politique et juridique**, registre formel, orateurs surtout masculins ; c'est justement le registre du test « dialogue ».
