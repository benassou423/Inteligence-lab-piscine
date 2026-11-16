# Variante complète `complet_20261110` : essayée le 10/11/2026, non retenue

État au 16/11/2026 : la variante a été construite et mesurée le 10/11/2026.
Moyenne de deux tirages à 25 001 984 tokens : **1,632570 BPB**, contre
**1,622266** pour la référence `nettoye` (écart de 0,010304 ; plus bas est
meilleur). Le critère de confirmation n'est pas satisfait : la référence est
conservée. Sur 50 paires de quasi-doublons contrôlées, 38 décrivaient des
entités différentes écrites sur un même gabarit. Les trois ajouts (masquage,
qualité web, quasi-doublons) n'ont pas été mesurés séparément. Seul le masquage
des coordonnées a été repris : `tools/prepare_masked_reference.py` l'applique
au mélange final. Preuves : `experiments/ablation_complete_20261110.md` et
`docs/remise/RAPPORT.md`, § 5. La suite de cette fiche décrit la variante telle
qu'elle a été construite.

## 1. Ce qui est ajouté — lecture : 3 minutes

Le corpus `nettoye` reste la référence. La variante `complet_20261110` ajoute :

1. **Coordonnées masquées** : emails, téléphones au format français, IBAN français
   avec clé valide, adresses IP précédées d'un libellé explicite. Le texte reçoit
   des marqueurs `[EMAIL]`, `[TELEPHONE]`, `[IBAN]`, `[IP]`. Les compteurs ne stockent
   pas les valeurs trouvées. Dates, populations, versions et faux IBAN sont testés.
   Les téléphones ont des séparateurs cohérents et ne peuvent être un sous-groupe
   d'une longue série numérique. Dans les cinq sources soignées, un contexte de
   téléphone/contact est exigé à proximité ; les nombres des tables sont préservés.
2. **Qualité du web** : moins de 10 mots ; faible proportion de lettres (<55 %) ;
   mots de longueur moyenne hors [2,20] ; plus de 50 % des caractères dans des
   répétitions de lignes ; trigramme dominant couvrant >30 % des mots. Une page
   de 60 mots au plus composée d'un message de formulaire/cookies est retirée.
   Un texte avec au moins cinq coordonnées et plus de deux coordonnées pour
   100 mots est assimilé à un annuaire. Seuls Common Crawl et FineWeb sont visés.
3. **Quasi-doublons** : textes très semblables, sans être nécessairement identiques.
   Un groupe de cinq mots consécutifs est appelé un *shingle*. Deux textes sont
   comparés par Jaccard : groupes communs / groupes présents dans au moins un texte.

## 2. Comment chercher sans comparer des milliards de paires — lecture : 2 minutes

**MinHash** résume chaque ensemble de groupes de mots par 128 nombres. **LSH**
range ces nombres dans 16 bandes de huit : une bande commune suffit à proposer
une paire. Cette proposition ne provoque jamais seule une suppression.

Le Jaccard est recalculé sur les ensembles uniques de shingles encodés par hash
64 bits. Un retrait exige une similarité d'au moins **0,80** avec un document
**conservé**. Source prioritaire, fichier trié, ligne d'entrée : cet ordre stable
décide du témoin gardé. Une chaîne A proche de B, B proche de C ne suffit pas à
retirer C si C n'est pas assez proche de A. Les textes de moins de 50 mots ne
passent pas dans cette recherche ; les doublons exacts sont traités ensuite.

Le seuil 0,80 correspond à une forte proximité de document, pas simplement au même
sujet. Il s'inspire du seuil de [Gopher, annexe A](https://arxiv.org/html/2112.11446v2),
avec cinq mots plutôt que treize pour tolérer de petites retouches. Cette
adaptation a été évaluée le 10/11 (voir l'état en tête de fiche). Les règles de
qualité sont également adaptées : les seuils anglais et les listes de mots de
Gopher ne sont pas repris aveuglément.

Une proximité élevée peut aussi venir d'un même gabarit utilisé pour des entités
différentes : villages, espèces ou monuments. Le retrait fait alors perdre des
faits spécifiques. La relecture des paires distingue ce cas d'une autre édition
du même livre ou d'un vrai article recopié. Le seuil était une hypothèse ;
l'ablation du 10/11 ne l'a pas confirmée (voir l'état en tête de fiche). Il ne
prouve pas une équivalence sémantique des documents.

La recherche est probabiliste : à Jaccard 0,80, la probabilité théorique d'une bande
commune est `1 - (1 - 0,8^8)^16 = 94,70 %`, dans le modèle d'indépendance idéal.
Des quasi-doublons peuvent rester. La vérification limite les faux retraits,
mais les hashes 64 bits peuvent en théorie entrer en collision. Voir la
[documentation datasketch](https://ekzhu.com/datasketch/lsh.html).

## 3. Ce qui protège les textes et le disque — lecture : 1 minute

L'essai initial sur 2 000 documents a révélé que le seuil de 40 mots supprimait
de vraies définitions et brèves françaises. Il a été abaissé à 10 mots avant
le traitement complet. La relecture a aussi protégé les numéros explicitement
présentés comme identifiants (exemple réel : un numéro d'antenne) et un avis
municipal qui ne devait pas être assimilé à un annuaire pour trois coordonnées.
Un audit ultérieur des masquages à grande échelle a trouvé des faux téléphones
dans des tableaux anciens et des décimales. La première passe complète a été
arrêtée et rejetée avant tout entraînement. Les preuves sont conservées sous
`complet_20261110_pass1_rejected` (dossier de travail, non publié) ; des tests
de ces contre-exemples réels, des séparateurs cohérents et le contexte
téléphonique ont été ajoutés avant reprise.
Livres, poésie, débats, Wikipédia et Wikinews sont
exclus des heuristiques de qualité web ; les coordonnées y sont toujours masquées.

Le masquage n'est **pas une anonymisation** : noms, adresses postales, coordonnées
obfusquées et la plupart des téléphones internationaux ne sont pas couverts.
Les métadonnées d'attribution restent nécessaires et doivent être revues avant
une publication. Les diagnostics de langue de l'ancien texte sont conservés avec
leur provenance, sans les présenter comme recalculés après masquage.

La variante utilise un cache sur disque et une base SQLite pour l'index LSH.
L'option explicite `prune_intermediate_files` retire seulement les parts
intermédiaires générés dans son propre dossier, fichier par fichier, après écriture
réussie de la sortie suivante et de son reçu. Pour une étape globale, toutes les
décisions sont calculées avant cette libération ; aucun fichier encore utile à
la comparaison n'est retiré.
Statistiques, reçus et échantillons restent ; une reprise reconstruit les parts
manquants. Le corpus d'entrée, les autres expériences et les liens nosync sont
protégés. Une réserve disque de 2 Go interrompt la vérification si nécessaire.

La décontamination est rejouée après ces changements, puis la déduplication exacte.
Le contrôle du kit contre les 230 textes de validation doit signaler **0** avant
tout entraînement de comparaison. Les mesures finales et décisions sont consignées
dans `experiments/`, sans modifier la référence sur la seule apparence des textes.

## 4. Rejouer

```bash
uv run python -m tools.complete_pipeline_pilot
uv run python -m pipeline.run --config configs/pipeline/complet_20261110.yaml --workers 8
uv run python -m tools.check_contamination --config complet_20261110 --fast
```

Les longues commandes doivent être lancées par `nohup`, avec leur journal dans
`data/stages/logs/`. Les tests automatisés et relectures IA restent distingués
d'une annotation humaine.

La première commande relit un tirage de 2 000 documents,
`data/stages/targeted_clean_20261104_audit/pilot_input.jsonl.zst` (non publié),
écrit par `tools.targeted_clean_pilot`. Pour tout rejouer depuis la collecte :
`uv run --no-project python tools/reproduce.py --recipe complete`.
La durée sur le corpus entier n'est pas notée dans cette fiche.

Outils qui ont produit l'ablation du 10/11 : `tools/prepare_raw_ablation.py`
(témoins bruts), `tools/run_complete_pipeline.py` (lancement de la variante et
contrôle de contamination), `tools/audit_near_pairs.py` (50 paires de
quasi-doublons), `tools/audit_complete_privacy.py` (masquages réels) et
`tools/compare_complete_results.py` (tableau brut, nettoyé, complet).
L'index de tous les outils est dans [tools/README.md](../../tools/README.md).
