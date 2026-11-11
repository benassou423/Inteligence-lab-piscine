Prochaine action : préparer la soutenance ; première soumission Mac publiée le 11/11, contrôle T4 et classement caché à venir.

# Audit de remise - 11/11/2026

Les consignes sont conservées dans CONSIGNES_TRANSMISES.md. Un export local ne prouve ni sa publication ni son dépôt sur la plateforme.

## Code, données et expériences

| Exigence | État constaté | Preuve ou limite |
|---|---|---|
| Kit, validation et fichiers protégés | Vérifiés | 18 fichiers identiques à la référence, diff des fichiers protégés vide |
| Sept liens nosync | Conservés | Gardes et tests de création/reprise ; jonctions Windows prévues |
| Sources et fiches | Sept actives, archive HF distincte | Origine, licences déclarées, volumes, périodes et risques dans docs/sources/ |
| Common Crawl brut | 10 WARC intégraux d'un même segment | 9,3 Go lus, 11 865 pages françaises avec HTML brut ; pas tout le répertoire du segment |
| Wikipédia refaite, HF2023 conservée | Fait | frwiki20261001 ; essai 2 000 et audit complet documentés |
| Nettoyage de référence | Exécuté | 3 119 274 documents, seuil fr 0,65 ; encodage, extraction, langue, paragraphes |
| Déduplication exacte | Exécutée | Hash normalisé global, tests de priorité et reprise |
| MinHash/LSH | Exécuté et évalué | 128 permutations, 16×8 bandes, cinq mots, Jaccard ≥0,80 ; 8 722 retraits |
| Seuil de quasi-doublons | Contrôlé et limité | 50 paires intégrales vérifiées sans hash ; 38 décrivent des entités différentes avec un même gabarit |
| Qualité et masquage | Implémentés et testés | Variante complète évaluée ; mélange final : 548 textes masqués, 423 emails, 486 téléphones, 4 IP |
| Échantillon de langue manuel | Écart documenté | Classement automatique : 80 extraits, sept sources ; aucune annotation manuelle attestée |
| Précision du filtre | Mesurée relativement aux classes automatiques | 99,69 %, sensibilité aux cas limites 99,56–99,80 % ; 17 cas limites, 46 extraits tronqués |
| Ablation brut/nettoyé/complet | Six runs terminés | Contrat identique, 25 001 984 tokens, tokenizer figé ; moyennes 1,657747 / 1,622266 / 1,632570 BPB |
| Décision | Référence retenue, coordonnées masquées | Complet moins nettoyé +0,010304 BPB ; variante non adoptée |
| Reproduction par une commande | Exécutée avec les bruts locaux et le cache | tools/reproduce.py --recipe final --skip-collect ; même hash final ; nouveau téléchargement intégral non attesté |
| Tokenizer 8 192 sans perte | Vérifié | SHA 35c0f184... ; vraie recette restaurée dans configs/bench/proxy_wiki20261001_fr065.yaml |
| Données synthétiques | Aucune ajoutée explicitement | Origines non détectées du web possibles ; aucun générateur utilisé dans cette recette |

## Modèle et remise

| Exigence | État constaté | Preuve ou travail restant |
|---|---|---|
| Architecture et initialisation | Vérifiées | 10 166 400 paramètres, kit inchangé, pas de poids pré-entraînés |
| Budget et paramètres officiels | Vérifiés | 3 052 pas, 100 007 936 tokens, tous les paramètres d'entraînement identiques au kit |
| Flux sans répétition involontaire | Vérifié | 114 701 016 tokens disponibles pour 100 007 936 vus |
| Modèle final masqué | Terminé et contrôlé | full_reference_masked_20261110, BPB 1,288292 sur 230 documents, MPS float32, 2 039,9 secondes |
| Empreinte finale | Vérifiée | b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee |
| Provenance | Vérifiée ligne par ligne | 72 612 lignes, hashes avant/après, 0 URL/licence déclarée manquante |
| Contamination | Zéro document signalé | Contrôle officiel du mélange final ; cela ne signifie pas zéro phrase commune |
| Logs, versions et cinq générations | Conservés | Run final, CSV d'expériences, metadata et samples.txt |
| Vérification CUDA préalable | Facultative, non exécutée | Modèle Mac retenu par Ben le 11/11 ; ftb.submit n’impose pas CUDA. Le paquet PC reste disponible en option |
| Réentraînement T4 à ±0,02 BPB | Non attesté | Le résultat MPS et la conversion des poids ne prouvent pas l'équivalence d'un entraînement CUDA |
| Paquet produit par le kit | Créé et publié | submissions/2026-11-11_153839 ; tous les poids exportés concordent avec les poids entraînés convertis en float16 |
| URL du dataset | Publiée et vérifiée | GitHub Release ftb-final-20261110, dix fichiers retéléchargés sans authentification ; empreintes et flux de tokens identiques |
| Dépôt officiel | Première soumission du 11/11 | Dossier submissions/2026-11-11_153839 publié sur le dépôt pour collecte par le mentor ; évaluation cachée non encore attestée |
| Commit et push | Réalisés avec accord du 11/11 | Sources, documents et paquet officiel présents sur main ; contrôles de branche distante et des fichiers publiés |

## Livrables et soutenance

| Livrable | État | Preuve ou limite |
|---|---|---|
| Autograd, bigramme, BPE et transformer | Construits et testés | minigpt/ et fiches docs/comprendre/minigpt_* |
| Molière : courbe, BPB et cinq générations | Exécuté | BPB 1,738328, bigrammes 2,494216 et 3,354389 |
| Rapport d'ablation ≤2 pages | Préparé | output/pdf/rapport_ablation.pdf, six runs et limites |
| Datasheet et carte | Préparées | output/pdf/datasheet.pdf, DATASET_CARD.md ; licences déclarées distinctes d'une validation juridique indépendante |
| Support oral | Sauvegardé, modifiable | 14 diapositives, graphiques/tableaux natifs et notes actualisées |
| Appropriation orale | À vérifier avec Ben | Les supports ne démontrent pas la compréhension de chaque ligne à sa place |
| Dépôt effectif avant le 16/11 | Première remise le 11/11 | Dataset public et soumission Mac ; contrôle préalable CUDA facultatif, réentraînement T4 de l’orga non attesté |

Les mélanges, répétitions, ordre des données et comparaison HF2023/dump2026 à
recette identique restent des pistes supplémentaires, non présentées comme réalisées.
