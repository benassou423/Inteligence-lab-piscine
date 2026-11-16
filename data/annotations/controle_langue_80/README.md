# Contrôle du filtre de langue

Classement automatique de 80 extraits, sept sources ; 17 cas limites. Les classes ne sont pas une relecture manuelle indépendante. Les extraits sources ne sont pas recopiés dans ce dossier : leurs IDs et SHA-256 permettent de les retrouver. Les scores, poids, décisions et limites sont conservés pour reproduire les calculs.

Refaire le calcul : `uv run python -m tools.langid_report --out-dir data/annotations/controle_langue_80`

Attention : cette commande réécrit `langid_report.md` dans une forme longue (7 834 caractères contre 743 publiés). Les chiffres doivent rester ceux de `report_summary.json` : pour la règle actuelle en lecture stricte, précision `0.9968824423576331` et rappel `0.9696477677785833`. Pour retrouver le résumé publié : `git checkout -- data/annotations/controle_langue_80/langid_report.md`. Pour ne rien écraser, copier d'abord le dossier ailleurs et donner cette copie à `--out-dir`.

## Fichiers

| Fichier | Contenu |
|---|---|
| `langid_sample.jsonl` | Les 80 documents tirés : identifiant, source, adresse, strate, scores du détecteur, poids, empreinte de l'extrait (`excerpt_sha256`). Pas de texte. |
| `langid_sample.meta.json` | Le tirage du 10/11/2026 : règle `fr`, seuil 0,65, quotas par source, taille de chaque cellule source × strate. |
| `blind_decisions.jsonl` | Les 80 décisions prises sans voir les scores : classe `f` (français), `n` (pas de prose française exploitable) ou `m` (mélangé), motif, cas limite ou non. |
| `blind_decisions_locked.json` | Verrou : empreinte SHA-256 de `blind_decisions.jsonl`, posée avant la comparaison aux scores. Vérifiée identique le 16/11/2026. |
| `langid.jsonl` | Les 80 décisions jointes aux scores et aux poids. |
| `report_summary.json` | Précision et rappel pondérés : lecture stricte, lecture souple, et balayage des seuils pour les règles `fr` et `family`. |
| `sensitivity.json` | Bornes obtenues en faisant varier les 17 cas limites. Ce ne sont pas des intervalles de confiance. |
| `annotation_audit.json` | Comptes de contrôle : 63 `f`, 10 `n`, 7 `m` ; 46 extraits tronqués ; 0 annotation humaine. |
| `langid_report.md` | Résumé court (743 caractères). Aucun outil du dépôt ne produit ce texte : il a été écrit à part. |

## Origine

Ce dossier est un export sans les extraits (`public_snapshot` dans `langid_sample.meta.json`). `tools/finalize_langid_ai_80.py` travaille dans `data/annotations/ia-20261110`, dossier de travail non publié : il y écrit `langid.jsonl`, `report_summary.json`, `sensitivity.json` et `annotation_audit.json`, complète `langid_sample.meta.json` et écrit un `langid_report.md` long, différent du résumé publié ici. Il ne tourne pas sur ce dossier-ci : son dossier est écrit en dur et il lit le texte des extraits. L'étape d'export n'a pas de script dans le dépôt. La version publiée de `annotation_audit.json` a deux champs que cet outil n'écrit pas : `excerpt_text_mismatches` et `source_file_sha256`.

Ne pas confondre avec `data/annotations/ia-20261104` : c'est un autre contrôle (le premier tirage, 300 extraits), cité par `experiments/langid_full_audit_20261104.md` et non publié.

Chiffres repris dans `docs/remise/DATASHEET.md`, `docs/remise/EXIGENCES.md` et `docs/remise/RAPPORT.md`.
