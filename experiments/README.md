# Expériences : quel fichier prouve quel chiffre

Ce dossier garde les résultats des entraînements du concours (du 03/11 au 10/11/2026) et le reçu de la soumission du 11/11/2026. Tous ces runs ont tourné sur un Mac (colonne `device` : `mps`, précision float32). BPB signifie bits par octet : plus bas est meilleur. Les noms de fichiers sans dossier, et `chatbot_rtx/`, sont dans ce dossier ; les autres chemins partent de la racine du dépôt.

Le chatbot, hors concours, a son propre sous-dossier `chatbot_rtx/` : voir la dernière section.

## Fichiers du concours

| Date | Fichier | Ce qu'il contient | Où lire le chiffre |
|---|---|---|---|
| 03/11 | `throughput_mps.json` | Mesure de débit sur le Mac, pour régler le budget des essais courts. | `tokens_per_s` : 37758, avec des micro-lots de 16. |
| 03/11 au 10/11 | `results.csv` | Une ligne par run du banc d'essai : 12 runs. | BPB global et par genre (4 décimales), tokens vus, durées, empreinte du jeu. |
| 04/11 | `proxy_wiki20261001_fr065.md` | Référence locale à 25 001 984 tokens vus, après reconstruction de Wikipédia. | Ligne `proxy_wiki20261001_fr065` de `results.csv`. |
| 04/11 | `langid_full_audit_20261104.md` | Audit de langue sur documents complets. Décision : garder `fr >= 0,65`. | Dans le rapport ; les dossiers de détail ne sont pas publiés. |
| 04/11 | `targeted_clean_v1_20261104.md`, `.json`, `_verification.json` | Essai de nettoyage ciblé, rejeté : moins bon sur les deux tirages. | Dans le `.json`, `average.delta_bpb` : `0.01736472078399487`. |
| 10/11 | `ablation_complete_20261110.md`, `.json` | Ablation brut, nettoyé, complet : six runs, deux tirages. | Dans le `.json`, `average.brut.bpb`, `average.nettoye.bpb`, `average.complet.bpb` et `pairs`. |
| 10/11 | `full_reference_20261110.json` | Run à 100 007 936 tokens vus sur le mélange de référence, avant masquage des coordonnées. | `val_bpb` : `1.2873910457219313` ; `train_time_s` : 4423.5. |
| 10/11 | `full_reference_masked_20261110.json` | Le modèle soumis : même mélange, coordonnées masquées. | `evaluation.bpb` : `1.2882923104766877` sur 230 documents ; `preparation.counts` : 548 textes masqués, 423 courriels, 486 téléphones, 4 adresses IP ; `meta.train_time_s` : 2039.9. |
| 11/11 | `submission_20261111.json` | Reçu de la soumission `submissions/2026-11-11_153839`. | `files_sha256` : les six empreintes, recalculées identiques le 16/11/2026. |

Ces chiffres sont repris, arrondis, dans le `README.md` de la racine, `docs/remise/RAPPORT.md` et `docs/remise/EXIGENCES.md`.

## Pour lire ces fichiers

- **Dossiers non publiés.** Les rapports citent `runs/`, `data/stages/` et `data/mixes/`. Ce sont des dossiers de travail, ignorés par git. Le jeu final est dans la release GitHub `ftb-final-20261110`. Ne sont pas publiés non plus : `data/annotations/ia-20261104/`, `comparison_spec.json` et le script `data/stages/logs/langid-full-summarize.py`.
- **Chemins dans les JSON.** Le 16/11/2026, les chemins absolus des clés `ckpt` et `data` ont été remplacés par des chemins relatifs à la racine du dépôt dans `ablation_complete_20261110.json` (12 chaînes), `full_reference_masked_20261110.json` (2 chaînes) et `targeted_clean_v1_20261104.json` (8 chaînes). Aucun chiffre n'a changé. L'empreinte `comparison_sha256` de `targeted_clean_v1_20261104_verification.json` est celle de la version d'origine de ce dernier fichier.
- **« ATTENTION flux » dans `results.csv`.** `bench/run.py` ajoute cette mention quand le flux est relu depuis le début, ou quand le budget coupe la fin d'un flux que le banc n'a pas mélangé lui-même (ordre `as_is`, ou plusieurs blocs). Ici, c'est le second cas : ordre `as_is`. Pour le modèle soumis : 100 007 936 tokens vus sur 114 701 016 disponibles. Le mélange masqué garde l'ordre du mélange `full_reference_20261110`, qui était déjà mélangé (`shuffle`). Rien n'est relu deux fois.
- **« satisfait : False » dans `ablation_complete_20261110.md`.** Le critère d'adoption de la variante complète n'est pas rempli : elle n'a pas été adoptée (`proxy_adoption_gate_passed: false` dans le `.json`).
- **`submission_20261111.json`.** `organizer_hidden_score: null` et `organizer_t4_retraining_verified: false` décrivent l'état au 11/11/2026. Le score du test caché et le contrôle sur T4 relèvent de l'organisateur ; ce dépôt n'en contient pas le résultat.

## Chatbot (hors concours)

[`chatbot_rtx/20261113_1956_M/`](chatbot_rtx/20261113_1956_M/) contient les rapports du lancement du 13/11/2026 sur la RTX 5090, sans les poids. [`chatbot_rtx/20261116_modele_publie/`](chatbot_rtx/20261116_modele_publie/) contient les rapports de l'installation du modèle publié (S03) le 16/11/2026 : seuls l'export et l'évaluation ont été relancés, rien n'a été réentraîné. Les rapports du modèle publié sont donc dans `chatbot_rtx/20261116_modele_publie/` et ceux du lancement du 13/11/2026 (chatbot Molière = A0) dans `chatbot_rtx/20261113_1956_M/` ; la base et le chatbot « français courant » sont les mêmes dans les deux dossiers. Le récit daté est dans [`docs/chatbot/HISTORIQUE.md`](../docs/chatbot/HISTORIQUE.md). Les preuves des jugements à l'aveugle sont décrites dans [`chatbot_rtx/jugements/README.md`](chatbot_rtx/jugements/README.md). Le jugement final du 16/11/2026 (S03 reste le modèle retenu, aucun candidat n'atteint l'écart exigé) est dans `chatbot_rtx/jugements/final_neuf2/`. Les petits scripts qui ont préparé la synthèse, fabriqué les recettes, mené les essais et calculé les gardes sont dans [`chatbot_rtx/outils/`](chatbot_rtx/outils/README.md).
