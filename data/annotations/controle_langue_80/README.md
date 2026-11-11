# Contrôle du filtre de langue

Classement automatique de 80 extraits, sept sources ; 17 cas limites. Les classes ne sont pas une relecture manuelle indépendante. Les extraits sources ne sont pas recopiés dans ce dossier : leurs IDs et SHA-256 permettent de les retrouver. Les scores, poids, décisions et limites sont conservés pour reproduire les calculs.

Commande : uv run python -m tools.langid_report --out-dir data/annotations/controle_langue_80
