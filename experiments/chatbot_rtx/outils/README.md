# Outils des essais du chatbot (13 au 16/11/2026)

Ce dossier contient les petits scripts qui ont servi à préparer les données de synthèse, à fabriquer les
recettes, à mener les essais et à contrôler le dépôt. Ils sont publiés tels qu'ils ont servi, pour qu'un
lecteur puisse voir comment chaque fichier publié a été obtenu. Seul changement : les quatre scripts shell
partaient d'un dossier propre au poste ; ils partent maintenant de la racine du dépôt.

Ce ne sont pas des outils généraux. Les outils du projet sont dans [`tools/`](../../../tools/README.md).
Tous se lancent depuis la racine du dépôt. Sur le PC Windows à carte NVIDIA, `python` se lit
`.venv\Scripts\python.exe` (voir [`PC_RTX.md`](../../../PC_RTX.md)).

Plusieurs lisent ou écrivent sous `runs/` et `data/stages/`, deux dossiers de travail qui ne sont dans aucun
clone. Les données de synthèse se récupèrent avec la pièce jointe `synthese_moliere.zip` de la release
`chatbot-moliere-20261116`, décompressée à la racine.

## Données de synthèse

| Fichier | Ce qu'il fait |
| --- | --- |
| `campagne_nuit.sh` | Lance la génération : un flux par graine, reprise automatique, serveur Ollama relancé s'il ne répond plus. Le générateur lui-même est `tools/campagne_synthese.py`. |
| `donnees_v6.sh` | Chaîne complète du fichier « v4 + v5 » : fusion et refiltrage de la campagne v5, union avec la v4, retrait des conversations qui contiennent un message de test, tirage de 120 conversations pour le contrôle de qualité. Résultat : 27 624 conversations. |
| `unir_synthese.py` | Réunit deux fichiers de synthèse déjà filtrés en écartant les doublons entre les deux. |
| `garde_protocole.py` | Écarte toute conversation dont un message d'utilisateur est un message des jeux de test : le modèle ne doit pas avoir vu les messages qui le jugent. |
| `traces_publiables.py` | Remplace, dans les traces `.campagne.json`, un chemin absolu par un chemin relatif au dépôt. |

## Recettes

Chaque script écrit des fichiers `chatbot/recettes_rtx_*.json` à partir de `chatbot/recettes_rtx.json`, sans
le modifier. La table des essais est dans [`chatbot/README.md`](../../../chatbot/README.md).

| Fichier | Recettes écrites |
| --- | --- |
| `faire_recette_v2.py` | `recettes_rtx_v2.json`, et `recettes_rtx_v3.json` avec les options `--w` et `--sortie` : première synthèse (essais B et C, non retenus). |
| `faire_recettes_v4.py` | `recettes_rtx_v4.json`, `_v4a.json`, `_v4e.json` : données nettoyées (essais D, F, E, résultat négatif). |
| `faire_recettes_v5.py` | `recettes_rtx_v5_s03.json` à `_s15.json` : campagne de synthèse v4 à 3, 5, 10 et 15 %. |
| `faire_recettes_v6.py` | `recettes_rtx_v6_s03.json` à `_s15.json` : synthèse v4 + v5 aux mêmes parts (essais du 16/11). |
| `faire_recette_publiee.py` | `recettes_rtx_publie.json` : la recette du modèle publié. |

## Essais et gardes

| Fichier | Ce qu'il fait |
| --- | --- |
| `essais_v5.sh` | Les affinages du 15/11 (campagne v4) : entraînement, mesures, rangement, modèle en place remis et vérifié par son empreinte. |
| `essais_v6.sh` | Les affinages du 16/11 (v4 + v5) : mesure des deux témoins, puis quatre candidats. |
| `gardes_v6.py` | Calcule les deux gardes de chaque modèle : BPB sur Molière et par-cœur. Résultat publié dans [`../jugements/final_neuf2/gardes.json`](../jugements/final_neuf2/gardes.json). |
| `decouper_lots.py` | Découpe les réponses mélangées en lots de 72 pour les juges, sans la clé. |

Le mélange des réponses, le dépouillement des notes et la comparaison appariée sont dans
[`../jugements/`](../jugements/README.md) (`aveugle.py`, `recompter_jugements.py`, `apparie.py`).

## Contrôles du dépôt

| Fichier | Ce qu'il vérifie |
| --- | --- |
| `empreinte_code.py` | L'empreinte du code de chaque fichier Python, commentaires et docstrings exclus. Elle prouve qu'une retouche de documentation n'a changé aucun comportement. |
| `empreintes_des_rapports.py` | Les empreintes de code consignées dans un rapport d'entraînement ou d'évaluation, comparées aux fichiers du dépôt. |
| `liens_morts.py` | Chaque lien relatif d'un fichier Markdown mène à un fichier présent et suivi par git. |
| `paquets_release.py` | Fabrique les deux paquets de la release (`chatbot_moliere_s03.zip`, `synthese_moliere.zip`) et leur liste d'empreintes. Deux fabrications donnent les mêmes octets. |

## Ce qui n'est pas ici

Les brouillons, les notes de relecture et les scripts d'un seul usage dont le résultat n'est cité nulle part
ne sont pas publiés. Le journal de travail brut non plus : son récit publiable est
[`docs/chatbot/HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md).
