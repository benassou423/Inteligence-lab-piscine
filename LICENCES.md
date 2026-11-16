# Licences

Ce dépôt contient du code, des modèles entraînés et des données. Ils n'ont pas la même licence, et une
partie du dépôt n'appartient pas à son auteur. Cette page dit ce qui s'applique à quoi.

Auteur du dépôt : `benassou423` (GitHub), 2026.

## Ce que l'auteur du dépôt met sous licence

| Quoi | Où | Licence |
| --- | --- | --- |
| Code et documentation écrits pour ce dépôt | `chatbot/` (sauf les consignes de synthèse, dernière ligne du tableau), `pipeline/`, `tools/`, `minigpt/`, `bench/`, `configs/`, `tests/` (sauf `tests/test_kit.py`), `docs/` (sauf `docs/KIT_README.md`), `README.md`, `PC_RTX.md`, `Makefile`, `pyproject.toml`, `uv.lock` | [Apache 2.0](LICENSE) |
| Rapports d'expérience, journaux, classements automatiques et documents PDF produits par l'auteur, rangés avec la documentation | `experiments/` (sauf les conversations de synthèse recopiées dans `experiments/chatbot_rtx/jugements/controle_synthese_*/`, couvertes par la dernière ligne du tableau), `data/annotations/`, `output/pdf/`, et dans `submissions/` les journaux, métadonnées et générations (`train_log.jsonl`, `submission.json`, `data_meta.json`, `samples.txt`) | [Apache 2.0](LICENSE) |
| Poids des modèles entraînés par l'auteur | modèle du chatbot Molière (`ckpt.pt` et les fichiers qui l'accompagnent ; paquet `chatbot_moliere_s03.zip` de la release `chatbot-moliere-20261116`), les deux chatbots de l'archive des résultats du lanceur (`RESULTATS_CHATBOT_RTX_20261113_1956_M.zip`, même release), modèle déposé dans `submissions/2026-11-11_153839/` (`ckpt.pt` et son `tokenizer.json`) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.fr) |
| Conversations de synthèse et consignes qui les ont produites | `essai_synthetique_*.jsonl.zst` (release `chatbot-donnees-20261113`), conversations de synthèse recopiées pour contrôle dans `experiments/chatbot_rtx/jugements/controle_synthese_*/`, les fichiers de la pièce jointe `synthese_moliere.zip` de la release `chatbot-moliere-20261116` (dont `moliere.jsonl.zst`, le fichier de synthèse du modèle publié), `chatbot/prompts_synthetiques*.json`, `chatbot/axes_consignes_*.json` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.fr) |

Quand deux lignes du tableau couvrent le même fichier, la ligne la plus précise l'emporte.

Pour citer : « benassou423, *Inteligence-lab-piscine*, 2026,
https://github.com/benassou423/Inteligence-lab-piscine ».

Pourquoi le partage à l'identique pour les poids : une partie des textes d'entraînement (Wikipédia,
Wikisource) est sous CC BY-SA 4.0. Savoir si cette licence s'étend à un modèle entraîné n'est pas tranché ;
la licence choisie est celle qui reste valable dans les deux cas.

Les conversations de synthèse ont été écrites par des modèles à poids ouverts sous licence Apache 2.0 :
`gemma4:31b-it-q4_K_M` pour les campagnes v4 et v5, `qwen3:8b` pour le premier essai et pour les deux
petits échantillons `essai_synthetique_*.jsonl.zst` de la release `chatbot-donnees-20261113` (recette
d'essai du lanceur). Elles ont ensuite été
triées par des règles automatiques. Un texte écrit par une machine n'est pas forcément protégé par le
droit d'auteur : la licence CC BY 4.0 vaut pour ce qui peut l'être.

## Ce que ces licences ne couvrent pas

- **Le kit du concours** : `ftb/`, `config/official.json`, `corpus/`, `data/validation.jsonl`,
  `data/VALIDATION_SOURCES.md`, `tests/test_kit.py`, `examples/`, `notebooks/`, `docs/KIT_README.md`,
  `requirements.txt`. Il vient du concours « Feed the Beast » (Intelligence Lab, ECE Paris, 2026 ; dépôt
  `Meuleur/Piscine`, commit `3709c85`). Il est copié ici sans modification parce que les modèles ne
  tournent pas sans lui. Aucune licence n'accompagne cette copie : pour réutiliser le kit en dehors de ce
  dépôt, il faut demander à son auteur.
- **Les textes d'entraînement** distribués dans les releases du dépôt (Wikipédia, Wikisource, Project
  Gutenberg, FineWeb-2, Common Crawl, Wikinews, Assemblée nationale, théâtre, dialogues tirés de romans de
  Gutenberg et de Wikisource). Chacun garde la licence de sa source ; l'auteur du dépôt n'en accorde aucune
  autre. Le détail, source par source, et les réserves (dont le statut en France de certains livres) sont
  dans [`docs/chatbot/FICHE_DONNEES.md`](docs/chatbot/FICHE_DONNEES.md), § 7.
- **Les logiciels utilisés** (PyTorch, tokenizers, Ollama, etc.) : chacun a sa propre licence.

## Crédits

Le code de `minigpt/` a été réécrit pour apprendre, d'après les cours d'Andrej Karpathy (« Let's build
GPT », « Let's build the GPT Tokenizer ») et ses dépôts micrograd, minbpe et nanoGPT, publiés sous licence
MIT. Les fichiers `micrograd.py`, `bpe.py` et `gpt.py` le disent dans leur en-tête.

## Sans garantie

Le code, les modèles et les données sont fournis tels quels, sans garantie d'aucune sorte, comme l'écrivent
les licences ci-dessus. Les modèles sont de petits modèles d'étude : ils écrivent des phrases fausses et ne
doivent servir à aucune décision.
