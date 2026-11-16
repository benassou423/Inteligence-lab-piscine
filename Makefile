# Makefile historique (macOS/Linux) : `make all` rejoue la référence de base (collecte, nettoyage, run proxy).
# Il ne reconstruit ni le jeu final du concours ni le chatbot.
# Jeu final du concours : uv run --no-project python tools/reproduce.py --recipe final (docs/remise/REPRODUCTION.md).
# Chatbot Molière (hors concours) : python -m chatbot.pipeline_rtx (guide : PC_RTX.md).
# PC Windows à carte NVIDIA : ne pas utiliser ce Makefile (`uv sync` et `uv run` y installeraient un PyTorch sans CUDA).
PY := uv run python

.PHONY: setup test collect clean-v1 proxy all

setup:
	uv sync

test:
	uv run pytest -q

# Collecte des sources brutes → data/raw/
collect:
	$(PY) -m pipeline.collect all

# Nettoyage v1 (encodage, langue, boilerplate, doublons exacts) → data/stages/nettoye/
clean-v1:
	$(PY) -m pipeline.run --config configs/pipeline/nettoye.yaml

# Run proxy local (kit officiel, budget réduit) : mix, tokenizer, entraînement, BPB par genre → experiments/results.csv
proxy:
	$(PY) -m bench.run --config configs/bench/proxy.yaml --stage all

all: collect clean-v1 proxy
