# Une commande pour tout regénérer : make all
PY := uv run python

.PHONY: setup test collect clean-v1 val proxy all

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

# Val provisoire (en attendant le kit) → data/val/provisional/
val:
	$(PY) -m bench.val

# Run proxy local : tokenizer, entraînement, BPB par domaine → experiments/results.csv
proxy:
	$(PY) -m bench.run --config configs/bench/proxy.yaml --stage all

all: collect clean-v1 val proxy
