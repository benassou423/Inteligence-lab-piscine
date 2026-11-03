"""Banc d'essai : fine couche au-dessus du kit officiel (`ftb/`, jamais modifié).

    uv run python -m bench.run --config configs/bench/<run>.yaml   mix → tokenizer → ftb.prepare → ftb.train → ftb.evaluate
    uv run python -m bench.mix --config configs/bench/<run>.yaml   seulement le dataset ordonné (data/mixes/<run>/)
    uv run python -m bench.throughput                              débit de ftb.train et budget proxy conseillé
    uv run python -m bench.compare runs/A runs/B                   écart de BPB et intervalle de confiance
    uv run python -m bench.kaggle push|run|pull                    run complet sur Kaggle (T4), avec reprise

Modules : mix (sorties du pipeline → train.jsonl.gz ordonné), run (enchaîne le kit, results.csv),
config (valeurs par défaut des configs), throughput (débit), compare (deux runs, bootstrap),
kaggle (run complet sur Kaggle), common (outils).
"""
