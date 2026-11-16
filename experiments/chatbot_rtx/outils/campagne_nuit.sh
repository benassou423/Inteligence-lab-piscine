#!/usr/bin/env bash
# Génération de nuit : un flux par graine, reprise automatique, serveur Ollama relancé s'il ne répond plus.
# Usage : campagne_nuit.sh GRAINE:APPELS [GRAINE:APPELS …]
cd "$(dirname "$0")/../../.." || exit 1  # racine du dépôt
O="${OLLAMA:-ollama}"  # programme Ollama
D=${CAMP_D:-data/stages/chatbot_20261114/synthetique_v4/campagne}
mkdir -p $D
C=${CAMP_C:-chatbot/prompts_synthetiques_v4.json}
serveur() {
  curl -s -m 10 http://127.0.0.1:11434/api/version > /dev/null 2>&1 && return 0
  echo "$(date +%H:%M:%S) serveur absent : relance" >> $D/nuit.log
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  (LLAMA_ARG_CACHE_RAM=0 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=3 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=60m nohup "$O" serve >> runs/ollama_serve_11434.log 2>&1 &)
  sleep 10
}
flux() {
  local S=${1%%:*} N=${1##*:}
  for essai in $(seq 1 60); do
    serveur
    PYTHONUTF8=1 .venv/Scripts/python.exe tools/campagne_synthese.py generer --consignes $C --n 100000 --seed $S --max-appels $N --port 11434 --out $D/gemma_s$S.jsonl.zst >> $D/gemma_s$S.log 2>&1 && { echo "$(date +%H:%M:%S) graine $S finie (essai $essai)" >> $D/nuit.log; return 0; }
    echo "$(date +%H:%M:%S) graine $S : arrêt, reprise dans 30 s (essai $essai)" >> $D/nuit.log
    sleep 30
  done
}
echo "$(date +%H:%M:%S) début : $*" >> $D/nuit.log
for x in "$@"; do flux "$x" & done
wait
echo "$(date +%H:%M:%S) FIN" >> $D/nuit.log
