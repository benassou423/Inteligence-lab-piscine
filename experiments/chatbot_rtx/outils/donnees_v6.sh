#!/usr/bin/env bash
# Données de synthèse « v6 » = campagne v4 (déjà fusionnée et refiltrée) + campagne v5b, pour les essais du 16/11.
# 1. fusion puis refiltrage de la v5 avec SES consignes ; 2. union avec la v4 (anti-doublon entre les deux) ;
# 3. retrait des conversations dont un message est un message de test (4 jeux) ; 4. tirage de 120 conversations
# pour le contrôle de qualité par lecteurs. Processeur seul. Refuse de partir si la génération n'est pas finie.
cd "$(dirname "$0")/../../.." || exit 1  # racine du dépôt
PY="env PYTHONUTF8=1 NUIT_SANS_TORCH=1 .venv/Scripts/python.exe"
V4=data/stages/chatbot_20261114/synthetique_v4
V5=data/stages/chatbot_20261114/synthetique_v5
V6=data/stages/chatbot_20261114/synthetique_v6
C5=chatbot/prompts_synthetiques_v5.json
L=runs/evals_manuelles/campagne/donnees_v6.log
grep -q "FIN" $V5/ecart/nuit.log 2>/dev/null || { echo "ARRET : génération pas finie (pas de FIN dans $V5/ecart/nuit.log)"; exit 1; }
[ -e $V6 ] && { echo "ARRET : $V6 existe déjà"; exit 1; }
set -e
{
  echo "== $(date '+%d/%m %H:%M:%S') fusion v5"
  $PY tools/campagne_synthese.py fusion --consignes $C5 --motif "$V5/campagne/gemma_s33*.jsonl.zst" --dst $V5/fusion
  echo "== refiltrage v5"
  $PY tools/campagne_synthese.py refiltrer --consignes $C5 --fichier $V5/fusion/moliere.jsonl.zst
  echo "== union v4 + v5"
  $PY experiments/chatbot_rtx/outils/unir_synthese.py $V4/fusion/moliere.jsonl.zst $V5/fusion/moliere.jsonl.zst $C5 $V6/union/moliere.jsonl.zst
  echo "== garde des messages de test"
  $PY experiments/chatbot_rtx/outils/garde_protocole.py $V6/union/moliere.jsonl.zst $V6/entrainement/moliere.jsonl.zst
  echo "== tirage du contrôle de qualité (120 conversations, graine 7)"
  $PY tools/nuit_moliere.py controle --fichier $V6/entrainement/moliere.jsonl.zst --out runs/evals_manuelles/campagne/final_v6_controle.txt --n 120 --graine 7
  echo "== $(date '+%d/%m %H:%M:%S') FIN"
} 2>&1 | tee -a $L
