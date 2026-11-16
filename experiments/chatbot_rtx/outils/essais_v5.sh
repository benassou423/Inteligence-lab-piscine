#!/usr/bin/env bash
# Trois affinages de la campagne de synthèse (S05, S10, S15), base v1 gardée. Après chacun : mesures, candidat rangé
# dans archives/20261115_essai_<NOM>, modèle livré A0 remis à sa place et vérifié par son empreinte.
cd "$(dirname "$0")/../../.." || exit 1  # racine du dépôt
PY="env PYTHONUTF8=1 .venv/Scripts/python.exe"
R=runs/chatbot_rtx; A0=e3f5ed741f2726b97925758a92f16685e0aa618153ae17ddc72894e5891d5089
S=data/stages/chatbot_20261114/synthetique_v4
L=runs/evals_manuelles/campagne/essais_v5.log
mesures() {  # $1 nom, $2 dossier du modèle
  $PY tools/nuit_moliere.py protocole --model $2 --jeu neuf --out runs/evals_manuelles/$1/neuf --nom $1 --device cuda --regler ancre_mu=1 candidates=32 > runs/evals_manuelles/campagne/protocole_$1.log 2>&1
  mkdir -p runs/evals_manuelles/$1
  $PY tools/nuit_moliere.py bpb --model $2 --fichiers data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst $S/entrainement/moliere.jsonl.zst $S/ecart/gemma_s290.jsonl.zst $S/ecart/gemma_s291.jsonl.zst $S/ecart/gemma_s292.jsonl.zst --out runs/evals_manuelles/$1/bpb_campagne.json --device cuda 2>&1 | grep BPB >> $L
}
echo "== $(date +%H:%M:%S) A0" >> $L
[ "$(sha256sum $R/modeles/chatbot_moliere/ckpt.pt | cut -c1-64)" = "$A0" ] || { echo "ARRET : le modèle en place n'est pas A0" >> $L; exit 1; }
cp $R/bilan.json $R/bilan_A0_avant_v5.json; cp $R/etat_lanceur.json $R/etat_lanceur_A0_avant_v5.json
[ -f runs/evals_manuelles/A0n/bpb_campagne.json ] || mesures A0n $R/modeles/chatbot_moliere
for N in "${@:-s05 s10 s15}"; do
  echo "== $(date +%H:%M:%S) $N" >> $L
  avant=$(ls $R/archives | sort | tail -1)
  $PY -m chatbot.pipeline_rtx --recettes chatbot/recettes_rtx_v5_$N.json lancer --taille M --seulement moliere > $R/lanceur_v5_$N.log 2>&1 || { echo "ECHEC du lancement $N (code $?)" >> $L; }
  grep -E "terminé|données|moliere :" $R/lanceur_v5_$N.log | cut -c1-200 >> $L
  arch=$(ls $R/archives | grep -v essai | sort | tail -1)
  mesures $N $R/modeles/chatbot_moliere
  E=$R/archives/20261115_essai_$N; mkdir -p $E/runs $E/modeles $E/evaluations
  mv $R/runs/moliere_etape1 $E/runs/ && mv $R/modeles/chatbot_moliere $E/modeles/ && mv $R/evaluations/moliere $E/evaluations/
  cp $R/bilan.json $R/bilan_v5_$N.json; cp $R/etat_lanceur.json $R/etat_lanceur_v5_$N.json
  mv $R/archives/$arch/runs/moliere_etape1 $R/runs/ && mv $R/archives/$arch/modeles/chatbot_moliere $R/modeles/ && mv $R/archives/$arch/evaluations/moliere $R/evaluations/
  cp $R/bilan_A0_avant_v5.json $R/bilan.json; cp $R/etat_lanceur_A0_avant_v5.json $R/etat_lanceur.json
  [ "$(sha256sum $R/modeles/chatbot_moliere/ckpt.pt | cut -c1-64)" = "$A0" ] && echo "   A0 remis en place (archive $arch), empreinte conforme" >> $L || { echo "ARRET : A0 non retrouvé après $N" >> $L; exit 1; }
done
echo "== $(date +%H:%M:%S) FIN" >> $L
