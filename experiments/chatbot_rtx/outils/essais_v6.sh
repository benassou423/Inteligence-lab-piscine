#!/usr/bin/env bash
# Essais « v6 » (règle du 15/11 16:00) : mesures des témoins S03 et A0 sur le jeu « neuf2 », puis affinages
# s03, s05, s10, s15 sur la synthèse v4 + v5, base v1 gardée. Après chacun : jeu « neuf2 » avec le décodage adopté,
# BPB (moliere_val, synthèse entraînée, 3 fichiers v5 tenus à l'écart), candidat rangé dans
# archives/20261116_essai_v6_<NOM>, modèle en place A0 remis à sa place et vérifié par son empreinte.
cd "$(dirname "$0")/../../.." || exit 1  # racine du dépôt
PY="env PYTHONUTF8=1 .venv/Scripts/python.exe"
R=runs/chatbot_rtx; A0=e3f5ed741f2726b97925758a92f16685e0aa618153ae17ddc72894e5891d5089
S=data/stages/chatbot_20261114/synthetique_v6/entrainement/moliere.jsonl.zst
E5=data/stages/chatbot_20261114/synthetique_v5/ecart
L=runs/evals_manuelles/campagne/essais_v6.log
mesures() {  # $1 nom, $2 dossier du modèle
  mkdir -p runs/evals_manuelles/$1
  $PY tools/nuit_moliere.py protocole --model $2 --jeu neuf2 --out runs/evals_manuelles/$1/neuf2 --nom $1 --device cuda --regler ancre_mu=1 candidates=32 > runs/evals_manuelles/campagne/protocole_$1.log 2>&1 || echo "ECHEC du protocole $1" >> $L
  $PY tools/nuit_moliere.py bpb --model $2 --fichiers data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst $S $E5/gemma_s390.jsonl.zst $E5/gemma_s391.jsonl.zst $E5/gemma_s392.jsonl.zst --out runs/evals_manuelles/$1/bpb_campagne.json --device cuda 2>&1 | grep BPB >> $L
}
echo "== $(date '+%d/%m %H:%M:%S') témoins" >> $L
[ "$(sha256sum $R/modeles/chatbot_moliere/ckpt.pt | cut -c1-64)" = "$A0" ] || { echo "ARRET : le modèle en place n'est pas A0" >> $L; exit 1; }
[ -f $S ] || { echo "ARRET : $S absent" >> $L; exit 1; }
cp $R/bilan.json $R/bilan_A0_avant_v6.json; cp $R/etat_lanceur.json $R/etat_lanceur_A0_avant_v6.json
[ -f runs/evals_manuelles/v6_A0/bpb_campagne.json ] || mesures v6_A0 $R/modeles/chatbot_moliere
[ -f runs/evals_manuelles/v6_S03t/bpb_campagne.json ] || mesures v6_S03t $R/archives/20261115_essai_s03/modeles/chatbot_moliere
for N in ${@:-s03 s05 s10 s15}; do
  echo "== $(date +%H:%M:%S) $N" >> $L
  [ -f chatbot/recettes_rtx_v6_$N.json ] || { echo "recette absente pour $N" >> $L; continue; }
  $PY -m chatbot.pipeline_rtx --recettes chatbot/recettes_rtx_v6_$N.json lancer --taille M --seulement moliere > $R/lanceur_v6_$N.log 2>&1 || { echo "ECHEC du lancement $N (code $?)" >> $L; }
  grep -E "terminé|données|moliere :" $R/lanceur_v6_$N.log | cut -c1-200 >> $L
  arch=$(ls $R/archives | grep -v essai | sort | tail -1)
  mesures v6_$N $R/modeles/chatbot_moliere
  E=$R/archives/20261116_essai_v6_$N; mkdir -p $E/runs $E/modeles $E/evaluations
  mv $R/runs/moliere_etape1 $E/runs/ && mv $R/modeles/chatbot_moliere $E/modeles/ && mv $R/evaluations/moliere $E/evaluations/
  cp $R/bilan.json $R/bilan_v6_$N.json; cp $R/etat_lanceur.json $R/etat_lanceur_v6_$N.json
  mv $R/archives/$arch/runs/moliere_etape1 $R/runs/ && mv $R/archives/$arch/modeles/chatbot_moliere $R/modeles/ && mv $R/archives/$arch/evaluations/moliere $R/evaluations/
  cp $R/bilan_A0_avant_v6.json $R/bilan.json; cp $R/etat_lanceur_A0_avant_v6.json $R/etat_lanceur.json
  [ "$(sha256sum $R/modeles/chatbot_moliere/ckpt.pt | cut -c1-64)" = "$A0" ] && echo "   A0 remis en place (archive $arch), empreinte conforme" >> $L || { echo "ARRET : A0 non retrouvé après $N" >> $L; exit 1; }
done
echo "== $(date '+%d/%m %H:%M:%S') FIN" >> $L
