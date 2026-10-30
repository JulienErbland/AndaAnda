#!/bin/bash
# Learning curve of the iteration-0 baseline (R_f + B1_f, 50/50, switch 30): both members trained
# on k DFL matches (k = 6 is the baseline's own held-out models; 7 is not possible held-out).
# Bench (Y_v2) and PFF (reported only) at switch 30.
set -e
cd "$(dirname "$0")/.."
L=reports/compare
for k in 1 2 3 5; do
  for m in "R restart" "B1 touch"; do
    set -- $m; d=models_exp/lc_${1}_k$k; lab=""; [ $1 = B1 ] && lab=hackathon-v1
    [ -f $d/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=$lab MTAH_MODEL_DIR=$d uv run python -m \
      possession.learned train --features $2 --draws 2 --dead-mix --train-k $k > $L/lc_${1}_k$k.log 2>&1
  done
  mkdir -p models_exp/ens_lc_k$k
  echo "{\"members\": [[\"models_exp/lc_R_k$k\", 0.5], [\"models_exp/lc_B1_k$k\", 0.5]]}" > models_exp/ens_lc_k$k/ensemble.json
  [ -f $L/lc_k$k.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/ens_lc_k$k uv run python \
    scripts/compare_models.py eval --switches 30 --out $L/lc_k$k.json > $L/lc_k$k.eval.log 2>&1
  uv run python scripts/pff_eval.py sweep lc_R_k$k lc_B1_k$k 0.5 30 > $L/pff_lc_k$k.log 2>&1
done
touch $L/learning_curve.done
