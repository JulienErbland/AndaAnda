#!/bin/bash
# Round 9 = iteration 0 (clean-up): F5 bench (degrader knobs from the model's own calls), F6 (B on
# v2 vs v1 labels), convention (kick vs reception). Decision rules: tasks/todo.md, written before.
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round9.done
L=reports/compare
train() {  # dir labels features
  [ -f models_exp/$1/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=$2 MTAH_MODEL_DIR=models_exp/$1 \
    uv run python -m possession.learned train --features $3 --draws 2 --dead-mix > $L/$1_train.log 2>&1
}
train R_f "" restart
train B_f hackathon touch
train B1_f hackathon-v1 touch
train R_rxf reception restart
train B_rxf hackathon-rx touch
SW="20 30 40 60"
ev() {  # tag modeldir labels
  [ -f $L/$1.json ] || MTAH_DFL_LABELS=$3 MTAH_MODEL_DIR=$2 uv run python scripts/compare_models.py eval \
    --switches $SW --out $L/$1.json > $L/$1.log 2>&1
}
ev sub_v2 models hackathon
ev K0_v2 models_exp/ens_K0 hackathon
ev K0_v1 models_exp/ens_K0 hackathon-v1
ev K0_rx models_exp/ens_K0 hackathon-rx
ev K1_v2 models_exp/ens_K1 hackathon
ev K1_v1 models_exp/ens_K1 hackathon-v1
ev K2_v2 models_exp/ens_K2 hackathon
ev K2_rx models_exp/ens_K2 hackathon-rx
ev K0h_v2 models_exp/ens_K0h hackathon
uv run python scripts/real_rates.py models_exp/ens_K0 models_exp/ens_K1 models_exp/ens_K2 --switches $SW > $L/real_rates_r9.log 2>&1
for k in "R_f B_f" "R_f B1_f" "R_rxf B_rxf"; do
  set -- $k; uv run python scripts/pff_eval.py sweep $1 $2 0.75 $SW > $L/pff_sweep_$1_$2.log 2>&1
done
touch $L/round9.done
