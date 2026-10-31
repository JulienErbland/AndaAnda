#!/bin/bash
# Iteration 3: the 4-member mix (CNN + LightGBM) confirmed on fresh degrader seeds 3-5. Plan: tasks/todo.md.
set -e
cd "$(dirname "$0")/.."
L=reports/compare
R="uv run"
ev() { [ -f $L/$1.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=$2 $R python scripts/compare_models.py eval \
    --seeds 3 4 5 --switches 30 40 --out $L/$1.json > $L/$1.log 2>&1; }
ev iter3_base_s345 models
ev iter3_TL_s345 models_exp/ens_TL
$R python scripts/pff_eval.py mix T_R:0.25,T_B1:0.25,models/R:0.25,models/B:0.25 30 40 > $L/iter3_pff.log 2>&1
touch $L/iter3.done
