#!/bin/bash
# Iteration 2: temporal CNN evidence (possession/tcn.py). Plan: tasks/todo.md (written before).
set -e
cd "$(dirname "$0")/.."
L=reports/compare
R="uv run"
[ -f models_exp/T_R/lomo/dfl-DFL-MAT-J03WR9.pt ] || MTAH_MODEL_DIR=models_exp/T_R $R python -m possession.tcn > $L/iter2_T_R_train.log 2>&1
[ -f models_exp/T_B1/lomo/dfl-DFL-MAT-J03WR9.pt ] || MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models_exp/T_B1 $R python -m possession.tcn > $L/iter2_T_B1_train.log 2>&1
SW="20 30 40 60"
ev() { [ -f $L/$1.json ] || MTAH_DFL_LABELS=$3 MTAH_MODEL_DIR=$2 $R python scripts/compare_models.py eval \
    --switches $SW --out $L/$1.json > $L/$1.log 2>&1; }
ev iter2_T_v2 models_exp/ens_T hackathon
ev iter2_T_v1 models_exp/ens_T hackathon-v1
ev iter2_TL_v2 models_exp/ens_TL hackathon
$R python scripts/real_rates.py models_exp/ens_T models_exp/ens_TL --switches $SW > $L/iter2_real.log 2>&1
$R python scripts/pff_eval.py sweep T_R T_B1 0.5 $SW > $L/iter2_pff.log 2>&1
touch $L/iter2.done
