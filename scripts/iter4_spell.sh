#!/bin/bash
# Iteration 4: spell-balanced training weights (MTAH_SPELL_WEIGHT=10). Plan: tasks/todo.md (written before).
set -e
cd "$(dirname "$0")/.."
L=reports/compare
R="uv run"
export MTAH_SPELL_WEIGHT=10
(
[ -f models_exp/R_sw/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_sw $R python -m possession.learned train --features restart --draws 2 --dead-mix > $L/iter4_R_train.log 2>&1
[ -f models_exp/B1_sw/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models_exp/B1_sw $R python -m possession.learned train --features touch --draws 2 --dead-mix > $L/iter4_B1_train.log 2>&1
) &
LGB=$!
[ -f models_exp/T_R_sw/lomo/dfl-DFL-MAT-J03WR9.pt ] || MTAH_MODEL_DIR=models_exp/T_R_sw $R python -m possession.tcn --features restart > $L/iter4_T_R_train.log 2>&1
[ -f models_exp/T_B1_sw/lomo/dfl-DFL-MAT-J03WR9.pt ] || MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models_exp/T_B1_sw $R python -m possession.tcn > $L/iter4_T_B1_train.log 2>&1
wait $LGB
unset MTAH_SPELL_WEIGHT
ev() { [ -f $L/$1.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=$2 $R python scripts/compare_models.py eval \
    --switches 30 40 60 80 --out $L/$1.json > $L/$1.log 2>&1; }
ev iter4_SW_v2 models_exp/ens_SW
ev iter4_base_v2 models
touch $L/iter4.done
