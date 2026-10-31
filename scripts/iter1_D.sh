#!/bin/bash
# Iteration 1: D (learned switch costs) on the iteration-0 baseline. Plan: tasks/todo.md.
set -e
cd "$(dirname "$0")/.."
L=reports/compare
export MTAH_CHANGE_DIR=models_exp/changes_f
[ -f $MTAH_CHANGE_DIR/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon-v1 uv run python -m possession.changes train > $L/iter1_changes_train.log 2>&1
MTAH_DFL_LABELS=hackathon uv run python scripts/eval_changes.py bench --betas 0 1 2 4 8 --switches 30 40 --out $L/iter1_v2 > $L/iter1_v2.log 2>&1
MTAH_DFL_LABELS=hackathon-v1 uv run python scripts/eval_changes.py bench --betas 0 1 2 4 8 --switches 30 --out $L/iter1_v1 > $L/iter1_v1.log 2>&1
for b in 1 2 4 8; do
  uv run python scripts/eval_changes.py pff --beta $b --switch 30 --out $L/iter1_pff_$b > $L/iter1_pff_$b.log 2>&1
  uv run python scripts/eval_changes.py real --beta $b --switch 30 > $L/iter1_real_$b.log 2>&1
done
touch $L/iter1.done
