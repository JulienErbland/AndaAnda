#!/bin/bash
# Round 5b (F0): train on both dead-ball behaviours (moving / parked), 2 draws per match.
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round5b.done
[ -f models_exp/R_v/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_v \
  uv run python -m possession.learned train --features restart --draws 2 --dead-mix > reports/compare/R_v_train.log 2>&1
[ -f models_exp/B_v/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/B_v \
  uv run python -m possession.learned train --features touch --draws 2 --dead-mix > reports/compare/B_v_train.log 2>&1
[ -f reports/compare/ens_RvBv_f0.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/ens_RvBv \
  uv run python scripts/compare_models.py eval --switches 10 14 20 30 --out reports/compare/ens_RvBv_f0.json \
  > reports/compare/ens_RvBv_f0.log 2>&1
uv run python scripts/pff_eval.py sweep R_v B_v 0.75 10 14 20 30 > reports/compare/pff_sweep_v.log 2>&1
uv run python scripts/real_rates.py models_exp/ens_RvBv --switches 10 14 20 30 > reports/compare/real_rates_v.log 2>&1
touch reports/compare/round5b.done
