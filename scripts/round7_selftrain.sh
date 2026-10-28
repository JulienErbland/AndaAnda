#!/bin/bash
# Round 7 (F4): self-training on the real match, two variants, R member only (B = B_g).
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round7.done
[ -f models_exp/R_sc/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_sc uv run python -m possession.learned \
  train --features restart --draws 2 --dead-mix --self-train 2 --self-train-mode control > reports/compare/R_sc_train.log 2>&1
[ -f models_exp/R_ss/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_ss uv run python -m possession.learned \
  train --features restart --draws 2 --dead-mix --self-train 2 --self-train-mode conservative > reports/compare/R_ss_train.log 2>&1
for v in sc ss; do
  [ -f reports/compare/ens_R${v}Bg_g.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/ens_R${v}Bg \
    uv run python scripts/compare_models.py eval --switches 10 20 30 --out reports/compare/ens_R${v}Bg_g.json \
    > reports/compare/ens_R${v}Bg_g.log 2>&1
  uv run python scripts/pff_eval.py sweep R_$v B_g 0.75 10 14 20 30 > reports/compare/pff_sweep_$v.log 2>&1
done
uv run python scripts/real_rates.py models_exp/ens_RscBg models_exp/ens_RssBg --switches 10 20 30 > reports/compare/real_rates_s.log 2>&1
touch reports/compare/round7.done
