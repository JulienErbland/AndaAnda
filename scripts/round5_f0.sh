#!/bin/bash
# Round 5 (F0): moving dead ball in the degrader + ball speed always derived -> retrain both
# members, then compare with the current ensemble on the new bench, PFF and the real match.
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round5.done
[ -f models_exp/R_d/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_d \
  uv run python -m possession.learned train --features restart > reports/compare/R_d_train.log 2>&1
[ -f models_exp/B_d/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/B_d \
  uv run python -m possession.learned train --features touch > reports/compare/B_d_train.log 2>&1
for m in ens_RB3 ens_RdBd; do      # both on the NEW bench
  [ -f reports/compare/${m}_f0.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/$m \
    uv run python scripts/compare_models.py eval --switches 10 14 20 30 --out reports/compare/${m}_f0.json \
    > reports/compare/${m}_f0.log 2>&1
done
uv run python scripts/pff_eval.py sweep R_d B_d 0.75 10 14 20 30 > reports/compare/pff_sweep_d.log 2>&1
uv run python scripts/real_rates.py models_exp/ens_RB3 models_exp/ens_RdBd --switches 10 14 20 30 \
  > reports/compare/real_rates_f0.log 2>&1
uv run python scripts/domain_shift.py > reports/domain_shift_f0.log 2>&1
touch reports/compare/round5.done
