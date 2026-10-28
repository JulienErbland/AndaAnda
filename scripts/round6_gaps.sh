#!/bin/bash
# Round 6 (F3): ball-gap split recalibrated (live 7% / dead 80%, gaps end at restarts) ->
# retrain both members (dead-ball mix), compare with the submission on the new bench, PFF, real.
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round6.done
[ -f models_exp/R_g/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_g \
  uv run python -m possession.learned train --features restart --draws 2 --dead-mix > reports/compare/R_g_train.log 2>&1
[ -f models_exp/B_g/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/B_g \
  uv run python -m possession.learned train --features touch --draws 2 --dead-mix > reports/compare/B_g_train.log 2>&1
for m in models ens_RgBg; do
  d=models_exp/$m; [ $m = models ] && d=models
  [ -f reports/compare/${m}_g.json ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=$d \
    uv run python scripts/compare_models.py eval --switches 10 14 20 30 --out reports/compare/${m}_g.json \
    > reports/compare/${m}_g.log 2>&1
done
uv run python scripts/pff_eval.py sweep R_g B_g 0.75 10 14 20 30 > reports/compare/pff_sweep_g.log 2>&1
uv run python scripts/real_rates.py models_exp/ens_RgBg --switches 10 14 20 30 > reports/compare/real_rates_g.log 2>&1
touch reports/compare/round6.done
