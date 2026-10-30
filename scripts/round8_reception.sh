#!/bin/bash
# Round 8 (F8): dead until the first reception (PR #1), scored on hackathon-rx labels.
# submission as is / + decoder rule; R, B retrained on reception labels / + decoder rule.
set -e
cd "$(dirname "$0")/.."
rm -f reports/compare/round8.done
[ -d external/dfl_rx ] || uv run python -m bench reception > reports/compare/rx_labels.log 2>&1
[ -f models_exp/R_rx/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=reception MTAH_MODEL_DIR=models_exp/R_rx \
  uv run python -m possession.learned train --features restart --draws 2 --dead-mix > reports/compare/R_rx_train.log 2>&1
[ -f models_exp/B_rx/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_DFL_LABELS=hackathon-rx MTAH_MODEL_DIR=models_exp/B_rx \
  uv run python -m possession.learned train --features touch --draws 2 --dead-mix > reports/compare/B_rx_train.log 2>&1
SW="20 30 40 60"
for m in models ens_RrxBrx; do
  d=models_exp/$m; [ $m = models ] && d=models
  for r in "" 2.0; do
    tag=${m}_rx${r:+_dec}
    [ -f reports/compare/$tag.json ] || MTAH_DFL_LABELS=hackathon-rx MTAH_MODEL_DIR=$d \
      uv run python scripts/compare_models.py eval --switches $SW ${r:+--reception $r} \
      --out reports/compare/$tag.json > reports/compare/$tag.log 2>&1
  done
done
uv run python scripts/real_rates.py models models_exp/ens_RrxBrx --switches $SW > reports/compare/real_rates_rx.log 2>&1
uv run python scripts/real_rates.py models models_exp/ens_RrxBrx --switches $SW --reception 2.0 > reports/compare/real_rates_rx_dec.log 2>&1
touch reports/compare/round8.done
