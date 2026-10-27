#!/bin/bash
# Round 3 (Oct 27): retrain with restart features, evaluate everything on the v2 labels.
set -e
cd "$(dirname "$0")/.."
[ -f models_exp/R_r/lomo/dfl-DFL-MAT-J03WR9.txt ] || MTAH_MODEL_DIR=models_exp/R_r uv run python -m possession.learned train --features restart > reports/compare/R_r_train.log 2>&1
for m in models models_exp/A_r models_exp/ens_AB models_exp/ens_AB3 models_exp/R_r; do
  tag=$(basename $m)_k
  [ -f reports/compare/$tag.json ] && continue   # resumable: finished evals are kept
  MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=$m uv run python scripts/compare_models.py eval --out reports/compare/$tag.json > reports/compare/$tag.log 2>&1
done
echo done
