#!/bin/bash
# Round 4, fix 3: retrain both ensemble members with Metrica sample games 1-3 added.
set -e
cd "$(dirname "$0")/.."
[ -f models_exp/B_m/lomo/metrica-game3.txt ] || MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/B_m \
  uv run python -m possession.learned train --features touch --with-metrica > reports/compare/B_m_train.log 2>&1
[ -f models_exp/R_m/lomo/metrica-game3.txt ] || MTAH_MODEL_DIR=models_exp/R_m \
  uv run python -m possession.learned train --features restart --with-metrica > reports/compare/R_m_train.log 2>&1
echo trained
