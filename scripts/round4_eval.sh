#!/bin/bash
# Round 4: evaluate the new members inside the 75/25 ensemble (bench v2 labels, switches 10-30, chosen on the benches).
set -e
cd "$(dirname "$0")/.."
mk() { mkdir -p models_exp/$1; echo "{\"members\": [[\"$2\", 0.75], [\"$3\", 0.25]]}" > models_exp/$1/ensemble.json; }
mk ens_RBm models_exp/R_r models_exp/B_m
mk ens_RmBm models_exp/R_m models_exp/B_m
for t in ens_RBm ens_RmBm; do
  [ -f reports/compare/${t}_lo.json ] && continue
  MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models_exp/$t uv run python scripts/compare_models.py eval \
    --switches 10 14 20 30 --out reports/compare/${t}_lo.json > reports/compare/${t}_lo.log 2>&1
done
echo evaluated
