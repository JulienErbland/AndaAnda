#!/bin/bash
# Round 3b: low switch penalties for the ensembles, whose real-match rate stays under 4.25 at 5-20.
set -e
cd "$(dirname "$0")/.."
for m in models_exp/ens_AB models_exp/ens_AB3; do
  tag=$(basename $m)_lo
  [ -f reports/compare/$tag.json ] && continue
  MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=$m uv run python scripts/compare_models.py eval \
    --switches 1 2 3 4 5 --out reports/compare/$tag.json > reports/compare/$tag.log 2>&1
done
echo done
