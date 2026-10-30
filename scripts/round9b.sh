#!/bin/bash
# Round 9b: the two adopted round-9 decisions together (F6 -> B1, L3 -> 50/50) = K1h, scored on
# both v2 and v1 yardsticks; K0h on v1 for the record. PFF and the real rate reported only.
set -e
cd "$(dirname "$0")/.."
L=reports/compare
SW="20 30 40 60"
ev() { [ -f $L/$1.json ] || MTAH_DFL_LABELS=$3 MTAH_MODEL_DIR=$2 uv run python scripts/compare_models.py eval \
    --switches $SW --out $L/$1.json > $L/$1.log 2>&1; }
ev K1h_v2 models_exp/ens_K1h hackathon
ev K1h_v1 models_exp/ens_K1h hackathon-v1
ev K0h_v1 models_exp/ens_K0h hackathon-v1
uv run python scripts/real_rates.py models_exp/ens_K0h models_exp/ens_K1h --switches $SW > $L/real_rates_r9b.log 2>&1
for k in "R_f B_f" "R_f B1_f"; do
  set -- $k; uv run python scripts/pff_eval.py sweep $1 $2 0.5 $SW > $L/pff_sweep_$1_$2_h.log 2>&1
done
touch $L/round9b.done
