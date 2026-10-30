#!/bin/bash
# Round 8, PFF (reported only, decides nothing): submission and reception-trained mixes, switch 20-60.
cd "$(dirname "$0")/.."
until [ -f reports/compare/round8.done ]; do sleep 20; done
uv run python scripts/pff_eval.py sweep R_g B_g 0.75 20 30 40 60 > reports/compare/pff_sweep_g8.log 2>&1
uv run python scripts/pff_eval.py sweep R_rx B_rx 0.75 20 30 40 60 > reports/compare/pff_sweep_rx.log 2>&1
touch reports/compare/round8_pff.done
