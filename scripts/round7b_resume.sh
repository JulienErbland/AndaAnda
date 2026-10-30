#!/bin/bash
# Resume (Oct 30): round 7 (F4 self-training evals), then the MBR temperature grid (C).
cd "$(dirname "$0")/.."
scripts/round7_selftrain.sh > reports/compare/round7.log 2>&1; echo "exit $?" >> reports/compare/round7.log
MTAH_DFL_LABELS=hackathon uv run python scripts/eval_mbr.py bench --temps 0.5 1 2 4 8 > reports/compare/mbr_bench.log 2>&1
echo "exit $?" >> reports/compare/mbr_bench.log
touch reports/compare/mbr_bench.done
