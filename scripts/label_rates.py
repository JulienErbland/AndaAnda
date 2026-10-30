"""True change rate (possessions per live minute) on every labelled open match: the range check
for the real match's predicted rate, from other games instead of the public window's 17
possessions. Aggregates per match only; open data (DFL, PFF), never the hackathon match.

    uv run python scripts/label_rates.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import EXTERNAL_DIR  # noqa: E402
from bench.match import Match  # noqa: E402
from bench.relabel import Rule, hackathon_labels  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

SETS = {
    "DFL, DFL's labels": ("dfl", None),
    "DFL, hackathon-style": ("dfl_hk", None),
    "DFL, hackathon-style + reception": ("dfl_hk_rx", None),
    "PFF, PFF's labels": ("pff", None),
    "PFF, hackathon-style rule": ("pff", Rule()),
}


def main():
    for name, (d, rule) in SETS.items():
        rates = []
        for p in sorted(glob.glob(os.path.join(EXTERNAL_DIR, d, "*.npz"))):
            m = Match.load(p)
            lab = m.labels if rule is None else hackathon_labels(m, rule)
            rates.append(spell_rate(m, lab))
        r = np.array(rates)
        print("{:<34} n {:>2}  mean {:.2f}  min {:.2f}  p5 {:.2f}  p95 {:.2f}  max {:.2f}".format(
            name, len(r), r.mean(), r.min(), *np.percentile(r, [5, 95]), r.max()), flush=True)
        if len(r) <= 10:
            print("    per match: " + " ".join("{:.2f}".format(x) for x in r))


if __name__ == "__main__":
    main()
