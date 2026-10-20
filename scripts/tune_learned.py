"""Sweep the decoder penalties of possession.learned on held-out probabilities.

    uv run python scripts/tune_learned.py

Probabilities are computed once per match (held-out models on DFL, the full model on the
hackathon match). With log-probability evidence only the penalty/weight ratios matter, so weight
stays 1. Same rule as M1: the switch penalty is chosen by the predicted possession rate on the
full hackathon match (~4.25 per live minute in the public window), not by DFL spellF1, whose
labels change possession ~2x more often. Hackathon output is aggregate-only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.evaluate import score  # noqa: E402
from possession import learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

SWITCHES = [3, 5, 7, 10, 20, 30, 40, 60]
STOPS = [10, 20, 40, 80]


def main():
    dfl = [(m, learned.probabilities(m.without_labels()))
           for m in (load_degraded(p, "hackathon", 0) for p in match_paths(["dfl"]))]
    hack = [(m, learned.probabilities(m.without_labels()))
            for m in (load_degraded(p, "hackathon", 0) for p in match_paths(["hackathon"]))]
    print("{:>6} {:>5} {:>10} {:>10} {:>12} {:>10}".format(
        "switch", "stop", "DFL spell", "DFL macro", "public spell", "rate/min"))
    for stop in STOPS:
        for sw in SWITCHES:
            p = learned.Params(switch=sw, stop=stop, restart=stop)
            r = [score(m.labels, learned.decode(m, p, pr)) for m, pr in dfl]
            hm, hp = hack[0]
            hpred = learned.decode(hm, p, hp)
            print("{:>6} {:>5} {:>10.4f} {:>10.4f} {:>12.4f} {:>10.2f}".format(
                sw, stop, np.mean([x["spell_f1_mean"] for x in r]),
                np.mean([x["macro_f1"] for x in r]), score(hm.labels, hpred)["spell_f1_mean"],
                spell_rate(hm, hpred)), flush=True)


if __name__ == "__main__":
    main()
