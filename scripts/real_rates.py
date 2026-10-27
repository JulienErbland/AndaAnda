"""Real-match aggregates per model and switch: change rate (public window: 4.25 per live minute),
public-window spellF1 / macroF1 and possessions predicted there (true 17). Aggregates only.

    uv run python scripts/real_rates.py models models_exp/A_r ... [--switches 1 2 3 5 7 10]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.evaluate import pm, score  # noqa: E402
from bench.match import X, Match  # noqa: E402
from possession import learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

PUBLIC_RATE = 4.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--switches", type=float, nargs="+", default=[1, 2, 3, 5, 7, 10, 14])
    a = ap.parse_args()
    hm = Match.load(match_paths(["hackathon"])[0])
    view = hm.without_labels()
    win = hm.labels != X
    g = hm.labels.astype(np.int64)
    n_true = len(pm.possession_spells(g, pm.valid_runs(g)))
    for d in a.models:
        pr = learned.probabilities(view, d)
        rows = []
        for sw in a.switches:
            pred = learned.decode(view, learned.Params(switch=sw), pr)
            r = score(hm.labels, pred)
            p = np.where(win, pred, X).astype(np.int64)
            n = len(pm.possession_spells(p, pm.valid_runs(np.where(win, 0, X).astype(np.int64))))
            rows.append((sw, spell_rate(hm, pred), r["spell_f1_mean"], r["macro_f1"], n))
        best = min(rows, key=lambda x: abs(x[1] - PUBLIC_RATE))
        print("{}: matched switch {:g} (rate {:.2f})".format(d, best[0], best[1]))
        for sw, rate, sp, mac, n in rows:
            print("   switch {:>4g}: rate {:.2f}  public spellF1 {:.3f} macroF1 {:.3f}  "
                  "possessions {}/{}".format(sw, rate, sp, mac, n, n_true))


if __name__ == "__main__":
    main()
