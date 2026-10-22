"""Smart Tagging bonus on the real match: sanity band only (aggregates).

    uv run python scripts/eval_tags.py

The weight is fixed in possession/tags.py (LOGODDS, from the public-window agreement); the
other strengths are shown for sensitivity, not to choose from. No bench score is possible: the
DFL matches have no Smart Tagging.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.evaluate import pm, score  # noqa: E402
from bench.match import FPS, X, Match, graded_mask  # noqa: E402
from possession import learned, tags as T  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402


def main():
    (hp,) = match_paths(["hackathon"])
    hm = Match.load(hp)
    hv = hm.without_labels()
    pr = learned.probabilities(hv)
    tg = T.load()
    g = graded_mask(hm)
    base = learned.decode(hv, probs=pr)
    print("public labels: 4.25 possessions / live min, durations p10/p50/p90 2.2 / 7.6 / 38.1 s")
    for w in (0.0, T.LOGODDS, 2 * T.LOGODDS, 4 * T.LOGODDS):
        out = learned.decode(hv, probs=pr, bonus=T.bonus(hv, tg, w) if w else None)
        full = np.where(g, out, X).astype(np.int64)
        sp = pm.possession_spells(full, pm.valid_runs(np.where(g, 0, X).astype(np.int64)))
        d = np.percentile([(b - a) / FPS for a, b, _ in sp], [10, 50, 90])
        r = score(hm.labels, out)
        print("tags {:<4.1f}{} rate {:.2f}  spells {}  p10/p50/p90 {:.1f} / {:.1f} / {:.1f} s  "
              "frames changed {:.2%}  public spell {:.3f} macro {:.3f}".format(
                  w, " (chosen)" if w == T.LOGODDS else "         ", spell_rate(hm, out), len(sp),
                  *d, float((out[g] != base[g]).mean()), r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()
