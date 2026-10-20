"""Tune possession.hmm.Params on the bench, in stages that avoid DFL's label-definition gap.

DFL labels change possession ~2x as often as the hackathon GT, so the switch penalty must not be
fitted to DFL spellF1. Instead:

  1. dead-ball weights   -> maximise DFL alive/dead F1 (what "dead" means transfers well)
  2. team-evidence       -> maximise DFL team accuracy on frames both call alive
  3. switch penalty      -> sweep and REPORT: DFL spellF1, hackathon public-window spellF1, and the
                            predicted possession rate on the full hackathon match (a count, no GT)

    uv run python scripts/tune_hmm.py [--stage 1 2 3]

Hackathon output is aggregate-only. Coordinate search, one parameter at a time, few values: the
evidence terms are few and roughly separable, and a big grid would overfit 7 matches.
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import is_private, load_degraded, match_paths  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import A, D, H, graded_mask  # noqa: E402
from possession import features, hmm  # noqa: E402

STAGE1 = {"walk_speed": [1.3, 1.5, 1.7], "walk_scale": [0.3, 0.5, 0.8],
          "w_out": [3, 6, 12], "w_ball_missing": [0, 0.25, 0.5]}
STAGE2 = {"control_m": [1.0, 1.5, 2.5], "gap_scale_m": [0.25, 0.5, 1.0], "w_team": [1, 2, 4, 8]}
SWITCHES = [10, 20, 30, 40, 60, 80, 120]


def load(selectors):
    out = []
    for path in match_paths(selectors):
        m = load_degraded(path, "hackathon", 0)
        out.append((m, features.compute(m)))
    return out


def run(matches, p):
    return [(m, hmm.decode(m.without_labels(), p, f)) for m, f in matches]


def mean_metric(results, key):
    return float(np.mean([score(m.labels, pred)[key] for m, pred in results]))


def coordinate_search(matches, p, grid, key):
    for _ in range(2):
        for name, values in grid.items():
            best = max(values, key=lambda v: mean_metric(run(matches, replace(p, **{name: v})), key))
            p = replace(p, **{name: best})
            print("  {} = {}  ->  {} {:.4f}".format(
                name, best, key, mean_metric(run(matches, p), key)), flush=True)
    return p


def spell_rate(m, pred):
    """Predicted possessions per minute of predicted live ball, over graded frames."""
    g = graded_mask(m)
    live = g & ((pred == H) | (pred == A))
    edges = np.flatnonzero(np.diff(np.where(g, pred, D).astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    n = sum(1 for s in starts if g[s] and pred[s] in (H, A))
    return n / (live.sum() / 25 / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, nargs="*", default=[1, 2, 3])
    args = ap.parse_args()

    dfl = load(["dfl"])
    p = hmm.Params()
    print("start: DFL spellF1 {:.4f}".format(mean_metric(run(dfl, p), "spell_f1_mean")))
    if 1 in args.stage:
        print("stage 1: dead-ball weights on DFL alive/dead F1")
        p = coordinate_search(dfl, p, STAGE1, "alive_dead_f1")
    if 2 in args.stage:
        print("stage 2: team evidence on DFL team accuracy (both alive)")
        p = coordinate_search(dfl, p, STAGE2, "team_acc_both_alive")
    if 3 in args.stage:
        hack = load(["hackathon"])
        print("stage 3: switch penalty (choose by judgement, see module docstring)")
        print("  {:>6} {:>10} {:>10} {:>12} {:>14}".format(
            "switch", "DFL spell", "DFL macro", "public spell", "full rate/min"))
        for s in SWITCHES:
            q = replace(p, switch=s)
            d = run(dfl, q)
            (hm, hp), = run(hack, q)
            print("  {:>6} {:>10.4f} {:>10.4f} {:>12.4f} {:>14.2f}".format(
                s, mean_metric(d, "spell_f1_mean"), mean_metric(d, "macro_f1"),
                score(hm.labels, hp)["spell_f1_mean"], spell_rate(hm, hp)), flush=True)
    print("params:", p)


if __name__ == "__main__":
    main()
