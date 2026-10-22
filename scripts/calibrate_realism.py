"""Calibrate the degrader's realism knobs on the real tracking (aggregates only, no labels).

    uv run python scripts/calibrate_realism.py

Each knob against the statistic it controls (scripts/degrader_realism.py measures them):
  break_duel_gain                        -> share of track ends with an opponent within 2 m
  ball_false_per_min                     -> one-frame ball jumps > 2 m per minute
  crowd_gain, fast_ball_gain, fast_ball_ms -> ball speed (p50, p90) and crowding at gap starts
Grid on 3 DFL matches, then the chosen profile checked on all 7. Prints the values to put in
bench/degrade.py Profile; changes nothing itself.
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import PROFILES, degrade  # noqa: E402
from bench.match import Match  # noqa: E402
from scripts.degrader_realism import measure  # noqa: E402


def mean_stats(matches, prof):
    per = [measure(degrade(c, prof, seed=sum(map(ord, c.name)))) for c in matches]
    return {k: float(np.nanmean([r[k] for r in per])) for k in per[0]}


def logdist(a, b):
    return abs(np.log(max(a, 1e-6) / max(b, 1e-6)))


def stage(name, few, base, grid, keys, real):
    """Try each {knob: value} in grid on top of base; return the closest one on `keys`."""
    print(name, "(real: {})".format(", ".join("{} {:.2f}".format(k, real[k]) for k in keys)))
    best = None
    for knobs in grid:
        st = mean_stats(few, replace(base, **knobs))
        d = float(np.mean([logdist(st[k], real[k]) for k in keys]))
        print("   {:<58} {}  dist {:.3f}".format(str(knobs), "  ".join(
            "{:.2f}".format(st[k]) for k in keys), d), flush=True)
        if best is None or d < best[0]:
            best = (d, knobs)
    print("   ->", best[1])
    return best[1]


def main():
    real = measure(Match.load(match_paths(["hackathon"])[0]).without_labels())
    all_m = [Match.load(p) for p in match_paths(["dfl"])]
    few = all_m[:3]
    base = PROFILES["hackathon"]
    chosen = {}
    chosen.update(stage("A. gaps start as the ball leaves play", few, base,
                        [{"gap_from_ballout": False}] +
                        [{"gap_from_ballout": True, "gap_ballout_lead_s": l} for l in (0.5, 1.0, 2.0)],
                        ("gap start: ball speed p50", "gap start: ball speed p90",
                         "gap start: players within 3 m (mean)"), real))
    base = replace(base, **chosen)
    chosen.update(stage("B. false ball blips", few, base,
                        [{"ball_false_per_min": r, "ball_false_in_gaps": g}
                         for r in (0.5, 1.0, 1.5) for g in (0.0, 0.5, 0.9)],
                        ("jumps/min", "implied speed <2 s (median m/s)",
                         "implied speed 2-10 s (median m/s)"), real))
    base = replace(base, **chosen)
    chosen.update(stage("C. identity breaks in duels", few, base,
                        [{"break_duel_gain": g} for g in (10.0, 20.0, 40.0, 80.0)],
                        ("track end: opponent within 2 m", "track ends/min"), real))
    print("\nchosen:", chosen)
    final = mean_stats(all_m, replace(PROFILES["hackathon"], **chosen))
    print("\ncheck on all 7 DFL matches:")
    print("{:<40} {:>10} {:>10}".format("", "real", "bench"))
    for k in real:
        print("{:<40} {:>10.3f} {:>10.3f}".format(k, real[k], final[k]))


if __name__ == "__main__":
    main()
