"""Calibrate the degrader's ball smoothing (Profile.ball_smooth_s) on the real tracking.

    uv run python scripts/calibrate_ball_smoothing.py

Target: the ball's high-frequency motion, which a filtering tracker removes whoever plays:
percentiles of |acceleration| over 1, 2 and 4 frames, on visible-ball stretches of graded
frames. Fitted on the whole real match (unlabelled, aggregates only); no labels and no model
output are involved, so nothing is tuned on the 17 public possessions.

Check, not fit: touch-detector events per minute (controls, deflections, contacts) for the
chosen smoothing, against the real match and the spread across the DFL matches (event counts
also depend on playing style, so they validate rather than calibrate).
"""

import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import PROFILES, degrade  # noqa: E402
from bench.match import FPS, Match, graded_mask  # noqa: E402
from scripts.touch_rates import rates  # noqa: E402

LAGS = (1, 2, 4)
QS = (50, 90, 99)
GRID = (0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.28)


def accel(match):
    """{lag: |acceleration| samples (m/s^2)} on graded frames with the ball seen at t, t +- lag."""
    xy = match.ball[:, :2].astype(np.float64) * np.asarray(match.pitch)
    g = graded_mask(match)
    out = {}
    for k in LAGS:
        a = np.full(len(xy), np.nan)
        a[k:-k] = np.linalg.norm(xy[2 * k:] - 2 * xy[k:-k] + xy[:-2 * k], axis=1) * FPS ** 2 / k ** 2
        out[k] = a[g & ~np.isnan(a)]
    return out


def quantiles(samples):
    return np.array([np.percentile(samples[k], q) for k in LAGS for q in QS])


def main():
    (hp,) = match_paths(["hackathon"])
    real = Match.load(hp).without_labels()
    q_real = quantiles(accel(real))
    clean = [Match.load(p) for p in match_paths(["dfl"])]
    head = "  ".join("a{}:p{}".format(k, q) for k in LAGS for q in QS)
    print("{:<22} {}".format("", head))
    print("{:<22} {}".format("real (hackathon)", "  ".join("{:7.1f}".format(x) for x in q_real)))
    best = None
    for sig in GRID:
        prof = replace(PROFILES["hackathon"], ball_smooth_s=sig)
        pooled = {k: [] for k in LAGS}
        for c in clean:
            a = accel(degrade(c, prof, seed=sum(map(ord, c.name))))
            for k in LAGS:
                pooled[k].append(a[k])
        q = quantiles({k: np.concatenate(v) for k, v in pooled.items()})
        dist = float(np.mean(np.abs(np.log(q / q_real))))
        print("{:<22} {}   dist {:.3f}".format("DFL smooth {:.2f} s".format(sig),
                                              "  ".join("{:7.1f}".format(x) for x in q), dist),
              flush=True)
        if best is None or dist < best[0]:
            best = (dist, sig)
    sig = best[1]
    print("\nclosest ball_smooth_s = {:.2f} s".format(sig))

    print("\ncheck: touch events per graded minute")
    prof = replace(PROFILES["hackathon"], ball_smooth_s=sig)
    for label, p in (("DFL degraded, no smoothing", PROFILES["hackathon"]),
                     ("DFL degraded, smooth {:.2f}".format(sig), prof)):
        per = [rates(degrade(c, p, seed=sum(map(ord, c.name))).without_labels()) for c in clean]
        print("  {:<30} ".format(label) + "  ".join(
            "{} {:.1f} [{:.1f}-{:.1f}]".format(k.split("/")[0], np.mean([r[k] for r in per]),
                                            min(r[k] for r in per), max(r[k] for r in per))
            for k in ("controls/min", "deflections/min", "contacts/min")))
    r = rates(real)
    print("  {:<30} ".format("real (hackathon)") + "  ".join(
        "{} {:.1f}".format(k.split("/")[0], r[k]) for k in ("controls/min", "deflections/min",
                                                            "contacts/min")))


if __name__ == "__main__":
    main()
