"""Dead-ball behaviour, real vs bench (aggregates only): on dead frames with a visible ball, the
ball's speed (from positions), share standing still, distance to the nearest player, share
outside the lines. Calibrates bench/degrade.py's dead-ball knobs.

Real dead frames come from ALGORITHMIC calls over the whole match, never from the public ground
truth (RULES.md: the public 5 minutes are not a tuning set): the submission model's confident
dead frames, and a model-free one (players at walking pace for >= 5 s). They agree; the public
window's 4 stoppages did not (Oct 30).

    uv run python scripts/dead_ball_realism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.epts import derive_speed  # noqa: E402
from bench.match import D, X, Match, graded_mask  # noqa: E402


def stats(m, mask):
    sp = derive_speed(m.ball[:, :2], m.pitch).astype(float)
    P = m.players[..., :2] * np.asarray(m.pitch)
    B = m.ball[:, :2] * np.asarray(m.pitch)
    dd = np.linalg.norm(P - B[:, None], axis=-1)
    dd = np.where(np.isnan(dd), np.inf, dd).min(1)
    L, W = m.pitch
    out = (B[:, 0] < 0) | (B[:, 0] > L) | (B[:, 1] < 0) | (B[:, 1] > W)
    v = mask & ~np.isnan(sp)
    return np.r_[np.percentile(sp[v], [10, 50, 90]), np.mean(sp[v] < 0.3), np.median(dd[v]), np.mean(out[v])]


def real_targets():
    from bench.match import FPS
    from possession import learned
    hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
    g = graded_mask(hm)
    pr = learned.probabilities(hm)
    sp = hm.players[..., 2].astype(float)
    ms = np.nansum(sp, 1) / np.maximum((~np.isnan(sp)).sum(1), 1)
    k = 5 * FPS
    walk = np.convolve(ms < 1.0, np.ones(k), "same") >= k
    return {"model-confident dead": stats(hm, g & (pr[:, 2] > 0.9)),
            "walking pace >= 5 s": stats(hm, g & walk)}


def bench(profile="hackathon"):
    rows = []
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        rows.append(stats(load_degraded(p, profile, 0), graded_mask(c) & (c.labels == D)))
    return np.mean(rows, axis=0)


FMT = "{:<22} speed p10/50/90 {:5.2f} {:5.2f} {:5.2f} | still {:4.0%} | nearest player {:4.1f} m | outside {:4.0%}"


def main():
    for name, v in real_targets().items():
        print(FMT.format("real: " + name, *v))
    print(FMT.format("bench (mean)", *bench()))


if __name__ == "__main__":
    main()
