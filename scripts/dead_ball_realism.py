"""Dead-ball behaviour, real vs bench (aggregates only): on dead frames with a visible ball, the
ball's speed (from positions), share standing still, distance to the nearest player, share
outside the lines. Real: the public window's dead frames (the only labelled ones). Calibrates
bench/degrade.py's dead-ball knobs.

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


def main():
    fmt = "{:<14} speed p10/50/90 {:5.2f} {:5.2f} {:5.2f} | still {:4.0%} | nearest player {:4.1f} m | outside {:4.0%}"
    hm = Match.load(match_paths(["hackathon"])[0])
    print(fmt.format("real (public)", *stats(hm, (hm.labels != X) & (hm.labels == D))))
    rows = []
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        rows.append(stats(load_degraded(p, "hackathon", 0), graded_mask(c) & (c.labels == D)))
    print(fmt.format("bench (mean)", *np.mean(rows, axis=0)))


if __name__ == "__main__":
    main()
