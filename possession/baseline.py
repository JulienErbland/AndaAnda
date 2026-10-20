"""Method 0: the organisers' baseline (starter/baseline.py), ported to the common format.

Nearest tracked player to the ball gets the ball for their team; no ball means D; then a centred
majority vote over +/- `window` frames, D where the window saw no ball at all, ties to home.
Kept identical in behaviour so the bench can be checked against the published sweep.
"""

import numpy as np

from bench.match import A, D, H

NOBODY = -1


def nearest_team(match):
    metres = np.asarray(match.pitch, dtype=np.float32)
    dist = np.sqrt((((match.players[..., :2] - match.ball[:, None, :2]) * metres) ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    seen = np.isfinite(dist[np.arange(len(dist)), nearest])
    return np.where(seen, match.slot_team[nearest], NOBODY).astype(np.int8)


def smooth(codes, window):
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    return np.where(h + a == 0, D, np.where(h >= a, H, A)).astype(np.int8)


def predict(match, window=50):
    return smooth(nearest_team(match), window)
