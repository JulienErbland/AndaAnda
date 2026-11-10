"""Definition check: does the ground truth go live at the restart kick or at the first reception?

    uv run python scripts/restart_convention.py

For each stoppage that ends inside the public window (4), prints aggregates around the GT's
first live frame s, nothing else: when the ball comes back into view, when it first moves fast
(the kick / throw), and how far the nearest player is at s (a reception would put one on the
ball). A label-meaning check (CHALLENGE.md: the public GT is the reference for what labels
mean); no constant is fitted from it.

Nov 10: the ball reappears moving 10-12 frames before s on the two restarts where it was hidden,
and the nearest player at s is 2.1-4.1 m away on all four -> live shortly after the kick, not at
the first reception. The kick convention stays.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.match import FPS, D, X, Match  # noqa: E402

hm = Match.load(match_paths(["hackathon"])[0])
g = hm.labels
live = (g != D) & (g != X)
gt_starts = np.flatnonzero(live[1:] & (g[:-1] == D)) + 1
# Model-free (the restart detector finds only a third of the real restarts: the ball is mostly
# untracked in dead time): around each GT live start s, when does the ball reappear, when does it
# first move fast, and how far is the nearest player at s.
pitch = np.asarray(hm.pitch, dtype=np.float64)
ball = hm.ball[:, :2].astype(np.float64) * pitch
seen = ~np.isnan(ball[:, 0])
speed = np.full(len(ball), np.nan)
speed[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * FPS / 2
dist = np.linalg.norm(hm.players[..., :2].astype(np.float64) * pitch - ball[:, None], axis=-1)
dmin = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
print("{} stoppages end in the public window".format(len(gt_starts)))
for i, s in enumerate(gt_starts):
    w = np.arange(s - 5 * FPS, s + 2 * FPS)
    back = np.flatnonzero(seen[w] & ~seen[w - 1])            # the ball comes back into view
    fast = np.flatnonzero(np.nan_to_num(speed[w]) >= 4.0)    # the ball is moving (kick / throw)
    print("  restart {}: ball seen at s: {}; reappears at {} frames; first >= 4 m/s at {} frames; "
          "seen share in the 5 s before s: {:.0%}; nearest player at s: {:.1f} m".format(
              i + 1, bool(seen[s]), [int(x) for x in w[back] - s], int(w[fast[0]] - s) if len(fast) else None,
              seen[s - 5 * FPS:s].mean(), dmin[s]))
