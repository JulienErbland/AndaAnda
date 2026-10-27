"""Does the tracking park a dead ball where it left the pitch, then jump it to the restart spot?
Aggregates only: share of >= 1.5 s still-ball stretches outside the lines, and of those followed
by a jump (> 5 m in 0.2 s). DFL clean / degraded vs the real match.

    uv run python scripts/restart_realism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import FPS, Match  # noqa: E402
from possession import restarts  # noqa: E402


def stats(m):
    ball = restarts._ball_m(m)
    L, W = m.pitch
    runs, _ = restarts._still_runs(ball)
    out, jump, n = 0, 0, 0
    for a, b in runs:
        seen = ~np.isnan(ball[a:b, 0])
        if seen.sum() < restarts.STILL_S * FPS or b + 5 >= len(ball):
            continue
        n += 1
        x, y = np.median(ball[a:b][seen], axis=0)
        out += (x < 0) or (x > L) or (y < 0) or (y > W)
        d = np.linalg.norm(ball[b:b + 8] - ball[b - 1], axis=1)
        jump += np.nan_to_num(d).max() > 5.0
    return n, out, jump


rows = []
for p in match_paths(["dfl"]):
    c = Match.load(p)
    rows.append(("clean", stats(c)))
    rows.append(("degraded", stats(degrade(c, "hackathon", seed=sum(map(ord, c.name))))))
for kind in ("clean", "degraded"):
    n, o, j = np.sum([r for k, r in rows if k == kind], axis=0)
    print("DFL {:<9} still stretches {:5d}; outside the lines {:5.1%}; followed by a jump {:5.1%}".format(
        kind, n, o / n, j / n))
hp = match_paths(["hackathon"])
if hp:
    n, o, j = stats(Match.load(hp[0]).without_labels())
    print("real match    still stretches {:5d}; outside the lines {:5.1%}; followed by a jump {:5.1%}".format(n, o / n, j / n))
