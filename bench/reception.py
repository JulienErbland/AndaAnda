"""A restart stays dead until someone receives the ball, as in the hackathon ground truth.

In the ground truth a throw-in, free kick, corner, goal kick or kick-off is still `D` while the
ball travels; play resumes at the first reception. This is a definition, read from the public
window as CHALLENGE.md says to; no constant here comes from it. DFL's BallStatus turns live at the
kick itself (the ball passes 3 m/s a median 3 frames later, for every kind of restart), and the
first reception comes 1-2 s after that, so a model trained on DFL as is ends every stoppage early.

Rule (`dead_until_reception`). At the start of each live stretch that follows a stoppage, the
taker is the player nearest the ball. `D` is extended until the first frame where another player
is the nearest one within near_m of the ball, on two frames running. No ball at the restart, or no
reception within max_s: the stretch is left as it is.

It is used twice:
- on clean DFL, to build training and scoring labels (1.5 m, 6 s; DFL possession everywhere
  else): `python -m bench reception` writes external/dfl_rx/*.npz, `MTAH_DFL_LABELS=reception`
  reads them; `python -m bench reception --source hackathon` does the same on the
  hackathon-style labels -> external/dfl_hk_rx, `MTAH_DFL_LABELS=hackathon-rx`. Each file keeps DFL's BallStatus as `ball_dead`, so the degrader still hides the ball
  where it is really out of play;
- on the predicted labels (possession/learned.py), with a wider radius for the noisier tracking.
"""

import glob
import os
from dataclasses import replace

import numpy as np

from . import EXTERNAL_DIR
from .match import FPS, A, D, H, Match

OUT_DIR = os.path.join(EXTERNAL_DIR, "dfl_rx")
NEAR_M = 1.5
MAX_S = 6.0


def dead_until_reception(match, labels, near_m=NEAR_M, max_s=MAX_S):
    """(labels with each restart moved to the first reception, extensions in frames)."""
    out = labels.copy()
    m = np.asarray(match.pitch, dtype=np.float64)
    d = np.sqrt(((match.players[..., :2] * m - match.ball[:, None, :2] * m) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    who = d.argmin(1)
    near = d[np.arange(len(who)), who] < near_m
    live = np.isin(labels, (H, A))
    starts = np.flatnonzero(live[1:] & ~live[:-1]) + 1
    ends = np.flatnonzero(live & ~np.concatenate((live[1:], [False]))) + 1
    ext = []
    for a in starts[~np.isnan(match.ball[starts, 0])]:
        b = ends[np.searchsorted(ends, a + 1)]
        t = np.arange(a + 1, min(b, a + int(max_s * FPS)) - 1)
        got = near[t] & near[t + 1] & (who[t] != who[a]) & (who[t + 1] == who[t])
        if got.any():
            r = t[np.argmax(got)]
            out[a:r] = D
            ext.append(r - a)
    return out, np.asarray(ext)


def write(paths, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        m = Match.load(p)
        labels, ext = dead_until_reception(m, m.labels)
        replace(m, labels=labels, ball_dead=m.labels == D).save(
            os.path.join(out_dir, os.path.basename(p)))
        print("  {}: {} restarts extended, median {:.0f} frames (p10 {:.0f}, p90 {:.0f}); dead "
              "share {:.1%} -> {:.1%}".format(m.name, len(ext), *np.percentile(ext, [50, 10, 90]),
                                              np.mean(m.labels == D), np.mean(labels == D)),
              flush=True)


def dfl_paths(source="dfl"):
    """source: dfl (DFL's possession) or hackathon (bench/relabel.py's labels, the bench's)."""
    d = os.path.join(EXTERNAL_DIR, "dfl" if source == "dfl" else "dfl_hk")
    return sorted(glob.glob(os.path.join(d, "*.npz")))
