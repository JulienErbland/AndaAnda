"""DFL labels with the hackathon's end of a stoppage: the ball stays dead until someone receives it.

In the hackathon ground truth a restart (throw-in, free kick, corner, goal kick, kick-off) is still
`D` while the ball travels: play resumes at the first reception. DFL's BallStatus turns live at the
kick itself (the ball passes 3 m/s a median 3 frames later, for every kind of restart), and the
first reception comes 1-2 s after that. A model trained on DFL as is learns to end every stoppage
too early.

Rule, on clean tracking. At the start of each live stretch, the taker is the player nearest the
ball. `D` is extended until the first frame where another player is the nearest one within NEAR_M
of the ball, on two frames running. Nobody receives within MAX_S (a penalty shot, a ball that goes
straight out): the stretch keeps DFL's start. Possession is DFL's own everywhere else, and nothing
here comes from the hackathon labels.

    python -m bench reception          # writes external/dfl_rx/*.npz (same match names)
    MTAH_DFL_LABELS=reception ...      # every bench script then reads those labels
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


def reception_labels(clean):
    """(labels, extensions in frames, restarts with no reception) for a clean DFL match."""
    out = clean.labels.copy()
    m = np.asarray(clean.pitch, dtype=np.float64)
    d = np.sqrt(((clean.players[..., :2] * m - clean.ball[:, None, :2] * m) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    who = d.argmin(1)
    near = d[np.arange(len(who)), who] < NEAR_M
    live = np.isin(out, (H, A))
    starts = np.flatnonzero(live & ~np.concatenate(([False], live[:-1])))
    ends = np.flatnonzero(live & ~np.concatenate((live[1:], [False]))) + 1
    ext, missed = [], 0
    for a, b in zip(starts, ends):
        t = np.arange(a + 1, min(b, a + int(MAX_S * FPS)) - 1)
        got = near[t] & near[t + 1] & (who[t] != who[a]) & (who[t + 1] == who[t])
        if not got.any():
            missed += 1
            continue
        r = t[np.argmax(got)]
        out[a:r] = D
        ext.append(r - a)
    return out, np.asarray(ext), missed


def write(paths):
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in paths:
        m = Match.load(p)
        labels, ext, missed = reception_labels(m)
        replace(m, labels=labels).save(os.path.join(OUT_DIR, os.path.basename(p)))
        print("  {}: {} restarts extended, median {:.0f} frames (p10 {:.0f}, p90 {:.0f}); {} with "
              "no reception; dead share {:.1%} -> {:.1%}".format(
                  m.name, len(ext), *np.percentile(ext, [50, 10, 90]), missed,
                  np.mean(m.labels == D), np.mean(labels == D)), flush=True)


def dfl_paths():
    return sorted(glob.glob(os.path.join(EXTERNAL_DIR, "dfl", "*.npz")))
