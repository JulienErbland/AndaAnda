"""Snap M2's hand-overs to the frame where the new team takes control.

The path decoder (path.py) lost to M2 by creating changes of its own. This keeps M2's decisions
and only moves each direct H <-> A change onto the start of the nearest control episode by the
new team (touches.py), within WINDOW_S. Restarts after dead balls are left alone, and so are
changes with no detected control nearby (typically in ball gaps). A change never moves past
its neighbouring changes. Ceiling: perfect timing of every change is worth about +0.03 spellF1
(benchmark oracle), so this is a small, cheap refinement.

    refine(match, labels) -> labels
"""

import numpy as np

from bench.match import FPS, A, H

from . import touches

WINDOW_S = 2.0


def refine(match, labels, window_s=WINDOW_S):
    out = labels.copy()
    starts = {H: [], A: []}
    for t in touches.detect(match):
        if t.kind == "control":
            starts[t.team].append(t.frame)
    starts = {k: np.asarray(sorted(v), dtype=np.int64) for k, v in starts.items()}
    change = np.flatnonzero(np.diff(labels.astype(np.int64))) + 1
    w = int(round(window_s * FPS))
    for i, c in enumerate(change):
        old, new = labels[c - 1], labels[c]
        if not ({old, new} == {H, A}):
            continue                                   # restarts and stoppages stay
        cand = starts[new]
        if not len(cand):
            continue
        j = np.searchsorted(cand, c)
        near = [cand[k] for k in (j - 1, j) if 0 <= k < len(cand) and abs(cand[k] - c) <= w]
        if not near:
            continue
        f = min(near, key=lambda x: abs(x - c))
        lo = change[i - 1] + 1 if i > 0 else 1         # keep at least one frame of each spell
        hi = change[i + 1] - 1 if i + 1 < len(change) else len(labels) - 1
        f = int(np.clip(f, lo, hi))
        if f < c:
            out[f:c] = new
        elif f > c:
            out[c:f] = old
    return out
