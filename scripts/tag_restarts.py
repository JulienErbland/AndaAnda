"""Smart Tagging SET PIECES vs the restart detector on the real match. Aggregates only.

    uv run python scripts/tag_restarts.py

Per tag: is a detected restart (or a kick-off line-up) within 3 s of its start, timing offset,
same team (when both name one), same kind. Per detection: share near a tag.
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.match import FPS, Match, graded_mask  # noqa: E402
from possession import restarts, tags  # noqa: E402

KIND = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
        "free kick": "free_kick", "kick off (after goal)": "kickoff", "kick off (start)": "kickoff"}

hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
g = graded_mask(hm)
rs = restarts.detect(hm)
rs += [restarts.Restart(b, None, "kickoff") for _, b in restarts.lineups(hm)]
rs = [r for r in rs if g[r.frame]]
fr = np.asarray(sorted(r.frame for r in rs))
by_frame = {r.frame: r for r in rs}
sp = [t for t in tags.load() if t.code == tags.SET_PIECES and 0 <= t.start < hm.n_frames and g[t.start]]
found, off, team_ok, kind_ok = [], [], [], []
for t in sp:
    j = np.flatnonzero(np.abs(fr - t.start) <= 3 * FPS) if len(fr) else []
    found.append(len(j) > 0)
    if len(j):
        r = by_frame[fr[j[np.argmin(np.abs(fr[j] - t.start))]]]
        off.append((r.frame - t.start) / FPS)
        if r.team is not None and t.team is not None:
            team_ok.append(r.team == t.team)
        if t.kind in KIND:
            kind_ok.append(r.kind == KIND[t.kind])
tf = np.asarray([t.start for t in sp])
near = [len(tf) and np.min(np.abs(tf - f)) <= 3 * FPS for f in fr]
print("SET PIECES tags on graded frames: {} (team N/A {:.0%}); by type {}".format(
    len(sp), np.mean([t.team is None for t in sp]), dict(Counter(t.kind for t in sp))))
print("  with a detected restart within 3 s: {:.0%}; detection - tag start median {:+.1f} s "
      "(quartiles {:+.1f} / {:+.1f})".format(np.mean(found), *np.percentile(off, [50, 25, 75]))
      if off else "  no tag has a detected restart within 3 s")
print("  same team {} / {}; same kind {} / {}".format(sum(team_ok), len(team_ok), sum(kind_ok),
                                                     len(kind_ok)))
print("detections on graded frames: {}; near a SET PIECES tag {:.0%}".format(len(fr), np.mean(near)))
