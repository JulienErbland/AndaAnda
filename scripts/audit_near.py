"""Bench audit (aggregates only): ball visibility and the ball_vis feature near possession changes,
bench (true and held-out predicted changes) vs the real match (predicted changes; the public
window is never used).

    MTAH_DFL_LABELS=hackathon uv run python scripts/audit_near.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.match import Match, graded_mask  # noqa: E402
from possession import context, learned  # noqa: E402
from scripts.domain_shift import near_changes  # noqa: E402


def stat(m, mask):
    f = learned.frame_features(m, "restart")
    X_, names = context.matrix(f)
    bv = X_[:, names.index("ball_vis")]
    g = graded_mask(m) & mask
    return np.mean(~np.isnan(m.ball[g, 0])), np.nanmean(bv[g]), np.mean(np.isnan(bv[g])), g.sum()


FMT = "{:<34} ball seen {:5.1%} | ball_vis mean {:.3f} nan {:5.1%} | frames {}"
rows = {"true": [], "pred": []}
for p in match_paths(["dfl"]):
    c = Match.load(p)
    v = load_degraded(p, "hackathon", 0).without_labels()
    rows["true"].append(stat(v, near_changes(c.labels)))
    rows["pred"].append(stat(v, near_changes(learned.predict(v))))
for k, r in rows.items():
    print(FMT.format("bench, near %s changes (mean of 7)" % k, *np.mean(r, 0)[:3], int(np.sum(r, 0)[3])))
hm = load_degraded(match_paths(["hackathon"])[0], "hackathon", 0).without_labels()
raw = Match.load(match_paths(["hackathon"])[0]).without_labels()
print("real degraded == raw ball:", np.array_equal(hm.ball, raw.ball, equal_nan=True))
print(FMT.format("real, near predicted changes", *stat(hm, near_changes(learned.predict(hm)))))
