"""Touch-detector event rates per graded minute: real hackathon tracking vs degraded DFL.

If the real tracking shows fewer controls / control-team changes than the degraded bench, models
trained on the bench see more evidence of hand-overs than they will on the real match.
Aggregates only (counts per minute).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import FPS, Match, graded_mask  # noqa: E402
from possession import touches  # noqa: E402


def rates(m):
    g = graded_mask(m)
    minutes = g.sum() / FPS / 60
    ts = [t for t in touches.detect(m) if g[t.frame]]
    ctrl = [t for t in ts if t.kind == "control"]
    changes = sum(1 for a, b in zip(ctrl, ctrl[1:]) if a.team != b.team)
    return {"ball seen": float((~np.isnan(m.ball[g, 0])).mean()),
            "controls/min": len(ctrl) / minutes, "control-team changes/min": changes / minutes,
            "deflections/min": sum(t.kind == "deflection" for t in ts) / minutes,
            "contacts/min": sum(t.kind == "contact" for t in ts) / minutes}


def main():
    rows = {}
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        for name, m in (("DFL clean", c), ("DFL degraded", degrade(c, "hackathon", seed=sum(map(ord, c.name))))):
            for k, v in rates(m.without_labels()).items():
                rows.setdefault(name, {}).setdefault(k, []).append(v)
    hp = match_paths(["hackathon"])
    if hp:
        for k, v in rates(Match.load(hp[0]).without_labels()).items():
            rows.setdefault("hackathon (real)", {})[k] = [v]
    keys = list(next(iter(rows.values())))
    print("{:<18}".format("") + "".join("{:>26}".format(k) for k in keys))
    for name, r in rows.items():
        print("{:<18}".format(name) + "".join("{:>26.3f}".format(np.mean(r[k])) for k in keys))


if __name__ == "__main__":
    main()
