"""What does Smart Tagging's Team tag mean, per code? Measured on the public window.

    uv run python scripts/tag_agreement.py

For each code: how many tags overlap the public 5 minutes, and on their live frames, the share
where the ball is with the tagged team (near 100%: Team = possessing team; near 0%: Team = the
other team; near 50%: no team information). Also the share of the tags' frames that are dead,
and for SET PIECES how close each tag start is to a real restart. Aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.match import FPS, A, D, H, X, Match  # noqa: E402
from possession import tags as T  # noqa: E402


def main():
    (hp,) = match_paths(["hackathon"])
    gt = Match.load(hp).labels
    public = gt != X
    tags = T.load()
    rows = {}
    for t in tags:
        r = rows.setdefault(t.code, {"n": 0, "n_pub": 0, "with": 0, "live": 0, "dead": 0, "frames": 0})
        r["n"] += 1
        a, b = max(0, t.start), min(len(gt), t.end)
        sel = np.zeros(len(gt), bool)
        sel[a:b] = True
        sel &= public
        if not sel.any():
            continue
        r["n_pub"] += 1
        lab = gt[sel]
        r["frames"] += len(lab)
        r["dead"] += int((lab == D).sum())
        if t.team is not None:
            live = np.isin(lab, (H, A))
            r["live"] += int(live.sum())
            r["with"] += int((lab[live] == t.team).sum())
    print("{:<32} {:>5} {:>7} {:>16} {:>10}".format("code", "tags", "public", "with tagged team", "dead"))
    for code, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
        w = "{:.0%}".format(r["with"] / r["live"]) if r["live"] else "-"
        d = "{:.0%}".format(r["dead"] / r["frames"]) if r["frames"] else "-"
        print("{:<32} {:>5} {:>7} {:>16} {:>10}".format(code, r["n"], r["n_pub"], w, d))

    # set pieces vs real restarts (dead -> live switches) inside the public window
    g = gt.astype(np.int64)
    sw = [i for i in range(1, len(g)) if g[i - 1] == D and g[i] in (H, A) and public[i]]
    sp = [t for t in tags if t.code == T.SET_PIECES and public[min(max(t.start, 0), len(g) - 1)]]
    if sw and sp:
        d = [min(abs(t.start - s) for s in sw) / FPS for t in sp]
        team_ok = [g[min(sw, key=lambda s: abs(t.start - s))] == t.team for t in sp if t.team is not None]
        print("\nSET PIECES in the public window: {} tags, {} real restarts; median distance of a tag"
              " start to the nearest restart {:.1f} s; team = restarting team {}/{}".format(
                  len(sp), len(sw), float(np.median(d)), sum(team_ok), len(team_ok)))


if __name__ == "__main__":
    main()
