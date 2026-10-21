"""Held-out check of the ball filling model (possession/ballfill.py) on the open matches.

    uv run python scripts/eval_ballfill.py [--seeds 0]

On gap frames where the clean data has the true ball: distance error of the filled ball against
the straight line between sightings and against "stay where last seen", by time since the
ball was lost; whether the predicted error radius is honest; and, on live frames, whether the
team of the nearest player to the filled ball is the team in possession. Aggregates only.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import A, H, Match  # noqa: E402
from possession import ballfill  # noqa: E402

BUCKETS = [(0, 1), (1, 3), (3, 10), (10, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    err = {k: [] for k in ("model", "line", "last")}
    since, radius, team_hit = [], [], {"model": [], "line": []}
    for p in match_paths(["dfl", "metrica"]):
        clean = Match.load(p)
        m = np.asarray(clean.pitch, dtype=np.float32)
        for seed in args.seeds:
            seen = degrade(clean, "hackathon", seed=seed + sum(map(ord, clean.name))).without_labels()
            frames = ballfill.gap_frames(seen)
            frames = frames[~np.isnan(clean.ball[frames, 0])]
            X_, I = ballfill.features(seen, frames)
            xy, r = ballfill.fill(seen)
            truth = clean.ball[frames, :2] * m
            F = ballfill.FEATURES
            L = I + X_[:, [F.index("lx"), F.index("ly")]]
            L = np.where(np.isnan(L), I, L)
            for k, q in (("model", xy[frames] * m), ("line", I), ("last", L)):
                err[k].append(np.linalg.norm(q - truth, axis=1))
            since.append(X_[:, F.index("t_since")])
            radius.append(r[frames])
            live = np.isin(clean.labels[frames], (H, A))
            P = seen.players[frames][live, :, :2] * m
            for k, q in (("model", xy[frames][live] * m), ("line", I[live])):
                d = np.linalg.norm(P - q[:, None], axis=-1)
                ok = ~np.isnan(d).all(1)
                near = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
                team = seen.slot_team[near]
                team_hit[k].append((team == clean.labels[frames][live])[ok])
        print(clean.name, "done", flush=True)

    e = {k: np.concatenate(v) for k, v in err.items()}
    s, r = np.concatenate(since), np.concatenate(radius)
    print("\nerror (m) on gap frames      median    p75    p90      n")
    for lo, hi in BUCKETS:
        sel = (s >= lo) & (s < hi)
        label = "{}-{} s since lost".format(lo, hi if hi < 1e9 else "inf")
        for k in e:
            q = np.percentile(e[k][sel], [50, 75, 90])
            print("  {:<18} {:<6} {:7.1f} {:6.1f} {:6.1f} {:7d}".format(label, k, *q, sel.sum()))
    for k in e:
        q = np.percentile(e[k], [50, 75, 90])
        print("  {:<18} {:<6} {:7.1f} {:6.1f} {:6.1f} {:7d}".format("all", k, *q, len(s)))
    print("\npredicted 80% radius honest: share of true errors <= radius {:.1%} (target 80%); <= 2x radius {:.1%}"
          .format(np.mean(e["model"] <= r), np.mean(e["model"] <= 2 * r)))
    for k, v in team_hit.items():
        print("live gap frames, nearest player to the {} ball is on the possessing team: {:.1%}"
              .format(k, np.concatenate(v).mean()))


if __name__ == "__main__":
    main()
