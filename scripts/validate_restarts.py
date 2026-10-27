"""Check possession/restarts.py against DFL official set pieces (clean and degraded tracking).

    uv run python scripts/validate_restarts.py

Per set piece (DFL event, snapped to the labels' dead -> live switch): found (a detection within
1 s), right team, right kind; per detection: precision (a set piece within 1 s), and the share
of the unmatched ones that sit in live play. On the real match (aggregates only): detections per
graded minute by kind, and the kick-off check of each half.
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import dfl_events  # noqa: E402
from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import FPS, A, H, Match, graded_mask  # noqa: E402
from possession import restarts  # noqa: E402

TOL = FPS


def check(match, events):
    rs = restarts.detect(match)
    sp = [e for e in events if "restart" in e and e["snapped"] and e["restart"] != "penalty"]
    fr = np.asarray([r.frame for r in rs])
    found, team_ok, kind_ok = [], [], []
    for e in sp:
        j = np.flatnonzero(np.abs(fr - e["row"]) <= TOL) if len(fr) else []
        found.append(len(j) > 0)
        if len(j):
            r = rs[j[np.argmin(np.abs(fr[j] - e["row"]))]]
            team_ok.append(r.team is not None and (0 if r.team == H else 1) == e["team"])
            kind_ok.append(r.kind == e["restart"])
    ef = np.asarray([e["row"] for e in sp])
    matched = [len(ef) and np.min(np.abs(ef - r.frame)) <= TOL for r in rs]
    live = [match.labels[min(r.frame + 5, match.n_frames - 1)] in (H, A) for r in rs]
    stray_live = [lv for m, lv in zip(matched, live) if not m]

    ko = [e["row"] for e in events if e.get("restart") == "kickoff" and e["snapped"]]
    lu = np.asarray([b for _, b in restarts.lineups(match)])
    ko_err = [np.min(np.abs(lu - k)) / FPS if len(lu) else np.inf for k in ko]
    lu_true = [len(ko) and np.min(np.abs(np.asarray(ko) - f)) <= 3 * FPS for f in lu]
    return dict(found=found, team=team_ok, kind=kind_ok, precision=matched,
                stray_live=stray_live, lineup_found_3s=[e <= 3 for e in ko_err],
                lineup_precision_3s=lu_true,
                by_kind=Counter((e["restart"], f) for e, f in zip(sp, found)),
                team_named=[r.team is not None for r in rs])


def main():
    for profile in ("clean", "hackathon"):
        agg = {}
        kinds = Counter()
        for path in match_paths(["dfl"]):
            clean = Match.load(path)
            m = clean if profile == "clean" else degrade(clean, profile, seed=sum(map(ord, clean.name)))
            res = check(m, dfl_events.load(clean.name))
            kinds += res.pop("by_kind")
            for k, v in res.items():
                agg.setdefault(k, []).extend(v)
        print(profile)
        for k, v in agg.items():
            print("  {:<12} {:6.1%}  (n={})".format(k, np.mean(v) if v else np.nan, len(v)))
        print("  found by kind: " + ", ".join(
            "{} {}/{}".format(k, kinds[(k, True)], kinds[(k, True)] + kinds[(k, False)])
            for k in sorted({k for k, _ in kinds})))
    hp = match_paths(["hackathon"])
    if hp:
        hm = Match.load(hp[0]).without_labels()
        rs = restarts.detect(hm)
        g = graded_mask(hm)
        mins = g.sum() / FPS / 60
        inside = [r for r in rs if g[r.frame]]
        print("real match: {} restarts in graded frames ({:.2f}/min); by kind {}; team unknown {:.0%}"
              .format(len(inside), len(inside) / mins, dict(Counter(r.kind for r in inside)),
                      np.mean([r.team is None for r in inside]) if inside else np.nan))
        for half, first, last in hm.halves:
            ko = [r.frame - first for r in rs if r.kind == "kickoff" and first <= r.frame <= last]
            lu = [f - first for _, f in restarts.lineups(hm) if first <= f <= last]
            print("  half {}: kick-off restarts at {} s, line-up ends at {} s after the graded start"
                  .format(half, [round(k / FPS, 1) for k in ko], [round(k / FPS, 1) for k in lu]))


if __name__ == "__main__":
    main()
