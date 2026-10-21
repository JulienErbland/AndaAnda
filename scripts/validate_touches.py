"""Check possession/touches.py against DFL official events and labels (clean and degraded).

    uv run python scripts/validate_touches.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import dfl_events  # noqa: E402
from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import FPS, A, H, Match  # noqa: E402
from possession import touches  # noqa: E402


def check(match, events):
    ctrl_team, _, _ = touches._frame_state(match)
    ts = touches.detect(match)
    ctrl = [t for t in ts if t.kind == "control"]
    starts = np.asarray([t.frame for t in ctrl])
    team_of = {H: 0, A: 1}
    res = {}
    passes = [e for e in events if e["kind"] == "Play" and e["snapped"] and e["team"] is not None]
    ok = [np.any(ctrl_team[max(0, e["row"] - 8):e["row"] + 1] == e["team"]) for e in passes]
    res["passer controlling before pass"] = (np.mean(ok), len(ok))
    wins = [e for e in events if e["snapped"] and e["team"] is not None and (
        e["kind"] == "BallClaiming" or (e["kind"] == "TacklingGame" and e.get("possession_change")))]
    ok = []
    for e in wins:
        near = [t for t in ctrl if abs(t.frame - e["row"]) <= FPS // 2]
        ok.append(any(team_of[t.team] == e["team"] for t in near))
    res["interception/tackle -> control by winner (+-0.5 s)"] = (np.mean(ok) if ok else np.nan, len(ok))
    lab = match.labels
    agree = [lab[min(len(lab) - 1, t.frame + 5)] == t.team for t in ctrl]
    res["control team == possession label (+0.2 s)"] = (np.mean(agree), len(agree))
    live_min = np.isin(lab, (H, A)).sum() / FPS / 60
    res["controls per live minute"] = (len(ctrl) / live_min, None)
    res["deflections per live minute"] = (sum(t.kind == "deflection" for t in ts) / live_min, None)
    return res


def main():
    for profile in ("clean", "hackathon"):
        agg = {}
        for path in match_paths(["dfl"]):
            clean = Match.load(path)
            m = clean if profile == "clean" else degrade(clean, profile, seed=sum(map(ord, clean.name)))
            for k, (v, n) in check(m, dfl_events.load(clean.name)).items():
                agg.setdefault(k, []).append((v, n or 0))
        print(profile)
        for k, vals in agg.items():
            v = np.array([x for x, _ in vals])
            n = sum(c for _, c in vals)
            print("  {:<52} {:>6.2f}{}".format(k, np.nanmean(v), "   (n={})".format(n) if n else ""))


if __name__ == "__main__":
    main()
