"""Does the degraded bench look like the real tracking on things the degrader does not model yet?

    uv run python scripts/degrader_realism.py

Aggregates only, real hackathon tracking vs degraded DFL (current profile):
1. ball jumps: frames where the ball moves more than JUMP_M in one frame (> 50 m/s: not a
   kick, a false detection or a switch to another object), per graded minute;
2. where ball gaps start: ball speed and players within 3 m on the last frame before the gap;
3. implied ball speed across gaps: distance from last to next sighting / gap duration, by gap
   length (live passes vs stoppages);
4. where player tracks end: distance from the track's last position to the nearest opponent
   (identity switches happen in duels).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import FPS, Match, graded_mask  # noqa: E402

JUMP_M = 2.0


def measure(m):
    pitch = np.asarray(m.pitch, dtype=np.float64)
    g = graded_mask(m)
    xy = m.ball[:, :2].astype(np.float64) * pitch
    seen = ~np.isnan(xy[:, 0])
    pl = m.players[..., :2].astype(np.float64) * pitch
    step = np.full(len(xy), np.nan)
    step[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    out = {"jumps/min": float(np.nansum(step[g] > JUMP_M) / (g.sum() / FPS / 60))}
    # gap starts / ends
    st = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    en = np.flatnonzero(~seen[:-1] & seen[1:]) + 1
    st = st[g[st]]
    speed, crowd, implied = [], [], {"<2 s": [], "2-10 s": [], ">10 s": []}
    for s in st:
        k = s - 1
        if k >= 2 and seen[k - 2]:
            speed.append(np.linalg.norm(xy[k] - xy[k - 2]) * FPS / 2)
        d = np.linalg.norm(pl[k] - xy[k], axis=1)
        crowd.append(int(np.nansum(d < 3.0)))
        j = np.searchsorted(en, s)
        if j < len(en) and g[en[j]]:
            dur = (en[j] - s) / FPS
            v = np.linalg.norm(xy[en[j]] - xy[k]) / max(dur, 1 / FPS)
            implied["<2 s" if dur < 2 else "2-10 s" if dur < 10 else ">10 s"].append(v)
    out["gap start: ball speed p50"] = float(np.median(speed))
    out["gap start: ball speed p90"] = float(np.percentile(speed, 90))
    out["gap start: players within 3 m (mean)"] = float(np.mean(crowd))
    for k, v in implied.items():
        out["implied speed {} (median m/s)".format(k)] = float(np.median(v)) if v else np.nan
    # track ends: nearest opponent distance
    tracked = ~np.isnan(pl[..., 0])
    ends = []
    for s in range(pl.shape[1]):
        e = np.flatnonzero(tracked[:-1, s] & ~tracked[1:, s])
        for t in e[g[e]]:
            opp = m.slot_team != m.slot_team[s]
            d = np.linalg.norm(pl[t, opp] - pl[t, s], axis=1)
            if np.isfinite(d).any():
                ends.append(np.nanmin(d))
    ends = np.array(ends)
    out["track ends/min"] = float(len(ends) / (g.sum() / FPS / 60))
    out["track end: opponent within 2 m"] = float(np.mean(ends < 2.0))
    return out


def main():
    rows = {"real (hackathon)": measure(Match.load(match_paths(["hackathon"])[0]).without_labels())}
    per = [measure(degrade(Match.load(p), "hackathon", seed=sum(map(ord, Match.load(p).name))))
           for p in match_paths(["dfl"])]
    keys = list(rows["real (hackathon)"])
    print("{:<40} {:>18} {:>30}".format("", "real (hackathon)", "degraded DFL mean [range]"))
    for k in keys:
        v = [r[k] for r in per]
        print("{:<40} {:>18.3f} {:>16.3f} [{:.3f}-{:.3f}]".format(
            k, rows["real (hackathon)"][k], np.nanmean(v), np.nanmin(v), np.nanmax(v)))


if __name__ == "__main__":
    main()
