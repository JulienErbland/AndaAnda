"""Hackathon-style possession labels on DFL: fix the bench's yardstick.

DFL's BallPossession changes about twice as often as the hackathon ground truth (8.4 against
4.3 possessions per live minute), so the bench rewards changing too often. The hackathon GT is
a mechanical rule on corrected tracking ("loose or contested ball: possession stays with the
last team that had it; touches, deflections and aerial duels without control do not change
possession"). DFL gives clean tracking, so the same kind of rule can be applied to it:

- dead frames: DFL's BallStatus, unchanged;
- each live stretch starts with DFL's possession at its first frame (the restarting team);
- afterwards possession changes only when a player of the other team CONTROLS the ball: within
  near_m of it, moving with it (relative speed under rel_ms) for at least control_s;
- anything else (passes in flight, deflections, duels, touches) keeps the last team.

The knobs are calibrated so the relabelled matches show the public GT's possession rate and
duration quantiles (`python -m bench relabel --calibrate`); the public window is 17 possessions,
so the rate and quantiles are targets with error bars, not exact values.

    python -m bench relabel            # writes external/dfl_hk/*.npz (same match names)
    MTAH_DFL_LABELS=hackathon ...      # every bench script then reads those labels
    python -m bench relabel --control-s 0.2 --near-m 1.0 --tag lo   # robustness variant
    MTAH_DFL_LABELS=hackathon-lo ...   # reads external/dfl_hk_lo
"""

import glob
import os
from dataclasses import dataclass, replace

import numpy as np

from . import EXTERNAL_DIR
from .match import FPS, A, H, X, Match, graded_mask

OUT_DIR = os.path.join(EXTERNAL_DIR, "dfl_hk")
# Public GT (evaluation/gt, frames 11185-18684): 17 possessions.
PUBLIC_RATE = 4.25
PUBLIC_Q = {10: 2.2, 25: 3.5, 50: 7.6, 75: 20.0, 90: 38.1}


@dataclass(frozen=True)
class Rule:
    control_s: float = 0.4
    near_m: float = 1.5
    rel_ms: float = 2.0
    # v2: a control still DECIDES the change, but it is DATED from the new team's first touch:
    # walking back from the control (at most lookback_s), after the last frame where the old team
    # was the nearest player within touch_m of the ball. The public labels change near the new
    # team's first touch (+0.12 s median), not at its control start (+0.94 s).
    date_from_touch: bool = True
    touch_m: float = 2.0
    lookback_s: float = 3.0


def _control_team(clean, rule):
    """(T,) team (H/A) controlling the ball under `rule`, -1 elsewhere; clean tracking."""
    m = np.asarray(clean.pitch, dtype=np.float64)
    ball = clean.ball[:, :2].astype(np.float64) * m
    pl = clean.players[..., :2].astype(np.float64) * m
    k = 2
    vb = np.full(ball.shape, np.nan)
    vp = np.full(pl.shape, np.nan)
    vb[k:-k] = (ball[2 * k:] - ball[:-2 * k]) * FPS / (2 * k)
    vp[k:-k] = (pl[2 * k:] - pl[:-2 * k]) * FPS / (2 * k)
    dist = np.sqrt(((pl - ball[:, None]) ** 2).sum(-1))
    rel = np.sqrt(((vp - vb[:, None]) ** 2).sum(-1))
    ok = (np.nan_to_num(dist, nan=np.inf) < rule.near_m) & (np.nan_to_num(rel, nan=np.inf) < rule.rel_ms)
    d = np.where(ok, dist, np.inf)
    who = np.argmin(d, axis=1)
    has = np.isfinite(d[np.arange(len(who)), who])
    team = np.where(has, clean.slot_team[who], -1)
    # keep only runs of one team lasting control_s
    min_len = max(1, int(round(rule.control_s * FPS)))
    out = np.full(len(team), -1, dtype=np.int64)
    change = np.flatnonzero(np.diff(team)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(team)]))
    for a, b in zip(starts, ends):
        if team[a] >= 0 and b - a >= min_len:
            out[a:b] = team[a]
    return out


def _nearest_team(clean, radius_m):
    """(T,) team of the nearest player within radius_m of the ball, -1 if nobody."""
    m = np.asarray(clean.pitch, dtype=np.float64)
    d = np.sqrt(((clean.players[..., :2] * m - (clean.ball[:, None, :2] * m)) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    who = np.argmin(d, axis=1)
    near = d[np.arange(len(who)), who] < radius_m
    return np.where(near, clean.slot_team[who], -1)


def hackathon_labels(clean, rule=Rule()):
    """(T,) labels under the hackathon-style rule; X where DFL has no label."""
    src = clean.labels
    control = _control_team(clean, rule)
    nearest = _nearest_team(clean, rule.touch_m) if rule.date_from_touch else None
    back = int(round(rule.lookback_s * FPS))
    out = src.copy()
    live = (src == H) | (src == A)
    starts = np.flatnonzero(live & ~np.concatenate(([False], live[:-1])))
    ends = np.flatnonzero(live & ~np.concatenate((live[1:], [False]))) + 1
    for a, b in zip(starts, ends):
        cur, last_change = src[a], a
        for t in range(a, b):
            c = control[t]
            if c >= 0 and c != cur:
                cur = c          # control by the other team: the only way possession changes
                if nearest is not None:
                    s = t
                    lo = max(last_change, t - back)
                    while s > lo and nearest[s - 1] != 1 - c:   # back to the old team's last touch
                        s -= 1
                    while s < t and nearest[s] != c:            # then the new team's first touch
                        s += 1
                    out[s:t] = cur
                last_change = t
            out[t] = cur
    return out


def stats(labels, match):
    """(possessions per live minute, duration quantiles in s) over graded frames."""
    lab = np.where(graded_mask(match), labels, X).astype(np.int64)
    live = (lab == H) | (lab == A)
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(lab)]))
    durs = np.array([(b - a) / FPS for a, b in zip(starts, ends) if lab[a] in (H, A)])
    return len(durs) / (live.sum() / FPS / 60), durs


def calibrate(paths, grid=None):
    """Print, per rule, the relabelled rate and duration quantiles against the public GT."""
    grid = grid or [Rule(c, n) for c in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5) for n in (1.0, 1.5, 2.0)]
    matches = [Match.load(p) for p in paths]
    rows = []
    for rule in [None] + grid:
        n_spells, live_min, durs, agree = 0, 0.0, [], []
        for m in matches:
            lab = m.labels if rule is None else hackathon_labels(m, rule)
            rate, d = stats(lab, m)
            live = graded_mask(m) & np.isin(lab, (H, A))
            n_spells += len(d)
            live_min += live.sum() / FPS / 60
            durs.append(d)
            both = live & np.isin(m.labels, (H, A))
            agree.append((lab[both] == m.labels[both]).mean())
        q = np.percentile(np.concatenate(durs), list(PUBLIC_Q))
        # distance to the public GT: rate and quantiles on a log scale
        dist = abs(np.log(n_spells / live_min / PUBLIC_RATE)) + np.mean(
            np.abs(np.log(q / np.array(list(PUBLIC_Q.values())))))
        rows.append((dist, rule, n_spells / live_min, q, float(np.mean(agree))))
        name = "DFL original" if rule is None else "control {:.1f} s, near {:.1f} m".format(
            rule.control_s, rule.near_m)
        print("{:<28} rate {:5.2f}  q10/25/50/75/90 {}  agree {:.1%}  dist {:.3f}".format(
            name, n_spells / live_min, " ".join("{:5.1f}".format(x) for x in q),
            np.mean(agree), dist), flush=True)
    print("{:<28} rate {:5.2f}  q10/25/50/75/90 {}".format(
        "public GT (17 possessions)", PUBLIC_RATE,
        " ".join("{:5.1f}".format(x) for x in PUBLIC_Q.values())))
    best = min((r for r in rows if r[1] is not None), key=lambda r: r[0])
    print("closest rule:", best[1])
    return best[1]


def out_dir(tag=None):
    """external/dfl_hk (main rule) or external/dfl_hk_<tag> (robustness variants)."""
    return OUT_DIR if not tag else OUT_DIR + "_" + tag


def write(paths, rule, tag=None):
    out_d = out_dir(tag)
    os.makedirs(out_d, exist_ok=True)
    for p in paths:
        m = Match.load(p)
        out = replace(m, labels=hackathon_labels(m, rule).astype(np.int8))
        out.save(os.path.join(out_d, os.path.basename(p)))
        rate, d = stats(out.labels, out)
        print("  {} rate {:.2f} -> {}".format(m.name, rate, os.path.relpath(out_d)), flush=True)


def dfl_paths():
    return sorted(glob.glob(os.path.join(EXTERNAL_DIR, "dfl", "*.npz")))
