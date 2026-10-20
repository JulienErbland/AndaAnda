"""Direction 3: describe each ball gap as a whole, so the model can tell live gaps from stoppages
and see hand-overs that happen while the ball is invisible.

features(match) -> dict of (T,) arrays, constant over each ball gap; NaN outside gaps (the team
keys and the flip are 0 outside gaps):

    gap_len_s              length of the gap
    gap_progress           0 at the gap's first frame, 1 at its last
    gap_implied_speed      last-seen to next-seen ball position over the gap duration (m/s):
                           a live pass travels, a stoppage ball barely moves
    gap_lost_team          +1 home / -1 away / 0: nearest player within 2 m when the ball was lost
    gap_found_team         same when it reappears
    gap_peak_player_speed  mean over the gap of the fastest tracked player's speed (live gaps
                           have sprinters, dead ones walkers)
    gap_shape_flip         -1 home went from advancing to retreating (it lost the ball), +1 the
                           reverse, 0 neither: home's centroid forward velocity, first vs second
                           half of the gap, each beyond +-0.5 m/s

Status (Oct 20): built and spec-tested, NOT used by the model. Added to the LightGBM features it
changed nothing beyond noise (DFL 0.5220 -> 0.5205; gap scenarios +-1-4 points): context.py
already carries the same information (ball since/until seen, last/next position, last/next
control team, team velocities). Kept as the base for a gap-level model.

Direction of play per half: the median sign of (home centroid x - away centroid x), as in
context.py (defenders stand between attackers and their own goal).
"""

import warnings

import numpy as np

from bench.match import FPS

from .features import rolling_nanmean

TEAM_RADIUS_M = 2.0
FLIP_MS = 0.5
KEYS_NAN = ("gap_len_s", "gap_progress", "gap_implied_speed", "gap_peak_player_speed")
KEYS_ZERO = ("gap_lost_team", "gap_found_team", "gap_shape_flip")


def _runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return zip(e[::2], e[1::2])


def features(match):
    n = match.n_frames
    out = {k: np.full(n, np.nan) for k in KEYS_NAN}
    out.update({k: np.zeros(n) for k in KEYS_ZERO})
    metres = np.asarray(match.pitch, dtype=np.float64)
    ball = match.ball[:, :2].astype(np.float64) * metres
    pl = match.players[..., :2].astype(np.float64) * metres
    tracked = ~np.isnan(pl[..., 0])
    sign = np.where(match.slot_team == 0, 1.0, -1.0)
    with np.errstate(invalid="ignore", all="ignore"):
        peak = np.where(tracked.any(1),
                        np.nanmax(np.where(tracked, match.players[..., 2], -np.inf), 1), np.nan)

    def centroid(team):
        m = match.slot_team == team
        c = tracked[:, m].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tracked[:, m], pl[:, m, 0], 0).sum(1) / c

    home_x, away_x = centroid(0), centroid(1)

    def team_at(frame):
        if frame < 0 or frame >= n or np.isnan(ball[frame, 0]):
            return 0.0
        d = np.sqrt(((pl[frame] - ball[frame]) ** 2).sum(-1))
        d = np.where(np.isnan(d), np.inf, d)
        i = int(np.argmin(d))
        return float(sign[i]) if d[i] <= TEAM_RADIUS_M else 0.0

    for _, first, last in match.halves:
        lo, hi = first, last + 1
        home_dir = 1.0 if np.nanmedian(home_x[lo:hi] - away_x[lo:hi]) < 0 else -1.0
        home_vel = np.gradient(rolling_nanmean(home_dir * home_x[lo:hi], FPS)) * FPS
        home_vel = rolling_nanmean(home_vel, FPS)
        missing = np.isnan(ball[lo:hi, 0])
        for a, b in _runs(missing):
            ga, gb = lo + a, lo + b          # gap frames [ga, gb)
            length = gb - ga
            sl = slice(ga, gb)
            out["gap_len_s"][sl] = length / FPS
            out["gap_progress"][sl] = np.linspace(0, 1, length) if length > 1 else 0.0
            if ga - 1 >= lo and gb < hi:
                out["gap_implied_speed"][sl] = np.hypot(*(ball[gb] - ball[ga - 1])) / (
                    (gb - ga + 1) / FPS)
            out["gap_lost_team"][sl] = team_at(ga - 1) if ga - 1 >= lo else 0.0
            out["gap_found_team"][sl] = team_at(gb) if gb < hi else 0.0
            out["gap_peak_player_speed"][sl] = np.nanmean(peak[sl])
            if length < FPS:  # too short for a change of shape to mean anything
                continue
            mid = a + length // 2
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # home untracked all gap long
                v1, v2 = np.nanmean(home_vel[a:mid]), np.nanmean(home_vel[mid:b])
            if v1 > FLIP_MS and v2 < -FLIP_MS:
                out["gap_shape_flip"][sl] = -1.0
            elif v1 < -FLIP_MS and v2 > FLIP_MS:
                out["gap_shape_flip"][sl] = 1.0
    return out
