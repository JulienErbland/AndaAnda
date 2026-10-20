"""Per-frame, team-level features from tracking alone (no labels, no player identities).

Everything is a (T,) float array in SI units; NaN where it cannot be computed (no ball, nobody
tracked). Slots are anonymous, so features are collective: nearest player per team, team means.
"""

import numpy as np

from bench.match import FPS

SMOOTH_S = 2.0          # centred window for the mean-player-speed signal
OUT_MARGIN_M = 0.5      # ball this far beyond a line counts as out of the pitch


def rolling_nanmean(x, window):
    """Centred rolling mean ignoring NaN; NaN where the whole window is NaN."""
    ok = ~np.isnan(x)
    kernel = np.ones(window)
    total = np.convolve(np.where(ok, x, 0.0), kernel, "same")
    count = np.convolve(ok.astype(float), kernel, "same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def compute(match):
    metres = np.asarray(match.pitch, dtype=np.float64)
    ball = match.ball[:, :2].astype(np.float64) * metres
    players = match.players[..., :2].astype(np.float64) * metres
    tracked = ~np.isnan(players[..., 0])

    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))  # NaN if ball or player missing
    nearest = {}
    for team, name in ((0, "home"), (1, "away")):
        d = np.where(match.slot_team[None, :] == team, dist, np.nan)
        with np.errstate(all="ignore"):
            nearest[name] = np.where(np.isnan(d).all(1), np.nan, np.nanmin(np.nan_to_num(d, nan=np.inf), 1))

    speed = np.where(tracked, np.nan_to_num(match.players[..., 2]), 0.0)
    count = tracked.sum(1)
    mean_speed = np.where(count > 0, speed.sum(1) / np.maximum(count, 1), np.nan)

    ball_visible = ~np.isnan(ball[:, 0])
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - metres[0],
                                 -ball[:, 1], ball[:, 1] - metres[1]])
    return {
        "ball_visible": ball_visible,
        "ball_speed": match.ball[:, 2].astype(np.float64),
        "ball_out": ball_visible & (outside > OUT_MARGIN_M),
        "d_home": nearest["home"],
        "d_away": nearest["away"],
        "player_speed": rolling_nanmean(mean_speed, int(SMOOTH_S * FPS)),
        "n_tracked": tracked.sum(1),
    }
