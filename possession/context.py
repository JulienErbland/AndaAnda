"""Per-frame context features for the learned evidence model (M2). No labels, no identities.

Built for the ball-gap problem: most features still exist when the ball is not tracked (team
speeds and shape, how play is slowing down, where and how the ball was last and will next be
seen, which team last and next controlled it). Offline, so windows look both ways.

Direction of play: home's attacking direction per half is the sign that makes home's centroid sit
behind away's on average. Whichever team attacks, the defenders stand between the attackers and
their own goal, so home's centroid is nearer home's goal in both phases. "Forward" features are
then expressed in each team's own attacking direction, which makes the model side-agnostic.
"""

import numpy as np

from bench.match import FPS

from .features import rolling_nanmean

CONTROL_M = 1.5
CAP_S = 60.0


def _ffill_index(mask):
    """Index of the last True at or before t, -1 if none."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(mask):
    """Index of the next True at or after t, len if none."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _since(mask):
    last = _ffill_index(mask)
    return np.where(last >= 0, (np.arange(len(mask)) - last) / FPS, CAP_S).clip(0, CAP_S), last


def _until(mask):
    nxt = _bfill_index(mask)
    return np.where(nxt < len(mask), (nxt - np.arange(len(mask))) / FPS, CAP_S).clip(0, CAP_S), nxt


def _take(values, idx, valid):
    out = np.full(len(idx), np.nan)
    ok = valid & (idx >= 0) & (idx < len(values))
    out[ok] = values[idx[ok]]
    return out


def _half_block(match, x):
    """Apply `x(lo, hi) -> dict of arrays` per graded half; NaN elsewhere."""
    out = {}
    for _, first, last in match.halves:
        part = x(first, last + 1)
        for k, v in part.items():
            out.setdefault(k, np.full(match.n_frames, np.nan))[first:last + 1] = v
    return out


def compute(match):
    L, W = match.pitch
    metres = np.asarray(match.pitch, dtype=np.float64)

    def block(lo, hi):
        ball = match.ball[lo:hi, :2].astype(np.float64) * metres
        bspeed = match.ball[lo:hi, 2].astype(np.float64)
        pl = match.players[lo:hi, :, :2].astype(np.float64) * metres
        ps = match.players[lo:hi, :, 2].astype(np.float64)
        tracked = ~np.isnan(pl[..., 0])
        n = hi - lo
        f = {}

        team_mask = [match.slot_team == t for t in (0, 1)]
        cx, cy, spd, cnt, sx, sy, xmin, xmax = [], [], [], [], [], [], [], []
        for m in team_mask:
            tr = tracked[:, m]
            c = tr.sum(1)
            px, py = np.where(tr, pl[:, m, 0], 0), np.where(tr, pl[:, m, 1], 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, my = px.sum(1) / c, py.sum(1) / c
                cx.append(mx)
                cy.append(my)
                spd.append(np.where(tr, np.nan_to_num(ps[:, m]), 0).sum(1) / c)
                sx.append(np.sqrt(np.where(tr, (pl[:, m, 0] - mx[:, None]) ** 2, 0).sum(1) / c))
                sy.append(np.sqrt(np.where(tr, (pl[:, m, 1] - my[:, None]) ** 2, 0).sum(1) / c))
            xmin.append(np.where(c > 0, np.nanmin(np.where(tr, pl[:, m, 0], np.inf), 1), np.nan))
            xmax.append(np.where(c > 0, np.nanmax(np.where(tr, pl[:, m, 0], -np.inf), 1), np.nan))
            cnt.append(c)

        # +1 if home attacks towards x = L in this half
        diff = cx[0] - cx[1]
        home_dir = 1.0 if np.nanmedian(diff) < 0 else -1.0
        dirs = (home_dir, -home_dir)

        def fwd(x, d):  # metres from the centre line towards the team's attacking goal
            return d * (x - L / 2)

        for t, name in ((0, "home"), (1, "away")):
            d = dirs[t]
            f[name + "_n"] = cnt[t]
            f[name + "_speed"] = rolling_nanmean(spd[t], FPS)
            f[name + "_fwd"] = fwd(cx[t], d)
            vel = np.gradient(rolling_nanmean(fwd(cx[t], d), FPS)) * FPS
            f[name + "_fwd_vel"] = rolling_nanmean(vel, FPS)
            f[name + "_spread_x"] = sx[t]
            f[name + "_spread_y"] = sy[t]
            front = xmax[t] if d > 0 else xmin[t]
            back = xmin[t] if d > 0 else xmax[t]
            f[name + "_front"] = fwd(front, d)
            f[name + "_back"] = fwd(back, d)
        f["centroid_gap"] = np.hypot(cx[0] - cx[1], cy[0] - cy[1])

        with np.errstate(invalid="ignore", divide="ignore"):
            all_speed = np.where(tracked, np.nan_to_num(ps), 0).sum(1) / tracked.sum(1)
        for s in (1, 3, 8):
            f["speed_{}s".format(s)] = rolling_nanmean(all_speed, s * FPS)
        past = rolling_nanmean(np.concatenate([np.full(3 * FPS, np.nan), all_speed]),
                               3 * FPS)[:n]  # mean over roughly the previous 3 s
        future = rolling_nanmean(np.concatenate([all_speed, np.full(3 * FPS, np.nan)]),
                                 3 * FPS)[3 * FPS:]
        f["speed_past_minus_future"] = past - future
        f["n_tracked"] = tracked.sum(1)

        vis = ~np.isnan(ball[:, 0])
        line = np.minimum.reduce([ball[:, 0], L - ball[:, 0], ball[:, 1], W - ball[:, 1]])
        f["ball_vis"] = vis.astype(float)
        f["ball_speed"] = np.where(vis, bspeed, np.nan)
        f["ball_line_dist"] = line  # negative when outside the pitch
        f["ball_fwd_home"] = fwd(ball[:, 0], dirs[0])
        since, last = _since(vis)
        until, nxt = _until(vis)
        f["ball_since_seen"], f["ball_until_seen"] = since, until
        f["ball_gap_len"] = np.minimum(since + until, 2 * CAP_S)
        f["ball_last_line_dist"] = _take(line, last, np.ones(n, bool))
        f["ball_next_line_dist"] = _take(line, nxt, np.ones(n, bool))
        f["ball_last_speed"] = _take(bspeed, last, np.ones(n, bool))
        f["ball_last_fwd_home"] = _take(fwd(ball[:, 0], dirs[0]), last, np.ones(n, bool))

        dist = np.sqrt(((pl - ball[:, None, :]) ** 2).sum(-1))
        dist = np.where(np.isnan(dist), np.inf, dist)
        dh = dist[:, team_mask[0]].min(1) if team_mask[0].any() else np.full(n, np.inf)
        da = dist[:, team_mask[1]].min(1) if team_mask[1].any() else np.full(n, np.inf)
        f["d_home"] = np.where(np.isfinite(dh), dh, np.nan)
        f["d_away"] = np.where(np.isfinite(da), da, np.nan)
        with np.errstate(invalid="ignore"):
            gap = np.clip(np.nan_to_num((da - dh) / 0.25, nan=0.0), -1, 1)
        control = np.where(np.minimum(dh, da) < CONTROL_M, gap, 0.0)
        for s in (0.5, 2, 5):
            f["control_{}s".format(s)] = rolling_nanmean(control, max(1, int(s * 2 * FPS)))
        held = np.abs(control) > 0.5
        since_c, last_c = _since(held)
        until_c, next_c = _until(held)
        f["control_since"], f["control_until"] = since_c, until_c
        f["control_last_team"] = np.nan_to_num(_take(np.sign(control), last_c, np.ones(n, bool)))
        f["control_next_team"] = np.nan_to_num(_take(np.sign(control), next_c, np.ones(n, bool)))
        return f

    return _half_block(match, block)


def matrix(feats):
    names = sorted(feats)
    return np.column_stack([feats[k] for k in names]).astype(np.float32), names
