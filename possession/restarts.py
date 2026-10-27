"""Direction 1: restart detector -- when play restarts, where (kind) and who takes it.

Every restart is a visible ball standing still for >= STILL_S that is then kicked or thrown.
Where it stood gives the kind (centre spot with both teams in their own halves = kick-off,
touchline = throw-in, corner arc = corner, goal area = goal kick, anywhere else = free kick); the
player at the ball when it moves gives the restarting team. Needs the ball: restarts whose ball
is not tracked are not found (the real tracking misses the ball on ~61% of dead frames).

A keeper holding the ball, or a player standing on it, can look the same. The detections are
evidence for the learned model, which weighs them; only the half's kick-off is used directly
(`kickoff_frames`: a half is dead until its kick-off).

    detect(match)   -> [Restart(frame, team, kind)], frame-ordered; team H/A, or None
    features(match) -> per-frame arrays for the learned model (see FEATURES)
"""

from dataclasses import dataclass

import numpy as np

from bench.match import FPS, A, H

STILL_MS = 1.0         # ball speed under this is standing still
STILL_S = 1.5          # for at least this long
STILL_SPREAD_M = 1.5   # and within this of where it stands
GAP_TOL_S = 0.3        # missing frames tolerated inside a still stretch
KICK_S = 1.0           # it must then move within this
MOVE_MS = 4.0          # at this speed, or be seen MOVE_M away
MOVE_M = 2.0
TAKER_M = 3.0          # restarting player within this of the ball when it moves
PARKED_M = 1.0         # a still ball further than this beyond a line is parked, not being taken
SEEN_GAP = 2           # missing frames tolerated between the still ball and its first move
VEL_HALF = 2
KINDS = ("kickoff", "throw_in", "corner", "goal_kick", "free_kick")
CAP_S = 60.0


@dataclass(frozen=True)
class Restart:
    frame: int        # first frame the ball moves
    team: object      # H, A, or None (nobody tracked at the ball)
    kind: str         # one of KINDS


def _ball_m(match):
    return match.ball[:, :2].astype(np.float64) * np.asarray(match.pitch, dtype=np.float64)


def _still_runs(ball):
    """[(a, b)] frame ranges where the ball stands still (short tracking gaps bridged)."""
    k = VEL_HALF
    speed = np.full(len(ball), np.nan)
    speed[k:-k] = np.linalg.norm(ball[2 * k:] - ball[:-2 * k], axis=1) * FPS / (2 * k)
    still = np.nan_to_num(speed, nan=np.inf) < STILL_MS
    idx = np.flatnonzero(still)
    if not len(idx):
        return [], speed
    tol = int(round(GAP_TOL_S * FPS)) + 1
    runs, a, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        # bridge only missing frames, and only if the ball is still where it stood
        if t - prev <= tol and np.isnan(ball[prev + 1:t, 0]).all() \
                and np.linalg.norm(ball[t] - ball[prev]) < STILL_SPREAD_M:
            prev = t
            continue
        runs.append((a, prev + 1))
        a = prev = t
    runs.append((a, prev + 1))
    return runs, speed


def _beyond_lines(p, pitch):
    """Metres outside the pitch (0 inside)."""
    return max(0.0, -p[0], p[0] - pitch[0], -p[1], p[1] - pitch[1])


def _kind(spot, pitch, own_halves):
    L, W = pitch
    x, y = spot
    if (x < 2.5 or x > L - 2.5) and (y < 2.5 or y > W - 2.5):
        return "corner"
    if y < 1.5 or y > W - 1.5:
        return "throw_in"
    if (x < 6.5 or x > L - 6.5) and abs(y - W / 2) < 10.5:
        return "goal_kick"
    if np.hypot(x - L / 2, y - W / 2) < 2.0 and own_halves:
        return "kickoff"
    return "free_kick"


def _own_halves(match, t, spot):
    """Both teams have >= 80% of their tracked players (away from the ball) on their own side."""
    L = match.pitch[0]
    x = match.players[t, :, 0].astype(np.float64) * L
    y = match.players[t, :, 1].astype(np.float64) * match.pitch[1]
    ok = ~np.isnan(x) & (np.hypot(x - spot[0], y - spot[1]) > 3.0)
    for team in (0, 1):
        xs = x[ok & (match.slot_team == team)]
        if len(xs) < 3:
            return False
        side = np.sign(np.mean(xs) - L / 2)
        if np.mean(np.sign(xs - L / 2) == side) < 0.8:
            return False
    return True


def detect(match):
    ball = _ball_m(match)
    pitch = np.asarray(match.pitch, dtype=np.float64)
    pl = match.players[..., :2].astype(np.float64) * pitch
    runs, speed = _still_runs(ball)
    n, w = len(ball), int(round(KICK_S * FPS))
    out = []
    for a, b in runs:
        seen = ~np.isnan(ball[a:b, 0])
        if seen.sum() < STILL_S * FPS or b >= n:
            continue
        spot = np.median(ball[a:b][seen], axis=0)
        if np.linalg.norm(ball[a:b][seen] - spot, axis=1).max() > STILL_SPREAD_M:
            continue
        if _beyond_lines(spot, pitch) > PARKED_M:
            continue      # a dead ball parked where it went out: the restart happens elsewhere
        after = slice(b, min(n, b + w))
        moved = np.nan_to_num(speed[after]) >= MOVE_MS
        far = np.nan_to_num(np.linalg.norm(ball[after] - spot, axis=1)) > MOVE_M
        if not (moved.any() or far.any()):
            continue
        # Where it is taken: the first seen position once it moves. Some trackers park a dead
        # ball where it left the pitch and then jump it to the restart spot (DFL does).
        left = np.flatnonzero(np.nan_to_num(np.linalg.norm(ball[after] - spot, axis=1)) > 0.3)
        seen_leaving = len(left) and np.isnan(ball[b:b + left[0], 0]).sum() <= SEEN_GAP
        if not seen_leaving:
            # it vanished and reappeared elsewhere: a restart, but the taker is not visible
            out.append(Restart(int(b), None, _kind(spot, pitch, False)))
            continue
        b = b + left[0]
        spot = ball[b]
        # the taker: nearest player to that spot around the moment the ball moves
        lo = max(0, b - int(0.3 * FPS))
        d = np.linalg.norm(pl[lo:b + 1] - spot, axis=-1)
        d = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=0)
        s = int(np.argmin(d))
        team = None
        if np.isfinite(d[s]) and d[s] <= TAKER_M:
            team = H if match.slot_team[s] == 0 else A
        out.append(Restart(int(b), team, _kind(spot, pitch, _own_halves(match, b, spot))))
    return out


LINEUP_S = 2.0         # kick-off line-up: held this long,
LINEUP_SHARE = 0.9     # with this share of each team's tracked players in its own half,
LINEUP_LINE_M = 2.0    # (players within this of the halfway line count on either side)
LINEUP_MIN = 7         # at least this many tracked per team,
LINEUP_SPEED = 1.5     # and players at walking pace (mean m/s)


def lineups(match):
    """Kick-offs found without the ball: [(start, end)] of each line-up (end: first frame after
    it), where both teams stand
    in opposite halves (LINEUP_SHARE of each team's tracked players), at walking pace, for at
    least LINEUP_S. Live play rarely does this at walking pace for 2 s."""
    L = match.pitch[0]
    x = match.players[..., 0].astype(np.float64) * L
    sides, ok = [], np.ones(len(x), dtype=bool)
    if not len(x):
        return []
    for team in (0, 1):
        xs = x[:, match.slot_team == team]
        n = (~np.isnan(xs)).sum(1)
        with np.errstate(invalid="ignore"):
            side = np.sign(np.nansum(xs, axis=1) / np.maximum(n, 1) - L / 2)
            across = (side[:, None] * (xs - L / 2) < -LINEUP_LINE_M).sum(1)
            share = 1 - across / np.maximum(n, 1)
        ok &= (n >= LINEUP_MIN) & (share >= LINEUP_SHARE)
        sides.append(side)
    ok &= sides[0] * sides[1] < 0
    sp = match.players[..., 2].astype(np.float64)
    n_sp = (~np.isnan(sp)).sum(1)
    ok &= np.nansum(sp, axis=1) / np.maximum(n_sp, 1) < LINEUP_SPEED
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= LINEUP_S * FPS and b < len(x)]


KICKOFF_SEARCH_S = 180.0   # a half's kick-off is looked for in its first 3 minutes
KICKOFF_DEAD_P = 0.5       # each step to it must be dead on average for the model,
KICKOFF_WALK_MS = 1.5      # or at walking pace (mean player speed; live play: 3% of 15 s windows on DFL)


def kickoff_frames(match, p_dead, restarts=None, lineup=None):
    """[(first, kick)] per graded half: the half is dead from its first frame until its kick-off.

    A half graded from its kick-off opens inside the line-up: nothing to do. Otherwise the
    kick-off is the last of the early line-ups / kick-off restarts (first KICKOFF_SEARCH_S of
    the half) reached through dead ball only: each step must be dead on average for the model
    (p_dead, (T,)) or at walking pace, so a later kick-off after a goal can never force minutes of
    live play dead."""
    rs = detect(match) if restarts is None else restarts
    lu = lineups(match) if lineup is None else lineup
    cand = sorted([r.frame for r in rs if r.kind == "kickoff"] + [b for _, b in lu])
    out = []
    for _, first, last in match.halves:
        if any(a <= first < b for a, b in lu):
            continue
        cur = first
        for f in cand:
            if not first < f <= min(last, first + KICKOFF_SEARCH_S * FPS):
                continue
            sp = match.players[cur:f, :, 2].astype(np.float64)
            walk = np.nansum(sp) / max(1, (~np.isnan(sp)).sum()) < KICKOFF_WALK_MS
            if np.mean(p_dead[cur:f]) < KICKOFF_DEAD_P and not walk:
                break
            cur = f
        if cur > first:
            out.append((int(first), int(cur)))
    return out


FEATURES = ("restart_last_team", "restart_since", "restart_next_team", "restart_until",
            "restart_last_kind", "restart_next_kind")


def features(match, restarts=None):
    """Per-frame restart features (team +1 home / -1 away / 0 unknown; kind 1-5 as KINDS, 0 none;
    times in s, capped)."""
    from .touches import _last_next
    rs = detect(match) if restarts is None else restarts
    # kick-offs seen only as a line-up (ball not tracked): added, team unknown
    near = np.asarray([r.frame for r in rs], dtype=np.int64)
    for _, b in lineups(match):
        if not len(near) or np.min(np.abs(near - b)) > 3 * FPS:
            rs = rs + [Restart(b, None, "kickoff")]
    rs = sorted(rs, key=lambda r: r.frame)
    f = np.asarray([r.frame for r in rs], dtype=np.int64)
    team = np.asarray([0.0 if r.team is None else (1.0 if r.team == H else -1.0) for r in rs])
    kind = np.asarray([1.0 + KINDS.index(r.kind) for r in rs])
    n = match.n_frames
    out = {}
    out["restart_last_team"], out["restart_since"], out["restart_next_team"], \
        out["restart_until"] = _last_next(f, team, n)
    out["restart_last_kind"], _, out["restart_next_kind"], _ = _last_next(f, kind, n)
    for k in ("restart_since", "restart_until"):
        out[k] = np.clip(out[k], 0, CAP_S)
    return out
