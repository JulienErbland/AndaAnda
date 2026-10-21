"""Direction 2: touch detector -- who really plays the ball, beyond "a player is near it".

Proximity alone cannot tell a control from a pass flying by, a duel or a deflection, and the
benchmark shows that is where M2 loses most (86% of missed possessions start with a hand-over,
mostly with the ball tracked). Velocity can tell them apart:

- control: a player of one team is within NEAR_M of the ball and moves WITH it (relative
  velocity under REL_MS) for at least CONTROL_S. A pass flying by has the ball's full speed
  relative to the player, so it never qualifies, however close it passes.
- deflection: next to a player, the ball turns by more than TURN_DEG while moving fast, and that
  team does not control it within CONTROL_S.
- contact: any other sharp change of the ball's velocity (> CONTACT_MS over +-3 frames) next to a
  player: kicks, blocks, clearances, one-touch interceptions. On DFL it catches 87% of
  interceptions and possession-changing tackles on clean tracking (61% on hackathon-like
  tracking), where a 0.4 s control catches 22%.

Team-level, not slot-level: on the real tracking, slots swap identities, so a control episode is
"the nearest moving-with-the-ball player belongs to team T" on consecutive frames, whichever slot
it is. Needs the ball: frames without a tracked ball produce no touches.

    detect(match)   -> [Touch(frame, team, kind)], frame-ordered
    features(match) -> per-frame arrays for the learned model (see FEATURES)
"""

from dataclasses import dataclass

import numpy as np

from bench.match import FPS, A, H

NEAR_M = 1.5
REL_MS = 2.0
CONTROL_S = 0.4
TURN_DEG = 30.0
MIN_SPEED_MS = 3.0
CONTACT_MS = 3.0
DEDUP_S = 0.3
VEL_HALF = 2          # velocity by central difference over +-2 frames (0.16 s)
CAP_S = 60.0


@dataclass(frozen=True)
class Touch:
    frame: int
    team: int            # H or A
    kind: str            # "control", "deflection" or "contact"


def _velocity(xy):
    """(T, ..., 2) m/s by central difference over +-VEL_HALF frames; NaN at gaps."""
    v = np.full(xy.shape, np.nan)
    k = VEL_HALF
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    return v


def _frame_state(match, with_slot=False):
    """Per frame: team controlling (H/A/-1) and team of the nearest player within NEAR_M
    (+ the controlling slot, -1 if none, when with_slot)."""
    m = np.asarray(match.pitch, dtype=np.float64)
    ball = match.ball[:, :2].astype(np.float64) * m
    pl = match.players[..., :2].astype(np.float64) * m
    vb, vp = _velocity(ball), _velocity(pl)
    dist = np.sqrt(((pl - ball[:, None, :]) ** 2).sum(-1))
    rel = np.sqrt(((vp - vb[:, None, :]) ** 2).sum(-1))
    near = np.nan_to_num(dist, nan=np.inf) < NEAR_M
    moving_with = near & (np.nan_to_num(rel, nan=np.inf) < REL_MS)
    # nearest qualifying player decides the controlling team
    d_with = np.where(moving_with, dist, np.inf)
    who = np.argmin(d_with, axis=1)
    has = np.isfinite(d_with[np.arange(len(who)), who])
    control_team = np.where(has, match.slot_team[who], -1)
    d_near = np.where(near, dist, np.inf)
    nwho = np.argmin(d_near, axis=1)
    nhas = np.isfinite(d_near[np.arange(len(nwho)), nwho])
    near_team = np.where(nhas, match.slot_team[nwho], -1)
    if with_slot:
        return control_team, near_team, vb, np.where(has, who, -1)
    return control_team, near_team, vb


def detect(match):
    control_team, near_team, vb = _frame_state(match)
    n = len(control_team)
    min_len = int(round(CONTROL_S * FPS))
    touches = []
    controls = np.zeros(n, dtype=bool)
    # control episodes: runs of the same controlling team, >= min_len frames
    start = 0
    for t in range(1, n + 1):
        if t == n or control_team[t] != control_team[start]:
            team = control_team[start]
            if team >= 0 and t - start >= min_len:
                touches.append(Touch(int(start), H if team == 0 else A, "control"))
                controls[start:t] = True
            start = t
    # deflections: a sharp turn at speed next to a player, not followed by that team's control
    speed = np.sqrt((vb ** 2).sum(-1))
    k = 3
    before = vb[:-2 * k]
    after = vb[2 * k:]
    with np.errstate(invalid="ignore"):
        cos = (before * after).sum(-1) / (np.linalg.norm(before, axis=-1) *
                                          np.linalg.norm(after, axis=-1))
    turn = np.full(n, False)
    turn[k:-k] = (np.degrees(np.arccos(np.clip(np.nan_to_num(cos, nan=1.0), -1, 1))) > TURN_DEG) \
        & (np.nan_to_num(speed[:-2 * k]) > MIN_SPEED_MS) & (np.nan_to_num(speed[2 * k:]) > MIN_SPEED_MS)
    last = -10 ** 9
    for t in np.flatnonzero(turn & (near_team >= 0)):
        if t - last < int(0.3 * FPS):
            continue
        team = near_team[t]
        follow = control_team[t:t + min_len + 1]
        if np.any(follow == team) and controls[t:t + min_len + 1].any():
            continue
        touches.append(Touch(int(t), H if team == 0 else A, "deflection"))
        last = t
    # contacts: a sharp velocity change next to a player, unless already a deflection/control start
    dv = np.zeros(n)
    dv[k:-k] = np.nan_to_num(np.linalg.norm(vb[2 * k:] - vb[:-2 * k], axis=1))
    taken = {(t.frame // int(DEDUP_S * FPS), t.team) for t in touches}
    last = {H: -10 ** 9, A: -10 ** 9}
    for t in np.flatnonzero((dv > CONTACT_MS) & (near_team >= 0)):
        team = H if near_team[t] == 0 else A
        if t - last[team] < int(DEDUP_S * FPS):
            continue
        last[team] = t
        if any(abs(x.frame - t) < int(DEDUP_S * FPS) and x.team == team for x in touches[-20:]
               if x.kind != "contact") or (t // int(DEDUP_S * FPS), team) in taken:
            continue
        touches.append(Touch(int(t), team, "contact"))
    return sorted(touches, key=lambda x: x.frame)


FEATURES = ("touch_control_now",
            "control_last_team", "control_since", "control_next_team", "control_until",
            "touch_last_team", "touch_since", "touch_next_team", "touch_until",
            "deflection_since")


def _last_next(frames, values, n):
    """(last value, seconds since, next value, seconds until) per frame; 0 / CAP_S if none."""
    idx = np.arange(n)
    if not len(frames):
        return np.zeros(n), np.full(n, CAP_S), np.zeros(n), np.full(n, CAP_S)
    il = np.searchsorted(frames, idx, side="right") - 1
    ok = il >= 0
    last = np.where(ok, values[np.maximum(il, 0)], 0.0)
    since = np.where(ok, (idx - frames[np.maximum(il, 0)]) / FPS, CAP_S)
    inx = np.searchsorted(frames, idx, side="left")
    ok = inx < len(frames)
    j = np.minimum(inx, len(frames) - 1)
    nxt = np.where(ok, values[j], 0.0)
    until = np.where(ok, (frames[j] - idx) / FPS, CAP_S)
    return last, np.clip(since, 0, CAP_S), nxt, np.clip(until, 0, CAP_S)


def features(match):
    """Per-frame touch features (team as +1 home / -1 away / 0 none; times in s, capped)."""
    n = match.n_frames
    control_team, _, _ = _frame_state(match)
    ts = detect(match)
    sign = {H: 1.0, A: -1.0}
    out = {"touch_control_now": np.where(control_team == 0, 1.0,
                                         np.where(control_team == 1, -1.0, 0.0))}
    for prefix, kinds in (("control", {"control"}), ("touch", {"control", "deflection", "contact"})):
        sel = [t for t in ts if t.kind in kinds]
        f = np.asarray([t.frame for t in sel], dtype=np.int64)
        v = np.asarray([sign[t.team] for t in sel])
        (out[prefix + "_last_team"], out[prefix + "_since"],
         out[prefix + "_next_team"], out[prefix + "_until"]) = _last_next(f, v, n)
    d = np.asarray([t.frame for t in ts if t.kind == "deflection"], dtype=np.int64)
    out["deflection_since"] = _last_next(d, np.ones(len(d)), n)[1]
    return out
