"""M1: hand-built per-frame evidence + a 3-state (H, A, D) Viterbi decoder.

Why a decoder: possession is an interval, not a per-frame vote. Viterbi picks the single label
sequence that best balances per-frame evidence against a cost for every change of state, so a
flicker of contrary evidence is not worth a switch, while sustained evidence is. Decoding the
whole half at once also removes the centred window's built-in lead/lag.

Evidence, as log-odds, all scaled by `Params` (no learning in M1):

- dead vs alive: walking pace is a stoppage (mean player speed, 2 s window; validated on DFL and
  on the public window), the ball beyond a line, and -- weakly -- the ball not being tracked
  (it is missing ~3x more often in dead time than in live play);
- home vs away, only when someone is close enough to play the ball: the gap between the two
  teams' nearest distances. A ball in flight or loose gives no team evidence, so the last team
  carries forward -- which is also what happens through the long ball gaps.

The per-frame terms are correlated over consecutive frames, so they are not real likelihoods;
`Params` weights and penalties are energies, tuned on the bench.
"""

from dataclasses import dataclass

import numpy as np

from bench.match import A, D, H, X

from . import features

STATES = (H, A, D)


@dataclass(frozen=True)
class Params:
    # dead vs alive
    walk_speed: float = 1.5       # m/s; mean player speed at which dead and alive are even
    walk_scale: float = 0.5       # m/s per unit of log-odds
    walk_cap: float = 3.0
    w_walk: float = 1.0
    w_out: float = 12.0           # ball beyond a line
    w_ball_missing: float = 0.25  # ball not tracked (vs tracked: -w_ball_missing / 2)
    # home vs away
    control_m: float = 1.5        # someone within this of the ball can be playing it
    gap_scale_m: float = 0.25     # (d_away - d_home) metres per unit of log-odds, capped at +-1
    w_team: float = 2.0
    # transition costs (log units)
    switch: float = 40.0          # H <-> A
    stop: float = 40.0            # H/A -> D
    restart: float = 40.0         # D -> H/A


def evidence(f, p):
    """(T, 3) log-scores for H, A, D."""
    walk = np.clip((p.walk_speed - f["player_speed"]) / p.walk_scale, -p.walk_cap, p.walk_cap)
    dead = (p.w_walk * np.nan_to_num(walk)
            + p.w_out * f["ball_out"]
            + np.where(f["ball_visible"], -p.w_ball_missing / 2, p.w_ball_missing))

    d_home = np.nan_to_num(f["d_home"], nan=np.inf)
    d_away = np.nan_to_num(f["d_away"], nan=np.inf)
    near = np.minimum(d_home, d_away) < p.control_m
    with np.errstate(invalid="ignore"):  # inf - inf where neither team is tracked: no evidence
        gap = np.clip(np.nan_to_num((d_away - d_home) / p.gap_scale_m, nan=0.0), -1, 1)
    team = np.where(near, p.w_team * gap, 0.0)

    return np.stack([team, -team, dead], axis=1)


def transitions(p):
    """(3, 3) log transition scores, from row state to column state."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -p.switch
    t[H, D] = t[A, D] = -p.stop
    t[D, H] = t[D, A] = -p.restart
    return t


def viterbi(scores, trans):
    """Most likely state index sequence for (T, 3) scores under (3, 3) log transitions."""
    n = len(scores)
    back = np.empty((n, 3), dtype=np.int8)
    best = scores[0].copy()
    for t in range(1, n):
        cand = best[:, None] + trans
        back[t] = cand.argmax(0)
        best = cand.max(0) + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(match, p=Params(), feats=None):
    feats = feats if feats is not None else features.compute(match)
    return decode_scores(match, evidence(feats, p), p)


def decode_scores(match, scores, p):
    """Viterbi over each graded half for (T, 3) scores; X outside the halves."""
    trans = transitions(p)
    out = np.full(match.n_frames, X, dtype=np.int8)
    state_codes = np.asarray(STATES, dtype=np.int8)
    for _, first, last in match.halves:
        out[first:last + 1] = state_codes[viterbi(scores[first:last + 1], trans)]
    return out


def predict(match):
    return decode(match)
