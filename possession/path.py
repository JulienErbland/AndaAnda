"""Path decoder: follow the ball from player to player, then read possession off the path.

The H/A/D decoder (hmm.py) only knows teams, so "a loose ball stays with the last team" and
"a deflection changes nothing" have to be learned by the evidence. Here they hold by
construction. The hidden state per frame is where the ball is in the game:

    slot i         a tracked player carries it (a few candidate slots per frame)
    flight H / A   nobody carries it; home / away touched it last
    out            dead ball

Transitions (costs in log-probability units; -inf = impossible):

    carrier keeps it               0
    carrier -> flight, own team    release
    flight T -> slot j             receive, + switch if j is not on team T: THE possession change
    flight H <-> flight A          unseen + switch (a hand-over whose receiver is untracked)
    slot i -> slot j               identity break (i's track ends, j starts where i was): 0;
                                   tackle within TACKLE_M: tackle; else impossible.
                                   + switch if the teams differ
    live -> out / out -> live      stop / restart, as in M2 (no switch: the restart team is free)

Evidence per state: log P(slot i carries) from carrier.py for slot states, log P(nobody) for
flight, plus M2's log P(H/A/D) for the state's team, so everything M2 knows is kept.

    decode(match, p, probs=None, cprobs=None) -> (T,) labels, X outside the graded halves
"""

from dataclasses import dataclass

import numpy as np

from bench.match import A, D, H, X

FLIGHT_H, FLIGHT_A, OUT = -1, -2, -3
NEG = -1e18


@dataclass(frozen=True)
class Params:
    k: int = 3                # candidate carrier slots per frame (top by carrier probability)
    release: float = 2.0
    receive: float = 2.0
    tackle: float = 4.0
    unseen: float = 8.0
    switch: float = 10.0
    stop: float = 40.0
    restart: float = 40.0
    w_carrier: float = 1.0
    w_m2: float = 1.0
    break_m: float = 2.0      # identity break: new slot starts within this of the ended one
    tackle_m: float = 3.0
    min_carrier: float = 1e-3


def _log(p):
    return np.log(np.clip(p, 1e-4, 1.0))


def _candidates(pc_t, prev_slots, k, floor):
    """Top-k tracked slots by carrier probability, plus last frame's still-plausible ones."""
    ok = ~np.isnan(pc_t)
    top = np.flatnonzero(ok)[np.argsort(-pc_t[ok])[:k]]
    keep = [s for s in prev_slots if ok[s] and pc_t[s] >= floor]
    return np.unique(np.concatenate([top, np.asarray(keep, dtype=np.int64)])).astype(np.int64)


def _team(states, slot_team):
    t = np.where(states == FLIGHT_H, H, np.where(states == FLIGHT_A, A, D))
    s = states >= 0
    t[s] = slot_team[states[s]]
    return t


def _transitions(prev, cur, team_p, team_c, xy_prev, xy_cur, tracked_cur, p):
    """(len(prev), len(cur)) transition scores between frame t-1 and t."""
    n_p, n_c = len(prev), len(cur)
    C = np.full((n_p, n_c), NEG)
    P, Q = prev[:, None], cur[None, :]
    tp, tc = team_p[:, None], team_c[None, :]
    live_p, live_c = P != OUT, Q != OUT
    diff_team = (tp != tc) & live_p & live_c
    C[P == Q] = 0.0
    C[live_p & ~live_c] = -p.stop
    C[~live_p & live_c] = -p.restart
    fp, fc = (P == FLIGHT_H) | (P == FLIGHT_A), (Q == FLIGHT_H) | (Q == FLIGHT_A)
    sp, sc = P >= 0, Q >= 0
    C[fp & fc & (P != Q)] = -p.unseen - p.switch
    C[sp & fc & (tp == tc)] = -p.release                       # released by the carrier's team
    rec = fp & sc
    C[rec] = -p.receive - p.switch * diff_team[rec]
    # slot -> other slot: identity break or tackle
    pairs = sp & sc & (P != Q)
    if pairs.any():
        ii, jj = np.nonzero(pairs)
        si, sj = prev[ii], cur[jj]
        d_break = np.linalg.norm(xy_prev[si] - xy_cur[sj], axis=-1)
        d_tackle = np.linalg.norm(xy_cur[si] - xy_cur[sj], axis=-1)   # NaN if i is gone
        ended = ~tracked_cur[si]
        cost = np.full(len(ii), NEG)
        brk = ended & (d_break < p.break_m)
        tk = ~ended & (np.nan_to_num(d_tackle, nan=np.inf) < p.tackle_m)
        cost[brk] = 0.0
        cost[tk] = -p.tackle
        cost -= p.switch * (team_p[ii] != team_c[jj])
        C[ii, jj] = cost
    return C


def _emissions(states, team, pc_t, pnone_t, lp_m2_t, p):
    e = p.w_m2 * lp_m2_t[team]
    slot = states >= 0
    e[slot] += p.w_carrier * _log(pc_t[states[slot]])
    flight = (states == FLIGHT_H) | (states == FLIGHT_A)
    e[flight] += p.w_carrier * _log(pnone_t)
    return e


def _viterbi_half(pc, pnone, lp_m2, xy, slot_team, p):
    n = len(pc)
    specials = np.array([FLIGHT_H, FLIGHT_A, OUT], dtype=np.int64)
    tracked = ~np.isnan(xy[..., 0])
    states_t, back_t = [], []
    slots = _candidates(pc[0], [], p.k, p.min_carrier)
    states = np.concatenate([slots, specials])
    team = _team(states, slot_team)
    delta = _emissions(states, team, pc[0], pnone[0], lp_m2[0], p)
    states_t.append(states)
    back_t.append(None)
    for t in range(1, n):
        slots = _candidates(pc[t], states[states >= 0], p.k, p.min_carrier)
        cur = np.concatenate([slots, specials])
        team_c = _team(cur, slot_team)
        C = _transitions(states, cur, team, team_c, xy[t - 1], xy[t], tracked[t], p)
        tot = delta[:, None] + C
        bp = np.argmax(tot, axis=0)
        delta = tot[bp, np.arange(len(cur))] + _emissions(cur, team_c, pc[t], pnone[t],
                                                          lp_m2[t], p)
        delta -= delta.max()            # keep the numbers small; argmax is unchanged
        states, team = cur, team_c
        states_t.append(states)
        back_t.append(bp)
    # backtrace
    labels = np.empty(n, dtype=np.int8)
    j = int(np.argmax(delta))
    for t in range(n - 1, -1, -1):
        labels[t] = _team(states_t[t][j:j + 1], slot_team)[0]
        if t:
            j = int(back_t[t][j])
    return labels


def decode(match, p=Params(), probs=None, cprobs=None):
    """probs: M2 (T, 3) P(H, A, D); cprobs: carrier (T, S) per-slot probabilities."""
    from . import carrier, learned
    probs = learned.probabilities(match) if probs is None else probs
    cprobs = carrier.probabilities(match) if cprobs is None else cprobs
    pnone = carrier.team_probabilities(match, cprobs)[:, 2]
    xy = match.players[..., :2] * np.asarray(match.pitch, dtype=np.float32)
    lp = _log(probs)
    out = np.full(match.n_frames, X, dtype=np.int8)
    for _, a, b in match.halves:
        sl = slice(a, b + 1)
        out[sl] = _viterbi_half(cprobs[sl], pnone[sl], lp[sl], xy[sl], match.slot_team, p)
    return out


def predict(match):
    return decode(match)
