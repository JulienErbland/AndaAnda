import numpy as np

from bench.match import A, H, Match
from possession import path

N = 200


def _setup():
    """Slots 0, 1 home and 2, 3 away, standing 15 m apart; neutral M2 evidence."""
    players = np.full((N, 4, 3), np.nan, dtype=np.float32)
    for s, (x, y) in enumerate([(20, 20), (35, 20), (20, 40), (35, 40)]):
        players[:, s, 0], players[:, s, 1], players[:, s, 2] = x / 105, y / 68, 0
    match = Match(name="toy", pitch=(105.0, 68.0), ball=np.full((N, 3), np.nan, np.float32),
                  players=players, slot_team=np.array([0, 0, 1, 1], np.int8),
                  halves=np.array([[1, 0, N - 1]]))
    probs = np.tile([0.45, 0.45, 0.10], (N, 1))
    cprobs = np.full((N, 4), 0.005, dtype=np.float32)
    return match, probs, cprobs


def test_a_deflection_by_the_other_team_changes_nothing():
    match, probs, cprobs = _setup()
    cprobs[:50, 0] = 0.9            # home 0 carries, passes
    cprobs[110:113, 2] = 0.3        # an away player brushes the ball in flight
    cprobs[150:, 1] = 0.9           # home 1 receives
    out = path.decode(match, probs=probs, cprobs=cprobs)
    assert (out == H).all()


def test_an_interception_changes_possession_at_the_control():
    match, probs, cprobs = _setup()
    cprobs[:50, 0] = 0.9
    cprobs[100:, 2] = 0.9           # away 2 controls from frame 100
    out = path.decode(match, probs=probs, cprobs=cprobs)
    assert (out[:95] == H).all()    # the loose ball stays with home while in flight
    assert (out[105:] == A).all()
    assert abs(int(np.flatnonzero(out == A)[0]) - 100) <= 2


def test_a_hand_over_nobody_saw_can_still_happen_on_strong_evidence():
    match, probs, cprobs = _setup()
    cprobs[:50, 0] = 0.9
    match.players[50:150] = np.nan  # blackout: no player tracked
    cprobs[50:150] = np.nan
    probs[60:150] = [0.05, 0.9, 0.05]   # M2 is sure away has it
    out = path.decode(match, probs=probs, cprobs=cprobs)
    assert (out[:50] == H).all() and (out[70:] == A).all()


def test_slot_to_slot_moves_need_a_break_or_a_tackle():
    p = path.Params()
    team = np.array([0, 0, 1, 1], np.int8)
    xy_prev = np.array([[20, 20], [50, 20], [21, 20], [60, 40]], np.float32)
    xy_cur = xy_prev.copy()
    xy_cur[0] = np.nan                             # slot 0's track ended
    tracked = ~np.isnan(xy_cur[:, 0])
    prev, cur = np.array([0, 1]), np.array([2, 3])
    C = path._transitions(prev, cur, path._team(prev, team), path._team(cur, team),
                          xy_prev, xy_cur, tracked, p)
    assert C[0, 0] == -p.switch                    # 0 continues as slot 2 (1 m away): free break
    assert C[0, 1] < -1e17                         # slot 3 is 40 m away: impossible
    assert C[1, 0] < -1e17 and C[1, 1] < -1e17     # slot 1 still tracked, nobody within 3 m
