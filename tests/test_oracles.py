"""The benchmark's oracles must fix exactly their own error type and nothing else."""

import numpy as np

from bench.match import A, D, H
from scripts import benchmark as b

GT = np.array([H] * 50 + [D] * 30 + [A] * 40 + [H] * 10 + [A] * 70, dtype=np.int8)


class Seen:  # only .ball is read (by the ball-gap oracle)
    def __init__(self, missing):
        self.ball = np.zeros((len(GT), 3))
        self.ball[missing, 0] = np.nan


def test_timing_oracle_moves_late_changes_back_but_not_far_ones():
    late = np.roll(GT, 10)
    late[:10] = H
    assert (b.oracle_timing(GT, late, None) == GT).all()        # 10 f late: all fixed
    very = np.roll(GT, 40)
    very[:40] = H
    assert (b.oracle_timing(GT, very, None) == very).all()      # 40 f: beyond 1 s, untouched


def test_dead_oracle_fixes_dead_calls_only():
    pred = GT.copy()
    pred[50:80] = H                    # missed stoppage
    pred[150:160] = D                  # invented stoppage inside an A spell
    pred[120:130] = A                  # a team error: must stay
    fixed = b.oracle_dead(GT, pred, None)
    assert (fixed[50:80] == D).all() and (fixed[150:160] == A).all()
    assert (fixed[120:130] == A).all()


def test_team_oracle_keeps_the_dead_structure():
    pred = np.where(GT == D, D, H).astype(np.int8)
    pred[60:70] = H                    # wrong: should be D, team oracle must not touch it
    fixed = b.oracle_team(GT, pred, None)
    assert (fixed[np.isin(GT, (H, A)) & (pred != D)] == GT[np.isin(GT, (H, A)) & (pred != D)]).all()
    assert (fixed[60:70] == H).all()


def test_window_oracles_only_touch_their_events():
    pred = np.full(len(GT), H, dtype=np.int8)
    r = b.oracle_restarts(GT, pred, None)                       # restart at frame 80
    assert (r[30:130] == GT[30:130]).all() and (r[:30] == H).all()
    h = b.oracle_handovers(GT, pred, None)                      # H<->A at 120, 130
    assert (h[95:155] == GT[95:155]).all() and (h[:95] == H).all()
    s = b.oracle_short_spells(GT, np.full(len(GT), A, np.int8), None)
    assert (s[120:130] == H).all() and (s[130:] == A).all()     # the 10-frame H spell only


def test_ball_gap_oracle_copies_only_missing_ball_frames():
    pred = np.full(len(GT), H, dtype=np.int8)
    fixed = b.oracle_ball_gaps(GT, pred, Seen(slice(55, 75)))
    assert (fixed[55:75] == D).all() and (fixed[75:80] == H).all()
