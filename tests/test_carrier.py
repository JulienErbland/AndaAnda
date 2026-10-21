from dataclasses import replace

import numpy as np

from bench.match import FPS, H, Match
from possession import carrier


def _match():
    """Home player 0 dribbles for 1.2 s, then the ball flies away from everyone."""
    n = int(2.4 * FPS)
    t = np.arange(n) / FPS
    players = np.full((n, 2, 3), np.nan, dtype=np.float32)
    players[:, 0, 0] = (30 + 4 * t) / 105            # home, jogging at 4 m/s
    players[:, 0, 1] = 34 / 68
    players[:, 1, 0] = 60 / 105                      # away, standing far off
    players[:, 1, 1] = 20 / 68
    ball = np.zeros((n, 3), dtype=np.float32)
    ball[:, 0] = players[:, 0, 0] + 0.5 / 105        # at his feet
    ball[:, 1] = 34 / 68
    fly = t >= 1.2
    ball[fly, 0] = ball[fly, 0] + (20 * (t[fly] - 1.2)) / 105   # kicked forward at 24 m/s
    return Match(name="toy", pitch=(105.0, 68.0), ball=ball, players=players,
                 slot_team=np.array([0, 1], np.int8), halves=np.array([[1, 0, n - 1]]),
                 labels=np.full(n, H, np.int8))


def test_carrier_follows_the_player_into_his_new_slot():
    clean = _match()
    seen = replace(clean, players=clean.players[:, ::-1].copy(),
                   slot_team=clean.slot_team[::-1].copy())   # the degrader repacks slots
    lab = carrier.labels(clean, seen)
    dribble = slice(5, int(1.0 * FPS))
    assert (lab[dribble] == 1).all()                 # player 0 now sits in slot 1
    assert (lab[int(1.5 * FPS):] == -1).all()        # in flight: nobody carries it


def test_features_one_row_per_tracked_slot():
    m = _match()
    X_, f, s = carrier.features(m, np.arange(10, 20))
    assert X_.shape == (20, len(carrier.FEATURES)) and set(s) == {0, 1}
    near = X_[:, carrier.FEATURES.index("r_ball")] == 0
    assert (s[near] == 0).all()                      # the dribbler is nearest to the ball
