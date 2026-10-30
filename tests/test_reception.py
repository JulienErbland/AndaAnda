import numpy as np

from bench import reception
from bench.match import FPS, A, D, H, Match


def _throw_in(receiver=True):
    """Dead for 1 s, home 0 throws at 1 s, the ball flies 10 m/s to home 1 (12 m away) at 2.2 s."""
    n = 4 * FPS
    t = np.arange(n) / FPS
    players = np.zeros((n, 3, 3), dtype=np.float32)
    ball = np.zeros((n, 3), dtype=np.float32)
    x = np.clip(30 + 10 * (t - 1), 30, 42)
    ball[:, 0], ball[:, 1] = x / 105, 0.5 / 68
    players[:, 0, 0], players[:, 0, 1] = 30 / 105, 0.2 / 68
    players[:, 1, 0], players[:, 1, 1] = (42 if receiver else 70) / 105, 1 / 68
    players[:, 2, 0], players[:, 2, 1] = 60 / 105, 30 / 68
    labels = np.full(n, H, np.int8)
    labels[:FPS] = D
    return Match(name="toy", pitch=(105.0, 68.0), ball=ball, players=players,
                 slot_team=np.array([0, 0, 1], np.int8), halves=np.array([[1, 0, n - 1]]),
                 labels=labels)


def test_dead_ball_lasts_until_another_player_receives():
    m = _throw_in()
    lab, ext, missed = reception.reception_labels(m)
    first_live = int(np.flatnonzero(lab != D)[0])
    # the ball is within 1.5 m of home 1 from 42 - 1.5 = 40.5 m, i.e. at 2.05 s
    assert abs(first_live - int(2.05 * FPS)) <= 1
    assert (lab[first_live:] == H).all() and missed == 0 and len(ext) == 1


def test_no_reception_keeps_dfl_s_restart():
    m = _throw_in(receiver=False)
    lab, ext, missed = reception.reception_labels(m)
    assert (lab == m.labels).all() and missed == 1 and len(ext) == 0
    assert not (lab == A).any()
