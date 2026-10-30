from dataclasses import replace

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
    lab, ext = reception.dead_until_reception(m, m.labels)
    first_live = int(np.flatnonzero(lab != D)[0])
    # the ball is within 1.5 m of home 1 from 42 - 1.5 = 40.5 m, i.e. at 2.05 s
    assert abs(first_live - int(2.05 * FPS)) <= 1
    assert (lab[first_live:] == H).all() and len(ext) == 1


def test_no_reception_keeps_dfl_s_restart():
    m = _throw_in(receiver=False)
    lab, ext = reception.dead_until_reception(m, m.labels)
    assert (lab == m.labels).all() and len(ext) == 0
    assert not (lab == A).any()


def test_no_ball_at_the_restart_leaves_it_alone():
    m = _throw_in()
    m.ball[FPS - 2:FPS + 5] = np.nan                  # the tracker loses the ball at the throw
    lab, ext = reception.dead_until_reception(m, m.labels)
    assert (lab == m.labels).all() and len(ext) == 0


def test_methods_never_see_the_ball_status_and_the_degrader_uses_it():
    from bench.degrade import PROFILES, degrade
    m = _throw_in()
    lab, _ = reception.dead_until_reception(m, m.labels)
    rx = replace(m, labels=lab, ball_dead=m.labels == D)
    assert rx.without_labels().ball_dead is None
    # the ball is hidden by the real status, so the extended frames are degraded as live play
    a = degrade(replace(m, labels=lab), PROFILES["hackathon"], seed=0)
    b = degrade(rx, PROFILES["hackathon"], seed=0)
    c = degrade(m, PROFILES["hackathon"], seed=0)
    assert np.array_equal(np.isnan(b.ball[:, 0]), np.isnan(c.ball[:, 0]))
    # ... and moved by it too (F0's dead-ball motion): the kick's flight is left as tracked
    assert np.array_equal(b.ball, c.ball, equal_nan=True)
    assert not np.array_equal(np.isnan(a.ball[:, 0]), np.isnan(c.ball[:, 0]))
