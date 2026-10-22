import numpy as np

from bench import relabel
from bench.match import FPS, A, D, H, Match


def _match():
    """Home 0 dribbles 1 s, ball flies past away 1 (brushes him 0.12 s), away 2 controls 1 s."""
    n = 4 * FPS
    t = np.arange(n) / FPS
    players = np.zeros((n, 3, 3), dtype=np.float32)
    ball = np.zeros((n, 3), dtype=np.float32)
    x = np.where(t < 1, 30 + 3 * t, np.where(t < 3, 33 + 10 * (t - 1), 53 + 3 * (t - 3)))
    ball[:, 0], ball[:, 1] = x / 105, 34 / 68
    players[:, 0, 0], players[:, 0, 1] = (30 + 3 * np.minimum(t, 1)) / 105, 34 / 68
    players[:, 1, 0], players[:, 1, 1] = 43 / 105, 34.5 / 68          # the ball passes him
    players[:, 2, 0] = np.where(t < 3, 53, 53 + 3 * (t - 3)) / 105       # away 2 receives at 3 s
    players[:, 2, 1] = 34 / 68
    labels = np.full(n, H, np.int8)
    labels[int(1.95 * FPS):int(2.1 * FPS)] = A                         # DFL-style flicker
    labels[3 * FPS:] = A
    labels[:5] = D
    return Match(name="toy", pitch=(105.0, 68.0), ball=ball, players=players,
                 slot_team=np.array([0, 1, 1], np.int8), halves=np.array([[1, 0, n - 1]]),
                 labels=labels)


def test_only_a_real_control_by_the_other_team_changes_possession():
    m = _match()
    lab = relabel.hackathon_labels(m, relabel.Rule(date_from_touch=False))   # v1 dating
    assert (lab[:5] == D).all()                       # dead frames kept from DFL
    assert (lab[5:3 * FPS] == H).all()                # the brush and the flicker change nothing
    first_a = int(np.flatnonzero(lab == A)[0])
    assert 3 * FPS - 3 <= first_a <= 3 * FPS + 3      # changes at away 2's control
    assert (lab[first_a:] == A).all()


def test_v2_dates_the_change_from_the_new_team_s_first_touch():
    m = _match()
    m.players[:, 2, 0] = np.where(np.arange(m.n_frames) / FPS < 2.6, 50.5, np.nan) / 105
    # away 2 now steps in at 50.5 m: the ball (10 m/s from 33 m at 1 s) reaches 2 m of him at
    # ~2.55 s, but he only controls it from 3 s (he is replaced by his moving track then)
    m.players[:, 1, 0] = np.where(np.arange(m.n_frames) / FPS >= 2.6,
                                  np.minimum(53 + 3 * (np.arange(m.n_frames) / FPS - 3), 60), 43) / 105
    m.players[:, 1, 1] = np.where(np.arange(m.n_frames) / FPS >= 2.6, 34 / 68, 34.5 / 68)
    v1 = relabel.hackathon_labels(m, relabel.Rule(date_from_touch=False))
    v2 = relabel.hackathon_labels(m, relabel.Rule(date_from_touch=True))
    c1, c2 = int(np.flatnonzero(v1 == A)[0]), int(np.flatnonzero(v2 == A)[0])
    assert c2 < c1                                    # dated earlier than the control start
    assert (v2[c2:] == A).all() and (v2[5:c2] == H).all()


def test_v2_still_ignores_a_brush_when_the_brushing_team_never_controls():
    m = _match()
    m.players[3 * 25:, 2, 0] = np.nan                 # away 2 never takes the ball
    m.ball[3 * 25:, 0] = np.nan
    lab = relabel.hackathon_labels(m)                 # v2, the default
    assert not (lab == A).any()
