import numpy as np

from bench.match import Match
from possession import ballfill


def test_gap_features_interpolate_between_sightings():
    n = 50
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[:, 0] = np.linspace(0.2, 0.6, n)           # straight run along x
    ball[:, 1] = 0.5
    ball[20:30, :2] = np.nan                         # a 10-frame gap
    players = np.full((n, 2, 3), np.nan, dtype=np.float32)
    players[:, 0, :2], players[:, 1, :2] = (0.3, 0.4), (0.5, 0.6)
    match = Match(name="toy", pitch=(105.0, 68.0), ball=ball, players=players,
                  slot_team=np.array([0, 1], np.int8), halves=np.array([[1, 0, n - 1]]))
    frames = ballfill.gap_frames(match)
    assert frames.tolist() == list(range(20, 30))
    X_, I = ballfill.features(match, frames, cprobs=np.zeros((n, 2), np.float32))
    assert X_.shape == (10, len(ballfill.FEATURES))
    truth = np.linspace(0.2, 0.6, n)[frames] * 105
    assert np.allclose(I[:, 0], truth, atol=0.05)    # the straight line is the baseline
    lvx = X_[:, ballfill.FEATURES.index("lvx")]
    assert np.allclose(lvx, (0.4 / (n - 1)) * 105 * 25, rtol=0.01)   # arriving speed, m/s
