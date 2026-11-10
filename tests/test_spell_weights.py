import numpy as np

from bench.match import FPS, A, D, H
from possession.learned import spell_weights


def test_short_runs_weigh_more_and_total_weight_per_short_run_is_equal():
    lab = np.r_[np.full(20 * FPS, H), np.full(2 * FPS, A), np.full(5 * FPS, H), np.full(10, D)]
    w = spell_weights(lab, t0_s=10.0, cap=8.0)
    assert len(w) == len(lab)
    assert np.all(w[:20 * FPS] == 1.0)                       # longer than t0: untouched
    assert np.allclose(w[20 * FPS:22 * FPS], 5.0)            # 2 s run: 10 / 2
    assert np.allclose(w[22 * FPS:27 * FPS], 2.0)
    assert np.all(w[-10:] == 8.0)                            # capped
    assert np.isclose(w[20 * FPS:22 * FPS].sum(), w[22 * FPS:27 * FPS].sum())
