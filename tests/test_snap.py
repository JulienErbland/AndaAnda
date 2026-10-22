import numpy as np

from bench.match import FPS, A, D, H
from possession import snap
from tests.test_relabel import _match   # home dribbles, ball flies, away 2 controls from 3 s


def test_a_late_hand_over_moves_to_the_control_frame():
    m = _match().without_labels()
    labels = np.full(m.n_frames, H, np.int8)
    labels[:5] = D
    labels[int(3.8 * FPS):] = A                       # M2 switched 0.8 s late
    out = snap.refine(m, labels)
    first_a = int(np.flatnonzero(out == A)[0])
    assert abs(first_a - 3 * FPS) <= 3                # now at away 2's control
    assert (out[:5] == D).all() and (out[5:first_a] == H).all() and (out[first_a:] == A).all()


def test_no_control_nearby_leaves_the_change_alone():
    m = _match().without_labels()
    labels = np.full(m.n_frames, H, np.int8)
    labels[int(1.2 * FPS):] = A                       # nobody of away controls near 1.2 s
    out = snap.refine(m, labels, window_s=0.5)
    assert (out == labels).all()
