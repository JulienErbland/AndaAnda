import numpy as np

from bench.match import A, H, Match
from possession import tags as T


def test_tag_features_use_the_sign_per_code():
    n = 200
    m = Match(name="toy", pitch=(105.0, 68.0), ball=np.zeros((n, 3), np.float32),
              players=np.zeros((n, 2, 3), np.float32), slot_team=np.array([0, 1], np.int8),
              halves=np.array([[1, 0, n - 1]]))
    tags = [T.Tag(10, 50, "build up", H, ""),                      # home attacks
            T.Tag(60, 100, "defending in middle third", H, ""),    # home DEFENDS: away has it
            T.Tag(110, 150, "ball in the box", A, ""),             # no team information
            T.Tag(160, 170, "set pieces", None, "throw-in")]       # N/A team
    f = T.features(m, tags)
    assert (f["tag_team"][10:50] == 1).all()
    assert (f["tag_team"][60:100] == -1).all()
    assert (f["tag_team"][110:150] == 0).all() and (f["tag_phase"][110:150] == 1).all()
    assert f["setpiece_team"][150] == 0 and abs(f["setpiece_until"][150] - 10 / 25) < 1e-9


def test_one_tag_is_one_piece_of_evidence_however_long():
    n = 1000
    m = Match(name="toy", pitch=(105.0, 68.0), ball=np.zeros((n, 3), np.float32),
              players=np.zeros((n, 2, 3), np.float32), slot_team=np.array([0, 1], np.int8),
              halves=np.array([[1, 0, n - 1]]))
    short = T.bonus(m, [T.Tag(0, 10, "build up", H, "")])
    long_ = T.bonus(m, [T.Tag(0, 500, "build up", H, "")])
    for b in (short, long_):
        assert abs((b[:, H] - b[:, A]).sum() - T.LOGODDS) < 1e-9   # same total, any length
    assert (T.bonus(m, [T.Tag(0, 10, "build up", A, "")])[:10, A] > 0).all()
