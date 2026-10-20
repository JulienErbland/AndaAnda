import numpy as np

from bench import dfl, epts, import_starter
from bench.degrade import Profile, degrade
from bench.evaluate import diagnose
from bench.match import A, D, H, X, Match, graded_mask
from possession import baseline

INFO = """<?xml version="1.0"?><PutDataRequest><MatchInformation>
<Environment PitchX="105.00" PitchY="68.00"/>
<Teams><Team TeamId="T-HOME" Role="home"/><Team TeamId="T-AWAY" Role="guest"/></Teams>
</MatchInformation></PutDataRequest>"""


def _frames(n0, rows, ball=False):
    out = []
    for i, (x, y, *flags) in enumerate(rows):
        extra = ' BallPossession="{}" BallStatus="{}"'.format(*flags) if ball else ""
        out.append('<Frame N="{}" X="{}" Y="{}" S="36.0"{}/>'.format(n0 + i, x, y, extra))
    return "".join(out)


def test_dfl_parse_labels_and_layout(tmp_path):
    ball1 = [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 2, 1), (3, 0, 2, 0)]
    ball2 = [(0, 0, 2, 0), (0, 0, 2, 1)]
    xml = ('<?xml version="1.0"?><PutDataRequest><Positions>'
           '<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">{}</FrameSet>'
           '<FrameSet GameSection="firstHalf" TeamId="T-AWAY" PersonId="P2">{}</FrameSet>'
           '<FrameSet GameSection="firstHalf" TeamId="T-HOME" PersonId="P1">{}</FrameSet>'
           '<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="R1">{}</FrameSet>'
           '<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">{}</FrameSet>'
           '</Positions></PutDataRequest>').format(
        _frames(10000, ball1, ball=True), _frames(10001, [(-52.5, 34)]),
        _frames(10000, [(52.5, -34)] * 4), _frames(10000, [(0, 0)] * 4),
        _frames(100000, ball2, ball=True))
    (tmp_path / "i.xml").write_text(INFO)
    (tmp_path / "p.xml").write_text(xml)
    m = dfl.parse(str(tmp_path / "i.xml"), str(tmp_path / "p.xml"), "t")

    gap = dfl.HALFTIME_GAP
    assert m.halves.tolist() == [[1, 0, 3], [2, 4 + gap, 5 + gap]]
    assert m.labels[:4].tolist() == [H, H, A, D]
    assert (m.labels[4:4 + gap] == X).all()
    assert m.labels[-2:].tolist() == [D, A]
    assert m.slot_team.tolist() == [0, 1]                    # home first; referee dropped
    np.testing.assert_allclose(m.players[0, 0, :2], [1, 1])  # (+52.5, -34) m -> bottom-right
    np.testing.assert_allclose(m.players[1, 1, :2], [0, 0])  # away player, top-left corner
    assert np.isnan(m.players[0, 1, 0])                      # not on the pitch yet
    np.testing.assert_allclose(m.ball[0, 2], 10.0)           # 36 km/h -> 10 m/s


def test_metrica_labels_from_events():
    def ev(kind, team, start, end, sub=None, period=1):
        return {"type": {"name": kind}, "team": {"id": team}, "period": period,
                "subtypes": sub, "start": {"frame": start}, "end": {"frame": end}}
    events = [ev("SET PIECE", "HOME", 11, 11, {"name": "KICK OFF"}),
              ev("PASS", "HOME", 11, 20), ev("BALL LOST", "HOME", 21, 22),
              ev("RECOVERY", "AWAY", 25, 25), ev("BALL OUT", "AWAY", 26, 30),
              ev("SET PIECE", "HOME", 41, 41, {"name": "THROW IN"}), ev("PASS", "HOME", 41, 50),
              ev("PASS", "AWAY", 60, 70, period=2)]
    labels, halves = epts.metrica_labels(events, 80, "HOME")
    assert halves.tolist() == [[1, 10, 49], [2, 59, 69]]
    assert (labels[:10] == X).all()
    assert (labels[10:24] == H).all()   # BALL LOST does not hand over; RECOVERY does
    assert (labels[24:29] == A).all()
    assert (labels[29:40] == D).all()   # from the frame the ball crosses the line to the restart
    assert (labels[40:50] == H).all()
    assert (labels[59:70] == A).all()


def _toy_match(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.where(np.arange(n) % 1000 < 400, D, np.where(np.arange(n) % 2000 < 1000, H, A))
    walk = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22, 2)), axis=0), 0, 1)
    players = np.concatenate([walk, np.full((n, 22, 1), 3.0)], -1).astype(np.float32)
    ball = np.concatenate([players[:, 0, :2], np.full((n, 1), 8.0)], -1).astype(np.float32)
    return Match(name="toy", pitch=(105.0, 68.0), ball=ball, players=players,
                 slot_team=np.repeat([0, 1], 11).astype(np.int8),
                 halves=np.array([[1, 0, n - 1]]), labels=labels.astype(np.int8))


def test_degrade_hits_targets_and_keeps_labels():
    m = _toy_match()
    # Short gaps: the toy's 16-24 s label blocks are too short for the calibrated ~10 s gaps to
    # hit separate alive/dead targets (on real DFL they do: see `python -m bench stats dfl`).
    p = Profile(ball_missing_alive=0.2, ball_missing_dead=0.7, ball_gap_median_s=1.0,
                ball_flicker_share=0)
    d = degrade(m, p, seed=1)
    g, dead, nb = graded_mask(d), d.labels == D, np.isnan(d.ball[:, 0])
    assert abs(nb[g & ~dead].mean() - 0.2) < 0.03
    assert abs(nb[g & dead].mean() - 0.7) < 0.03
    assert np.array_equal(d.labels, m.labels)
    vis = (~np.isnan(d.players[..., 0])).sum(1).mean()
    assert 19.5 < vis < 21.5
    again = degrade(m, p, seed=1)
    assert np.array_equal(np.isnan(again.players), np.isnan(d.players))


def test_merge_of_a_short_spell_is_diagnosed_as_merged():
    # GT: A20.0 H0.8 A16.0 (the pattern from the public window); prediction merges it.
    gt = np.array([A] * 500 + [H] * 20 + [A] * 400, dtype=np.int8)
    m = _toy_match(len(gt))
    m.labels = gt
    pred = np.full(len(gt), A, dtype=np.int8)
    causes = diagnose(m, pred)["miss_causes"]
    assert causes["merged"] == 1             # A16.0 lost: its predicted spell also covers A20.0
    assert causes["wrong_team"] == 1         # H0.8 predicted as A


def test_baseline_smoothing_matches_organisers():
    import_starter()
    import importlib.util
    import os
    from bench import STARTER_DIR
    spec = importlib.util.spec_from_file_location(
        "starter_baseline", os.path.join(STARTER_DIR, "starter", "baseline.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    codes = np.random.default_rng(0).integers(-1, 2, 5000).astype(np.int8)
    for window in (0, 25, 50):
        letters = ref.smooth(codes, window)
        mine = baseline.smooth(codes, window)
        assert np.array_equal(np.array(["H", "A", "D"])[mine], letters)


def test_viterbi_is_optimal_and_ignores_flicker():
    import itertools
    from possession import hmm
    rng = np.random.default_rng(0)
    trans = hmm.transitions(hmm.Params(switch=1.5, stop=1.0, restart=2.0))
    for _ in range(20):  # exact against brute force on short random sequences
        scores = rng.normal(0, 1, (6, 3))
        best = max(itertools.product(range(3), repeat=6), key=lambda s: sum(
            scores[t, s[t]] + (trans[s[t - 1], s[t]] if t else 0) for t in range(6)))
        assert tuple(hmm.viterbi(scores, trans)) == best
    # a 3-frame flicker of contrary evidence is not worth two switches; 30 frames are
    scores = np.zeros((100, 3))
    scores[:, 0] = 1
    scores[40:43] = [0, 2, 0]
    scores[70:] = [0, 2, 0]
    path = hmm.viterbi(scores, hmm.transitions(hmm.Params(switch=5)))
    assert (path[:70] == 0).all() and (path[70:] == 1).all()
