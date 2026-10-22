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


def test_dfl_events_parse_sync_and_snap(tmp_path):
    from bench import dfl_events
    xml = """<?xml version="1.0"?><PutDataRequest>
<Event EventTime="2022-08-26T18:00:00.000+02:00" X-Position="52.5" Y-Position="34.0">
  <KickOff GameSection="firstHalf"><Play Team="T-HOME" Player="P1"><Pass/></Play></KickOff></Event>
<Event EventTime="2022-08-26T18:00:05.000+02:00" X-Position="52.5" Y-Position="34.0"
       CalculatedFrame="10100" CalculatedTimestamp="2022-08-26T18:00:04.000+02:00">
  <Play Team="T-HOME" Player="P1" Evaluation="successfullyCompleted"><Pass/></Play></Event>
<Event EventTime="2022-08-26T18:00:06.000+02:00"><Delete/></Event>
<Event EventTime="2022-08-26T18:00:09.000+02:00" X-Position="40.0" Y-Position="0.0">
  <ThrowIn Team="T-AWAY" DecisionTimestamp="2022-08-26T18:00:07.000+02:00"><Play Team="T-AWAY"/></ThrowIn></Event>
<Event EventTime="2022-08-26T18:50:00.000+02:00" X-Position="52.5" Y-Position="34.0">
  <KickOff GameSection="secondHalf"><Play Team="T-AWAY"><Pass/></Play></KickOff></Event>
</PutDataRequest>"""
    (tmp_path / "e.xml").write_text(xml)
    ev = dfl_events.parse(str(tmp_path / "e.xml"), "T-HOME")
    assert [e["kind"] for e in ev] == ["KickOff", "Play", "ThrowIn", "KickOff"]  # Delete dropped
    assert [e["team"] for e in ev] == [0, 0, 1, 1]
    assert ev[2]["restart"] == "throw_in" and ev[0]["restart"] == "kickoff"

    # 12 s of half 1 at 25 fps (+ a 2nd half): home kicks at 4 s (frame 100 = N 10100),
    # dead from 6.5 s, away throws in at 8.4 s (row 210).
    n1, n2 = 300, 50
    labels = np.array([H] * 162 + [D] * 48 + [A] * 90 + [D] * n2, dtype=np.int8)
    ball = np.zeros((n1 + n2, 3), np.float32)
    ball[:, :2] = (0.5, 0.5)
    ball[105:, 2] = 9.0                                     # kicked at frame 100 (0.2 s lead)
    players = np.full((n1 + n2, 2, 3), np.nan, np.float32)
    players[:, 0, :2] = ball[:, :2]                          # home player on the ball
    m = Match(name="t", pitch=(105.0, 68.0), ball=ball, players=players,
              slot_team=np.array([0, 1], np.int8),
              halves=np.array([[1, 0, n1 - 1], [2, n1, n1 + n2 - 1]]), labels=labels)
    ev[3]["t"] = ev[0]["t"] + 3000  # keep the second half well after the first
    stats = dfl_events.sync(ev, m)
    play = ev[1]
    assert play["anchor_row"] == 100 and stats["lag_s"][0] == 1.0  # logged 1 s late
    assert abs(play["row"] - 100) <= 1 and play["snapped"]
    ev[1]["x"], ev[1]["y"] = 80.0, 10.0                      # logged 35 m away from the ball
    dfl_events.sync(ev, m)
    assert not ev[1]["snapped"]                              # never snap to the wrong place
    assert ev[2]["row"] == 210 and ev[2]["snapped"]          # the dead -> live switch


def test_degraded_positions_are_smooth_like_the_real_tracking():
    """Real ATD jitter: median second difference 0.1 cm (players), 0.9 cm (ball). The added error
    must be a slow offset (tens of cm), not frame-to-frame jitter."""
    clean = _toy_match()
    keep_slots = Profile(blackout_share=0, player_blackouts_per_min=0, player_missing=0,
                         tracklet_s=1e9, wrong_team=0, duplicates_per_min=0,
                         ball_smooth_s=0)   # the deliberate ball smoothing has its own test
    m = degrade(clean, keep_slots, seed=0)
    pitch = np.asarray(m.pitch)
    for added in ((m.players[..., :2] - clean.players[..., :2]) * pitch,
                  (m.ball[:, :2] - clean.ball[:, :2]) * pitch):
        jitter = np.sqrt((np.diff(added, 2, axis=0) ** 2).sum(-1))
        offset = np.sqrt((added ** 2).sum(-1))
        assert np.nanmedian(jitter) < 0.01                       # < 1 cm frame to frame
        assert 0.1 < np.nanmedian(offset) < 0.5                  # but a real position error


def test_metrica_csv_reader_and_events(tmp_path):
    # 2 players per team, 3 frames; player 2 of the away team is a substitute (NaN)
    head = ",,,{0},,{0},,,\n,,,1,,2,,,\nPeriod,Frame,Time [s],P1,,P2,,Ball,\n"
    rows = ["1,{},{:.2f},0.1,0.5,0.2,0.5,0.5,0.5".format(f, f / 25) for f in (1, 2, 3)]
    away_rows = ["1,{},{:.2f},0.9,0.5,NaN,NaN,0.5,0.5".format(f, f / 25) for f in (1, 2, 3)]
    (tmp_path / "h.csv").write_text(head.format("Home") + "\n".join(rows) + "\n")
    (tmp_path / "a.csv").write_text(head.format("Away") + "\n".join(away_rows) + "\n")
    ball, players, slot_team = epts.read_metrica_csv(tmp_path / "h.csv", tmp_path / "a.csv")
    assert ball.shape == (3, 3) and players.shape == (3, 4, 3)
    assert slot_team.tolist() == [0, 0, 1, 1]
    assert players[0, 2, 0] == np.float32(0.9) and np.isnan(players[:, 3, 0]).all()
    (tmp_path / "e.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,"
        "Start X,Start Y,End X,End Y\n"
        "Away,SET PIECE,KICK OFF,1,1,0.04,0,0,P1,,NaN,NaN,NaN,NaN\n"
        "Away,PASS,,1,1,0.04,3,0.12,P1,P2,0.5,0.5,0.6,0.5\n")
    ev = epts.csv_events(tmp_path / "e.csv")
    assert [e["type"]["name"] for e in ev] == ["SET PIECE", "PASS"]
    assert ev[1]["start"]["frame"] == 1 and ev[1]["end"]["frame"] == 3 and ev[1]["to"] == "P2"


def test_ball_smoothing_rounds_kicks_and_keeps_gaps():
    from bench.degrade import smooth_track
    t = np.arange(100)
    xy = np.zeros((100, 2))
    xy[:, 0] = np.where(t < 50, 0.0, (t - 50) * 1.0)       # a kick at frame 50: 25 m/s at once
    xy[70:75] = np.nan
    out = smooth_track(xy, 0.2)
    acc = np.abs(np.diff(out[:, 0], 2))
    assert np.nanmax(acc) < 0.25                           # raw: 1.0 per frame^2 at the kick
    assert np.isnan(out[70:75]).all() and not np.isnan(out[:70]).any()
    assert abs(out[90, 0] - 40.0) < 0.5                    # steady motion is left alone


def test_realism_knobs_switch_off_cleanly_and_work_when_on():
    from dataclasses import replace as _r
    m = _toy_match(n=6000)
    off = Profile(ball_smooth_s=0, ball_false_per_min=0, break_duel_gain=0, gap_from_ballout=False)
    a = degrade(m, off, seed=3)
    b = degrade(m, _r(off, ball_false_in_gaps=0.7, gap_ballout_lead_s=2.0), seed=3)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)            # off = unaffected by the
    assert np.array_equal(a.players, b.players, equal_nan=True)      # dependent sub-knobs
    f = degrade(m, _r(off, ball_false_per_min=20), seed=3)
    step = np.linalg.norm(np.diff(f.ball[:, :2] * np.asarray(f.pitch), axis=0), axis=1)
    step0 = np.linalg.norm(np.diff(a.ball[:, :2] * np.asarray(a.pitch), axis=0), axis=1)
    assert np.nansum(step > 2) > np.nansum(step0 > 2) + 5            # false detections jump
    assert np.array_equal(f.players, a.players, equal_nan=True)      # players untouched


def test_vary_draws_neighbours_of_the_calibrated_profile():
    from bench.degrade import PROFILES, VARIED, vary
    base = PROFILES["hackathon"]
    rng = np.random.default_rng(0)
    draws = [vary(base, rng, 0.25) for _ in range(200)]
    for k in VARIED:
        ratio = np.array([getattr(d, k) for d in draws]) / getattr(base, k)
        assert 0.85 < np.median(ratio) < 1.15 and ratio.std() > 0.1     # centred, really varied
    assert all(d.ball_missing_dead <= 0.9 for d in draws)
    assert vary(base, np.random.default_rng(0), 0.0) == base
