"""Edge cases with a known right answer: scenario finders/checks, and HMM behaviour on them."""

import numpy as np

from bench import scenarios
from bench.match import A, D, H, Match
from possession import hmm

S = 25
PITCH = (105.0, 68.0)


def toy(labels, ball_xy, carrier=None, speed=4.0):
    """22 players running at `speed` around the centre; `carrier` = (team, slot) per frame keeps
    one player on the ball. ball_xy (T, 2) in metres, NaN = not tracked."""
    n = len(labels)
    rng = np.random.default_rng(0)
    base = np.array([[30 + 4 * i, 10 + 2.5 * i] for i in range(22)], dtype=float)
    xy = np.repeat(base[None], n, 0) + rng.normal(0, 0.05, (n, 22, 2))
    if carrier is not None:
        for t, slot in enumerate(carrier):
            if slot is not None and not np.isnan(ball_xy[t, 0]):
                xy[t, slot] = ball_xy[t] + [0.4, 0]
    players = np.concatenate([xy / PITCH, np.full((n, 22, 1), speed)], -1).astype(np.float32)
    ball = np.concatenate([ball_xy / PITCH, np.full((n, 1), 3.0)], -1).astype(np.float32)
    return Match(name="toy", pitch=PITCH, ball=ball, players=players,
                 slot_team=np.repeat([0, 1], 11).astype(np.int8),
                 halves=np.array([[1, 0, n - 1]]), labels=np.asarray(labels, dtype=np.int8))


def ctx_of(m):
    return scenarios.context("toy", m, m)


# -- finders and checks ------------------------------------------------------------------------

def test_restart_finders_and_check():
    labels = [H] * 5 * S + [D] * 5 * S + [A] * 5 * S
    m = toy(labels, np.full((len(labels), 2), 50.0))
    ctx = ctx_of(m)
    other, same = scenarios.find_restart_other(ctx), scenarios.find_restart_same(ctx)
    assert len(other) == 1 and len(same) == 0
    clip = other[0]
    assert clip.anchor == 5 * S and clip.info == {"restart": 10 * S, "team": A}
    gt = np.asarray(labels)
    assert scenarios.check_restart(clip, gt, gt)                       # perfect passes
    assert not scenarios.check_restart(clip, gt, np.full(len(gt), H))  # never stopping fails
    late = gt.copy(); late[10 * S:12 * S] = D                          # restart 2 s late fails
    assert not scenarios.check_restart(clip, gt, late)


def test_short_spell_and_interception_finders():
    labels = [A] * 4 * S + [H] * 20 + [A] * 4 * S
    m = toy(labels, np.full((len(labels), 2), 50.0))
    ctx = ctx_of(m)
    short = scenarios.find_short_spell(ctx)
    assert len(short) == 1 and short[0].info["team"] == H
    gt = np.asarray(labels)
    assert scenarios.check_short_spell(short[0], gt, gt)
    assert not scenarios.check_short_spell(short[0], gt, np.full(len(gt), A))  # merged away
    assert scenarios.find_interception(ctx) == []   # 0.8 s spell is too short to be one


def test_transition_tolerance():
    gt = np.array([H] * 100 + [A] * 100)
    clip = scenarios.Clip("toy", 100, 0, 200, {"frm": H, "to": A})
    shifted = np.array([H] * 110 + [A] * 90)
    assert scenarios.check_transition_half_second(clip, gt, shifted)        # 10 f late: ok
    assert not scenarios.check_transition_half_second(clip, gt, np.roll(gt, 20))  # 20 f: no


# -- HMM behaviour on the situations -----------------------------------------------------------

def test_hmm_keeps_possession_through_a_ball_gap():
    n = 16 * S
    ball = np.tile([45.0, 30.0], (n, 1))
    ball[3 * S:13 * S] = np.nan                          # 10 s tracker gap, players still running
    m = toy([H] * n, ball, carrier=[0] * n)
    pred = hmm.decode(m.without_labels())
    assert np.mean(pred[3 * S:13 * S] == H) >= 0.9


def test_hmm_stops_when_the_ball_goes_out():
    n = 12 * S
    ball = np.tile([45.0, 30.0], (n, 1))
    ball[4 * S:9 * S] = [45.0, -3.0]                     # beyond the touchline for 5 s
    m = toy([H] * 4 * S + [D] * 5 * S + [H] * 3 * S, ball, speed=1.0)
    pred = hmm.decode(m.without_labels())
    assert np.mean(pred[5 * S:8 * S] == D) >= 0.9


def test_hmm_finds_an_interception_on_time():
    n = 12 * S
    ball = np.tile([45.0, 30.0], (n, 1))
    carrier = [0] * (6 * S) + [11] * (6 * S)             # home slot 0, then away slot 11
    m = toy([H] * 6 * S + [A] * 6 * S, ball, carrier=carrier)
    pred = hmm.decode(m.without_labels())
    clip = scenarios.Clip("toy", 6 * S, 0, n, {"frm": H, "to": A})
    assert scenarios.check_transition_half_second(clip, m.labels, pred)


# -- learned model guards -----------------------------------------------------------------------

def test_swap_teams_swaps_labels_and_slots():
    from possession import learned
    m = toy([H, H, A, D], np.full((4, 2), 50.0))
    s = learned.swap_teams(m)
    assert s.labels.tolist() == [A, A, H, D]
    assert (s.slot_team == 1 - m.slot_team).all()
    assert m.labels.tolist() == [H, H, A, D]  # original untouched


def test_learned_uses_held_out_model_for_training_matches(monkeypatch):
    import json
    import lightgbm
    from possession import learned
    with open(learned.META) as f:
        trained_on = json.load(f)["trained_on"]
    opened = []
    monkeypatch.setattr(lightgbm, "Booster", lambda model_file: opened.append(model_file))
    learned._model_for(toy([H], np.full((1, 2), 50.0)))             # unseen match
    m = toy([H], np.full((1, 2), 50.0))
    m.name = trained_on[0]
    try:
        learned._model_for(m)                                        # a training match
    except IOError:
        pass  # held-out models not built on this machine: refusing is also correct
    assert opened[0] == learned.FULL_MODEL
    assert all(p != learned.FULL_MODEL for p in opened[1:])


# -- semi-Markov decoder -------------------------------------------------------------------------

def _brute_force_segments(scores, p):
    import itertools
    from possession import hsmm
    trans = hsmm._transitions(p)
    best, arg = -np.inf, None
    for seq in itertools.product(range(3), repeat=len(scores)):
        total, start = 0.0, 0
        for t in range(1, len(seq) + 1):
            if t == len(seq) or seq[t] != seq[start]:
                s, d = seq[start], t - start
                med, sig = ((p.dead_median_s, p.dead_sigma) if s == hsmm.DEAD
                            else (p.alive_median_s, p.alive_sigma))
                total += scores[start:t, s].sum() + hsmm._log_duration(d, p, med, sig)[-1]
                if t < len(seq):
                    total += trans[s, seq[t]]
                start = t
        if total > best:
            best, arg = total, seq
    return arg


def test_segment_viterbi_matches_brute_force():
    from possession import hsmm
    rng = np.random.default_rng(1)
    p = hsmm.Params(switch=1.0, stop=0.5, restart=1.5, w_dur=1.0, block=1,
                    alive_median_s=0.12, dead_median_s=0.2, alive_max_s=1, dead_max_s=1)
    for _ in range(15):
        scores = rng.normal(0, 1.5, (6, 3))
        segs = hsmm.segment_viterbi(scores, p)
        path = tuple(s for a, b, s in segs for _ in range(b - a))
        assert path == _brute_force_segments(scores, p)


def test_gap_weighting_keeps_possession_through_a_weak_dead_lean():
    from possession import hsmm
    n = 40 * S
    scores = np.zeros((n, 3))
    scores[:, 0] = np.log(0.9)
    scores[:, 1] = np.log(0.05)
    scores[:, 2] = np.log(0.05)
    gap = slice(10 * S, 30 * S)                 # 20 s ball gap, slight lean to dead
    scores[gap] = np.log([0.42, 0.03, 0.55])
    ball = np.tile([45.0, 30.0], (n, 1))
    ball[gap] = np.nan
    m = toy([H] * n, ball)
    plain = hsmm.decode_scores(m, scores, hsmm.Params())
    weighted = hsmm.decode_scores(m, scores, hsmm.Params(kappa=50))
    assert np.mean(plain[gap] == D) > 0.5       # the failure mode
    assert np.mean(weighted[gap] == H) == 1.0   # fixed


def test_refinement_puts_the_boundary_on_the_exact_frame():
    from possession import hsmm
    n = 400
    scores = np.full((n, 3), -5.0)
    scores[:203, 0] = 0.0                       # H until frame 203, not a block multiple
    scores[203:, 1] = 0.0
    m = toy([H] * n, np.full((n, 2), 50.0))
    pred = hsmm.decode_scores(m, scores, hsmm.Params(switch=1.0))
    assert (pred[:203] == H).all() and (pred[203:] == A).all()
