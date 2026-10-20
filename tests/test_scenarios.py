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
