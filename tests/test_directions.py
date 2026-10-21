"""Specifications for the three improvement directions (see tasks/todo.md, "Next directions").

Each test pins down one behaviour of a component that does not exist yet, on a synthetic
situation with a known right answer. They are xfail(strict=True): they report as expected
failures until the component is built, then XPASS turns into a failure, which is the cue to
delete the marker so the behaviour stays protected.

    Direction 1  possession/restarts.py   restart detector: when, where (kind), who
    Direction 2  possession/touches.py    touch detector: control vs deflection vs nothing
    Direction 3  possession/gaps.py       ball-gap reasoning features
"""

import numpy as np
import pytest

from bench.match import FPS, A, H

from .toys import HOME, Toy

TODO_1 = pytest.mark.xfail(strict=True, reason="direction 1: possession/restarts.py not built")
TODO_2 = lambda f: f  # direction 2 built: possession/touches.py
TODO_3 = lambda f: f  # direction 3 built: possession/gaps.py


# -- Direction 1: restart detector ---------------------------------------------------------------
# restarts.detect(match) -> list of objects with .frame (first frame the ball moves), .team (H/A,
# the team of the player at the ball) and .kind in {"kickoff", "throw_in", "corner", "goal_kick",
# "free_kick"}. A restart is a visible ball standing still for >= 1.5 s that then moves.

def _restart_toy(spot, taker, taker_pos, kick_to, still_s=4.0):
    t = Toy(still_s + 4)
    t.jog(0.8, 0, still_s)            # players walking while the ball is dead
    t.jog(3.5, still_s)               # play resumes
    kick_s = max(0.4, float(np.hypot(*np.subtract(kick_to, spot))) / 15.0)  # a 15 m/s kick
    t.ball_still(0, still_s, spot)
    t.ball_move(still_s, still_s + kick_s, spot, kick_to)
    t.ball_move(still_s + kick_s, still_s + 4, kick_to, kick_to)
    t.put(taker, 0, still_s + 0.3, taker_pos)  # after jog(); still at the ball when it moves
    return t.match(), int(still_s * FPS)


@TODO_1
def test_kickoff_is_detected_with_the_kicking_team():
    from possession import restarts
    m, kick = _restart_toy((52.5, 34.0), 9, (52.0, 34.6), (42.0, 34.0))
    found = restarts.detect(m)
    assert len(found) == 1
    r = found[0]
    assert r.kind == "kickoff" and r.team == H and abs(r.frame - kick) <= FPS // 2


@TODO_1
def test_throw_in_is_detected_for_the_thrower():
    from possession import restarts
    m, throw = _restart_toy((40.0, 0.0), 15, (40.0, -0.6), (45.0, 10.0))
    (r,) = restarts.detect(m)
    assert r.kind == "throw_in" and r.team == A and abs(r.frame - throw) <= FPS // 2


@TODO_1
@pytest.mark.parametrize("spot,kind", [((104.5, 0.5), "corner"), ((3.0, 34.0), "goal_kick"),
                                       ((70.0, 40.0), "free_kick")])
def test_restart_kind_follows_the_ball_position(spot, kind):
    from possession import restarts
    taker = 12 if spot[0] > 52.5 else 0
    m, _ = _restart_toy(spot, taker, (spot[0] - 0.6, spot[1]), (spot[0] * 0.8 + 10, 30.0))
    (r,) = restarts.detect(m)
    assert r.kind == kind


@TODO_1
def test_no_restart_in_open_play():
    from possession import restarts
    t = Toy(20).jog(3.5)
    for k in range(10):  # passes every 2 s, the ball never stops
        a = (30 + 4 * k, 20 + (k % 3) * 8)
        b = (34 + 4 * k, 20 + ((k + 1) % 3) * 8)
        t.ball_move(2 * k, 2 * k + 2, a, b)
    assert restarts.detect(t.match()) == []


# -- Direction 2: touch detector -----------------------------------------------------------------
# touches.detect(match) -> list of objects with .frame, .team (H/A) and .kind in {"control",
# "deflection"}. Control: the ball comes within 1.5 m of a player and stays with them (similar
# velocity) for >= 0.4 s; .frame = first frame of that. Deflection: the ball's direction changes by
# more than 30 degrees within 1.5 m of a player, without control following.

def _pass_toy(receiver, receiver_pos, controls=True, deflect_to=None):
    """Home slot 5 passes along y=30 at 15 m/s from x=40; the ball reaches x=55 at t=1 s."""
    t = Toy(6).jog(2.0, slots=[s for s in range(22) if s not in (5, receiver)])
    t.put(5, 0, 6, (39.5, 30.0))
    t.ball_move(0, 1, (40.0, 30.0), (55.0, 30.0))
    if deflect_to is not None:
        t.ball_move(1, 2, (55.0, 30.0), deflect_to)
        t.ball_move(2, 6, deflect_to, deflect_to)
        t.put(receiver, 0, 6, receiver_pos)
    elif controls:
        t.ball_move(1, 4, (55.0, 30.0), (61.0, 30.0))     # carried at 2 m/s
        t.ball_move(4, 6, (61.0, 30.0), (61.0, 30.0))
        t.put(receiver, 0, 1, receiver_pos)
        t.follow(receiver, 1, 6)
    else:
        t.ball_move(1, 2, (55.0, 30.0), (70.0, 30.0))     # keeps flying past
        t.ball_move(2, 6, (70.0, 30.0), (70.0, 30.0))
        t.put(receiver, 0, 6, receiver_pos)
        t.put(7, 0, 2, (70.4, 30.0))
        t.follow(7, 2, 6)
    return t.match(), FPS  # arrival frame at x=55


@TODO_2
def test_interception_is_a_control_by_the_interceptor():
    from possession import touches
    m, arrival = _pass_toy(15, (55.4, 30.0), controls=True)
    controls = [x for x in touches.detect(m) if x.kind == "control" and x.team == A]
    assert len(controls) == 1 and abs(controls[0].frame - arrival) <= 5


@TODO_2
def test_pass_flying_past_an_opponent_is_not_his_touch():
    from possession import touches
    m, _ = _pass_toy(15, (55.0, 31.2), controls=False)   # 1.2 m from the ball's line
    found = touches.detect(m)
    assert not [x for x in found if x.team == A]
    assert [x for x in found if x.team == H and x.kind == "control"]  # received at x=70


@TODO_2
def test_deflection_is_not_a_control():
    from possession import touches
    m, arrival = _pass_toy(15, (55.4, 30.0), deflect_to=(62.0, 42.0))  # ~60 degrees off
    mine = [x for x in touches.detect(m) if x.team == A]
    assert [x.kind for x in mine] == ["deflection"] and abs(mine[0].frame - arrival) <= 5


@TODO_2
def test_short_control_is_still_a_control():
    """The 0.8 s possessions of the public window: a brief but real control counts."""
    from possession import touches
    t = Toy(6).jog(2.0, slots=[s for s in range(22) if s not in (5, 15)])
    t.put(5, 0, 6, (39.5, 30.0))
    t.ball_move(0, 1, (40.0, 30.0), (55.0, 30.0))
    t.ball_move(1, 1.8, (55.0, 30.0), (56.6, 30.0))      # away carries it for 0.8 s
    t.ball_move(1.8, 6, (56.6, 30.0), (56.6, 30.0))
    t.put(15, 0, 1, (55.4, 30.0))
    t.follow(15, 1, 1.8)
    t.put(15, 1.8, 6, (57.5, 31.5))
    t.put(4, 0, 1.0, (59.0, 31.0))
    t.put(4, 1.0, 1.8, (59.0, 31.0), (57.2, 30.3))        # closes in, >= 1.5 m away until 1.8 s
    t.follow(4, 1.8, 6)                                   # home wins it back
    teams = [x.team for x in touches.detect(t.match()) if x.kind == "control"]
    assert A in teams and teams[-1] == H


# -- Direction 3: ball-gap reasoning features -----------------------------------------------------
# gaps.features(match) -> dict of (T,) arrays, constant over each ball gap, NaN (team keys: 0)
# outside gaps:
#   gap_len_s, gap_implied_speed (m/s, last-seen to next-seen position over the gap),
#   gap_lost_team / gap_found_team (+1 home, -1 away, 0 nobody within 2 m),
#   gap_peak_player_speed (mean over the gap of the fastest tracked player's speed),
#   gap_shape_flip (-1 home went from advancing to retreating, +1 the reverse, 0 neither;
#                   from home's centroid forward velocity in the gap's first vs second half)

def _gap_toy(players_speed, lost, found, lost_slot, found_slot, gap=(2.0, 8.0), flip=False):
    t = Toy(10).jog(players_speed)
    if flip:  # before placing the two ball players, so the drift does not move them
        t.drift(0, gap[0], 5.0, +2.0).drift(1, gap[0], 5.0, +2.0)    # home attacks
        t.drift(0, 5.0, gap[1], -2.5).drift(1, 5.0, gap[1], -2.5)    # home retreats
    # No teleports (they would spike the speed channel): the loser carries the ball from t=0 and
    # stays where he lost it; the finder waits all along where the ball reappears.
    t.ball_move(0, gap[0], (lost[0] - 4, lost[1]), lost)
    t.follow(lost_slot, 0, gap[0])
    t.put(lost_slot, gap[0], 10, (lost[0] + 0.4, lost[1]))
    t.ball_lost(*gap)
    t.ball_move(gap[1], 10, found, found)
    t.put(found_slot, 0, 10, (found[0] + 0.4, found[1]))
    return t.match(), slice(int(gap[0] * FPS), int(gap[1] * FPS))


@TODO_3
def test_gap_features_in_a_live_gap_kept_by_the_same_team():
    from possession import gaps
    m, g = _gap_toy(5.0, (40.0, 30.0), (70.0, 30.0), 3, 8)
    f = gaps.features(m)
    assert np.allclose(f["gap_len_s"][g], 6.0, atol=0.1)
    assert np.allclose(f["gap_implied_speed"][g], 5.0, atol=0.3)    # 30 m in 6 s
    assert (f["gap_lost_team"][g] == 1).all() and (f["gap_found_team"][g] == 1).all()
    assert f["gap_peak_player_speed"][g].min() > 4.0
    assert np.isnan(f["gap_len_s"][:g.start]).all()


@TODO_3
def test_gap_features_in_a_dead_gap():
    from possession import gaps
    m, g = _gap_toy(0.6, (40.0, 1.0), (41.0, 0.0), 3, 15)          # walking, ends on the line
    f = gaps.features(m)
    assert f["gap_peak_player_speed"][g].max() < 1.5
    assert f["gap_implied_speed"][g].max() < 0.5
    assert (f["gap_found_team"][g] == -1).all()


@TODO_3
def test_gap_shape_flip_reveals_a_change_of_possession():
    from possession import gaps
    m, g = _gap_toy(3.0, (40.0, 30.0), (35.0, 30.0), 3, 15, flip=True)
    f = gaps.features(m)
    assert (f["gap_shape_flip"][g] == -1).all()
    assert (f["gap_lost_team"][g] == 1).all() and (f["gap_found_team"][g] == -1).all()


def test_toy_builder_sanity():
    """Not a direction test: guards the builder the specs above rely on."""
    t = Toy(4).jog(3.0)
    t.ball_move(0, 2, (40.0, 30.0), (50.0, 30.0)).ball_lost(2, 4)
    m = t.match()
    speeds = m.players[10:90, list(HOME), 2]
    assert np.allclose(speeds, 3.0, atol=0.2)
    assert np.isclose(np.nanmedian(m.ball[5:45, 2]), 5.0, atol=0.2)
    assert np.isnan(m.ball[60:, 0]).all()
