"""Bench acceptance gates for the submission method, on the 7 held-out DFL matches.

    uv run --group dev pytest -m slow           # ~1 min; needs `bench fetch`, `bench relabel`
                                                # and the held-out models (`learned train`)

FLOORS pass today and must keep passing: a direction may not buy its gain by breaking what works.
TARGETS are what each direction is for. They are xfail(strict=True) until reached; when one
passes, pytest reports XPASS as a failure, which is the cue to move it into FLOORS.

Scored on the hackathon-style labels (bench/relabel.py v2), the bench's yardstick since Oct 22,
with the dead ball moving like the real one (degrader, Oct 28). Re-based Oct 28 for the
dead-ball-mixed R + B ensemble at switch 20; comments give its value, then the previous
ensemble's (R_r + B_r, parked-dead-ball training) on this same bench. The previous one called
nearly every visible dead ball live here (dead_ball_visible 5%). Two scenarios are below it and
stay recorded as regressions: slow_play (74% vs 81%) and ball_gap_hold (52% vs 56%).
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow
os.environ.setdefault("MTAH_DFL_LABELS", "hackathon")

METHOD = "learned"

FLOORS = {
    "spell_f1_mean": 0.66,          # 0.681 / 0.567
    "macro_f1": 0.87,               # 0.890 / 0.806
    "pass_past_opponents": 0.88,    # 91% / 91%
    "long_stoppage": 0.94,          # 97% / 60%
    "dead_ball_visible": 0.74,      # 78% / 5%
    "duel_flurry": 0.40,            # 50% / 50% (6 clips)
    "slow_play": 0.70,              # 74% / 81% (regression)
    "interception": 0.22,           # 25% / 26%
    "restart_same": 0.50,           # 53% / 20% (direction 1)
    "restart_other": 0.47,          # 50% / 22% (direction 1)
    "ball_gap_hold": 0.48,          # 52% / 56% (regression; direction 3 target 45%, reached)
}

TARGETS = [
    # direction 1: restart detector
    ("restart_same", 0.70, 1),      # 53%
    ("restart_other", 0.70, 1),     # 50%
    ("kickoff", 0.50, 1),           # 12% (8 clips)
    # direction 2: touch detector
    ("short_spell", 0.15, 2),       # 7% (15 clips on these labels)
    # direction 3: ball-gap reasoning
    ("change_in_ball_gap", 0.40, 3),  # 32%
    ("blackout_change", 0.30, 3),   # 19%
]


@pytest.fixture(scope="module")
def bench():
    from bench import scenarios
    from bench.__main__ import match_paths, resolve_method
    from bench.degrade import degrade
    from bench.evaluate import score
    from bench.match import Match
    from possession import learned

    try:
        paths = match_paths(["dfl"])
    except SystemExit:
        pytest.skip("no DFL data: run `python -m bench fetch dfl`")
    if not all(os.path.isdir(os.path.join(d, "lomo")) for d, _ in learned.members()):
        pytest.skip("no held-out models: run `python -m possession.learned train`")

    method = resolve_method(METHOD)
    contexts, preds, results = [], {}, []
    for path in paths:
        clean = Match.load(path)
        seen = degrade(clean, "hackathon", seed=sum(map(ord, clean.name)))
        contexts.append(scenarios.context(clean.name, clean, seen))
        preds[clean.name] = method(seen.without_labels())
        results.append(score(clean.labels, preds[clean.name]))
    table, _ = scenarios.evaluate(contexts, {METHOD: preds})
    rates = {name: (p / t if t else float("nan")) for name, row in table.items()
             for p, t in [row[METHOD]]}
    rates["spell_f1_mean"] = float(np.mean([r["spell_f1_mean"] for r in results]))
    rates["macro_f1"] = float(np.mean([r["macro_f1"] for r in results]))
    return rates


@pytest.mark.parametrize("name,floor", sorted(FLOORS.items()))
def test_floor(bench, name, floor):
    assert bench[name] >= floor, "{} fell to {:.3f} (floor {:.2f})".format(name, bench[name], floor)


@pytest.mark.parametrize("name,target,direction", TARGETS)
def test_target(bench, name, target, direction, request):
    request.node.add_marker(pytest.mark.xfail(
        strict=True, reason="direction {} target: {} >= {:.0%}".format(direction, name, target)))
    assert bench[name] >= target, "{} at {:.0%}, target {:.0%}".format(name, bench[name], target)
