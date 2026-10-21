"""Bench acceptance gates for the submission method, on the 7 held-out DFL matches.

    uv run --group dev pytest -m slow           # ~1 min; needs `bench fetch` + `learned train`

FLOORS pass today and must keep passing: a direction may not buy its gain by breaking what works.
TARGETS are what each direction is for. They are xfail(strict=True) until reached; when one
passes, pytest reports XPASS as a failure, which is the cue to move it into FLOORS.

Current values (M2 learned, Oct 20) are in the comments; floors sit just below them so that a
retrained model's small wobble does not fail them, while a real regression does.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

METHOD = "learned"

FLOORS = {
    "spell_f1_mean": 0.53,          # 0.550 (M2 + touches, Oct 21)
    "macro_f1": 0.87,               # 0.890
    "pass_past_opponents": 0.92,    # 95%
    "long_stoppage": 0.96,          # 99%
    "dead_ball_visible": 0.88,      # 93%
    "duel_flurry": 0.70,            # 74%
    "slow_play": 0.74,              # 83%
    "interception": 0.37,           # 40%: direction 2 target reached Oct 21
}

TARGETS = [
    # direction 1: restart detector
    ("restart_same", 0.70, 1),      # 56%
    ("restart_other", 0.70, 1),     # 55%
    ("kickoff", 0.50, 1),           # 12% (8 clips)
    # direction 2: touch detector
    ("short_spell", 0.15, 2),       # 8%
    # direction 3: ball-gap reasoning
    ("change_in_ball_gap", 0.40, 3),  # 25%
    ("ball_gap_hold", 0.45, 3),       # 23%
    ("blackout_change", 0.30, 3),     # 14%
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
    if not os.path.isdir(learned.LOMO_DIR):
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
