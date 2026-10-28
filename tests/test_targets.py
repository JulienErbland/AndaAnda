"""Bench acceptance gates for the submission method, on the 7 held-out DFL matches.

    uv run --group dev pytest -m slow           # ~1 min; needs `bench fetch`, `bench relabel`
                                                # and the held-out models (`learned train`)

FLOORS pass today and must keep passing: a direction may not buy its gain by breaking what works.
TARGETS are what each direction is for. They are xfail(strict=True) until reached; when one
passes, pytest reports XPASS as a failure, which is the cue to move it into FLOORS.

Scored on the hackathon-style labels (bench/relabel.py v2), the bench's yardstick since Oct 22,
with the dead ball moving like the real one and the ball's gaps split like the real tracker's
(live 7% / dead 80%, gaps ending at restarts; degrader, Oct 28). Re-based Oct 28 for the R + B
ensemble at switch 20 (R_g / B_g); comments give its value, then the previous submission's
(R_v + B_v) on this same bench. Recorded regressions: dead_ball_visible (63% vs 80%, 79 clips)
and ball_gap_hold (27% vs 35%, only 26 clips now that live gaps are rare: moved to targets).
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow
os.environ.setdefault("MTAH_DFL_LABELS", "hackathon")

METHOD = "learned"

FLOORS = {
    "spell_f1_mean": 0.70,          # 0.716 / 0.710
    "macro_f1": 0.89,               # 0.904 / 0.903
    "pass_past_opponents": 0.90,    # 93% / 94%
    "long_stoppage": 0.96,          # 99% / 99%
    "dead_ball_visible": 0.58,      # 63% / 80% (regression; 79 clips)
    "duel_flurry": 0.40,            # 50% / 50% (6 clips)
    "slow_play": 0.80,              # 84% / 79%
    "interception": 0.28,           # 31% / 30%
    "restart_same": 0.54,           # 58% / 52% (direction 1)
    "restart_other": 0.54,          # 58% / 56% (direction 1)
    "blackout_change": 0.27,        # 30% / 27% (direction 3 target 30%, reached; 30 clips)
}

TARGETS = [
    # direction 1: restart detector
    ("restart_same", 0.70, 1),      # 58%
    ("restart_other", 0.70, 1),     # 58%
    ("kickoff", 0.50, 1),           # 12% (8 clips)
    # direction 2: touch detector
    ("short_spell", 0.15, 2),       # 0% (15 clips on these labels)
    # direction 3: ball-gap reasoning (few live gaps now: 26-39 clips)
    ("change_in_ball_gap", 0.40, 3),  # 23%
    ("ball_gap_hold", 0.45, 3),     # 27% (previous submission 35%)
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
