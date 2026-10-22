"""Where do the public labels put a hand-over, relative to what the tracking shows?

    uv run python scripts/change_timing.py            # public labels on the real tracking
    uv run python scripts/change_timing.py --bench    # also DFL / v1 / v2 labels on degraded DFL

For each H <-> A change in the public window: signed time to the nearest control start and to
the nearest touch of any kind (control, deflection, contact) by the new team, in the real
tracking. Prints only summary statistics (count, median, quartiles, share within 0.5 s).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.match import FPS, A, H, X, Match  # noqa: E402
from possession import touches  # noqa: E402


def offsets(match, labels, view):
    """Per H <-> A change: signed seconds to the nearest control start / any touch by the new team."""
    gt = labels.astype(np.int64)
    ts = touches.detect(view)
    out = {"control start": [], "any touch": []}
    for c in np.flatnonzero(np.diff(gt)) + 1:
        if {gt[c - 1], gt[c]} != {H, A}:
            continue
        for name, kinds in (("control start", {"control"}),
                            ("any touch", {"control", "deflection", "contact"})):
            fr = [t.frame for t in ts if t.team == gt[c] and t.kind in kinds]
            if fr:
                out[name].append((min(fr, key=lambda f: abs(f - c)) - c) / FPS)
    return out


def report(label, offs):
    for name, d in offs.items():
        d = np.array(d)
        print("{:<26} {:<14} n {:5d}  median {:+.2f} s  quartiles {:+.2f} / {:+.2f} s  within 0.5 s {:.0%}"
              .format(label, name, len(d), np.median(d), *np.percentile(d, [25, 75]),
                      np.mean(np.abs(d) <= 0.5)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    hm = Match.load(match_paths(["hackathon"])[0])
    report("public labels (real)", offsets(hm, hm.labels, hm.without_labels()))
    if not args.bench:
        return
    from bench import relabel
    from bench.degrade import degrade
    pooled = {}
    for p in relabel.dfl_paths():
        c = Match.load(p)
        view = degrade(c, "hackathon", seed=sum(map(ord, c.name))).without_labels()
        for name, lab in (("DFL original", c.labels),
                          ("relabel v1 (control start)", relabel.hackathon_labels(c)),
                          ("relabel v2 (first touch)", relabel.hackathon_labels(
                              c, relabel.Rule(date_from_touch=True)))):
            for k, v in offsets(c, lab, view).items():
                pooled.setdefault(name, {}).setdefault(k, []).extend(v)
    for name, offs in pooled.items():
        report(name, offs)


if __name__ == "__main__":
    main()
