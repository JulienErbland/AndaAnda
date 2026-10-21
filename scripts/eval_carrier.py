"""Held-out check of the carrier model (possession/carrier.py) on the open matches.

    uv run python scripts/eval_carrier.py [--seeds 0 1]

Per match and degrader seed, on graded frames (1 in 5), with the leave-one-match-out model:
- frames with a carrier: is the top slot the carrier, is its team right; ball seen vs missing;
  against simple rules (nearest to the ball; nearest to where the ball was last seen; the
  player who stood on the ball when it was last seen);
- carrier presence: mean summed probability on carrier / no-carrier live / dead frames.
Aggregates only.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import D, X, Match, graded_mask  # noqa: E402
from possession import carrier  # noqa: E402

F = carrier.FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    tot = {}

    def add(key, hit, n=1):
        a = tot.setdefault(key, [0.0, 0])
        a[0] += hit
        a[1] += n

    for path in match_paths(["dfl", "metrica"]):
        clean = Match.load(path)
        for seed in args.seeds:
            seen = degrade(clean, "hackathon", seed=seed + sum(map(ord, clean.name)))
            lab = carrier.labels(clean, seen)
            frames = np.flatnonzero(graded_mask(clean) & (clean.labels != X))[::5]
            view = seen.without_labels()
            X_, f, s = carrier.features(view, frames)
            p = carrier._model_for(view).predict(X_)
            # per frame: argmax slot by model and by each rule
            order = np.lexsort((-p, f))
            f_o, s_o, p_o = f[order], s[order], p[order]
            first = np.r_[True, f_o[1:] != f_o[:-1]]
            top = dict(zip(f_o[first], s_o[first]))
            uf, inv = np.unique(f, return_inverse=True)
            frame_sum = dict(zip(uf, np.bincount(inv, weights=p)))

            def rule(col):
                v = X_[:, F.index(col)]
                ok = v == 0
                return dict(zip(f[ok], s[ok]))

            rules = {"nearest to ball": rule("r_ball"), "nearest to last-seen ball": rule("r_last"),
                     "stood on last-seen ball": rule("r_at_last")}
            ball_seen = ~np.isnan(view.ball[:, 0])
            team = view.slot_team
            for t in uf:
                c = lab[t]
                kind = "seen" if ball_seen[t] else "missing"
                if c >= 0:
                    add(f"model top-1 ({kind})", top.get(t) == c)
                    add(f"model team ({kind})", team[top[t]] == team[c] if t in top else 0)
                    for name, r in rules.items():
                        if t in r:
                            add(f"{name} top-1 ({kind})", r[t] == c)
                            add(f"{name} team ({kind})", team[r[t]] == team[c])
                        else:
                            add(f"{name} top-1 ({kind})", 0)
                    add(f"P(someone) | carrier ({kind})", frame_sum[t])
                elif clean.labels[t] == D:
                    add(f"P(someone) | dead ({kind})", frame_sum[t])
                else:
                    add(f"P(someone) | live, no carrier ({kind})", frame_sum[t])
        print(clean.name, "done", flush=True)

    print()
    for k in sorted(tot):
        h, n = tot[k]
        print("{:<46} {:6.1%}  (n={})".format(k, h / n, n))


if __name__ == "__main__":
    main()
