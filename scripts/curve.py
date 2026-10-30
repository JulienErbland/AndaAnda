"""Bench curves and paired comparisons from compare_models.py eval JSONs, the MATCH as the unit
(seeds averaged within each match first).

    uv run python scripts/curve.py A.json [B.json ...]              # spellF1 per switch: mean + per match
    uv run python scripts/curve.py --vs BASE.json:20 X.json:30      # paired per-match gain, wins
"""
import json
import sys
from collections import defaultdict

import numpy as np


def per_match(path):
    """{switch: {match: mean spellF1 over seeds}}, {switch: {match: mean rate}}."""
    rows = json.load(open(path))["dfl"]
    acc, rate = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for name, _seed, sw, spell, _macro, r in rows:
        acc[float(sw)][name].append(spell)
        rate[float(sw)][name].append(r)
    return ({s: {m: float(np.mean(v)) for m, v in d.items()} for s, d in acc.items()},
            {s: {m: float(np.mean(v)) for m, v in d.items()} for s, d in rate.items()})


def curve(path):
    sp, rt = per_match(path)
    real = json.load(open(path)).get("hackathon", {})
    names = sorted(next(iter(sp.values())))
    print(path)
    print("  switch   mean   bench-rate  real-rate | " + " ".join(n[-6:] for n in names))
    for s in sorted(sp):
        rr = real.get(str(s), real.get(str(int(s)), {})).get("rate", float("nan"))
        print("  {:>6g}  {:.4f}   {:.2f}       {:.2f}    | {}".format(
            s, np.mean(list(sp[s].values())), np.mean(list(rt[s].values())), rr,
            " ".join("{:.3f} ".format(sp[s][n]) for n in names)))


def paired(base, other):
    (bp, bs), (op, os_) = (x.rsplit(":", 1) for x in (base, other))
    b, o = per_match(bp)[0][float(bs)], per_match(op)[0][float(os_)]
    names = sorted(set(b) & set(o))
    d = np.array([o[n] - b[n] for n in names])
    print("{} vs {}: mean {:+.4f} (sd {:.4f}, se {:.4f}), better on {}/{}; per match {}".format(
        other, base, d.mean(), d.std(ddof=1), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()),
        len(d), " ".join("{:+.3f}".format(x) for x in d)))
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--vs":
        for x in a[2:]:
            paired(a[1], x)
    else:
        for p in a:
            curve(p)
