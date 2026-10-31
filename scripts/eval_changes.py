"""Review D: learned switch costs vs the constant penalty, for the submission model.

    MTAH_DFL_LABELS=hackathon uv run python scripts/eval_changes.py bench
    uv run python scripts/eval_changes.py pff --beta B --switch S      # evaluation-only
    uv run python scripts/eval_changes.py real --beta B --switch S     # aggregates only

Bench: 7 DFL matches x 3 degrader seeds, held-out evidence and change models, paired per match
against beta 0 at the first switch.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import changes, learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

BETAS = (0.0, 0.5, 1.0, 2.0, 4.0)
SWITCHES = (14.0, 20.0, 30.0)


def _one(args):
    path, seed, betas, switches = args
    clean = Match.load(path)
    view = load_degraded(path, "hackathon", seed).without_labels()
    pr, q = learned.probabilities(view), changes.probability(view)
    out = {"name": clean.name}
    for b in betas:
        for sw in switches:
            pred = changes.decode(view, learned.Params(switch=sw), pr, b, q)
            r = score(clean.labels, pred)
            out[(b, sw)] = (r["spell_f1_mean"], spell_rate(clean, pred), r["macro_f1"])
    out["seed"] = seed
    return out


def run(groups, seeds, betas, switches, out=None):
    jobs = [(p, s, betas, switches) for p in match_paths(groups) for s in seeds]
    with Pool(4) as pool:
        res = list(pool.imap(_one, jobs))
    if out:   # one compare_models-style JSON per beta (scripts/curve.py reads them)
        for b in betas:
            rows = [[r["name"], r["seed"], sw, r[(b, sw)][0], r[(b, sw)][2], r[(b, sw)][1]]
                    for r in res for sw in switches]
            with open("{}_b{:g}.json".format(out, b), "w") as f:
                json.dump({"dfl": rows, "hackathon": {}}, f)
    names = sorted({r["name"] for r in res})
    per = lambda k: np.array([np.mean([r[k][0] for r in res if r["name"] == n]) for n in names])  # noqa: E731
    ref = (0.0, switches[0])
    base = per(ref)
    print("{} matches x {} seeds; reference beta 0, switch {:g}: {:.4f}".format(len(names), len(seeds), ref[1], base.mean()))
    for b in betas:
        for sw in switches:
            d = per((b, sw)) - base
            print("beta {:>3g} switch {:>3g}: spellF1 {:.4f}  {:+.4f} ± {:.4f}, better on {}/{} | rate {:.2f}".format(
                b, sw, per((b, sw)).mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()),
                len(d), np.mean([r[(b, sw)][1] for r in res])))


def real(beta, switch):
    hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
    pr, q = learned.probabilities(hm), changes.probability(hm)
    for b in (0.0, beta):
        pred = changes.decode(hm, learned.Params(switch=switch), pr, b, q)
        print("real match, beta {} switch {}: rate {:.2f}".format(b, switch, spell_rate(hm, pred)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("where", choices=["bench", "pff", "real"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--switch", type=float, default=20.0)
    ap.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    ap.add_argument("--switches", type=float, nargs="+", default=list(SWITCHES), help="first = reference")
    ap.add_argument("--out", help="bench: write <out>_b<beta>.json per beta")
    a = ap.parse_args()
    if a.where == "bench":
        run(["dfl"], [0, 1, 2], a.betas, a.switches, a.out)
    elif a.where == "pff":
        run(["pff"], [0], (0.0, a.beta), (a.switch,), a.out)
    else:
        real(a.beta, a.switch)
