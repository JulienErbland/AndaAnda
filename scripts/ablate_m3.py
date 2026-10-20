"""M3 ablation: M2 Viterbi vs the semi-Markov decoder with/without duration priors and gap weighting.

    uv run python scripts/ablate_m3.py

Held-out probabilities are computed once (leave-one-match-out on DFL, full model on the
hackathon match). Reports the DFL board, the public window, the predicted possession rate on the
full hackathon match (count only) and the pass rate of the scenarios M3 targets.
"""

import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import scenarios  # noqa: E402
from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import hsmm, learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

KEY = ["ball_gap_hold", "change_in_ball_gap", "short_spell", "restart_other", "interception",
       "dead_ball_visible"]
BASE = hsmm.Params()
CONFIGS = [("A  M2 viterbi", None)]
CONFIGS += [("B  gap k={}".format(k), replace(BASE, kappa=k)) for k in (25, 50, 100, 200)]
CONFIGS += [("C  dur w={} med={}".format(w, m), replace(BASE, w_dur=w, alive_median_s=m))
            for w in (1, 3) for m in (4, 8)]
CONFIGS += [("D  k={} w={} med={}".format(k, w, m), replace(BASE, kappa=k, w_dur=w,
                                                            alive_median_s=m))
            for k in (50, 100) for w in (1, 3) for m in (4, 8)]


def decode(m, probs, p):
    if p is None:
        return learned.decode(m, probs=probs)
    return hsmm.decode_scores(m, np.log(np.clip(probs, 1e-4, 1.0)), p)


def main():
    dfl = []
    for path in match_paths(["dfl"]):
        clean = Match.load(path)
        seen = degrade(clean, "hackathon", seed=sum(map(ord, clean.name)))
        dfl.append((scenarios.context(clean.name, clean, seen),
                    learned.probabilities(seen.without_labels())))
    (hpath,) = match_paths(["hackathon"])
    hm = Match.load(hpath)
    hprobs = learned.probabilities(hm.without_labels())
    clips = {sc.name: [c for ctx, _ in dfl for c in sc.find(ctx)]
             for sc in scenarios.SCENARIOS if sc.name in KEY}
    checks = {sc.name: sc.check for sc in scenarios.SCENARIOS}

    print("{:<22} {:>7} {:>7} {:>7} {:>5}  ".format("config", "DFL sp", "macro", "public",
                                                   "rate") + " ".join(
        "{:>8}".format(k[:8]) for k in KEY) + "   sec")
    for name, p in CONFIGS:
        t0 = time.time()
        preds = {ctx.name: decode(ctx.seen, pr, p) for ctx, pr in dfl}
        res = [score(ctx.gt, preds[ctx.name]) for ctx, _ in dfl]
        hpred = decode(hm, hprobs, p)
        gts = {ctx.name: ctx.gt for ctx, _ in dfl}
        rates = [np.mean([checks[k](c, gts[c.match], preds[c.match]) for c in clips[k]])
                 for k in KEY]
        print("{:<22} {:>7.4f} {:>7.4f} {:>7.4f} {:>5.2f}  ".format(
            name, np.mean([r["spell_f1_mean"] for r in res]), np.mean([r["macro_f1"] for r in res]),
            score(hm.labels, hpred)["spell_f1_mean"], spell_rate(hm, hpred)) + " ".join(
            "{:>8.0%}".format(r) for r in rates) + "  {:>4.0f}".format(time.time() - t0),
            flush=True)


if __name__ == "__main__":
    main()
