"""Path decoder (possession/path.py) against M2, paired by match; plus a few cost variants.

    uv run python scripts/eval_path.py [--seeds 0] [--hackathon]

Held-out throughout (M2 and carrier pick their leave-one-match-out models). With --hackathon,
also the public-window spellF1 and the possessions per live minute of the real match
(aggregates only). The switch penalty stays M2's: it encodes the hackathon's change rate and is
never fitted to DFL.
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import carrier, learned, path  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

BASE = path.Params()
VARIANTS = {
    "path": BASE,
    "path w_carrier 0.5": replace(BASE, w_carrier=0.5),
    "path w_carrier 2": replace(BASE, w_carrier=2.0),
    "path release/receive 1": replace(BASE, release=1.0, receive=1.0),
    "path release/receive 4": replace(BASE, release=4.0, receive=4.0),
    "path unseen 4": replace(BASE, unseen=4.0),
    "path unseen 16": replace(BASE, unseen=16.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--hackathon", action="store_true")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    rows = {k: [] for k in ["M2"] + args.variants}
    for p in match_paths(["dfl"]):
        clean = Match.load(p)
        for seed in args.seeds:
            seen = degrade(clean, "hackathon", seed=seed + sum(map(ord, clean.name))).without_labels()
            pr, cp = learned.probabilities(seen), carrier.probabilities(seen)
            rows["M2"].append((clean.name, score(clean.labels, learned.decode(seen, probs=pr))))
            for k in args.variants:
                out = path.decode(seen, VARIANTS[k], probs=pr, cprobs=cp)
                rows[k].append((clean.name, score(clean.labels, out)))
        print(clean.name, " ".join("{} {:.4f}".format(k, v[-1][1]["spell_f1_mean"])
                                   for k, v in rows.items()), flush=True)

    ref = {}
    for n, r in rows["M2"]:
        ref.setdefault(n, []).append(r["spell_f1_mean"])
    print("\n{:<26} {:>8} {:>8} {:>18} {:>6}".format("method", "spellF1", "macroF1", "vs M2", "wins"))
    for k, v in rows.items():
        per = {}
        for n, r in v:
            per.setdefault(n, []).append(r["spell_f1_mean"])
        d = np.array([np.mean(per[n]) - np.mean(ref[n]) for n in ref])
        print("{:<26} {:8.4f} {:8.4f} {:>+9.4f} ± {:.4f} {:>4}/{}".format(
            k, np.mean([r["spell_f1_mean"] for _, r in v]), np.mean([r["macro_f1"] for _, r in v]),
            d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))

    if args.hackathon:
        (hp,) = match_paths(["hackathon"])
        hm = Match.load(hp)
        hv = hm.without_labels()
        pr, cp = learned.probabilities(hv), carrier.probabilities(hv)
        for k, out in [("M2", learned.decode(hv, probs=pr))] + [
                (k, path.decode(hv, VARIANTS[k], probs=pr, cprobs=cp)) for k in args.variants]:
            print("hackathon {:<26} public spellF1 {:.3f}  rate {:.2f} / live min".format(
                k, score(hm.labels, out)["spell_f1_mean"], spell_rate(hm, out)))


if __name__ == "__main__":
    main()
