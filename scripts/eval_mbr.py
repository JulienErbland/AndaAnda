"""Review C: MBR decoding vs Viterbi at switch 20, for the submission model.

    MTAH_DFL_LABELS=hackathon uv run python scripts/eval_mbr.py bench [--temps 1 2 4 8]
    uv run python scripts/eval_mbr.py pff --temp 4      # evaluation-only, 64 matches, seed 0
    uv run python scripts/eval_mbr.py real --temp 4     # real rate + chosen switches (aggregates)

Bench: 7 DFL matches x 3 degrader seeds, held-out probabilities computed once per match, paired
per match. Prints the switch MBR picked per half (how often each candidate wins).
"""

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import learned, mbr  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402


def _one(args):
    path, seed, temps = args
    clean = Match.load(path)
    view = load_degraded(path, "hackathon", seed).without_labels()
    pr = learned.probabilities(view)
    out = {"name": clean.name, "viterbi": score(clean.labels, learned.decode(view, learned.Params(), pr))["spell_f1_mean"]}
    for t in temps:
        ch = []
        pred = mbr.decode(view, probs=pr, temp=t, chosen=ch)
        out[t] = (score(clean.labels, pred)["spell_f1_mean"], ch, spell_rate(clean, pred))
    return out


def run(groups, seeds, temps):
    jobs = [(p, s, temps) for p in match_paths(groups) for s in seeds]
    with Pool(4) as pool:
        res = list(pool.imap(_one, jobs))
    names = sorted({r["name"] for r in res})
    base = {n: np.mean([r["viterbi"] for r in res if r["name"] == n]) for n in names}
    print("Viterbi switch 20: spellF1 {:.4f} ({} matches x {} seeds)".format(np.mean(list(base.values())), len(names), len(seeds)))
    for t in temps:
        per = {n: np.mean([r[t][0] for r in res if r["name"] == n]) for n in names}
        d = np.array([per[n] - base[n] for n in names])
        ch = Counter(sw for r in res for sw in r[t][1])
        print("MBR temp {:>4g}: spellF1 {:.4f}  vs Viterbi {:+.4f} ± {:.4f}, better on {}/{} | rate {:.2f} | switches picked {}".format(
            t, np.mean(list(per.values())), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d),
            np.mean([r[t][2] for r in res]), dict(sorted(ch.items()))))


def real(temp):
    hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
    pr = learned.probabilities(hm)
    ch = []
    pred = mbr.decode(hm, probs=pr, temp=temp, chosen=ch)
    print("real match, MBR temp {}: rate {:.2f} (Viterbi 20: {:.2f}); switch picked per half {}".format(
        temp, spell_rate(hm, pred), spell_rate(hm, learned.decode(hm, learned.Params(), pr)), ch))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("where", choices=["bench", "pff", "real"])
    ap.add_argument("--temps", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--temp", type=float, default=4)
    a = ap.parse_args()
    if a.where == "bench":
        run(["dfl"], [0, 1, 2], a.temps)
    elif a.where == "pff":
        run(["pff"], [0], [a.temp])
    else:
        real(a.temp)
