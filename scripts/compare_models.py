"""Evaluate an M2 model across switch penalties, then compare two such evaluations.

    MTAH_MODEL_DIR=models_exp/x uv run python scripts/compare_models.py eval --out x.json
    uv run python scripts/compare_models.py compare a.json b.json

eval: the model in MTAH_MODEL_DIR (default models/), held-out on the DFL matches (respects
MTAH_DFL_LABELS), 3 degrader seeds; spellF1 / macroF1 per match, seed and switch penalty; the
real match's possessions per live minute and public-window scores per switch (aggregates only);
scenario pass rates at the model's default switch.

compare: paired per-match spellF1 differences (b - a) at each common switch, and at matched
real-match rate (each model's switch whose rate is closest to the public 4.25), since a DFL gain
bought by changing possession more often does not transfer.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PUBLIC_RATE = 4.25


def cmd_eval(args):
    from bench import scenarios
    from bench.__main__ import match_paths
    from bench.degrade import degrade
    from bench.evaluate import score
    from bench.match import Match
    from possession import learned
    from scripts.tune_hmm import spell_rate

    prof = "hackathon"
    if args.profile == "stress":
        from scripts.benchmark import STRESS
        prof = STRESS
    out = {"model_dir": learned.MODEL_DIR, "profile": args.profile, "labels": os.environ.get("MTAH_DFL_LABELS", "dfl"),
           "switches": args.switches, "dfl": [], "hackathon": {}, "scenarios": {},
           "bench_rate": []}
    contexts, preds = [], {"m": {}}
    for p in match_paths(args.matches):
        clean = Match.load(p)
        for seed in args.seeds:
            pr_ = "pff" if clean.name.startswith("pff-") and args.profile == "hackathon" else prof
            seen = degrade(clean, pr_, seed=seed + sum(map(ord, clean.name)))
            view = seen.without_labels()
            pr = learned.probabilities(view)
            for sw in args.switches:
                pred = learned.decode(view, learned.Params(switch=sw), pr)
                r = score(clean.labels, pred)
                out["dfl"].append([clean.name, seed, sw, r["spell_f1_mean"], r["macro_f1"],
                                   spell_rate(clean, pred)])
                if seed == args.seeds[0] and sw == learned.Params().switch:
                    contexts.append(scenarios.context(clean.name, clean, seen))
                    preds["m"][clean.name] = pred
                    out["bench_rate"].append([spell_rate(clean, pred), spell_rate(clean, clean.labels)])
        print(clean.name, flush=True)
    table, _ = scenarios.evaluate(contexts, preds)
    out["scenarios"] = {k: [int(x) for x in v["m"]] for k, v in table.items()}
    hp = match_paths(["hackathon"])
    if hp:
        hm = Match.load(hp[0])
        hv = hm.without_labels()
        pr = learned.probabilities(hv)
        for sw in args.switches:
            pred = learned.decode(hv, learned.Params(switch=sw), pr)
            r = score(hm.labels, pred)
            out["hackathon"][str(sw)] = {"rate": spell_rate(hm, pred),
                                         "public_spell": r["spell_f1_mean"],
                                         "public_macro": r["macro_f1"]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("->", args.out)


def _per_match(ev, sw):
    d = {}
    for name, seed, s, sp, mac, *rate in ev["dfl"]:
        if s == sw:
            d.setdefault(name, []).append((sp, mac, rate[0] if rate else np.nan))
    return {n: np.mean(v, axis=0) for n, v in d.items()}


def _summary(ev, label):
    print("\n{} ({}, labels {})".format(label, ev["model_dir"], ev["labels"]))
    if ev.get("bench_rate"):
        br = np.array(ev["bench_rate"])
        print("  bench rate at default switch: predicted {:.2f} / true {:.2f} per live minute".format(
            *br.mean(0)))
    print("  {:>6} {:>8} {:>8} {:>10} {:>8} {:>8}".format("switch", "spellF1", "macroF1", "bench rate",
                                                       "rate", "public"))
    for sw in ev["switches"]:
        pm = _per_match(ev, sw)
        h = ev["hackathon"].get(str(sw), {})
        print("  {:>6} {:8.4f} {:8.4f} {:10.2f} {:>8} {:>8}".format(
            sw, np.mean([v[0] for v in pm.values()]), np.mean([v[1] for v in pm.values()]),
            np.mean([v[2] for v in pm.values()]),
            "{:.2f}".format(h["rate"]) if h else "-", "{:.3f}".format(h["public_spell"]) if h else "-"))


def _matched_switch(ev):
    if not ev["hackathon"]:
        return None
    return min(ev["switches"], key=lambda s: abs(ev["hackathon"][str(s)]["rate"] - PUBLIC_RATE))


def _paired(a, b, sa, sb):
    pa, pb = _per_match(a, sa), _per_match(b, sb)
    d = np.array([pb[n][0] - pa[n][0] for n in pa if n in pb])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)


def cmd_compare(args):
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    _summary(a, "A")
    _summary(b, "B")
    print("\nB - A, paired by match (seeds averaged):")
    for sw in [s for s in a["switches"] if s in b["switches"]]:
        m, se, w, n = _paired(a, b, sw, sw)
        print("  switch {:>3}: {:+.4f} ± {:.4f}  B better on {}/{}".format(sw, m, se, w, n))
    sa, sb = _matched_switch(a), _matched_switch(b)
    if sa is not None and sb is not None:
        m, se, w, n = _paired(a, b, sa, sb)
        print("  at matched real rate (A switch {}, B switch {}): {:+.4f} ± {:.4f}  B better on {}/{}"
              .format(sa, sb, m, se, w, n))
    print("\nscenarios at default switch (A / B):")
    for k in a["scenarios"]:
        (pa, na), (pb, nb) = a["scenarios"][k], b["scenarios"].get(k, (0, 0))
        print("  {:<22} {:5.0%} {:5.0%}  (n={})".format(k, pa / max(na, 1), pb / max(nb, 1), na))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--switches", type=float, nargs="+", default=[5, 7, 10, 14, 20])
    p.add_argument("--matches", nargs="+", default=["dfl"], help="match groups, e.g. dfl pff")
    p.add_argument("--profile", choices=["hackathon", "stress"], default="hackathon",
                   help="degrader profile the DFL matches are scored under")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
