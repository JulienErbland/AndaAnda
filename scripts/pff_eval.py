"""PFF World Cup 2022 as an EVALUATION-ONLY bench: 64 held-out matches no model was trained on.

    uv run python scripts/pff_eval.py run          # -> reports/compare/pff_eval.json (~30 min)
    uv run python scripts/pff_eval.py report       # paired table at each model's real-match switch
    uv run python scripts/pff_eval.py sweep R_r B_r 0.75 10 14 20 30   # one mix, more switches

Licence: PFF's Terms of Use forbid ML training on its data (results.md, Oct 23), so PFF is never
used to train or tune: only to compare DFL-trained candidates, each at the switch chosen on the
real match's change rate (CANDIDATES). Labels are PFF's possession (`home_ball` per possession
sequence), a definition close to but not the hackathon's; PFF's own rate is 4.67 per live minute.

Each base model's probabilities are computed once per match; ensembles are formed from them the
way learned.probabilities does (weighted geometric mean). Degrader profile "pff" (its ball is
already as missing as the real one), seed 0.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import REPORTS_DIR, ROOT  # noqa: E402

BASES = {"sub_old": "models_exp/sub_old", "A_r": "models_exp/A_r", "R_r": "models_exp/R_r",
         "B_r": "models_exp/B_r", "R_m": "models_exp/R_m", "B_m": "models_exp/B_m", "R_d": "models_exp/R_d", "B_d": "models_exp/B_d", "R_v": "models_exp/R_v", "B_v": "models_exp/B_v", "R_g": "models_exp/R_g", "B_g": "models_exp/B_g", "R_sc": "models_exp/R_sc", "R_ss": "models_exp/R_ss", "R_rx": "models_exp/R_rx", "B_rx": "models_exp/B_rx", "R_f": "models_exp/R_f", "B_f": "models_exp/B_f", "B1_f": "models_exp/B1_f", "R_rxf": "models_exp/R_rxf", "B_rxf": "models_exp/B_rxf"}
# name: ({base: weight}, switch whose real-match rate is closest to the public 4.25)
CANDIDATES = {
    "sub_old": ({"sub_old": 1}, 10),
    "A_r": ({"A_r": 1}, 7),
    "R_r": ({"R_r": 1}, 7),
    "B_r": ({"B_r": 1}, 1),
    "A_r+B_r 50/50": ({"A_r": 0.5, "B_r": 0.5}, 1),
    "A_r+B_r 75/25": ({"A_r": 0.75, "B_r": 0.25}, 3),
    "R_r+B_r 75/25 (submission)": ({"R_r": 0.75, "B_r": 0.25}, 3),
}
SWITCHES = (1, 2, 3, 5, 7, 10)
OUT = os.path.join(REPORTS_DIR, "compare", "pff_eval.json")


def _job(path):
    from bench.__main__ import load_degraded
    from bench.evaluate import score
    from bench.match import Match
    from possession import learned
    from scripts.tune_hmm import spell_rate
    clean = Match.load(path)
    view = load_degraded(path, "hackathon", 0).without_labels()     # pff profile for PFF
    cache = {}
    probs = {b: learned._single(view, os.path.join(ROOT, d), cache) for b, d in BASES.items()}
    out = {"name": clean.name, "true_rate": spell_rate(clean, clean.labels), "rows": []}
    for cand, (mix, _) in CANDIDATES.items():
        logp = sum(w * np.log(np.clip(probs[b], 1e-6, 1.0)) for b, w in mix.items())
        p = np.exp(logp / sum(mix.values()))
        p /= p.sum(1, keepdims=True)
        for sw in SWITCHES:
            pred = learned.decode(view, learned.Params(switch=sw), p)
            r = score(clean.labels, pred)
            out["rows"].append([cand, sw, r["spell_f1_mean"], r["macro_f1"], spell_rate(clean, pred)])
    print(clean.name, flush=True)
    return out


def run():
    from bench.__main__ import match_paths
    paths = match_paths(["pff"])
    with Pool(4) as pool:
        res = list(pool.imap(_job, paths))
    with open(OUT, "w") as f:
        json.dump(res, f)
    print("->", OUT)


def report():
    res = json.load(open(OUT))
    def at(cand, sw):
        return np.array([[r[2], r[3], r[4]] for m in res for r in m["rows"]
                         if r[0] == cand and r[1] == sw])
    true_rate = np.mean([m["true_rate"] for m in res])
    print("PFF evaluation-only bench: {} matches, true rate {:.2f} per live minute".format(
        len(res), true_rate))
    print("  {:<28} {:>6} {:>8} {:>8} {:>6}   all switches spellF1 ({})".format(
        "candidate", "switch", "spellF1", "macroF1", "rate", " ".join(map(str, SWITCHES))))
    for cand, (_, sw) in CANDIDATES.items():
        v = at(cand, sw).mean(0)
        sweep = " ".join("{:.3f}".format(at(cand, s)[:, 0].mean()) for s in SWITCHES)
        print("  {:<28} {:>6} {:8.4f} {:8.4f} {:6.2f}   {}".format(cand, sw, *v, sweep))
    ref = "R_r+B_r 75/25 (submission)"
    print("\npaired vs {} at the real-match switches (64 matches):".format(ref))
    b = at(ref, CANDIDATES[ref][1])[:, 0]
    for cand, (_, sw) in CANDIDATES.items():
        if cand == ref:
            continue
        d = b - at(cand, sw)[:, 0]
        print("  submission - {:<26} {:+.4f} ± {:.4f}  submission better on {}/{}".format(
            cand, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))


def _sweep_job(args):
    path, bases, weights, switches = args
    from bench.__main__ import load_degraded
    from bench.evaluate import score
    from bench.match import Match
    from possession import learned
    from scripts.tune_hmm import spell_rate
    clean = Match.load(path)
    view = load_degraded(path, "hackathon", 0).without_labels()
    cache = {}
    logp = sum(w * np.log(np.clip(learned._single(view, os.path.join(ROOT, BASES.get(b, b if "/" in b else "models_exp/" + b)), cache), 1e-6, 1.0))
               for b, w in zip(bases, weights))
    p = np.exp(logp / sum(weights))
    p /= p.sum(1, keepdims=True)
    rows = []
    for sw in switches:
        pred = learned.decode(view, learned.Params(switch=sw), p)
        rows.append([sw, score(clean.labels, pred)["spell_f1_mean"], spell_rate(clean, pred)])
    print(clean.name, flush=True)
    return rows


def sweep(a, b, wa, *switches):
    """Mix a (weight wa) + b (1 - wa) at more switches; prints spellF1 and rate per switch."""
    from bench.__main__ import match_paths
    switches = [float(s) for s in switches]
    jobs = [(p, (a, b), (float(wa), 1 - float(wa)), switches) for p in match_paths(["pff"])]
    with Pool(4) as pool:
        res = np.array(list(pool.imap(_sweep_job, jobs)))      # (matches, switches, 3)
    out = os.path.join(REPORTS_DIR, "compare", "pff_sweep_{}_{}_{}.json".format(a, b, wa))
    with open(out, "w") as f:
        json.dump(res.tolist(), f)
    for k, sw in enumerate(switches):
        print("switch {:>4g}: PFF spellF1 {:.4f}  rate {:.2f}".format(sw, res[:, k, 1].mean(), res[:, k, 2].mean()))


def mix(spec, *switches):
    """Any number of members, spec "dirA:wA,dirB:wB,..." (dirs under models_exp/ or with a "/")."""
    from bench.__main__ import match_paths
    bases, weights = zip(*((b, float(w)) for b, w in (x.split(":") for x in spec.split(","))))
    switches = [float(s) for s in switches]
    jobs = [(p, bases, weights, switches) for p in match_paths(["pff"])]
    with Pool(4) as pool:
        res = np.array(list(pool.imap(_sweep_job, jobs)))
    out = os.path.join(REPORTS_DIR, "compare", "pff_mix_{}.json".format("_".join(b.replace("/", "-") for b in bases)))
    with open(out, "w") as f:
        json.dump(res.tolist(), f)
    for k, sw in enumerate(switches):
        print("switch {:>4g}: PFF spellF1 {:.4f}  rate {:.2f}".format(sw, res[:, k, 1].mean(), res[:, k, 2].mean()))


if __name__ == "__main__":
    {"run": run, "report": report, "sweep": sweep, "mix": mix}[sys.argv[1]](*sys.argv[2:])
