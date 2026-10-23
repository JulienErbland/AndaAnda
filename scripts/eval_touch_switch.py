"""Cheaper switches at detected touches (learned.Params.switch_touch): grid on the DFL bench.

    MTAH_DFL_LABELS=hackathon [MTAH_MODEL_DIR=...] uv run python scripts/eval_touch_switch.py

The model's probabilities and the touches are computed once per (match, seed); only the decoder
changes. Prints, per setting: DFL spellF1 / macroF1 (held-out models, 3 degrader seeds), predicted
vs true possessions per live minute, recall of possessions under 5 s, and on the real match the
rate and the public-window score (sanity band only, never the selection criterion). Then the
best setting against the constant switch 10, paired by match. Aggregates only.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.evaluate import IOU, pm, score  # noqa: E402
from bench.match import FPS, Match  # noqa: E402
from possession import learned, touches  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

SEEDS = (0, 1, 2)
SETTINGS = [learned.Params(switch=sw) for sw in (10, 14, 20)] + [
    learned.Params(switch=sw, switch_touch=st, touch_window_s=w, touch_kinds=k)
    for sw, st, w, k in itertools.product((10, 14, 20), (1, 3, 5), (0.3, 0.6, 1.0),
                                          (("control",), ("control", "deflection", "contact")))]


def key(p):
    if p.switch_touch is None:
        return "switch {:g}".format(p.switch)
    return "switch {:g} touch {:g} w {:.1f} {}".format(
        p.switch, p.switch_touch, p.touch_window_s, "control" if len(p.touch_kinds) == 1 else "all")


def short_recall(labels, pred):
    runs = pm.valid_runs(labels.astype(np.int64))
    gt_sp = pm.possession_spells(labels.astype(np.int64), runs)
    pr_sp = pm.possession_spells(pm.fill_abstentions(pred.astype(np.int64), runs), runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), IOU)
    short = [h for (a, b, _), h in zip(gt_sp, hit) if b - a < 5 * FPS]
    return float(np.sum(short)), len(short)


def job(args):
    path, seed = args
    clean = Match.load(path)
    if seed is None:          # the real match: labels only on the public window
        seen = clean.without_labels()
    else:
        seen = load_degraded(path, "hackathon", seed).without_labels()
    probs = learned.probabilities(seen)
    tl = touches.detect(seen)
    out = {}
    for p in SETTINGS:
        pred = learned.decode(seen, p, probs, touch_list=tl)
        r = score(clean.labels, pred)
        row = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "rate": spell_rate(clean, pred)}
        if seed is not None:
            row["true_rate"] = spell_rate(clean, clean.labels)
            row["short_hit"], row["short_n"] = short_recall(clean.labels, pred)
        out[key(p)] = row
    return clean.name, seed, out


def main():
    jobs = [(p, s) for p in match_paths(["dfl"]) for s in SEEDS] + \
        [(p, None) for p in match_paths(["hackathon"])]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    bench = [r for r in res if r[1] is not None]
    real = next((r[2] for r in res if r[1] is None), {})
    with open(os.path.join("reports", "touch_switch.json"), "w") as f:
        json.dump({"model_dir": learned.MODEL_DIR, "bench": bench, "real": real}, f)

    def per_match(k, field):
        d = {}
        for name, _, out in bench:
            d.setdefault(name, []).append(out[k][field])
        return {n: np.mean(v) for n, v in d.items()}

    rows = []
    for p in SETTINGS:
        k = key(p)
        sh = sum(out[k]["short_hit"] for _, _, out in bench) / sum(out[k]["short_n"] for _, _, out in bench)
        rows.append((np.mean(list(per_match(k, "spell").values())), k,
                     np.mean(list(per_match(k, "macro").values())),
                     np.mean(list(per_match(k, "rate").values())),
                     np.mean(list(per_match(k, "true_rate").values())), sh,
                     real.get(k, {}).get("rate", np.nan), real.get(k, {}).get("spell", np.nan)))
    print("model", learned.MODEL_DIR)
    print("{:<40} {:>7} {:>7} {:>11} {:>8} {:>9} {:>7}".format(
        "setting", "spell", "macro", "rate/true", "<5s rec", "real rate", "public"))
    for r in sorted(rows, reverse=True)[:15] + [x for x in rows if x[1] == "switch 10"]:
        print("{:<40} {:7.4f} {:7.4f} {:5.2f}/{:<5.2f} {:8.2f} {:9.2f} {:7.3f}".format(r[1], r[0], *r[2:]))
    best = max(rows)[1]
    a, b = per_match("switch 10", "spell"), per_match(best, "spell")
    d = np.array([b[n] - a[n] for n in a])
    print("\nbest ({}) - switch 10, paired by match: {:+.4f} +- {:.4f}, better on {}/{}".format(
        best, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    main()
