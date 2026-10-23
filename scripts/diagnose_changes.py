"""Which true possessions does M2 miss, and do its per-frame probabilities even see them?

    MTAH_DFL_LABELS=hackathon [MTAH_MODEL_DIR=models_exp/B_r] uv run python scripts/diagnose_changes.py

DFL bench (degraded, hackathon profile, seed 0), held-out models, default decoder. Every true
possession is put in buckets by:
- start: after a dead ball (restart) or straight from the other team (turnover);
- the first DFL event of the new team within +-1 s of the start (pass, tackle, claim, ...);
- duration;
- ball visible in the degraded tracking over the second around the start;
and per bucket: count, recall (matched at IoU 0.5), and P(new team) from the model, mean over the
possession and its max over a 1 s running mean. If missed possessions have a low max, the
probabilities never show them and no decoder can find them. Aggregates only.
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import dfl_events  # noqa: E402
from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.evaluate import IOU, diagnose, pm  # noqa: E402
from bench.match import FPS, D, Match  # noqa: E402
from possession import learned  # noqa: E402

DUR = ((0, 2), (2, 5), (5, 15), (15, np.inf))
EVENT_WINDOW = FPS


def _dur(n):
    s = n / FPS
    return next("{}-{}s".format(a, b) if b != np.inf else ">{}s".format(a) for a, b in DUR if a <= s < b)


def _event_kind(events, start, team):
    near = [e for e in events if e["team"] == team and abs(e["row"] - start) <= EVENT_WINDOW
            and e["kind"] not in ("Substitution", "Caution", "Foul", "Offside")]
    if not near:
        return "no event"
    e = min(near, key=lambda e: abs(e["row"] - start))
    return e.get("restart") or e["kind"]


def rows_for(path):
    clean = Match.load(path)
    seen = load_degraded(path, "hackathon", 0).without_labels()
    probs = learned.probabilities(seen)
    pred = learned.decode(seen, probs=probs)
    gt = clean.labels.astype(np.int64)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred.astype(np.int64), runs)
    gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), IOU)
    events = dfl_events.load(clean.name) or []
    ball = ~np.isnan(seen.ball[:, 0])
    kernel = np.ones(FPS) / FPS
    d = diagnose(clean, pred)
    out = [{"causes": d["miss_causes"], "lag": d["transition_lag"]}]
    for i, (a, b, team) in enumerate(gt_sp):
        after_dead = a > 0 and gt[a - 1] == D
        if not after_dead and (i == 0 or gt_sp[i - 1][2] == team or gt_sp[i - 1][1] != a):
            continue          # half start, or not a real hand-over
        p = probs[a:b, team]
        run = np.convolve(p, kernel, mode="valid") if len(p) >= FPS else p
        out.append({"start": "restart" if after_dead else "turnover",
                    "event": _event_kind(events, a, team), "dur": _dur(b - a),
                    "ball": "ball seen" if ball[max(0, a - FPS // 2):a + FPS // 2].mean() >= 0.5
                    else "ball missing",
                    "hit": bool(hit[i]), "p_mean": float(p.mean()), "p_max1s": float(run.max())})
    return out


def table(rows, key, title):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print("\n{:<34} {:>5} {:>7} {:>8}   {:>10} {:>10}".format(
        title, "n", "recall", "misses", "P found", "P missed"))
    n_miss = sum(not r["hit"] for r in rows)
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        found = [r["p_max1s"] for r in g if r["hit"]]
        miss = [r["p_max1s"] for r in g if not r["hit"]]
        print("{:<34} {:5d} {:7.2f} {:7.0%}   {:>10} {:>10}".format(
            k, len(g), np.mean([r["hit"] for r in g]), len(miss) / max(n_miss, 1),
            "{:.2f}".format(np.median(found)) if found else "-",
            "{:.2f}".format(np.median(miss)) if miss else "-"))


def main():
    rows, causes, lags = [], defaultdict(int), []
    for p in match_paths(["dfl"]):
        extra, *r = rows_for(p)
        rows += r
        for k, v in extra["causes"].items():
            causes[k] += v
        lags.append(extra["lag"])
        print(os.path.basename(p), len(rows), flush=True)
    print("\nmodel", learned.MODEL_DIR, "| labels", os.environ.get("MTAH_DFL_LABELS", "dfl"),
          "| P = median over possessions of max 1 s mean P(new team)")
    table(rows, lambda r: "all", "")
    table(rows, lambda r: r["start"], "start")
    table(rows, lambda r: "{} / {}".format(r["start"], r["event"]), "start / first event of new team")
    table(rows, lambda r: r["dur"], "duration")
    table(rows, lambda r: r["ball"], "ball around the start")
    table(rows, lambda r: "{} / {}".format(r["start"], r["ball"]), "start / ball")
    miss = [r for r in rows if not r["hit"]]
    lo = np.mean([r["p_max1s"] < 0.5 for r in miss])
    tot = sum(causes.values())
    print("\nwhy missed (bench.evaluate miss causes, IoU 0.5): " + ", ".join(
        "{} {:.0%}".format(k, v / tot) for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))
    print("transitions found within 2 s: {:.0%}; median lag {:+.1f} frames; mean |lag| {:.1f}".format(
        np.mean([x["found_within_2s"] for x in lags]), np.mean([x["median_frames"] for x in lags]),
        np.mean([x["mean_abs_frames"] for x in lags])))
    print("\nmissed possessions whose P(new team) never reaches 0.5 over 1 s: {:.0%}".format(lo))


if __name__ == "__main__":
    main()
