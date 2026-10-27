"""Detailed benchmark of the best method (M2 learned) and its near alternatives.

    uv run python scripts/benchmark.py        # ~3 min; writes reports/benchmark/summary.json

1. Ranking: methods x 7 DFL matches x 3 degrader seeds; paired per-match differences vs M2.
2. Sensitivity: M2 under clean / ball-only / hackathon / stress degradation.
3. Oracles: fix one error type perfectly and measure the spellF1 gain = the ceiling of each
   improvement direction.
4. Failure anatomy of M2: lost possessions by cause, duration, start context, ball visibility;
   spurious possessions; probability calibration; feature importance.
5. Real match, aggregates only: public window, possession rate, duration quantiles.

Held-out throughout: DFL predictions use the leave-one-match-out models.
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import REPORTS_DIR  # noqa: E402
from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import PROFILES, Profile, degrade  # noqa: E402
from bench.evaluate import pm, score  # noqa: E402
from bench.match import FPS, A, D, H, X, Match  # noqa: E402
from possession import baseline, hmm, hsmm, learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

SEEDS = (0, 1, 2)
STRESS = replace(Profile(), ball_missing_alive=0.30, ball_missing_dead=0.70,
                 ball_gap_median_s=14.0, blackout_share=0.45)
LOG = lambda p: np.log(np.clip(p, 1e-4, 1.0))  # noqa: E731


def methods():
    """name -> f(match, probs) ; probs = held-out learned probabilities (or None)."""
    return {
        "smooth50": lambda m, pr: baseline.predict(m, 50),
        "M1 hmm": lambda m, pr: hmm.predict(m),
        "M2 learned": lambda m, pr: learned.decode(m, probs=pr),
        "M2 switch 5": lambda m, pr: learned.decode(m, learned.Params(switch=5), pr),
        "M2 switch 20": lambda m, pr: learned.decode(m, learned.Params(switch=20), pr),
        "M3 durations": lambda m, pr: hsmm.decode_scores(
            m, LOG(pr), hsmm.Params(w_dur=1, alive_median_s=4)),
        "M3 gap k=200": lambda m, pr: hsmm.decode_scores(m, LOG(pr), hsmm.Params(kappa=200)),
    }


def degraded(clean, profile, seed):
    return degrade(clean, profile, seed=seed + sum(map(ord, clean.name)))


# -- oracles: perfect fix of one error type ---------------------------------------------------

def _fill_team(pred):
    team = np.where(np.isin(pred, (H, A)), pred, -1)
    idx = np.where(team >= 0, np.arange(len(team)), -1)
    ff = team[np.maximum.accumulate(idx)]
    ff[np.maximum.accumulate(idx) < 0] = -1
    idxb = np.where(team >= 0, np.arange(len(team)), len(team))
    nb = np.minimum.accumulate(idxb[::-1])[::-1]
    bf = np.where(nb < len(team), team[np.minimum(nb, len(team) - 1)], -1)
    out = np.where(ff >= 0, ff, bf)
    return np.where(out < 0, H, out)


def oracle_dead(gt, pred, seen):
    p = pred.copy()
    p[gt == D] = D
    need = np.isin(gt, (H, A)) & (p == D)
    p[need] = _fill_team(pred)[need]
    return p


def oracle_team(gt, pred, seen):
    p = pred.copy()
    m = np.isin(p, (H, A)) & np.isin(gt, (H, A))
    p[m] = gt[m]
    return p


def _changes(codes):
    f = np.flatnonzero(np.diff(codes)) + 1
    return f, codes[f - 1], codes[f]


def oracle_timing(gt, pred, seen, tol=FPS):
    """Move each predicted change to the matching true change within 1 s (same direction)."""
    pf, pa, pb = _changes(pred)
    gf, ga, gb = _changes(gt)
    new = pf.copy()
    used = set()
    for i, (f, a, b) in enumerate(zip(pf, pa, pb)):
        cand = np.flatnonzero((ga == a) & (gb == b) & (np.abs(gf - f) <= tol))
        cand = [c for c in cand if c not in used]
        if cand:
            c = min(cand, key=lambda c: abs(gf[c] - f))
            used.add(c)
            new[i] = gf[c]
    if np.any(np.diff(new) <= 0):  # a move would cross another change: keep the order safe
        bad = np.flatnonzero(np.diff(new) <= 0)
        new[bad] = pf[bad]
        new[bad + 1] = pf[bad + 1]
    out = np.empty_like(pred)
    edges = np.concatenate(([0], new, [len(pred)]))
    vals = np.concatenate(([pred[0]], pb))
    for v, a, b in zip(vals, edges[:-1], edges[1:]):
        out[a:b] = v
    return out


def _copy_windows(gt, pred, frames, half_width):
    p = pred.copy()
    for f in frames:
        a, b = max(0, f - half_width), min(len(p), f + half_width)
        p[a:b] = gt[a:b]
    return p


def oracle_restarts(gt, pred, seen):
    f, a, b = _changes(gt)
    return _copy_windows(gt, pred, f[(a == D) & np.isin(b, (H, A))], 2 * FPS)


def oracle_stoppage_starts(gt, pred, seen):
    f, a, b = _changes(gt)
    return _copy_windows(gt, pred, f[np.isin(a, (H, A)) & (b == D)], 2 * FPS)


def oracle_handovers(gt, pred, seen):
    f, a, b = _changes(gt)
    return _copy_windows(gt, pred, f[np.isin(a, (H, A)) & np.isin(b, (H, A))], FPS)


def oracle_short_spells(gt, pred, seen):
    p = pred.copy()
    for a, b, k in pm.segments(gt.astype(np.int64), pm.valid_runs(gt.astype(np.int64))):
        if k in (H, A) and b - a < 2 * FPS:
            p[a:b] = gt[a:b]
    return p


def oracle_ball_gaps(gt, pred, seen):
    p = pred.copy()
    gap = np.isnan(seen.ball[:, 0]) & (gt != X)
    p[gap] = gt[gap]
    return p


ORACLES = {
    "dead/alive perfect": oracle_dead,
    "team perfect (when both alive)": oracle_team,
    "timing perfect (changes within 1 s)": oracle_timing,
    "restarts perfect (+-2 s)": oracle_restarts,
    "stoppage starts perfect (+-2 s)": oracle_stoppage_starts,
    "hand-overs perfect (+-1 s)": oracle_handovers,
    "short possessions perfect (< 2 s)": oracle_short_spells,
    "ball-gap frames perfect": oracle_ball_gaps,
}


# -- failure anatomy ------------------------------------------------------------------------------

def anatomy(gt, pred, seen):
    g = gt.astype(np.int64)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(pred.astype(np.int64), runs)
    gsp, psp = pm.possession_spells(g, runs), pm.possession_spells(filled, runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(gsp, psp), len(gsp), 0.5)
    seg = pm.segments(g, runs)
    start_after = {}
    for i, (a, b, k) in enumerate(seg):
        if k in (H, A):
            prev = seg[i - 1][2] if i and seg[i - 1][1] == a else None
            start_after[a] = "after stoppage" if prev == D else (
                "after hand-over" if prev in (H, A) else "half start")
    vis = ~np.isnan(seen.ball[:, 0])
    rows = []
    for (a, b, k), h in zip(gsp, hit):
        v = vis[a:b].mean()
        rows.append({"hit": bool(h), "dur_s": (b - a) / FPS, "start": start_after.get(a, "?"),
                     "ball_visible": v})
    # spurious predicted spells: no GT spell matched at IoU 0.5
    ov = pm.spell_overlaps(gsp, psp)
    matched_p = set()
    used_g = set()
    for iou, gi, pj in ov:
        if iou < 0.5:
            break
        if gi in used_g or pj in matched_p:
            continue
        used_g.add(gi)
        matched_p.add(pj)
    spurious = [{"dur_s": (b - a) / FPS, "ball_visible": float(vis[a:b].mean())}
                for j, (a, b, k) in enumerate(psp) if j not in matched_p]
    return rows, spurious


def bucket(x, edges, labels):
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        if lo <= x < hi:
            return lab
    return labels[-1]


def calibration(gt, probs, graded):
    """Reliability of P(dead) and of P(home | alive) on held-out frames (10 bins, ECE)."""
    out = {}
    m = graded & (gt != X)
    pd, yd = probs[m, 2], (gt[m] == D)
    alive = m & np.isin(gt, (H, A))
    ph = probs[alive, 0] / np.maximum(probs[alive, 0] + probs[alive, 1], 1e-9)
    yh = gt[alive] == H
    for name, p, y in (("P(dead)", pd, yd), ("P(home | alive)", ph, yh)):
        bins = np.clip((p * 10).astype(int), 0, 9)
        ece, table = 0.0, []
        for k in range(10):
            s = bins == k
            if s.any():
                ece += s.mean() * abs(p[s].mean() - y[s].mean())
                table.append((k / 10, float(p[s].mean()), float(y[s].mean()), int(s.sum())))
        out[name] = {"ece": float(ece), "bins": table}
    return out


def main():
    out_dir = os.path.join(REPORTS_DIR, "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    M = methods()
    paths = match_paths(["dfl"])
    res = {"ranking": {}, "sensitivity": {}, "oracles": {}, "anatomy": {}, "calibration": {},
           "per_match": {}, "hackathon": {}}

    # 1 + 3 + 4 on the hackathon profile, 3 seeds
    scores = {k: [] for k in M}           # (match, seed, spell, macro)
    oracle_scores = {k: [] for k in ORACLES}
    anat, spur, calib = [], [], []
    for path in paths:
        clean = Match.load(path)
        for seed in SEEDS:
            seen = degraded(clean, "hackathon", seed)
            view = seen.without_labels()
            probs = learned.probabilities(view)
            gt = clean.labels
            preds = {}
            for k, f in M.items():
                preds[k] = f(view, probs)
                r = score(gt, preds[k])
                scores[k].append((clean.name, seed, r["spell_f1_mean"], r["macro_f1"]))
            base = preds["M2 learned"]
            for k, f in ORACLES.items():
                oracle_scores[k].append(score(gt, f(gt, base, seen))["spell_f1_mean"])
            if seed == 0:
                rows, sp = anatomy(gt, base, seen)
                anat += rows
                spur += sp
                calib.append((gt, probs, (gt != X)))
                # ball oracle: same match, ball not degraded (players still degraded)
                with_ball = replace(seen, ball=clean.ball.copy())
                pb = learned.probabilities(with_ball.without_labels())
                oracle_scores.setdefault("ball always tracked (players still noisy)", []).append(
                    score(gt, learned.decode(with_ball.without_labels(), probs=pb))["spell_f1_mean"])
            print(clean.name, seed, "M2 {:.4f}".format(scores["M2 learned"][-1][2]), flush=True)

    m2 = {(n, s): sp for n, s, sp, _ in scores["M2 learned"]}
    for k, rows in scores.items():
        per_match = {}
        for n, s, sp, mac in rows:
            per_match.setdefault(n, []).append((sp, mac, sp - m2[(n, s)]))
        diffs = np.array([np.mean([d for _, _, d in v]) for v in per_match.values()])
        res["ranking"][k] = {
            "spell_mean": float(np.mean([r[2] for r in rows])),
            "spell_seed_sd": float(np.mean([np.std([sp for sp, _, _ in v]) for v in per_match.values()])),
            "macro_mean": float(np.mean([r[3] for r in rows])),
            "diff_vs_M2": float(diffs.mean()), "diff_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
            "wins_vs_M2": int((diffs > 0).sum()), "n_matches": len(diffs)}
        res["per_match"][k] = {n: float(np.mean([sp for sp, _, _ in v])) for n, v in per_match.items()}
    base_mean = res["ranking"]["M2 learned"]["spell_mean"]
    for k, v in oracle_scores.items():
        n = len(v)
        ref = np.mean([r[2] for r in scores["M2 learned"]][:n] if n == len(scores["M2 learned"])
                      else [r[2] for r in scores["M2 learned"] if r[1] == 0])
        res["oracles"][k] = {"spell": float(np.mean(v)), "gain": float(np.mean(v) - ref)}

    # anatomy tables
    dur_edges, dur_labels = [0, 1, 3, 10, 1e9], ["<1 s", "1-3 s", "3-10 s", ">=10 s"]
    vis_edges, vis_labels = [0, 0.25, 0.75, 1.01], ["ball seen <25%", "25-75%", ">75%"]
    def table(key, edges, labels, field):
        t = {}
        for r in anat:
            lab = bucket(r[field], edges, labels) if edges else r[field]
            d = t.setdefault(lab, [0, 0])
            d[0] += 1
            d[1] += int(r["hit"])
        return {k: {"n": n, "recall": h / n, "missed": n - h} for k, (n, h) in t.items()}
    res["anatomy"]["by_duration"] = table("d", dur_edges, dur_labels, "dur_s")
    res["anatomy"]["by_start"] = table("s", None, None, "start")
    res["anatomy"]["by_ball_visibility"] = table("v", vis_edges, vis_labels, "ball_visible")
    res["anatomy"]["n_gt_spells"] = len(anat)
    sp_t = {}
    for s in spur:
        lab = bucket(s["dur_s"], dur_edges, dur_labels)
        sp_t[lab] = sp_t.get(lab, 0) + 1
    res["anatomy"]["spurious_by_duration"] = sp_t
    res["anatomy"]["n_spurious"] = len(spur)

    gts = np.concatenate([g for g, _, _ in calib])
    prs = np.concatenate([p for _, p, _ in calib])
    gr = np.concatenate([m for _, _, m in calib])
    res["calibration"] = calibration(gts, prs, gr)

    import lightgbm as lgb
    # feature importance of the first (largest-weight) member of the submission model
    booster = lgb.Booster(model_file=os.path.join(learned.members()[0][0], "lgbm.txt"))
    gain = booster.feature_importance("gain")
    names = booster.feature_name()
    order = np.argsort(-gain)
    res["feature_importance"] = [(names[i], float(gain[i] / gain.sum())) for i in order[:15]]

    # 2 sensitivity of M2 to the degradation profile (seed 0)
    for prof_name, prof in (("clean", None), ("ball-only", PROFILES["ball-only"]),
                            ("hackathon", PROFILES["hackathon"]), ("stress", STRESS)):
        vals = []
        for path in paths:
            clean = Match.load(path)
            seen = clean if prof is None else degrade(clean, prof, seed=sum(map(ord, clean.name)))
            vals.append(score(clean.labels, learned.predict(seen.without_labels()))["spell_f1_mean"])
        res["sensitivity"][prof_name] = float(np.mean(vals))
        print("profile", prof_name, round(float(np.mean(vals)), 4), flush=True)

    # 5 real match, aggregates only
    (hpath,) = match_paths(["hackathon"])
    hm = Match.load(hpath)
    hview = hm.without_labels()
    hprobs = learned.probabilities(hview)
    pub = hm.labels.astype(np.int64)
    pub_sp = pm.possession_spells(pub, pm.valid_runs(pub))
    res["hackathon"]["public_gt_duration_q"] = list(np.percentile(
        [(b - a) / FPS for a, b, _ in pub_sp], [10, 25, 50, 75, 90]))
    for k, f in M.items():
        p = f(hview, hprobs)
        g = np.zeros(hm.n_frames, bool)
        for _, a, b in hm.halves:
            g[a:b + 1] = True
        full = np.where(g, p, X).astype(np.int64)
        sp = pm.possession_spells(full, pm.valid_runs(np.where(g, 0, X).astype(np.int64)))
        d = [(b - a) / FPS for a, b, _ in sp]
        res["hackathon"][k] = {"public_spell": score(hm.labels, p)["spell_f1_mean"],
                               "rate_per_live_min": spell_rate(hm, p), "n_spells": len(d),
                               "duration_q": list(np.percentile(d, [10, 25, 50, 75, 90]))}
    dfl_d = []
    for path in paths:
        g = Match.load(path).labels.astype(np.int64)
        dfl_d += [(b - a) / FPS for a, b, _ in pm.possession_spells(g, pm.valid_runs(g))]
    res["hackathon"]["dfl_gt_duration_q"] = list(np.percentile(dfl_d, [10, 25, 50, 75, 90]))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
