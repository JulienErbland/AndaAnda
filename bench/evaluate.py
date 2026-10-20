"""Score with the organisers' scorer, then explain *where* a prediction loses points.

`score` is `possession_metrics.evaluate` unchanged, so every headline number here is the number
the official board would print for the same labels. `diagnose` adds the breakdown the board does
not give: which possessions were missed and why, how late transitions are, and in which
situations frames go wrong.
"""

import numpy as np

from . import import_starter
from .match import FPS, A, D, H, X, graded_mask

import_starter()
import possession_metrics as pm  # noqa: E402

DURATION_BUCKETS_S = (0, 1, 3, 10, np.inf)
IOU = pm.SPELL_IOU_HEADLINE
LAG_WINDOW = 2 * FPS          # transitions further apart than this are not the same transition
NEAR_TRANSITION = FPS         # "near a hand-over" = within 1 s of a GT transition
WORST_WINDOW = 10 * FPS


def score(labels, pred, bootstrap_n=0):
    return pm.evaluate(labels.astype(np.int64), pred.astype(np.int64), bootstrap_n=bootstrap_n)


def diagnose(match, pred):
    gt = match.labels.astype(np.int64)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred.astype(np.int64), runs)
    gt_sp = pm.possession_spells(gt, runs)
    pr_sp = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_sp, pr_sp)
    gt_hit = pm.matched_gt_flags(overlaps, len(gt_sp), IOU)
    pr_hit = _matched_pred_flags(overlaps, len(pr_sp), IOU)

    return {
        "spells_by_duration": _by_duration(gt_sp, gt_hit, pr_sp, pr_hit),
        "miss_causes": _miss_causes(gt, filled, gt_sp, gt_hit, pr_sp),
        "transition_lag": _transition_lag(gt, filled, runs),
        "frame_error_by_context": _context_errors(match, gt, filled),
        "worst_stretches": worst_stretches(match, gt, filled),
    }


def _matched_pred_flags(overlaps, n_pred, threshold):
    """Mirror of pm.matched_gt_flags for the prediction side (same greedy, same order)."""
    flags = np.zeros(n_pred, dtype=bool)
    matched_gt = set()
    for iou, gi, pj in overlaps:
        if iou < threshold:
            break
        if flags[pj] or gi in matched_gt:
            continue
        flags[pj] = True
        matched_gt.add(gi)
    return flags


def _bucket(length):
    s = length / FPS
    for lo, hi in zip(DURATION_BUCKETS_S, DURATION_BUCKETS_S[1:]):
        if lo <= s < hi:
            return "{}-{}s".format(lo, hi) if hi != np.inf else ">={}s".format(lo)


def _by_duration(gt_sp, gt_hit, pr_sp, pr_hit):
    """Recall of GT spells and precision of predicted spells, per duration bucket, at IoU 0.5."""
    out = {}
    for spells, hits, n_key, rate_key in ((gt_sp, gt_hit, "n_gt", "recall"),
                                          (pr_sp, pr_hit, "n_pred", "precision")):
        for (a, b, _), hit in zip(spells, hits):
            row = out.setdefault(_bucket(b - a), {"n_gt": 0, "recall": 0, "n_pred": 0,
                                                  "precision": 0})
            row[n_key] += 1
            row[rate_key] += int(hit)
    for row in out.values():
        row["recall"] = row["recall"] / row["n_gt"] if row["n_gt"] else None
        row["precision"] = row["precision"] / row["n_pred"] if row["n_pred"] else None
    return dict(sorted(out.items(), key=lambda kv: float(kv[0].strip(">=").split("-")[0]
                                                           .rstrip("s"))))


def _miss_causes(gt, filled, gt_sp, gt_hit, pr_sp):
    """Why each unmatched GT spell was missed:

    wrong_team         most of it was predicted as the other team
    absorbed_by_dead   most of it was predicted dead
    merged             a same-team predicted spell also swallowed a neighbouring GT spell
    split              two or more same-team predicted spells cut it up
    boundary           one same-team predicted spell, but its edges are too far off for IoU 0.5
    """
    causes = {k: 0 for k in ("wrong_team", "absorbed_by_dead", "merged", "split", "boundary")}
    for (a, b, team), hit in zip(gt_sp, gt_hit):
        if hit:
            continue
        seg = filled[a:b]
        same, dead, other = np.mean(seg == team), np.mean(seg == D), np.mean(seg == 1 - team)
        if same < 0.5:
            causes["absorbed_by_dead" if dead >= other else "wrong_team"] += 1
            continue
        mine = [(pa, pb) for pa, pb, pt in pr_sp if pt == team and pa < b and a < pb]
        if any(sum(1 for ga, gb, gt_team in gt_sp if gt_team == team and ga < pb and pa < gb) > 1
               for pa, pb in mine):
            causes["merged"] += 1
        elif len(mine) > 1:
            causes["split"] += 1
        else:
            causes["boundary"] += 1
    return causes


def _transition_lag(gt, filled, runs):
    """Signed lag (pred - GT, frames) to the nearest same-direction predicted transition."""
    gt_t = pm.transitions(gt, runs)
    pr_t = pm.transitions(filled, runs)
    by_kind = {}
    for f, a, b in pr_t:
        by_kind.setdefault((a, b), []).append(f)
    by_kind = {k: np.asarray(v) for k, v in by_kind.items()}
    lags = []
    for f, a, b in gt_t:
        cand = by_kind.get((a, b))
        if cand is None:
            continue
        nearest = cand[np.argmin(np.abs(cand - f))]
        if abs(nearest - f) <= LAG_WINDOW:
            lags.append(int(nearest - f))
    lags = np.asarray(lags)
    found = len(lags) / len(gt_t) if gt_t else None
    if not len(lags):
        return {"found_within_2s": found}
    return {"found_within_2s": found, "median_frames": float(np.median(lags)),
            "p10_frames": float(np.percentile(lags, 10)),
            "p90_frames": float(np.percentile(lags, 90)),
            "mean_abs_frames": float(np.mean(np.abs(lags)))}


def _context_errors(match, gt, filled):
    """Per-frame error rate (filled prediction vs GT) in situations that matter."""
    graded = (gt != X) & graded_mask(match)
    wrong = filled != gt
    change = np.flatnonzero(np.diff(gt) != 0) + 1
    near = np.zeros(len(gt), dtype=bool)
    for f in change:
        near[max(0, f - NEAR_TRANSITION):f + NEAR_TRANSITION] = True
    ball_missing = np.isnan(match.ball[:, 0])
    contexts = {
        "all": graded,
        "ball_visible": graded & ~ball_missing,
        "ball_missing": graded & ball_missing,
        "gt_alive": graded & ((gt == H) | (gt == A)),
        "gt_dead": graded & (gt == D),
        "near_transition_1s": graded & near,
        "mid_spell": graded & ~near,
    }
    return {k: {"share_of_frames": float(m.sum() / graded.sum()),
                "error_rate": float(wrong[m].mean()) if m.any() else None}
            for k, m in contexts.items()}


def worst_stretches(match, gt, filled, k=10, window=WORST_WINDOW):
    """Top-k non-overlapping windows by number of wrong graded frames."""
    wrong = ((filled != gt) & (gt != X)).astype(np.int64)
    csum = np.concatenate(([0], np.cumsum(wrong)))
    totals = csum[window:] - csum[:-window]
    out, taken = [], np.zeros(len(totals), dtype=bool)
    for start in np.argsort(-totals, kind="stable"):
        if len(out) == k or totals[start] == 0:
            break
        if taken[start]:
            continue
        taken[max(0, start - window + 1):start + window] = True
        out.append({"start": int(start), "stop": int(start + window),
                    "clock": clock(match, start), "wrong_frames": int(totals[start]),
                    "gt": runs_string(gt[start:start + window]),
                    "pred": runs_string(filled[start:start + window])})
    return out


def clock(match, frame):
    """'H1 23:45' -- half and match time since that half's first graded frame."""
    for half, first, last in match.halves:
        if first <= frame <= last:
            s = (frame - first) // FPS
            return "H{} {:02d}:{:02d}".format(half, s // 60, s % 60)
    return "-"


def runs_string(codes):
    """'H3.2 A0.8 D5.0' -- label runs with durations in seconds."""
    letters = "HADX"
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return " ".join("{}{:.1f}".format(letters[codes[a]], (b - a) / FPS)
                    for a, b in zip(starts, stops))
