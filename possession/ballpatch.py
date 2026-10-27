"""Learned ball detector for the tactical video: a patch classifier trained on THIS video.

The classical detector (video.py) found candidates near the ball but could not tell the ball
from boots, socks or line fragments of the same size. Here the tracked ball is free supervision
(59% of frames): every frame with a tracked ball says which bright spot in the image is the ball.

1. candidates: bright compact peaks (white top-hat) in the search window around the ball-filling
   prior (same window as video.py), at most MAX_CANDS, brightest first;
2. per candidate: a PATCH x PATCH patch of grey, saturation, brightness and motion vs t-2 and
   t+2, plus scalars (top-hat strength and rank, expected ball size, distance to the prior);
3. LightGBM binary "this candidate is the ball", trained on first-half frames; the candidate
   nearest the projected tracked ball (within POS_PX) is the positive, candidates further than
   NEG_PX are negatives.

Self-check (go / no-go, aggregates only): second-half frames with a tracked ball, ball hidden
for ~1 s around each, prior from ball filling as in a real gap; the best-scoring candidate is
compared with the tracked ball.

    uv run python -m possession.ballpatch check [--train 2500 --test 1500]

PRIVACY: reads the gated video locally; prints only aggregates; never shows or saves an image.
The trained model and any detections are derived from the gated video: data/ (gitignored).
"""

import os
import sys
from dataclasses import replace

import numpy as np

from bench import ROOT

from . import video

PATCH = 11
MAX_CANDS = 150     # sized by `diagnose`: ball in the top 150 on 50% of frames (12 px)
TOPHAT_K = 15          # px, top-hat kernel (the ball is ~4.6-8.5 px wide)
MIN_TOPHAT = 8
POS_PX = 10.0       # the projected tracked ball sits ~7 px (median) from its image peak
NEG_PX = 25.0
HIDE = 12              # frames hidden on each side of a test frame (a ~1 s gap)
MODEL = os.path.join(ROOT, "data", "ballpatch.txt")
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
           feature_fraction=0.5, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
           force_row_wise=True, num_threads=4)
ROUNDS = 400


def _channels(frame, before, after, x0, y0, x1, y1):
    import cv2
    crop = frame[y0:y1, x0:x1]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    g16 = grey.astype(np.int16)
    mb = np.abs(g16 - cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16))
    ma = np.abs(g16 - cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_K, TOPHAT_K)))
    return np.stack([grey, hsv[..., 1], hsv[..., 2], mb.clip(0, 255), ma.clip(0, 255)],
                    -1).astype(np.float32), tophat


def candidates(frame, before, after, center, radius_px, diam_px):
    """(xy_px (N, 2), X (N, F)) for the candidates in the window, best (top-hat + motion) first."""
    import cv2
    h, w = frame.shape[:2]
    r = PATCH // 2
    x0, y0 = int(max(0, center[0] - radius_px - r)), int(max(0, center[1] - radius_px - r))
    x1, y1 = int(min(w, center[0] + radius_px + r + 1)), int(min(h, center[1] + radius_px + r + 1))
    if x1 - x0 < PATCH + 2 or y1 - y0 < PATCH + 2:
        return np.zeros((0, 2)), None
    ch, th = _channels(frame, before, after, x0, y0, x1, y1)
    peak = (th == cv2.dilate(th, np.ones((5, 5), np.uint8))) & (th >= MIN_TOPHAT)
    peak[:r], peak[-r:], peak[:, :r], peak[:, -r:] = False, False, False, False
    ys, xs = np.nonzero(peak)
    if not len(xs):
        return np.zeros((0, 2)), None
    # rank by brightness AND motion: lines and still kit are bright but do not move
    motion = np.minimum(ch[..., 3], ch[..., 4])
    motion = cv2.dilate(motion, np.ones((5, 5), np.uint8))
    order = np.argsort(-(th[ys, xs] + motion[ys, xs]), kind="stable")[:MAX_CANDS]
    ys, xs = ys[order], xs[order]
    patches = np.stack([ch[y - r:y + r + 1, x - r:x + r + 1] for y, x in zip(ys, xs)])
    n = len(xs)
    px = np.c_[xs + x0, ys + y0].astype(np.float64)
    dist = np.hypot(px[:, 0] - center[0], px[:, 1] - center[1]) / max(radius_px, 1.0)
    scal = np.c_[th[ys, xs], np.arange(n), np.full(n, diam_px), dist,
                 patches[..., 0].mean((1, 2)), patches[:, r, r, 0] - patches[..., 0].mean((1, 2))]
    return px, np.c_[patches.reshape(n, -1), scal].astype(np.float32)


def _frames_with_ball(match, valid, lo, hi, n, rng):
    seen = ~np.isnan(match.ball[:, 0]) & valid
    # the whole hidden span must have had a tracked ball (a clean test of a ~1 s gap)
    ok = np.convolve(seen.astype(int), np.ones(2 * HIDE + 1, int), "same") == 2 * HIDE + 1
    cand = np.flatnonzero(ok)
    cand = cand[(cand >= lo) & (cand <= hi)]
    keep, last = [], -10 ** 9
    for f in np.sort(rng.choice(cand, size=min(4 * n, len(cand)), replace=False)):
        if f - last > 2 * HIDE + 4:                     # no overlapping hidden spans
            keep.append(f)
            last = f
    return np.sort(rng.choice(keep, size=min(n, len(keep)), replace=False))


def collect(match, H, video_path, frames, prior_xy, prior_r):
    """One decoding pass: {frame: (cands_px, X)} for the requested frames."""
    import cv2
    want = {int(f): i for i, f in enumerate(frames)}
    need = set()
    for f in want:
        need.update((f - video.K, f, f + video.K))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open the video: " + video_path)
    buf, out, t, last = {}, {}, -1, max(need)
    while t < last:
        if not cap.grab():
            break
        t += 1
        if t in need:
            buf[t] = cap.retrieve()[1]
        f = t - video.K
        if f in want and f in buf and f - video.K in buf and t in buf:
            i = want[f]
            c = video.project(H[f], prior_xy[i:i + 1])[0]
            diam = video.ball_px(H[f], prior_xy[i], match.pitch)
            r_px = diam / video.BALL_M * float(np.clip(prior_r[i], video.MIN_RADIUS_M,
                                                       video.MAX_RADIUS_M))
            out[f] = candidates(buf[f], buf[f - video.K], buf[t], c, r_px, diam)
        for old in [k for k in buf if k < t - 2 * video.K]:
            del buf[old]
        if t and t % 20000 == 0:
            print("  decoded {} / {} frames".format(t, last), flush=True)
    cap.release()
    return out


def _labels(match, H, f, px):
    true = video.project(H[f], match.ball[f:f + 1, :2].astype(np.float64))[0]
    d = np.hypot(px[:, 0] - true[0], px[:, 1] - true[1])
    y = np.full(len(px), -1)
    y[d > NEG_PX] = 0
    if len(d) and d.min() <= POS_PX:
        y[np.argmin(d)] = 1
    return y, d


def diagnose(n=300, last_frame=30000, seed=0):
    """How far is the projected tracked ball from the nearest candidate, and at which rank?
    Sizes MAX_CANDS and POS_PX. Aggregates only."""
    from . import ballfill
    global MAX_CANDS
    match, H, valid, video_path = video._inputs()
    rng = np.random.default_rng(seed)
    pick = _frames_with_ball(match, valid, match.halves[0][1], last_frame, n, rng)
    hidden = match.ball.copy()
    for f in pick:
        hidden[f - HIDE:f + HIDE + 1] = np.nan
    xy, r = ballfill.fill(replace(match, ball=hidden))
    keep, MAX_CANDS = MAX_CANDS, 400
    got = collect(match, H, video_path, pick, xy[pick], r[pick])
    MAX_CANDS = keep
    near, rank, ncand = [], [], []
    for f in pick:
        px, _ = got.get(f, (np.zeros((0, 2)), None))
        ncand.append(len(px))
        if not len(px):
            near.append(np.inf)
            rank.append(10 ** 6)
            continue
        _, d = _labels(match, H, f, px)
        near.append(d.min())
        rank.append(int(np.argmin(d)))
    near, rank = np.array(near), np.array(rank)
    print("diagnose on {} frames: candidates per window p50 {:.0f} p90 {:.0f}".format(
        len(pick), *np.percentile(ncand, [50, 90])))
    print("  nearest candidate to the projected ball (px): p25 {:.1f} p50 {:.1f} p75 {:.1f}".format(
        *np.percentile(near[np.isfinite(near)], [25, 50, 75])))
    for px_ in (4, 8, 12, 20):
        print("  within {:>2} px: any rank {:.0%}, top 60 {:.0%}, top 150 {:.0%}".format(
            px_, np.mean(near <= px_), np.mean((near <= px_) & (rank < 60)),
            np.mean((near <= px_) & (rank < 150))))


def check(n_train=2500, n_test=1500, seed=0):
    import lightgbm as lgb
    from . import ballfill
    match, H, valid, video_path = video._inputs()
    halves = match.halves
    rng = np.random.default_rng(seed)
    train = _frames_with_ball(match, valid, halves[0][1], halves[0][2], n_train, rng)
    test = _frames_with_ball(match, valid, halves[1][1], halves[1][2], n_test, rng)
    frames = np.sort(np.r_[train, test])
    hidden = match.ball.copy()
    for f in frames:
        hidden[f - HIDE:f + HIDE + 1] = np.nan
    xy, r = ballfill.fill(replace(match, ball=hidden))
    print("frames: train {} (1st half), test {} (2nd half); decoding...".format(len(train), len(test)),
          flush=True)
    got = collect(match, H, video_path, frames, xy[frames], r[frames])

    rows, ys, recall = [], [], []
    for f in train:
        px, X_ = got.get(f, (None, None))
        if X_ is None:
            recall.append(False)
            continue
        y, d = _labels(match, H, f, px)
        recall.append(bool((y == 1).any()))
        keep = y >= 0
        rows.append(X_[keep])
        ys.append(y[keep])
    X_, y = np.concatenate(rows), np.concatenate(ys)
    print("train: ball among the candidates on {:.1%} of frames; rows {} ({} positives)".format(
        np.mean(recall), len(y), int(y.sum())), flush=True)
    booster = lgb.train(LGB, lgb.Dataset(X_, y), ROUNDS)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    booster.save_model(MODEL)

    m = np.asarray(match.pitch, dtype=np.float64)
    err, err_prior, in_cands, scores = [], [], [], []
    for f in test:
        prior_e = np.hypot(*((xy[f] - match.ball[f, :2]) * m))
        err_prior.append(prior_e)
        px, X_ = got.get(f, (None, None))
        if X_ is None:
            err.append(np.inf)
            in_cands.append(False)
            continue
        y, _ = _labels(match, H, f, px)
        in_cands.append(bool((y == 1).any()))
        s = booster.predict(X_)
        best = int(np.argmax(s))
        scores.append(s[best])
        pos = video.unproject(H[f], px[best:best + 1])[0]
        err.append(np.hypot(*((pos - match.ball[f, :2]) * m)))
    err, err_prior, scores = np.array(err), np.array(err_prior), np.array(scores)
    print("test (2nd half, {} hidden tracked balls):".format(len(test)))
    print("  ball among the candidates {:.1%}".format(np.mean(in_cands)))
    print("  best candidate within 1 m {:.1%}, within 2 m {:.1%}; median error {:.2f} m "
          "(prior alone {:.2f} m)".format(np.mean(err < 1), np.mean(err < 2),
                                          np.median(err), np.median(err_prior)))
    fin = err[np.isfinite(err)]
    for q in (0.5, 0.8, 0.9):
        thr = np.quantile(scores, 1 - q) if len(scores) else np.nan
        sel = fin[scores >= thr] if len(scores) == len(fin) else fin
        print("  most confident {:.0%}: within 1 m {:.1%}, median {:.2f} m".format(
            q, np.mean(sel < 1), np.median(sel)))
    go = np.mean(err < 1) >= 0.60 and np.median(err) < np.median(err_prior)
    print("GO" if go else "NO-GO", "(gate: >= 60% within 1 m and median error below the prior)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "diagnose"])
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--test", type=int, default=1500)
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose()
    else:
        check(a.train, a.test)
    sys.exit(0)
