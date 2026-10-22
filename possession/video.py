"""Ball detection in the tactical video, inside the tracking's ball gaps.

The tracker loses the ball on ~41% of graded frames, often exactly at hand-overs. The video
still shows it most of the time. This finds it, searching only where it can be:

1. prior: for each gap frame, the ball-filling model (ballfill.py) gives a pitch position and an
   error radius; the per-frame homography turns them into a pixel search window, and gives the
   pixel size a 22 cm ball should have there;
2. candidates: small, bright, unsaturated blobs in the window that MOVE relative to the frames
   2 before and 2 after (the ball is never still for long in a gap of live play), of plausible
   size;
3. linking: a candidate is kept only if a candidate in a neighbouring frame is within a
   plausible ball speed (MAX_SPEED_MS), which removes isolated false blobs.

Results are pitch positions (normalised) with a confidence, cached in data/ (gitignored: they
are frame-level, derived from the gated video, and are never committed).

    uv run python -m possession.video check [--n 3000]   # self-check, aggregates only
    uv run python -m possession.video detect             # gap frames -> data/video_ball.npz
    apply(match) -> match with gaps filled where the detection is confident

PRIVACY: reads the gated video locally; prints only aggregates; never shows or saves an image.
"""

import os
import sys

import numpy as np

from bench import ROOT
from bench.match import FPS

BALL_M = 0.22
MAX_SPEED_MS = 40.0
K = 2                       # motion: compare frame t with t-K and t+K
MIN_RADIUS_M = 3.0          # search radius floor (the prior's radius can be tiny)
MAX_RADIUS_M = 25.0
WHITE_V = 150               # brightness (HSV value) floor
WHITE_S = 90                # saturation ceiling (the ball is white-ish, the grass is not)
MOTION = 25                 # grey-level change vs both neighbours
SIZE_RANGE = (0.25, 4.0)    # blob area vs the expected ball area
CACHE = os.path.join(ROOT, "data", "video_ball.npz")


def project(H, xy):
    """(N, 2) normalised pitch -> (N, 2) pixels with a 3x3 homography."""
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def unproject(H, px):
    return project(np.linalg.inv(H), px)


def ball_px(H, xy, pitch):
    """Expected ball diameter in pixels at pitch position xy (normalised)."""
    d = np.array([BALL_M / pitch[0], 0.0])
    a, b = project(H, np.array([xy, xy + d]))
    return float(np.linalg.norm(b - a))


DEFAULT = dict(v=WHITE_V, s=WHITE_S, motion=MOTION, both=True, size_max=SIZE_RANGE[1])


MAX_CANDS = 5               # per frame, best scores first (keeps link() linear)


def window(frame, before, after, center, radius_px):
    """The per-frame arrays candidates() needs, computed once (a threshold sweep reuses them)."""
    import cv2
    h, w = frame.shape[:2]
    x0, y0 = int(max(0, center[0] - radius_px)), int(max(0, center[1] - radius_px))
    x1, y1 = int(min(w, center[0] + radius_px + 1)), int(min(h, center[1] + radius_px + 1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = frame[y0:y1, x0:x1]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return dict(x0=x0, y0=y0, hsv=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                db=np.abs(grey - cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)),
                da=np.abs(grey - cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)))


def candidates(frame, before, after, center, radius_px, diam_px, params=None, win=None):
    """Ball candidates [(x_px, y_px, score)] in a window of `frame` (BGR uint8) around center,
    at most MAX_CANDS, best first. params: v (brightness floor), s (saturation ceiling), motion,
    both (motion vs both neighbours, else either), size_max (blob area / expected area)."""
    import cv2
    P = dict(DEFAULT, **(params or {}))
    win = window(frame, before, after, center, radius_px) if win is None else win
    if win is None:
        return []
    x0, y0, hsv = win["x0"], win["y0"], win["hsv"]
    mb, ma = win["db"] > P["motion"], win["da"] > P["motion"]
    moving = (mb & ma) if P["both"] else (mb | ma)
    white = (hsv[..., 2] > P["v"]) & (hsv[..., 1] < P["s"])
    mask = (moving & white).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area_ball = np.pi * (max(diam_px, 1.0) / 2) ** 2
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (SIZE_RANGE[0] * area_ball <= a <= P["size_max"] * area_ball + 2):
            continue
        cx, cy = cent[i][0] + x0, cent[i][1] + y0
        dist = np.hypot(cx - center[0], cy - center[1]) / max(radius_px, 1.0)
        size_fit = np.exp(-abs(np.log(a / area_ball)))
        out.append((float(cx), float(cy), float(size_fit * np.exp(-dist ** 2))))
    return sorted(out, key=lambda c: -c[2])[:MAX_CANDS]


def link(found, pitch):
    """Keep a frame's best candidate only if a neighbouring frame (+-1, +-2) has a candidate within
    a plausible ball speed. found: {frame: [(x_norm, y_norm, score)]} -> {frame: (x, y, conf)}."""
    m = np.asarray(pitch, dtype=np.float64)
    out = {}
    for t, cands in found.items():
        best = None
        for x, y, sc in cands:
            support = 0.0
            for dt in (-2, -1, 1, 2):
                for x2, y2, sc2 in found.get(t + dt, ()):
                    if np.hypot((x2 - x) * m[0], (y2 - y) * m[1]) <= MAX_SPEED_MS * abs(dt) / FPS:
                        support = max(support, sc2)
            conf = sc * support
            if conf > 0 and (best is None or conf > best[2]):
                best = (x, y, conf)
        if best is not None:
            out[t] = best
    return out


def search(video_path, H, frames, prior_xy, prior_r, pitch, params_list=None):
    """Decode the video once, front to back, and collect candidates on `frames` (sorted).
    prior_xy (N, 2) normalised, prior_r (N,) metres. Returns {frame: [(x_norm, y_norm, score)]},
    or one such dict per entry of params_list (a threshold sweep in a single decoding pass)."""
    plist = params_list or [None]
    import cv2
    want = {int(f): i for i, f in enumerate(frames)}
    need = set()
    for f in want:
        need.update((f - K, f, f + K))
    last = max(need) if need else -1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open the video: " + video_path)
    buf, founds, t = {}, [{} for _ in plist], -1
    while t < last:
        ok, img = cap.read()
        t += 1
        if not ok:
            break
        if t in need:
            buf[t] = img
        f = t - K                                     # frame whose neighbours are now decoded
        if f in want and f in buf and f - K in buf and t in buf:
            i = want[f]
            Hf = H[f]
            c = project(Hf, prior_xy[i:i + 1])[0]
            diam = ball_px(Hf, prior_xy[i], pitch)
            r_m = float(np.clip(prior_r[i], MIN_RADIUS_M, MAX_RADIUS_M))
            r_px = diam / BALL_M * r_m
            win = window(buf[f], buf[f - K], buf[t], c, r_px)
            for found, prm in zip(founds, plist):
                cands = candidates(buf[f], buf[f - K], buf[t], c, r_px, diam, prm, win)
                if cands:
                    px = np.array([[x, y] for x, y, _ in cands])
                    xy = unproject(Hf, px)
                    found[f] = [(float(a), float(b), s) for (a, b), (_, _, s) in zip(xy, cands)]
        for old in [k for k in buf if k < t - 2 * K]:
            del buf[old]
        if t and t % 5000 == 0:
            print("  decoded {} / {} frames".format(t, last), flush=True)
    cap.release()
    return founds if params_list else founds[0]


def _inputs(game="BAR-ATH"):
    from bench import import_starter
    from bench.__main__ import match_paths
    from bench.match import Match
    import_starter()
    from games import asset_paths
    from starter import load_homography
    H, _, valid = load_homography.load(game)
    video = asset_paths(source="tactical")["video"]
    if not os.path.isfile(video):
        sys.exit("tactical video missing: download BAR-ATH_tactical.mp4 to " + os.path.dirname(video))
    match = Match.load(match_paths(["hackathon"])[0]).without_labels()
    return match, np.asarray(H), np.asarray(valid), video


def _runs_around(pick, n_frames, half=2):
    """pick plus its +-half neighbours, sorted and unique (link() needs neighbours searched)."""
    f = (np.asarray(pick)[:, None] + np.arange(-half, half + 1)[None]).ravel()
    return np.unique(f[(f >= 0) & (f < n_frames)])


def check(n=3000, seed=0):
    """Self-check on frames WITH a tracked ball: hide it, use the ball-filling prior as if it were
    a gap, and compare detections with the tracked ball. Aggregates only."""
    from dataclasses import replace
    from . import ballfill
    match, H, valid, video = _inputs()
    seen = np.flatnonzero(~np.isnan(match.ball[:, 0]) & valid)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(seen, size=min(n, len(seen)), replace=False))
    hidden = match.ball.copy()
    for f in pick:                                   # hide ~1 s around each, like a short gap
        hidden[max(0, f - 12):f + 13] = np.nan
    masked = replace(match, ball=hidden)
    xy, r = ballfill.fill(masked)
    runs = _runs_around(pick, match.n_frames)   # linking needs the neighbouring frames searched
    found = link(search(video, H, runs, xy[runs], r[runs], match.pitch), match.pitch)
    m = np.asarray(match.pitch)
    d = [np.hypot(*((np.array(found[f][:2]) - match.ball[f, :2]) * m)) for f in pick if f in found]
    d = np.array(d)
    print("self-check on {} frames with a tracked ball (hidden):".format(len(pick)))
    print("  detected {:.1%}; error vs tracked ball median {:.2f} m, p90 {:.2f} m, within 1 m {:.1%}"
          .format(len(d) / len(pick), np.median(d) if len(d) else np.nan,
                  np.percentile(d, 90) if len(d) else np.nan, np.mean(d < 1.0) if len(d) else np.nan))
    prior = np.hypot(*((xy[pick] - match.ball[pick, :2]) * m).T)
    print("  (ball-filling prior alone: median error {:.2f} m)".format(np.nanmedian(prior)))


def detect():
    """All gap frames -> data/video_ball.npz: frame, x, y (normalised), confidence."""
    from . import ballfill
    match, H, valid, video = _inputs()
    xy, r = ballfill.fill(match)
    gaps = np.flatnonzero(np.isnan(match.ball[:, 0]) & ~np.isnan(xy[:, 0]) & valid)
    found = link(search(video, H, gaps, xy[gaps], r[gaps], match.pitch), match.pitch)
    fr = np.array(sorted(found), dtype=np.int64)
    arr = np.array([found[f] for f in fr]).reshape(-1, 3)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, frame=fr, x=arr[:, 0], y=arr[:, 1], conf=arr[:, 2])
    print("gap frames searched {}; ball found on {} ({:.1%}) -> {}".format(
        len(gaps), len(fr), len(fr) / max(len(gaps), 1), os.path.relpath(CACHE, ROOT)))


def diagnose(n=1500, last_frame=40000, seed=0):
    """What does the ball look like in this video? Aggregates only (quantiles), never an image.

    At the tracked ball's projected pixel (max over a 5x5 patch) vs at a background point 3 m
    away: brightness (V), saturation (S, min over the patch), motion vs t-K and t+K (min of the
    two); the expected ball diameter in pixels; and the offset from the projected ball to the
    nearest bright moving pixel within 40 px (a timing / homography offset would show here)."""
    import cv2
    match, H, valid, video_path = _inputs()
    g = np.flatnonzero(~np.isnan(match.ball[:, 0]) & valid)
    g = g[(g > K) & (g < last_frame)]
    rng = np.random.default_rng(seed)
    pick = set(np.sort(rng.choice(g, size=min(n, len(g)), replace=False)).tolist())
    need = set()
    for f in pick:
        need.update((f - K, f, f + K))
    cap = cv2.VideoCapture(video_path)
    buf, rows, t = {}, [], -1
    m = np.asarray(match.pitch)
    while t < max(need):
        ok, img = cap.read()
        t += 1
        if not ok:
            break
        if t in need:
            buf[t] = img
        f = t - K
        if f in pick and f in buf and f - K in buf:
            fr, b, a = buf[f], buf[f - K], buf[t]
            xy = match.ball[f, :2].astype(np.float64)
            off = xy + np.array([3.0 / m[0], 0.0])
            for kind, q in (("ball", xy), ("bg", off)):
                px = project(H[f], q[None])[0]
                x, y = int(round(px[0])), int(round(px[1]))
                if not (5 <= x < fr.shape[1] - 5 and 5 <= y < fr.shape[0] - 5):
                    continue
                sl = (slice(y - 2, y + 3), slice(x - 2, x + 3))
                hsv = cv2.cvtColor(fr[sl], cv2.COLOR_BGR2HSV)
                gr = lambda im: cv2.cvtColor(im[sl], cv2.COLOR_BGR2GRAY).astype(np.int16)  # noqa: E731
                mot = np.minimum(np.abs(gr(fr) - gr(b)), np.abs(gr(fr) - gr(a)))
                rows.append((kind, hsv[..., 2].max(), hsv[..., 1].min(), mot.max(),
                             ball_px(H[f], xy, match.pitch) if kind == "ball" else np.nan, np.nan))
            # nearest bright moving pixel to the projected ball
            px = project(H[f], xy[None])[0]
            x, y = int(round(px[0])), int(round(px[1]))
            if 40 <= x < fr.shape[1] - 40 and 40 <= y < fr.shape[0] - 40:
                win = (slice(y - 40, y + 41), slice(x - 40, x + 41))
                hsv = cv2.cvtColor(fr[win], cv2.COLOR_BGR2HSV)
                gr2 = lambda im: cv2.cvtColor(im[win], cv2.COLOR_BGR2GRAY).astype(np.int16)  # noqa: E731
                mot = np.minimum(np.abs(gr2(fr) - gr2(b)), np.abs(gr2(fr) - gr2(a)))
                cand = (hsv[..., 2] > 140) & (mot > 15)
                ys, xs = np.nonzero(cand)
                d = np.hypot(xs - 40, ys - 40).min() if len(xs) else np.nan
                rows.append(("offset", np.nan, np.nan, np.nan, np.nan, d))
        for old in [k for k in buf if k < t - 2 * K]:
            del buf[old]
    cap.release()
    def q(kind, col):
        v = np.array([r[col] for r in rows if r[0] == kind], dtype=float)
        v = v[~np.isnan(v)]
        return "p10 {:.0f} / p50 {:.0f} / p90 {:.0f}".format(*np.percentile(v, [10, 50, 90])) if len(v) else "-"
    nb = sum(r[0] == "ball" for r in rows)
    print("diagnose on {} frames with a tracked ball (first {} frames):".format(nb, last_frame))
    for col, name in ((1, "brightness V (max, 5x5)"), (2, "saturation S (min, 5x5)"),
                      (3, "motion vs t+-2 (max, 5x5)")):
        print("  {:<28} ball {:<30} background {}".format(name, q("ball", col), q("bg", col)))
    v = np.array([r[4] for r in rows if r[0] == "ball"], dtype=float)
    print("  expected ball diameter (px)  p10 {:.1f} / p50 {:.1f} / p90 {:.1f}".format(*np.percentile(v, [10, 50, 90])))
    d = np.array([r[5] for r in rows if r[0] == "offset"], dtype=float)
    print("  nearest bright moving pixel to the projected ball (px): p25 {:.0f} / p50 {:.0f} / p75 {:.0f}; "
          "none within 40 px {:.0%}".format(*np.nanpercentile(d, [25, 50, 75]), np.isnan(d).mean()))


def sweep(n=1000, last_frame=40000, seed=0):
    """Threshold sweep, scored like check(), in one decoding pass. Aggregates only."""
    from dataclasses import replace
    from itertools import product
    from . import ballfill
    match, H, valid, video = _inputs()
    seen = np.flatnonzero(~np.isnan(match.ball[:, 0]) & valid)
    seen = seen[(seen > 30) & (seen < last_frame)]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(seen, size=min(n, len(seen)), replace=False))
    hidden = match.ball.copy()
    for f in pick:
        hidden[max(0, f - 12):f + 13] = np.nan
    xy, r = ballfill.fill(replace(match, ball=hidden))
    runs = _runs_around(pick, match.n_frames)
    grid = [dict(v=v, s=sv, motion=mo, both=b, size_max=sm)
            for v, sv, mo, b, sm in product((120, 135, 150), (90, 130, 256), (10, 20),
                                            (True, False), (4.0, 8.0))]
    founds = search(video, H, runs, xy[runs], r[runs], match.pitch, grid)
    m = np.asarray(match.pitch)
    prior = np.nanmedian(np.hypot(*((xy[pick] - match.ball[pick, :2]) * m).T))
    print("  linking and scoring {} settings".format(len(grid)), flush=True)
    rows = []
    for prm, fd in zip(grid, founds):
        got = link(fd, match.pitch)
        d = np.array([np.hypot(*((np.array(got[f][:2]) - match.ball[f, :2]) * m))
                      for f in pick if f in got])
        hit1 = float(np.sum(d < 1.0)) / len(pick) if len(d) else 0.0
        rows.append((hit1, len(d) / len(pick), np.median(d) if len(d) else np.nan, prm))
    rows.sort(key=lambda r: -r[0])
    print("sweep on {} hidden tracked balls (first {} frames); prior median error {:.2f} m".format(
        len(pick), last_frame, prior))
    print("  {:>9} {:>9} {:>9}  params".format("found<1m", "detected", "median m"))
    for hit1, det, med, prm in rows[:12]:
        print("  {:9.1%} {:9.1%} {:9.2f}  {}".format(hit1, det, med, prm))


MIN_CONF = 0.2


def apply(match, cache=CACHE, min_conf=MIN_CONF):
    """match with its ball gaps filled from the video detections (confidence >= min_conf); the
    tracked ball is never overwritten. Unchanged if there is no cache."""
    from dataclasses import replace
    if not os.path.isfile(cache):
        return match
    with np.load(cache) as z:
        f, x, y, c = z["frame"], z["x"], z["y"], z["conf"]
    ok = (c >= min_conf) & (f < match.n_frames)
    f, x, y = f[ok], x[ok], y[ok]
    ball = match.ball.copy()
    gap = np.isnan(ball[f, 0])
    ball[f[gap], 0], ball[f[gap], 1] = x[gap], y[gap]
    ball[f[gap], 2] = np.nan                       # speed unknown; features derive it if needed
    return replace(match, ball=ball)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "detect", "diagnose", "sweep"])
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.n)
    elif a.cmd == "diagnose":
        diagnose(a.n)
    elif a.cmd == "sweep":
        sweep(a.n)
    else:
        detect()
    sys.exit(0)
