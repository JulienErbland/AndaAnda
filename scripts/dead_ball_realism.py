"""Dead-ball behaviour, real vs bench (aggregates only): on dead frames with a visible ball, the
ball's speed (from positions), share standing still, distance to the nearest player, share
outside the lines. Calibrates bench/degrade.py's dead-ball knobs.

Real dead frames come from ALGORITHMIC calls over the whole match, never from the public ground
truth (RULES.md: the public 5 minutes are not a tuning set): the submission model's confident
dead frames, and a model-free one (players at walking pace for >= 5 s). They agree; the public
window's 4 stoppages did not (Oct 30).

    uv run python scripts/dead_ball_realism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.epts import derive_speed  # noqa: E402
from bench.match import D, X, Match, graded_mask  # noqa: E402


def stats(m, mask):
    sp = derive_speed(m.ball[:, :2], m.pitch).astype(float)
    P = m.players[..., :2] * np.asarray(m.pitch)
    B = m.ball[:, :2] * np.asarray(m.pitch)
    dd = np.linalg.norm(P - B[:, None], axis=-1)
    dd = np.where(np.isnan(dd), np.inf, dd).min(1)
    L, W = m.pitch
    out = (B[:, 0] < 0) | (B[:, 0] > L) | (B[:, 1] < 0) | (B[:, 1] > W)
    v = mask & ~np.isnan(sp)
    return np.r_[np.percentile(sp[v], [10, 50, 90]), np.mean(sp[v] < 0.3), np.median(dd[v]), np.mean(out[v])]


def real_targets():
    from bench.match import FPS
    from possession import learned
    hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
    g = graded_mask(hm)
    pr = learned.probabilities(hm)
    sp = hm.players[..., 2].astype(float)
    ms = np.nansum(sp, 1) / np.maximum((~np.isnan(sp)).sum(1), 1)
    k = 5 * FPS
    walk = np.convolve(ms < 1.0, np.ones(k), "same") >= k
    full = Match.load(match_paths(["hackathon"])[0])
    nopub = full.labels == X                              # public window dropped
    dec = learned.decode(hm, probs=pr)
    return {"model-confident dead": stats(hm, g & nopub & (pr[:, 2] > 0.9)),
            "decoded dead": stats(hm, g & nopub & (dec == D)),
            "walking pace >= 5 s": stats(hm, g & nopub & walk)}


def bench(profile="hackathon", decoded=False):
    """Mean over the DFL matches, on true dead frames (or, decoded=True, on the held-out
    models' decoded dead frames: the same mask as the real match's, to see its bias)."""
    from possession import learned
    rows = []
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        v = load_degraded(p, profile, 0)
        dead = learned.decode(v.without_labels()) == D if decoded else c.labels == D
        rows.append(stats(v, graded_mask(c) & dead))
    return np.mean(rows, axis=0)


FMT = "{:<28} speed p10/50/90 {:5.2f} {:5.2f} {:5.2f} | still {:4.0%} | nearest player {:4.1f} m | outside {:4.0%}"


def missing_split():
    """Ball-missing share on dead / live frames. Real: the submission's DECODED labels over the
    whole match minus the public window, and the model-free walking-pace mask. Bench: the same
    decoded calls (held-out models) next to the truth, to show the bias of judging "dead" by a
    model that itself reads a missing ball as dead."""
    from bench.match import FPS, H, A
    from possession import learned
    full = Match.load(match_paths(["hackathon"])[0])
    hm = full.without_labels()
    g = graded_mask(hm) & (full.labels == X)            # public window dropped
    pred = learned.decode(hm)
    miss = np.isnan(hm.ball[:, 0])
    sp = hm.players[..., 2].astype(float)
    ms = np.nansum(sp, 1) / np.maximum((~np.isnan(sp)).sum(1), 1)
    walk = np.convolve(ms < 1.0, np.ones(5 * FPS), "same") >= 5 * FPS
    live = np.isin(pred, (H, A))
    print("real, decoded:   missing on dead {:.1%}  live {:.1%}  (dead share {:.1%}); walking >= 5 s {:.1%}; "
          "overall {:.1%}".format(miss[g & (pred == D)].mean(), miss[g & live].mean(),
                                  (g & (pred == D)).sum() / g.sum(), miss[g & walk].mean(), miss[g].mean()))
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        v = load_degraded(p, "hackathon", 0)
        gm = graded_mask(c)
        pr = learned.decode(v.without_labels())
        mi = np.isnan(v.ball[:, 0])
        tl, pl_ = np.isin(c.labels, (H, A)), np.isin(pr, (H, A))
        print("bench {}: dead true {:.1%} / decoded {:.1%} | live true {:.1%} / decoded {:.1%}".format(
            c.name[-6:], mi[gm & (c.labels == D)].mean(), mi[gm & (pr == D)].mean(),
            mi[gm & tl].mean(), mi[gm & pl_].mean()), flush=True)


def main():
    if "--missing" in sys.argv:
        return missing_split()
    for name, v in real_targets().items():
        print(FMT.format("real: " + name, *v))
    print(FMT.format("bench, true dead", *bench()))
    print(FMT.format("bench, decoded dead", *bench(decoded=True)))


if __name__ == "__main__":
    main()
