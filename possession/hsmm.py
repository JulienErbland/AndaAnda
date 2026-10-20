"""M3: semi-Markov (explicit-duration) decoding of per-frame H/A/D evidence.

A plain Viterbi pays a fixed price per change and sums per-frame evidence. Two things go wrong
inside a long ball gap: the evidence there is one piece of information repeated hundreds of times,
so a slight lean towards "dead" adds up to more than any change penalty; and nothing knows how
long a stoppage or a possession usually lasts. This decoder scores whole segments instead:

    score(segment [a, b) in state s) = sum_t w_t * e_s(t) + w_dur * log P_s(duration)

plus the M1/M2 transition penalties between segments. Two options:
- duration priors: lognormal per state (stoppages: DFL fit, median 18 s; possessions: a wide
  prior whose median is a parameter, because DFL and hackathon possessions differ ~2x);
- gap weighting: in a ball gap of G frames each frame weighs min(1, kappa / G), so the whole gap
  counts as at most ~kappa frames of evidence.

The segment search runs on 5-frame blocks (0.2 s) for speed, then every boundary is refined to the
exact frame (+-2 blocks) from the per-frame evidence.
"""

from dataclasses import dataclass

import numpy as np

from bench.match import FPS, A, D, H, X

STATES = (H, A, D)
DEAD = 2  # state index of D


@dataclass(frozen=True)
class Params:
    switch: float = 10.0
    stop: float = 40.0
    restart: float = 40.0
    w_dur: float = 0.0                  # 0 disables the duration priors
    alive_median_s: float = 6.0
    alive_sigma: float = 1.2
    dead_median_s: float = 18.0
    dead_sigma: float = 0.92
    kappa: float = 0.0                  # gap weighting, in frames; 0 disables it
    block: int = 5
    alive_max_s: float = 180.0          # longer segments are split (self-transition, no cost)
    dead_max_s: float = 600.0
    refine_blocks: int = 2


def gap_weights(ball_missing, kappa):
    """Per-frame evidence weight: min(1, kappa / G) inside a ball gap of G frames, else 1."""
    w = np.ones(len(ball_missing))
    if kappa <= 0:
        return w
    e = np.flatnonzero(np.diff(np.concatenate(([0], ball_missing.astype(np.int8), [0]))))
    for a, b in zip(e[::2], e[1::2]):
        w[a:b] = min(1.0, kappa / (b - a))
    return w


def _log_duration(n_blocks, p, median_s, sigma):
    """log P(duration) for 1..n_blocks blocks, lognormal in seconds, discretised per block."""
    x = np.arange(1, n_blocks + 1) * p.block / FPS
    mu = np.log(median_s)
    logpdf = -np.log(x * sigma * np.sqrt(2 * np.pi)) - (np.log(x) - mu) ** 2 / (2 * sigma ** 2)
    return p.w_dur * (logpdf + np.log(p.block / FPS))


def _transitions(p):
    t = np.zeros((3, 3))  # self-transition 0: only used to continue segments beyond the max
    t[0, 1] = t[1, 0] = -p.switch
    t[0, DEAD] = t[1, DEAD] = -p.stop
    t[DEAD, 0] = t[DEAD, 1] = -p.restart
    return t


def segment_viterbi(block_scores, p):
    """[(start_block, stop_block, state)] maximising segment scores; block_scores (nb, 3)."""
    nb = len(block_scores)
    ce = np.vstack([np.zeros(3), np.cumsum(block_scores, axis=0)])
    dmax = [max(1, int(s * FPS / p.block)) for s in (p.alive_max_s, p.alive_max_s, p.dead_max_s)]
    logdur = [_log_duration(dmax[0], p, p.alive_median_s, p.alive_sigma)] * 2 + [
        _log_duration(dmax[2], p, p.dead_median_s, p.dead_sigma)]
    trans = _transitions(p)

    best = np.full((nb + 1, 3), -np.inf)
    enter = np.full((nb + 1, 3), -np.inf)
    enter_from = np.zeros((nb + 1, 3), dtype=np.int8)
    back_d = np.zeros((nb + 1, 3), dtype=np.int32)
    enter[0] = 0.0
    g = np.full((nb + 1, 3), -np.inf)  # enter[j, s] - ce[j, s]
    g[0] = -ce[0]
    for c in range(1, nb + 1):
        for s in range(3):
            d_hi = min(c, dmax[s])
            vals = g[c - 1::-1, s][:d_hi] + logdur[s][:d_hi]  # j = c-1 .. c-d_hi
            k = int(np.argmax(vals))
            back_d[c, s] = k + 1
            best[c, s] = vals[k] + ce[c, s]
        cand = best[c][:, None] + trans
        enter_from[c] = cand.argmax(0)
        enter[c] = cand.max(0)
        g[c] = enter[c] - ce[c]

    segs, c, s = [], nb, int(np.argmax(best[nb]))
    while c > 0:
        d = int(back_d[c, s])
        segs.append((c - d, c, s))
        c, s = c - d, int(enter_from[c - d, s])
    segs.reverse()
    merged = []  # self-transitions split long segments; join them back
    for a, b, s in segs:
        if merged and merged[-1][2] == s:
            merged[-1] = (merged[-1][0], b, s)
        else:
            merged.append((a, b, s))
    return merged


def _refine(segs, cum, radius):
    """Move each boundary to the frame (+-radius) best splitting the two neighbours' evidence."""
    out = [list(x) for x in segs]
    for i in range(len(out) - 1):
        left, right = out[i], out[i + 1]
        lo = max(left[0] + 1, left[1] - radius)
        hi = min(right[1] - 1, left[1] + radius)
        if hi < lo:
            continue
        cand = np.arange(lo, hi + 1)
        gain = cum[cand, left[2]] - cum[cand, right[2]]
        b = int(cand[np.argmax(gain)])
        left[1] = right[0] = b
    return out


def decode_scores(match, scores, p=Params()):
    """(T,) codes from (T, 3) per-frame log-scores; X outside the graded halves."""
    w = gap_weights(np.isnan(match.ball[:, 0]), p.kappa)
    weighted = scores * w[:, None]
    out = np.full(match.n_frames, X, dtype=np.int8)
    codes = np.asarray(STATES, dtype=np.int8)
    for _, first, last in match.halves:
        e = weighted[first:last + 1]
        n = len(e)
        nb = -(-n // p.block)
        padded = np.vstack([e, np.zeros((nb * p.block - n, 3))])
        blocks = padded.reshape(nb, p.block, 3).sum(1)
        segs = [(a * p.block, min(n, b * p.block), s) for a, b, s in segment_viterbi(blocks, p)]
        cum = np.vstack([np.zeros(3), np.cumsum(e, axis=0)])
        for a, b, s in _refine(segs, cum, p.refine_blocks * p.block):
            out[first + a:first + b] = codes[s]
    return out


def predict(match):
    from . import learned
    probs = learned.probabilities(match)
    return decode_scores(match, np.log(np.clip(probs, 1e-4, 1.0)), Params())
