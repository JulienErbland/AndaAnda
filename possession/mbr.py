"""Review C: minimum Bayes risk (MBR) decoding for spellF1.

Viterbi returns the single most probable label path, which is not the path with the best
EXPECTED spellF1. MBR approximates the latter, per graded half:

1. sample N_SAMPLES paths from the decoder's posterior by forward-filtering backward-sampling
   (FFBS), with the energies divided by a temperature: the per-frame terms are correlated, so
   the raw posterior is far too confident (TEMPERATURE is chosen on the bench);
2. candidates: the Viterbi path at each of SWITCHES;
3. pick the candidate with the best mean spellF1 against the samples, scored by the organisers'
   own `spell_scores`.

In effect the switch penalty is chosen per half from the model's own uncertainty instead of one
global value. Deterministic (seeded).
"""

import numpy as np

from bench import import_starter
from bench.match import X

from . import hmm, learned

SWITCHES = (5, 10, 14, 20, 30, 40)
N_SAMPLES = 64
TEMPERATURE = 4.0


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sample_paths(scores, trans, n, temp, rng):
    """(n, T) int8 state paths drawn from exp((scores, trans) / temp) by FFBS."""
    s, tr = scores / temp, trans / temp
    T = len(s)
    alpha = np.empty((T, 3))
    alpha[0] = s[0]
    for t in range(1, T):
        alpha[t] = s[t] + _lse(alpha[t - 1][:, None] + tr, axis=0)
    paths = np.empty((n, T), dtype=np.int8)
    g = rng.gumbel(size=(n, 3))
    paths[:, -1] = np.argmax(alpha[-1][None] + g, axis=1)
    for t in range(T - 2, -1, -1):
        logits = alpha[t][None, :] + tr[:, paths[:, t + 1]].T          # (n, 3): from i to next
        paths[:, t] = np.argmax(logits + rng.gumbel(size=(n, 3)), axis=1)
    return paths


def decode(match, p=None, probs=None, temp=TEMPERATURE, switches=SWITCHES, n=N_SAMPLES, seed=0,
           chosen=None):
    """(T,) codes; X outside the graded halves. `chosen`: optional list, receives the switch
    picked for each half."""
    import_starter()
    import possession_metrics as pm
    p = p or learned.Params()
    probs = learned.probabilities(match) if probs is None else probs
    scores = learned.scores(match, p, probs)
    states = np.asarray(hmm.STATES, dtype=np.int8)
    out = np.full(match.n_frames, X, dtype=np.int8)
    rng = np.random.default_rng(seed)
    post = hmm.transitions(hmm.Params(switch=p.switch, stop=p.stop, restart=p.restart))
    for _, first, last in match.halves:
        sc = scores[first:last + 1]
        runs = [(0, len(sc))]
        samples = states[sample_paths(sc, post, n, temp, rng)].astype(np.int64)
        best, best_sw, best_val = None, None, -1.0
        for sw in switches:
            tr = hmm.transitions(hmm.Params(switch=sw, stop=p.stop, restart=p.restart))
            cand = states[hmm.viterbi(sc, tr)].astype(np.int64)
            val = np.mean([pm.spell_scores(smp, cand, runs)["mean_f1"] for smp in samples])
            if val > best_val:
                best, best_sw, best_val = cand, sw, val
        out[first:last + 1] = best
        if chosen is not None:
            chosen.append(best_sw)
    return out


def predict(match):
    return decode(match)
