"""Self-training on the real match: teach the evidence model what the REAL tracking looks like.

Every model sees fewer hand-overs on the real tracking than on the bench, so the bench's
simulated noise misses something. The real match has no labels, but ~59% of its frames show the
ball, and there the hackathon-style rule (bench/relabel.py) can name the team in control:

1. scaffold: the submission's own dead / live call and restarting team (learned.predict);
2. pseudo-labels: the hackathon-style rule on top of it (possession changes only on a control
   by the other team), on the real tracking;
3. confident rows only: live frames during a control episode or within KEEP_S after one by the
   labelled team; dead frames where the model is sure (P(dead) >= DEAD_P);
4. the ball is then hidden on gap-shaped spans (degrader's gap generator, EXTRA_MISSING more of
   the frames), features are recomputed on the masked match, and the rows (plus their home /
   away swap) join the training set with weight `weight`. So the model learns to read the real
   players' movement without the ball, where it fails today.

The public window (evaluation/gt) is excluded, so it stays a held-out check. No ground truth is
used anywhere: labels come from an algorithm on the provided tracking (RULES.md allows it).
Deterministic (seeded). PRIVACY: everything stays local; only aggregate counts are printed.
"""

from dataclasses import replace

import numpy as np

import os

from bench import ROOT
from bench.match import FPS, A, D, H, X

PUBLIC = (11185, 18684)      # public GT window: never used for training
KEEP_S = 1.0
DEAD_P = 0.95
EXTRA_MISSING = 0.35
DRAWS = 2
SUBSAMPLE = 3
SCAFFOLD_DIR = os.path.join(ROOT, "models")   # the submission model makes the scaffold


def real_match():
    """The real match, labels dropped (the public GT is never read here)."""
    from bench.__main__ import match_paths
    from bench.match import Match
    from bench import epts
    paths = match_paths(["hackathon"])
    m = Match.load(paths[0]) if paths else epts.hackathon(save=False)
    return m.without_labels()


def pseudo_labels(match, probs=None):
    """(labels (T,) H/A/D/X, keep (T,) bool): pseudo-labels and the confident frames."""
    from bench import relabel
    from bench.match import graded_mask
    from . import learned
    view = match.without_labels()
    probs = learned.probabilities(view, SCAFFOLD_DIR) if probs is None else probs
    scaffold = learned.decode(view, learned.Params(), probs)    # the submission's decoder too
    lab = relabel.hackathon_labels(replace(view, labels=scaffold))
    control = relabel._control_team(view, relabel.Rule())
    # frames since the last control by the labelled team
    last = np.full(len(lab), -1)
    t_last = {H: -10 ** 9, A: -10 ** 9}
    for t in range(len(lab)):
        if control[t] in (H, A):
            t_last[control[t]] = t
        if lab[t] in (H, A):
            last[t] = t - t_last[lab[t]]
    live_ok = np.isin(lab, (H, A)) & (last >= 0) & (last <= KEEP_S * FPS)
    dead_ok = (lab == D) & (probs[:, learned.CLASS_OF[D]] >= DEAD_P)
    # no more confident dead frames than live ones (evenly spaced), or "dead" dominates the rows
    dead_idx = np.flatnonzero(dead_ok)
    n_live = int(live_ok.sum())
    if len(dead_idx) > n_live > 0:
        dead_ok[:] = False
        dead_ok[dead_idx[np.linspace(0, len(dead_idx) - 1, n_live).astype(int)]] = True
    keep = (live_ok | dead_ok) & graded_mask(view)
    keep[PUBLIC[0]:PUBLIC[1] + 1] = False
    return np.where(keep, lab, X).astype(np.int8), keep


def rows(match, feature_set, seed=0):
    """Training rows [(X, y, names)] from the real match: DRAWS masked copies, each with its
    home / away swap, confident frames every SUBSAMPLE-th."""
    from bench.degrade import Profile, _ball_gaps
    from . import learned
    lab, keep = pseudo_labels(match)
    live, dead = np.isin(lab, (H, A)), lab == D
    print("self-training pseudo-labels: {} live / {} dead confident frames (public window excluded)"
          .format(int(live.sum()), int(dead.sum())), flush=True)
    base = replace(match.without_labels(), labels=lab)
    already = np.isnan(match.ball[:, 0])
    out = []
    for k in range(DRAWS):
        rng = np.random.default_rng(seed + k)
        hide = _ball_gaps(replace(base, labels=np.where(lab == X, H, lab).astype(np.int8)),
                          Profile(ball_missing_alive=EXTRA_MISSING + already.mean(),
                                  ball_missing_dead=EXTRA_MISSING + already.mean()), rng)
        masked = replace(base, ball=np.where(hide[:, None], np.nan, match.ball).astype(np.float32))
        sel = keep & ((np.arange(match.n_frames) % SUBSAMPLE) == k % SUBSAMPLE)
        for v in (masked, learned.swap_teams(masked)):
            X_, names = learned.context.matrix(learned.frame_features(v.without_labels(), feature_set))
            y = np.vectorize(learned.CLASS_OF.get)(v.labels[sel])
            out.append((X_[sel], y, names))
        print("  draw {}: ball hidden on {:.0%} of graded frames".format(
            k, np.isnan(masked.ball[:, 0]).mean()), flush=True)
    return out
