"""Review D: learned, time-varying switch costs.

A second LightGBM predicts "a possession change (H <-> A) within ±WINDOW_S" per frame, from the
same features as the evidence model, trained on the bench's hackathon-style labels. The decoder's
H <-> A cost at frame t becomes

    switch - beta * (logit q_t - logit q_base),   clipped to [MIN_COST, MAX_FACTOR * switch]

so changes are cheaper where the detector expects one and dearer where it does not; beta = 0 is
the constant penalty. Unlike the dropped touch discounts (a hand-made rule), the cost is learned,
and it also raises the price of changes nothing supports.

    MTAH_DFL_LABELS=hackathon uv run python -m possession.changes train   # models_exp/changes
"""

import json
import os
import sys

import numpy as np

from bench import ROOT
from bench.match import FPS, A, H, X, graded_mask

from . import context, hmm, learned

WINDOW_S = 0.5
MODEL_DIR = os.environ.get("MTAH_CHANGE_DIR") or os.path.join(ROOT, "models_exp", "changes")
FEATURE_SET = "restart"
MIN_COST = 0.5
MAX_FACTOR = 3.0
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
           feature_fraction=0.8, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True,
           force_row_wise=True, num_threads=4)
ROUNDS = 300


def target(labels):
    """(T,) 1 within WINDOW_S of an H <-> A change of `labels`."""
    ch = np.flatnonzero(((labels[1:] == H) & (labels[:-1] == A)) | ((labels[1:] == A) & (labels[:-1] == H))) + 1
    out = np.zeros(len(labels), np.int8)
    w = int(round(WINDOW_S * FPS))
    for t in ch:
        out[max(0, t - w):t + w + 1] = 1
    return out


def _rows(path):
    from bench.__main__ import load_degraded
    from bench.match import Match
    clean = Match.load(path)
    rows = []
    for v in (load_degraded(path, "hackathon", 0), None):
        if v is None:
            v = learned.swap_teams(load_degraded(path, "hackathon", 0))
        X_, names = context.matrix(learned.frame_features(v.without_labels(), FEATURE_SET))
        keep = graded_mask(clean) & (clean.labels != X) & ((np.arange(clean.n_frames) % learned.SUBSAMPLE) == 0)
        rows.append((X_[keep], target(clean.labels)[keep], names))
    return clean.name, rows


def train(workers=4):
    import lightgbm as lgb
    from multiprocessing import Pool
    from bench.__main__ import match_paths
    with Pool(workers) as pool:
        data = dict(pool.imap(_rows, match_paths(["dfl"])))
    os.makedirs(os.path.join(MODEL_DIR, "lomo"), exist_ok=True)

    def fit(names):
        rs = [r for n in names for r in data[n]]
        y = np.concatenate([r[1] for r in rs])
        return lgb.train(LGB, lgb.Dataset(np.concatenate([r[0] for r in rs]), y,
                                          feature_name=rs[0][2]), ROUNDS), y.mean()

    full, base = fit(sorted(data))
    full.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump({"trained_on": sorted(data), "base_rate": float(base), "window_s": WINDOW_S,
                   "feature_set": FEATURE_SET}, f, indent=1)
    for held in sorted(data):
        fit([n for n in data if n != held])[0].save_model(os.path.join(MODEL_DIR, "lomo", held + ".txt"))
        print("  without", held, flush=True)
    print("change model -> {} (base rate {:.3f})".format(os.path.relpath(MODEL_DIR, ROOT), base))


def probability(match):
    """(T,) P(change within WINDOW_S), held-out for training matches."""
    import lightgbm as lgb
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    path = os.path.join(MODEL_DIR, "lomo", match.name + ".txt") if match.name in meta["trained_on"] \
        else os.path.join(MODEL_DIR, "lgbm.txt")
    X_, _ = context.matrix(learned.frame_features(match, FEATURE_SET))
    return lgb.Booster(model_file=path).predict(X_), meta["base_rate"]


def costs(q, base, switch, beta):
    """(T, 2) switch-into-H / into-A costs from change probabilities q."""
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))  # noqa: E731
    c = np.clip(switch - beta * (lg(q) - lg(base)), MIN_COST, MAX_FACTOR * switch)
    return np.stack([c, c], axis=1)


def decode(match, p=None, probs=None, beta=1.0, q=None):
    p = p or learned.Params()
    probs = learned.probabilities(match) if probs is None else probs
    if q is None:
        q, base = probability(match)
    else:
        q, base = q
    sc = learned.scores(match, p, probs)
    return hmm.decode_scores(match, sc, hmm.Params(switch=p.switch, stop=p.stop, restart=p.restart),
                             costs(q, base, p.switch, beta))


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
