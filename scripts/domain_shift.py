"""Adversarial validation: can a classifier tell the bench's frames from the real match's?

    uv run python scripts/domain_shift.py

Rows: the submission's per-frame features (feature set "restart") on graded frames, every
STEP-th frame, from the 7 degraded DFL matches (bench, label 0) and the real match (label 1).
LightGBM, 5 folds grouped by 1-minute blocks (neighbouring frames never straddle train / test).
Prints aggregates only: AUC overall and after dropping the most shifted features, per-feature
univariate AUC, and the classifier's top features. AUC 0.5 = indistinguishable.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import load_degraded, match_paths  # noqa: E402
from bench.match import FPS, graded_mask  # noqa: E402
from possession import context, learned  # noqa: E402

STEP = 10
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.8, verbose=-1, seed=0, deterministic=True, force_row_wise=True,
           num_threads=4)


def rows(match, label, gid):
    X_, names = context.matrix(learned.frame_features(match, "restart"))
    keep = np.flatnonzero(graded_mask(match))[::STEP]
    groups = np.array(["{}-{}".format(gid, t // (60 * FPS)) for t in keep])
    return X_[keep], np.full(len(keep), label), groups, names


def auc(y, s):
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def cv_auc(X_, y, groups, cols):
    import lightgbm as lgb
    ug = np.unique(groups)
    fold = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    s = np.zeros(len(y))
    imp = np.zeros(len(cols))
    for k in range(5):
        tr, te = f != k, f == k
        b = lgb.train(LGB, lgb.Dataset(X_[tr][:, cols], y[tr]), 200)
        s[te] = b.predict(X_[te][:, cols])
        imp += b.feature_importance("gain")
    return auc(y, s), imp / imp.sum()


def main():
    parts = [rows(load_degraded(p, "hackathon", 0).without_labels(), 0, i)
             for i, p in enumerate(match_paths(["dfl"]))]
    hm = load_degraded(match_paths(["hackathon"])[0], "hackathon", 0).without_labels()
    parts.append(rows(hm, 1, "real"))
    names = parts[0][3]
    X_ = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    print("rows: bench {} / real {} ({} features)".format((y == 0).sum(), (y == 1).sum(), len(names)))
    uni = []
    for j, n in enumerate(names):
        v = np.nan_to_num(X_[:, j], nan=-1e9)
        uni.append((abs(auc(y, v) - 0.5), n))
    uni.sort(reverse=True)
    cols = list(range(len(names)))
    a, imp = cv_auc(X_, y, groups, cols)
    print("domain classifier AUC, all features: {:.3f}".format(a))
    print("top features by classifier gain:")
    for j in np.argsort(-imp)[:12]:
        print("  {:<28} gain {:5.1%}".format(names[j], imp[j]))
    print("most shifted single features (|univariate AUC - 0.5|):")
    for d, n in uni[:12]:
        print("  {:<28} {:.3f}".format(n, d))
    dropped = []
    for k in (3, 6, 10):
        dropped = [names[j] for j in np.argsort(-imp)[:k]]
        keep = [j for j, n in enumerate(names) if n not in dropped]
        print("AUC without the top {} features: {:.3f}".format(k, cv_auc(X_, y, groups, keep)[0]))


if __name__ == "__main__":
    main()
