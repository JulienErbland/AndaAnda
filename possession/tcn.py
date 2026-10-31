"""Iteration 2: temporal CNN evidence (a dilated, non-causal 1-D CNN; Bai et al. 2018,
arXiv 1803.01271) over the per-frame feature sequence, in place of LightGBM's per-frame trees.
Same features, labels and decoder; a model dir with `tcn.json` is used by learned._single.

    MTAH_MODEL_DIR=models_exp/T_R uv run --group tcn python -m possession.tcn --features restart
    MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models_exp/T_B1 uv run --group tcn python -m possession.tcn

Runs at 5 fps (every STRIDE-th frame); training crops start at a random phase, prediction runs
every phase and interleaves them back to 25 fps. NaN features -> 0 plus a missing indicator.
Fixed budget (STEPS), no early stopping: nothing is tuned on a held-out match.
"""

import argparse
import json
import os

import numpy as np

from bench.match import X, graded_mask

from . import context, learned

STRIDE = 5
CH, BLOCKS, KERNEL, DROPOUT = 64, 6, 3, 0.1          # dilations 1..32: ~25 s receptive field
LR, BATCH, CROP, STEPS, SEED, THREADS = 1e-3, 8, 1024, 1500, 0, 4


def _torch():
    import torch
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    return torch


def _net(n_in):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(CH, CH, KERNEL, dilation=d, padding=d), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Conv1d(CH, CH, KERNEL, dilation=d, padding=d), nn.ReLU(), nn.Dropout(DROPOUT))

        def forward(self, x):
            return x + self.conv(x)

    return nn.Sequential(nn.Conv1d(n_in, CH, 1), *[Block(2 ** i) for i in range(BLOCKS)],
                         nn.Conv1d(CH, 3, 1))


def _inputs(X_, mean, std):
    miss = np.isnan(X_)
    Z = np.where(miss, 0.0, (X_ - mean) / std).astype(np.float32)
    return np.concatenate([Z, miss.astype(np.float32)], 1)


def _seq_job(args):
    """Worker: one degraded draw (as learned._rows_job) -> [(name, X (T, F), y (T,) with -1 off)]."""
    path, k, feature_set, dead_mix = args
    from dataclasses import replace
    from bench.__main__ import load_degraded
    from bench.degrade import PROFILES, degrade
    from bench.match import Match
    if k == 0:
        m = load_degraded(path, "hackathon", 0)
    else:
        clean = Match.load(path)
        prof = replace(PROFILES["hackathon"], dead_ball_moves=False) if dead_mix and k % 2 == 1 else "hackathon"
        m = degrade(clean, prof, seed=k + sum(map(ord, clean.name)))
    out = []
    for v in (m, learned.swap_teams(m)):
        X_, names = context.matrix(learned.frame_features(v.without_labels(), feature_set))
        y = np.full(v.n_frames, -1, np.int64)
        ok = graded_mask(v) & (v.labels != X)
        y[ok] = np.vectorize(learned.CLASS_OF.get)(v.labels[ok])
        out.append((v.name, X_, y, names))
    return out


def _fit(seqs, mean, std, seed):
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [(torch.from_numpy(_inputs(X_, mean, std)), torch.from_numpy(y)) for X_, y in seqs]
    net = _net(data[0][0].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    span = CROP * STRIDE
    net.train()
    for _ in range(STEPS):
        xs, ys = [], []
        for _ in range(BATCH):
            Z, y = data[rng.integers(len(data))]
            t0 = int(rng.integers(0, len(y) - span))
            xs.append(Z[t0:t0 + span:STRIDE].T)
            ys.append(y[t0:t0 + span:STRIDE])
        opt.zero_grad()
        loss = loss_fn(net(torch.stack(xs)), torch.stack(ys))
        loss.backward()
        opt.step()
    net.eval()
    return net


def train(feature_set="restart", draws=2, dead_mix=True, workers=4):
    from multiprocessing import Pool
    from bench.__main__ import match_paths
    torch = _torch()
    jobs = [(p, k, feature_set, dead_mix) for p in match_paths(["dfl"]) for k in range(draws)]
    seqs = {}
    with Pool(workers) as pool:
        for r in pool.imap(_seq_job, jobs):
            for name, X_, y, names in r:
                seqs.setdefault(name, []).append((X_, y))
    names_sorted = sorted(seqs)
    os.makedirs(os.path.join(learned.MODEL_DIR, "lomo"), exist_ok=True)

    def fit_on(keep, seed):
        tr = [s for n in keep for s in seqs[n]]
        allX = np.concatenate([x for x, _ in tr])
        mean, std = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
        mean, std = np.nan_to_num(mean), np.nan_to_num(std, nan=1.0)
        return _fit(tr, mean, std, seed), mean, std

    def save(res, path):
        net, mean, std = res
        torch.save({"state": net.state_dict(), "mean": mean, "std": std}, path)

    save(fit_on(names_sorted, SEED), os.path.join(learned.MODEL_DIR, "full.pt"))
    with open(os.path.join(learned.MODEL_DIR, "tcn.json"), "w") as f:
        json.dump({"trained_on": names_sorted, "feature_set": feature_set, "draws": draws,
                   "dead_mix": dead_mix, "labels": os.environ.get("MTAH_DFL_LABELS", ""),
                   "net": [CH, BLOCKS, KERNEL, DROPOUT], "fit": [LR, BATCH, CROP, STEPS, SEED],
                   "stride": STRIDE}, f, indent=1)
    print("full model done", flush=True)
    for held in names_sorted:
        save(fit_on([n for n in names_sorted if n != held], SEED),
             os.path.join(learned.MODEL_DIR, "lomo", held + ".pt"))
        print("  without", held, flush=True)


def predict(match, model_dir, cache):
    """(T, 3) P(H, A, D), held out for training matches; feature matrix shared via `cache`."""
    torch = _torch()
    with open(os.path.join(model_dir, "tcn.json")) as f:
        meta = json.load(f)
    fs = meta["feature_set"]
    if fs not in cache:
        cache[fs] = context.matrix(learned.frame_features(match, fs))[0]
    path = os.path.join(model_dir, "lomo", match.name + ".pt") if match.name in meta["trained_on"] \
        else os.path.join(model_dir, "full.pt")
    ck = torch.load(path, weights_only=False)
    Z = torch.from_numpy(_inputs(cache[fs], ck["mean"], ck["std"]))
    net = _net(Z.shape[1])
    net.load_state_dict(ck["state"])
    net.eval()
    out = np.empty((len(Z), 3), np.float64)
    with torch.no_grad():
        for ph in range(STRIDE):
            out[ph::STRIDE] = torch.softmax(net(Z[ph::STRIDE].T[None]), 1)[0].T.double().numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="restart")
    a = ap.parse_args()
    train(a.features)
