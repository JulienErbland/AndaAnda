"""Timeline strip for one window: GT, prediction, ball seen, errors. One PNG per window."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .evaluate import clock  # noqa: E402
from .match import FPS, X  # noqa: E402

LABEL_COLOURS = ListedColormap(["#1f5fbf", "#d1495b", "#9aa0a6", "#ffffff"])  # H A D X


def timeline(match, pred, start, length, path):
    stop = min(match.n_frames, start + length)
    gt = match.labels[start:stop]
    pr = pred[start:stop]
    seen = ~np.isnan(match.ball[start:stop, 0])
    wrong = (pr != gt) & (gt != X)

    rows = [("GT", gt, LABEL_COLOURS, 3), ("pred", pr, LABEL_COLOURS, 3),
            ("ball seen", seen.astype(int), ListedColormap(["#ffffff", "#2e7d32"]), 1),
            ("error", wrong.astype(int), ListedColormap(["#ffffff", "#000000"]), 1)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.4), sharex=True)
    t = np.arange(start, stop + 1) / FPS
    for ax, (name, values, cmap, vmax) in zip(axes, rows):
        ax.pcolormesh(t, [0, 1], values[None, :], cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("seconds from frame 0")
    fig.suptitle("{}  frames {}-{}  ({})   H blue, A red, D grey".format(
        match.name, start, stop, clock(match, start)), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
