"""MP4 preview of one scenario clip: the pitch as the method sees it, plus label strips.

Top: degraded tracking (what methods get), home blue / away red, tracked ball black, the TRUE ball
as a faint grey ring (so you can see where it really is during a tracker gap). Bottom: GT and each
method's labels for the clip, with a cursor. External (DFL) data only -- never hackathon frames.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

from .match import FPS  # noqa: E402

TEAM = ("#1f5fbf", "#d1495b")
LABELS = ListedColormap(["#1f5fbf", "#d1495b", "#9aa0a6", "#ffffff"])  # H A D X
STEP = 2  # render every 2nd frame: 12.5 fps keeps files small and motion readable


def _pitch(ax, length, width):
    ax.set_facecolor("#e8f1e4")
    line = dict(color="#7a8f74", lw=1, fill=False)
    ax.add_patch(Rectangle((0, 0), length, width, **line))
    ax.plot([length / 2] * 2, [0, width], color="#7a8f74", lw=1)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, **line))
    for x, sign in ((0, 1), (length, -1)):
        ax.add_patch(Rectangle((x if sign > 0 else x - 16.5, width / 2 - 20.16), 16.5, 40.32,
                               **line))
        ax.add_patch(Rectangle((x if sign > 0 else x - 5.5, width / 2 - 9.16), 5.5, 18.32, **line))
        ax.add_patch(Arc((x + sign * 11, width / 2), 18.3, 18.3, theta1=-53 if sign > 0 else 127,
                         theta2=53 if sign > 0 else 233, color="#7a8f74", lw=1))
    ax.set_xlim(-3, length + 3)
    ax.set_ylim(width + 3, -3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render(clip, scenario, ctx, preds, verdicts, path):
    """preds: {method: (T,) codes}; verdicts: {method: bool} for this clip."""
    seen, clean = ctx.seen, ctx.clean
    m = np.asarray(seen.pitch)
    lo, hi = clip.start, clip.stop
    frames = range(lo, hi, STEP)

    rows = [("truth", ctx.gt[lo:hi])] + [(k, v[lo:hi]) for k, v in preds.items()]
    fig = plt.figure(figsize=(6.4, 5.6), dpi=100)
    ax = fig.add_axes([0.02, 0.30, 0.96, 0.58])
    _pitch(ax, *m)
    strips = []
    for i, (name, codes) in enumerate(rows):
        sa = fig.add_axes([0.16, 0.20 - i * 0.06, 0.80, 0.045])
        sa.pcolormesh(np.arange(lo, hi + 1) / FPS, [0, 1], codes[None, :], cmap=LABELS,
                      vmin=0, vmax=3)
        sa.set_yticks([])
        sa.set_xticks([])
        tag = "" if name == "truth" else ("  PASS" if verdicts[name] else "  FAIL")
        sa.text(-0.01, 0.5, name + tag, transform=sa.transAxes, ha="right", va="center",
                fontsize=8, color="#2e7d32" if tag == "  PASS" else
                ("#c62828" if tag else "#222"))
        sa.axvline(clip.anchor / FPS, color="#000", lw=0.8, ls=":")
        strips.append(sa.axvline(lo / FPS, color="#000", lw=1.5))

    fig.text(0.02, 0.965, scenario.title, fontsize=11, weight="bold")
    fig.text(0.02, 0.925, "Expected: " + scenario.expected, fontsize=7.5, wrap=True)
    players = [ax.scatter([], [], s=28, c=TEAM[t], edgecolors="white", linewidths=0.5, zorder=3)
               for t in (0, 1)]
    true_ball = ax.scatter([], [], s=90, facecolors="none", edgecolors="#888", linewidths=1.2,
                           zorder=4)
    ball = ax.scatter([], [], s=22, c="#111", zorder=5)
    clock = ax.text(1, -1.2, "", fontsize=8)

    writer = FFMpegWriter(fps=FPS / STEP, bitrate=600)
    with writer.saving(fig, path, dpi=100):
        for t in frames:
            for team, sc in enumerate(players):
                xy = seen.players[t, seen.slot_team == team, :2] * m
                sc.set_offsets(xy[~np.isnan(xy[:, 0])] if len(xy) else np.empty((0, 2)))
            true_ball.set_offsets((clean.ball[t, :2] * m)[None, :])
            b = seen.ball[t, :2]
            ball.set_offsets((b * m)[None, :] if not np.isnan(b[0]) else np.empty((0, 2)))
            clock.set_text("{} t={:+.1f}s{}".format(
                ctx.name, (t - clip.anchor) / FPS,
                "   ball not tracked" if np.isnan(b[0]) else ""))
            for s in strips:
                s.set_xdata([t / FPS] * 2)
            writer.grab_frame()
    plt.close(fig)
