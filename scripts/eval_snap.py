"""M2 with and without hand-over snapping (possession/snap.py), paired by match.

    [MTAH_DFL_LABELS=hackathon] uv run python scripts/eval_snap.py [--seeds 0]

Model from MTAH_MODEL_DIR (default models/). Also the real match's rate and public window.
Aggregates only.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.evaluate import score  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import learned, snap  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    per = {}
    for p in match_paths(["dfl"]):
        c = Match.load(p)
        for seed in args.seeds:
            v = degrade(c, "hackathon", seed=seed + sum(map(ord, c.name))).without_labels()
            base = learned.predict(v)
            for k, out in (("M2", base), ("M2 + snap", snap.refine(v, base))):
                per.setdefault(k, {}).setdefault(c.name, []).append(score(c.labels, out)["spell_f1_mean"])
        print(c.name, flush=True)
    d = np.array([np.mean(per["M2 + snap"][n]) - np.mean(per["M2"][n]) for n in per["M2"]])
    print("labels {}: M2 {:.4f}  M2 + snap {:.4f}  diff {:+.4f} ± {:.4f}  better on {}/{}".format(
        os.environ.get("MTAH_DFL_LABELS", "dfl"),
        np.mean([np.mean(v) for v in per["M2"].values()]),
        np.mean([np.mean(v) for v in per["M2 + snap"].values()]),
        d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), len(d)))
    hp = match_paths(["hackathon"])
    if hp:
        hm = Match.load(hp[0])
        hv = hm.without_labels()
        base = learned.predict(hv)
        for k, out in (("M2", base), ("M2 + snap", snap.refine(hv, base))):
            print("real match {:<10} rate {:.2f}  public spell {:.3f}".format(
                k, spell_rate(hm, out), score(hm.labels, out)["spell_f1_mean"]))


if __name__ == "__main__":
    main()
