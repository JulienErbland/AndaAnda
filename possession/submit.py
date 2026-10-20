"""One command from the data bundle to a validated submission file.

    uv run python -m possession.submit [--method hmm] [--team AndaAnda]

Rebuilds the match from the raw bundle every time (official_data/ or MTAH_DATA_DIR), so the
output depends on the bundle and the code only -- what the organisers will re-run. Deterministic:
no randomness anywhere in the pipeline. Writes submissions/<team>.csv (gitignored: a submission is
frame-level and derived from the gated data) and runs the organisers' validate_submission.py.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from bench import ROOT, STARTER_DIR, epts
from bench.__main__ import resolve_method
from bench.match import LETTERS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", default="hmm")
    ap.add_argument("--team", default="AndaAnda")
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions"))
    args = ap.parse_args()

    match = epts.hackathon(save=False)
    pred = resolve_method(args.method)(match.without_labels())
    if pred.shape != (match.n_frames,):
        sys.exit("method returned shape {}, expected ({},)".format(pred.shape, match.n_frames))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "{}.csv".format(args.team))
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[pred]))
    print("wrote {} ({} rows, method {})".format(os.path.relpath(path, ROOT), len(pred),
                                                 args.method))
    validator = os.path.join(STARTER_DIR, "evaluation", "validate_submission.py")
    sys.exit(subprocess.call([sys.executable, validator, path]))


if __name__ == "__main__":
    main()
