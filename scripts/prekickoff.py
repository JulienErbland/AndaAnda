"""Second-half pre-kickoff: graded 110,808-113,741, all D in the GT (organisers' FAQ). How does
the model in MTAH_MODEL_DIR label it on the real match? Aggregates only.

    MTAH_MODEL_DIR=... uv run python scripts/prekickoff.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.match import A, D, H, LETTERS, Match  # noqa: E402
from possession import METHODS  # noqa: E402

FIRST, KICKOFF = 110808, 113742   # FAQ.md

hm = Match.load(match_paths(["hackathon"])[0]).without_labels()
method = sys.argv[1] if len(sys.argv) > 1 else "learned"
pred = METHODS[method](hm)
seg = pred[FIRST:KICKOFF]
counts = {LETTERS[c]: int((seg == c).sum()) for c in (H, A, D)}
live = np.isin(seg, (H, A)).astype(np.int8)
runs = int(np.sum(np.diff(np.concatenate(([0], live))) == 1))
print("{} ({}): pre-kickoff frames {} -> {}; H/A runs {}".format(
    method, os.environ.get("MTAH_MODEL_DIR", "models"), KICKOFF - FIRST, counts, runs))
