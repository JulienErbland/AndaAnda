"""Possessions predicted inside the public window vs the 17 true ones (one count per model).

    MTAH_MODEL_DIR=... uv run python scripts/public_count.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.evaluate import pm  # noqa: E402
from bench.match import X, Match  # noqa: E402
from possession import learned  # noqa: E402

hm = Match.load(match_paths(["hackathon"])[0])
win = hm.labels != X
pred = learned.predict(hm.without_labels())
p = np.where(win, pred, X).astype(np.int64)
g = hm.labels.astype(np.int64)
n_pred = len(pm.possession_spells(p, pm.valid_runs(np.where(win, 0, X).astype(np.int64))))
n_true = len(pm.possession_spells(g, pm.valid_runs(g)))
print("{}: public window possessions predicted {} / true {}".format(learned.MODEL_DIR, n_pred, n_true))
