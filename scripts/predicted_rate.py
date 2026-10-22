"""Predicted vs true possessions per live minute on the DFL bench, for the model in
MTAH_MODEL_DIR (respects MTAH_DFL_LABELS). One seed, default decoder. Aggregates only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.__main__ import match_paths  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.match import Match  # noqa: E402
from possession import learned  # noqa: E402
from scripts.tune_hmm import spell_rate  # noqa: E402

pred_r, true_r = [], []
for p in match_paths(["dfl"]):
    c = Match.load(p)
    v = degrade(c, "hackathon", seed=sum(map(ord, c.name))).without_labels()
    pred_r.append(spell_rate(c, learned.predict(v)))
    true_r.append(spell_rate(c, c.labels))
print("{}: bench predicted {:.2f} / true {:.2f} possessions per live minute".format(
    learned.MODEL_DIR, np.mean(pred_r), np.mean(true_r)))
