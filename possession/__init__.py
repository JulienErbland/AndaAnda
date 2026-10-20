"""Possession methods. A method is `predict(match) -> (T,) int codes` (H=0, A=1, D=2, X=3).

It receives `match.without_labels()`, so it can only use what the hackathon gives us.
Register a method in METHODS to make it runnable as `python -m bench run --method <name>`.
"""

from functools import partial

from . import baseline, hmm, learned

METHODS = {
    "raw": partial(baseline.predict, window=0),
    "smooth25": partial(baseline.predict, window=25),
    "smooth50": partial(baseline.predict, window=50),
    "smooth125": partial(baseline.predict, window=125),
    "hmm": hmm.predict,
    "learned": learned.predict,
}
