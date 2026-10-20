"""Offline test bench replicating the hackathon setting on external labelled matches.

Paths are resolved here once. The organisers' starter repo is the reference for scoring and
constants; it is expected next to this repo unless MTAH_STARTER_DIR says otherwise.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER_DIR = os.environ.get("MTAH_STARTER_DIR") or os.path.join(
    os.path.dirname(ROOT), "more-than-a-hack-2027")
EXTERNAL_DIR = os.path.join(ROOT, "external")
REPORTS_DIR = os.path.join(ROOT, "reports")
# The gated match bundle (gitignored). games.py reads MTAH_DATA_DIR at import time.
OFFICIAL_DATA_DIR = os.path.join(ROOT, "official_data")


def import_starter():
    """Make `games` and `possession_metrics` importable from the organisers' repo, with the
    match data looked up in official_data/ unless MTAH_DATA_DIR says otherwise."""
    if not os.environ.get("MTAH_DATA_DIR") and os.path.isdir(OFFICIAL_DATA_DIR):
        os.environ["MTAH_DATA_DIR"] = OFFICIAL_DATA_DIR
    if not os.path.isfile(os.path.join(STARTER_DIR, "games.py")):
        raise IOError("organisers' starter repo not found at {}; set MTAH_STARTER_DIR".format(
            STARTER_DIR))
    for path in (STARTER_DIR, os.path.join(STARTER_DIR, "evaluation")):
        if path not in sys.path:
            sys.path.insert(0, path)
