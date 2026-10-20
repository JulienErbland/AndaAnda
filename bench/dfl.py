"""DFL / IDSSE open data (Bassek et al., Sci Data 2025, CC-BY 4.0): 7 matches, TRACAB 25 fps.

Why this is the main benchmark: every ball frame carries BallStatus (1 alive, 0 dead) and
BallPossession (1 home, 2 away). The hackathon ground truth is built the same way -- mechanically,
from a dead flag and an owning team in corrected tracking -- so these labels are the closest
public analogue we have.

The position XML is ~350-420 MB per match and disk is tight, so `fetch` downloads one match,
converts it to a compact npz and deletes the XML before the next one. Parsing is a streaming
iterparse rather than kloppy, whose sportec loader holds the whole document in memory.
"""

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from . import EXTERNAL_DIR
from .match import A, D, H, X, Match

ARTICLE_API = "https://api.figshare.com/v2/articles/28196177"
DFL_DIR = os.path.join(EXTERNAL_DIR, "dfl")
HALVES = (("firstHalf", 1), ("secondHalf", 2))
# Frames of NaN/X between the halves, so no possession can straddle half-time.
HALFTIME_GAP = 250
KMH_TO_MS = 1 / 3.6


def match_path(match_id):
    return os.path.join(DFL_DIR, "{}.npz".format(match_id))


def list_files():
    """{match_id: {"info": url, "positions": url}} from the figshare article."""
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:
        files = json.load(response)["files"]
    out = {}
    for f in files:
        found = re.search(r"(matchinformation|positions_raw_observed).*_(DFL-MAT-\w+)\.xml$",
                          f["name"])
        if found:
            kind = "info" if found.group(1) == "matchinformation" else "positions"
            out.setdefault(found.group(2), {})[kind] = f["download_url"]
    return out


def fetch(match_ids=None, keep_raw=False):
    """Download, convert and (unless keep_raw) delete the XML. Skips matches already converted."""
    os.makedirs(os.path.join(DFL_DIR, "raw"), exist_ok=True)
    files = list_files()
    for match_id in match_ids or sorted(files):
        if os.path.isfile(match_path(match_id)):
            print("{}: already converted".format(match_id))
            continue
        raw = {kind: os.path.join(DFL_DIR, "raw", "{}_{}.xml".format(kind, match_id))
               for kind in ("info", "positions")}
        for kind, path in raw.items():
            if not os.path.isfile(path):
                print("{}: downloading {} ...".format(match_id, kind))
                urllib.request.urlretrieve(files[match_id][kind], path + ".part")
                os.replace(path + ".part", path)
        print("{}: parsing ...".format(match_id))
        match = parse(raw["info"], raw["positions"], match_id)
        match.save(match_path(match_id))
        if not keep_raw:
            for path in raw.values():
                os.remove(path)
        print("{}: {} frames, {} slots -> {}".format(
            match_id, match.n_frames, len(match.slot_team), match_path(match_id)))


def read_info(info_path):
    """(home_team_id, guest_team_id, (pitch_length, pitch_width))."""
    root = ET.parse(info_path).getroot()
    env = root.find(".//Environment")
    roles = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    if set(roles) != {"home", "guest"}:
        raise ValueError("{}: expected a home and a guest team, got {}".format(info_path, roles))
    return roles["home"], roles["guest"], (float(env.get("PitchX")), float(env.get("PitchY")))


def read_framesets(positions_path):
    """{(person_or_BALL, half): dict of arrays} streamed from the position XML."""
    sets = {}
    current = None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            half = dict(HALVES)[elem.get("GameSection")]
            key = ("BALL" if elem.get("TeamId") == "BALL" else elem.get("PersonId"), half)
            current = {"team": elem.get("TeamId"), "N": [], "X": [], "Y": [], "S": [],
                       "status": [], "possession": []}
            sets[key] = current
        elif event == "end" and elem.tag == "Frame":
            current["N"].append(int(elem.get("N")))
            current["X"].append(float(elem.get("X")))
            current["Y"].append(float(elem.get("Y")))
            current["S"].append(float(elem.get("S")))
            if key[0] == "BALL":
                current["status"].append(int(elem.get("BallStatus")))
                current["possession"].append(int(elem.get("BallPossession")))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()
    return sets


def parse(info_path, positions_path, name):
    home_team, guest_team, (length, width) = read_info(info_path)
    # Some matches also track the referees under their own TeamId: they are not players, and a
    # referee counted as an away player corrupts every nearest-player feature.
    sets = {k: v for k, v in read_framesets(positions_path).items()
            if k[0] == "BALL" or v["team"] in (home_team, guest_team)}

    # The ball frameset spans each half completely, so it defines the timeline.
    starts, lengths, offset = {}, {}, 0
    halves = []
    for _, half in HALVES:
        n = np.asarray(sets[("BALL", half)]["N"])
        starts[half], lengths[half] = int(n.min()), int(n.max() - n.min() + 1)
        halves.append((half, offset, offset + lengths[half] - 1))
        offset += lengths[half] + HALFTIME_GAP
    n_frames = offset - HALFTIME_GAP

    def rows(half, frames):
        return halves[half - 1][1] + np.asarray(frames) - starts[half]

    def normalise(fs):
        # Centre-origin metres -> 0-1 from a corner, y flipped so (0,0) is top-left like Metrica.
        x = (np.asarray(fs["X"]) + length / 2) / length
        y = (width / 2 - np.asarray(fs["Y"])) / width
        return x, y, np.asarray(fs["S"]) * KMH_TO_MS

    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    labels = np.full(n_frames, X, dtype=np.int8)
    for _, half in HALVES:
        fs = sets[("BALL", half)]
        r = rows(half, fs["N"])
        ball[r, 0], ball[r, 1], ball[r, 2] = normalise(fs)
        status, possession = np.asarray(fs["status"]), np.asarray(fs["possession"])
        labels[r] = np.where(status == 0, D, np.where(possession == 1, H, A))

    persons = sorted({k[0] for k in sets if k[0] != "BALL"},
                     key=lambda p: (_team_of(sets, p) != home_team, p))
    slot_team = np.array([0 if _team_of(sets, p) == home_team else 1 for p in persons],
                         dtype=np.int8)
    players = np.full((n_frames, len(persons), 3), np.nan, dtype=np.float32)
    for slot, person in enumerate(persons):
        for _, half in HALVES:
            fs = sets.get((person, half))
            if fs:
                r = rows(half, fs["N"])
                players[r, slot, 0], players[r, slot, 1], players[r, slot, 2] = normalise(fs)

    return Match(name="dfl-" + name, pitch=(length, width), ball=ball, players=players,
                 slot_team=slot_team, halves=np.asarray(halves, dtype=np.int64), labels=labels)


def _team_of(sets, person):
    return next(v["team"] for k, v in sets.items() if k[0] == person)
