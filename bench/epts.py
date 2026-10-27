"""Metrica FIFA/EPTS tracking: the hackathon's own format, and Metrica Sample Games 1-3.

One reader for both, driven by the metadata's DataFormatSpecifications. Each spec covers a frame
range and lists, in line order, which player each ';' group is. Sample Game 3 has several specs
(the 22 groups change owner at substitutions); the hackathon ATD is the simple case of one spec
with 46 fixed track slots. Groups are 'x,y' or 'x,y,speed'; missing speed is derived from
positions.

PRIVACY: `hackathon()` reads the gated match data. It is for the user to run locally; nothing
here prints rows.
"""

import json
import os
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from . import EXTERNAL_DIR, ROOT, import_starter
from .match import FPS, A, D, H, X, Match

METRICA_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
               "Sample_Game_3/Sample_Game_3_{}")
METRICA_DIR = os.path.join(EXTERNAL_DIR, "metrica")
METRICA_PATH = os.path.join(METRICA_DIR, "metrica-game3.npz")
METRICA_CSV_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
                   "Sample_Game_{0}/Sample_Game_{0}_{1}")
METRICA_PITCH = (105.0, 68.0)  # games 1-2 ship no metadata; game 3's says 105 x 68
HACKATHON_PATH = os.path.join(ROOT, "data", "hackathon-BAR-ATH.npz")  # data/ is gitignored


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_meta(meta_path):
    """(slot_ids, slot_team, specs, pitch). specs = [(start_frame, end_frame, [slot per group])]."""
    root = ET.parse(meta_path).getroot()
    for elem in root.iter():
        elem.tag = _local(elem.tag)

    team_ids = [t.get("id") for t in root.iter("Team")]
    score = root.find(".//Score")
    home = score.get("idLocalTeam") if score is not None and score.get("idLocalTeam") \
        else team_ids[0]  # FIFA/EPTS convention, and what the organisers' baseline assumes
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    slot_of = {pid: i for i, (pid, _) in enumerate(players)}
    slot_team = np.array([0 if team == home else 1 for _, team in players], dtype=np.int8)
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    channel_kind = {c.get("id"): c.get("channelId") for c in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        groups = spec.find("SplitRegister")  # the ';' register holding one ',' register per slot
        for g in groups.findall("SplitRegister"):
            kinds = [channel_kind[r.get("playerChannelId")] for r in g.findall("PlayerChannelRef")]
            if kinds[:2] != ["x", "y"]:
                raise ValueError("{}: unexpected channel order {}".format(meta_path, kinds))
        slots = [slot_of[channel_player[g.find("PlayerChannelRef").get("playerChannelId")]]
                 for g in groups.findall("SplitRegister")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))

    size = root.find(".//FieldSize")
    pitch = (float(size.findtext("Width")), float(size.findtext("Height")))
    return slot_of, slot_team, specs, pitch


def read_raw(raw_path, specs, n_slots, n_frames):
    """(ball, players) arrays; row = frame_id - 1."""
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    spec_i = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            while specs[spec_i][1] < frame:
                spec_i += 1
            row = frame - 1
            for slot, group in zip(specs[spec_i][2], parts[1].split(";")):
                values = group.split(",")
                players[row, slot, :len(values)] = [float(v) for v in values[:3]]
            values = parts[2].split(",")
            ball[row, :len(values)] = [float(v) for v in values[:3]]
    return ball, players


def derive_speed(xy, pitch):
    """m/s from normalised positions by central difference; NaN where a neighbour is missing."""
    metres = xy * np.asarray(pitch, dtype=np.float32)
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step = metres[2:] - metres[:-2]
    speed[1:-1] = np.sqrt((step ** 2).sum(-1)) * FPS / 2
    return speed


def read_epts(meta_path, raw_path, n_frames=None):
    slot_of, slot_team, specs, pitch = read_meta(meta_path)
    n_frames = n_frames or specs[-1][1]
    ball, players = read_raw(raw_path, specs, len(slot_team), n_frames)
    # The ball's speed is always derived from its positions, as on the bench. The hackathon ATD
    # gives a third ball value on only 7.8% of visible-ball frames (none in the public window), and
    # it is not a speed (correlation 0.06 with the speed from positions). Trusting it left every
    # ball-speed feature missing on 92% of real frames (found Oct 27, scripts/domain_shift.py).
    ball[:, 2] = derive_speed(ball[:, :2], pitch)
    if np.isnan(players[..., 2]).all():
        players[..., 2] = derive_speed(players[..., :2], pitch)
    return ball, players, slot_team, pitch


# -- Metrica Sample Game 3 ---------------------------------------------------------------------

# Events that say "this team has the ball now". BALL LOST / CHALLENGE / CARD are ignored: a loose
# or contested ball stays with the last team that had it, as in the hackathon definition.
POSSESSION_EVENTS = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def metrica_labels(events, n_frames, home_team_id):
    """(labels, halves) derived from synchronised events. APPROXIMATE -- the hackathon GT comes
    from corrected tracking, these come from event timing, so boundaries are event-precise
    at best. Use Metrica mainly for format and robustness checks, DFL for scoring.

    Dead from the frame a BALL OUT ends (ball crosses the line), a foul, or a goal, until the next
    possession event (normally the SET PIECE restart). Graded from each period's kick-off to its
    last event.
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    halves = []
    for period in (1, 2):
        evs = sorted((e for e in events if e["period"] == period),
                     key=lambda e: e["start"]["frame"])
        points = []  # (row, state)
        prev_end = None
        for e in evs:
            kind = e["type"]["name"]
            team = H if e["team"]["id"] == home_team_id else A
            start, end = e["start"]["frame"] - 1, e["end"]["frame"] - 1
            if kind == "SET PIECE" and prev_end is not None and prev_end < start:
                # Every restart ends a stoppage. Fouls and offsides often carry no event of their
                # own, so the stoppage starts when the last action before the restart ended.
                points.append((prev_end, D))
            prev_end = end
            if kind in POSSESSION_EVENTS:
                points.append((start, team))
            if kind == "BALL OUT":
                points.append((end, D))
            elif kind == "FAULT RECEIVED":
                points.append((start, D))
            elif kind == "SHOT" and "GOAL" in json.dumps(e.get("subtypes")):
                points.append((end, D))
        points.sort(key=lambda p: p[0])  # stable: same-frame order follows the event list
        first, last = points[0][0], max(e["end"]["frame"] for e in evs) - 1
        for (row, state), nxt in zip(points, points[1:] + [(last + 1, None)]):
            labels[row:nxt[0]] = state
        halves.append((period, first, last))
    return labels, np.asarray(halves, dtype=np.int64)


def fetch_metrica():
    raw = os.path.join(METRICA_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    paths = {}
    for name in ("metadata.xml", "tracking.txt", "events.json"):
        paths[name] = os.path.join(raw, "Sample_Game_3_" + name)
        if not os.path.isfile(paths[name]):
            print("metrica: downloading {} ...".format(name))
            urllib.request.urlretrieve(METRICA_URL.format(name), paths[name])

    ball, players, slot_team, pitch = read_epts(paths["metadata.xml"], paths["tracking.txt"])
    with open(paths["events.json"]) as handle:
        events = json.load(handle)["data"]
    root = ET.parse(paths["metadata.xml"]).getroot()
    home_id = next(e.get("idLocalTeam") for e in root.iter() if _local(e.tag) == "Score")
    labels, halves = metrica_labels(events, len(ball), home_id)
    match = Match(name="metrica-game3", pitch=pitch, ball=ball, players=players,
                  slot_team=slot_team, halves=halves, labels=labels)
    match.save(METRICA_PATH)
    print("metrica: {} frames, {} slots -> {}".format(len(ball), len(slot_team), METRICA_PATH))


def read_metrica_csv(home_csv, away_csv):
    """(ball, players, slot_team) from the Sample Game 1-2 CSV pair: 3 header lines, then
    Period, Frame, Time, x/y per player (one fixed column pair per player, substitutes NaN while
    off), Ball x/y. Same 0-1 coordinates as the EPTS files."""
    import pandas as pd
    halves_xy = []
    for path in (home_csv, away_csv):
        a = pd.read_csv(path, header=None, skiprows=3).to_numpy(np.float32)
        halves_xy.append(a)
    home, away = halves_xy
    if len(home) != len(away) or not np.array_equal(home[:, 1], away[:, 1]):
        raise ValueError("home and away tracking files are not frame-aligned")
    ball = home[:, -2:]
    teams = [h[:, 3:-2].reshape(len(h), -1, 2) for h in (home, away)]
    xy = np.concatenate(teams, axis=1)
    slot_team = np.repeat([0, 1], [teams[0].shape[1], teams[1].shape[1]]).astype(np.int8)
    players = np.concatenate([xy, derive_speed(xy, METRICA_PITCH)[..., None]], axis=-1)
    ball = np.concatenate([ball, derive_speed(ball, METRICA_PITCH)[:, None]], axis=-1)
    return ball, players, slot_team


def csv_events(events_csv):
    """Sample Game 1-2 events in the shape metrica_labels() reads from the game 3 JSON."""
    import pandas as pd
    out = []
    for r in pd.read_csv(events_csv).itertuples(index=False):
        out.append({"type": {"name": r.Type}, "team": {"id": r.Team}, "period": int(r.Period),
                    "start": {"frame": int(r[4])}, "end": {"frame": int(r[6])},
                    "subtypes": None if pd.isna(r.Subtype) else r.Subtype,
                    "from": r.From, "to": None if pd.isna(r.To) else r.To})
    return out


def fetch_metrica_csv(game):
    raw = os.path.join(METRICA_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    paths = {}
    for name in ("RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv",
                 "RawEventsData.csv"):
        paths[name] = os.path.join(raw, "Sample_Game_{}_{}".format(game, name))
        if not os.path.isfile(paths[name]):
            print("metrica: downloading game {} {} ...".format(game, name))
            urllib.request.urlretrieve(METRICA_CSV_URL.format(game, name), paths[name])
    ball, players, slot_team = read_metrica_csv(paths["RawTrackingData_Home_Team.csv"],
                                                paths["RawTrackingData_Away_Team.csv"])
    labels, halves = metrica_labels(csv_events(paths["RawEventsData.csv"]), len(ball), "Home")
    name = "metrica-game{}".format(game)
    out = os.path.join(METRICA_DIR, name + ".npz")
    Match(name=name, pitch=METRICA_PITCH, ball=ball, players=players, slot_team=slot_team,
          halves=halves, labels=labels).save(out)
    print("metrica: {} frames, {} slots -> {}".format(len(ball), len(slot_team), out))


# -- the hackathon match (gated data: run locally) ---------------------------------------------

def hackathon(save=True):
    """BAR-ATH tactical ATD in the common format, with the public 5-minute GT as labels."""
    import_starter()
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    import possession_metrics as pm

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical", GAME)
    ball, players, slot_team, _ = read_epts(meta_path, raw_path, n_frames)
    gt = os.path.join(os.path.dirname(pm.__file__), "gt", "BAR-ATH_possession_gt_public.csv")
    labels = pm.label_codes(pm.read_label_csv(gt)).astype(np.int8)
    match = Match(name="hackathon-BAR-ATH", pitch=GAMES[GAME]["pitch"], ball=ball,
                  players=players, slot_team=slot_team,
                  halves=np.asarray(half_video_blocks(GAME), dtype=np.int64), labels=labels)
    if save:
        os.makedirs(os.path.dirname(HACKATHON_PATH), exist_ok=True)
        match.save(HACKATHON_PATH)
        print("hackathon: {} frames, {} slots -> {}".format(n_frames, len(slot_team),
                                                           HACKATHON_PATH))
    return match
