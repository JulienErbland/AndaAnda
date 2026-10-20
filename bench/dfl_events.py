"""DFL official events for the 7 bench matches, synced to the bench timeline (frame rows).

    uv run python -m bench fetch dfl-events

What they add: real on-ball events (passes, tackles, interceptions, shots) and set pieces
(kick-off, throw-in, corner, free kick, goal kick, penalty) with the team that takes them. They
are the reference for the restart detector (direction 1) and the touch detector (direction 2).

Sync, in two steps:
1. Clock -> frame. 26-ish events per match carry CalculatedFrame + CalculatedTimestamp on the
   tracking clock (frame 12004 is exactly 80.16 s after frame 10000). They fix the frame <-> time
   map of each half. Halves without one fall back to aligning the kick-off event with the first
   live frame of that half.
2. Lag. EventTime runs 0.3-2 s late on the tracking clock, by a roughly constant amount per
   match (measured on the anchors: a real kick sits exactly on every anchor frame). The median
   lag per half is subtracted.
3. Snap. Each on-ball event moves to the frame, within +-SNAP_S, that best fits both its time
   and its position (event coordinates match the ball to ~1 m once y is flipped), among frames
   where a player of the event's team is next to the ball in the CLEAN tracking. Passes, shots
   and set pieces must also be a kick (ball speed jump); set pieces start from a standstill.
   Set pieces instead snap to the nearest dead -> live switch of the DFL labels (that switch IS
   the restart, by DFL's own definition), within +-6 s. Only the reference is built from labels;
   methods never see them. Events that cannot be snapped keep their clock frame and
   `snapped=False`.

Saved as external/dfl/<match_id>_events.json: a list of dicts with row (snapped bench frame),
clock_row, snapped, kind, team (0 home / 1 away / None), decision_row (set pieces: the referee's
call), and kind-specific fields.
"""

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from . import dfl
from .match import FPS, D, Match

RESTARTS = {"KickOff": "kickoff", "ThrowIn": "throw_in", "CornerKick": "corner",
            "FreeKick": "free_kick", "GoalKick": "goal_kick", "Penalty": "penalty"}
ON_BALL = {"Play", "ShotAtGoal", "OtherBallAction", "BallClaiming", "TacklingGame"}
HALF_START_N = {1: 10000, 2: 100000}  # DFL frame numbering; checked by the anchor residuals
SNAP_S = (2.0, 6.0)  # try +-2 s first; widen to +-6 s if no candidate is within POS_OK_M
POS_SCALE_M = 2.0    # 2 m of position mismatch costs as much as 1 s of time mismatch
POS_OK_M = 3.0       # a snap this close in position is trusted at the first window
POS_MAX_M = 5.0      # never snap to a frame further than this from the logged position
KICKS = {"Play", "ShotAtGoal"}
NEAR_M = 1.5
KICK_MS = 3.0     # ball speed gain over 0.2 s that counts as a kick


def events_path(match_id):
    return os.path.join(dfl.DFL_DIR, "{}_events.json".format(match_id))


def _t(s):
    return datetime.fromisoformat(s).timestamp() if s else None


def list_event_files():
    with urllib.request.urlopen(dfl.ARTICLE_API, timeout=60) as response:
        files = json.load(response)["files"]
    out = {}
    for f in files:
        found = re.search(r"(matchinformation|events_raw).*_(DFL-MAT-\w+)\.xml$", f["name"])
        if found:
            out.setdefault(found.group(2), {})[found.group(1)] = f["download_url"]
    return out


def parse(events_path_xml, home_team):
    """Raw events: list of dicts with t (epoch s), kind, team, and anchors when present."""
    out = []
    for e in ET.parse(events_path_xml).getroot().iter("Event"):
        if not len(e) or e[0].tag == "Delete":
            continue
        child = e[0]
        team_id = (child.get("Team") or child.get("WinnerTeam") or child.get("TeamFouled")
                   or (child[0].get("Team") if len(child) else None))  # KickOff: on its Play
        row = {"t": _t(e.get("EventTime")), "kind": child.tag,
               "team": None if team_id is None else int(team_id != home_team),
               "x": float(e.get("X-Position")) if e.get("X-Position") else None,
               "y": float(e.get("Y-Position")) if e.get("Y-Position") else None}
        if e.get("CalculatedFrame"):
            row["anchor_n"] = int(e.get("CalculatedFrame"))
            row["anchor_t"] = _t(e.get("CalculatedTimestamp"))
        if child.tag in RESTARTS:
            row["restart"] = RESTARTS[child.tag]
            row["decision_t"] = _t(child.get("DecisionTimestamp"))
            row["section"] = child.get("GameSection")
        if child.tag == "TacklingGame":
            row["possession_change"] = child.get("PossessionChange") == "true"
        if child.tag == "Play":
            play = child[0].tag if len(child) else None
            row["play"] = play
            row["successful"] = child.get("Evaluation") == "successfullyCompleted"
        out.append(row)
    out.sort(key=lambda r: r["t"])
    return out


def _halves_of(events):
    """Split time at the second-half kick-off (and drop anything after the last final whistle)."""
    second = [r["t"] for r in events if r.get("section") == "secondHalf"]
    if not second:
        raise ValueError("no second-half kick-off event")
    return second[0] - 60.0


def sync(events, match):
    """Add clock_row, row, snapped (and decision_row) to each event, in place. Returns stats."""
    split = _halves_of(events)
    half_row0 = {int(h): int(first) for h, first, _ in match.halves}
    t_n0, lag = {}, {}
    for half in (1, 2):
        anchors = [r for r in events if "anchor_n" in r and (r["anchor_t"] >= split) == (half == 2)]
        lag[half] = float(np.median([a["t"] - a["anchor_t"] for a in anchors])) if anchors else 0.0
        if anchors:
            t_n0[half] = float(np.median([a["anchor_t"] - (a["anchor_n"] - HALF_START_N[half]) / FPS
                                          for a in anchors]))
        else:  # fall back: kick-off event = first live frame of the half
            first, last = [(f, l) for h, f, l in match.halves if h == half][0]
            live = np.flatnonzero(match.labels[first:last + 1] != D)
            ko = [r["t"] for r in events if r.get("restart") == "kickoff"
                  and (r["t"] >= split) == (half == 2)][0]
            t_n0[half] = ko - (live[0] if len(live) else 0) / FPS

    def to_row(t, half):
        return half_row0[half] + int(round((t - t_n0[half]) * FPS))

    m = np.asarray(match.pitch)
    ball = match.ball[:, :2] * m
    speed = np.nan_to_num(match.ball[:, 2].astype(float))
    pl = match.players[..., :2] * m
    dist = np.nan_to_num(np.sqrt(((pl - ball[:, None, :]) ** 2).sum(-1)), nan=np.inf)
    nearest = dist.argmin(1)
    near_team = np.where(dist.min(1) < NEAR_M, match.slot_team[nearest], -1)
    lead = int(0.2 * FPS)
    gain = np.concatenate([speed[lead:] - speed[:-lead], np.zeros(lead)])
    still = np.convolve(speed < 1.0, np.ones(int(0.5 * FPS)), "full")[:len(speed)] >= int(0.5 * FPS)

    restart_rows = np.flatnonzero((match.labels[:-1] == D) & np.isin(match.labels[1:], (0, 1))) + 1
    restart_rows = np.concatenate([[int(f) for _, f, _ in match.halves
                                    if match.labels[f] in (0, 1)], restart_rows])  # live at start
    residual, snapped_n = [], 0
    for r in events:
        half = 2 if r["t"] >= split else 1
        r["half"] = half
        r["clock_row"] = to_row(r["t"] - lag[half], half)
        if r.get("decision_t"):
            r["decision_row"] = to_row(r["decision_t"] - lag[half], half)
        if "anchor_n" in r:
            truth = half_row0[half] + r["anchor_n"] - HALF_START_N[half]
            r["anchor_row"] = truth
        r["row"], r["snapped"] = r["clock_row"], False
        if "restart" in r:
            near = restart_rows[np.abs(restart_rows - r["clock_row"]) <= int(SNAP_S[1] * FPS)]
            if len(near):
                cost = np.abs(near - r["clock_row"]) / FPS
                if r["x"] is not None:
                    ex, ey = r["x"], match.pitch[1] - r["y"]
                    cost = cost + np.nan_to_num(np.hypot(ball[near, 0] - ex, ball[near, 1] - ey),
                                                nan=20.0) / POS_SCALE_M
                r["row"], r["snapped"] = int(near[np.argmin(cost)]), True
                snapped_n += 1
            if "anchor_row" in r:
                residual.append((r["clock_row"] - r["anchor_row"], r["row"] - r["anchor_row"]))
            continue
        if r["team"] is None or r["kind"] not in ON_BALL:
            continue
        for k, window in enumerate(SNAP_S):
            lo = max(1, r["clock_row"] - int(window * FPS))
            hi = min(match.n_frames - lead - 1, r["clock_row"] + int(window * FPS))
            if hi <= lo:
                break
            win = np.arange(lo, hi)
            ok = near_team[win] == r["team"]
            if r["kind"] in KICKS or "restart" in r:
                ok &= gain[win] > KICK_MS
            if "restart" in r:
                ok &= still[win - 1]
            if not ok.any():
                continue
            cand = win[ok]
            cost = np.abs(cand - r["clock_row"]) / FPS
            pos_err = np.zeros(len(cand))
            if r["x"] is not None:
                ex, ey = r["x"], match.pitch[1] - r["y"]  # event y axis is flipped
                pos_err = np.nan_to_num(np.hypot(ball[cand, 0] - ex, ball[cand, 1] - ey), nan=99.0)
                cost = cost + pos_err / POS_SCALE_M
            best = int(np.argmin(cost))
            if pos_err[best] > POS_MAX_M or (pos_err[best] > POS_OK_M and k + 1 < len(SNAP_S)):
                continue
            r["row"], r["snapped"] = int(cand[best]), True
            snapped_n += 1
            break
        if "anchor_row" in r:
            residual.append((r["clock_row"] - r["anchor_row"], r["row"] - r["anchor_row"]))
    res = np.asarray(residual) if residual else np.zeros((0, 2))
    return {"events": len(events), "snapped": snapped_n, "lag_s": [round(lag[1], 2), round(lag[2], 2)],
            "anchors": len(res),
            "anchor_err_clock_frames": float(np.median(np.abs(res[:, 0]))) if len(res) else None,
            "anchor_err_snapped_frames": float(np.median(np.abs(res[:, 1]))) if len(res) else None}


def fetch(match_ids=None):
    """Download events + match info for the converted DFL matches, sync, save json."""
    files = list_event_files()
    raw = os.path.join(dfl.DFL_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    for match_id in match_ids or sorted(files):
        if not os.path.isfile(dfl.match_path(match_id)):
            print("{}: tracking not converted, run `bench fetch dfl` first".format(match_id))
            continue
        paths = {}
        for kind, key in (("info", "matchinformation"), ("events", "events_raw")):
            paths[kind] = os.path.join(raw, "{}_{}.xml".format(kind, match_id))
            if not os.path.isfile(paths[kind]):
                urllib.request.urlretrieve(files[match_id][key], paths[kind])
        home, _, _ = dfl.read_info(paths["info"])
        events = parse(paths["events"], home)
        stats = sync(events, Match.load(dfl.match_path(match_id)))
        with open(events_path(match_id), "w") as f:
            json.dump(events, f)
        print("{}: {events} events, {snapped} snapped, lag {lag_s} s; anchors {anchors}: median "
              "error {anchor_err_clock_frames} frames on the clock -> {anchor_err_snapped_frames} "
              "after snapping".format(match_id, **stats))


def load(match_name):
    """Synced events for a bench match name ('dfl-DFL-MAT-...'), or None if not fetched."""
    path = events_path(match_name.replace("dfl-", "", 1))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
