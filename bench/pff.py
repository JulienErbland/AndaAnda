"""PFF FC World Cup 2022 (64 matches) -> the common Match format.

Broadcast tracking (Sportlogiq) synchronised with PFF events, 29.97 fps, metres from the pitch
centre (PFF FC Tracking Data Specification v2.2, shipped with the data). Proprietary: the files
stay in external/pff/ (gitignored) and nothing derived from them is committed.

Conversions:
- 29.97 fps -> 25 fps by linear interpolation between neighbouring source frames (only when
  both are present; else missing);
- metres from the centre -> 0-1 from a corner, y flipped so it grows downwards like the
  Metrica / hackathon convention (only consistency within a match matters);
- jersey numbers -> anonymous slots (one per player, as in the Metrica CSV games);
- players: ESTIMATED positions are KEPT (a broadcast camera shows ~7 players; the hackathon's
  tactical camera tracks ~20, so dropping them would be far less realistic); the bench then
  degrades them like any clean match (noise, fragments, slots);
- ball: VISIBLE only, the RAW track (the smoothed ball is placed at the ball carrier from the
  events: it would leak the answer). It is then missing on ~35% of live and ~90% of dead frames,
  already as much as the real tracking, so the "pff" degrader profile adds no ball gaps.

Labels, from the game events (each names the player in control of the ball):
- possession = `home_ball` of the latest game event (the team in control during that
  possession sequence) -- or, with label="player", the team of the latest controlling player;
- dead ball from an OUT event until the next event of another type (the restart);
- graded: each regular half, from its first to its last tracked frame (extra time dropped).

    python -m bench fetch pff        # external/pff/raw/... -> external/pff/<id>.npz
"""

import bz2
import glob
import json
import os

import numpy as np

from . import EXTERNAL_DIR
from .match import FPS, A, D, H, X, Match

RAW = os.path.join(EXTERNAL_DIR, "pff", "raw", "FIFA World Cup 2022")
OUT_DIR = os.path.join(EXTERNAL_DIR, "pff")


def ids():
    return sorted(os.path.basename(p).split(".")[0]
                  for p in glob.glob(os.path.join(RAW, "Tracking Data", "*.jsonl.bz2")))


def read_meta(game_id):
    with open(os.path.join(RAW, "Metadata", "{}.json".format(game_id))) as f:
        m = json.load(f)
    m = m[0] if isinstance(m, list) else m
    pitch = m["stadium"]["pitches"][0]
    return {"pitch": (float(pitch["length"]), float(pitch["width"])), "fps": float(m["fps"]),
            "home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"]}


def _xy(p, pitch, keep_estimated=False):
    if p is None or p.get("x") is None:
        return np.nan, np.nan
    if p.get("visibility") != "VISIBLE" and not keep_estimated:
        return np.nan, np.nan
    return (p["x"] + pitch[0] / 2) / pitch[0], (pitch[1] / 2 - p["y"]) / pitch[1]


def read_tracking(game_id, pitch):
    """Source-rate arrays: frames (N,), period (N,), ball (N, 2), players {(team, jersey): (N, 2)},
    events [(frame, type, home_team, home_ball)] in first-seen order."""
    path = os.path.join(RAW, "Tracking Data", "{}.jsonl.bz2".format(game_id))
    frames, period, ball, events, seen = [], [], [], [], set()
    pl = {}
    with bz2.open(path, "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            fn = r.get("frameNum")
            if fn is None:
                continue
            frames.append(int(fn))
            period.append(int(r.get("period") or 0))
            b = r.get("balls") or []
            ball.append(_xy(b[0] if isinstance(b, list) and b else None, pitch))
            for team, key in ((0, "homePlayers"), (1, "awayPlayers")):
                for p in r.get(key) or []:
                    j = (team, str(p.get("jerseyNum")))
                    pl.setdefault(j, {})[len(frames) - 1] = _xy(p, pitch, keep_estimated=True)
            ge = r.get("game_event")
            gid = r.get("game_event_id")
            if ge and gid is not None and gid not in seen:
                seen.add(gid)
                events.append((int(ge.get("start_frame") or fn), ge.get("game_event_type"),
                               bool(ge.get("home_team")), ge.get("home_ball")))
    n = len(frames)
    players = {}
    for j, d in pl.items():
        a = np.full((n, 2), np.nan)
        idx = np.fromiter(d.keys(), dtype=np.int64)
        a[idx] = np.array(list(d.values()), dtype=np.float64)
        players[j] = a
    return (np.asarray(frames), np.asarray(period), np.asarray(ball, dtype=np.float64),
            players, events)


def _resample(src_t, values, dst_t, max_dt):
    """Linear interpolation of (N, 2) values at dst_t; NaN unless both neighbours exist within
    max_dt of each other."""
    j = np.searchsorted(src_t, dst_t)
    lo, hi = np.clip(j - 1, 0, len(src_t) - 1), np.clip(j, 0, len(src_t) - 1)
    t0, t1 = src_t[lo], src_t[hi]
    w = np.where(t1 > t0, (dst_t - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)[:, None]
    out = values[lo] * (1 - w) + values[hi] * w
    bad = (t1 - t0 > max_dt) | (dst_t < src_t[0]) | (dst_t > src_t[-1])
    out[bad] = np.nan
    return out


def labels_from_events(events, src_frame_of, n_src, label="sequence"):
    """(n_src,) labels at source rate from the ordered game events."""
    lab = np.full(n_src, X, dtype=np.int8)
    ev = sorted((src_frame_of(f), t, ht, hb) for f, t, ht, hb in events if f is not None)
    state = X
    pts = []
    for k, t, home_team, home_ball in ev:
        if t in ("OUT",):
            state = D
        elif t in ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "G"):
            if label == "sequence" and home_ball is not None:
                state = H if home_ball else A
            else:
                state = H if home_team else A
        elif t in ("END", "VID"):
            state = X
        else:                      # ON / OFF / SUB: substitutions, no ball information
            continue
        pts.append((k, state))
    for (k, s), nxt in zip(pts, pts[1:] + [(n_src, None)]):
        lab[max(0, k):max(0, nxt[0])] = s
    return lab


def convert(game_id, label="sequence"):
    meta = read_meta(game_id)
    pitch, fps = meta["pitch"], meta["fps"]
    frames, period, ball, players, events = read_tracking(game_id, pitch)
    order = np.argsort(frames, kind="stable")
    frames, period, ball = frames[order], period[order], ball[order]
    players = {j: a[order] for j, a in players.items()}
    src_t = frames / fps
    # an event's start frame may have no tracking (camera cut): map it to the next tracked frame
    lab_src = labels_from_events(events, lambda f: int(np.searchsorted(frames, f)), len(frames), label)
    regular = (period == 1) | (period == 2)
    t0, t1 = src_t[regular].min(), src_t[regular].max()
    dst_t = np.arange(t0, t1, 1.0 / FPS)
    # each half spans only its own tracked time (half-time is not graded)
    span = {h: (src_t[period == h].min(), src_t[period == h].max()) for h in (1, 2) if (period == h).any()}
    max_dt = 1.5 / fps
    b = _resample(src_t, ball, dst_t, max_dt)
    keys = sorted(players)
    P = np.stack([_resample(src_t, players[k], dst_t, max_dt) for k in keys], axis=1)
    near = np.clip(np.searchsorted(src_t, dst_t), 0, len(src_t) - 1)
    labels = lab_src[near]
    per = np.zeros(len(dst_t), dtype=np.int64)
    for h, (a, b_) in span.items():
        per[(dst_t >= a) & (dst_t <= b_)] = h
    halves = []
    for h in (1, 2):
        idx = np.flatnonzero(per == h)
        if len(idx):
            halves.append((h, int(idx[0]), int(idx[-1])))
    labels[~np.isin(per, (1, 2))] = X
    m = np.asarray(pitch, dtype=np.float32)

    def speed(xy):
        s = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
        s[1:-1] = np.linalg.norm((xy[2:] - xy[:-2]) * m, axis=-1) * FPS / 2
        return s

    ball3 = np.concatenate([b, speed(b)[:, None]], axis=1).astype(np.float32)
    P3 = np.concatenate([P, speed(P)[..., None]], axis=-1).astype(np.float32)
    slot_team = np.array([k[0] for k in keys], dtype=np.int8)
    return Match(name="pff-{}".format(game_id), pitch=pitch, ball=ball3, players=P3,
                 slot_team=slot_team, halves=np.asarray(halves, dtype=np.int64),
                 labels=labels.astype(np.int8))


def _one(game_id):
    out = os.path.join(OUT_DIR, "{}.npz".format(game_id))
    if os.path.isfile(out):
        return game_id, "exists"
    try:
        convert(game_id).save(out)
        return game_id, "ok"
    except Exception as e:        # one bad file must not stop the other 63
        return game_id, "error: {}".format(e)


def fetch(workers=6):
    """Convert every PFF match (already downloaded to external/pff/raw/) in parallel."""
    from multiprocessing import Pool
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = ids()
    if not todo:
        raise SystemExit("no PFF tracking in {}; see README for the download".format(RAW))
    with Pool(workers) as pool:
        for gid, status in pool.imap_unordered(_one, todo):
            print("  pff {} {}".format(gid, status), flush=True)
