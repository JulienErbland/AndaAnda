"""Turn clean tracking into hackathon-like automatic tracking. Labels are never touched.

What the hackathon ATD is documented to look like (DATA.md / CHALLENGE.md), and how each part is
simulated:

- ball missing on 41% of graded frames, in few LONG gaps (measured: median 10.7 s, p99 105 s,
  11% of gaps under 1 s)
  -> gap process: per-frame start probability (scaled by crowding and ball speed), gap length a
     mixture of short flickers and a lognormal, start rates calibrated by bisection to target
     missing rates on alive and on dead frames separately (that split decides how much "ball
     missing" leaks "dead");
- tracker blackouts: the ball and many players vanish together (measured: with <=18 players
  tracked, the ball is missing 85% of the time)
  -> a share of ball gaps also hides each player with a probability drawn per gap;
- no player identities, 46 slots that fragment (measured: 11.5 tracklets/min, median 35 s)
  -> each player's visible time is cut into tracklets, which are packed into anonymous slots;
- ~20.5 tracked objects for 22 players, occasional duplicates
  -> blackouts, light per-player hiding gaps, plus short ghost copies of a player;
- identity switches that can cross teams
  -> a fraction of tracklets land in a slot of the wrong team (team-level features are all a
     method can use, so a same-team switch is invisible and not simulated);
- position noise.

Calibrated on 2026-10-20 against `python -m bench stats hackathon` (aggregates only). The
alive/dead split is an estimate: the public window covers 5 minutes only, so the full-match split
was inferred from ball-missing rates per mean-player-speed bin, weighted by DFL's dead share per
bin (the two matches have near-identical speed distributions).

ponytail: one gap-length distribution for alive and dead starts; split it only if a measurement
says they differ.
"""

from dataclasses import dataclass, replace

import numpy as np

from .match import A, D, H, Match, graded_mask


@dataclass(frozen=True)
class Profile:
    # Ball dropout. Targets are fractions of graded frames of that kind with no ball.
    ball_missing_alive: float = 0.19
    ball_missing_dead: float = 0.61
    ball_gap_median_s: float = 9.5     # lognormal part of the gap length
    ball_gap_sigma: float = 0.75
    ball_flicker_share: float = 0.11   # share of gaps that are short flickers
    ball_flicker_s: float = 0.4
    crowd_gain: float = 0.5            # start-rate multiplier per player within crowd_radius_m
    crowd_radius_m: float = 3.0
    fast_ball_gain: float = 1.0        # extra multiplier when ball speed > fast_ball_ms
    fast_ball_ms: float = 12.0
    ball_noise_m: float = 0.3
    # Tracker blackouts: share of ball gaps that also hide players, each with prob ~U(lo, hi).
    blackout_share: float = 0.3
    blackout_hide: tuple = (0.2, 0.6)
    # Player-only blackouts (ball still tracked), measured: 15% of <=18-player frames have a ball.
    player_blackouts_per_min: float = 0.2
    player_blackout_s: float = 5.0
    # Players.
    player_missing: float = 0.005      # independent hiding, on top of blackouts (rare: measured)
    player_gap_s: float = 3.0
    tracklet_s: float = 400.0          # mean time between identity breaks
    wrong_team: float = 0.02           # fraction of tracklets assigned to a wrong-team slot
    duplicates_per_min: float = 0.5
    duplicate_s: float = 2.0
    player_noise_m: float = 0.2


PROFILES = {
    "hackathon": Profile(),
    # Same ball as the hackathon, perfect players: isolates the cost of losing the ball.
    "ball-only": Profile(blackout_share=0, player_blackouts_per_min=0, player_missing=0,
                         tracklet_s=1e9, wrong_team=0, duplicates_per_min=0, player_noise_m=0),
    "clean": None,
}

FPS = 25


def degrade(match, profile="hackathon", seed=0):
    if isinstance(profile, str):
        profile = PROFILES[profile]
    if profile is None:
        return match
    rng = np.random.default_rng(seed)
    metres = np.asarray(match.pitch, dtype=np.float32)

    ball = match.ball.copy()
    missing = _ball_gaps(match, profile, rng)
    ball[missing] = np.nan
    ball[:, :2] += rng.normal(0, profile.ball_noise_m, (len(ball), 2)) / metres

    players, slot_team = _fragment_players(match, profile, rng,
                                           _blackouts(missing, match.players.shape[1], profile, rng))
    players[..., :2] += rng.normal(0, profile.player_noise_m, players[..., :2].shape) / metres
    return replace(match, ball=ball, players=players, slot_team=slot_team)


def _gap_cover(start_prob, lengths, u):
    """Frames covered by gaps: a gap of lengths[t] starts at t where u[t] < start_prob[t]."""
    t = np.arange(len(u))
    ends = np.where(u < start_prob, t + lengths, 0)
    return np.maximum.accumulate(ends) > t


def _ball_gaps(match, p, rng):
    n = match.n_frames
    graded = graded_mask(match)
    labels = match.labels if match.labels is not None else np.full(n, H)
    dead = labels == D
    alive = (labels == H) | (labels == A)

    xy = match.ball[:, :2] * np.asarray(match.pitch, dtype=np.float32)
    pl = match.players[..., :2] * np.asarray(match.pitch, dtype=np.float32)
    dist = np.sqrt(((pl - xy[:, None, :]) ** 2).sum(-1))
    crowd = np.nansum(dist < p.crowd_radius_m, axis=1)
    fast = np.nan_to_num(match.ball[:, 2]) > p.fast_ball_ms
    weight = (1 + p.crowd_gain * crowd) * (1 + p.fast_ball_gain * fast)

    flicker = rng.random(n) < p.ball_flicker_share
    lengths = np.where(flicker, rng.geometric(1 / (p.ball_flicker_s * FPS), n),
                       rng.lognormal(np.log(p.ball_gap_median_s * FPS), p.ball_gap_sigma, n))
    lengths = np.maximum(1, lengths).astype(np.int64)
    u = rng.random(n)
    already = np.isnan(match.ball[:, 0])

    scale = {"alive": 1e-3, "dead": 1e-3}
    targets = {"alive": (alive & graded, p.ball_missing_alive),
               "dead": (dead & graded, p.ball_missing_dead)}

    def cover():
        prob = weight * np.where(dead, scale["dead"], scale["alive"])
        return _gap_cover(prob, lengths, u) | already

    # Coordinate bisection on log-scale: the two rates interact only through gaps that run
    # across an alive/dead boundary, so two rounds are enough.
    for _ in range(2):
        for key, (mask, target) in targets.items():
            if not mask.any():
                continue
            lo, hi = -12.0, 0.0
            for _ in range(30):
                scale[key] = 10 ** ((lo + hi) / 2)
                if cover()[mask].mean() < target:
                    lo = (lo + hi) / 2
                else:
                    hi = (lo + hi) / 2
    return cover()


def _blackouts(ball_missing, n_people, p, rng):
    """(T, people) hide mask: a share of ball gaps also loses each player with a per-gap prob."""
    hide = np.zeros((len(ball_missing), n_people), dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ball_missing.astype(np.int8), [0]))))
    for t0, t1 in zip(edges[::2], edges[1::2]):
        if rng.random() < p.blackout_share:
            hide[t0:t1] = rng.random(n_people) < rng.uniform(*p.blackout_hide)
    n = len(ball_missing)
    for _ in range(rng.poisson(p.player_blackouts_per_min * n / FPS / 60)):
        t0 = rng.integers(n)
        t1 = min(n, t0 + max(1, int(rng.exponential(p.player_blackout_s * FPS))))
        hide[t0:t1] |= rng.random(n_people) < rng.uniform(*p.blackout_hide)
    return hide


def _fragment_players(match, p, rng, blackout):
    """(players, slot_team) after hiding, fragmenting, duplicating and repacking into slots."""
    n, s0, _ = match.players.shape
    tracklets = []  # (team, person, t0, t1, offset_xy or None)
    gap_len = p.player_gap_s * FPS
    start = p.player_missing / (gap_len * (1 - p.player_missing)) if p.player_missing else 0
    for person in range(s0):
        present = ~np.isnan(match.players[:, person, 0])
        hidden = _gap_cover(np.full(n, start), rng.geometric(1 / gap_len, n), rng.random(n)) \
            if start else np.zeros(n, dtype=bool)
        visible = present & ~hidden & ~blackout[:, person]
        breaks = rng.random(n) < 1 / (p.tracklet_s * FPS)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], visible.astype(np.int8), [0]))))
        for t0, t1 in zip(edges[::2], edges[1::2]):
            cuts = [t0] + [t for t in np.flatnonzero(breaks[t0:t1]) + t0 if t > t0] + [t1]
            for a, b in zip(cuts, cuts[1:]):
                tracklets.append((int(match.slot_team[person]), person, a, b, None))

    n_dup = rng.poisson(p.duplicates_per_min * n / FPS / 60)
    for _ in range(n_dup):
        person, t0 = rng.integers(s0), rng.integers(n)
        t1 = min(n, t0 + max(1, int(rng.exponential(p.duplicate_s * FPS))))
        if not np.isnan(match.players[t0, person, 0]):
            offset = rng.normal(0, 1.0, 2) / np.asarray(match.pitch)
            tracklets.append((int(match.slot_team[person]), person, t0, t1, offset))

    # Pack tracklets into anonymous slots, first free slot of the (possibly wrong) team.
    tracklets.sort(key=lambda tr: tr[2])
    slot_team, busy_until, assignment = [], [], []
    for team, person, t0, t1, offset in tracklets:
        if rng.random() < p.wrong_team:
            team = 1 - team
        # Strictly before t0: a slot freed on this very frame is the same track continuing, and
        # an identity break must show up as a new slot, not a seamless continuation.
        slot = next((i for i, (st, bu) in enumerate(zip(slot_team, busy_until))
                     if st == team and bu < t0), None)
        if slot is None:
            slot = len(slot_team)
            slot_team.append(team)
            busy_until.append(0)
        busy_until[slot] = t1
        assignment.append((slot, person, t0, t1, offset))

    players = np.full((n, len(slot_team), 3), np.nan, dtype=np.float32)
    for slot, person, t0, t1, offset in assignment:
        players[t0:t1, slot] = match.players[t0:t1, person]
        if offset is not None:
            players[t0:t1, slot, :2] += offset.astype(np.float32)
    return players, np.asarray(slot_team, dtype=np.int8)
