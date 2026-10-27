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
- the ball's path is smoothed (`ball_smooth_s`), as a filtering tracker does: kicks and
  deflections come out rounded (measured Oct 22: real deflections 2.1/min, unsmoothed bench 6.1);
- position error: SMOOTH, not jitter. The real ATD's frame-to-frame jitter is tiny (median
  second difference 0.1 cm for players, 0.9 cm for the ball; measured Oct 20), so its errors are
  slow offsets. Noise is white noise low-passed over `noise_smooth_s` and rescaled to the given
  standard deviation, which keeps velocities realistic.

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
    ball_noise_m: float = 0.3         # std of the smooth position error
    # Tracker blackouts: share of ball gaps that also hide players, each with prob ~U(lo, hi).
    blackout_share: float = 0.3
    blackout_hide: tuple = (0.2, 0.6)
    # Player-only blackouts (ball still tracked), measured: 15% of <=18-player frames have a ball.
    player_blackouts_per_min: float = 0.2
    player_blackout_s: float = 5.0
    # Players.
    player_missing: float = 0.005      # independent hiding, on top of blackouts (rare: measured)
    player_gap_s: float = 3.0
    tracklet_s: float = 300.0          # mean time between identity breaks (re-fitted Oct 22
                                       # with the duel weighting: 11.3 track ends/min, real 11.3)
    wrong_team: float = 0.02           # fraction of tracklets assigned to a wrong-team slot
    duplicates_per_min: float = 0.5
    duplicate_s: float = 2.0
    player_noise_m: float = 0.2
    noise_smooth_s: float = 1.0       # correlation time of the position error
    # The real tracker smooths the ball's path (kicks and deflections come out rounded):
    # Gaussian filter of this std. Calibrated Oct 22 by scripts/calibrate_ball_smoothing.py on the
    # whole real match (|acceleration| p90/p99 over 1-4 frames: real 18/37 m/s^2, smoothed DFL
    # 17/42, unsmoothed 21-37/103-356); checked on touch rates (deflections/min real 2.1, bench
    # 6.1 -> 2.3). ponytail: one knob; the real ball also has a slow wobble (median |a| 5.7 vs
    # 3.1 m/s^2) that a second noise term would add if it ever matters.
    ball_smooth_s: float = 0.20
    # Calibrated Oct 22 on the real tracking (scripts/degrader_realism.py measures them,
    # scripts/calibrate_realism.py fits them; aggregates only, no labels):
    # false ball detections (real: 1.1 one-frame jumps > 2 m per minute, bench 0.13; after gaps
    # under 2 s the real ball reappears ~66 m/s away), placed at a random player's position for
    # ~ball_false_s, anywhere in the graded frames (inside a gap they split it);
    ball_false_per_min: float = 0.5    # jumps/min: real 1.12, bench 0.13 -> 0.98
    ball_false_s: float = 0.3
    ball_false_in_gaps: float = 0.0     # share of false blips placed inside ball gaps (fit: 0)
    # Gaps covering a stoppage start when the ball LEAVES play (real gap starts: ball speed p50
    # 1.7 m/s; bench 0.5): a gap starting inside a dead stretch is moved back to just before it
    # (lead U(0, gap_ballout_lead_s)), keeping its end.
    # gap-start ball speed p50: real 1.71, bench 0.49 -> 1.82; crowd 1.04: 0.62 -> 0.93.
    # ponytail: the p90 overshoots (12 vs 6.4 m/s); a speed-dependent lead would fix the tail.
    gap_from_ballout: bool = True
    gap_ballout_lead_s: float = 0.5
    gap_ballout_share: float = 0.5     # share of such gaps that move (the rest start in the stoppage)
    # identity breaks in duels (real: 33% of track ends have an opponent within 2 m, bench 12%):
    # break rate x (1 + gain) near an opponent, rescaled to keep the mean rate (tracklet_s).
    break_duel_gain: float = 80.0     # track ends near an opponent: real 33%, bench 12% -> 33%
    duel_m: float = 2.0
    # Dead-ball motion (Oct 27). DFL parks a dead ball where it left the pitch; the real one is
    # thrown back, carried to the spot and placed. Public window, dead frames with a ball: speed
    # p10/50/90 1.0/2.7/9.3 m/s, still 1%, outside the lines 11% (DFL: 0.2/0.5/0.9, 22%, 52-73%).
    # On: the ball hops between nearby players, then the restart taker carries it to the spot.
    # Fitted Oct 27 (scripts/dead_ball_realism.py): bench now 0.7/2.9/8.4 m/s, still 2%, nearest
    # player 2.3 m, outside 7% (the real target is 4 stoppages: not fitted any closer).
    dead_ball_moves: bool = True
    dead_hop_ms: float = 7.0           # median speed of a throw / kick back between players
    dead_hold_s: tuple = (0.5, 2.0)    # time near a player between hops
    dead_offset_m: tuple = (1.0, 4.5)  # ball's distance to the player holding it
    dead_carry_s: tuple = (1.0, 4.0)   # carried by the taker to the restart spot
    dead_place_s: float = 0.4


PROFILES = {
    "hackathon": Profile(),
    # Same ball as the hackathon, perfect players: isolates the cost of losing the ball.
    "ball-only": Profile(blackout_share=0, player_blackouts_per_min=0, player_missing=0,
                         tracklet_s=1e9, wrong_team=0, duplicates_per_min=0, player_noise_m=0),
    "clean": None,
    # PFF broadcast tracking: its raw ball is already missing as often as the real tracking's
    # (live ~35%, dead ~90%), so no extra ball gaps; everything else as the hackathon profile.
    # Measured (degraded PFF vs real): gap-start ball speed p50/p90 2.26/6.1 vs 1.71/6.45, players
    # within 3 m 0.93 vs 1.04, 20.0 players tracked vs 20.5. Track ends/min 27 vs 11 (camera cuts).
    "pff": Profile(ball_missing_alive=0.0, ball_missing_dead=0.0, gap_from_ballout=False,
                   dead_ball_moves=False),
}

FPS = 25


# Parameters `vary` perturbs: what is calibrated but uncertain (a fit to one real match).
VARIED = ("ball_missing_alive", "ball_missing_dead", "ball_gap_median_s", "blackout_share",
          "tracklet_s", "player_noise_m", "ball_noise_m", "ball_smooth_s", "ball_false_per_min",
          "break_duel_gain", "crowd_gain")


def vary(profile, rng, spread=0.25):
    """A random neighbour of `profile`: each VARIED parameter times exp(N(0, spread)), missing
    rates kept below 0.9. Training on many neighbours puts the real tracker inside what the model
    has seen even where the calibration is off (domain randomisation)."""
    if isinstance(profile, str):
        profile = PROFILES[profile]
    out = {}
    for k in VARIED:
        v = getattr(profile, k) * float(np.exp(rng.normal(0, spread)))
        out[k] = min(v, 0.9) if k.startswith("ball_missing") else v
    return replace(profile, **out)


def degrade(match, profile="hackathon", seed=0):
    if isinstance(profile, str):
        profile = PROFILES[profile]
    if profile is None:
        return match
    rng = np.random.default_rng(seed)
    metres = np.asarray(match.pitch, dtype=np.float32)

    ball = match.ball.copy()
    if profile.dead_ball_moves:
        _dead_ball_motion(ball, match, profile, rng)
    if profile.ball_smooth_s > 0:
        from .epts import derive_speed
        ball[:, :2] = smooth_track(ball[:, :2], profile.ball_smooth_s)
        ball[:, 2] = derive_speed(ball[:, :2], match.pitch)
    missing = _ball_gaps(match, profile, rng)
    ball[missing] = np.nan
    ball[:, :2] += _smooth_noise(rng, (len(ball), 2), profile.ball_noise_m,
                                 profile.noise_smooth_s) / metres

    players, slot_team = _fragment_players(match, profile, rng,
                                           _blackouts(missing, match.players.shape[1], profile, rng))
    players[..., :2] += _smooth_noise(rng, players[..., :2].shape, profile.player_noise_m,
                                      profile.noise_smooth_s) / metres
    if profile.ball_false_per_min > 0:      # drawn last: off leaves every other draw unchanged
        _false_ball(ball, match, profile, rng)
    return replace(match, ball=ball, players=players, slot_team=slot_team)


def _dead_ball_motion(ball, match, p, rng):
    """In place, clean tracking: during each stoppage the ball hops between nearby players
    (thrown / kicked back at dead_hop_ms, then held near the player at an offset for dead_hold_s),
    the last hop reaching the player who restarts, who carries it to the restart spot, where it
    rests for up to dead_place_s. Needs labels (dead runs) and the ball at the restart frame."""
    if match.labels is None:
        return
    m = np.asarray(match.pitch, dtype=np.float64)
    pl_all = match.players[..., :2].astype(np.float64) * m
    dead = np.r_[False, match.labels == D, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(dead))
    for a, b in zip(edges[::2], edges[1::2]):
        if b >= match.n_frames or b - a < FPS or np.isnan(ball[b, 0]):
            continue
        spot = ball[b, :2].astype(np.float64) * m
        d = np.linalg.norm(pl_all[b] - spot, axis=1)
        if np.all(np.isnan(d)):
            continue
        taker = int(np.nanargmin(d))
        carry = int(round(rng.uniform(*p.dead_carry_s) * FPS))
        place = int(round(rng.uniform(0, p.dead_place_s) * FPS))
        end_hops = max(a, b - place - carry)
        pos = ball[a, :2].astype(np.float64) * m if not np.isnan(ball[a, 0]) else spot
        path = np.full((b - a, 2), np.nan)
        t = a
        while t < end_hops:
            last = t + 1 >= end_hops - FPS            # no room for another hop: go to the taker
            if last:
                who = taker
            else:
                dd = np.linalg.norm(pl_all[t] - pos, axis=1)
                near = np.argsort(np.where(np.isnan(dd), np.inf, dd))[:5]
                near = near[np.isfinite(dd[near])]
                if not len(near):
                    break
                who = int(rng.choice(near))
            ang = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(*p.dead_offset_m) * np.array([np.cos(ang), np.sin(ang)])
            speed = rng.lognormal(np.log(p.dead_hop_ms), 0.6)
            target_t = pl_all[min(t, b - 1), who]
            if np.isnan(target_t[0]):
                target_t = spot
            dur = int(np.clip(np.linalg.norm(target_t + off - pos) / speed * FPS, 3, max(3, end_hops - t)))
            t1 = min(end_hops, t + dur)
            f = np.linspace(0, 1, t1 - t, endpoint=False)[:, None]
            arrive = pl_all[t1 - 1, who] + off if not np.isnan(pl_all[t1 - 1, who, 0]) else target_t
            path[t - a:t1 - a] = pos * (1 - f) + arrive * f
            pos, t = arrive, t1
            if last or t >= end_hops:
                break
            hold = min(end_hops, t + int(round(rng.uniform(*p.dead_hold_s) * FPS)))
            seg = pl_all[t:hold, who] + off
            seg = np.where(np.isnan(seg), pos, seg)
            path[t - a:hold - a] = seg
            if hold > t:
                pos = seg[-1]
            t = hold
        # carried by the taker to the spot, then placed
        c0 = max(t, a)
        who_tr = pl_all[c0:b - place, taker]
        if len(who_tr):
            start_off = pos - (who_tr[0] if not np.isnan(who_tr[0, 0]) else pos)
            end_off = spot - (who_tr[-1] if not np.isnan(who_tr[-1, 0]) else spot)
            g_ = np.linspace(0, 1, len(who_tr))[:, None]
            seg = who_tr + start_off * (1 - g_) + end_off * g_
            path[c0 - a:b - place - a] = np.where(np.isnan(seg), spot, seg)
        path[b - place - a:] = spot
        path = np.where(np.isnan(path), spot, path)
        ball[a:b, :2] = (path / m).astype(ball.dtype)


def _false_ball(ball, match, p, rng):
    """In place: short spans where the 'ball' is another object near a random player."""
    g = np.flatnonzero(graded_mask(match))
    if not len(g):
        return
    n_false = rng.poisson(p.ball_false_per_min * len(g) / FPS / 60)
    gap = np.flatnonzero(np.isnan(ball[g, 0]))
    in_gap = rng.random(n_false) < p.ball_false_in_gaps
    starts = np.where(in_gap & (len(gap) > 0), g[gap[rng.integers(0, max(len(gap), 1), n_false)]]
                      if len(gap) else 0, rng.choice(g, size=n_false))
    for t0 in starts:
        t1 = min(match.n_frames, t0 + max(1, int(rng.exponential(p.ball_false_s * FPS))))
        who = np.flatnonzero(~np.isnan(match.players[t0, :, 0]))
        if not len(who):
            continue
        s = rng.choice(who)
        ball[t0:t1, :2] = match.players[t0:t1, s, :2] + rng.normal(0, 0.5, 2) / np.asarray(
            match.pitch, dtype=np.float32)
        ball[t0:t1, 2] = np.nan_to_num(match.players[t0:t1, s, 2])


def smooth_track(xy, sigma_s):
    """Gaussian smoothing along time, NaN-aware: never blends across missing frames' values,
    and missing frames stay missing."""
    sig = sigma_s * FPS
    r = int(np.ceil(3 * sig))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    ok = ~np.isnan(xy[:, 0])
    w = np.convolve(ok.astype(np.float64), k, "same")
    out = np.full(xy.shape, np.nan, dtype=xy.dtype)
    for c in range(xy.shape[1]):
        v = np.convolve(np.where(ok, xy[:, c], 0.0).astype(np.float64), k, "same")
        out[ok, c] = (v[ok] / w[ok]).astype(xy.dtype)
    return out


def _smooth_noise(rng, shape, std, smooth_s):
    """Gaussian noise low-passed along time (axis 0), rescaled to `std`.

    Three passes of a moving average approximate a Gaussian filter: one pass leaves steps at the
    window edges, which show up as ~8 cm of frame-to-frame jitter; three bring it under 1 cm.
    """
    if std <= 0:
        return np.zeros(shape, dtype=np.float32)
    w = max(1, int(round(smooth_s * FPS)))
    x = rng.normal(0, 1, (shape[0] + 3 * (w - 1),) + tuple(shape[1:]))
    for _ in range(3):
        c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)], axis=0)
        x = (c[w:] - c[:-w]) / w
    return (x * std / x.std()).astype(np.float32)


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

    if p.gap_from_ballout:
        lead = (rng.random(n) * p.gap_ballout_lead_s * FPS).astype(np.int64)
        moves = rng.random(n) < p.gap_ballout_share
        dead_start = np.flatnonzero(dead & ~np.concatenate(([False], dead[:-1])))
        run_start = np.zeros(n, dtype=np.int64)
        if len(dead_start):
            idx = np.searchsorted(dead_start, np.arange(n), side="right") - 1
            run_start = np.where(idx >= 0, dead_start[np.maximum(idx, 0)], 0)

    def cover():
        prob = weight * np.where(dead, scale["dead"], scale["alive"])
        c = _gap_cover(prob, lengths, u)
        if p.gap_from_ballout:
            starts = np.flatnonzero(c & ~np.concatenate(([False], c[:-1])))
            for s0 in starts[dead[starts] & moves[starts]]:
                c[max(0, run_start[s0] - lead[s0]):s0] = True   # begin as the ball leaves play
        return c | already

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


def _near_opponent(match, person, radius_m):
    """(T,) True where an opponent of `person` stands within radius_m (clean tracking)."""
    m = np.asarray(match.pitch, dtype=np.float32)
    opp = match.slot_team != match.slot_team[person]
    d = np.linalg.norm((match.players[:, opp, :2] - match.players[:, person:person + 1, :2]) * m,
                       axis=-1)
    return (np.nan_to_num(d, nan=np.inf) < radius_m).any(axis=1)


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
        rate = np.full(n, 1 / (p.tracklet_s * FPS))
        if p.break_duel_gain > 0:
            duel = _near_opponent(match, person, p.duel_m)
            w = 1 + p.break_duel_gain * duel
            rate = rate * w / max(w[present].mean() if present.any() else 1.0, 1e-9)
        breaks = rng.random(n) < rate
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
