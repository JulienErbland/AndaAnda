"""Edge-case scenarios: situations that break possession methods, mined from labelled matches.

Each scenario finds clips automatically (from the GT labels, the clean tracking for geometry and the
degraded tracking for what a method actually sees) and states one pass/fail rule for what a good
prediction does there. `run` reports the pass rate per scenario and method -- a targeted view the
global spellF1 hides (a method can gain overall while getting worse at restarts).

Only use on labelled external matches (DFL): clips are frame-level, so never on hackathon data.
"""

from dataclasses import dataclass, field

import numpy as np

from .evaluate import pm
from .match import FPS, A, D, H

S = FPS  # one second, in frames


@dataclass
class Clip:
    match: str
    anchor: int                 # the key frame of the situation
    start: int                  # preview window
    stop: int
    info: dict = field(default_factory=dict)


@dataclass
class Context:
    name: str
    clean: object               # Match, true tracking (geometry the finders may use)
    seen: object                # Match, degraded tracking (what the method sees)
    gt: np.ndarray
    segs: list                  # [(start, stop, code)] GT segments, contiguous within a half

    def contiguous(self, i):
        return i + 1 < len(self.segs) and self.segs[i][1] == self.segs[i + 1][0]


def _alive(k):
    return k in (H, A)


def _transition_near(pred, frame, a, b, tol):
    lo, hi = max(1, frame - tol), min(len(pred), frame + tol + 1)
    return bool(np.any((pred[lo - 1:hi - 1] == a) & (pred[lo:hi] == b)))


def _share(pred, lo, hi, code):
    return float(np.mean(pred[lo:hi] == code)) if hi > lo else 0.0


def _clip(ctx, anchor, before=4, after=4, **info):
    return Clip(ctx.name, int(anchor), max(0, int(anchor - before * S)),
                min(len(ctx.gt), int(anchor + after * S)), info)


def _nearest(match, team_of_ball_side=None):
    """(nearest-home distance, nearest-away distance) to the ball, metres, NaN-safe."""
    m = np.asarray(match.pitch)
    d = np.sqrt((((match.players[..., :2] - match.ball[:, None, :2]) * m) ** 2).sum(-1))
    d = np.nan_to_num(d, nan=np.inf)
    return (d[:, match.slot_team == 0].min(1), d[:, match.slot_team == 1].min(1))


def _runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2], e[1::2]))


# -- finders + checks ------------------------------------------------------------------------

def find_pass_past_opponents(ctx):
    """Fast ball inside a long spell, with an opponent nearer to it than any team-mate."""
    dh, da = _nearest(ctx.clean)
    speed = np.nan_to_num(ctx.clean.ball[:, 2])
    out = []
    for a, b, k in ctx.segs:
        if not _alive(k) or b - a < 8 * S:
            continue
        opp, mate = (da, dh) if k == H else (dh, da)
        hit = np.flatnonzero((speed[a + 2 * S:b - 2 * S] > 8) & (opp[a + 2 * S:b - 2 * S] < 2.5)
                             & (opp[a + 2 * S:b - 2 * S] < mate[a + 2 * S:b - 2 * S]))
        if len(hit):
            out.append(_clip(ctx, a + 2 * S + hit[0], team=int(k)))
    return out


def check_hold_team(clip, gt, pred):
    return _share(pred, clip.anchor - S, clip.anchor + S, clip.info["team"]) >= 0.9


def find_interception(ctx):
    out = []
    for i, (a, b, k) in enumerate(ctx.segs):
        if ctx.contiguous(i):
            a2, b2, k2 = ctx.segs[i + 1]
            if _alive(k) and _alive(k2) and b - a >= 3 * S and b2 - a2 >= 3 * S:
                out.append(_clip(ctx, b, frm=int(k), to=int(k2)))
    return out


def check_transition_half_second(clip, gt, pred):
    return _transition_near(pred, clip.anchor, clip.info["frm"], clip.info["to"], S // 2)


def find_short_spell(ctx):
    out = []
    for i in range(1, len(ctx.segs) - 1):
        (pa, pb, pk), (a, b, k), (na, nb, nk) = ctx.segs[i - 1:i + 2]
        if (ctx.contiguous(i - 1) and ctx.contiguous(i) and _alive(k) and b - a < 1.5 * S
                and pk == nk == 1 - k and pb - pa >= 3 * S and nb - na >= 3 * S):
            out.append(_clip(ctx, a, team=int(k), stop_spell=int(b)))
    return out


def check_short_spell(clip, gt, pred):
    a, b, k = clip.anchor, clip.info["stop_spell"], clip.info["team"]
    best = 0.0
    for ra, rb in _runs(pred == k):
        if ra < b and a < rb:
            best = max(best, (min(b, rb) - max(a, ra)) / (max(b, rb) - min(a, ra)))
    return best >= 0.3


def _find_restart(ctx, same):
    out = []
    for i in range(1, len(ctx.segs) - 1):
        (pa, pb, pk), (a, b, k), (na, nb, nk) = ctx.segs[i - 1:i + 2]
        if (ctx.contiguous(i - 1) and ctx.contiguous(i) and k == D and _alive(pk) and _alive(nk)
                and 3 * S <= b - a < 30 * S and (pk == nk) == same):
            out.append(Clip(ctx.name, int(a), max(0, a - 3 * S), min(len(ctx.gt), b + 3 * S),
                            {"restart": int(b), "team": int(nk)}))
    return out


def find_restart_same(ctx):
    return _find_restart(ctx, True)


def find_restart_other(ctx):
    return _find_restart(ctx, False)


def check_restart(clip, gt, pred):
    stop, restart, team = clip.anchor, clip.info["restart"], clip.info["team"]
    stopped = np.any(pred[stop - S:stop + S] == D)
    resumed = _share(pred, restart + S, restart + 3 * S, team) >= 0.8
    return bool(stopped and resumed and _share(pred, stop + S, restart - S, D) >= 0.8)


def find_long_stoppage(ctx):
    return [Clip(ctx.name, int(a), max(0, a - 3 * S), min(len(ctx.gt), a + 20 * S),
                 {"stop": int(b)}) for a, b, k in ctx.segs if k == D and b - a >= 30 * S]


def check_long_stoppage(clip, gt, pred):
    return _share(pred, clip.anchor + 5 * S, clip.info["stop"] - 5 * S, D) >= 0.9


def find_ball_gap_hold(ctx):
    missing = np.isnan(ctx.seen.ball[:, 0])
    out = []
    for a, b, k in ctx.segs:
        if _alive(k):
            for ga, gb in _runs(missing[a + S:b - S]):
                if gb - ga >= 10 * S:
                    out.append(Clip(ctx.name, int(a + S + ga), max(0, a + S + ga - 3 * S),
                                    min(len(ctx.gt), a + S + gb + 3 * S),
                                    {"team": int(k), "gap_stop": int(a + S + gb)}))
    return out


def check_ball_gap_hold(clip, gt, pred):
    return _share(pred, clip.anchor, clip.info["gap_stop"], clip.info["team"]) >= 0.9


def find_change_in_ball_gap(ctx):
    missing = np.isnan(ctx.seen.ball[:, 0])
    out = []
    for i, (a, b, k) in enumerate(ctx.segs):
        if ctx.contiguous(i):
            k2 = ctx.segs[i + 1][2]
            if _alive(k) and _alive(k2) and missing[max(0, b - 2 * S):b + 2 * S].all():
                out.append(_clip(ctx, b, frm=int(k), to=int(k2)))
    return out


def check_transition_two_seconds(clip, gt, pred):
    return _transition_near(pred, clip.anchor, clip.info["frm"], clip.info["to"], 2 * S)


def find_blackout_change(ctx):
    tracked = (~np.isnan(ctx.seen.players[..., 0])).sum(1)
    out = []
    for i, (a, b, k) in enumerate(ctx.segs):
        if ctx.contiguous(i):
            k2 = ctx.segs[i + 1][2]
            if _alive(k) and _alive(k2) and tracked[b] <= 18:
                out.append(_clip(ctx, b, frm=int(k), to=int(k2), tracked=int(tracked[b])))
    return out


def check_transition_one_second(clip, gt, pred):
    return _transition_near(pred, clip.anchor, clip.info["frm"], clip.info["to"], S)


def find_duel_flurry(ctx):
    changes = [s for s in range(1, len(ctx.gt))
               if _alive(ctx.gt[s - 1]) and _alive(ctx.gt[s]) and ctx.gt[s - 1] != ctx.gt[s]]
    out, last = [], -10 ** 9
    for j in range(len(changes) - 2):
        if changes[j + 2] - changes[j] <= 5 * S and changes[j] > last + 10 * S:
            end = changes[j] + 5 * S
            n = sum(1 for c in changes[j:] if c <= end)
            out.append(_clip(ctx, changes[j], before=2, after=7, end=int(end), n_changes=n))
            last = changes[j]
    return out


def check_duel_flurry(clip, gt, pred):
    lo, end = clip.anchor - S, clip.info["end"]
    p = pred[lo:end]
    n_pred = int(np.sum(np.isin(p[:-1], (H, A)) & np.isin(p[1:], (H, A)) & (p[:-1] != p[1:])))
    after = gt[end + S // 2]
    return n_pred <= clip.info["n_changes"] and (not _alive(after) or pred[end + S // 2] == after)


def find_kickoff(ctx):
    out = []
    for _, first, last in ctx.clean.halves:
        i = next((i for i, (a, b, k) in enumerate(ctx.segs) if a >= first and _alive(k)), None)
        if i is not None and i > 0 and ctx.segs[i - 1][2] == D:
            a, _, k = ctx.segs[i]
            out.append(_clip(ctx, a, before=5, after=5, team=int(k)))
    return out


def check_kickoff(clip, gt, pred):
    return (_share(pred, clip.anchor - 3 * S, clip.anchor - S, D) >= 0.8
            and _share(pred, clip.anchor + S, clip.anchor + 3 * S, clip.info["team"]) >= 0.8)


def find_dead_ball_visible(ctx):
    seen = ~np.isnan(ctx.seen.ball[:, 0])
    return [Clip(ctx.name, int(a), max(0, a - 2 * S), min(len(ctx.gt), b + 2 * S),
                 {"stop": int(b)})
            for a, b, k in ctx.segs if k == D and 5 * S <= b - a < 30 * S and seen[a:b].mean() >= 0.8]


def check_dead_ball_visible(clip, gt, pred):
    return _share(pred, clip.anchor + S, clip.info["stop"] - S, D) >= 0.8


def find_slow_play(ctx):
    tracked = ~np.isnan(ctx.clean.players[..., 0])
    speed = np.where(tracked, np.nan_to_num(ctx.clean.players[..., 2]), 0).sum(1) / np.maximum(
        tracked.sum(1), 1)
    slow = np.convolve(speed, np.ones(4 * S) / (4 * S), "same") < 1.3
    out = []
    for a, b, k in ctx.segs:
        if _alive(k) and b - a >= 8 * S:
            hit = np.flatnonzero(slow[a + 2 * S:b - 2 * S])
            if len(hit):
                out.append(_clip(ctx, a + 2 * S + hit[0], before=3, after=5, team=int(k)))
    return out


def check_slow_play(clip, gt, pred):
    return _share(pred, clip.anchor - 2 * S, clip.anchor + 2 * S, clip.info["team"]) >= 0.8


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    what: str
    expected: str
    find: object
    check: object


SCENARIOS = [
    Scenario("pass_past_opponents", "Pass flies past opponents",
             "A fast ball inside a long possession, with an opponent nearer to it than any "
             "team-mate (the nearest-player trap).",
             "No change of possession: the passing team keeps it on >=90% of the +-1 s around it.",
             find_pass_past_opponents, check_hold_team),
    Scenario("interception", "Interception / tackle",
             "Direct hand-over between two possessions of >=3 s each, no stoppage.",
             "A change in the same direction within +-0.5 s of the true hand-over.",
             find_interception, check_transition_half_second),
    Scenario("short_spell", "Very short possession",
             "A possession under 1.5 s sandwiched between two opponent possessions of >=3 s "
             "(the public window has one of 0.8 s).",
             "A predicted spell of that team overlapping it with IoU >= 0.3, not merged away.",
             find_short_spell, check_short_spell),
    Scenario("restart_same", "Stoppage, same team restarts",
             "Ball dead 3-30 s, then the team that had it restarts (own throw-in, foul suffered).",
             "Dead predicted within 1 s of the stoppage, dead >=80% of it, the right team >=80% of "
             "the 2 s after the restart.",
             find_restart_same, check_restart),
    Scenario("restart_other", "Stoppage, other team restarts",
             "Ball dead 3-30 s, then the opponents restart (ball out, foul committed).",
             "Dead predicted within 1 s of the stoppage, dead >=80% of it, the NEW team >=80% of "
             "the 2 s after the restart.",
             find_restart_other, check_restart),
    Scenario("long_stoppage", "Long stoppage",
             "Ball dead >= 30 s: goal, injury, substitution, VAR.",
             "Dead on >=90% of the stoppage (5 s margins at both ends).",
             find_long_stoppage, check_long_stoppage),
    Scenario("ball_gap_hold", "Ball lost by the tracker, no change",
             "The tracking loses the ball for >=10 s inside one possession.",
             "Keep the possession: the right team on >=90% of the gap, no invented change or stop.",
             find_ball_gap_hold, check_ball_gap_hold),
    Scenario("change_in_ball_gap", "Change of possession while the ball is invisible",
             "A hand-over happening with the ball untracked for +-2 s around it.",
             "A change in the same direction within +-2 s, from player movement alone.",
             find_change_in_ball_gap, check_transition_two_seconds),
    Scenario("blackout_change", "Change during a tracker blackout",
             "A hand-over while 18 or fewer players are tracked.",
             "A change in the same direction within +-1 s.",
             find_blackout_change, check_transition_one_second),
    Scenario("duel_flurry", "Contested ball / duel",
             ">=3 hand-overs within 5 s (duels, ricochets).",
             "No more predicted changes than true ones in the window, and the right team 0.5 s "
             "after it.",
             find_duel_flurry, check_duel_flurry),
    Scenario("kickoff", "Half start: dead to kick-off",
             "The first possession of a half, after the pre-kickoff dead stretch.",
             "Dead before (-3 to -1 s), the kicking team after (+1 to +3 s).",
             find_kickoff, check_kickoff),
    Scenario("dead_ball_visible", "Dead ball, ball still tracked",
             "Ball dead 5-30 s while the tracker still sees it (set-piece preparation, foul).",
             "Dead on >=80% of the stoppage: a visible ball is not proof of play.",
             find_dead_ball_visible, check_dead_ball_visible),
    Scenario("slow_play", "Slow build-up",
             "Live possession while players walk (mean speed < 1.3 m/s over 4 s): the trap for a "
             "speed-based dead-ball detector.",
             "Stay alive with the right team on >=80% of the +-2 s.",
             find_slow_play, check_slow_play),
]


def context(name, clean, seen):
    gt = clean.labels.astype(np.int64)
    return Context(name, clean, seen, gt, pm.segments(gt, pm.valid_runs(gt)))


def evaluate(contexts, predictions):
    """{scenario: {method: (passed, total)}} and {scenario: [Clip]}; predictions[method][name]."""
    clips, table = {}, {}
    for sc in SCENARIOS:
        clips[sc.name] = [c for ctx in contexts for c in sc.find(ctx)]
        gts = {ctx.name: ctx.gt for ctx in contexts}
        table[sc.name] = {
            method: (sum(sc.check(c, gts[c.match], preds[c.match]) for c in clips[sc.name]),
                     len(clips[sc.name]))
            for method, preds in predictions.items()}
    return table, clips
