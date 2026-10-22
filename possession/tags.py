"""Smart Tagging: Metrica's automatic phase tags for the hackathon match, as evidence.

638 intervals, each with a code (BUILD UP, DEFENDING IN MIDDLE THIRD, SET PIECES, ...) and tags
(Team, Half, Type, ...). Inferred automatically, so noisy; phases, not possessions; `Team` is
N/A on 36 instances, every throw-in among them (DATA.md). Never copied as labels.

What `Team` means is code-dependent (the possessing team for BUILD UP, possibly the defending
team for DEFENDING IN ... THIRD). `scripts/tag_agreement.py` measures it on the public window
(aggregates only); SIGN holds the result: +1 the tagged team has the ball, -1 the other team.

    load(match) -> [Tag(start, end, code, team, kind)]  video frames, end exclusive; team H/A/None
    features(match, tags) -> per-frame arrays (team as +1 home / -1 away / 0 none)
    bonus(match, tags) -> (T, 3) additive scores for the H/A/D decoder

PRIVACY: reads the gated file; prints nothing frame-level.
"""

from dataclasses import dataclass

import numpy as np

from bench import import_starter
from bench.match import A, H

TEAMS = {"barcelona": H, "atletic club": A}   # sic, the file's spelling
SET_PIECES = "set pieces"
# Per code: does the tagged team have the ball (+1) or not (-1)? Absent = no team information.
# Measured Oct 22 with scripts/tag_agreement.py on the public window (share of the tags' live
# frames with the ball at the tagged team; only 1-5 tags per code there, so this fixes the SIGN,
# not the weight -- the weight comes from simulated tags on DFL):
#   build up 100%, creating chances 100%, long ball 90%, progression 73%, attacking transition 67%
#   defending in attacking / defensive third 0%, defending in middle third 27%, defensive
#   transition 33%; in the box / final third 44-54% (no team information); set pieces 96%.
SIGN = {"build up": 1, "creating chances": 1, "long ball": 1, "progression": 1,
        "attacking transition": 1,
        "defending in attacking third": -1, "defending in middle third": -1,
        "defending in defensive third": -1, "defensive transition": -1}


@dataclass(frozen=True)
class Tag:
    start: int
    end: int
    code: str
    team: object      # H, A or None
    kind: str         # Type tag, lower case ("throw-in", "corner kick", ...) or ""


def load(game="BAR-ATH"):
    import_starter()
    from starter import load_smart_tagging as st
    out = []
    for e in st.load(game):
        team = TEAMS.get((e["tags"].get("Team") or "").strip().lower())
        out.append(Tag(e["start_frame"], e["end_frame"], (e["code"] or "").strip().lower(), team,
                       (e["tags"].get("Type") or "").strip().lower()))
    return out


def features(match, tags):
    """tag_team: signed team of the covering phase tags (mean if several, via SIGN);
    tag_phase: 1 inside any phase tag; setpiece_since / setpiece_until: seconds to the nearest
    SET PIECES start; setpiece_team: its team (0 if N/A)."""
    n = match.n_frames
    tot, cnt, phase = np.zeros(n), np.zeros(n), np.zeros(n)
    sp = []
    for t in tags:
        a, b = max(0, t.start), min(n, t.end)
        if b <= a:
            continue
        if t.code == SET_PIECES:
            sp.append((t.start, 0.0 if t.team is None else (1.0 if t.team == H else -1.0)))
            continue
        phase[a:b] = 1
        s = SIGN.get(t.code)
        if s is None or t.team is None:
            continue
        tot[a:b] += s * (1.0 if t.team == H else -1.0)
        cnt[a:b] += 1
    out = {"tag_team": np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), "tag_phase": phase}
    from .touches import _last_next
    sp.sort()
    f = np.asarray([x for x, _ in sp], dtype=np.int64)
    v = np.asarray([y for _, y in sp])
    last, since, nxt, until = _last_next(f, v, n)
    out.update(setpiece_since=since, setpiece_until=until, setpiece_team=nxt)
    return out


# One tag is ONE piece of evidence, however long: its log-odds are spread over its frames, so a
# 10 s phase does not count 250 times (the correlated-evidence mistake M3 exposed). Weight: the
# lowest public-window agreement among the informative codes, 67% -> log(0.67 / 0.33) ~ 0.7.
# Conservative on purpose; only checked on the real match's rate and durations, never tuned.
LOGODDS = 0.7


def bonus(match, tags, logodds=LOGODDS):
    """(T, 3) scores to add to M2's log-probabilities: +-logodds/2 per tag on H vs A, spread
    evenly over the tag's frames; dead-ball scores untouched. Set pieces are not used (yet)."""
    from bench.match import D
    n = match.n_frames
    diff = np.zeros(n)
    for t in tags:
        s = SIGN.get(t.code)
        a, b = max(0, t.start), min(n, t.end)
        if s is None or t.team is None or b <= a:
            continue
        diff[a:b] += s * (1.0 if t.team == H else -1.0) * logodds / (b - a)
    out = np.zeros((n, 3))
    out[:, H], out[:, A] = diff / 2, -diff / 2
    out[:, D] = 0.0
    return out
