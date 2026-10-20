# Benchmark of the best method (Oct 20)

Reproduce with `uv run python scripts/benchmark.py` (about 5 min; raw numbers in
`reports/benchmark/summary.json`). Every DFL number is held-out: each match is predicted by the
model trained without it. The oracles are covered by `tests/test_oracles.py`.

Setup: 7 DFL matches × 3 degrader seeds (the same matches with different simulated tracking
noise), hackathon noise profile, official scorer.

## 1. Ranking: is M2 really the best?

Paired per-match comparison against M2 (seeds averaged; ± is the standard error over 7 matches).

| method | spellF1 | macroF1 | vs M2 | matches better than M2 |
|---|---|---|---|---|
| M2, switch 5 | **0.526** | 0.885 | **+0.006 ± 0.002** | **7 / 7** |
| **M2 learned** (switch 10, current) | 0.519 | 0.884 | — | — |
| M3 duration priors | 0.514 | 0.884 | −0.005 ± 0.002 | 1 / 7 |
| M3 gap weighting κ=200 | 0.508 | 0.880 | −0.011 ± 0.002 | 0 / 7 |
| M2, switch 20 | 0.502 | 0.882 | −0.017 ± 0.003 | 0 / 7 |
| M1 hmm | 0.355 | 0.801 | −0.164 ± 0.011 | 0 / 7 |
| smooth50 | 0.251 | 0.583 | −0.268 ± 0.012 | 0 / 7 |

- M2 is the best family. The M3 decoders lose to it on almost every match.
- A switch penalty of 5 beats 10 on **every** match, but only by 0.006, which is tiny next to the
  0.04 noise band of a single match. On the real match it predicts 4.57 possessions per live
  minute (public window: 4.25, ±1 from only 17 possessions), and its predicted durations
  match the public labels' quantiles more closely (see §5). It's a free but small change.
- Different noise realisations move a match's score by ±0.01, so method differences below about
  0.01 need the paired comparison above to be believed.
- Per-match M2 ranges from 0.458 to 0.581. Hard matches stay hard for every method.

## 2. Sensitivity to the tracking noise

| tracking given to M2 | spellF1 |
|---|---|
| clean (perfect tracking) | 0.579 |
| ball degraded only | 0.539 |
| hackathon profile (calibrated) | 0.522 |
| stress (more and longer gaps, more blackouts) | 0.497 |

M2 degrades gracefully. More importantly, **perfect tracking only reaches 0.579**. Most of the lost
points are not caused by bad tracking; they come from deciding who has the ball. Fixing the
tracking (for example by re-detecting the ball in the video) can win at most ~0.04–0.06.

## 3. Oracles: how much each perfect fix would gain

Each line replaces M2's output with the truth for one kind of error only (base: 0.519).

| perfect fix | spellF1 | gain |
|---|---|---|
| team, on frames M2 and the truth both call live | 0.909 | **+0.390** |
| possessions shorter than 2 s | 0.755 | **+0.236** |
| hand-overs (±1 s around each H↔A change) | 0.699 | **+0.180** |
| frames where the ball is missing | 0.693 | **+0.174** |
| stoppage starts (±2 s around live → dead) | 0.632 | +0.112 |
| restarts (±2 s around dead → live) | 0.562 | +0.043 |
| ball always tracked (players still noisy) | 0.558 | +0.036 |
| timing of changes (moved to the true one within 1 s) | 0.547 | +0.028 |
| dead/alive call (team kept from M2) | 0.540 | +0.021 |

The oracles overlap (a short possession is usually also a hand-over), so the gains don't add up.
Read them as ceilings for each direction.

## 4. Failure anatomy of M2 (3,061 true possessions, seed 0)

**By duration** (recall = matched at IoU 0.5)

| duration | possessions | recall | missed |
|---|---|---|---|
| < 1 s | 384 | 2% | 378 |
| 1–3 s | 934 | 17% | 777 |
| 3–10 s | 1,104 | 52% | 525 |
| ≥ 10 s | 639 | 87% | 82 |

**By how the possession starts**

| start | possessions | recall | missed |
|---|---|---|---|
| after a hand-over | 2,351 | 36% | **1,512 (86% of all misses)** |
| after a stoppage | 699 | 65% | 247 |
| at a half start | 11 | 73% | 3 |

**By ball visibility during the possession**

| ball tracked | possessions | recall | missed |
|---|---|---|---|
| > 75% of frames | 2,187 | 45% | **1,208** |
| 25–75% | 168 | 70% | 50 |
| < 25% | 706 | 29% | 504 |

M2 also predicts 541 possessions that match nothing, mostly 3–10 s long (236) and 1–3 s (164).

**Calibration** of the per-frame probabilities is excellent: the expected calibration error is
0.004 for P(dead) and 0.007 for P(home | live). The model is not over- or under-confident, so
nothing is to be gained from recalibrating; gains must come from new information.

**Most-used features** (share of total split gain): mean player speed over 8 s 22%, team-control
evidence over 2 s 21%, distance of the next-seen ball to a line 7%, speed over 3 s 6%, control
over 5 s 6%. The team-shape and gap features each count under 3%.

## 5. The real match (aggregates only)

| | public spellF1 | possessions / live min | durations p10 / p25 / p50 / p75 / p90 (s) |
|---|---|---|---|
| public ground truth (17 possessions) | — | 4.25 | 2.2 / 3.5 / 7.6 / 20.0 / 38.1 |
| DFL ground truth (for contrast) | — | 8.4 | 0.8 / 1.7 / 3.7 / 8.5 / 18.6 |
| M2 learned | 0.622 | 4.16 | 2.4 / 4.0 / 9.5 / 21.4 / 33.2 |
| M2, switch 5 | 0.622 | 4.57 | 1.8 / 3.5 / 7.9 / 19.4 / 31.4 |
| M3 duration priors | 0.622 | 3.92 | 2.7 / 4.1 / 10.0 / 22.4 / 36.2 |
| smooth50 | 0.326 | 8.29 | 0.4 / 1.4 / 4.6 / 9.2 / 18.0 |

On the real match, M2's possessions look like the real ones, and switch 5 even more so. The
public window can't separate the M2 variants (all 0.622): 17 possessions are too few.

## 6. What to really aim at

1. **Who has the ball during live play, at hand-overs.** This is the dominant loss:
   - team perfect: +0.39;
   - 86% of the missed possessions start with a hand-over;
   - most misses happen with the ball tracked (1,208 of 1,762).

   So it's not mainly a ball-gap problem: proximity ("a player within 1.5 m") is too crude to
   tell a control from a touch, a duel or a pass flying by. This is **direction 2 (touch/control
   detector)**, which should go first. Its specs are written, and the DFL events now give real
   passes, interceptions and tackles to check it against.
2. **Short possessions** (+0.24 ceiling), but **discount it for the hackathon**: DFL's labels
   create many more short possessions (p10 0.8 s) than the hackathon's (p10 2.2 s, and 2 of 17
   public possessions under 1.5 s). On the real match the payoff is probably a third of that or
   less. It mostly comes for free with direction 2 anyway.
3. **The possession just before a stoppage, and the stoppage start** (+0.11). This is bigger than
   restarts themselves (+0.04). The perfect dead/alive call only gains +0.02, so the gain around
   stoppages is about *who* had the last possession (often a short, contested one: a shot or a
   cross going out), not about the dead-ball timing. **Direction 1 should be re-aimed** at
   stoppage starts: which team touched it last before it went out. The restart detector alone has
   a low ceiling.
4. **Ball gaps** (+0.17 ceiling) remain valuable but are second. The gap features showed that the
   current inputs already describe gaps; what's missing is training examples (PFF World Cup:
   64 matches) or recovering the ball itself (video).
5. **Not worth pursuing now**:
   - decoder variants: M3 is ≤ 0, and the switch penalty is ±0.006;
   - timing refinement: +0.03;
   - dead/alive detection: +0.02;
   - ball re-detection alone: +0.04 (its value is through the gaps' *team* question, item 4).
6. **Free small change:** switch penalty 10 → 5. It wins on 7 of 7 matches (+0.006), and its
   real-match durations are closer to the public labels.
