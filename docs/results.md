# Results log

Every result we obtained, newest last. Add a row with each experiment, adopted or not. The
explanation of the methods is in `report.html`; the full benchmark tables are in `benchmark.md`.

How to read the numbers:

- **DFL** is held-out spellF1 on the bench: 7 public DFL matches, degraded to look like the
  hackathon tracking. Each match is predicted by a model trained without it (leave-one-match-out).
  It is the main score.
- **Public** is spellF1 on the 5 public minutes of the real match. There are only 17 possessions,
  so use it as a sanity check, not for ranking.
- **Rate** is the number of predicted possessions per live minute over the whole real match. The
  public labels give 4.25. It checks the change rate, because DFL's labels change possession about
  twice as often as the hackathon's.
- **Noise band.** For one match, spellF1 differences below about 0.04 are noise. Across the 7
  bench matches, a difference needs a paired per-match comparison before it can be believed: a new
  noise draw moves a match by about ±0.01.

## Methods

| date | method | DFL spellF1 | DFL macroF1 | public | rate | status |
|---|---|---|---|---|---|---|
| Oct 20 | smooth50 (organisers' baseline) | 0.247 | 0.585 | 0.326 | 8.29 | floor |
| Oct 20 | M1: hand-built evidence + Viterbi (switch 40) | 0.352 | 0.80 | 0.413 | 4.35 | superseded |
| Oct 20 | M2: LightGBM evidence + Viterbi (switch 10) | 0.522 | 0.883 | 0.622 | 4.16 | **adopted** |
| Oct 20 | M3: semi-Markov, 17 variants | 0.463–0.517 | – | 0.622 | 3.54–4.11 | rejected |
| Oct 20 | M2 + ball-gap features (direction 3) | 0.5205 | – | – | – | rejected (null) |
| Oct 21 | M2 retrained on the corrected degrader (base, 42 features) | 0.523 | 0.883 | 0.622 | 4.16 | reference |
| Oct 21 | M2 base, switch 5 | 0.529 | 0.884 | 0.622 | 4.51 | +0.006 ± 0.001, 7/7; not adopted yet |
| Oct 21 | **M2 + touch features (direction 2), 52 features** | **0.550** | **0.890** | 0.622 | 4.32 | **adopted**: +0.028 ± 0.006 vs base, 7/7 |
| Oct 21 | M2 + touches, switch 5 | 0.554 | 0.891 | 0.552 | 4.69 | +0.004 vs switch 10; public drops; not adopted |

Public-window macroF1: base 0.894, touches 0.861 (17 possessions; bench macroF1 went up). Watch it.
| Oct 21 | M2 + touches + carrier features (61 features), switch 10 | 0.560 | 0.894 | 0.533 | 4.60 | rejected: +0.0095 ± 0.0016 (7/7) on DFL, but only by changing possession more often |
| Oct 21 | same, switch 17 (real rate back to 4.32) | 0.543 | – | 0.533 | 4.32 | rejected: at equal real-match rate it is no better than M2 + touches (0.547) |
| Oct 21 | path decoder (player-level), default costs, seed 0 | 0.532 | 0.887 | 0.607 | 4.53 | rejected: −0.015 ± 0.006 vs M2 (1/7); 6 cost variants all −0.013 to −0.022 |
| Oct 30 | M2 + touches + dead until the first reception (decoder), today's code | 0.507 on reception labels / 0.510 on DFL's | 0.886 / 0.883 | 0.622 | 4.43 | off by default: +0.016 ± 0.002 (7/7) or −0.024 ± 0.002 (0/7) depending on the convention; see below |

Switch sweep, seed 0 (DFL spellF1 / real rate / public): M2 + touches 10: 0.547 / 4.32 / 0.622;
12: 0.543 / 4.23; 14: 0.538 / 4.21; 20: 0.529 / 4.16. M2 + carrier 10: 0.556 / 4.60 / 0.533;
14: 0.552 / 4.48; 17: 0.543 / 4.32; 20: 0.535 / 4.20 / 0.600. Lesson: a DFL gain must be read at the
real match's change rate, because DFL's labels change possession twice as often.

## Benchmark of M2 (Oct 20, before the degrader noise fix)

7 matches × 3 noise draws. "Wins" counts the matches on which a method beats M2, seeds averaged.

| method | spellF1 | macroF1 | vs M2 (± s.e.) | wins |
|---|---|---|---|---|
| M2, switch 5 | 0.526 | 0.885 | +0.006 ± 0.002 | 7 / 7 |
| M2, switch 10 | 0.519 | 0.884 | – | – |
| M3, duration priors | 0.514 | 0.884 | −0.005 ± 0.002 | 1 / 7 |
| M3, gap weighting κ=200 | 0.508 | 0.880 | −0.011 ± 0.002 | 0 / 7 |
| M2, switch 20 | 0.502 | 0.882 | −0.017 ± 0.003 | 0 / 7 |

Sensitivity of M2 to the tracking quality:

| tracking | spellF1 |
|---|---|
| clean | 0.579 |
| ball degraded only | 0.539 |
| hackathon-like | 0.522 |
| stress | 0.497 |

Oracle ceilings: the gain over 0.519 if one kind of error were fixed perfectly.

| fix | gain |
|---|---|
| team in live play | +0.390 |
| possessions under 2 s | +0.236 |
| hand-overs | +0.180 |
| ball-missing frames | +0.174 |
| stoppage starts | +0.112 |
| restarts | +0.043 |
| ball always tracked | +0.036 |
| timing | +0.028 |
| dead/alive | +0.021 |

## Edge-case scenarios (M2, seed 0)

Pass rates on situations mined from the DFL matches (`python -m bench scenarios`).

| scenario | clips | smooth50 | M1 | M2 (Oct 20) | M2 base (Oct 21) | **M2 + touches** |
|---|---|---|---|---|---|---|
| pass past opponents | 633 | – | – | 95% | 93% | 93% |
| interception | 770 | – | – | 28% | 30% | **40%** |
| short possession | 84 | – | – | 4% | 6% | 8% |
| restart, same team | 120 | – | – | 56% | 48% | 57% |
| restart, other team | 355 | 1% | 23% | 55% | 54% | 55% |
| long stoppage | 210 | – | – | 99% | 99% | 99% |
| hold through a 10 s+ ball gap | 201 | 0% | 23% | 23% | 21% | 22% |
| change while the ball is missing | 423 | 4% | 0% | 25% | 25% | 23% |
| change during a blackout | 195 | – | – | 14% | 18% | 16% |
| duel flurry | 201 | – | – | 74% | 75% | 75% |
| kick-off | 8 | – | – | 12% | 12% | 12% |
| dead ball, ball visible | 111 | – | – | 93% | 93% | 95% |
| slow live play | 126 | – | – | 79% | 79% | 83% |

"–": not recorded. The Oct 20 and Oct 21 columns differ in the degrader (position-noise fix), so
small moves between them are bench changes, not method changes.

## Bench and data changes that move the numbers

| date | change | effect |
|---|---|---|
| Oct 20 | Degrader calibrated on the real tracking: ball gaps, blackouts, tracklets | smooth50 on bench 0.247 vs 0.243 on the real match |
| Oct 20 | Fix: identity breaks reused the same slot | slots no longer follow one player forever |
| Oct 20 | Fix: DFL referees were parsed as away players (2 matches) | the nearest "away player" is never the referee |
| Oct 20 | DFL official events synced (9,961 events) | passes agree with the labels 98%; set pieces 97–100% |
| Oct 21 | Fix: the degrader position noise is now smooth (frame-to-frame jitter 58–88 cm → 1.2–2.2 cm; real: 0.1–0.9 cm) | velocities are usable; M2 retrained: 0.522 → 0.523 |
| Oct 21 | Metrica sample games 1–2 added (CSV reader); 10 open matches for the carrier and ball-filling models | M2 still trains and is judged on DFL only (Metrica labels come from events) |

## Touch detector (direction 2), checked against DFL events

| check | clean | hackathon-like |
|---|---|---|
| detected control agrees with the label | 87% | 87% |
| passer controls just before the pass | 78% | 59% |
| interception or won tackle: control by the winner | 22% | 16% |
| interception or won tackle: contact by the winner | 87% | 61% |

## Carrier model (Oct 21), held-out, 10 open matches × 2 noise draws

| frames with a carrier | right player | right team |
|---|---|---|
| ball seen: model / nearest to the ball | 98.7% / 98.3% | 98.9% / 98.6% |
| ball missing: model | **57.7%** | **76.9%** |
| ball missing: nearest to the last-seen ball (best rule) | 40.0% | 74.1% |

Summed P(someone carries), ball missing: 28% when someone does, 12% live when nobody does, 5% dead.

## Ball filling (Oct 21), held-out, 10 open matches, gap frames with a true ball

Median / p90 error in metres (v2: L1 loss, out-of-fold 80% radius).

| time since lost | model | straight line | stays where last seen |
|---|---|---|---|
| 0–1 s | 0.9 / 4.1 | 0.5 / 3.1 | **0.4 / 2.5** |
| 1–3 s | 1.2 / **8.4** | 1.3 / 10.1 | **0.4** / 10.9 |
| 3–10 s | 2.7 / **15.2** | 4.5 / 21.2 | **0.5** / 27.9 |
| over 10 s (58% of gap frames) | **9.6 / 29.9** | 15.3 / 44.5 | 17.8 / 63.2 |

80% radius covers 77% of true errors (v1, in-sample log-error model: 13%). Team of the nearest
player to the filled ball = possessing team on 50% of live gap frames (straight line: 53%), so
the filled ball is not a team signal by itself.

## Where this leaves us (Oct 21)

- Best: M2 + touches (DFL 0.550, real rate 4.32, public 0.622). It is the submission.
- Most potential: evidence about who really plays the ball (touches, carrier, path decoder as a
  refinement of M2's changes).
- Main risk: the bench's labels define possession differently (DFL changes it twice as often), so
  DFL gains must be read at the real match's change rate. Next fix: relabel DFL with a
  hackathon-style rule matched to the public labels' rate and durations.

## Fixing the bench: hackathon-style labels on DFL (Oct 22)

`bench/relabel.py`: dead from DFL's BallStatus; possession changes only when the other team
controls the ball (>= control_s within near_m, moving with it); everything else keeps the last
team. Calibrated against the public labels (17 possessions, so targets with error bars).

| labels | possessions / live min | durations p10 / p25 / p50 / p75 / p90 (s) |
|---|---|---|
| DFL original | 8.30 | 0.8 / 1.7 / 3.7 / 8.5 / 18.6 |
| control 0.2 s, 1.0 m | 4.34 | 2.0 / 4.3 / 8.8 / 18.0 / 31.3 |
| **control 0.4 s, 1.5 m (chosen)** | **4.35** | **2.2 / 4.3 / 8.8 / 17.8 / 31.3** |
| control 0.6 s, 2.0 m | 4.13 | 2.4 / 4.8 / 9.4 / 19.1 / 32.4 |
| control 1.0 s, 1.5 m | 3.21 | 3.2 / 6.4 / 13.0 / 24.1 / 41.5 |
| public labels | 4.25 | 2.2 / 3.5 / 7.6 / 20.0 / 38.1 |

Per match, the chosen rule gives 3.76–5.18 possessions per live minute. The relabelled team
agrees with DFL's on 85% of live frames: the definitions differ mostly in short changes.
Caveat: the rule uses the same notion of control as our touch detector, so a model with touch
features may find these labels a little easier than Metrica's real (unknown) rule.

### Retraining on the fixed labels (Oct 22)

`scripts/compare_models.py` (3 seeds, 7 matches, scored on the hackathon-style labels):

| switch 10 | DFL spellF1 (new labels) | predicted rate on bench (true 4.38) | real rate | public |
|---|---|---|---|---|
| A: M2 + touches, trained on DFL's labels | 0.586 | 5.38 | 4.32 | 0.622 |
| B: same, trained on the new labels | **0.732** (+0.146 ± 0.006, 7/7) | 4.14 | **3.55** | **0.433** |

B is far better and well calibrated on the bench (interception 24% → 56%), but on the real
match it changes possession too rarely at every switch (3.37–3.64) and the public window drops.
Not adopted. Both models lose 0.6–1.0 changes per live minute from bench to real match.

Touch-detector events per graded minute (`scripts/touch_rates.py`):

| | ball seen | controls | control-team changes | deflections | contacts |
|---|---|---|---|---|---|
| DFL clean | 100% | 10.6 | 1.91 | 7.7 | 18.9 |
| DFL degraded (bench) | 61% | 7.9 | 1.48 | 6.1 | 14.7 |
| real hackathon tracking | 59% | 5.6 | 1.34 | **2.1** | 12.1 |

Gaps are well calibrated, but the real ball's path is much smoother (a third of the sharp
turns): the real tracker most likely filters the ball. The degrader does not, so bench models
expect more touch evidence than the real tracking gives. Next: calibrate a ball smoothing in the
degrader on these rates (aggregates only), then retrain B and re-check the real rate.

### Ball smoothing in the degrader (Oct 22)

`scripts/calibrate_ball_smoothing.py`: one knob (Gaussian std of the ball path), fitted on the
whole real match's |acceleration| percentiles over 1/2/4 frames (no labels, no model output).

| ball | a1 p50 / p90 / p99 | a4 p50 / p90 / p99 (m/s²) |
|---|---|---|
| real hackathon | 5.7 / 18.3 / 36.8 | 5.7 / 18.1 / 37.0 |
| bench, no smoothing | 9.8 / 37.2 / 356.4 | 2.7 / 22.1 / 103.0 |
| **bench, smoothing 0.20 s (chosen)** | 3.1 / 17.4 / 42.0 | 3.1 / 17.1 / 40.2 |

Check (not fitted): touch events per minute, mean [range over 7 DFL matches].

| | controls | deflections | contacts |
|---|---|---|---|
| bench, no smoothing | 7.9 [6.4–9.0] | 6.1 [5.1–9.2] | 14.7 [12.5–20.7] |
| bench, smoothing 0.20 s | 6.1 [5.0–7.1] | 2.3 [2.0–2.5] | 10.2 [7.5–12.6] |
| real hackathon | 5.6 | 2.1 | 12.1 |

Not matched: the real ball's median |acceleration| (5.7 vs 3.1), a slow wobble a second noise
term could add. Robustness label rules built: lo (0.2 s, 1.0 m) and hi (0.6 s, 2.0 m).

### Smart Tagging (Oct 22)

`possession/tags.py`, `scripts/tag_agreement.py`, `scripts/eval_tags.py`. What `Team` means per
code, measured on the public window (share of the tags' live frames with the ball at the tagged
team; 1–5 tags per code there):

| codes | with tagged team | meaning |
|---|---|---|
| build up, creating chances, long ball, progression, attacking transition | 67–100% | team with the ball |
| defending in attacking / middle / defensive third, defensive transition | 0–33% | team without the ball |
| players in the box, ball in the box, ball in final third | 44–54% | no team information |
| set pieces | 96%, restarting team 2/2 | restarting team (when not N/A) |

Used as one piece of evidence per tag (log-odds 0.7, from the lowest agreement, 67%, spread over
the tag's frames) added to M2's H/A scores. Real match: frames changed < 0.01% (0.12% at 4× the
weight), rate 4.32 and public 0.622 unchanged. M2 already agrees with the tags' team on 84.9%
of their live frames; it disagrees on 32 of 306 informative tags, and nothing tells who is right.
Not used in the submission (redundant at a safe weight; a stronger one cannot be validated).

### Snapping hand-overs to the control frame (Oct 22)

`possession/snap.py`: moves each of M2's direct H <-> A changes onto the start of the nearest
control by the new team (within 2 s); adds no changes. Current model, smoothed bench, seed 0:

| bench labels | M2 | M2 + snap | paired difference |
|---|---|---|---|
| DFL original | 0.508 | 0.496 | −0.012 ± 0.003 (0/7) |
| hackathon-style | 0.593 | 0.609 | +0.016 ± 0.003 (7/7) |
| real match | rate 4.32, public 0.622 | rate 4.32, public 0.622 | none |

Not adopted: the gain on the hackathon-style labels is circular (they place changes at control
starts by construction); DFL's labels place them elsewhere. Which matches Metrica's rule is
unknown. Possible check (aggregates only): distance of the public window's changes to the
nearest control start by the new team.

Change timing in the public window (`scripts/change_timing.py`, 12 H <-> A changes, real
tracking): nearest control start by the new team is +0.94 s after the label change (quartiles
−3.3 / +3.4 s, 8% within 0.5 s); nearest touch of any kind is +0.12 s (−0.7 / +1.0 s, 33% within
0.5 s). Metrica's labels change near the new team's first touch, not at its control start, so
the relabel rule gets the count right but probably dates changes late. Next: rule v2.

### Relabel v2 and degrader realism (Oct 22)

Relabel v2 dates each change from the new team's first touch (the control still decides it):
same rate (4.35) and durations, and the timing of changes vs detected touches now resembles the
real labels.

| labels | to new team's control start: median, within 0.5 s | to any touch: median, within 0.5 s |
|---|---|---|
| public labels (real, 12 changes) | +0.94 s, 8% | +0.12 s, 33% |
| DFL original | +1.16 s, 15% | +0.04 s, 42% |
| relabel v1 | +0.04 s, 60% | +0.00 s, 68% |
| relabel v2 | +1.08 s, 27% | +0.36 s, 46% |

Things the degrader did not model (`scripts/degrader_realism.py`):

| | real | bench mean [range] |
|---|---|---|
| ball jumps > 2 m in one frame, per min | 1.12 | 0.13 [0.07–0.33] |
| implied speed across gaps < 2 s | 66 m/s | 0.6 m/s |
| gap start: ball speed p50 / p90 | 1.7 / 6.4 m/s | 0.5 / 1.6 m/s |
| gap start: players within 3 m | 1.04 | 0.57 |
| track ends per min | 11.3 | 11.0 |
| track ends with an opponent within 2 m | 33% | 12% |

New off-by-default knobs: false ball detections, identity breaks in duels; calibration pending.

### Retraining on the smoothed bench + robustness (Oct 22)

All scored on the main hackathon-style labels (3 seeds), switch 10:

| model | bench spellF1 | bench rate pred / true | real rate | public | public-window possessions (true 17) |
|---|---|---|---|---|---|
| current submission (M2 + touches, unsmoothed bench) | – | – | 4.32 | 0.622 | 10 |
| A_s: DFL labels, smoothed bench | 0.628 | 5.03 / 4.38 | 3.90 | 0.622 | 10 |
| B (v1 labels, unsmoothed) | 0.732 | 4.14 / 4.38 | 3.55 | 0.433 | 7 |
| B_s: v1 labels, smoothed | 0.735 (+0.106 ± 0.005, 7/7) | 4.18 / 4.38 | 3.22 | 0.330 | 6 |
| B_lo: rule 0.2 s / 1.0 m | 0.731 (+0.102, 7/7) | 4.07 / 4.38 | 3.23 | 0.365 | 6 |
| B_hi: rule 0.6 s / 2.0 m | 0.730 (+0.102, 7/7) | 4.00 / 4.38 | 3.17 | 0.450 | 7 |

- Robust to the rule: lo / main / hi give the same picture.
- Ball smoothing made the bench's ball realistic but widened the bench -> real gap (B 3.55 ->
  3.22, A 4.32 -> 3.90): it is not what hides real hand-overs.
- Every model, the submission included, sees far fewer changes in the public window than there
  are (10 / 17 at best): hand-overs are much harder to see in the real tracking than on the
  bench. Suspects: false ball blips inside gaps, gaps starting as the ball leaves play, identity
  breaks in duels (structural knobs added; calibration v2 running).

### Degrader realism v2: structural knobs, calibrated (Oct 22)

New defaults (bench/degrade.py): false ball blips 0.5/min; stoppage gaps start as the ball
leaves play for half of them (lead ≤ 0.5 s); identity breaks ×81 near an opponent, mean track
length 300 s. All 7 DFL matches, mean [range]:

| | real | bench before | bench now |
|---|---|---|---|
| ball jumps per min | 1.12 | 0.13 | 0.97 [0.73–1.23] |
| gap start: ball speed p50 | 1.71 | 0.49 | 2.24 [0.99–3.54] |
| gap start: ball speed p90 | 6.45 | 1.63 | 12.5 [10.7–13.8] (overshoot) |
| gap start: players within 3 m | 1.04 | 0.57 | 0.96 [0.71–1.12] |
| implied speed, gaps < 2 s | 66 m/s | 0.6 | 13 [6–28] (still low) |
| implied speed, gaps 2–10 s | 1.16 | 0.13 | 2.4 [0.8–3.6] |
| track ends per min | 11.3 | 11.0 | 11.2 [9.7–12.8] |
| track ends near an opponent | 33% | 12% | 35% [32–39%] |

Relabel v2 is now the default (external/dfl_hk); v1 kept in external/dfl_hk_v1.

### Retraining on the realistic bench, v2 labels (Oct 22)

Scored on the v2 labels, 3 seeds, switch 10:

| model | bench spellF1 | bench rate pred / true | real rate | public | public possessions (of 17) |
|---|---|---|---|---|---|
| current submission | 0.611 | 5.21 / 4.38 | 4.32 | 0.622 | 10 |
| A_r: DFL labels, realistic bench | 0.649 (+0.038 ± 0.007 vs submission, 7/7) | 5.11 / 4.38 | 3.98 | 0.622 | 10 |
| B_r: v2 labels, realistic bench | 0.740 (+0.091 ± 0.006 vs A_r, 7/7) | 4.25 / 4.38 | 3.42 | 0.496 | 8 |
| (B_s: v1 labels, smoothed only) | 0.735 | 4.18 / 4.38 | 3.22 | 0.330 | 6 |

The structural realism knobs improved transfer (B: 6 -> 8 public possessions, public 0.33 ->
0.50, rate 3.22 -> 3.42) but did not close the bench -> real gap. Not adopted: B_r (the real-match
evidence still prefers the current behaviour); A_r is a small, consistent bench gain with the
same public result, held for now.

## Refresh on the fixed bench (Oct 22)

`MTAH_DFL_LABELS=hackathon uv run python scripts/benchmark.py`: hackathon-style labels (relabel
v2) and the realistic degrader; the submission model (M2 + touches).

| method | spellF1 | macroF1 | vs M2 | matches better |
|---|---|---|---|---|
| M2, switch 20 | 0.641 | 0.887 | +0.030 ± 0.002 | 7 / 7 |
| M3 duration priors | 0.625 | 0.886 | +0.014 ± 0.001 | 7 / 7 |
| M3 gap weighting κ=200 | 0.622 | 0.884 | +0.011 ± 0.001 | 7 / 7 |
| **M2 learned** (switch 10) | 0.611 | 0.885 | — | — |
| M2, switch 5 | 0.586 | 0.884 | −0.025 ± 0.002 | 0 / 7 |
| M1 hmm | 0.534 | 0.814 | −0.077 ± 0.008 | 0 / 7 |
| smooth50 | 0.275 | 0.616 | −0.335 ± 0.008 | 0 / 7 |

Sensitivity: clean 0.679, ball only 0.659, hackathon 0.619, stress 0.593.

Oracle gains: team +0.279, ball-gap frames +0.072, stoppage starts +0.039, short possessions
+0.038, dead/alive +0.033, restarts +0.026, ball always tracked +0.023, timing +0.020 (hand-overs
±1 s: −0.035, an artefact of the window copy with v2-dated changes).

Recall by duration: < 1 s 2%, 1–3 s 32%, 3–10 s 64%, ≥ 10 s 84%. By start: after a hand-over
66%, after a stoppage 70%. Calibration error 0.009–0.010.

Real match (all decoders): public 0.622 for M2 switch 10/20 and M3; rates 4.15–4.32.

Scenarios (learned): pass past opponents 87%, interception 27%, restarts 59% / 63%, long stoppage
97%, hold through a ball gap 46%, change during a ball gap 41%, change during a blackout 25%,
dead ball visible 95%, slow play 75%, kick-off 12%.

Reading: who has the ball (+0.28) is still the dominant loss; short possessions were mostly a DFL
artefact (+0.24 -> +0.04); fewer changes win on the fixed bench, the reverse of DFL's labels.

### Clean vs one degraded draw vs varied draws (Oct 22)

M2 + touches trained on the v2 labels; scored on the fixed bench (3 seeds, switch 10):

| trained on | bench (hackathon profile) | bench (stress) | real rate | public | public possessions (of 17) |
|---|---|---|---|---|---|
| clean data | 0.624 (−0.116 ± 0.007, 0/7) | 0.554 (−0.134, 0/7) | 3.95 | 0.492 | 9 |
| one degraded draw (B_r) | 0.740 | 0.689 | 3.42 | 0.496 | 8 |
| 3 varied draws, spread 0.25 | 0.746 (+0.006 ± 0.004, 4/7) | 0.699 (+0.010 ± 0.005, 5/7) | 3.38 | 0.330 | 6 |

Degrading the training data is essential (clean −0.12). Varying the degradation adds nothing
beyond noise and is worse on the public window: not adopted.

### Ball detection in the tactical video, classical detector (Oct 22)

`possession/video.py` (search window from the ball-filling prior + homography; bright moving
blobs of the expected size; temporal linking). Self-check: hide the tracked ball on frames that
have it, search, compare (aggregates only).

Diagnostic at the tracked ball's pixel (1,500 frames): the ball is ~6 px wide (p10–p90 4.6–8.5);
a bright moving blob is within 7 px of the projected ball (median) and within 40 px 95% of the
time; but its saturation is not low (median 99: blur mixes it with the grass) and its motion vs
t±2 is often weak (slow balls).

Threshold sweep (72 settings, 1,000 hidden balls, first 40,000 frames): best settings find a
candidate on 98.7% of frames, but only 24.7% within 1 m of the ball; median error 2.68 m vs 2.57 m
for the ball-filling prior alone. Not adopted: hand-written rules cannot tell the ball from boots,
socks or line fragments of the same size. (Two bugs fixed on the way: the self-check did not
search the neighbour frames linking needs; linking was quadratic in candidates.)

### PFF World Cup 2022 added (Oct 22)

`bench/pff.py`: 64 matches, broadcast tracking (Sportlogiq) resampled 29.97 -> 25 fps; estimated
players kept, ball visible-only (raw track; the event-smoothed ball would leak); labels from
PFF's `home_ball` per possession sequence. Pooled: 4.67 possessions per live minute, durations
p10/p50/p90 1.6 / 7.0 / 31.6 s (public 4.25, 2.2 / 7.6 / 38.1); ball missing 59%. Degrader
profile "pff": no extra ball gaps (gap-start ball speed p50/p90 2.26/6.1 vs real 1.71/6.45).

M2 + touches trained on DFL (v2 labels) + PFF, held out by group (each DFL match; PFF in 5
folds): models_exp/P. Switch 10:

| | DFL bench (3 seeds) | PFF held-out (64) | PFF rate pred / true | real rate | public | public possessions (of 17) |
|---|---|---|---|---|---|---|
| current submission | 0.611 | – | – | 4.32 | 0.622 | 10 |
| B_r (DFL v2 only) | 0.740 | 0.483 | 3.71 / 4.72 | 3.42 | 0.496 | 8 |
| P (DFL + PFF) | 0.692 | **0.579** (+0.096 ± 0.004, 64/64) | 3.39 / 4.72 | 3.35 | 0.586 | **11** |

PFF reproduces the real-match under-changing (both models predict ~3.4–3.7 changes per live
minute vs 4.72 true), which the DFL bench never showed: the first bench where the real problem
is measurable with real labels. Next: tune P's switch penalty on the PFF folds (predicted rate
= true), then check the real rate and public window.

Switch sweep on the PFF folds (`reports/compare/P_pff_sweep.json`, seed 0):

| switch | PFF spellF1 | PFF rate (true 4.72) | real rate (public 4.25) | public |
|---|---|---|---|---|
| 1 | 0.535 | 4.43 | 3.80 | 0.507 |
| 2 | 0.552 | 4.09 | 3.66 | 0.507 |
| 3 | 0.561 | 3.91 | 3.51 | 0.586 |
| 5 | 0.570 | 3.69 | 3.38 | 0.586 |
| 7 | 0.576 | 3.54 | 3.35 | 0.586 |
| 10 | **0.579** | 3.39 | 3.35 | 0.586 |

Lowering the penalty adds wrong changes faster than it finds missed ones: spellF1 falls at every
step and even switch 1 stays under the true rate. The missed changes are not marginal decoder
calls; the per-frame probabilities do not show them. Switch stays 10.

Licence (Oct 23): the PFF files ship no licence, every page of the specs reads "PFFA Acquisition
LLC proprietary information", and PFF's general Terms of Use (pff.com/terms) forbid use "for
machine learning or training of artificial intelligence models" and derivative works, unless
supplemental terms say otherwise. Until the download form's own terms are checked, P is not a
submission candidate.

### Which possessions are missed (DFL, Oct 23)

`scripts/diagnose_changes.py` (DFL v2 labels, degraded seed 0, held-out models, switch 10): every
true hand-over, by start, the new team's first DFL event within 1 s, duration and ball
visibility, with recall at IoU 0.5 and P(new team) (max of a 1 s running mean).

| | submission | B_r |
|---|---|---|
| recall, all 1,592 hand-overs | 0.67 | 0.73 |
| recall 0-2 s / 2-5 s / 5-15 s / >15 s | 0.19 / 0.46 / 0.75 / 0.85 | 0.24 / 0.54 / 0.80 / 0.91 |
| share of misses under 5 s | 56% | 60% |
| recall turnover, ball missing (173) | 0.51 | 0.47 |
| misses whose P(new team) never reaches 0.5 | 32% | 38% |

Submission misses by cause: wrong team 35%, predicted dead 26%, boundary 16%, merged 13%,
split 11%; 66% of transitions found within 2 s, mean |lag| 13.9 frames.

Two different problems. (1) Short possessions (< 5 s, over half the misses): P(new team) for a
missed one peaks at a median 0.36 (0-2 s) / 0.59 (2-5 s), so the model barely sees them; and a
switch penalty that charges per change suppresses them even when it does. (2) Longer missed
possessions are seen (median peak 0.86-0.98) but lost to the wrong team at the start, a dead
label over live play, or boundaries. Worst contexts: turnovers with the ball missing, deflections
/ clearances (OtherBallAction, recall 0.53), throw-ins (19% of misses), and turnovers with no
event of the new team within 1 s (30% of misses: contested balls, where the label rule itself
is least certain).

### Cheaper switches at detected touches (Oct 23): rejected

`learned.Params.switch_touch` (off by default): the switch into a team costs `switch_touch`
instead of `switch` within `touch_window_s` of one of that team's detected touches.
`scripts/eval_touch_switch.py`, submission model, DFL v2 labels, 3 seeds, 57 settings
(switch 10/14/20 x touch cost 1/3/5 x window 0.3/0.6/1.0 s x controls only / all touches):

| setting | DFL spellF1 | rate pred / true | recall < 5 s | real rate | public |
|---|---|---|---|---|---|
| switch 10 (submission) | 0.611 | 5.30 / 4.38 | 0.40 | 4.32 | 0.622 |
| switch 20 | **0.641** (+0.030 ± 0.003, 7/7) | 4.81 / 4.38 | 0.38 | 4.16 | 0.622 |
| switch 20, touch 5, 0.6 s, controls | 0.635 | 4.98 / 4.38 | 0.39 | 4.17 | 0.622 |
| switch 20, touch 1, 0.3 s, controls | 0.631 | 5.05 / 4.38 | 0.39 | 4.28 | 0.622 |

Every discount is worse than the same constant switch, and all-touch discounts never reach the
top 15: the extra changes are mostly wrong, and short-possession recall moves by 0.01-0.02. The
detected touches do not mark the short possessions precisely enough (on degraded tracking the
detector finds ~61% of real hand-overs, and contacts fire in duels). On this bench the submission
model over-changes (5.30 vs 4.38), the opposite of P on PFF, and switch 20 is +0.03 better; the
real-match rate at switch 20 (4.16) stays near the public 4.25.

## Round 3 (Oct 27): kick-off, model and switch, ensembles, restarts, learned video ball

All bench numbers: DFL v2 hackathon-style labels, realistic degrader, 3 seeds, held-out models.
Adoption rule: paired per-match comparison read at the real match's change rate (each model at
the switch whose real rate is closest to the public 4.25); the public window is a sanity check.

### Second half before kick-off (item 6)

The real second half is graded from 110,808 but kicks off at 113,742 (FAQ.md): 2,934 frames,
all D. The decoder starts at 110,808 and the bench never shows this (DFL halves start at their
kick-off). `scripts/prekickoff.py` (real match, aggregates):

| model | H/A frames before kick-off | false possessions there |
|---|---|---|
| submission (M2 + touches), before | 1,059 | 6 |
| A_r, before | 325 | 2 |
| submission, with the kick-off rule | 27 | 1* |
| A_r, with the kick-off rule | 13 | 1* |

*the first real possession starting under a second early, not a false one.

Rule (`restarts.kickoff_frames`, `learned.Params.kickoff_dead`, on by default): a half that does
not open inside a kick-off line-up is dead until its kick-off, found as the last early line-up
(both teams in their own halves, walking pace, >= 2 s; no ball needed) or kick-off restart in
its first 3 minutes, reached through stretches that are dead for the model or at walking pace
(15 s windows of live play average walking pace 3% of the time on DFL). The first half is graded
from its kick-off and opens inside the line-up: untouched. Bench scores are unchanged; the
submission's real rate goes 4.32 -> 4.28.

### Restart detector (item 11, direction 1)

`possession/restarts.py`: a visible ball standing still >= 1.5 s that is then kicked; the
position where it starts moving gives the kind, the nearest player then gives the team. The 6
direction-1 specs pass. Checked against the 681 DFL official set pieces
(`scripts/validate_restarts.py`):

| | clean | hackathon-like |
|---|---|---|
| set pieces found (within 1 s) | 84% | 37% |
| team right (when named) | 94% | 14/15 checked by hand on one match; named on ~50% |
| kind right | 91% | 74% |
| detections that are set pieces | 83% | 70% |
| kick-off line-ups found (within 3 s) | 58% | 39% |

DFL parks a dead ball where it left the pitch and jumps it to the restart spot: 57% of DFL's
still-ball stretches lie outside the lines vs 11.5% on the real tracking
(`scripts/restart_realism.py`), so parked stretches are skipped and the restart is read where
the ball starts moving. On the real match the detector finds 0.29 restarts per graded minute
(team unknown on 29%), and a restart in 34% of the tagged set pieces, close to its 37% recall on
the degraded bench.

R_r (A_r's recipe + 6 restart features, `learned train --features restart`):

| | bench spellF1 at matched real rate | real rate | public |
|---|---|---|---|
| A_r, switch 7 | 0.637 | 4.24 | 0.622 |
| R_r, switch 7 | 0.644 (+0.007 ± 0.003 vs A_r, 6/7) | 4.22 | 0.622 |

Small but consistent. Restart scenarios vs the submission: restart_same 59 -> 64%, restart_other
63 -> 68%; kick-off stays 12% (8 clips, mostly kick-offs after goals with the ball missing).

Smart Tagging SET PIECES (`scripts/tag_restarts.py`): the tags are fixed 35 s windows; the
model's restart sits +5.7 s after the tag start (quartiles +3.7 / +8.6 s), and 53% have no team.
Too vague to date or attribute a restart: not used.

### Switch and model (item 7)

With the kick-off rule (`reports/compare/{models,A_r}_k.json`):

| | switch | bench spellF1 | real rate | public |
|---|---|---|---|---|
| submission (M2 + touches) | 10 | 0.611 | 4.28 | 0.622 |
| A_r (same features, realistic bench) | 7 | 0.637 (+0.026 ± 0.006, 7/7) | 4.24 | 0.622 |

### Learned ball detection in the video (item 12): NO-GO

`possession/ballpatch.py`: candidates = bright compact peaks (white top-hat) ranked by
brightness + motion in the ball-filling search window; LightGBM on 11×11 patches (grey,
saturation, brightness, motion vs t±2) + scalars, trained on THIS video with the tracked ball as
free labels (1st half), tested on 1,000 hidden tracked balls of the 2nd half (~1 s gaps).

| | within 1 m | median error |
|---|---|---|
| ball-filling prior alone | – | 1.96 m |
| classical detector (Oct 22) | 24.7% | 2.68 m |
| **learned patches** | 42.6% (most confident half: 56%) | **1.23 m** |
| gate | >= 60% | below the prior |

Better than the classical detector and than the prior, but below the gate, and the test gaps
(~1 s) are far easier than the real ones (median 10.7 s). The ceiling is the candidate stage:
the ball is among the top 150 candidates on only ~52% of frames (`ballpatch diagnose`: the
projected tracked ball sits 7 px median from its image peak, windows hold > 400 bright peaks).
With the bench's ceiling for a perfectly tracked ball at +0.02, not pursued; a deep detector
would be the next step.

### Ensembles (item 8) and the new submission

`models_exp/ens_*/ensemble.json`: weighted geometric mean of member models' probabilities
(`learned.members`, `learned.probabilities`); each member stays held-out on its own training
matches. B_r (hackathon-style labels) is better calibrated on the bench but under-changes on the
real match; A_r / R_r (DFL labels) over-change on the bench. Mixing them, then picking the switch
whose real rate is closest to 4.25:

| model | switch | bench spellF1 | real rate | public | vs submission (paired, 7 matches) |
|---|---|---|---|---|---|
| submission (M2 + touches) | 10 | 0.611 | 4.28 | 0.622 | – |
| A_r | 7 | 0.637 | 4.24 | 0.622 | +0.026 ± 0.006, 7/7 |
| R_r (+ restarts) | 7 | 0.644 | 4.22 | 0.622 | +0.033 ± 0.007, 7/7 |
| A_r + B_r 50/50 | 1 | 0.668 | 4.24 | 0.593 | +0.057 ± 0.008, 7/7 |
| A_r + B_r 75/25 | 3 | 0.665 | 4.18 | 0.622 | +0.054 ± 0.008, 7/7 |
| **R_r + B_r 75/25 (adopted)** | **3** | **0.667** | **4.18** | **0.622** | **+0.056 ± 0.007, 7/7** |

R_r + B_r vs R_r alone: +0.023 ± 0.003 (7/7); vs A_r + B_r 75/25: +0.002 (noise). The 50/50 mix
needs switch 1 (the grid's edge) to reach the real rate and loses public points: 75/25 kept.

Adopted (Oct 27): `models/ensemble.json` = models/R (R_r, DFL labels, restart features) × 0.75 +
models/B (B_r, hackathon-style labels, touch features) × 0.25; switch 3; kick-off rule on.
Real match: 220 possessions (mean 14.4 s; public GT 14.2 s), rate 4.18, public spellF1 0.622,
public possessions 10 / 17 (unchanged), public macroF1 0.861 -> 0.839 (17 possessions; bench
macroF1 0.885 -> 0.907). Byte-identical across runs; validator passes.

Caveat: B is trained on the same hackathon-style label rule the bench scores with, so part of
any B-family bench gain is circular. The guard is the real-match change rate, at which every
row above is read.

Gates (`pytest -m slow`) now score the v2 labels, the bench yardstick since Oct 22; floors were
re-based on the new model, which is at or above the old submission on every one of them;
ball_gap_hold (48%) and change_in_ball_gap (44%) reached their direction-3 targets.

### Adversarial validation finds a data problem (Oct 27)

`scripts/domain_shift.py` (degraded DFL vs real frames, 54 features, grouped 5-fold): AUC 0.998.
77% of the classifier's gain is `ball_last_speed`. The ATD's third ball value is not a speed:
correlation 0.06 with the speed from positions; median 1.1, p10 0.23, p99 5.3, never 0 (most
likely the ball's height in m). The bench derives a real speed. Players' speeds agree on both.

| ball speed on the real match | public-window dead frames called dead (of 1,469) | macroF1 | real rate |
|---|---|---|---|
| file's value (current; = height) | 1,026 | 0.839 | 4.18 |
| derived from positions (consistent with the bench) | 475 | ~0.71 | 4.38 |

The consistent value makes the current models miss dead balls: DFL parks a dead ball motionless,
while the real one is carried and thrown back, so the bench taught "still ball = dead". Reader
kept as validated (`bench/epts.py`, comment) until the degrader moves the dead ball and the
models are retrained (tasks/todo.md, F0). Other shifted features: the team-sign features
(Barcelona keeps the ball far more than a typical DFL home team: expected, not a bug).

### PFF evaluation-only bench, and the switch chosen on the benches (Oct 27)

Review: the public window's 4.25 changes per live minute comes from 17 possessions (exact Poisson
95% interval 9.9-27.2 over 4.02 live minutes: 2.5-6.8 per live minute), and one possession moves
its spellF1 by ~0.03-0.07. Matching "closest to 4.25" fitted noise. New rule: the benches decide,
the real rate is a range check, the public window decides nothing.

PFF World Cup 2022 as an evaluation-only bench (`scripts/pff_eval.py`; no model trains on it:
licence): 64 matches, PFF's own possession labels (4.72 per live minute), seed 0.

| at the switch chosen by the old rule | switch | PFF spellF1 | PFF rate |
|---|---|---|---|
| old submission (M2 + touches) | 10 | 0.479 | 4.50 |
| R_r | 7 | 0.483 | 4.75 |
| R_r + B_r 75/25 | 3 | 0.468 | 5.04 |

The ensemble looked worse than the old model on PFF only because of switch 3: at the same switch
10 it is +0.024 ± 0.003 (57/64). Switch sweep of the ensemble:

| switch | DFL bench spellF1 | bench rate (true 4.38) | PFF spellF1 | PFF rate (true 4.72) | real rate | public |
|---|---|---|---|---|---|---|
| 3 | 0.667 | 5.23 | 0.468 | 5.04 | 4.18 | 0.622 |
| 10 | 0.703 | 4.72 | 0.503 | 4.07 | 3.65 | 0.622 |
| **20 (adopted)** | **0.724** | **4.35** | **0.514** | 3.61 | **3.45** | 0.622 |
| 30 | 0.734 | 4.09 | 0.514 | 3.36 | 3.29 | 0.480 |
| 40 | 0.734 | 3.91 | 0.510 | 3.20 | 3.25 | 0.480 |

Switch 20: PFF's peak (flat 20-30), the DFL bench within 0.01 of its peak with its predicted rate
equal to its true one, real rate inside the range. vs switch 3: +0.057 bench, +0.047 PFF.
Submission: 181 possessions, mean 17.4 s (public GT 14.2 s), byte-identical, validated.

Trade-off (gates, seed 0): hold through a ball gap 48 -> 54%, restarts 66/68 -> 69/69%, pass
past opponents 88 -> 90%; change in a ball gap 44 -> 38% and blackout change 25 -> 21%, both now
below the old submission (41%, 25%): moved back to targets in `tests/test_targets.py`.

Both benches under-change at their best switch on PFF (3.61 vs 4.72) and the real match sits at
3.45: the models still miss hand-overs; a higher switch only stops the wrong ones.

### Metrica sample games as extra training data (Oct 27): rejected

`learned train --with-metrica`: Metrica sample games 1-3 added to both members (B_m: relabelled
with the hackathon-style rule, 3.5-3.7 changes per live minute, `external/metrica_hk`; R_m: their
event-derived labels). Same tracker family as the hackathon data, so the hope was transfer.

| vs the submission (R_r + B_r 75/25), paired | DFL bench, switch 20 (7 matches) | PFF, switch 20 (64 matches) |
|---|---|---|
| R_r + B_m | −0.003 ± 0.001 (1/7) | – |
| R_m + B_m | +0.000 ± 0.002 (5/7) | −0.006 ± 0.002 (19/64) |

No gain on either bench. Metrica's labels come from events (approximate), and its clean ball is
missing almost exactly when the ball is dead, which the degrader only partly hides: the extra
matches add noise more than information.

### F0: the ball speed fixed, the dead ball made to move (Oct 28)

Root cause, measured more closely: the ATD gives a third ball value on only 7.8% of visible-ball
frames (29 stretches, none in the public window), and it is not a speed (correlation 0.06; not a
height either: a resting ball reads 1.1). Trusting it left every ball-speed feature MISSING on
92% of real frames, where the bench always has a speed. Now `bench/epts.py` always derives it.

That exposed a second bench artefact: DFL parks a dead ball where it left the pitch. Dead frames
with a visible ball (`scripts/dead_ball_realism.py`):

| | speed p10/50/90 (m/s) | still | nearest player | outside the lines |
|---|---|---|---|---|
| real (public window; 4 stoppages) | 1.0 / 2.7 / 9.3 | 1% | 2.9 m | 11% |
| DFL bench before | 0.2 / 0.5 / 0.9 | 22% | 4-5 m | 52-73% |
| **DFL bench now** (`Profile.dead_ball_moves`) | 0.7 / 2.9 / 8.4 | 2% | 2.3 m | 7% |
| PFF (pff profile) | 0.3 / 1.0 / 4.3 | 10% | 2.6 m | 14% (dead ball seen 10%) |

The degrader now moves the dead ball: hops between nearby players (thrown / kicked back), then
the restart taker carries it to the spot (`_dead_ball_motion`, tested). Models trained on the old
bench read "still ball = dead" and break on a moving dead ball.

| switch 20 | corrected DFL bench | PFF | real public (no decision) | real rate |
|---|---|---|---|---|
| previous ensemble (R_r + B_r) | 0.567 | 0.514 | 0.444, macro 0.709* | 3.71* |
| F0 pure (moving dead ball only) | 0.683 | 0.496 | 0.667, macro 0.865 | 3.54 |
| **F0 mixed (adopted)**: 2 draws per match, dead ball moving / parked (`--dead-mix`) | **0.683** | **0.509** | 0.622, macro **0.871** | 3.56 |

*with the derived ball speed; with the old missing speed it was 0.622 / 0.839 / 3.45.
Mixed vs previous, paired: corrected bench +0.117 ± 0.006 (7/7); PFF −0.005 ± 0.002 (22/64;
PFF's dead ball is 90% invisible, so the "still = dead" shortcut still pays there). Mixed vs pure:
bench +0.000, PFF +0.014 (58/64). Switch 20 kept: 20 vs 30 within noise on both benches.
Domain classifier AUC 0.998 -> 0.994; `ball_last_speed` 77% -> 28% of its gain; the rest is
team shape (Barcelona's high line) and team-sign features (Barcelona keeps the ball).

Submission: 191 possessions, mean 16.8 s; byte-identical; validated. Gates re-based on the
corrected bench (`tests/test_targets.py`); regressions vs the previous model there: slow play
74% vs 81%, hold through a ball gap 52% vs 56%.

### F3: adversarial validation near changes -> the ball-gap split was wrong (Oct 28)

`scripts/domain_shift.py --near-changes` (frames within 2 s of an H <-> A change; real: the
public window's 12 changes, or the submission's 90 predicted ones). Most shifted features there:
ball visibility, then control / touch evidence. Measured directly:

| near possession changes | bench | real (public, 12) | real (predicted, 90) |
|---|---|---|---|
| ball seen | 77% | 96% | 93% |
| control detected (share of frames) | 21% | 13% | 17% |
| new team's control within 1 s of the change | 36% | 17% | 30% |
| any touch by the new team within 1 s | 57% | 58% | 60% |

The real tracker SEES the ball at hand-overs; the bench hid it. Cause: the Oct 20 live / dead
split of missing ball (19% / 61%, from a speed-bin proxy) was wrong. Real: ~7% of live frames
(public window 6.5%; frames the model calls live 7%, confident ones 4%) and ~75-80% of dead ones.
Gap placement weights do not change it (gaps are long); what does: gaps that start in a stoppage
end at the restart (+ up to 1 s), applied per gap (`Profile.gap_ends_at_restart`, tested).

| | real | bench before | bench now |
|---|---|---|---|
| ball missing overall / live / dead | 41% / ~7% / ~75-80% | 41% / 19% / 61% | 40% / 7% / 80% |
| ball seen near changes | 93-96% | 77% | 91% |
| gap length median / p90 (not fitted) | 10.7 / 34 s | – | 11.3 / 35 s |

Retrained (dead-ball mix), switch 20:

| | recalibrated bench | PFF | real rate | public |
|---|---|---|---|---|
| previous submission (R_v + B_v) | 0.710 | 0.509 | 3.56 | 0.622 |
| **R_g + B_g (adopted)** | **0.725** (+0.015 ± 0.004, 6/7) | **0.537** (+0.028 ± 0.003, 61/64) | 3.69 | 0.622 |
| Oct 27 model (R_r + B_r), on PFF | – | 0.514 (new: +0.023 ± 0.003, 56/64) | – | – |

Submission: 198 possessions, mean 16.2 s; byte-identical; validated. Gates re-based; recorded
regressions: dead ball visible 63% vs 80% (79 clips), ball-gap hold 27% vs 35% (26 clips).
The smaller control-evidence gap (new-team control within 1 s: 17-30% real vs 36% bench) is
second-order: the ball moves with the nearest player on 55-59% of close frames on the real
tracking vs 65% on the bench.

### F4: self-training on the real match (Oct 30): rejected

`learned train --self-train 2 --self-train-mode control|conservative` (R member; B = B_g): the
real match's frames with algorithmic pseudo-labels (no human label; public window excluded), the
ball hidden in gap shapes, added to every fit with weight 2. control: the hackathon-style rule
on the submission's scaffold (27k live frames, 4.0% disagree with the submission);
conservative: the submission's confident labels >= 2 s from any change (43k live frames).

| vs the submission, switch 20 | bench (7 matches) | PFF (64) | real rate |
|---|---|---|---|
| control | +0.001 ± 0.001 | −0.004 ± 0.002 (29/64) | 3.66 |
| conservative | +0.001 ± 0.002 | −0.005 ± 0.002 (22/64) | 3.61 |

Neutral on the bench, slightly worse on PFF at every switch. After F0 / F3 the bench and the real
tracking agree closely, so the real match's own frames add no new information and the model
leans slightly towards this fixture. Code kept (`possession/selftrain.py`), off by default.

### C: minimum Bayes risk decoding (Oct 30): rejected

`scripts/eval_mbr.py bench`, 7 matches x 3 seeds, vs Viterbi at switch 20 (0.7247): temp 0.5
+0.0002 ± 0.0002 (1/7), 1 −0.0004, 2 −0.0068, 4 −0.0226, 8 −0.0442. The tempered posterior
spreads mass over more changes, so MBR picks lower switches; at low temperature it is Viterbi.

### D: learned switch costs (Oct 30): grid edge, re-run higher

`possession/changes.py` (LightGBM "change within ±0.5 s", leave-one-match-out; base rate 0.022),
costs scaled by beta. Bench vs beta 0 / switch 20 (0.7246): switch 14 −0.012 for every beta;
switch 20 −0.001 to −0.005; switch 30 +0.0065 (beta 0, 7/7) to +0.0102 (beta 4, 7/7). Two
findings: the submission's switch is now too low on the bench (F3's retraining moved it), and D
adds +0.004 at switch 30. Both at the edge of the grid: re-run at 30-80 after the F8 retraining.
### Dead ball until the first reception (Oct 30): built, off by default

In the hackathon ground truth a restart stays `D` until another player first receives the ball.
That is a definition, read from the public window as CHALLENGE.md says to (and agreed with
Julien); none of the constants below come from it. DFL's BallStatus goes live at the kick. On
clean DFL the ball passes 3 m/s a median 3 frames after the switch, for every kind of restart,
and the first reception by another player comes later:

| restart | n | first reception after DFL's switch, p25 / p50 / p75 (frames) |
|---|---|---|
| corner | 51 | 27 / 31 / 36 |
| free kick | 161 | 22 / 33 / 45 |
| throw-in | 206 | 20 / 37 / 70 |
| kick-off | 31 | 24 / 37 / 58 |
| goal kick | 107 | 38 / 50 / 62 |

So every model trained on DFL learns to end stoppages 1-2 s early. `bench/reception.py` has one
rule for this: from each restart, the taker is the player nearest the ball, and `D` lasts until
another player is the nearest one within a radius, two frames running (no ball at the restart, or
no reception in time: unchanged). It serves twice:

- scoring labels: on clean DFL, 1.5 m, 6 s, DFL possession elsewhere (`python -m bench reception`,
  then `MTAH_DFL_LABELS=reception`). 90-111 restarts extended per match, median 24-33 frames; dead
  share +1.9 to +2.5 points.
- decoder: `learned.Params(reception_m=2.0, reception_s=3.0)` applies it to the predicted labels,
  on the tracking the method sees (off by default). 2 m because that tracking is noisy; 3 s is
  about the 90th percentile of DFL's reception delay.

Bench bug found on the way: the degrader hid the ball by the labels' `D`, so with these labels the
ball in flight after a kick went missing at the dead-ball rate (61% instead of 19%). The tracker
sees the ball, not the convention: `Match.ball_dead` now carries DFL's BallStatus in the
reception files, the degrader uses it, and `without_labels()` drops it.

On the fixed bench, scored on the reception labels, 3 seeds, switch 10. S0 is the submission
recipe retrained with today's code; R2 is the same recipe trained on the reception labels.

| | spellF1 | macroF1 | vs S0 (paired) |
|---|---|---|---|
| S0 | 0.491 | 0.877 | – |
| **S0 + reception in the decoder** | **0.507** | **0.886** | **+0.016 ± 0.002, 7/7** |
| R2 (trained on the reception labels) | 0.503 | 0.883 | +0.012 (−0.004 ± 0.002 vs the decoder rule, 2/7) |

| scenario (3 seeds) | n | S0 | S0 + reception | R2 |
|---|---|---|---|---|
| hold through a ball gap | 604 | 23.2% | 23.2% | 19.4% |
| change in a ball gap | 1099 | 25.1% | 25.1% | 20.8% |
| change during a blackout | 510 | 18.8% | 18.8% | 14.7% |
| restart, same team | 354 | 63.6% | 64.1% | 60.2% |
| restart, other team | 1050 | 59.0% | 60.2% | 53.0% |
| dead ball, ball visible | 533 | 96.6% | 99.1% | 97.7% |
| slow live play | 348 | 88.2% | 87.9% | 86.5% |
| interception | 2196 | 26.9% | 26.9% | 26.6% |
| kick-off | 24 | 12.5% | 12.5% | 12.5% |

The decoder rule only acts right after a predicted restart, so ball-gap play is untouched; slow
play loses one clip of 348. Training on the new labels instead makes the model see dead ball in
live ball gaps (the extended frames look like a pass in flight), so R2 is rejected. A first run,
before the degrader fix, had shown R +0.002 and the ball-gap losses; both came mostly from the bug.

Scored on DFL's own labels instead (live at the kick), the same rule costs −0.024 ± 0.002 (0/7;
restarts 69.7% -> 65.6% and 64.9% -> 61.0%). The bench cannot settle it: the sign depends only on
which convention the hackathon ground truth follows, and the evidence is mixed:

- for "until the reception": Julien's reading of the ground truth, and a teammate's (throw-ins);
- against, or for a shorter delay: in Metrica's open Sample Game 3 (same provider and format as
  the hackathon tracking) the ball reappears exactly at the restart kick on 74 of 74 restarts,
  22-40 frames before the reception (not the corrected tracking's dead flag, so not conclusive).

Real match, submission model (`models/lgbm.txt`), aggregates: dead share of graded frames 48.6% ->
49.9%, possessions per live minute 4.32 -> 4.43. Kept off until the convention is settled.

S0 is not `models/lgbm.txt`: the submission model was trained on Oct 21, before the Oct 22
degrader changes, so `python -m possession.learned train` no longer rebuilds it.

### Frozen or glued ball in the real tracking (Oct 30)

Does the real tracker report a ball that is frozen, or stuck to one player, more often than the
bench? Aggregates over the whole real match against degraded DFL (`degrader_realism`-style):

| | real | DFL degraded [range] | DFL clean |
|---|---|---|---|
| ball still for >= 1 s while players run (> 2 m/s), per min | 0.019 | 0.007 [0.000-0.020] | 0.256 |
| same, nobody within 3 m, per min | 0.009 | 0.001 [0.000-0.010] | 0.171 |
| same player within 1 m: run p90 / p99 (s) | 1.7 / 4.1 | 1.8 / 4.5 | 1.7 / 5.0 |
| same player within 1 m for >= 6 s, per min | 0.038 | 0.066 | 0.148 |
| same player within 0.3 m for >= 2 s, per min | 0.038 | 0.020 | 0.139 |

Both happen about 1-3 times per match, at rates close to the bench. The real ball sits within
0.3 m of a player more often than the degraded one (5.7% vs 3.0% of seen frames), but the clean
DFL ball does too (3.9%): the degrader's position noise blurs it, and runs are no longer. Not worth
a degrader artefact or a consistency rule.
