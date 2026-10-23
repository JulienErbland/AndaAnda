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
