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

### Round 8: reception convention and switch 20-60 (Oct 30)

Scored on hackathon-rx labels (the hackathon-style rule + "dead until the first reception",
`bench reception --source hackathon`), OLD bench (before F5). Match = unit (3 seeds averaged),
`scripts/curve.py`. Real rate: range 3.2-7.6 (Bench caveats). PFF: reported only, PFF's own
labels (live at the kick), 64 matches, bootstrap 95% vs the submission at switch 20 (0.5370).

| | switch 20 | 30 | 40 | 60 |
|---|---|---|---|---|
| submission (R_g + B_g), bench | 0.6730 | 0.6796 (+0.007, 7/7) | 0.6840 (+0.011, 7/7) | 0.6782 (+0.005, 4/7) |
| + decoder rule (2 m) | 0.6858 (+0.013, 7/7) | 0.6925 (+0.020, 7/7) | 0.6964 (+0.023, 7/7) | 0.6904 |
| **R_rx + B_rx (trained on reception labels)** | 0.6959 (+0.023, 7/7) | **0.6997 (+0.027, 7/7)** | 0.6970 (+0.024, 7/7) | 0.6933 (+0.020, 7/7) |
| R_rx + B_rx + decoder rule | 0.6829 | 0.6880 (−0.012 vs without, 0/7) | 0.6858 | 0.6826 |
| real rate: submission / R_rx + B_rx | 3.69 / 3.60 | 3.49 / 3.45 | 3.32 / 3.29 | 3.23 / 3.23 |
| PFF: submission | 0.5370 | 0.5381 [−0.001, +0.003] | 0.5362 [−0.004, +0.002] | 0.5285 [−0.012, −0.005] |
| PFF: R_rx + B_rx | 0.5220 [−0.018, −0.012] | 0.5209 | 0.5176 | 0.5093 |

Per match (bench, R_rx + B_rx, switch 20 / 30 / 40): J03WMX .673/.678/.676, J03WN1 .615/.612/.612,
J03WOH .713/.714/.713, J03WOY .685/.690/.687, J03WPY .738/.737/.733, J03WQQ .747/.756/.750,
J03WR9 .700/.711/.708. Flat 20-40 within 0.004 on the mean; 60 is worse on every curve and on PFF.

Reading: on reception labels, training on them beats the decoder rule (opposite of PR #1's result
with the older model); stacking both double-counts. The switch curve is flat over 20-40 on both
benches; its middle is 30. Best-of-4 on 7 matches is optimistic by about the neighbours' spread
(~0.003). On PFF the reception models lose 0.015-0.028: PFF's labels go live at the kick, so this
measures the convention, not the model. Public check (reported only): 10/17 possessions at switch
20, 8/17 from 30 on (one possession pair merged; not an alignment bug: same at 30, 40, 60 and for
every model). Which convention the hackathon GT follows is decided in round 9 (minimax rule).

### Round 9: iteration 0 on the F5 bench (Oct 31)

All models retrained on the F5 bench (`scripts/round9_clean.sh`); decision rules in
`tasks/todo.md` ("Decision rules for L1 and the convention"), written Oct 30 22:35 before scoring.
Numbers come from the DFL bench only (match = unit, 3 seeds averaged, `scripts/curve.py`); PFF,
the real rate and the public window are reported only.

| bench spellF1 | yardstick | switch 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|
| submission (old models) | Y_v2 | 0.7409 | 0.7501 | 0.7513 | 0.7432 |
| K0 = R_f + B_f (v2), 75/25 | Y_v2 | 0.7439 | **0.7508** | 0.7500 | 0.7409 |
| K0 | Y_v1 | 0.7190 | 0.7278 | 0.7281 | 0.7211 |
| K0 | Y_rx | 0.6966 | 0.7036 | 0.7038 | 0.6952 |
| K1 = R_f + B1_f (v1), 75/25 | Y_v2 | 0.7435 | 0.7478 | 0.7500 | 0.7394 |
| K1 | Y_v1 | 0.7392 | 0.7439 | **0.7451** | 0.7349 |
| K2 = R_rxf + B_rxf (reception) | Y_v2 | 0.7049 | 0.7112 | 0.7088 | 0.7039 |
| K2 | Y_rx | 0.7108 | **0.7159** | 0.7131 | 0.7070 |
| K0h = K0 at 50/50 | Y_v2 | 0.7668 | **0.7698** | 0.7679 | 0.7578 |
| real rate K0 / K1 / K2 | | 3.74 / 3.74 / 4.00 | 3.64 / 3.59 / 3.79 | 3.59 / 3.48 / 3.61 | 3.39 / 3.31 / 3.49 |
| PFF K0 / K1 / K2 (75/25) | | 0.5384 / 0.5349 / 0.5225 | 0.5386 / 0.5355 / 0.5221 | 0.5360 / 0.5317 / 0.5178 | 0.5269 / 0.5239 / 0.5088 |

Decisions (switch = each candidate's best on its own yardstick; every curve is flat over 30-40):
- **F6 (L1): K1 wins.** Max regret K0 0.0172 (on Y_v1, K1 better 7/7), K1 0.0008 (on Y_v2,
  2/7, within noise). Gap 0.016 > 0.005, so B is trained on v1 labels (dated at the control start,
  no first-touch dating learned from the public window). Same verdict with both at switch 30.
- **Convention: kick (K0) wins.** Max regret K0 0.0124 (on Y_rx, K2 better 6/7), K2 0.0395 (on
  Y_v2, K0 better 7/7). Not a tie, so no question for Julien; the F7 draft's Q2 stays open.
- **L3: 50/50 replaces 75/25.** K0h vs K0 at switch 30: +0.0190, 7/7 (rule: >= +0.005, >= 6/7).
- The F5 bench itself moved the submission from 0.725 to 0.741 (switch 20, same models): the
  degrader change, not a model change. Retraining on it changes nothing (K0 vs submission at 30:
  −0.0007, 5/7).
- Switch: 30 (middle of the flat 30-40 on K0, K0h, K1). Best-of-4 optimism ~0.002 (neighbours).

**Round 9b: both adopted changes together (K1h = R_f + B1_f, 50/50), `scripts/round9b.sh`.**

| switch 30 | Y_v2 | Y_v1 | real rate | PFF (vs submission 0.5370, bootstrap 95%) | public (reported) |
|---|---|---|---|---|---|
| K0 (75/25, v2) | 0.7508 | 0.7278 | 3.64 | 0.5386 (+0.002 [−0.002, +0.006]) | 8/17 |
| K1 (75/25, v1) | 0.7478 | 0.7439 | 3.59 | 0.5355 (−0.002 [−0.006, +0.003]) | 8/17 |
| K0h (50/50, v2) | 0.7698 | 0.7450 | 3.50 | 0.5307 (−0.006 [−0.011, −0.002]) | 7/17 |
| **K1h (50/50, v1)** | 0.7588 | **0.7734** | 3.44 | 0.5221 (−0.015 [−0.019, −0.011], 9/64) | 8/17 |

At 50/50 the F6 minimax still picks v1: max regret K0h 0.028 (Y_v1, 7/7), K1h 0.011 (Y_v2, 1/7).
K1h vs the submission at its own switch 20 on Y_v2: +0.018 (6/7).

**Iteration 0 fails its ±0.01 check, so it stops here.** Bench 0.759 (Y_v2) vs 0.725: +0.034, of
which +0.016 is the F5 bench change (same models). PFF 0.522 vs 0.537: −0.015, outside the band
and its interval entirely below 0. Both adopted changes lose on PFF (50/50 −0.008, v1 −0.009 at 50/50).
PFF's labels are PFF's own definition (live at the kick, its own possession dating), so this is
either a definition mismatch or real overfitting to the hackathon-style rule. With PFF reported
only, the bench can't tell which. **Julien accepted K1h as the baseline (Oct 31), with the PFF drop
as a caveat.** Installed in models/ (switch 30); `possession.submit` is byte-identical over two
runs and the validator passes; real rate 3.44, public 8/17 (reported only).

**Learning curve (K1h, switch 30, `scripts/learning_curve.sh`, `--train-k`).** Both members fit on
k DFL matches. Each held-out model uses the k matches after its own, cyclically. k = 6 is K1h's
own held-out models; k = 7 is not possible held-out.

| k | 1 | 2 | 3 | 5 | 6 |
|---|---|---|---|---|---|
| bench (Y_v2) | 0.7228 | 0.7463 | 0.7489 | 0.7581 | 0.7588 |
| PFF (reported only) | 0.4932 | 0.5035 | 0.5187 | 0.5210 | 0.5221 |

EXTRAPOLATION (fit a + b log k, not measured): the bench gains ~+0.015-0.019 per doubling of
training matches (k = 12: ~0.770-0.775). PFF fits ~+0.017 per log unit over all points, but only
~+0.005 over k = 3-6 (k = 12: ~0.525-0.536). The bench is still rising and PFF is flattening. More
matches with the same labels would help the bench more than PFF.

### Iteration 1: D (learned switch costs) on the baseline (Oct 31): FAIL, not adopted

Plan written before scoring (`tasks/todo.md`). The change model was retrained on the F5 bench with v1
labels (`models_exp/changes_f`); evidence = the baseline; switch 30; beta in {0, 1, 2, 4, 8};
`scripts/iter1_D.sh`. Numbers: DFL bench (decides), PFF / real rate / public reported only.

| beta | bench Y_v2 (vs 0.7588) | Y_v1 (reported; D's own labels) | PFF (vs 0.5221, bootstrap 95%) | real rate |
|---|---|---|---|---|
| 1 (selected: middle of the flat 0-2) | 0.7580, −0.0008, 3/7 | +0.0024, 5/7 | +0.0001 [−0.0005, +0.0008] | 3.46 |
| 2 | 0.7595, +0.0007, 4/7 | +0.0060, 7/7 | −0.0008 [−0.0018, +0.0003] | 3.46 |
| 4 | 0.7573, −0.0015, 2/7 | +0.0086, 6/7 | −0.0025 [−0.0038, −0.0012] | 3.52 |
| 8 | 0.7498, −0.0091, 0/7 | +0.0059, 6/7 | −0.0071 [−0.0093, −0.0050] | 3.61 |

Grades: mine PARTIAL (a fails, b, c, d pass); separate grader FAIL (a fails; d not shown for a D
submission, which isn't wired in). Lower grade = **FAIL**. The old-bench +0.004 was measured
against the Oct 28 submission's evidence at switch 30 on the pre-F5 bench. With the retrained 50/50
evidence it is gone on Y_v2, and it survives only on the change model's own labels (circular).
D doesn't fix the undercount (bench rate 3.99 vs true 4.38).

### Iteration 2: temporal CNN evidence (Oct 31): FAIL, not adopted

`possession/tcn.py`: a dilated 1-D CNN (Bai et al. 2018, arxiv.org/abs/1803.01271) over the
per-frame feature sequence at 5 fps (~25 s receptive field), PyTorch 2.15 CPU, a fixed budget (1500
steps), seeded and deterministic (same seed -> identical weights). Two members mirroring the
baseline (T_R on DFL labels, T_B1 on v1), 50/50, same decoder; `scripts/iter2_tcn.sh`. Plan in
`tasks/todo.md`, written before scoring.

| switch | bench Y_v2 (baseline 0.7588) | Y_v1 (reported) | PFF vs 0.5221 (reported, 95%) | real rate | public (reported) |
|---|---|---|---|---|---|
| 20 | 0.7471 (−0.012, 1/7) | 0.7616 | −0.0046 [−0.011, +0.002] | 3.87 | 11/17 |
| 30 | 0.7544 (−0.004, 3/7) | 0.7686 | −0.0052 [−0.011, +0.001] | 3.77 | 11/17 |
| 40 | 0.7565 (−0.002, 3/7) | 0.7705 | −0.0063 [−0.012, −0.001] | 3.70 | 11/17 |
| 60 (grid top) | 0.7608 (+0.002, 4/7) | 0.7750 (+0.002, 4/7) | −0.0093 [−0.015, −0.003] | 3.46 | 11/17 |

Grades: mine FAIL, the separate grader's FAIL (a: +0.002, 4/7; b: interval entirely below −0.003
at the selected switch 60; d: not wired into the submission). The curve still rises at 60 (grid
too narrow), but PFF falls as the switch rises. The CNN alone matches LightGBM on the bench
(per-match differences of ±0.02 both ways), so it's a different model with the same accuracy.
Reported only: **the 4-member mix** (T_R, T_B1, models/R, models/B, 1/4 each, fixed before scoring):
0.7719 / 0.7723 / 0.7763 / 0.7743 at switch 20 / 30 / 40 / 60, +0.0135 at 30 (5/7); real rate 3.43
at 30. Taken to iteration 3 with a fresh-seed confirmation, since these numbers have been seen.

### Iteration 3: CNN + LightGBM mix, confirmed on fresh seeds (Oct 31): GOOD, adopted

TL = models/R + models/B + T_R + T_B1, 1/4 each (weights fixed before any scoring), switch 40 (the
best of the seen seed 0-2 curve). Graded on FRESH degrader seeds 3-5, never scored before;
`scripts/iter3_TL.sh`. Plan in `tasks/todo.md`, written before the run.

| fresh seeds 3-5 | bench Y_v2 | PFF (reported, vs 0.5221, 95%) | real rate | public (reported) |
|---|---|---|---|---|
| baseline (switch 30) | 0.7481 | 0.5221 | 3.44 | 8/17 |
| **TL, switch 40** | **0.7726** (+0.0245 ± 0.0045, 6/7) | 0.5287 (+0.0065 [+0.0023, +0.0110], 44/64) | 3.32 | 8/17 |
| TL, switch 30 (reported) | 0.7736 (+0.0255, 6/7) | 0.5305 (+0.0084 [+0.0036, +0.0130]) | 3.43 | 8/17 |

Per match (TL@40 − baseline): +0.029 +0.030 +0.021 +0.029 −0.001 +0.028 +0.036. Seeds 0-2: 0.7763
vs 0.7588. Grades: mine GOOD, the separate grader's GOOD (a, b, c, d all pass). (d): the official
`possession.submit` is byte-identical over two runs, equal to the graded candidate's file; the
validator passes; 179 possessions, mean 18.1 s (baseline 185 / 17.4 s).

Caveats (the grader's): the fresh seeds re-draw the degradation, not the matches; on them, switch
30 is as good as 40 (switch 40 is kept, as graded, rather than re-chosen after the fact); the real
rate 3.32 is close to the 3.2 floor and the bench undercounts (3.87 vs 4.38); 6/7 rests on one
−0.001 match; torch results were checked on one machine (4 threads pinned in `tcn.py`; another
CPU could differ in the last bits).

### Bench audit (Oct 31, step 2)

- **Match spread**: the baseline's per-match spellF1 on Y_v2 spans 0.709-0.797 (switch 30), so
  the noise between matches is 10x the +0.005 adoption bar. Hence the 6/7 rule.
- **Adversarial validation** (`scripts/domain_shift.py`, the real side = predicted changes or the
  whole match, never the public window): the classifier separates bench from real (AUC 0.995 overall,
  0.917 near changes, 0.79 without the top 6). The top features are team shape (home_back / away_back,
  spread): one real match vs 7 DFL ones, so team style is expected to differ. Ball visibility near
  changes is now matched: 96.7% bench (near held-out predicted changes) / 94.5% (true) vs 97.9%
  real (`scripts/audit_near.py`).
- **Bug fixed in the audit tool**: `domain_shift.auc` didn't average tied ranks, so binary
  features (ball_vis, team ids) showed univariate AUCs near 1. The classifier AUCs were barely
  affected; F3's findings stand, since they came from directly measured shares (table in F3).
- **PFF**: scored on PFF's own labels (dead from OUT to the restart only, live at the kick); the
  hackathon-style rule on PFF is unusable (2.64 changes per live minute, ball missing on ~35%).
- **Possessions**: real 185 (mean 17.4 s); bench rate 3.97 vs true 4.38 (−9%): the decoder
  undercounts changes.

## PFF decisions log

PFF Terms of Use (pff.com/terms, updated Oct 12, 2026), checked Oct 30: §1.2 "PFF grants you the
right to access and use … the PFF Data solely for your own personal, non-commercial use"; §1.5(i)
you will not "use … any PFF Data … for machine learning or training of artificial intelligence
models or tools"; §1.6 exempts "traditional machine learning, statistical, or predictive models …
that you develop and run locally … solely for Personal Use". The World Cup 2022 release
(blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) links no other licence. These terms do not
clearly allow model selection, so **from Oct 30 PFF decides nothing: every choice is made on DFL
and PFF is reported only.** Past choices PFF influenced (all before Oct 30):

| date | choice | PFF's role | bench alone would have said |
|---|---|---|---|
| Oct 27 | switch 20 (vs 30) | "PFF's peak (flat 20-30)" | 30-40 (0.734 vs 0.724) |
| Oct 28 | F0 dead-ball mix vs pure | decided: bench +0.000, PFF +0.014 (58/64) | tie |
| Oct 28 | F0 switch 20 kept | "20 vs 30 within noise on both benches" | tie |
| Oct 27 | Metrica as training data rejected | confirmed: bench +0.000 (5/7), PFF −0.006 | reject (no gain) |
| Oct 30 | F4 self-training rejected | confirmed: bench +0.001, PFF −0.004 / −0.005 | reject (< +0.005) |
| Oct 28 | F3 retraining adopted | confirmed: bench +0.015 (6/7), PFF +0.028 | adopt |

The first three are re-decided on DFL in iteration 0 (switch from round 8 / 9; the dead-ball mix
stays in every candidate since the bench saw no difference and removing it is a new change).

## Bench caveats

- **Calibrated on the model's own calls (F5, Oct 30).** The degrader's dead-ball motion and the
  dead / live missing-ball split are fitted to the real match's frames as DECODED by the
  submission, public window excluded. On the bench, that mask's bias is small (missing +1.8 / −1.2
  points, motion stats within 0.1 m/s), but it is measured where the model works well: the bench
  may under-penalise the model's own blind spots (dead balls it calls live, and vice versa).
- **"Still" ball under-reproduced**: 7% of dead frames with a ball vs 15% real; the degrader's
  position noise moves a resting ball.
- **Label definition**: the bench scores a hackathon-style rule (control >= 0.4 s within 1.5 m,
  dated from the first touch) on DFL; the reception convention (dead until the first reception)
  is a second yardstick. Neither can be checked against the hackathon GT without the public window.
- **7 matches**: the unit is the match; a best-of-N switch on 7 matches is optimistic by roughly
  the spread of its neighbours.
- **Range check** for the real rate: 3.2-7.6 changes per live minute, the min-max over the 7 DFL
  matches (hackathon-style labels, 3.76-5.18) and the 64 PFF matches (PFF's labels, 3.22-7.59),
  `scripts/label_rates.py`; replaces the public window's 2.5-6.8 from Oct 30. Set after the
  submission's rates (3.69 / 3.49 at switch 20 / 30) were known; both are inside either range.
- **PFF with the hackathon-style rule is not usable**: 2.64 changes per live minute (DFL 4.38),
  because the rule needs the ball and PFF's raw ball is missing on ~35% of live frames.
- **PFF's own labels**: dead only from an OUT event to the restart (fouls / offsides stay live),
  live at the kick; so the reception convention loses on PFF by construction.
