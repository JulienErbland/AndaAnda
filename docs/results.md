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

## Touch detector (direction 2), checked against DFL events

| check | clean | hackathon-like |
|---|---|---|
| detected control agrees with the label | 87% | 87% |
| passer controls just before the pass | 78% | 59% |
| interception or won tackle: control by the winner | 22% | 16% |
| interception or won tackle: contact by the winner | 87% | 61% |
