# Pipeline at a glance (Oct 27)

One label per frame of the tactical video: `H` (Barcelona), `A` (Athletic), `D` (dead ball), `X`
outside the graded halves. Ranked on spellF1 (possessions matched by temporal IoU). Details:
`problem.md` (task), `report.html` (full report), `results.md` (every number).

## Data

| data | role | licence / status |
|---|---|---|
| **Hackathon tactical ATD** (BAR-ATH, 25 fps) | the only input of the submission | gated; never committed; scripts print aggregates only |
| Hackathon public GT (5 min, 17 possessions) | sanity checks: alignment, real change rate (4.25 / live min) | public |
| Hackathon homography + tactical video | ball detection in gaps | tried, not adopted |
| Hackathon Smart Tagging | team evidence, set pieces | tried, not adopted (redundant / too vague) |
| **DFL** (Bassek et al. 2025), 7 matches | training + main bench | CC-BY 4.0 |
| DFL official events | reference for touch / restart detectors | CC-BY 4.0 |
| Metrica sample games 1-3 | same file format as the hackathon; carrier / ball-filling models | public |
| PFF World Cup 2022, 64 matches | **evaluation only** (second held-out bench) | Terms of Use forbid ML training |

## The bench (`bench/`)

Clean DFL tracking is made to look like the hackathon ATD, then scored with the organisers' own
scorer, held-out by match.

- **Degrader** (`degrade.py`, calibrated on the real ATD's aggregates): long ball gaps (~41% of
  frames: ~7% live / ~80% dead, a stoppage's gap ending at the restart), tracker blackouts, anonymous fragmented slots (~11 tracklets /
  min, ~20.5 players), wrong-team tracklets, smoothed ball path, false ball blips, identity breaks
  in duels, and a dead ball that moves (thrown back, carried to the restart spot) like the real
  one instead of DFL's parked ball. Seeded.
- **Labels** (`relabel.py`): DFL's own labels change possession twice as often as the hackathon
  GT, so DFL is relabelled with a hackathon-style rule: dead from DFL's ball status; possession
  changes only when the other team *controls* the ball (>= 0.4 s within 1.5 m, moving with it),
  dated from its first touch. Matches the public GT's rate (4.35 vs 4.25) and durations.
- **Adoption rule**: paired per-match comparison (7 matches × 3 noise draws) on the DFL bench,
  confirmed on PFF (64 matches, evaluation only, independent labels). The real match's change
  rate is only a range check (17 public possessions: 2.5-6.8 per live minute); the public window
  decides nothing (one possession moves it by 0.03-0.07).

## The method (`possession/`)

```
ATD ──► features ──► 2 LightGBM models ──► geometric mean ──► Viterbi (H/A/D) ──► CSV
         (per frame)    P(H, A, D)            0.75 / 0.25        + kick-off rule
```

1. **Per-frame features**, no player identities, offline (past and future context):
   - `context.py` (38): team speeds and shape in each team's attacking direction, slowing down,
     last / next seen ball (position, speed, distance to a line), last / next team near the ball;
   - `touches.py` (10): control (a player moving *with* the ball >= 0.4 s), deflection (sharp turn
     next to a player, no control after), contact; last / next team and time since / until;
   - `restarts.py` (6): restarts (still ball >= 1.5 s then kicked; kind from where it is taken,
     team from the taker; kick-off line-ups without the ball); last / next team, kind, time.
2. **Four evidence models**: two LightGBM (multiclass, 300 rounds, every 3rd frame, home/away swap
   augmentation, deterministic), trained on the 7 DFL matches degraded twice (dead ball moving /
   parked, so neither tracker's dead-ball habit becomes a shortcut):
   - `models/R` × 0.5: DFL's labels, context + touches + restarts (54 features);
   - `models/B` × 0.5: hackathon-style labels dated at the control start (v1), context + touches
     (48 features). Weights and label version decided on the DFL bench (round 9, Oct 31).
   - `models/TR`, `models/TB1` × 0.25 each (and R, B × 0.25): temporal CNNs (`tcn.py`, dilated
     1-D CNN over the feature sequence at 5 fps, ~25 s receptive field, PyTorch CPU, seeded and
     deterministic, 4 threads) on the same labels as R and B1 (iteration 3, Oct 31).
   B is better calibrated on the bench but changes possession too rarely on the real match; R the
   reverse. The CNN alone scores like LightGBM but errs differently, so the 4-way geometric
   mean (`models/ensemble.json`) beats the LightGBM pair on the bench and on PFF.
3. **Decoder** (`hmm.py`): 3-state Viterbi per half on the log-probabilities; penalties switch 40
   (H <-> A, chosen on the DFL bench), stop 40, restart 40. Turns flickering evidence into possessions.
4. **Kick-off rule**: a half that is not graded from its kick-off is dead until it (the real
   second half is graded ~2 min early, all `D`).
5. **Submission** (`submit.py`): one command from the raw bundle, organisers' validator,
   byte-identical across runs.

## Where it stands

| | bench spellF1 (hackathon-style labels) | PFF spellF1 (eval only) | real rate (range 3.2-7.6) | public window |
|---|---|---|---|---|
| organisers' baseline (smooth50) | 0.275 | – | 8.29 | 0.326 |
| M2 single model (until Oct 27) | 0.611 | 0.479 | 4.28 | 0.622 |
| R + B ensemble, switch 20 (Oct 27; old bench) | 0.724 | 0.514 | 3.45 | 0.622 |
| R + B dead-ball-mixed, switch 20 (corrected bench) | 0.683 | 0.509 | 3.56 | 0.622 |
| R + B retrained on the recalibrated gaps, switch 20 (until Oct 31) | 0.725 | 0.537 | 3.69 | 0.622 |
| *F5 bench (Oct 31): dead-ball knobs from the model's own calls; same models score 0.741* | | | | |
| R + B1 (v1 labels), 50/50, switch 30 (iteration-0 baseline) | 0.759 | 0.522 | 3.44 | 0.496 |
| **R + B1 + TR + TB1 (CNNs), 1/4 each, switch 40 (submission, iteration 3)** | **0.776** (fresh seeds: 0.773 vs 0.748) | **0.529** | **3.32** | **0.480** |

The bench was corrected twice on Oct 28 (moving dead ball; ball gaps split live 7% / dead 80%),
so its numbers are not comparable across those rows; PFF is unchanged throughout. The baseline
loses 0.015 on PFF (PFF's own label definition; reported only, accepted by Julien Oct 31).

Main open problem: hand-overs are much harder to see on the real tracking than on the bench
(10 of the 17 public possessions found). Fixed Oct 28: the ATD's ball "speed" was missing on 92%
of frames (now derived from positions), DFL's parked dead ball taught "still ball = dead", and
the bench hid the ball at 23% of hand-overs where the real tracker sees it at 93-96%.

Tried and dropped (all in `results.md`): semi-Markov decoder, ball-gap features, player-level
carrier model and path decoder, snapping changes to controls, Smart Tagging, varied degradation,
cheaper switches at touches, classical and learned ball detection in the video.

## Reproduce

```bash
uv sync                                         # Python 3.12, uv.lock
uv run python -m possession.submit              # data bundle -> submissions/AndaAnda.csv (validated)
# retrain the models from public data (optional):
uv run python -m bench fetch && uv run python -m bench relabel
MTAH_MODEL_DIR=models/R uv run python -m possession.learned train --features restart --draws 2 --dead-mix
MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models/B uv run python -m possession.learned train --features touch --draws 2 --dead-mix
```
