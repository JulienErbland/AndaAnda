# AndaAnda: ball possession from automatic tracking

More than a Hack 2027, Phase 1. FC Barcelona vs Athletic Club, one label per frame of the tactical
video: `H`, `A`, `D` (dead ball), `X` (abstain).
Code: <https://github.com/JulienErbland/AndaAnda> (MIT). Submission file: `AndaAnda.csv`.

## 1. Summary

We use the automatic tracking (ATD) as the only input. Per-frame features describe the players,
the ball and touch and restart events, with no player identities. Four models trained only on
external, openly licensed matches (DFL Bundesliga) turn them into probabilities P(H, A, D). A
3-state Viterbi decoder turns those probabilities into possessions. Nothing is trained, fitted or
tuned on the provided match's ground truth or video, and no labels on the provided match were
made by hand.

The labelled match is a test set we can barely see: 17 public possessions. So every decision was
taken on an offline **bench**. Clean DFL tracking is degraded to look like the hackathon ATD, then
scored with the organisers' own scorer, held out by match. The decisions were then confirmed on a
second, independent evaluation-only bench (PFF World Cup 2022, 64 matches).

| | bench spellF1 (DFL, held-out) | PFF spellF1 (eval only) | public window spellF1 |
|---|---|---|---|
| organisers' baseline (smooth50) | 0.275 | – | 0.326 |
| **submission** | **0.776** | **0.529** | 0.480 |

The public window is reported for completeness only. One possession moves it by 0.03–0.07, and it
was never used to choose anything in the current pipeline.

## 2. Approach

```
ATD ─► per-frame features ─► 4 evidence models ─► geometric mean ─► Viterbi (H/A/D) ─► CSV
        (no identities)       P(H, A, D)            1/4 each         + kick-off rule
```

**Why this shape.** spellF1 matches whole possessions by temporal overlap. Its big failure mode is
fragmentation: a per-frame classifier that flickers creates dozens of false possessions. So we
split the work:

- per-frame models estimate *how likely* each state is from local evidence;
- a decoder picks the single label sequence that balances that evidence against a cost per change
  of state.

A brief contrary signal is then not worth a switch, and sustained evidence is.

**Features** (`possession/context.py`, `touches.py`, `restarts.py`). There are 54 per-frame
features. They are computed offline, so they can use past and future context:

- *team context* (38): each team's speed and shape in its attacking direction, slowing down,
  last / next seen ball (position, speed, distance to a line), last / next team near the ball;
- *touches* (10): a touch detector that tells **control** (a player moving *with* the ball for at
  least 0.4 s within 1.5 m) from **deflection** (a sharp turn next to a player, no control after) and
  **contact**, with the last / next team and the time since / until;
- *restarts* (6): a restart detector (still ball >= 1.5 s, then kicked; the kind comes from where
  it is taken, the team from the taker; kick-off line-ups are detected without the ball).

**Evidence models.** Two LightGBM classifiers and two temporal CNNs:

| member | model | labels it learns | features |
|---|---|---|---|
| `models/R` | LightGBM, multiclass, 300 rounds | DFL's own possession labels | all 54 |
| `models/B` | LightGBM, same settings | hackathon-style labels (section 5) | context + touches (48) |
| `models/TR` | dilated 1-D CNN, 5 fps, ~25 s receptive field | DFL's own | all 54 |
| `models/TB1` | same CNN | hackathon-style | all 54 |

- **Training data.** Each model is trained on the 7 DFL matches. Every match is degraded twice: the
  dead ball moves in one copy and is parked in the other, so neither tracker's dead-ball habit
  becomes a shortcut. Home and away are also swapped as augmentation.
- **Why both label sets.** The two disagree in useful ways. Models on DFL's labels change
  possession too often; models on the hackathon-style labels change too rarely.
- **Why both model types.** The CNN alone scores like LightGBM but makes different errors.
- **Combining them.** The equal-weight geometric mean of all four beats the LightGBM pair on the
  DFL bench (+0.025 on fresh seeds) and on PFF. The weights were fixed before scoring.

**Decoder** (`possession/hmm.py`). A 3-state Viterbi runs per graded half on the weighted
log-probabilities. Each kind of change has a penalty: H↔A switch 40, live→dead 40, dead→live 40.
All three were chosen on the DFL bench, and 30 vs 40 was re-checked on 12 fresh seeds (a tie, so
40 stayed). Decoding a whole half at once also removes the lag of centred smoothing windows.

**Kick-off rule.** A half is `D` until its kick-off, which the restart detector finds from the
players lining up. The organisers' 2nd-half grading starts ~2 min before kick-off, and our
detector finds it from the tracking. No frame number is hard-coded. Ungraded frames (from
`games.half_video_blocks`) are written `X`. Inside the graded halves we always guess, because an
`X` across a hand-over costs spellF1.

## 3. Assets: used and dropped

| asset | status | why |
|---|---|---|
| **Tactical ATD** (tracking + metadata) | **used, the only input** | |
| Smart Tagging | tried, dropped | as soft decoder evidence it changed < 0.01% of frames (the model already agreed on 85% of tagged live frames); its SET PIECES windows are fixed 35 s blocks, too vague to date a restart |
| Tactical video + homography | tried, dropped | classical ball detection in the tracking's ball gaps: too many false candidates; the bench puts the ceiling of a perfectly tracked ball at only +0.02 spellF1. A learned patch detector trained on this video was also tried; it was never in the submission and was removed (code and model) after the organisers' Nov clarification |
| Broadcast video | not used | own timeline, not aligned to the tactical one |
| Public 5-minute GT | checks only | alignment, file format, label meaning (it reproduces the published baseline scores exactly) |

Other methods tried and dropped (full log in `docs/results.md`):

- semi-Markov decoder;
- minimum-Bayes-risk decoding;
- learned per-frame switch costs;
- a player-level ball-carrier model and a path decoder;
- ball-gap features;
- snapping changes to detected controls;
- Metrica sample games as extra training data;
- training weights that make short possessions count as much as long ones (the models change
  possession more often, but the bench score does not move: +0.001), and an 8-member mix of the
  weighted and unweighted models (+0.0004);
- self-training on the provided match's tracking (pseudo-labels from our own detector, no GT; it
  gained +0.001, below the adoption bar).

## 4. Training and evaluation data

None of it comes from the provided game.

| dataset | role | labels: who made them, how | amount | licence |
|---|---|---|---|---|
| **DFL / IDSSE open data** (Bassek et al., Sci. Data 2025) | **training** + main bench | `BallStatus` and `BallPossession` per frame, produced by the data provider (DFL / Sportec Solutions) from its tracking; we also relabel them with a mechanical rule (section 5) | 7 full matches, 25 fps | CC-BY 4.0 |
| DFL official event data (same matches) | reference for the touch and restart detectors only, not a training target | the provider's event feed (on-ball events, set pieces) | 7 matches | CC-BY 4.0 |
| PFF FC World Cup 2022 | **evaluation only** (never trained on) | possession and dead ball derived mechanically from PFF's event data | 64 matches | PFF terms of use (no ML training, not redistributed) |
| Metrica Sample Games 1–3 | tried as training data, dropped | event-derived | 3 matches | public sample data |

We made no hand labels on any game.

## 5. Possession rules implemented

The ground truth comes mechanically from corrected tracking, so we encoded its stated rules in the
labels the models learn and in the decoder:

- **Loose or contested ball: the last team keeps it.** The hackathon-style relabelling of DFL
  (`bench/relabel.py`) changes possession only when a player of the other team **controls** the
  ball: within 1.5 m, moving with it (relative speed < 2 m/s), for >= 0.4 s. Passes in flight,
  deflections, duels and single touches keep the last team. The change is dated at the start of
  control. Those constants are the touch detector's control definition, set before this rule
  existed; robustness variants (0.2 s / 1.0 m, 0.6 s / 2.0 m) move the bench by <= 0.005.
- **Short possessions count.** The ground truth counts sub-second ownership, so there is no
  minimum duration; the switch cost alone decides.
- **Dead ball.** Learned from DFL's `BallStatus`, from evidence such as players at walking pace,
  the ball beyond a line, a still ball before a restart, and (weakly) the ball not being tracked.
  Every stoppage ends a possession, so `H D H` counts as two.
- **Restarts.** The ball goes live at the kick or throw (DFL's convention), and the restarting
  team owns the next possession. A "dead until the first reception" variant was built and
  evaluated, then rejected on the bench. We also checked the definition on the public window's
  four restarts, as CHALLENGE.md invites (aggregates only): the ground truth goes live about
  0.4 s after the ball starts moving, with no player within 2 m of it, so it follows the kick,
  not the reception. No constant was taken from that check.
- **Halves.** Each half is decoded independently and is dead until its detected kick-off.

## 6. Missing and noisy tracking

The ATD loses the ball on 41% of graded frames, in long gaps (median ~10 s) that often cover the
hand-overs. Player slots are anonymous and fragment (~11 tracklets per minute, ~20.5 players
visible).

- **No identities needed.** Every feature is defined per team (nearest player, team shape, team
  speed), never per player.
- **Ball gaps.** Features carry the last and the next sighting (position, speed, team near it)
  across a gap. The models learn from player movement whether possession changed inside it. The
  decoder keeps the last team unless the evidence for a change holds up.
- **Ball speed** is derived from positions, because the ATD's own speed field is missing on most
  frames.
- **Realistic training data.** The bench degrader (`bench/degrade.py`) makes clean DFL tracking
  look like the ATD: long ball gaps (mostly in dead time, ending at restarts), tracker blackouts,
  anonymous fragmented slots, wrong-team tracklets, a smoothed ball path, false ball blips and a
  dead ball that moves.
- **Where its settings come from.** Aggregate statistics of the provided tracking (gap lengths,
  jitter, ball smoothness, tracklet counts). The dead-ball motion and the dead / live split of
  missing ball also use our own model's decoded calls on the match, with the public window
  excluded. No ground truth and no video enter the degrader, and no frame of the match is a
  training target. The models therefore train on DFL tracking that resembles the provided
  tracking in its *noise*, while their labels stay DFL's.

## 7. Validation discipline and compliance

- **How a change is adopted.** It must win in a paired per-match comparison on the DFL bench
  (7 matches × 3 noise draws, organisers' scorer), beyond noise, without breaking an edge-case
  scenario that already works (restarts, holding through a ball gap, interceptions…). It must
  also be confirmed on PFF.
- **The real match.** Its predicted possession rate is a range check against the rates of the
  71 external matches, nothing more.
- **Earlier practice we dropped.** Before Oct 27, a switch value was once picked by the public
  window's possession rate. We dropped that practice, and the current settings all come from the
  DFL bench. On Nov 10 two variants tied with the submission on the bench and scored higher on
  the public window; we kept the submission, since the window may not choose between versions. The choice of dating a change from the new team's first touch (bench labels v2)
  came from looking at the public window. That choice is about what counts as possession, and it
  was checked against PFF's independent labels.
- **What we never did.** No training, fitting or tuning on the ground truth or the video. No hand
  labels or manual corrections on the provided game. Our output was never compared with the
  video. No match data was uploaded to any AI service: scripts that touch the match print
  aggregates only, and nothing frame-level derived from it is committed.

## 8. Limitations

- **Under-segmentation.** The decoder merges possessions: on the bench it predicts 3.87 changes
  per live minute against 4.38 in the labels. Short possessions are the main misses (under 1 s:
  2% recalled; 1–3 s: 32%; over 10 s: 84%).
- **Bench vs real gap.** A classifier still tells bench frames from real ones (AUC 0.995 overall,
  0.917 near changes), mostly from team shape: one real match against seven DFL ones. The real
  match changes possession less often than the bench (3.32 per live minute, near the bottom of
  the external range).
- **Hard edge cases.** Possession changes inside a ball gap, interceptions, and kick-offs.
- **Label definitions are checked on little.** The kick convention at restarts rests on DFL and
  on the four restarts of the public window. Our rule for dating a change of possession (the
  start of control) was not checked against the ground truth.
- **The bench has stopped separating candidates.** The last variants tried sit within 0.001 of
  the submission while changing possession 3-5% more often; which rate suits the real match
  cannot be told without tuning on it.
- **With more time:**
    - a deep ball detector in the tactical video to fill ball gaps;
    - more training matches from the same competition (team-shape shift);
    - a decoder that sees short possessions better.

## 9. Reproduction

```bash
git clone https://github.com/JulienErbland/AndaAnda.git
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git   # next to AndaAnda
cd AndaAnda && uv sync                           # Python 3.12, exact versions from uv.lock
# data bundle in official_data/ (or MTAH_DATA_DIR=...)
uv run python -m possession.submit               # -> submissions/AndaAnda.csv, validated
```

The trained models are committed. Prediction is deterministic: LightGBM in deterministic mode,
torch CPU with deterministic algorithms and fixed threads, no sampling. Re-running gives a
byte-identical file in ~30 s on a laptop CPU.

Retraining the four models from public data (optional; downloads DFL, ~200 MB kept):

```bash
uv run python -m bench fetch
uv run python -m bench relabel --date-at-control --tag v1        # hackathon-style labels
MTAH_MODEL_DIR=models/R uv run python -m possession.learned train \
    --features restart --draws 2 --dead-mix
MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models/B uv run python -m possession.learned train \
    --features touch --draws 2 --dead-mix
MTAH_MODEL_DIR=models/TR uv run python -m possession.tcn --features restart
MTAH_DFL_LABELS=hackathon-v1 MTAH_MODEL_DIR=models/TB1 uv run python -m possession.tcn
```

Training is seeded. Everything is in the repository, with details in `README.md`,
`docs/pipeline.md` (method), `docs/benchmark.md` (bench) and `docs/results.md` (every experiment).
