# Test environment ("bench"): plan

Goal: replicate the hackathon conditions offline, with known H/A/D labels on several full matches,
scored by the organisers' exact scorer, with diagnostics showing *where* a draft fails.
Implementation of real methods comes after this.

## Plan

- [x] 1. Repo skeleton: `pyproject.toml` (uv, Python 3.12, pinned), `.gitignore` (external data,
      reports, hackathon data), README, `docs/problem.md` (problem analysis for the team)
- [x] 2. Common match format (`bench/match.py`): tactical-ATD-shaped arrays (normalised 0-1 coords,
      25 fps, team-only slots, NaN = missing), labels as organiser codes, graded half blocks; saved as `.npz`
- [x] 3. DFL source (`bench/dfl.py`): fetch one match at a time from figshare, stream-parse the
      position XML, labels from per-frame BallStatus + BallPossession, save npz, delete XML (disk is tight)
      - [x] verify: label stats (possessions/min, mean duration, D share) vs public GT
- [x] 4. EPTS source (`bench/epts.py`): the Metrica FIFA/EPTS reader shared by the hackathon data
      and Metrica Sample Game 3; Metrica labels derived from events (weaker, marked as such)
- [x] 5. Degrader (`bench/degrade.py`): ball dropout ~41% clustered on crowded / fast-ball frames,
      tracklet fragmentation into slots (no IDs), cross-team swaps, missing players (~20.5 visible),
      position jitter. Seeded. Calibration targets in one dict
- [x] 6. Scoring (`bench/evaluate.py`): import organiser `possession_metrics` (no copy)
- [x] 7. Diagnostics: spell counts/durations, missed/spurious spells by duration bucket,
      miss causes (merge / split / wrong team / absorbed by D / boundary), transition latency,
      per-frame error by context (ball visible, near transition, dead ball), worst stretches
- [x] 8. Timeline plot (`bench/plot.py`): GT / pred / ball-visible / error strips for a window
- [x] 9. Method 0 (`possession/baseline.py`): organisers' baseline ported to the common format
- [x] 10. CLI (`python -m bench ...`): fetch, stats, run, plot
- [x] 11. Sanity check: on degraded DFL, smooth50 beats raw by a wide margin (like the real board)
- [x] 12. Small `tests/` for the parsers' label logic, degrader rates and diagnostics

## Privacy rule (from ../CLAUDE.md)

External data (DFL, Metrica) can be inspected freely. On hackathon data, every command prints
aggregates only, and plots/worst-stretch lists stay on the user's machine.

## Review

- The bench is live on the real data (Oct 20). Public-window baselines match the published scores exactly.
- Degrader recalibrated on the real ATD aggregates (gap shape, blackouts, tracklets, live/dead split).
  Calibration script: `python -m bench stats hackathon dfl` gives side-by-side numbers.
- Bugs found while calibrating, both fixed with tests:
  1. identity breaks reused the same slot, so they were invisible;
  2. DFL referees were parsed as away players (J03WMX, J03WN1).
- Open: the live/dead split is an estimate (speed proxy). DFL changes possession about twice as
  often as the hackathon GT, so don't tune switching rates on DFL.

---

# M1: dead-ball detector + HMM decoder (hand-built evidence) + submission pipeline

Target: DFL mean spellF1 clearly above smooth50 (0.2468), by more than the 0.04 noise band; dead-ball
frame error and "absorbed_by_dead" misses down; sane on the public window; predicted possession
rate on the full hackathon match plausible (~3-5 per live minute, count only).

- [x] 1. `possession/features.py`: per-frame team-level features (nearest home/away distance, ball
      speed, ball visible, ball out of pitch, smoothed mean player speed, tracked count). No labels.
- [x] 2. `possession/hmm.py`: per-frame log-evidence for H/A/D + 3-state Viterbi per graded half,
      X outside; switch/dead/restart penalties
- [x] 3. Verify Viterbi on a toy sequence (test), timing per match
- [x] 4. Tune a handful of evidence weights on DFL (mean spellF1), features cached;
      switch penalty sanity-checked via predicted spell rate on the hackathon match, not tuned on DFL
- [x] 5. Compare vs smooth50: board + diagnostics (dead error, miss causes, lag), public window
- [x] 6. `possession/submit.py`: one command -> submissions/<team>.csv (gitignored) -> organisers'
      validate_submission.py; deterministic
- [x] 7. Review section + docs update

## M1 review (Oct 20)

- HMM tuned in 3 stages (`scripts/tune_hmm.py`). Switch penalty kept at 40: it gives 4.35
  possessions per live minute on the full hackathon match, against 4.25 observed in the public
  window. Switch 20 scores higher on DFL (0.381) but predicts 5.9 per minute: it fits DFL's label
  definition, not the hackathon's.
- DFL mean spellF1 went from 0.247 (smooth50) to 0.352; macroF1 from 0.585 to 0.80. Public window 0.413 (smooth50 0.326).
- Submission: `python -m possession.submit` runs in 15 s, validates, and gives 252 possessions with
  a mean of 13.8 s (public GT: 14.2 s).
- Edge-case suite: `python -m bench scenarios --preview` + `scripts/scenario_page.py` (13 scenarios,
  published page). Weak spots for M2:
  - ball_gap_hold 23%;
  - change_in_ball_gap 0%;
  - blackout_change 1%;
  - short_spell 2%;
  - restarts 23%;
  - kickoff 12%.

---

# M2: learned evidence (LightGBM) inside the HMM

Why: M1 fails whenever the ball is invisible. In 116 of 154 failed ball-gap clips, slow live play
was called dead (speed + no ball can't separate them), and hand-overs in gaps are found 0% of the
time. A per-frame model trained on degraded DFL can combine many weak clues; the HMM keeps
the temporal structure.

- [x] 1. `possession/context.py`: rich per-frame features, ball-free where possible, all offline
      (past and future context allowed): multi-scale player speeds and deceleration, per-team
      centroid/forward velocity/spread in each team's attacking direction (from kick-off
      positions), ball visible/speed/out, time since/until ball seen, last/next seen ball position
      and speed, last/next controlling team and time since/until, team-evidence rolling means
- [x] 2. `possession/learned.py`: LightGBM multiclass H/A/D, trained on degraded DFL with home/away
      swap augmentation, leave-one-match-out for evaluation; deterministic (fixed seed, threads)
- [x] 3. Emissions = lambda * log p in the M1 Viterbi; tune lambda and the penalties with the staged
      logic (switch checked against the hackathon possession rate, not tuned on DFL)
- [x] 4. Compare with M1: board, diagnostics, edge-case scenarios (target: ball_gap_hold,
      change_in_ball_gap, restarts), public window, full-match rate
- [x] 5. Ship: model trained on all 7 DFL matches saved to `models/` (public data, so OK to commit);
      `python -m possession.train` regenerates it; submit uses it
- [x] 6. Tests, docs, review

## M2 review (Oct 20)

| | smooth50 | M1 hmm | M2 learned |
|---|---|---|---|
| DFL held-out spellF1 (7 matches) | 0.247 | 0.352 | **0.522** |
| DFL macroF1 | 0.585 | 0.80 | **0.883** |
| public window spellF1 (17 spells, noisy) | 0.326 | 0.413 | **0.622** |
| full hackathon match, possessions per live min | - | 4.35 | **4.16** (public window: 4.25) |

- Decoder: switch 10, stop/restart 40. DFL is flat for switch 3-10; 10 is the closest to the real
  possession rate. With confident evidence, the DFL and real-match criteria now agree.
- The submission is byte-identical across runs: 227 possessions, mean 14.4 s.
- Scenarios improved in 11 of 13. Restarts 23% -> 55%, change in ball gap 0% -> 25%, dead ball
  with the ball visible 77% -> 93%.
- Known limit, next milestone: ball_gap_hold stays at 23%. P(dead) rises with gap length (correct
  on average), and per-frame log-probs summed over hundreds of correlated frames outweigh the
  stop+restart penalty. Fix: a duration-aware (semi-Markov) decoder, or tempering correlated
  evidence.
- Still to do: Smart Tagging priors (with simulated DFL tags), ball re-detection from video.

---

# M3: semi-Markov (explicit-duration) decoder

Why: ball_gap_hold stays at 23%. A slight per-frame lean towards "dead", summed over hundreds of
correlated frames in a long ball gap, outweighs the stop+restart penalty and splits a live
possession. Two remedies, measured separately:
  (a) duration priors per state (how long possessions and stoppages really last), via a
      semi-Markov decoder that scores whole segments;
  (b) correlation correction: a ball gap of G frames contributes ~kappa frames of evidence
      (per-frame weight min(1, kappa/G)), since its frames repeat the same information.

- [x] 1. Duration stats: GT segment lengths per state on DFL and the public window -> lognormal priors
- [x] 2. `possession/hsmm.py`: segment Viterbi at 5-frame (0.2 s) blocks with prefix-sum segment
      evidence, lognormal duration log-prob, same transition penalties; boundary refinement to the
      exact frame (+-5 f) from per-frame evidence; gap weighting option
- [x] 3. Test: exact against brute-force segmentation on short sequences; refinement on a toy
- [x] 4. Ablation on held-out DFL probs (A: M2, B: +gap weights, C: +durations, D: both):
      spellF1, macroF1, public window, full-match rate, key scenarios
- [x] 5. Adopt the winner only if it beats M2 beyond noise (or fixes ball_gap_hold without
      losing elsewhere); docs, review, commit

## M3 review (Oct 20): negative result, M2 kept

| variant | DFL spellF1 | ball_gap_hold | change_in_ball_gap |
|---|---|---|---|
| A: M2 Viterbi | **0.522** | 23% | **25%** |
| B: gap weighting k=25-200 | 0.481-0.507 | 27-30% | 4-14% |
| C: duration priors | 0.502-0.517 | 23% | 21-24% |
| D: both | 0.463-0.492 | 28-30% | 3-9% |

- Gap weighting assumed that gap frames repeat one piece of information. Wrong: last/next control
  and team movement locate hand-overs inside gaps.
- Duration priors are redundant with the learned probabilities plus the switch penalty.
- `possession/hsmm.py` is kept (exact vs brute force) but not adopted. The next lever for ball gaps
  is better gap features (what separates a live gap from a stoppage), not the decoder.

---

# Next directions (Oct 20): three ways to fix M2's weak spots

Specs written first:
- `tests/test_directions.py`: 13 unit specs on synthetic situations (`tests/toys.py`),
  xfail(strict) until built; the toy geometry was checked numerically.
- `tests/test_targets.py`: bench gates, run with `pytest -m slow`, about 25 s. 7 floors pass today;
  8 targets are xfail until reached. When an XPASS is reported, move that target into FLOORS.

## Direction 1: restart detector (`possession/restarts.py`)
Weak spots: restart_same 56%, restart_other 55%, kickoff 12%.
Idea: every restart is a visible ball standing still for >= 1.5 s that then moves. Where it stands
gives the kind (centre spot = kick-off, touchline = throw-in, corner, goal area = goal kick, else
free kick). The nearest player when it moves gives the restarting team. Turn detections into
per-frame features for the LightGBM ("time since/until a restart", "restart team", "kind") and
sharpen the decoder's D -> H/A switch at a detected restart. On the hackathon match, also use
Smart Tagging `SET PIECES` (Type/Team) as a second source (throw-ins have Team = N/A: take the
team from the detector).
- [ ] specs: kick-off, throw-in, kind by position (corner, goal kick, free kick), none in open play
- [ ] targets: restart_same >= 70%, restart_other >= 70%, kickoff >= 50%

## Direction 2: touch / control detector (`possession/touches.py`)
Weak spots: interception 28%, short_spell 4%.
Idea: replace "nearest player within 1.5 m" with real touch events. A control is the ball
slowing to a player's pace and staying with him for >= 0.4 s; a deflection is a > 30 degree turn
of the ball near a player, with no control after it. A pass flying past an opponent is no touch.
Feed "last/next control team, time since/until" from these events into the model, and refine
each predicted hand-over to the frame of the control that caused it.
- [x] specs: interception = control, pass past an opponent = nothing, deflection != control,
      a 0.8 s control still counts
- [ ] targets: interception >= 40% (reached: 40%, now a floor), short_spell >= 15% (8%)
- Review (Oct 21): adopted. Degrader noise fixed first (white -> smooth; jitter 58-88 cm -> 1.2-2.2 cm,
  real 0.1-0.9 cm), M2 retrained (0.523). Touch features: DFL 0.523 -> 0.550, +0.028 +- 0.006,
  7/7 matches; interception 30 -> 40%, restart_same 48 -> 57%; real rate 4.32. Public macroF1
  0.894 -> 0.861 (17 spells, watch it). Not done yet: snapping hand-overs to the control frame.

## Direction 3: ball-gap reasoning (`possession/gaps.py`)
Weak spots: change_in_ball_gap 25%, ball_gap_hold 23%, blackout_change 14%. M3 showed the
decoder isn't the problem: the evidence in gaps is.
Idea: describe each ball gap as a whole: who lost and who found the ball; the implied ball speed
from the last-seen to the next-seen point (live passes are plausible, a stoppage is ~0); the
fastest players' speed during the gap (live gaps have sprinters); and whether the team shapes flip
mid-gap (an attacking team retreating means it lost the ball). Add these as features. Later, if
needed, re-detect the ball in the tactical video (largest ceiling, needs the video and a CV model).
- [x] specs: live gap kept by one team, dead gap, shape flip = change of possession
      (`possession/gaps.py` built, 3 specs pass)
- [ ] targets: change_in_ball_gap >= 40%, ball_gap_hold >= 45%, blackout_change >= 30%
- Result v1 (Oct 20): null. Gap features added to LightGBM: DFL 0.5220 -> 0.5205,
  ball_gap_hold 23 -> 24%, change_in_ball_gap 25 -> 26%, blackout 14 -> 10%. The context features
  already carry the same information, so the model was rolled back to M2 and gaps.py is unused.
- Next for this direction: (a) more training data. Gap hand-overs are rare (423 clips over 7
  matches), and PFF WC2022 would bring 64 more matches. (b) Ball re-detection from the tactical
  video, which removes gaps at the source.

Order suggestion: 3 first (the largest share of real frames: ball missing 41%), then 1, then 2.
Each direction must keep `pytest -m slow` floors green.

---

# DFL official events (Oct 20): added

`bench/dfl_events.py`, run with `python -m bench fetch dfl-events`: 9,961 events over the 7 bench
matches, synced to bench frames.
- Clock: frame-anchored events (CalculatedFrame) fix the map. A real kick sits exactly on every
  anchor, so the frame numbering is right.
- Lag: the hand-logged EventTime runs -1.2 to +2.5 s late per half; the median is subtracted.
- Snap: on-ball events go to the kick or touch by an event-team player that best fits time and
  position (event positions match the ball to ~1 m once y is flipped), within +-2 s, then +-6 s.
  Set pieces go to the nearest dead -> live switch of the labels.
- Quality:
  - anchors at 1-3.5 frames median (raw clock: 18-44);
  - pass team agrees with the possession label 98%;
  - set pieces 97-100% snapped and 96-100% consistent (dead just before, taking team after);
  - kick-offs 31 of 32.
- Use: reference for direction 1 (restart frame, kind, team) and direction 2 (passes,
  interceptions, tackles), and simulated Smart Tagging set pieces for the hackathon priors.

---

# Benchmark (Oct 20): docs/benchmark.md, scripts/benchmark.py

The priorities are revised by the oracle ceilings (gain over M2 0.519 if one error type were
perfect):
- team in live play +0.39;
- short possessions +0.24 (DFL-inflated);
- hand-overs +0.18;
- ball gaps +0.17;
- stoppage starts +0.11;
- restarts +0.04;
- ball always tracked +0.04;
- timing +0.03;
- dead/alive +0.02.

New order:
1. Direction 2, touch/control detector (team at hand-overs; 86% of misses start with a hand-over,
   and most happen with the ball visible).
2. Direction 1 re-aimed at stoppage starts: who touched it last before the ball went out.
   Restarts alone have a low ceiling.
3. Direction 3 (gaps) via more data (PFF) or video, not more features.
- Optional free change: switch 10 -> 5 (+0.006, wins on 7 of 7 matches, real-match durations
  closer to the public labels).

---

# Carrier model (Oct 21): who has the ball, at player level

Base trained on open data (DFL 7 + Metrica 3); the hackathon match is only a later fine-tuning
step (self-supervised: visible-ball controls, ball hidden on purpose), never the base.
- [x] Metrica sample games 1 and 2 (CSV format) in the bench: `python -m bench fetch metrica`
- [x] `possession/carrier.py`:
  - labels on CLEAN tracking: frames inside a control episode (touches.py) -> the controlling
    player; live otherwise -> "flight"; dead -> "out". Mapped onto the degraded slots by
    position (the degrader repacks players into new slots).
  - one row per (frame, tracked slot), features computed on DEGRADED tracking: ball distance and
    relative speed when seen, distance to the last/next seen ball, pressure (opponents near,
    closing speed), ranks within the frame. Ball-free features carry the gaps.
  - LightGBM binary "this slot is the carrier"; leave-one-match-out models.
  - outputs per frame: per-slot P(carrier), P(home carrier), P(away carrier).
- [x] standalone check, held-out: ball missing, right player 58% (best rule 40%), right team
      77% (74%); ball seen ~99% for everything.
- [x] carrier features in M2 (out-of-fold): REJECTED. +0.0095 on DFL (7/7) at switch 10 but
      real rate 4.60 and public 0.533; at switch 17 (rate 4.32) DFL 0.543 < 0.547. The gain was
      DFL-style extra changes.
- [ ] later: hackathon fine-tuning (continued boosting).

# Path decoder (Oct 21): follow the ball from player to player (`possession/path.py`)

State per frame: carrier slot (top 3 by carrier probability + last frame's), flight after home,
flight after away, out. A possession change is a receive by the other team (or a tackle, or an
unseen hand-over at a high cost); flights never change team, so deflections change nothing;
identity breaks pass the ball to the continuing slot for free. Evidence: carrier per-slot
probabilities + M2's H/A/D, so nothing M2 knows is lost. Switch/stop/restart stay M2's.
- [x] decoder + 4 specs (deflection, interception at the control frame, unseen hand-over,
      break vs tackle rules), `tests/test_path.py`
- [x] held-out comparison: REJECTED for now. −0.015 ± 0.006 vs M2 (1/7), all 6 variants −0.013
      to −0.022, and MORE changes on the real match (4.4-4.6). Likely double counting: M2's
      probabilities already hold touch/proximity evidence, carrier adds the same again, and
      flight -> other team is too cheap in gaps. Next try: use it only to move each of M2's
      own changes onto the control frame (no new changes).

# Ball filling (Oct 21): where the ball is during gaps (`possession/ballfill.py`)

Free supervision: the degrader's gaps on clean open data (DFL 7 + Metrica 3), true ball kept.
LightGBM predicts the correction to the straight line between sightings (x, y) from the
players (carrier-weighted position, nearest players, convergence of nearby players' runs,
team centroids), plus a model of its own error radius. Leave-one-match-out.
- [x] model, `finetune(match)` (hide visible hackathon ball in gap-shaped spans, continued
      boosting; run locally as the last step), 1 spec (`tests/test_ballfill.py`)
- [x] held-out check: long gaps (>10 s) 9.6 m median vs 15.3 (line) / 17.8 (last seen); short
      gaps still worse than "stays where last seen" (0.9 vs 0.4 m in the first second). v2 radius
      honest (77% inside the 80% radius). Not a team signal by itself (nearest player 50%).
- [ ] then: touches and carrier on the filled ball where the radius is small

---

# Fixes for the open issues (Oct 22)

## 1. Fix the bench: hackathon-style labels on DFL
The hackathon GT is a rule applied to corrected tracking; DFL gives clean tracking. Relabel DFL
with a hackathon-like rule so the bench measures the right definition of possession.
- [x] rule on clean tracking (`bench/relabel.py`, `tests/test_relabel.py`): possession changes only on a control by the other team (touches.py
      control episodes); a loose / contested ball stays with the last team; dead from DFL's
      BallStatus
- [x] tune its few knobs (control duration, distance) until the relabelled DFL matches the public
      labels: ~4.3 possessions per live minute, durations p10 / p50 / p90 ~ 2.2 / 7.6 / 38 s
- [x] `python -m bench relabel [--calibrate]` -> external/dfl_hk/; `MTAH_DFL_LABELS=hackathon`
      switches every script to them; originals untouched. Calibration: control >= 0.4 s within
      1.5 m gives rate 4.35 (public 4.25), p10/p50/p90 2.2/8.8/31.3 s (public 2.2/7.6/38.1);
      DFL original 8.30, 0.8/3.7/18.6. 85% of live frames agree with DFL's team.
- [x] retrain M2 + touches on the new labels (models_exp/hk): bench 0.586 -> 0.732, but real rate
      3.55 and public 0.433 -> not adopted. Bench->real shift found: the real ball path is
      smoothed (deflections 2.1/min vs 6.1 on the bench, controls 5.6 vs 7.9).
- [x] degrader: ball smoothing 0.20 s (`Profile.ball_smooth_s`), fitted on the real ball's
      acceleration percentiles, checked on touch rates (deflections 6.1 -> 2.3/min, real 2.1)
- [x] retrain on the smoothed bench: A_s, B_s, B_lo, B_hi (robust to the rule; see results.md)
- [x] benchmark and scenarios re-run on the fixed bench (docs/benchmark.md, Oct 22 section);
      report.html updated (summary, §4, new §5b, next steps)

## 2. Re-test carrier and path on the fixed bench
- [ ] M2 + carrier features vs M2 + touches, paired by match, at the real-match change rate
- [x] path decoder used narrowly (`possession/snap.py`): −0.012 on DFL labels, +0.016 on the
      hackathon-style labels (circular), no change on the real match -> not adopted
- [x] where Metrica's labels put a change (`scripts/change_timing.py`, 12 public hand-overs):
      nearest control start by the new team +0.94 s median (8% within 0.5 s); nearest touch of
      any kind +0.12 s (33% within 0.5 s) -> changes sit at the new team's FIRST TOUCH
- [x] relabel rule v2 (`Rule(date_from_touch=True)`, external/dfl_hk_v2): same rate 4.35 and
      durations; change timing vs touches now like the real labels (control start +1.08 s, 27%
      within 0.5 s; real +0.94 s, 8%; v1 +0.04 s, 60%)
- [x] v2 is the default (v1 kept in external/dfl_hk_v1)
- [ ] full path decoder again, with M2's team evidence down-weighted, for comparison

## 3. Degrader realism
- [x] measured (`scripts/degrader_realism.py`), real vs bench: ball jumps 1.12 vs 0.13/min;
      implied speed across <2 s gaps 66 vs 0.6 m/s (the ball reappears elsewhere); gap-start
      ball speed p50 1.7 vs 0.5 m/s, crowd 1.04 vs 0.57; track ends near an opponent 33% vs 12%
- [x] knobs added, off by default (byte-identical when off, tested): ball_false_per_min /
      ball_false_s, break_duel_gain / duel_m; gap-start weights already existed
- [x] calibrated and set as defaults (+ structural knobs: false blips in gaps, gaps from
      ball-out with a share, duel-weighted breaks, tracklet_s 300); remaining mismatches: gap-start
      speed tail, implied speed across short gaps
- [x] retrain on the realistic bench with v2 labels: B_r bench 0.740 but real rate 3.42, public
      0.496, 8/17 found (B_s: 3.22, 0.330, 6/17) -> better transfer, gap not closed, not adopted;
      A_r +0.038 over the submission on the bench, same public (0.622, 10/17) -> held
- [ ] remaining realism gaps: false blips (implied speed across short gaps 13 vs 66 m/s),
      gap-start speed tail; then ball detection from video (item 5)
- [x] varied degradation in training (`degrade.vary`, `learned train --draws N --vary S
      --profile P`; guard against writing experiments into models/)
- [x] check: clean −0.116 (0/7), varied +0.006 / +0.010 under stress (within noise), public worse
      -> one degraded draw stays; varied not adopted

## 4. Smart Tagging
- [x] reader for the hackathon tags (`possession/tags.py`); Team meaning per code measured on the
      public window (`scripts/tag_agreement.py`): attacking codes = team with the ball, defending
      codes = the other team, box / final third = no information
- [x] (dropped) simulating tags on DFL: the weight would only reflect the invented noise level;
      used the simpler, conservative public-window weight instead
- [x] soft evidence in the decoder, one piece per tag (log-odds 0.7 spread over its frames;
      `learned.decode(bonus=...)`); set pieces not used yet
- [x] check on the real match: < 0.01% of frames change; M2 agrees with the tags on 85% of their
      live frames -> redundant at a safe weight; NOT used in the submission

## 5. Ball gaps from the tactical video
- [x] detector (`possession/video.py`): search window from the ball-filling prior + homography;
      small bright unsaturated MOVING blobs of the expected pixel size; temporal linking. Classical
      CV (OpenCV), no deep model yet. 5 tests on synthetic frames / a synthetic MP4
- [x] pixels -> pitch through the per-frame homography; `apply(match)` fills gaps only
- [x] self-check + diagnose + 72-setting sweep: best 24.7% within 1 m, median 2.68 m vs prior
      2.57 m -> the classical detector does not work (cannot tell the ball from similar blobs)
- [ ] learned patch classifier trained on THIS video, the tracked ball as free labels (59% of
      frames); go/no-go on the self-check (>= 60% within 1 m, error below the prior)
- [ ] only then `detect` and compare M2 with/without filled balls on the real match

---

# PFF World Cup 2022 (Oct 22)

- [x] `bench/pff.py` reader (64 matches, `python -m bench fetch pff` after downloading to
      external/pff/raw/); labels from `home_ball`; degrader profile "pff"
- [x] grouped held-out training (`learned train --with-pff`: DFL matches alone, PFF in 5 folds;
      rows computed once, in parallel)
- [x] P (DFL + PFF): PFF held-out 0.579 vs 0.483 (64/64); real 11/17 public possessions (best so
      far), public 0.586; PFF reproduces the real under-changing (3.4 vs 4.72 per live min)
- [ ] tune P's switch penalty on the PFF folds so the predicted rate matches the truth; check the
      real rate / public window; candidate to replace the submission

---

# Round 3 (Oct 27): items 6, 7, 8, 11, 12 of the review

Rule for every adoption: paired DFL comparison (v2 labels, realistic bench, 3 seeds) read at the
real-match change rate (switch whose real rate is closest to the public 4.25); public window as a
sanity check only. Real-match outputs are aggregates only.

- [x] 6. Second-half pre-kickoff (110,808-113,741 is all D): measure the submission's non-D frames
      and possessions there; if any, force D before the detected kick-off of each half
      (`scripts/prekickoff.py`). Before: submission 1,059 H frames / 6 false possessions, A_r 325 / 2.
      Now (`restarts.kickoff_frames` + `learned.Params.kickoff_dead`): 27 / 13 H frames, 1 run each,
      inside the last 1.2 s before the kick (the first real possession starting early)
- [x] 7. Switch / model: A_r vs the submission at matched real rate (reports/compare/{sub_r,A_r}.json).
      Old decoder: A_r switch 7 -> real rate 4.25, public 0.622, bench 0.637 vs submission 0.611
      at switch 10; at matched real rate +0.013 +- 0.005, 7/7. Re-check with the kick-off rule
      (models_k vs A_r_k): A_r switch 7 +0.026 ± 0.006, 7/7, rate 4.24 -> superseded by item 8
- [x] 8. (built: `learned.members` / `probabilities` read ensemble.json) -> R_r + B_r 75/25 ADOPTED
      Ensemble: `ensemble.json` in a model dir = weighted log-mean of member models'
      probabilities (A_r + B_r); sweep weight and switch; paired vs the item-7 winner
- [x] 11. (built, 6 specs pass; R_r +0.007 ± 0.003 vs A_r, 6/7; in the submission as models/R) Restart detector `possession/restarts.py` (specs in tests/test_directions.py, direction 1):
      detect(match) -> [Restart(frame, team, kind)]; features into M2 (feature set "restart");
      retrain held-out; paired comparison; Smart Tagging SET PIECES agreement with the detector on
      the real match (aggregate), used only if it adds information
- [x] 12. NO-GO (42.6% within 1 m, gate 60%; median 1.23 m vs prior 1.96 m) Learned ball-patch classifier on the tactical video (tracked ball = free labels):
      go/no-go on the self-check (>= 60% within 1 m on held-out frames, error below the prior);
      only on a go: fill gaps, compare M2 with / without on the real match
- [x] Docs: README, docs/results.md, docs/benchmark.md, docs/report.html, tasks/todo.md

## Round 3: state when paused (Oct 27, 18:55)

Measured so far:
- restarts (`scripts/validate_restarts.py`, DFL events): clean found 84%, team 94%, kind 91%;
  degraded found 37%, team right 14/15 when named (named on ~50%). DFL parks a dead ball where it
  left the pitch (57% of still stretches outside the lines; real match 11.5%,
  `scripts/restart_realism.py`): parked stretches are skipped. Kick-off line-ups (no ball):
  found 58% clean / 39% degraded.
- Smart Tagging SET PIECES (`scripts/tag_restarts.py`): fixed 35 s windows, the model's restart
  sits +5.7 s after the tag start (IQR +3.7/+8.6), team N/A 53% -> too vague, not used.

Resume (all resumable, ~1 h):
1. `scripts/round3.sh > reports/compare/round3.log 2>&1` (5 evals; skips finished ones)
2. `uv run python -m possession.ballpatch check > reports/ballpatch_check.log 2>&1` (~20 min)
3. compare: `uv run python scripts/compare_models.py compare reports/compare/models_k.json
   reports/compare/<X>_k.json` for X in A_r, ens_AB, ens_AB3, R_r; adopt at matched real rate
4. adopt winner into models/ (+ switch in learned.Params), `python -m possession.submit`,
   `pytest -m slow`, then docs (README, results.md, benchmark.md, report.html)

## Round 3 review (Oct 27)

- Submission is now `models/ensemble.json`: models/R (R_r: DFL labels, touch + restart features)
  × 0.75 + models/B (B_r: hackathon-style labels, touch features) × 0.25, switch 3, kick-off rule.
  Bench (v2 labels) 0.667 vs 0.611, +0.056 ± 0.007, 7/7 at matched real rate; real rate 4.18;
  public 0.622 (unchanged), 10/17 public possessions; 220 possessions, mean 14.4 s; byte-identical
  across runs; validator passes. Public macroF1 0.861 -> 0.839 (bench macroF1 up): watch it.
- Pre-kickoff false possessions 6 -> 0 (item 6).
- `pytest -m slow` re-based on the v2 labels (12 floors pass; ball-gap targets reached).
- Not adopted: learned video ball detector (NO-GO), Smart Tagging SET PIECES (35 s windows).
- Caveat for the tech doc: B is trained on the rule the bench scores with; the real change rate is
  the guard every adoption was read at.
- Open: items 1-5 of the review (tech doc, fresh-clone reproduction incl. `bench relabel` for B,
  opencv to an optional group, team size, final validation); the bench -> real gap (adversarial
  validation, self-training on the real match).

---

# Review of the pipeline (Oct 27): fixes and different approaches

Estimates are bench spellF1 gains (hackathon-style labels, paired, 7 matches) unless stated; the
real-match effect is always less certain than the bench's.

## Fixes

- [x] **F0. The ATD's ball "speed" is not a speed (found Oct 27, highest priority).** DONE Oct 28:
      speed always derived; degrader moves the dead ball (`dead_ball_moves`); models trained on
      both behaviours (`--draws 2 --dead-mix`): corrected bench +0.117 (7/7), PFF −0.005 (22/64),
      real public macroF1 0.871. Height feature: dropped (the value is not a height either).
      `scripts/domain_shift.py`: bench vs real frames are separable at AUC 0.998, 77% of it from
      `ball_last_speed`. The ATD's third ball value correlates 0.06 with the speed from positions
      (median 1.1, p99 5.3, never 0): most likely the ball's HEIGHT in m. The bench derives a real
      speed, so on the real match every ball-speed feature reads height. Deriving the speed on the
      real match is the consistent fix but, as is, makes the current models miss dead balls
      (public window: 475 vs 1,026 of 1,469 dead frames): DFL parks a dead ball motionless, the
      real one is carried / thrown back. Reverted until:
  - [ ] degrader: a visible dead ball moves like the real one (follows the nearest player, or the
        real dead-ball speed distribution; aggregates only: derived ball speed on the public
        window's dead frames vs the bench's dead frames);
  - [ ] derive the ball speed everywhere; retrain R and B; re-run the bench + PFF + real checks;
  - [ ] optional: DFL tracking has the ball's z: a height feature on both sides (check
        `bench/dfl.py` keeps it).
      Estimate: unknown but potentially the largest real-match gain of the list (the model's
      ball-speed evidence is currently wrong on every real frame); bench unchanged by construction.
      Effort ~1 day with retraining.
- [x] **F1. Choose on the bench; the public rate is a range check only.** DONE: switch 20 (bench 0.724, PFF 0.514, real 3.45). Checked: exact Poisson
      95% interval for 17 possessions is 9.9-27.2 events; the window has 4.02 live minutes, so
      2.5-6.8 changes per live minute. "Closest to 4.25" (switch 3, the 0.75/0.25 mix) fits noise.
      New rule: paired bench comparison decides; the real rate must fall in [2.5, 6.8]; the PFF
      evaluation-only bench (independent labels, not the bench's rule) breaks ties and guards
      against the B-family's circular advantage. Effort: hours (existing evals re-read).
      Consequence to expect: higher switches and more weight on B (bench best: ens 50/50 at
      switch 20, 0.743; B_r alone 0.74).
- [x] **F2. The public window never decides.** Adopted as policy. One possession moves spellF1 by ~0.03-0.07 here
      (2·TP/(n_pred + 17) with ~10 predicted); several past "not adopted: public drops" calls
      were within that step. Keep it as an alignment / format check only.
- [x] **F3 (= review A). Adversarial validation near changes.** DONE Oct 28: the bench hid the
      ball at 23% of hand-overs (real: 4-7%); ball-gap split recalibrated (live 7% / dead 80%,
      gaps end at restarts); retrained: bench +0.015 (6/7), PFF +0.028 (61/64). The overall run found F0; repeat
      on frames within ±2 s of a possession change (bench: true changes; real: the public
      window's 12 changes + the model's predicted changes) to see which touch / control features
      fail to transfer there. Effort 1-2 h. Then fix the degrader or drop the features.
- [x] **F4 (= review B). Self-training on the real match.** REJECTED Oct 30: bench +0.001 (both modes), PFF −0.004 / −0.005 (`results.md`). Built (`possession/selftrain.py`,
      `learned train --self-train W`: control-based pseudo-labels, ball hidden in gap shapes,
      public window excluded; disagrees with the submission on 3.1% of confident live frames).
      Blocked on F0 (its scaffold reads the wrong ball speed). Then run two variants: as built,
      and the reviewer's conservative one (confident mid-possession frames only). Judge on the
      bench (must not drop), PFF and the real-rate range. Estimate +0.00-0.03 real, ~0 bench;
      risk: reinforces current misses.

## Different approaches

- [x] **C. Minimum Bayes risk decoding for spellF1.** REJECTED Oct 30: best temp 0.5 +0.0002
      (1/7); higher temps pick lower switches and lose up to −0.044 (`results.md`). Forward-filtering backward-sampling of a
      few hundred paths per half from the HMM posterior (with a temperature on the scores: the
      per-frame terms are correlated, so the raw posterior is over-confident), then pick among
      candidates (Viterbi at several switches + samples) the one with the best mean spellF1
      against the samples, with the organisers' scorer. Feasible: numpy FFBS vectorised over
      samples, ~1-2 min per match. Bonus: picks the switch per match, which removes F1's noisy
      choice. Estimate +0.00-0.02 bench (the gain is bounded by how often a different path wins
      on expected spellF1); effort ~1 day.
- [ ] **D. Learned, time-varying switch costs.** Oct 30 bench grid (beta 0-4 x switch 14/20/30):
      at switch 30 beta 4 +0.004 over beta 0, beta rising with the switch; and switch 30 alone
      beats 20 by +0.0065 (7/7): grid edge. Re-run at switch 30-80 after F8's retraining. A second LightGBM predicts "change within
      ±0.5 s" (bench labels); its log-odds set the switch cost per frame (IO-HMM-like), with one
      or two scale parameters tuned on the bench. Differs from the rejected touch discounts
      (rule-based) but overlaps them: the misses are mostly short possessions the evidence
      barely sees. Estimate +0.005-0.02; effort ~1 day.
- [ ] **E. PathCRF as extra features** (arXiv 2602.12080, Feb 2026, code
      github.com/hyunsungkim-ds/pathcrf; training data to verify in the paper). Set-attention
      edge scores + CRF over player-to-player possession paths, no ball. Catches: needs fixed
      player nodes (our slots break ~11 times per minute: join tracklets on short windows), live
      play only, DFL's possession definition, a torch dependency. Use: per-team summed edge
      probabilities as LightGBM features. Estimate +0.00-0.03 bench, real transfer doubtful;
      effort 2-4 days. Low priority; only after F0-F4.
- [ ] **F. Ball-free deep model from scratch** (Peral et al. WACV 2026; Ball Radar,
      arXiv 2306.08206): skip. Trained on 283 proprietary matches, no code; with 7 matches it is
      unlikely to beat LightGBM. Estimate <= current.

- [x] **Review fix 3 (earlier list): Metrica sample games as training data.** Rejected: DFL bench
      +0.000 / −0.003, PFF −0.006 ± 0.002 (19/64) at switch 20 (`results.md`).

## Rule changes (organisers, Oct 30): no tuning on the public 5 minutes

- [ ] **F5. Bench calibration from algorithmic calls only.** F0's dead-ball knobs and F3's split
      were fitted on the public window's labels. `scripts/dead_ball_realism.py` now takes real
      targets from the model's confident dead frames and a model-free walking-pace mask (they
      agree; `Profile.dead_wait_s` added). Grid so far: still 9-11% vs 17-20% real. Waits on F8
      (the targets depend on which frames count as dead).
- [x] **F6. Relabel rule off the public window.** CONTROL_S 0.4 / NEAR_M 1.5 were set in
      `touches.py` on Oct 21 (a0c9f1a), before the relabel calibration (f5ac33c, Oct 22); B_lo /
      B_hi showed the result barely moves. Left: `date_from_touch` came from the public window's
      12 changes -> v1 vs v2 on PFF (independent labels) in F8's round; drop PUBLIC_RATE/Q from
      `calibrate()`.
- [ ] **F7. Question for the organisers** (issue on their repo): is matching a bench label
      definition to the public window's aggregate rate "tuning"? Draft in `docs/`.
- [ ] **F8. Dead-ball convention (PR #1, max/dev).** Hackathon GT (Julien's reading): a restart
      stays D until the first reception; DFL and PFF go live at the kick; PFF is dead only for
      ball OUT (fouls / offsides stay live). Merge the PR into this branch; apply the reception
      rule on top of the hackathon-style labels (`hackathon-rx`), keep `ball_dead` for the
      degrader (gaps AND F0's dead-ball motion); retrain R/B; compare decoder rule vs trained
      labels; check PFF's convention before trusting PFF deltas on restarts.

Suggested order: F8 -> F5 -> F6 (v1/v2) -> D at higher switches -> E.

## State when paused (Oct 28, 01:25)

Submission: R_g + B_g (`models/`), switch 20; bench 0.725, PFF 0.537 (commit 2692053).
- F4 self-training: R_sc (control) and R_ss (conservative) trained in models_exp/; their
  evaluation was stopped. Resume: `scripts/round7_selftrain.sh > reports/compare/round7.log 2>&1`
  (skips the trained models; ~45 min: bench, PFF, real for both). Adopt only if the bench does
  not drop and PFF gains (vs `reports/compare/ens_RgBg_g.json`, `pff_sweep_R_g_B_g_0.75.json`).
  Watch: the self-training rows hide the ball on 82% of frames (real 41%).
- C (MBR decoding): built (`possession/mbr.py`, `scripts/eval_mbr.py`, FFBS test), not yet
  evaluated. One DFL match: temp 4 picks switch 10 and loses 0.021 -> the temperature matters.
  Resume: `MTAH_DFL_LABELS=hackathon uv run python scripts/eval_mbr.py bench --temps 0.5 1 2 4 8`
  (~20 min), then `pff --temp <best>` and `real --temp <best>`.
- Then D, E (order unchanged).
# JEPA (Oct 28): worth a closer look?

Open question, nothing built. A JEPA learns without labels by predicting the representation of a
masked part of the input. Our limits are few labels (7 DFL matches, PFF blocked by its licence)
and the gap between the bench and the real tracking, so self-supervised pretraining could help
with both. No published work applies JEPA to football tracking yet.
- [ ] read further before building anything:
      T-JEPA / HiT-JEPA (trajectories), V-JEPA 2 / 2.1 (video, MIT code, needs a GPU in practice),
      PathCRF (ball-free possession path on the same 7 DFL matches, code public),
      "Multi-Modal Soccer Scene Analysis with Masked Pre-Training" (tactical-camera crops)
- [ ] best candidate: a small JEPA on tracking (DFL + Metrica + the real ATD, run locally), masking
      the ball and some players; its embeddings become extra M2 features, decoder unchanged
- [ ] go/no-go: linear probe of H/A/D on the embeddings, held out per DFL match; then paired
      comparison vs the submission at the same real-match change rate
- Constraints: CPU only on this machine; weights derived from the real match can't be committed,
  so pretraining would have to run inside the deterministic submit command (adds torch)
