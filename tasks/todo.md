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
