# The problem, precisely

More than a Hack 2027, Phase 1: label every frame of FC Barcelona vs Athletic Club (LaLiga 2024-25,
MD2) as `H` (Barcelona in possession, ball alive), `A` (Athletic), `D` (dead ball) or `X`
(abstain). Sources: the organisers' `CHALLENGE.md`, `EVALUATION.md`, `DATA.md` and
`evaluation/possession_metrics.py` (the metric spec), plus our own measurements, marked *measured*.

## 1. Output

- 199,319 rows, `frame,label`, frame = 0 … 199318 of the **tactical** video, 25 fps.
- Graded: 11,185–90,651 and 110,808–190,197 (158,857 frames). Anything else is ignored.
- Free points: 110,808–113,741 is graded and all `D` (the 2nd half kicks off at 113,742).
- Alignment: ATD `frame_id − 1`; Smart Tagging `round(seconds × 25)`; homography is already
  indexed by row. Always run `validate_submission.py` on the exact file.

## 2. What the ground truth means

- **Mechanical, from Metrica's *corrected* tracking**: `D` where that tracking flags the ball
  dead, otherwise the team it names as owner. No smoothing, no manual edits. The English
  definition ("controlled possession", "a deflection does not count") describes intent; the GT is
  the rule. When they disagree, the GT wins.
- Three states only. A loose or contested ball stays with the last team that had it.
- A possession is a **maximal run of H or of A**. `H H D D H H` is two possessions: every stoppage
  ends one.

The public 5 minutes (*measured*):

```
A6.6s H8.1s A2.8s H41.6s A7.4s D5.5s H28.6s A3.2s H42.4s A20.0s H0.8s A16.0s
H1.3s A3.6s D20.4s A35.7s H3.5s D18.3s A7.6s D14.5s H11.8s
```

17 possessions from 0.8 s to 42 s (5 of 17 under 3.6 s), 4 stoppages of 5–20 s (20% of frames).
The GT does count sub-second ownership (`A20.0 H0.8 A16.0`), so minimum-duration rules must be
short and heavy smoothing deletes real possessions.

## 3. How the scoring shapes the design

`spellF1`: predicted possessions are matched one-to-one (same team, greedy by temporal IoU) to
true ones, F1 averaged over IoU 0.3–0.7.

| mistake | cost |
|---|---|
| a short possession missed between two long same-team ones (`A20 H0.8 A16` → one A36) | ~2.6 of 3 possessions lost (A36 matches A20 only at IoU ≤ 0.54) |
| a long possession split in two | ~1.4 possessions (one half matches up to IoU 0.5, the other is a false positive) |
| short spells and timing | a 0.8 s spell is 20 frames: 5 frames off gives IoU 0.6, 8 frames off 0.43 |
| systematic lag | 3 f → 0.98, 8 f → 0.89, 1 s → 0.65 |
| `X` | forward-filled with the previous label: free inside a possession, costly across a hand-over |
| noise | full-match gaps below ~0.04 spellF1 are noise; on the public window one spell ≈ 6 points |

macroF1 (tie-break) is per frame and nearly blind to all of the above.

## 4. Main challenges (ordered by points at stake, our judgement)

1. **Fragmentation vs deleting real short possessions**, the central trade-off. Merges cost more
   than splits.
2. **Control is not proximity**: passes in flight, duels and deflections fool nearest-player rules,
   yet the GT does count brief real ownership.
3. **Ball missing on 41% of graded frames**, not at random (crowds, aerial duels, tackles), which
   are exactly the hand-over moments.
4. **Dead-ball detection**: a large share of frames, both edges must be timed, and every stoppage
   splits possessions. Fouls aren't visible from movement alone. The restarting team owns the next
   possession.
5. **Transition timing**: 0.5–1 s of lag costs a lot; smoothing windows add lag.
6. **Noisy players**: 46 slots for ~20.5 visible players, no IDs, swaps, duplicates. Only team
   membership per slot is usable, and it can be wrong.
7. **Geometry**: 0–1 coordinates (scale by 105.3 × 68 m), ends swap at half-time.
8. **Noisy priors**: Smart Tagging phases are not possessions; `Team = N/A` on all throw-ins;
   `Atletic Club` is misspelt; tags are automatic.
9. **Almost no validation data**: 17 public possessions, which pick the wrong hyperparameter even for
   the baseline. Hence the offline bench (see README).
10. **Constraints**: no match data may reach a cloud LLM; labels from an algorithm only; the
    public repo must regenerate the exact file with one command (deterministic).

## 5. What the bench taught us (*measured*, Oct 19)

- **DFL labels change possession about twice as often as the hackathon GT**: 8.4 vs 4.3 spells per
  minute of live ball. Merging sub-2 s same-team interruptions only brings DFL to 6.4, so it isn't
  just touches. Likely part definition, part playing style (Barça keep the ball), part the small
  sample (17 spells). **Don't tune a possession-switch rate on DFL.** Tune things that should
  transfer (evidence thresholds, dead-ball rules) and check counts on the public window.
- **In Metrica's sample data, a missing ball means a dead ball**: missing on 4% of live frames and
  96% of dead ones. Undegraded, it leaks the D label, so always score Metrica degraded. Whether the
  hackathon ATD behaves the same way is the first thing to measure (`bench stats hackathon`).
- **The real ATD loses the ball in few, LONG gaps** (Oct 20, `bench stats hackathon`, aggregates):
  150 gaps over the match, median 10.7 s, p90 34 s, longest 221 s. So possession changes often
  happen with no ball at all: player movement has to carry the evidence.
- **Tracker blackouts**: the ball and many players vanish together. On the 11% of frames with 18 or
  fewer players tracked, the ball is missing 85% of the time. Independent single-player dropouts are
  rare: there are 11.5 tracklets per minute, median 35 s.
- **Estimated full-match split: ball missing on ~19% of live frames and ~61% of dead ones.** The
  public window (5 min) says 6.5% / 34%, but the full-match rate is 40.7%. We inferred the split from
  missing rates per mean-player-speed bin, weighted by DFL's dead share per bin (the two matches
  have near-identical speed distributions). Walking pace (<1.5 m/s) is mostly dead ball in both.
- **The recalibrated bench reproduces the real board**. Baseline spellF1 averaged over the 7 DFL
  matches: raw 0.037, smooth25 0.240, **smooth50 0.247**, smooth125 0.191. The real full match
  gives 0.049, 0.200, **0.243** and 0.224.
- On the real public window, the bench reproduces the published baseline scores exactly (0.0378 /
  0.2868 / 0.3256 / 0.4242), so loading, alignment and scoring match the organisers.
