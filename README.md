# More than a Hack 2027: ball possession

Our solution to [More than a Hack 2027, Phase 1](../more-than-a-hack-2027/CHALLENGE.md): one
label per frame of the tactical video (`H` / `A` / `D` / `X`), ranked on spellF1.

- `docs/pipeline.md`: one-page summary of the data, the bench and the method.
- `docs/problem.md`: the problem, the metric's consequences and the main challenges. **Read first.**
- `docs/report.html`: the project report: main challenges, current state, where it fails, and the
  whole pipeline in depth (theory, diagrams, interactive demos). Open it in a browser.
- `docs/results.md`: log of every result obtained (kept up to date with each experiment).
- `bench/`: offline test bench that replicates the hackathon conditions on public labelled matches.
- `possession/`: the methods.
  - `baseline.py`: the organisers' baseline (method 0, `smooth50`).
  - `hmm.py`: M1, hand-built evidence and a Viterbi decoder.
  - `learned.py`: M2, LightGBM evidence (`context.py` + `touches.py` + `restarts.py` features)
    in the same decoder; a model dir may be an ensemble (`ensemble.json`). The submission
    default is `models/ensemble.json`: models/R (DFL labels, restart features) × 0.5 +
    models/B (hackathon-style v1 labels, touch features) × 0.5, switch 30; both trained on two
    degraded copies of each match, the dead ball moving in one and parked in the other.
  - `touches.py`: touch detector (control / deflection / contact) feeding M2.
  - `restarts.py`: restart detector (when, kind, team) and kick-off line-ups; a half is dead
    until its kick-off (the real second half is graded ~2 min before it).
  - `video.py`, `ballpatch.py`: ball detection in the tactical video (classical / learned
    patches); evaluated, not adopted.

Bench numbers on the current yardstick (DFL relabelled with the hackathon-style rule, realistic
degrader, held-out, 3 seeds), plus PFF World Cup 2022 as an evaluation-only bench (64 matches,
never trained on). The benches choose; the real match's change rate is only a range check (the
public window's 17 possessions put it anywhere in 2.5-6.8 per live minute). Full log:
`docs/results.md`.

| method | switch | bench spellF1 | PFF spellF1 | real rate | public window |
|---|---|---|---|---|---|
| smooth50 | – | 0.275 | – | 8.29 | 0.326 |
| hmm (M1) | 40 | 0.534 | – | 4.35 | 0.413 |
| M2 + touches (submission until Oct 27) | 10 | 0.611 | 0.479 | 4.28 | 0.622 |
| R_r + B_r ensemble | 3 | 0.667 | 0.468 | 4.18 | 0.622 |
| R_r + B_r ensemble | 20 | 0.724 | 0.514 | 3.45 | 0.622 |
| *bench corrected Oct 28 (moving dead ball); ball speed derived on the real match:* | | | | | |
| R_r + B_r ensemble | 20 | 0.567 | 0.514 | 3.71 | 0.444 |
| R + B, dead-ball-mixed training | 20 | 0.683 | 0.509 | 3.56 | 0.622 |
| *bench recalibrated Oct 28 (ball gaps: live 7% / dead 80%, ending at restarts):* | | | | | |
| R + B, dead-ball-mixed training | 20 | 0.710 | 0.509 | 3.56 | 0.622 |
| R + B retrained on it (until Oct 31) | 20 | 0.725 | 0.537 | 3.69 | 0.622 |
| *F5 bench Oct 31 (dead-ball knobs from the model's own calls): the row above scores 0.741* | | | | | |
| **R + B1 (v1 labels), 50/50 (current default)** | **30** | **0.759** | **0.522** | **3.44** | **0.496** |

## Setup

```bash
uv sync                      # Python 3.12, pinned via uv.lock
```

The organisers' starter repo must sit next to this one (`../more-than-a-hack-2027`), or set
`MTAH_STARTER_DIR`. Its scorer is imported as-is, so bench numbers are official-scorer numbers.

## The bench

```bash
uv run python -m bench fetch                 # 7 DFL matches + Metrica Sample Game 3 (~200 MB kept)
uv run python -m bench fetch dfl-events      # DFL official events, synced to the bench frames (1-4 f)
uv run python -m bench run --method smooth50 # board + "where the points go" over all DFL matches
uv run python -m bench run --method smooth50 all --worst      # + Metrica, + worst 10 s stretches
uv run python -m bench plot --run smooth50_hackathon_s0 --match dfl-DFL-MAT-J03WOH   # PNGs
uv run python -m bench stats all             # data stats as degraded (for calibration)
uv run python -m bench scenarios --preview   # edge-case pass rates + one MP4 replay per scenario
uv run python scripts/scenario_page.py       # -> reports/scenarios/index.html
uv run python scripts/tune_hmm.py            # staged tuning of possession/hmm.py
uv run python -m bench relabel               # hackathon-style labels -> external/dfl_hk/ (for B)
MTAH_MODEL_DIR=models/R uv run python -m possession.learned train --features restart --draws 2 --dead-mix
MTAH_DFL_LABELS=hackathon MTAH_MODEL_DIR=models/B uv run python -m possession.learned train --features touch --draws 2 --dead-mix
                                             # each: <dir>/lgbm.txt + held-out <dir>/lomo/
uv run python scripts/compare_models.py eval --out x.json   # bench + real rate per switch
uv run python scripts/validate_restarts.py   # restart detector vs DFL official set pieces
uv run python scripts/tune_learned.py        # decoder penalties for the learned method
uv run python -m possession.submit           # official data -> submissions/AndaAnda.csv, validated
uv run --group dev pytest                   # fast suite (+ xfail specs for the next directions)
uv run --group dev pytest -m slow           # bench acceptance gates on held-out DFL, v2 labels (~75 s)
```

**Matches.**
- **DFL** (Bassek et al. 2025, CC-BY 4.0) is the main benchmark: 7 Bundesliga / 2. Bundesliga
  matches with per-frame BallStatus + BallPossession, i.e. labels built mechanically from tracking,
  like the hackathon GT.
- **Metrica Sample Game 3** has the hackathon's exact file format (FIFA/EPTS). Its labels are
  derived from events, so they are approximate.

**Profiles** (`--profile`). Clean tracking is degraded to look like the hackathon ATD.
- `hackathon` (default), calibrated on the real ATD's aggregates:
  - ball missing ~41% in long gaps (median ~10 s), estimated at 19% of live frames and 61% of
    dead ones;
  - tracker blackouts that hide the ball and many players together;
  - anonymous slots with ~11 tracklets per minute and ~20.5 visible players;
  - wrong-team tracklets, duplicates and jitter.
- `ball-only`: the same ball dropout with perfect players.
- `clean`: no degradation.

Settings are in `bench/degrade.py:Profile`, seeded and deterministic.

**Diagnostics** (printed by `run`, full JSON in `reports/<tag>/`):
- recall and precision of possessions by duration bucket;
- why each missed possession was lost (merged / split / wrong team / absorbed by dead / boundary);
- transition lag;
- per-frame error rate by context (ball visible or missing, near a hand-over, live or dead);
- the worst 10 s stretches, which `plot` turns into timeline strips.

**Adding a method**: write `predict(match) -> (T,) int8 codes` (H=0, A=1, D=2, X=3) against the
common format in `bench/match.py`, then register it in `possession/__init__.py` or call it as
`--method possession.mymodule:predict`. Methods receive `match.without_labels()`.

## When the hackathon data arrives (Oct 20)

```bash
# unpack the bundle into ../more-than-a-hack-2027/data/ (or set MTAH_DATA_DIR), then:
uv run python -m bench build-hackathon       # -> data/hackathon-BAR-ATH.npz (gitignored)
uv run python -m bench stats hackathon       # ball-missing split alive/dead -> recalibrate Profile
uv run python -m bench run --method smooth50 hackathon   # public 5-minute window only
```

**Privacy.** The terms forbid giving match data (or extracts) to any cloud LLM. The hackathon
commands print aggregates only. Reports and plots for the hackathon match stay on your machine:
don't paste them into a chat. Nothing frame-level derived from it is committed (`data/`,
`reports/`, `*.npz` are gitignored).
