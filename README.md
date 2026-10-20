# More than a Hack 2027: ball possession

Our solution to [More than a Hack 2027, Phase 1](../more-than-a-hack-2027/CHALLENGE.md): one
label per frame of the tactical video (`H` / `A` / `D` / `X`), ranked on spellF1.

- `docs/problem.md`: the problem, the metric's consequences and the main challenges. **Read first.**
- `docs/pipeline.html`: the whole pipeline explained in depth (theory, diagrams, interactive demos). Open it in a browser.
- `bench/`: offline test bench that replicates the hackathon conditions on public labelled matches.
- `possession/`: the methods.
  - `baseline.py`: the organisers' baseline (method 0, `smooth50`).
  - `hmm.py`: M1, hand-built evidence and a Viterbi decoder.
  - `learned.py`: M2, LightGBM evidence (`context.py` features) in the same decoder. This is the
    submission default.

| method | DFL held-out spellF1 | public window |
|---|---|---|
| smooth50 | 0.247 | 0.326 |
| hmm (M1) | 0.352 | 0.413 |
| learned (M2) | **0.522** | **0.622** |
| hsmm (M3, not adopted) | 0.463–0.517 | 0.622 |

## Setup

```bash
uv sync                      # Python 3.12, pinned via uv.lock
```

The organisers' starter repo must sit next to this one (`../more-than-a-hack-2027`), or set
`MTAH_STARTER_DIR`. Its scorer is imported as-is, so bench numbers are official-scorer numbers.

## The bench

```bash
uv run python -m bench fetch                 # 7 DFL matches + Metrica Sample Game 3 (~200 MB kept)
uv run python -m bench run --method smooth50 # board + "where the points go" over all DFL matches
uv run python -m bench run --method smooth50 all --worst      # + Metrica, + worst 10 s stretches
uv run python -m bench plot --run smooth50_hackathon_s0 --match dfl-DFL-MAT-J03WOH   # PNGs
uv run python -m bench stats all             # data stats as degraded (for calibration)
uv run python -m bench scenarios --preview   # edge-case pass rates + one MP4 replay per scenario
uv run python scripts/scenario_page.py       # -> reports/scenarios/index.html
uv run python scripts/tune_hmm.py            # staged tuning of possession/hmm.py
uv run python -m possession.learned train    # models/lgbm.txt (+ held-out models/lomo/, ~5 min)
uv run python scripts/tune_learned.py        # decoder penalties for the learned method
uv run python -m possession.submit           # official data -> submissions/AndaAnda.csv, validated
uv run --group dev pytest
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
