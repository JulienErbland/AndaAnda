
## Waiting on background jobs (Oct 21)
- `while pgrep -f "<pattern>"` inside a shell whose own command line contains `<pattern>` waits
  forever: pgrep matches the waiting shell. Wait on a PID (`while kill -0 $PID`) or chain the
  steps with `&&` in one shell instead.

## Helper scripts belong in the repo (Oct 22)
- Ad-hoc evaluation scripts kept in the session scratchpad were gone the next session, and a
  chained job failed on them. Anything a later step depends on goes in `scripts/` (results in
  the gitignored `reports/`), not in the scratchpad.

## Never train into models/ by accident (Oct 22)
- A chained experiment command had one line without MTAH_MODEL_DIR and started training into
  models/ (the committed submission model). Caught before it saved. Now `possession.learned
  train` refuses experiment options without MTAH_MODEL_DIR. When chaining experiment commands,
  export MTAH_MODEL_DIR per step and never leave a bare `train` line in the chain.

## pkill matches its own shell too (Oct 27)
- `pkill -f "<pattern>"` killed the Bash call running it (exit 144): its command line contains the
  pattern. Same trap as pgrep. Use a bracket pattern (`pkill -f "[c]ompare_models"`) or PIDs.

## "No overlap" filters must compare with the last KEPT item (Oct 27)
- Sorting random picks and dropping any closer than k to the previous PICK threw away ~80% of
  them (69 training frames instead of 2,500) and produced a misleading NO-GO. Filter greedily
  against the last kept item, and print the sample sizes before reading any result.

## Gates must follow the yardstick (Oct 27)
- The slow gates still scored DFL's original labels five days after the project moved its
  yardstick to the hackathon-style labels, so they failed a model that is better on the
  yardstick. When the yardstick changes, re-base the gates in the same change.

## A bracket pattern does not protect a waiting loop that names the script it will run (Oct 27)
- `while pgrep -f "[r]ound4_metrica.sh"; do sleep; done; scripts/round4_metrica.sh ...` waited
  forever: the loop's own `bash -c` command line contains the literal `round4_metrica.sh` (in the
  command it runs next), and the bracket only hides the pattern string itself. Wait on a PID
  (`while kill -0 $PID`) or on a file the job writes when it finishes.
