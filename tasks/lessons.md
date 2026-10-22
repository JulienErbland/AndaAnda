
## Waiting on background jobs (Oct 21)
- `while pgrep -f "<pattern>"` inside a shell whose own command line contains `<pattern>` waits
  forever: pgrep matches the waiting shell. Wait on a PID (`while kill -0 $PID`) or chain the
  steps with `&&` in one shell instead.

## Helper scripts belong in the repo (Oct 22)
- Ad-hoc evaluation scripts kept in the session scratchpad were gone the next session, and a
  chained job failed on them. Anything a later step depends on goes in `scripts/` (results in
  the gitignored `reports/`), not in the scratchpad.
