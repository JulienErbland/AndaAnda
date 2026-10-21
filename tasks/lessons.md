
## Waiting on background jobs (Oct 21)
- `while pgrep -f "<pattern>"` inside a shell whose own command line contains `<pattern>` waits
  forever: pgrep matches the waiting shell. Wait on a PID (`while kill -0 $PID`) or chain the
  steps with `&&` in one shell instead.
