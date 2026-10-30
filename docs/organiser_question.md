# Draft issue for the organisers' repo (F7). Not posted; Julien posts it.

**Title:** Two clarifications: label-definition calibration, and when a restart becomes live

Hi, two questions on RULES.md / CHALLENGE.md after the Oct 30 clarifications.

**1. Matching a training-label definition to the public window's aggregates.**
We train on other games (open DFL tracking) whose possession labels follow a different definition
from yours, so we relabel them with a mechanical rule written from CHALLENGE.md's text ("loose or
contested ball: possession stays with the last team that had it…"). Is it acceptable to check that
rule's *aggregate* output (possessions per minute, duration quantiles) against the 5 public minutes,
or does that count as "tuning on the public window"? We no longer do it, and our rule's constants
were set before we looked, but we'd like to describe this correctly in the tech doc.

**2. When does a restart stop being `D`?**
After a throw-in, free kick, corner, goal kick or kick-off, is the ball `D` until the kick/throw
itself, or until another player first receives it? Our training sources disagree (DFL and PFF go
live at the kick), and the difference is 1–1.5 s per restart, ~100 restarts per match. The
question is about the definition, not about specific frames.

Thanks!
