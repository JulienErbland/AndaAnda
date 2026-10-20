"""The one in-memory shape every source is converted to and every method reads.

It mirrors the hackathon tactical ATD, so a method written against it runs unchanged on the real
data:

- coordinates normalised 0-1 along (length, width), origin at a corner, raw direction of play
  (teams swap ends at half-time, nothing is flipped);
- 25 fps, one row per video frame;
- players are anonymous *slots* with a team only, NaN where a slot is not tracked;
- the ball is NaN where it is not tracked;
- labels use the organisers' codes (H=0, A=1, D=2, X=3), X where there is no ground truth.
"""

from dataclasses import dataclass, replace

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25


@dataclass
class Match:
    name: str
    pitch: tuple            # (length, width) in metres
    ball: np.ndarray        # (T, 3) float32: x, y (0-1), speed (m/s)
    players: np.ndarray     # (T, S, 3) float32: x, y (0-1), speed (m/s)
    slot_team: np.ndarray   # (S,) int8: 0 home, 1 away
    halves: np.ndarray      # (n, 3) int64: half, first frame, last frame (inclusive)
    labels: np.ndarray = None  # (T,) int8 organiser codes, None when unknown

    @property
    def n_frames(self):
        return len(self.ball)

    def without_labels(self):
        """What a method gets to see: everything but the answer."""
        return replace(self, labels=None)

    def save(self, path):
        arrays = dict(ball=self.ball, players=self.players, slot_team=self.slot_team,
                      halves=self.halves, pitch=np.asarray(self.pitch, dtype=np.float64),
                      name=np.asarray(self.name))
        if self.labels is not None:
            arrays["labels"] = self.labels
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(name=str(z["name"]), pitch=tuple(z["pitch"]), ball=z["ball"],
                       players=z["players"], slot_team=z["slot_team"], halves=z["halves"],
                       labels=z["labels"] if "labels" in z else None)


def graded_mask(match):
    """True on frames inside a graded half block."""
    mask = np.zeros(match.n_frames, dtype=bool)
    for _, first, last in match.halves:
        mask[first:last + 1] = True
    return mask
