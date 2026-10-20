"""Synthetic match builder for behaviour tests: write situations in metres and seconds.

    t = Toy(10)                      # 10 s, 22 players at kick-off positions, standing
    t.jog(3.0)                       # everyone runs small circles at 3 m/s (centroids stay put)
    t.ball_still(0, 5, (52.5, 34))   # ball on the centre spot for 5 s
    t.ball_move(5, 6, (52.5, 34), (40, 34))
    t.put(9, 0, 5, (52.0, 34.6))     # home slot 9 stands next to the ball
    m = t.match()

Home = slots 0-10 in the left half (attacking +x), away = 11-21 in the right half. Speeds in the
tracking's speed channel are derived from the final positions, as the real ATD would report them.
"""

import numpy as np

from bench.match import FPS, X, Match

PITCH = (105.0, 68.0)
HOME, AWAY = range(11), range(11, 22)


class Toy:
    def __init__(self, seconds):
        self.n = int(round(seconds * FPS))
        home = [(12 + 3.5 * i, 6 + 5.5 * i) for i in range(11)]
        away = [(93 - 3.5 * i, 6 + 5.5 * i) for i in range(11)]
        self.xy = np.repeat(np.array(home + away, dtype=float)[None], self.n, 0)
        self.ball = np.full((self.n, 2), np.nan)
        self.labels = np.full(self.n, X, dtype=np.int8)

    def _f(self, s):
        return int(round(s * FPS))

    def jog(self, speed, t0=0, t1=None, radius=3.0, slots=range(22)):
        """Circular running at `speed` m/s; random phases keep team centroids nearly fixed."""
        a, b = self._f(t0), self._f(t1) if t1 is not None else self.n
        t = np.arange(b - a) / FPS
        phase = np.random.default_rng(0).uniform(0, 2 * np.pi, 22)
        w = speed / radius
        for s in slots:
            self.xy[a:b, s, 0] += radius * (np.cos(w * t + phase[s]) - np.cos(phase[s]))
            self.xy[a:b, s, 1] += radius * (np.sin(w * t + phase[s]) - np.sin(phase[s]))
        return self

    def drift(self, team, t0, t1, vx):
        """Move a whole team along x at vx m/s from t0 to t1 (and stay shifted after)."""
        a, b = self._f(t0), self._f(t1)
        shift = np.concatenate([np.arange(b - a) / FPS * vx, np.full(self.n - b, (b - a) / FPS * vx)])
        for s in (HOME if team == 0 else AWAY):
            self.xy[a:, s, 0] += shift
        return self

    def put(self, slot, t0, t1, p0, p1=None):
        a, b = self._f(t0), self._f(t1)
        p1 = p0 if p1 is None else p1
        f = np.linspace(0, 1, b - a)[:, None]
        self.xy[a:b, slot] = np.asarray(p0) * (1 - f) + np.asarray(p1) * f
        return self

    def follow(self, slot, t0, t1, offset=(0.4, 0.0)):
        """Slot stays next to the ball (controlling it) from t0 to t1."""
        a, b = self._f(t0), self._f(t1)
        self.xy[a:b, slot] = self.ball[a:b] + np.asarray(offset)
        return self

    def ball_still(self, t0, t1, p):
        self.ball[self._f(t0):self._f(t1)] = p
        return self

    def ball_move(self, t0, t1, p0, p1):
        a, b = self._f(t0), self._f(t1)
        f = np.linspace(0, 1, b - a, endpoint=False)[:, None]
        self.ball[a:b] = np.asarray(p0) * (1 - f) + np.asarray(p1) * f
        return self

    def ball_lost(self, t0, t1):
        self.ball[self._f(t0):self._f(t1)] = np.nan
        return self

    def match(self):
        def speed(xy):
            v = np.gradient(xy, axis=0) * FPS
            return np.sqrt((v ** 2).sum(-1))
        pitch = np.asarray(PITCH)
        players = np.concatenate([self.xy / pitch, speed(self.xy)[..., None]], -1)
        ball = np.concatenate([self.ball / pitch, speed(self.ball)[:, None]], -1)
        return Match(name="toy", pitch=PITCH, ball=ball.astype(np.float32),
                     players=players.astype(np.float32),
                     slot_team=np.repeat([0, 1], 11).astype(np.int8),
                     halves=np.array([[1, 0, self.n - 1]]), labels=self.labels.copy())
