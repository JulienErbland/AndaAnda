"""Ball detection on synthetic frames (no real video): a moving white disc on grass is found,
a static white line is not, and pixels map back to the pitch through the homography."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from possession import video  # noqa: E402

PITCH = (105.0, 68.0)
H = np.array([[1800.0, 0, 60], [0, 900.0, 90], [0, 0, 1]])   # pitch 0-1 -> 1920x1080-ish


def _frame(ball_px=None):
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:] = (40, 120, 40)                                      # grass (BGR)
    cv2.line(img, (300, 0), (300, 1079), (235, 235, 235), 3)   # a static white line
    if ball_px is not None:
        cv2.circle(img, (int(ball_px[0]), int(ball_px[1])), 3, (245, 245, 245), -1)
    return img


def test_a_moving_ball_is_found_and_mapped_to_the_pitch():
    true_xy = np.array([0.5, 0.5])
    p = video.project(H, true_xy[None])[0]
    before, now, after = _frame(p - [30, 0]), _frame(p), _frame(p + [30, 0])
    diam = video.ball_px(H, true_xy, PITCH)
    c = video.candidates(now, before, after, p + [15, 10], 80, diam)
    assert c, "the ball should be found"
    x, y, _ = max(c, key=lambda k: k[2])
    back = video.unproject(H, np.array([[x, y]]))[0]
    assert np.hypot(*((back - true_xy) * PITCH)) < 0.2


def test_a_static_white_line_is_not_a_ball():
    img = _frame()
    assert video.candidates(img, img, img, (300, 500), 100, 6.0) == []


def test_linking_drops_isolated_blobs_and_keeps_tracks():
    track = {t: [(0.5 + 0.001 * t, 0.5, 0.9)] for t in range(10, 15)}
    track[40] = [(0.1, 0.1, 0.9)]                               # isolated: no neighbour
    out = video.link(track, PITCH)
    assert set(out) == set(range(10, 15))


def test_search_decodes_a_video_and_tracks_the_ball(tmp_path):
    n, w, h = 40, 640, 360
    Hs = np.array([[1800.0, 0, -700], [0, 900.0, -270], [0, 0, 1]])   # real-video scale
    path = str(tmp_path / "toy.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    xs = np.linspace(0.42, 0.58, n)                             # ~0.4 m per frame
    for t in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (40, 120, 40)
        px = video.project(Hs, np.array([[xs[t], 0.5]]))[0]
        cv2.circle(img, (int(px[0]), int(px[1])), 2, (250, 250, 250), -1)
        out.write(img)
    out.release()
    frames = np.arange(5, 35)
    prior = np.c_[xs[frames] + 0.01, np.full(len(frames), 0.52)]   # a slightly-off prior
    found = video.link(video.search(path, np.repeat(Hs[None], n, 0), frames, prior,
                                    np.full(len(frames), 5.0), PITCH), PITCH)
    assert len(found) >= 0.8 * len(frames)
    err = [np.hypot((found[f][0] - xs[f]) * PITCH[0], (found[f][1] - 0.5) * PITCH[1]) for f in found]
    assert np.median(err) < 0.5


def test_apply_fills_gaps_only(tmp_path):
    from bench.match import Match
    n = 10
    ball = np.full((n, 3), np.nan, np.float32)
    ball[:5] = [0.3, 0.3, 1.0]
    m = Match(name="toy", pitch=PITCH, ball=ball, players=np.zeros((n, 2, 3), np.float32),
              slot_team=np.array([0, 1], np.int8), halves=np.array([[1, 0, n - 1]]))
    cache = str(tmp_path / "vb.npz")
    np.savez(cache, frame=np.array([2, 6, 7]), x=np.array([0.9, 0.6, 0.6]),
             y=np.array([0.9, 0.5, 0.5]), conf=np.array([0.9, 0.9, 0.05]))
    out = video.apply(m, cache)
    assert out.ball[2, 0] == np.float32(0.3)         # tracked ball never overwritten
    assert out.ball[6, 0] == np.float32(0.6)         # confident detection fills the gap
    assert np.isnan(out.ball[7, 0])                  # low confidence: left missing


def test_runs_around_includes_the_neighbours_link_needs():
    r = video._runs_around(np.array([10, 11, 50]), 100)
    assert r.tolist() == [8, 9, 10, 11, 12, 13, 48, 49, 50, 51, 52]
