"""Score replays of one frozen episode against each other, beat by beat.

A replay (``scripts/otr_canonical_api_run.py --replay-from <bundle>``) renders
the video phase again from the SAME stills, prompts and seeds, so the opening
segment of every beat is a paired comparison between two recipes. This is the
measurement that caught recipe v4 of the 8 GB LTX lane (PBUG-20260928-05): it
matched the vendor's canonical graph and still made the picture cut away from
its still within half a second. A recipe change to a video lane is scored here
before it ships.

Per arm it prints:

* ADHERENCE -- the mean abs RGB distance of frames 12, 24 and 48 from the
  beat's still (both at 512x288), and how many beats passed 40 at frame 24,
  which is a picture that has left the still's composition: a cut or a new
  scene. Measured 2026-09-28: v3 17.0, v4 30.1 at frame 24 on the same beats.
* DETAIL -- the variance of the Laplacian at those frames.
* PULSE -- every 8th frame's detail against the mean of its two neighbours.
  A tiled VAE decode's seam shows as a dip below 1 (0.72 at a 16-frame tile,
  PBUG-20260928-04).
* JOINS -- with ``--segment-frames N`` (the lane's chained segment length, 161
  for ltx_8gb, whose successors drop their first frame), the frame change at
  each chain join against the four frames either side, and the signed
  brightness step across it net of the local drift (-2.08 before
  PBUG-20260928-03, -0.28 after).

The first ``--arm`` is the baseline the others are counted against.

    python scripts/otr_replay_ab.py --bundle <output>/otr/episodes/_replay/<episode> ^
        --arm v3=<output>/otr/episodes/<original> ^
        --arm v5=<output>/otr/episodes/<replay> --segment-frames 161

Read-only: it opens clips and stills and writes nothing.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics as st
import sys

import av
import numpy as np
from PIL import Image

#: The frames scored against the still: inside the first half second, where a
#: cut away from the still shows, and at two seconds.
PICKS = (12, 24, 48)
#: Mean abs RGB distance past which a frame no longer shows its still.
LEFT_THE_STILL = 40.0
#: The size stills and frames are compared at.
COMPARE_SIZE = (512, 288)
_STILL_NAME = re.compile(r"^still_(?P<beat>.+)_[0-9a-f]{12}\.png$")


def still_beats(stills_dir):
    """``{beat_id: still path}`` from a bundle's ``still_<beat>_<hash>.png``."""
    out = {}
    for name in sorted(os.listdir(stills_dir)):
        m = _STILL_NAME.match(name)
        if m:
            out[m.group("beat")] = os.path.join(stills_dir, name)
    return out


def clip_for(clips_dir, beat):
    """The one clip ``shot_<beat>_<role>_<engine>.mp4`` in ``clips_dir``, or None.

    The underscore after the beat keeps ``b1`` from matching ``b10``."""
    hits = [n for n in os.listdir(clips_dir)
            if n.startswith("shot_%s_" % beat) and n.endswith(".mp4")]
    return os.path.join(clips_dir, hits[0]) if len(hits) == 1 else None


def decode_frames(path):
    """Every frame of a clip as ``(rgb, luma)`` uint8 arrays: RGB for the
    distance from the still, the luma plane for detail and brightness, which
    is what the eye tracks. One decode, two conversions."""
    with av.open(path) as box:
        return [(f.to_ndarray(format="rgb24"), f.to_ndarray(format="gray"))
                for f in box.decode(video=0)]


def detail(luma):
    """Variance of the 4-neighbour Laplacian of a luma plane."""
    g = luma.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def _resized(rgb):
    if (rgb.shape[1], rgb.shape[0]) == COMPARE_SIZE:
        return rgb.astype(np.int16)
    return np.asarray(Image.fromarray(rgb).resize(COMPARE_SIZE, Image.LANCZOS)).astype(np.int16)


def score_clip(frames, still_rgb, segment_frames=None):
    """One clip's numbers from ``decode_frames`` output. ``still_rgb`` is
    already at ``COMPARE_SIZE``."""
    n = len(frames)
    out = {"frames": n, "distance": {}, "detail": {}}
    for p in PICKS:
        if p < n:
            out["distance"][p] = float(np.abs(_resized(frames[p][0]) - still_rgb).mean())
            out["detail"][p] = detail(frames[p][1])
    lap = [detail(f[1]) for f in frames]
    pulse = [lap[i] / ((lap[i - 1] + lap[i + 1]) / 2)
             for i in range(8, n - 1, 8) if lap[i - 1] + lap[i + 1] > 0]
    out["pulse"] = st.mean(pulse) if pulse else None
    out["joins"], out["steps"] = [], []
    if segment_frames:
        grey = [f[1].astype(np.float32) for f in frames]
        luma = [float(g.mean()) for g in grey]
        diffs = [float(np.abs(grey[i] - grey[i - 1]).mean()) for i in range(1, n)]
        # A successor drops its first frame, so joins fall every
        # (segment_frames - 1) frames after the first full segment.
        for j in range(segment_frames, n, segment_frames - 1):
            if j + 4 >= n or j - 5 < 0:
                continue
            around = diffs[j - 5:j - 1] + diffs[j:j + 4]
            out["joins"].append(diffs[j - 1] / max(st.mean(around), 0.05))
            drift = st.mean([luma[i] - luma[i - 1]
                             for i in list(range(j - 4, j)) + list(range(j + 1, j + 5))])
            out["steps"].append((luma[j] - luma[j - 1]) - drift)
    return out


def score_arms(stills_dir, arms, segment_frames=None):
    """``{label: {beat: score}}`` over the beats every arm rendered."""
    beats = still_beats(stills_dir)
    scores = {label: {} for label, _ in arms}
    for beat, still_path in beats.items():
        paths = [(label, clip_for(os.path.join(d, "clips"), beat)) for label, d in arms]
        if any(p is None for _, p in paths):
            continue
        still = _resized(np.asarray(Image.open(still_path).convert("RGB")))
        for label, path in paths:
            scores[label][beat] = score_clip(decode_frames(path), still, segment_frames)
    return scores


def _median(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else float("nan")


def report(scores, out=sys.stdout):
    labels = list(scores)
    common = sorted(set.intersection(*(set(s) for s in scores.values()))) if scores else []
    common = [b for b in common if all(p in scores[l][b]["distance"] for l in labels for p in PICKS)]
    out.write("beats compared: %d\n" % len(common))
    if not common:
        return
    out.write("%-12s %s  %s  %s  %s  %s\n" % (
        "arm", "  ".join("dist f%-3d" % p for p in PICKS), "left@f24",
        "  ".join("detail f%-3d" % p for p in PICKS), " pulse", "joins (median, >=3x) / luma step"))
    for label in labels:
        s = scores[label]
        dist = "  ".join("%8.1f" % _median(s[b]["distance"][p] for b in common) for p in PICKS)
        left = sum(1 for b in common if s[b]["distance"][24] > LEFT_THE_STILL)
        det = "  ".join("%10.0f" % _median(s[b]["detail"][p] for b in common) for p in PICKS)
        pulse = _median(s[b]["pulse"] for b in common)
        joins = [j for b in common for j in s[b]["joins"]]
        steps = [x for b in common for x in s[b]["steps"]]
        tail = ("%.2fx, %d of %d / %+.2f" % (st.median(joins), sum(1 for j in joins if j >= 3),
                                             len(joins), st.median(steps))) if joins else "-"
        out.write("%-12s %s  %4d/%-3d  %s  %6.2f  %s\n" % (
            label, dist, left, len(common), det, pulse, tail))
    base = labels[0]
    for label in labels[1:]:
        further = sum(1 for b in common
                      if scores[label][b]["distance"][24] > scores[base][b]["distance"][24])
        out.write("  %s sits further from the still than %s at frame 24 in %d of %d beats\n"
                  % (label, base, further, len(common)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True,
                    help="the frozen replay bundle (its stills/ folder is read)")
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=EPISODE_DIR",
                    help="an episode folder holding clips/; repeat, the first is the baseline")
    ap.add_argument("--segment-frames", type=int, default=None,
                    help="the lane's chained segment length (161 for ltx_8gb) to score joins")
    args = ap.parse_args(argv)
    arms = []
    for spec in args.arm:
        label, sep, path = spec.partition("=")
        if not sep or not os.path.isdir(os.path.join(path, "clips")):
            ap.error("--arm wants LABEL=EPISODE_DIR with a clips/ folder, got %r" % spec)
        arms.append((label, path))
    report(score_arms(os.path.join(args.bundle, "stills"), arms, args.segment_frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
