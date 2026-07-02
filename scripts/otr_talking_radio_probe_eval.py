"""Talking-radio probe evaluator -- the PROGRAMMATIC AID for the Sub-plan C
written criterion (docs/2026-07-01-talking-radio/EYEBALL.md).

For each probe obs mp4 it computes, over the OPEN bookend segment:
  * per-frame MOTION ENERGY in the mouth region (lower-central box of the
    frame, frame-differenced, gray) -- the grille-mouth articulation signal;
  * the AUDIO ONSET-STRENGTH envelope (mono RMS per video-frame window,
    half-wave-rectified positive difference) -- the speech/music transients;
  * their Pearson correlation r, maximised over a small +-3-frame lag (mux
    offset tolerance).

Criterion (pre-registered): face1 r1 >= 0.35 AND r1 - r0 >= 0.15, with the
face0 leg as the ambient control. This script only REPORTS numbers; the
GO/NO-GO verdict is the operator's eyeball. UTF-8, no BOM, ASCII-only.

Usage:
  python scripts/_otr_talking_radio_probe_eval.py <face0.mp4> <face1.mp4>
      [--start 0] [--dur 14] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def _probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (st.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den or 1)
    return fps, int(st["width"]), int(st["height"])


def _mouth_motion(path, start, dur, fps):
    """Per-frame motion energy inside the mouth box: central 60% width,
    35%..95% height (the grille-mouth of a face-forward radio still). Decoded
    small (160x96 gray) -- articulation is a coarse signal."""
    w, h = 160, 96
    vf = ("crop=iw*0.6:ih*0.6:(iw-ow)/2:ih*0.35,scale=%d:%d,format=gray"
          % (w, h))
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path,
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    if n < int(3 * fps):
        raise RuntimeError("too few frames decoded from %s (n=%d)" % (path, n))
    frames = np.frombuffer(raw[: n * w * h], dtype=np.uint8)
    frames = frames.reshape(n, h, w).astype(np.float32)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return d          # length n-1, one value per frame transition


def _audio_onsets(path, start, dur, fps, nframes):
    sr = 16000
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path,
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"],
        capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size < sr:
        raise RuntimeError("too little audio decoded from %s" % path)
    win = int(round(sr / fps))
    m = min(nframes + 1, x.size // win)
    rms = np.sqrt(np.mean(
        x[: m * win].reshape(m, win) ** 2, axis=1) + 1e-12)
    onset = np.maximum(np.diff(rms), 0.0)     # positive RMS change = transient
    return onset       # length m-1, aligned to frame transitions


def _pearson(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.mean(a * b))


def _best_lag_r(motion, onset, max_lag=3):
    n = min(motion.size, onset.size)
    motion, onset = motion[:n], onset[:n]
    best, best_lag = -math.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = motion[lag:], onset[: n - lag]
        else:
            a, b = motion[: n + lag], onset[-lag:]
        if a.size < 10:
            continue
        r = _pearson(a, b)
        if r > best:
            best, best_lag = r, lag
    return best, best_lag


def evaluate(path, start, dur):
    fps, w, h = _probe(path)
    motion = _mouth_motion(path, start, dur, fps)
    onset = _audio_onsets(path, start, dur, fps, motion.size)
    r, lag = _best_lag_r(motion, onset)
    return {"mp4": path, "fps": round(fps, 3), "src_w": w, "src_h": h,
            "segment": [start, dur], "frames_analyzed": int(motion.size),
            "mouth_motion_mean": round(float(motion.mean()), 4),
            "r_mouth_vs_onset": round(r, 4), "best_lag_frames": lag}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("face0")
    ap.add_argument("face1")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=14.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    r0 = evaluate(args.face0, args.start, args.dur)
    r1 = evaluate(args.face1, args.start, args.dur)
    delta = r1["r_mouth_vs_onset"] - r0["r_mouth_vs_onset"]
    crit = (r1["r_mouth_vs_onset"] >= 0.35) and (delta >= 0.15)
    out = {"face0": r0, "face1": r1, "r_delta": round(delta, 4),
           "criterion_r1_ge_0.35_and_delta_ge_0.15": bool(crit),
           "note": ("numbers are a VERDICT AID; GO/NO-GO is the operator "
                    "eyeball per EYEBALL.md")}
    print(json.dumps(out, indent=1, ensure_ascii=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
