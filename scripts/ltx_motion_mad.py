"""_ltx_motion_mad.py -- BUG-LOCAL-112 motion measurement.

Sample 5 evenly-spaced frames from a video and compute mean absolute
pixel difference (MAD) between consecutive frames. Used to verify the
LTX motion fix:

    Real motion (dial sweep / dolly forward): sustained 15-30 MAD
    Subtle motion: 5-15 MAD
    Effectively static: <5 MAD

Usage:
    python scripts\\_ltx_motion_mad.py --mp4 path\\to\\clip.mp4
    python scripts\\_ltx_motion_mad.py --mp4 path\\to\\clip.mp4 --samples 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def probe_dur(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip()) if r.stdout.strip() else 0.0


def sample_frame(src: str, ts: float, out_path: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", src,
         "-frames:v", "1", out_path],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and os.path.exists(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mp4", required=True, help="Path to the video file")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of evenly-spaced frames to sample (default 5)")
    args = parser.parse_args()

    src = args.mp4
    if not os.path.exists(src):
        print(f"ERROR: file not found: {src}", flush=True)
        return 2

    dur = probe_dur(src)
    if dur <= 0.5:
        print(f"ERROR: clip too short to sample ({dur:.2f}s)", flush=True)
        return 3

    print(f"clip:     {src}", flush=True)
    print(f"duration: {dur:.2f}s", flush=True)

    n = max(2, args.samples)
    # Avoid the very start (i2v init image artifact) and very end
    # (encoder padding); spread within [10%, 90%].
    tss = [dur * (0.10 + (0.80 * i / (n - 1))) for i in range(n)]

    tmpdir = tempfile.mkdtemp(prefix="ltx_mad_")
    frames: list[tuple[float, np.ndarray]] = []
    for i, ts in enumerate(tss):
        out = os.path.join(tmpdir, f"f{i}.png")
        if not sample_frame(src, ts, out):
            print(f"  @ {ts:.2f}s: ffmpeg failed", flush=True)
            continue
        arr = np.array(Image.open(out).convert("RGB")).astype(np.int16)
        frames.append((ts, arr))

    if len(frames) < 2:
        print("ERROR: <2 frames sampled successfully", flush=True)
        return 4

    print(f"\n=== inter-frame MAD ({len(frames)} samples) ===", flush=True)
    mads = []
    prev_ts, prev = frames[0]
    for ts, arr in frames[1:]:
        mad = float(np.mean(np.abs(arr - prev)))
        mads.append(mad)
        verdict = (
            "STATIC" if mad < 2.0 else
            "subtle" if mad < 8.0 else
            "moving" if mad < 25.0 else
            "very dynamic"
        )
        print(f"  @ {prev_ts:5.2f}s -> {ts:5.2f}s  MAD={mad:6.2f}  ({verdict})",
              flush=True)
        prev_ts, prev = ts, arr

    avg = sum(mads) / len(mads)
    high = max(mads)
    low = min(mads)
    print(f"\nsummary: avg={avg:.2f}  range={low:.2f}-{high:.2f}", flush=True)
    if avg < 5.0:
        print("VERDICT: clip is effectively STATIC. BUG-112 fix did not unlock motion.",
              flush=True)
        return 1
    if avg < 15.0:
        print("VERDICT: clip has SUBTLE motion. Some improvement; may need further tuning.",
              flush=True)
        return 0
    print("VERDICT: clip is MOVING (real per-frame motion detected). BUG-112 fix unlocked motion.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
