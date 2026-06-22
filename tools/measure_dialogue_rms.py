#!/usr/bin/env python3
r"""Measure active-window RMS loudness (dBFS) of a dialogue WAV.

Calibration helper for the baked-in per-segment RMS leveling
(nodes/scene_sequencer.py). Run it on a real dialogue stem to confirm /
refine OTR_SEGMENT_TARGET_RMS_DBFS (default -16.0 dBFS). Uses the SAME
active-sample gate as production so the number is apples-to-apples.

Usage:
  python tools/measure_dialogue_rms.py <file.wav> [more.wav ...] [--gate -50]

Stdlib + numpy only (no scipy / soundfile dependency).
"""

import argparse
import sys
import wave

import numpy as np


def _read_wav_mono_float(path):
    """Read a PCM WAV as a mono float64 array in [-1, 1] (stdlib `wave`)."""
    with wave.open(path, "rb") as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        raise ValueError(f"{path}: unsupported sample width {sw*8}-bit")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sw == 1:                      # 8-bit PCM is unsigned, centered at 128
        data = (data - 128.0) / 128.0
    else:
        data = data / float(2 ** (8 * sw - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)   # channel-average to mono
    return data


def active_rms_dbfs(x, gate_dbfs=-50.0):
    """Active-window RMS in dBFS (mirrors _loudness_normalize_clip)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return None
    gate_amp = 10.0 ** (gate_dbfs / 20.0)
    active = x[np.abs(x) >= gate_amp]
    use = active if active.size else x
    rms = float(np.sqrt(np.mean(np.square(use))))
    if rms <= 0.0:
        return None
    return 20.0 * np.log10(max(rms, 1e-10))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wavs", nargs="+", help="WAV file(s) to measure")
    ap.add_argument("--gate", type=float, default=-50.0, help="noise-floor gate dBFS")
    args = ap.parse_args()

    vals = []
    for p in args.wavs:
        try:
            d = active_rms_dbfs(_read_wav_mono_float(p), args.gate)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {p}: {e}")
            continue
        if d is None:
            print(f"  {p}: silent / below gate")
            continue
        vals.append(d)
        print(f"  {p}: {d:+.1f} dBFS active RMS")

    if vals:
        mean = sum(vals) / len(vals)
        print(f"\nmean active RMS over {len(vals)} file(s): {mean:+.1f} dBFS")
        print(f"suggested OTR_SEGMENT_TARGET_RMS_DBFS = {mean:+.1f}  "
              f"(set to preserve current loudness; default is -16.0)")
    else:
        print("\nno measurable dialogue found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
