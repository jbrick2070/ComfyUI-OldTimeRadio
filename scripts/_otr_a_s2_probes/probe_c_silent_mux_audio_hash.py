"""A-S2 probe (c) -- the silent mux is byte-identical (mux-LAST, -c:a copy, NO -shortest).

The whole audio invariant (V-1/V-2): only the terminal mux adds audio, it COPIES
the frozen master (never re-encodes), and ``-shortest`` is forbidden (it would
truncate to the silent video and drop audio tail). This probe muxes a frozen
master onto a silent clip with ``-c:a copy`` (no ``-shortest``) and asserts the
DECODED PCM of the muxed output is bit-for-bit equal to the master.

If OTR_MASTER_AUDIO / OTR_SILENT_CLIP are unset it synthesizes a 3 s sine master
+ a black silent clip with ffmpeg. Needs ffmpeg/ffprobe on PATH.

OPERATOR-RUN (needs ffmpeg). The agent did not run it.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

MASTER = os.getenv("OTR_MASTER_AUDIO", "").strip()
SILENT = os.getenv("OTR_SILENT_CLIP", "").strip()


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _pcm_sha(path):
    """sha256 of decoded little-endian s16 mono @24k -- codec-agnostic audio identity."""
    p = _run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a",
              "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", "-"])
    if p.returncode != 0:
        return None
    # ffmpeg writes binary PCM to stdout; re-run capturing bytes
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a", "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", "-"],
        capture_output=True)
    return hashlib.sha256(raw.stdout).hexdigest()


def main() -> int:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print("PROBE c: NO-GO: ffmpeg/ffprobe not on PATH")
        return 1
    tmp = tempfile.mkdtemp(prefix="otr_probe_c_")
    master = MASTER
    silent = SILENT
    if not master:
        master = os.path.join(tmp, "master.wav")
        _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
              "-i", "sine=frequency=440:duration=3", "-ar", "24000", "-ac", "1", master])
    if not silent:
        silent = os.path.join(tmp, "silent.mkv")
        _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
              "-i", "color=c=black:s=64x64:d=3", "-r", "25", "-an", silent])

    out = os.path.join(tmp, "muxed.mkv")
    # mux-LAST: copy the master audio, copy video, NO -shortest.
    mux_cmd = ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", master,
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", out]
    assert "-shortest" not in mux_cmd, "PROBE c: -shortest must never appear"
    p = _run(mux_cmd)
    if p.returncode != 0:
        print(f"PROBE c: NO-GO: mux failed :: {p.stderr.strip()[:300]}")
        return 1

    h_master = _pcm_sha(master)
    h_out = _pcm_sha(out)
    print(f"PROBE c: master_pcm={str(h_master)[:16]} muxed_pcm={str(h_out)[:16]} "
          f"(-shortest absent: True)")
    if h_master and h_out and h_master == h_out:
        print("PROBE c: PASS -- muxed audio PCM is byte-identical to the master (no re-encode, no truncation)")
        return 0
    print("PROBE c: NO-GO: muxed audio differs from the master (re-encode or truncation)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
