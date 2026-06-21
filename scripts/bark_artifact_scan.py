"""Bark artifact QA scan -- flag high-pitched non-speech squeals in rendered audio.

Runs the deterministic high-band edge gate (nodes/_otr_bark_lib.high_band_edge_ratio)
over a WAV file or a directory of WAVs and prints the per-clip head/tail high-band
(>4 kHz) energy ratio + a FLAG when it meets the threshold. Use it for the bark
artifact before/after re-soak: render a few bark-forced legs with the legacy
behavior (OTR_BARK_SPEECH_ONLY=0 OTR_BARK_DISABLE_THROAT_CLEAR=0) and with the B1
defaults, then compare the flagged counts.

Usage:
  python scripts/bark_artifact_scan.py <file.wav | dir> [threshold]

Pure-stdlib + numpy + soundfile; no torch, no GPU. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scan_one(path, threshold):
    import numpy as np  # noqa: F401
    import soundfile as sf

    from nodes._otr_bark_lib import flag_high_band_artifact

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    flagged, ratio = flag_high_band_artifact(audio, sr, threshold=threshold)
    return flagged, ratio, len(audio) / sr if sr else 0.0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    target = Path(argv[0])
    threshold = float(argv[1]) if len(argv) > 1 else 0.5

    if target.is_dir():
        wavs = sorted(target.rglob("*.wav"))
    elif target.is_file():
        wavs = [target]
    else:
        print(f"not found: {target}")
        return 1

    if not wavs:
        print(f"no .wav files under {target}")
        return 1

    flagged_n = 0
    print(f"threshold={threshold}  files={len(wavs)}")
    print(f"{'FLAG':<5} {'ratio':>7} {'dur_s':>7}  file")
    for w in wavs:
        try:
            flagged, ratio, dur = _scan_one(w, threshold)
        except Exception as exc:  # noqa: BLE001
            print(f"{'ERR':<5} {'-':>7} {'-':>7}  {w.name}: {type(exc).__name__}: {exc}")
            continue
        flagged_n += int(flagged)
        mark = "ART" if flagged else "ok"
        print(f"{mark:<5} {ratio:>7.3f} {dur:>7.2f}  {os.path.basename(str(w))}")
    print(f"\nFLAGGED {flagged_n} / {len(wavs)} clips (high-band edge artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
