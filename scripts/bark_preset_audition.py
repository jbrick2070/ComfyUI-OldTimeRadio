r"""Bark preset audition (S3) -- render one fixed-seed line across the v2/en_speaker
presets and write a metrics CSV + a run manifest, to justify a recommended_speakers
list for the char_bark_v1 profile.

NOTE (touch-scope): the BARK_VOICE_UPDATE_PLAN names only _otr_bark_lib.py /
eng_bark.py / audio_engine_profiles.yaml / tests. S3 inherently needs a render
SCRIPT + CSV/manifest outside that list -- this script is that flagged addition.
It is GPU-only and commits NO audio binaries (WAVs are written to a local temp dir).

Usage (on a GPU box with the suno/bark weights cached):
    .venv\Scripts\python.exe scripts\bark_preset_audition.py
Outputs:
    docs/2026-06-17-bark-voice/audition/preset_audition.csv
    docs/2026-06-17-bark-voice/audition/manifest.json
    <temp>/bark_audition/*.wav   (LOCAL ONLY -- not committed)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

# repo root on sys.path so `nodes` imports resolve when run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

LINE = ("This is the evening broadcast. Stay tuned for the next chapter of our "
        "story.")
SEED = 42
PRESETS = [f"v2/en_speaker_{i}" for i in range(10)]
OUT_DIR = os.path.join(_REPO, "docs", "2026-06-17-bark-voice", "audition")


def _dbfs(x, ref=1.0):
    v = float(np.abs(x).max()) if x.size else 0.0
    return 20.0 * np.log10(max(v, 1e-9) / ref)


def _rms_dbfs(x):
    if not x.size:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(max(r, 1e-9))


def _lufs(x, sr):
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:  # noqa: BLE001 -- LUFS is best-effort
        return None


def main():
    import torch
    from nodes._otr_bark_lib import _generate_single_line, _load_bark
    from nodes._otr_audio_engines.eng_bark import BarkEngine

    if not torch.cuda.is_available():
        print("CUDA not available -- Bark render is GPU-only. Aborting.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    wav_dir = os.path.join(tempfile.gettempdir(), "bark_audition")
    os.makedirs(wav_dir, exist_ok=True)

    eng = BarkEngine()
    sem, crs, fin = eng._resolve_stage_temps()
    print(f"stage temps (from char_bark_v1): semantic={sem} coarse={crs} fine={fin}")

    model, processor = _load_bark("suno/bark")
    import transformers

    rows = []
    for preset in PRESETS:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        t0 = time.time()
        audio, sr = _generate_single_line(
            LINE, preset, model, processor, is_first_line=False,
            semantic_temp=sem, coarse_temp=crs, fine_temp=fin)
        dt = time.time() - t0
        audio = np.asarray(audio, dtype=np.float32)
        dur = len(audio) / float(sr) if sr else 0.0
        peak = _dbfs(audio)
        rms = _rms_dbfs(audio)
        lufs = _lufs(audio, sr)
        # local-only WAV (not committed)
        try:
            import soundfile as sf
            sf.write(os.path.join(wav_dir, f"{preset.replace('/', '_')}.wav"),
                     audio, sr)
        except Exception as e:  # noqa: BLE001
            print(f"  (wav save skipped for {preset}: {e})")
        row = {"preset": preset, "peak_dbfs": round(peak, 2),
               "rms_dbfs": round(rms, 2),
               "lufs": round(lufs, 2) if lufs is not None else "",
               "duration_s": round(dur, 3), "render_s": round(dt, 2),
               "notes": ""}
        rows.append(row)
        print(f"  {preset}: peak={row['peak_dbfs']} rms={row['rms_dbfs']} "
              f"lufs={row['lufs']} dur={row['duration_s']}s")

    csv_path = os.path.join(OUT_DIR, "preset_audition.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "seed": SEED, "line": LINE, "presets": PRESETS,
        "stage_temps": {"semantic": sem, "coarse": crs, "fine": fin},
        "transformers": transformers.__version__,
        "model_id": "suno/bark",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "wav_dir_local_only": wav_dir,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {csv_path}")
    print("Pick recommended_speakers = the fullest/least-thin presets (highest RMS "
          "/ LUFS, clean timbre), then add them to char_bark_v1.default_params.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
