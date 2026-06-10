# One-shot diagnostic: duration + RMS of every IndexTTS2 ref WAV (pass-2 Gemini claim check).
import os
import numpy as np
import soundfile as sf

D = r"C:\ComfyUI-Models\TTS\refs\indextts2"
rows = []
for name in sorted(os.listdir(D)):
    if not name.lower().endswith(".wav"):
        continue
    p = os.path.join(D, name)
    data, sr = sf.read(p, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    dur = len(data) / float(sr)
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    rows.append((name, sr, dur, rms, peak))

print(f"{'ref':38s} {'sr':>6s} {'dur_s':>7s} {'rms':>7s} {'peak':>6s}")
for name, sr, dur, rms, peak in rows:
    print(f"{name:38s} {sr:6d} {dur:7.2f} {rms:7.4f} {peak:6.3f}")
durs = [r[2] for r in rows]
print(f"\nn={len(rows)}  dur min/med/max = {min(durs):.2f}/{sorted(durs)[len(durs)//2]:.2f}/{max(durs):.2f}s")
print("short (<5s):", sum(1 for d in durs if d < 5.0), " near-12s (>=11.5s):", sum(1 for d in durs if d >= 11.5))
