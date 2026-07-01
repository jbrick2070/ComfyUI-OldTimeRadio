r"""Cosmic Radio Mandala -- pycairo prototype (viz_mxc, 2026-06-30).

The Tuning Eye of the Multiverse: a recursive radio-dial mandala + pulsing tuning
eye, iridescent multi-hue, audio-reactive. Pure CPU (cairo + numpy + ffmpeg),
cross-platform. Reuses scope_draw's audio analysis + silent encoder.

Reactivity: bass -> outer ring radius + stroke; mids -> spoke rotation + count;
treble -> filigree flicker/detail; onset (RMS delta) -> symmetry-flip + signal-lock
flash; spectral centroid -> global hue drift.

Usage: python mandala_proto.py <master.wav> <out.mp4> [seconds]
"""
from __future__ import annotations

import colorsys
import math
import os
import sys

import numpy as np
import cairo

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from nodes._otr_shared import scope_draw as sd  # noqa: E402

W, H, FPS = 1472, 832, 25


def _hue(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))
    return r, g, b


def _band(freq, a, b):
    seg = freq[a:b]
    return float(np.mean(seg)) if len(seg) else 0.0


def _centroid(freq):
    f = np.asarray(freq, dtype=np.float32)
    s = float(f.sum())
    if s <= 1e-6:
        return 0.0
    return float((np.arange(len(f)) * f).sum() / s / max(1, len(f)))


def paint_mandala(ctx, w, h, fi, total, fps, vol, freq, sig, onset):
    cx, cy = w / 2.0, h / 2.0
    t = fi / float(fps or 25)
    bass = _band(freq, 0, 5)
    mid = _band(freq, 5, 16)
    treble = _band(freq, 16, 32)
    base_hue = (_centroid(freq) * 0.6 + t * 0.02) % 1.0

    # --- background: deep radio-bronze radial wash ---
    bg = cairo.RadialGradient(cx, cy, 0, cx, cy, w * 0.6)
    bg.add_color_stop_rgb(0.0, 0.05, 0.03, 0.02)
    bg.add_color_stop_rgb(1.0, 0.01, 0.01, 0.02)
    ctx.set_source(bg)
    ctx.paint()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # --- OUTER SOLID SPECTRUM BAND RING (filled wedges = density + "thick bands") ---
    n_band = 48
    r_in = (0.40 + bass * 0.06) * min(w, h)
    band_rot = t * (0.05 + mid * 0.4)
    for k in range(n_band):
        a0 = 2 * math.pi * k / n_band + band_rot
        a1 = 2 * math.pi * (k + 1) / n_band + band_rot
        mag = max(0.0, min(1.0, float(freq[k % len(freq)])))
        r_out = r_in + (0.02 + 0.10 * (0.4 + mag)) * min(w, h)
        hue = (base_hue + k / float(n_band)) % 1.0
        r, g, b = _hue(hue, 0.6, 0.32 + 0.55 * mag)
        ctx.set_source_rgba(r, g, b, 0.55 + 0.4 * mag)   # SOLID
        ctx.move_to(cx + r_in * math.cos(a0), cy + r_in * math.sin(a0))
        ctx.arc(cx, cy, r_in, a0, a1)
        ctx.arc_negative(cx, cy, r_out, a1, a0)
        ctx.close_path()
        ctx.fill()

    # --- concentric rings (bass) -- bolder + more solid ---
    n_rings = 9
    for i in range(n_rings):
        rr = (0.10 + 0.055 * i) * min(w, h) * (1.0 + bass * 0.30)
        hue = (base_hue + i * 0.10) % 1.0
        r, g, b = _hue(hue, 0.58, 0.40 + 0.5 * (0.4 + bass))
        ctx.set_source_rgba(r, g, b, 0.55 + 0.4 * bass)
        ctx.set_line_width(max(1.5, (3.0 + bass * 7.0) * (1.0 - i / (n_rings * 2.2))))
        ctx.arc(cx, cy, rr, 0, 2 * math.pi)
        ctx.stroke()

    # --- radial spokes (mids), kaleidoscopic; symmetry flips on onset ---
    sym = 24 + (12 if onset > 0.5 else 0)
    rot = t * (0.15 + mid * 1.6)
    inner = 0.14 * min(w, h)
    outer = (0.52 + bass * 0.1) * min(w, h)
    for k in range(sym):
        a = 2 * math.pi * k / sym + rot
        hue = (base_hue + k / float(sym)) % 1.0
        r, g, b = _hue(hue, 0.5, 0.3 + 0.6 * mid)
        ctx.set_source_rgba(r, g, b, 0.28 + 0.5 * mid)
        ctx.set_line_width(max(1.0, 1.0 + mid * 3.0))
        ctx.move_to(cx + inner * math.cos(a), cy + inner * math.sin(a))
        ctx.line_to(cx + outer * math.cos(a), cy + outer * math.sin(a))
        ctx.stroke()
        # filigree ticks (treble) along the spoke
        if treble > 0.06:
            steps = 3 + int(treble * 8)
            for s2 in range(steps):
                rr = inner + (outer - inner) * (s2 + 1) / (steps + 1)
                tick = (2 + treble * 8) * (0.5 + 0.5 * math.sin(t * 6 + k))
                hue2 = (hue + 0.5) % 1.0
                r2, g2, b2 = _hue(hue2, 0.6, 0.4 + 0.6 * treble)
                ctx.set_source_rgba(r2, g2, b2, 0.5 * treble + 0.15)
                px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
                ctx.arc(px, py, max(0.8, tick * 0.4), 0, 2 * math.pi)
                ctx.fill()

    # --- the TUNING EYE: pulsing iris + oscilloscope crosshair ---
    eye_r = (0.05 + vol * 0.05) * min(w, h)
    for j in range(5):
        rr = eye_r * (1.0 - j * 0.16)
        r, g, b = _hue((base_hue + 0.5 + j * 0.05) % 1.0, 0.45, 0.5 + 0.4 * sig)
        ctx.set_source_rgba(r, g, b, 0.85 - j * 0.12)
        ctx.arc(cx, cy, max(1.0, rr), 0, 2 * math.pi)
        ctx.fill()
    # signal-lock flash on onset
    if onset > 0.5:
        ctx.set_source_rgba(1.0, 0.95, 0.8, min(0.8, onset))
        ctx.set_line_width(3.0)
        ctx.arc(cx, cy, eye_r * 2.6, 0, 2 * math.pi)
        ctx.stroke()
    # crosshair
    ctx.set_source_rgba(0.9, 0.85, 0.6, 0.5)
    ctx.set_line_width(1.0)
    ctx.move_to(cx - eye_r * 1.6, cy); ctx.line_to(cx + eye_r * 1.6, cy); ctx.stroke()
    ctx.move_to(cx, cy - eye_r * 1.6); ctx.line_to(cx, cy + eye_r * 1.6); ctx.stroke()


def surface_to_rgb(surface, w, h):
    buf = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=surface.get_data())
    # cairo ARGB32 is BGRA in memory (little-endian) -> take B,G,R -> RGB
    rgb = buf[:, :, [2, 1, 0]].copy()
    return rgb


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else ""
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_REPO, "mandala_out.mp4")
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    total = int(secs * FPS)

    if wav and os.path.exists(wav):
        import soundfile as sf
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[: int(sr * secs)] if len(audio) > int(sr * secs) else audio
    else:
        sr = 24000
        audio = np.zeros(int(sr * secs), dtype=np.float32)
        print("[mandala] no wav -> idle")

    volume, freqs, waves = sd.analyze_audio_np(audio, int(sr), total, FPS)
    signal, _trig, _loss = sd.dual_ema(volume)
    vol_arr = np.asarray(volume, dtype=np.float32)
    onsets = np.zeros(total, dtype=np.float32)
    for i in range(1, total):
        d = vol_arr[i] - vol_arr[i - 1]
        onsets[i] = 1.0 if d > 0.06 else 0.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    def _frames():
        for fi in range(total):
            paint_mandala(ctx, W, H, fi, total, FPS, float(volume[fi]),
                          freqs[fi], float(signal[fi]), float(onsets[fi]))
            surface.flush()
            yield surface_to_rgb(surface, W, H)

    sd.encode_silent_mp4(_frames(), total, out, W, H, FPS,
                         os.environ.get("OTR_FFMPEG", "ffmpeg"))
    print("[mandala] wrote", out, total, "frames")


if __name__ == "__main__":
    main()
