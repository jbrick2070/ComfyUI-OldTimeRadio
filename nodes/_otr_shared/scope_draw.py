r"""Shared CRT scope draw routines -- the full-colour procedural visualizer look.

Torch-FREE (numpy + PIL + stdlib only) per-frame draw + audio-analysis + silent
ffmpeg-encode helpers, COPIED from ``nodes/video_engine.py``'s full-frame renderer
(the ring / orbiting particles / grid / mirrored waveform / freq bars / CRT post)
so the ``visualizer`` engine can paint a standalone 16:9 beat clip WITHOUT importing
or invoking the floor node (the SEPARATION INVARIANT, 2026-06-17 scope-visualizer
plan section 0.2). v1 COPIES rather than EXTRACTS so the floor's behaviour is
provably unchanged -- a later refactor can make the floor import this module.

The title-card / ident / gap branches are deliberately OMITTED -- those only make
sense for the whole-episode floor; a per-beat clip is pure procedural art.

Self-contained on purpose: ``_analyze_audio_np`` / ``_dual_ema`` / ``_encode_silent_mp4``
mirror the ``otr_scene_aware_scopes`` helpers byte-for-byte but are copied here so the
engine couples to NOTHING (neither the floor node nor the overlay node). UTF-8, no BOM.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -- full-colour CRT palette (from video_engine.py) --------------------------
CRT_BG = (8, 8, 16)
CRT_GREEN = (0, 255, 65)
CRT_CYAN = (0, 200, 200)
CRT_AMBER = (255, 176, 0)
CRT_DARK = (0, 50, 14)


# --------------------------------------------------------------------------- #
# Deterministic per-frame RNG (blake2s stable-hash; same key+fi+salt -> same).
# --------------------------------------------------------------------------- #
def _rng(key, fi, salt):
    seed = int.from_bytes(
        hashlib.blake2s(f"{key}|{int(fi)}|{salt}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Pure-numpy audio analysis (mirrors video_engine._analyze_audio at 25 fps).
# --------------------------------------------------------------------------- #
def analyze_audio_np(audio_np, sample_rate, total_frames, fps):
    spf = sample_rate // fps
    volume, freqs, waves = [], [], []
    for i in range(total_frames):
        s = i * spf
        e = min(s + spf, len(audio_np))
        chunk = audio_np[s:e] if s < len(audio_np) else np.zeros(spf)
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        volume.append(rms)
        if len(chunk) > 0:
            fft = np.abs(np.fft.rfft(chunk))
            n = len(fft)
            if n >= 32:
                bs = n // 32
                bins = np.array([np.mean(fft[j * bs:(j + 1) * bs]) for j in range(32)])
            else:
                bins = np.zeros(32)
                bins[:n] = fft[:n]
        else:
            bins = np.zeros(32)
        freqs.append(bins)
        if len(chunk) > 200:
            idx = np.linspace(0, len(chunk) - 1, 200, dtype=int)
            waves.append(chunk[idx])
        else:
            waves.append(chunk)
    vmax = max(volume) if volume and max(volume) > 0 else 1.0
    volume = [v / vmax for v in volume]
    fmax = max((np.max(f) for f in freqs), default=1.0) if freqs else 1.0
    if fmax > 0:
        freqs = [f / fmax for f in freqs]
    return volume, freqs, waves


def dual_ema(volume):
    """signal (slow ambient, a=0.05) + trig (fast lock, a=0.30) + loss=1-signal."""
    v = np.asarray(volume, dtype=np.float32)
    n = len(v)
    sig = np.zeros(n, dtype=np.float32)
    trg = np.zeros(n, dtype=np.float32)
    if n > 0:
        sig[0] = trg[0] = float(v[0])
        a_s, a_t = 0.05, 0.30
        for i in range(1, n):
            sig[i] = sig[i - 1] + a_s * (float(v[i]) - sig[i - 1])
            trg[i] = trg[i - 1] + a_t * (float(v[i]) - trg[i - 1])
    return sig, trg, (1.0 - sig).astype(np.float32)


# --------------------------------------------------------------------------- #
# Geometry + pre-built CRT overlays (from video_engine.py __init__).
# --------------------------------------------------------------------------- #
def ring_geom(w, h):
    """(ring_cx, ring_cy, ring_r, pad, divider_y) -- the floor's exact values."""
    return (w // 2, int(h * 0.42), min(w, h) // 5, w // 48, h // 10)


def build_scanlines(w, h):
    """RGBA scan-line overlay (every ``max(2, h//360)`` rows, alpha 45)."""
    sl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sl)
    step = max(2, h // 360)
    for y in range(0, h, step):
        sd.line([(0, y), (w, y)], fill=(0, 0, 0, 45))
    return sl


def build_vignette(w, h):
    """numpy vignette multiplier ``clip(1 - dist*0.35, 0.45, 1.0)``."""
    cy, cx = h / 2.0, w / 2.0
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 / (cx ** 2) + (Y - cy) ** 2 / (cy ** 2))
    return np.clip(1.0 - dist * 0.35, 0.45, 1.0).astype(np.float32)


def _small_font(h):
    """A small chrome font; truetype if available, else PIL's default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", max(9, h // 72))
    except Exception:  # noqa: BLE001 -- headless / no truetype -> bitmap default
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Per-section draw helpers (copied from video_engine.py; geometry-by-params).
# --------------------------------------------------------------------------- #
def _waveform_mirror(draw, wave, x, y, w, h, vol):
    mid = y + h // 2
    n = len(wave)
    if n < 1:
        return
    pts_top, pts_bot = [], []
    for i in range(min(n, w)):
        px = x + int(i * w / n)
        amp = float(wave[i]) * h * 0.45
        pts_top.append((px, mid - int(amp)))
        pts_bot.append((px, mid + int(amp)))
    brightness = min(1.0, 0.3 + vol * 0.9)
    col_top = tuple(min(255, int(c * brightness)) for c in CRT_GREEN)
    col_bot = tuple(min(255, int(c * brightness * 0.5)) for c in CRT_CYAN)
    if len(pts_top) > 1:
        draw.line(pts_top, fill=col_top, width=2)
        draw.line(pts_bot, fill=col_bot, width=1)
    draw.line([(x, mid), (x + w, mid)], fill=CRT_DARK, width=1)


def _freq_bars_wide(draw, freq, x, y, w, h, vol):
    n = min(32, len(freq))
    if n < 1:
        return
    bw = max(1, w // n - 1)
    for i in range(n):
        bh = max(1, int(float(freq[i]) * h * 1.5))
        bx = x + i * (bw + 1)
        by = y + h - min(bh, h)
        ratio = i / max(1, n - 1)
        if ratio < 0.5:
            r, g, b = int(ratio * 2 * 255), 255, 20
        else:
            r, g, b = 255, int((1.0 - (ratio - 0.5) * 2) * 200), 20
        brightness = 0.25 + float(freq[i]) * 0.75
        col = (min(255, int(r * brightness)), min(255, int(g * brightness)),
               min(255, int(b * brightness)))
        draw.rectangle([(bx, by), (bx + bw, y + h)], fill=col)


def freq_bars_green(draw, freq, x, y, w, h):
    """GREEN-ONLY frequency-bar strip -- the bottom-bars overlay look.

    Same bar geometry as :func:`_freq_bars_wide` but the palette is CRT_GREEN
    scaled by per-bin magnitude (NO amber/red gradient), so it honors the
    overlay's green-only invariant (OTR_SceneAwareScopes deliberately ships no
    colored CRT constants). ``freq`` is one frame's 32-bin spectrum (0..1).
    Geometry by params so the overlay node and the engine can both call it."""
    n = min(32, len(freq))
    if n < 1:
        return
    bw = max(1, w // n - 1)
    for i in range(n):
        mag = max(0.0, min(1.0, float(freq[i])))
        bh = max(1, int(mag * h * 1.5))
        bx = x + i * (bw + 1)
        by = y + h - min(bh, h)
        brightness = 0.25 + mag * 0.75
        col = tuple(min(255, int(c * brightness)) for c in CRT_GREEN)
        draw.rectangle([(bx, by), (bx + bw, y + h)], fill=col)


def paint_frame(w, h, fi, total, fps, vol, freq, wave, signal, loss,
                scanlines, vignette, rng_key="visualizer", font_small=None):
    """Paint ONE full-frame 16:9 CRT visualizer picture (sections 2-8 of the
    floor renderer; NO title card / ident / gap). Returns a PIL RGB Image.

    ``freq`` is one frame's 32-bin spectrum, ``wave`` one frame's samples (the
    full-colour look uses single-frame inputs, unlike the green windowed helpers)."""
    cx0, cy0, base_r, pad, ly = ring_geom(w, h)
    t = fi / float(fps or 25)
    img = Image.new("RGB", (w, h), CRT_BG)
    draw = ImageDraw.Draw(img)

    # divider chrome
    draw.line([(pad, ly), (w - pad, ly)], fill=CRT_DARK, width=1)

    # -- 2. circular frequency ring -------------------------------------
    r = base_r + int(vol * base_r * 0.3)
    drift = int(round(loss * (w // 120)))
    cx = max(r, min(w - r, cx0 + drift))
    cy = cy0
    n_bars = min(32, len(freq))
    for i in range(n_bars):
        angle = 2 * math.pi * i / n_bars - math.pi / 2
        bar_len = int(float(freq[i]) * h * 0.18) + 2
        x0 = cx + int(r * math.cos(angle))
        y0 = cy + int(r * math.sin(angle))
        x1 = cx + int((r + bar_len) * math.cos(angle))
        y1 = cy + int((r + bar_len) * math.sin(angle))
        g = int(255 * (1.0 - float(freq[i]) * 0.6))
        rb = int(float(freq[i]) * 180)
        col = (rb, g, max(20, 65 - int(float(freq[i]) * 50)))
        draw.line([(x0, y0), (x1, y1)], fill=col, width=max(2, w // 400))
    ring_bright = min(1.0, 0.3 + vol * 0.7)
    ring_col = tuple(min(255, int(c * ring_bright)) for c in CRT_GREEN)
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=ring_col, width=2)

    # -- 3. orbiting particles ------------------------------------------
    n_particles = 12
    flen = len(freq)
    for p in range(n_particles):
        phase = 2 * math.pi * p / n_particles
        orbit_r = r + int(h * 0.12) + int(vol * 30)
        fv = float(freq[p % flen]) if flen else 0.0
        speed = 0.3 + fv * 2.0
        angle = phase + t * speed
        px = cx + int(orbit_r * math.cos(angle))
        py = cy + int(orbit_r * math.sin(angle) * 0.6)
        size = max(2, int(3 + fv * 8))
        hue_shift = (p / n_particles + t * 0.05) % 1.0
        pcol = CRT_GREEN if hue_shift < 0.33 else (
            CRT_CYAN if hue_shift < 0.66 else CRT_AMBER)
        bright = min(1.0, 0.3 + fv * 0.7)
        pcol = tuple(min(255, int(c * bright)) for c in pcol)
        draw.ellipse([(px - size, py - size), (px + size, py + size)], fill=pcol)

    # -- 4. geometric grid ----------------------------------------------
    grid_step = max(40, w // 24)
    grid_alpha = max(6, int((15 + vol * 25) * (0.35 + 0.65 * signal)))
    grid_col = (0, grid_alpha, int(grid_alpha * 0.4))
    for gx in range(pad, w - pad, grid_step):
        wob = int(math.sin(gx * 0.01 + t * 2.0) * vol * 12)
        draw.line([(gx + wob, ly + pad), (gx - wob, h - pad * 2)], fill=grid_col, width=1)
    for gy in range(ly + pad, h - pad * 2, grid_step):
        wob = int(math.sin(gy * 0.01 + t * 1.5) * vol * 8)
        draw.line([(pad + wob, gy), (w - pad - wob, gy)], fill=grid_col, width=1)

    # -- 5. mirrored waveform + 6. frequency bars -----------------------
    if wave is not None and len(wave) > 1:
        _waveform_mirror(draw, wave, pad, int(h * 0.72), w - pad * 2, int(h * 0.12), vol)
    if freq is not None and len(freq) > 0:
        _freq_bars_wide(draw, freq, pad, int(h * 0.86), w - pad * 2, int(h * 0.06), vol)

    # -- 7. bottom chrome bar -------------------------------------------
    f_small = font_small or _small_font(h)
    by = h - pad
    draw.line([(pad, by - pad // 3), (w - pad, by - pad // 3)], fill=CRT_DARK, width=1)
    draw.text((pad, by - pad // 6), "OTR v1.0  x  %dx%d  x  %dfps" % (w, h, fps),
              fill=CRT_DARK, font=f_small)
    draw.text((w - pad - w // 8, by - pad // 6), "frame %05d/%05d" % (fi, total),
              fill=CRT_DARK, font=f_small)

    # -- 8. CRT post (scanlines composite + vignette multiply + noise) --
    img = Image.alpha_composite(img.convert("RGBA"), scanlines).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr *= vignette[:, :, np.newaxis]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if vol > 0.3:
        a = np.array(img, dtype=np.int16)
        intensity = int(vol * 12)
        noise = _rng(rng_key, fi, "noise").integers(
            -intensity, intensity + 1, size=a.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(a + noise, 0, 255).astype(np.uint8))
    return img


def _hue_rgb(hue, sat, val):
    """HSV -> (r,g,b) 0..255. For the OTR look pass a MUTED sat (< ~0.6) so the
    rainbow reads period/desaturated, not neon. colorsys is stdlib (cross-platform)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, max(0.0, min(1.0, sat)),
                                  max(0.0, min(1.0, val)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _rainbow_bars(draw, freq, x, y, w, h, base_hue):
    """A bottom SIGNAL-SPECTRUM strip: 32 bars, each its own muted rainbow hue,
    height by per-bin magnitude. The 'radio signal spectrum' motif."""
    n = min(32, len(freq))
    if n < 1:
        return
    bw = max(1, w // n - 1)
    for i in range(n):
        mag = max(0.0, min(1.0, float(freq[i])))
        bh = max(1, int(mag * h * 1.4))
        bx = x + i * (bw + 1)
        col = _hue_rgb((base_hue + i / float(n)) % 1.0, 0.5, 0.3 + mag * 0.6)
        draw.rectangle([(bx, y + h - min(bh, h)), (bx + bw, y + h)], fill=col)


def paint_rainbow_frame(w, h, fi, total, fps, vol, freq, wave, signal, loss,
                        scanlines, vignette, rng_key="viz_mxc", font_small=None):
    """Paint ONE full 16:9 OTR 'mxc' MULTI-COLORED frame -- a vintage radio-dial /
    SIGNAL-SPECTRUM sweep, NOT party rainbows. The 32-bin FFT drives a MUTED rainbow
    spectrum ring; a glowing tuning DIAL + magic-eye needle sweeps with RMS; the same
    CRT scanlines + vignette + film grain as :func:`paint_frame` give the period
    mystique. Same signature + silent-clip contract. Audio-optional: on silence the
    ring/needle idle (all inputs ~0). Returns a PIL RGB Image."""
    cx0, cy0, base_r, pad, ly = ring_geom(w, h)
    t = fi / float(fps or 25)
    img = Image.new("RGB", (w, h), CRT_BG)
    draw = ImageDraw.Draw(img)
    draw.line([(pad, ly), (w - pad, ly)], fill=CRT_DARK, width=1)

    # slow global hue drift = the "tuning sweep" (onsets nudge it)
    base_hue = (t * 0.03 + float(signal) * 0.15) % 1.0
    cx, cy = cx0, cy0
    r = base_r + int(float(vol) * base_r * 0.25)

    # -- spectrum ring: 32 bars, each a muted rainbow hue, length by magnitude --
    n = min(32, len(freq))
    for i in range(n):
        angle = 2 * math.pi * i / max(1, n) - math.pi / 2
        mag = max(0.0, min(1.0, float(freq[i])))
        bar_len = int(mag * h * 0.16) + 2
        col = _hue_rgb((base_hue + i / float(max(1, n))) % 1.0, 0.55, 0.32 + mag * 0.6)
        x0 = cx + int(r * math.cos(angle))
        y0 = cy + int(r * math.sin(angle))
        x1 = cx + int((r + bar_len) * math.cos(angle))
        y1 = cy + int((r + bar_len) * math.sin(angle))
        draw.line([(x0, y0), (x1, y1)], fill=col, width=max(2, w // 400))

    # -- the dial ring (amber period rim) + the tuning NEEDLE (magic-eye) --
    ring_bright = min(1.0, 0.35 + float(vol) * 0.65)
    ring_col = tuple(min(255, int(c * ring_bright)) for c in CRT_AMBER)
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=ring_col, width=2)
    needle_a = -math.pi / 2 + (float(signal) - 0.5) * math.pi * 0.9
    nx = cx + int((r - 4) * math.cos(needle_a))
    ny = cy + int((r - 4) * math.sin(needle_a))
    draw.line([(cx, cy), (nx, ny)], fill=_hue_rgb(base_hue, 0.4, 0.9),
              width=max(2, w // 350))
    draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=CRT_AMBER)

    # -- bottom rainbow SIGNAL-SPECTRUM strip --
    if freq is not None and len(freq) > 0:
        _rainbow_bars(draw, freq, pad, int(h * 0.86), w - pad * 2, int(h * 0.08),
                      base_hue)
    f_small = font_small or _small_font(h)
    by = h - pad
    draw.text((pad, by - pad // 6), "OTR mxc  x  %dx%d" % (w, h),
              fill=CRT_DARK, font=f_small)

    # -- CRT post: scanlines + vignette + film grain (same period look) --
    img = Image.alpha_composite(img.convert("RGBA"), scanlines).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr *= vignette[:, :, np.newaxis]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    a = np.array(img, dtype=np.int16)
    intensity = int(4 + float(vol) * 10)
    noise = _rng(rng_key, fi, "grain").integers(
        -intensity, intensity + 1, size=a.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(a + noise, 0, 255).astype(np.uint8))
    return img


def _golden_mix(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _golden_scale(col, k):
    return tuple(max(0, min(255, int(c * float(k)))) for c in col)


def _golden_camera_body(draw, cx, cy, s, vol, signal):
    """Draw the centered movie camera icon used by viz_camera."""
    gold = (231, 165, 30)
    amber = (255, 205, 92)
    ember = (225, 57, 31)
    shadow = (43, 25, 13)
    body_w = int(s * 1.38)
    body_h = int(s * 0.62)
    body = (cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2)
    glow = min(1.0, 0.28 + float(vol) * 0.55 + float(signal) * 0.25)

    for pad, alpha in ((16, 0.16), (9, 0.25), (4, 0.34)):
        col = _golden_scale(amber, alpha + glow * 0.22)
        draw.rounded_rectangle(
            (body[0] - pad, body[1] - pad, body[2] + pad, body[3] + pad),
            radius=max(4, s // 9), outline=col, width=max(1, s // 32))

    draw.rounded_rectangle(body, radius=max(5, s // 10), fill=shadow,
                           outline=_golden_scale(gold, 0.92), width=max(2, s // 34))
    draw.rectangle((body[0] + s // 14, body[1] + s // 9,
                    body[2] - s // 14, cy + s // 16),
                   fill=(68, 39, 18), outline=_golden_scale(gold, 0.48))

    lens_r = max(8, s // 5)
    lens_cx = body[2] - lens_r // 3
    draw.ellipse((lens_cx - lens_r, cy - lens_r, lens_cx + lens_r, cy + lens_r),
                 fill=(18, 13, 10), outline=amber, width=max(2, s // 32))
    draw.ellipse((lens_cx - lens_r // 2, cy - lens_r // 2,
                  lens_cx + lens_r // 2, cy + lens_r // 2),
                 fill=(49, 75, 63), outline=_golden_scale(gold, 0.6))
    draw.ellipse((lens_cx - lens_r // 5, cy - lens_r // 5,
                  lens_cx + lens_r // 5, cy + lens_r // 5),
                 fill=_golden_mix((15, 50, 45), ember, glow))

    reel_y = body[1] - s // 3
    reel_r = max(10, s // 4)
    for off in (-s // 3, s // 3):
        rcx = cx + off
        draw.ellipse((rcx - reel_r, reel_y - reel_r, rcx + reel_r, reel_y + reel_r),
                     fill=(21, 14, 9), outline=gold, width=max(2, s // 36))
        spin = float(signal) * math.pi * 2.0
        for k in range(6):
            a = spin + k * math.pi / 3.0
            rr = reel_r * 0.72
            x1 = rcx + int(rr * math.cos(a))
            y1 = reel_y + int(rr * math.sin(a))
            draw.line((rcx, reel_y, x1, y1), fill=_golden_scale(amber, 0.78),
                      width=max(1, s // 48))
        hole = max(3, reel_r // 5)
        draw.ellipse((rcx - hole, reel_y - hole, rcx + hole, reel_y + hole),
                     fill=(8, 6, 5), outline=_golden_scale(gold, 0.55))

    cone = [(body[2] - s // 12, cy - s // 6), (body[2] + s // 2, cy - s // 3),
            (body[2] + s // 2, cy + s // 3), (body[2] - s // 12, cy + s // 6)]
    draw.polygon(cone, fill=(56, 31, 15), outline=_golden_scale(gold, 0.82))
    draw.line((cx, body[3], cx, body[3] + s // 2), fill=_golden_scale(gold, 0.78),
              width=max(2, s // 35))
    draw.line((cx, body[3] + s // 5, cx - s // 2, body[3] + s // 2),
              fill=_golden_scale(gold, 0.58), width=max(2, s // 42))
    draw.line((cx, body[3] + s // 5, cx + s // 2, body[3] + s // 2),
              fill=_golden_scale(gold, 0.58), width=max(2, s // 42))


def paint_golden_camera_frame(w, h, fi, total, fps, vol, freq, wave, signal, loss,
                              scanlines, vignette, rng_key="viz_camera",
                              font_small=None):
    """Paint ONE full 16:9 Golden Flicker camera visualizer frame.

    This is an OTR-native PIL implementation inspired by the operator-approved
    camera centerpiece: twin spinning reels, a warm projector beam, mandala-like
    gold rings, audio-reactive spectrum spokes, CRT scanlines/vignette/grain.
    It has the same signature and silent-clip contract as paint_rainbow_frame.
    """
    del font_small
    cx0, cy0, base_r, pad, ly = ring_geom(w, h)
    t = fi / float(fps or 25)
    bg = (10, 7, 5)
    gold = (231, 165, 30)
    amber = (255, 205, 92)
    ember = (225, 57, 31)
    deep = (38, 22, 12)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img, "RGBA")
    cx = cx0
    cy = cy0 + int(h * 0.02)
    r = base_r + int(base_r * (0.08 + float(vol) * 0.22))
    rng = _rng(rng_key, fi, "golden-camera")

    draw.line([(pad, ly), (w - pad, ly)], fill=(*_golden_scale(gold, 0.2), 120),
              width=1)

    # Soft projector cone from the left, widening around the camera.
    beam_y = cy - int(r * 0.05)
    for i in range(8):
        spread = int((i + 1) * h * 0.022)
        alpha = max(8, 58 - i * 6)
        draw.polygon([(pad, beam_y - spread // 3),
                      (cx - r // 3, beam_y - spread),
                      (cx + r, beam_y + spread // 2),
                      (pad, beam_y + spread // 3)],
                     fill=(255, 185, 62, alpha))

    # Film gate / occult mandala rings.
    pulse = 0.6 + float(signal) * 0.4
    for j in range(8):
        rr = int(r * (0.42 + j * 0.115) + math.sin(t * 0.8 + j) * r * 0.015)
        col = _golden_mix(deep, gold, 0.28 + 0.08 * j + float(vol) * 0.2)
        width = max(1, int(w / 560) + (1 if j % 3 == 0 else 0))
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr),
                     outline=(*col, int(74 + 70 * pulse)), width=width)

    n = min(32, len(freq))
    spin = t * 0.35 + float(signal) * 0.35
    for i in range(max(1, n)):
        mag = max(0.0, min(1.0, float(freq[i]) if n else 0.0))
        a = 2 * math.pi * i / max(1, n) - math.pi / 2 + spin
        inner = r * (0.52 + 0.04 * math.sin(i + t))
        outer = r * (0.82 + mag * 0.46)
        col = _golden_mix(gold, ember, min(1.0, mag * 0.75 + float(vol) * 0.25))
        draw.line((cx + int(inner * math.cos(a)), cy + int(inner * math.sin(a)),
                   cx + int(outer * math.cos(a)), cy + int(outer * math.sin(a))),
                  fill=(*col, int(105 + mag * 120)), width=max(1, w // 520))

    # Petal lattice, camera as the machine inside the mandala.
    petals = 18
    for i in range(petals):
        a = 2 * math.pi * i / petals + spin * 0.55
        a2 = a + math.pi / petals
        p0 = (cx + int(r * 0.36 * math.cos(a)), cy + int(r * 0.36 * math.sin(a)))
        p1 = (cx + int(r * 0.86 * math.cos(a2)), cy + int(r * 0.86 * math.sin(a2)))
        p2 = (cx + int(r * 0.36 * math.cos(a + 2 * math.pi / petals)),
              cy + int(r * 0.36 * math.sin(a + 2 * math.pi / petals)))
        draw.line((p0, p1, p2), fill=(*_golden_scale(amber, 0.55), 80),
                  width=max(1, w // 700))

    # Perforated film loop around the outer ring.
    holes = 44
    for i in range(holes):
        a = 2 * math.pi * i / holes - spin * 0.9
        rr = r * 1.04
        x = cx + int(rr * math.cos(a))
        y = cy + int(rr * math.sin(a))
        hw = max(2, w // 420)
        hh = max(2, h // 350)
        alpha = 70 + int(80 * (0.5 + 0.5 * math.sin(i + t * 2.0)))
        draw.rectangle((x - hw, y - hh, x + hw, y + hh),
                       outline=(*_golden_scale(gold, 0.72), alpha), width=1)

    _golden_camera_body(draw, cx, cy, max(42, int(base_r * 0.78)), vol, signal)

    # Wave glints under the camera: subtle, so it stays OTR-noir rather than EDM.
    if wave is not None and len(wave) > 1:
        samples = np.asarray(wave, dtype=np.float32)
        count = min(120, len(samples))
        idx = np.linspace(0, len(samples) - 1, count, dtype=int)
        pts = []
        y0 = int(h * 0.77)
        amp = max(4, int(h * 0.045 * (0.35 + float(vol))))
        for k, j in enumerate(idx):
            x = pad + int(k * (w - pad * 2) / max(1, count - 1))
            y = y0 + int(float(samples[j]) * amp)
            pts.append((x, y))
        draw.line(pts, fill=(*_golden_scale(amber, 0.62), 130), width=max(1, w // 520))

    # Deterministic dust and scratches before CRT post.
    for _ in range(max(12, w // 55)):
        x = int(rng.integers(0, max(1, w)))
        y = int(rng.integers(0, max(1, h)))
        a = int(rng.integers(22, 88))
        draw.point((x, y), fill=(255, 216, 130, a))
    for _ in range(3):
        x = int(rng.integers(pad, max(pad + 1, w - pad)))
        y0 = int(rng.integers(pad, max(pad + 1, h - pad)))
        y1 = min(h - pad, y0 + int(rng.integers(h // 12, max(h // 12 + 1, h // 4))))
        draw.line((x, y0, x + int(rng.integers(-2, 3)), y1),
                  fill=(255, 210, 110, int(rng.integers(22, 58))), width=1)

    img = Image.alpha_composite(img.convert("RGBA"), scanlines).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr *= vignette[:, :, np.newaxis]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    a = np.array(img, dtype=np.int16)
    intensity = int(4 + float(vol) * 10)
    noise = _rng(rng_key, fi, "grain").integers(
        -intensity, intensity + 1, size=a.shape, dtype=np.int16)
    return Image.fromarray(np.clip(a + noise, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Cosmic Radio Mandala (viz_mxc_mandala, 2026-06-30) -- pycairo painter +
# cairo->numpy conversion + the shared CRT post applied AFTER conversion (the
# native-cairo CRT glue was cut for v1; PIL roundtrip reuses the SAME
# scanlines/vignette/grain machinery as paint_frame/paint_rainbow_frame).
# cairo is imported LOCALLY inside paint_mandala only -- NOT at module scope
# anywhere in this file (V-12 cold-import clean; pycairo is not in
# the main requirements, so a box without system libcairo never breaks any
# OTHER engine's install). Ported from the operator-approved prototype
# (docs/2026-06-30-viz-rainbow/mandala_proto.py); ring/spoke/band coefficients
# are a build-time LOOK pass (2026-06-30: 48 solid spectrum wedges + 9 bolder
# rings, denser look approved), NOT frozen constants -- a future look-pass may
# retune them; only the "core rings/eye must not clip on 16:9" invariant holds.
# --------------------------------------------------------------------------- #
def _mandala_band(freq, a, b):
    seg = freq[a:b]
    return float(np.mean(seg)) if len(seg) else 0.0


def _mandala_centroid(freq):
    f = np.asarray(freq, dtype=np.float32)
    s = float(f.sum())
    if s <= 1e-6:
        return 0.0
    return float((np.arange(len(f)) * f).sum() / s / max(1, len(f)))


def paint_mandala(ctx, w, h, fi, total, fps, vol, freq, sig, onset):
    """Paint ONE 16:9 'Cosmic Radio Mandala' frame on a cairo Context: a
    centered tuning-eye + radio-dial rings/spokes + an outer solid spectrum
    band, muted iridescent hues. CRT post is applied SEPARATELY (see
    :func:`apply_crt_post_rgb`) after the cairo->numpy conversion.

    Reactivity: bass(0-5)->ring radius+stroke+band inner-radius;
    mids(5-16)->spoke count/rotation+band rotation; treble(16-32)->filigree
    flicker; onset(RMS delta)->symmetry-flip+signal-lock flash; spectral
    centroid->global hue drift; vol->tuning-eye pulse. Fully deterministic
    (no RNG here -- only fi/t/audio-features feed the geometry).
    ``ctx.save()/restore()`` brackets the whole body so no cairo state (line
    width/cap/source) leaks across frames sharing one reused context."""
    import cairo

    ctx.save()
    try:
        cx, cy = w / 2.0, h / 2.0
        t = fi / float(fps or 25)
        bass = _mandala_band(freq, 0, 5)
        mid = _mandala_band(freq, 5, 16)
        treble = _mandala_band(freq, 16, 32)
        base_hue = (_mandala_centroid(freq) * 0.6 + t * 0.02) % 1.0

        # -- background: deep radio-bronze radial wash. OPAQUE (rgb-only color
        # stops, full ctx.paint()) so premultiplied-alpha == straight -- the
        # later ARGB32->RGB read needs no unpremultiply step. --
        bg = cairo.RadialGradient(cx, cy, 0, cx, cy, w * 0.6)
        bg.add_color_stop_rgb(0.0, 0.05, 0.03, 0.02)
        bg.add_color_stop_rgb(1.0, 0.01, 0.01, 0.02)
        ctx.set_source(bg)
        ctx.paint()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        # -- outer solid spectrum band ring (filled wedges; MAY bleed past 16:9) --
        n_band = 48
        r_in = (0.40 + bass * 0.06) * min(w, h)
        band_rot = t * (0.05 + mid * 0.4)
        for k in range(n_band):
            a0 = 2 * math.pi * k / n_band + band_rot
            a1 = 2 * math.pi * (k + 1) / n_band + band_rot
            mag = max(0.0, min(1.0, float(freq[k % len(freq)])))
            r_out = r_in + (0.02 + 0.10 * (0.4 + mag)) * min(w, h)
            hue = (base_hue + k / float(n_band)) % 1.0
            r, g, b = _hue_rgb(hue, 0.6, 0.32 + 0.55 * mag)
            ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, 0.55 + 0.4 * mag)
            ctx.move_to(cx + r_in * math.cos(a0), cy + r_in * math.sin(a0))
            ctx.arc(cx, cy, r_in, a0, a1)
            ctx.arc_negative(cx, cy, r_out, a1, a0)
            ctx.close_path()
            ctx.fill()

        # -- concentric core rings (bass; must NOT clip on 1472x832) --
        n_rings = 9
        for i in range(n_rings):
            rr = (0.10 + 0.055 * i) * min(w, h) * (1.0 + bass * 0.30)
            hue = (base_hue + i * 0.10) % 1.0
            r, g, b = _hue_rgb(hue, 0.58, 0.40 + 0.5 * (0.4 + bass))
            ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, 0.55 + 0.4 * bass)
            ctx.set_line_width(max(1.5, (3.0 + bass * 7.0) * (1.0 - i / (n_rings * 2.2))))
            ctx.arc(cx, cy, rr, 0, 2 * math.pi)
            ctx.stroke()

        # -- radial spokes (mids), kaleidoscopic; symmetry flips on onset --
        sym = 24 + (12 if onset > 0.5 else 0)
        rot = t * (0.15 + mid * 1.6)
        inner = 0.14 * min(w, h)
        outer = (0.52 + bass * 0.1) * min(w, h)
        for k in range(sym):
            a = 2 * math.pi * k / sym + rot
            hue = (base_hue + k / float(sym)) % 1.0
            r, g, b = _hue_rgb(hue, 0.5, 0.3 + 0.6 * mid)
            ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, 0.28 + 0.5 * mid)
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
                    r2, g2, b2 = _hue_rgb(hue2, 0.6, 0.4 + 0.6 * treble)
                    ctx.set_source_rgba(r2 / 255.0, g2 / 255.0, b2 / 255.0,
                                        0.5 * treble + 0.15)
                    px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
                    ctx.arc(px, py, max(0.8, tick * 0.4), 0, 2 * math.pi)
                    ctx.fill()

        # -- the tuning eye: pulsing iris + oscilloscope crosshair (core; no clip) --
        eye_r = (0.05 + vol * 0.05) * min(w, h)
        for j in range(5):
            rr = eye_r * (1.0 - j * 0.16)
            r, g, b = _hue_rgb((base_hue + 0.5 + j * 0.05) % 1.0, 0.45, 0.5 + 0.4 * sig)
            ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, 0.85 - j * 0.12)
            ctx.arc(cx, cy, max(1.0, rr), 0, 2 * math.pi)
            ctx.fill()
        if onset > 0.5:
            ctx.set_source_rgba(1.0, 0.95, 0.8, min(0.8, onset))
            ctx.set_line_width(3.0)
            ctx.arc(cx, cy, eye_r * 2.6, 0, 2 * math.pi)
            ctx.stroke()
        ctx.set_source_rgba(0.9, 0.85, 0.6, 0.5)
        ctx.set_line_width(1.0)
        ctx.move_to(cx - eye_r * 1.6, cy)
        ctx.line_to(cx + eye_r * 1.6, cy)
        ctx.stroke()
        ctx.move_to(cx, cy - eye_r * 1.6)
        ctx.line_to(cx, cy + eye_r * 1.6)
        ctx.stroke()
    finally:
        ctx.restore()


def mandala_surface_to_rgb(surface, w, h):
    """Convert a cairo ARGB32 ``ImageSurface`` to a contiguous HxWx3 uint8 RGB
    array. ``flush()`` FIRST (else ``get_data`` can return a blank/partial
    frame), then read the REAL stride via ``get_stride()`` (never assume
    ``w*4`` -- cairo may pad rows for alignment). cairo ARGB32 is BGRA in
    memory (little-endian); the background is painted fully opaque so
    premultiplied-alpha == straight and no unpremultiply step is needed."""
    surface.flush()
    stride = surface.get_stride()
    bgra = np.ndarray(shape=(h, w, 4), dtype=np.uint8,
                       buffer=surface.get_data(), strides=(stride, 4, 1))
    rgb = np.ascontiguousarray(bgra[:, :, [2, 1, 0]])
    assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8
    return rgb


def apply_crt_post_rgb(rgb, scanlines, vignette, fi, rng_key, vol=0.0):
    """CRT post-process over an ALREADY-RENDERED RGB frame (the mandala's
    cairo->numpy array, or any HxWx3 uint8 array): composite the scanline
    overlay, multiply the vignette, then add seed-keyed film grain (the same
    ``_rng(rng_key, fi, "grain")`` used by :func:`paint_frame` /
    :func:`paint_rainbow_frame`, intensity ``int(4 + vol*10)``). ``fi`` is
    REQUIRED -- without it the grain would be frame-frozen. Deterministic on
    (rng_key, fi); returns a NEW HxWx3 uint8 array, never mutates ``rgb``."""
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    img = Image.alpha_composite(img, scanlines).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr *= vignette[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.int16)
    intensity = int(4 + float(vol) * 10)
    noise = _rng(rng_key, fi, "grain").integers(
        -intensity, intensity + 1, size=arr.shape, dtype=np.int16)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Silent ffmpeg encode (copied; SILENT -- only OTR_MasterAudioMux adds audio).
# --------------------------------------------------------------------------- #
def find_ffmpeg(ffmpeg):
    if ffmpeg and (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)):
        return ffmpeg
    return shutil.which("ffmpeg")


#: The smallest canvas h264_nvenc will open, per NVIDIA's H.264 NVENC spec and
#: MEASURED on this box 2026-07-28 (144x48 refused, 146x50 accepted).
_NVENC_MIN_W, _NVENC_MIN_H = 145, 49


def _read_sink(sink):
    """Whatever ffmpeg wrote to its error sink, best effort."""
    try:
        sink.seek(0)
        return sink.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 -- diagnostics must never raise
        return ""


def _reap(proc, sink=None):
    """Close the pipes and make sure ffmpeg is GONE.

    A refusal raised part-way through the frame stream used to leave the child
    running: it was still waiting for the rest of its input, still holding the
    OUTPUT FILE open. On Windows that also stops the caller deleting the
    directory it wrote into -- the first refusal test written against this
    encoder failed on a PermissionError from its own TemporaryDirectory
    cleanup, not on the refusal it was checking."""
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 -- best effort; never mask the refusal
        pass
    if sink is not None:
        try:
            sink.close()
        except OSError:
            pass


def _has_nvenc(ffmpeg):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-codecs"],
                             capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in (out.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def encode_silent_mp4(frames_iter, total, out_path, w, h, fps, ffmpeg):
    """Stream ``frames_iter`` to a SILENT bt709 / yuv420p H.264 mp4.

    THE SECOND CLIP ENCODER of this build (filed 2026-07-28): four live
    engines -- ``viz_green``, ``viz_camera``, ``viz_mxc_mandala`` and
    ``viz_mxc_cpu`` -- write every clip they emit through here.

    Three things this used to do that it no longer does, all of them silent:

    * ``total`` was accepted and NEVER READ. The signature promised a frame
      count and checked nothing, so a generator that stopped early wrote a
      short clip and the engines went on to stamp the requested number as
      ``CanonicalClip.frame_count`` -- the integer timing authority the
      composite positions the beat by. Now the frames are counted as they are
      piped, and a divergence is REFUSED by name.
    * The rawvideo ``-s`` was built from the CALLER'S ``w``/``h`` while the
      pipe carried whatever the generator actually painted. Two derivations of
      one fact: when they disagree ffmpeg slices the byte stream on the wrong
      boundaries and writes a skewed clip behind a clean exit code. The
      declared size now comes from the FIRST FRAME, and a caller whose ``w``/
      ``h`` disagree with it is refused rather than quietly re-cut. (This is
      the same defect class filed against ``ffmpeg_silent_mp4_cmd`` on
      2026-07-28, where ``even_dim()`` declares one size and ``tobytes()``
      pipes another -- it is NOT copied in here.)
    * A frame of a different shape or dtype midway through the generator went
      straight down the pipe and desynchronised everything after it.

    It still returns ``out_path`` alone: the count PROVEN AGAINST THE FILE is
    the caller's to take, because this encoder is also used for the
    post-upscale bars layer, which is a compositing input and not a
    CanonicalClip and owes no clip contract. Raises ``RuntimeError``, NAMED,
    and never falls back.
    """
    fb = find_ffmpeg(ffmpeg)
    if not fb:
        raise RuntimeError("scope_draw: ffmpeg not found.")
    frames = iter(frames_iter)
    try:
        first = np.ascontiguousarray(next(frames))
    except StopIteration:
        raise RuntimeError(
            "scope_draw: asked to encode ZERO frames to %r -- there is no clip "
            "here to write. NO FALLBACK (a zero-length beat is a planning bug "
            "upstream, not an empty video)." % out_path)
    if first.ndim != 3 or first.shape[2] != 3:
        raise RuntimeError(
            "scope_draw: expected (H,W,3) rgb24 frames for %r, got shape %r"
            % (out_path, (first.shape,)))
    if first.dtype != np.uint8:
        raise RuntimeError(
            "scope_draw: expected uint8 frames for %r, got dtype %r (%d bytes "
            "per sample): the rawvideo pipe is 8-bit, so this sends %dx the "
            "bytes and every frame boundary after the first is wrong."
            % (out_path, first.dtype, first.dtype.itemsize,
               first.dtype.itemsize))
    fh, fw = int(first.shape[0]), int(first.shape[1])
    if (fw, fh) != (int(w), int(h)):
        raise RuntimeError(
            "scope_draw: caller declared %dx%d for %r but the first frame is "
            "%dx%d. The rawvideo -s is the DECLARED size while the pipe "
            "carries the REAL bytes, so ffmpeg would slice the stream on the "
            "wrong boundaries and write a skewed clip behind a clean exit "
            "code. NO FALLBACK." % (int(w), int(h), out_path, fw, fh))
    # h264_nvenc REFUSES a canvas below its documented 145x49 minimum, and it
    # refuses it with "Error while opening encoder - maybe incorrect
    # parameters such as bit_rate, rate, width or height", which names four
    # things and not the one that is wrong. MEASURED on this box 2026-07-28:
    # 144x48 fails, 146x50 encodes, libx264 encodes every size tried from
    # 96x64 up. Selecting nvenc purely on "the box has it" therefore made a
    # small-canvas beat die on a machine WITH a GPU and succeed on one
    # without -- a box-dependent failure, found when the viz contract tests
    # stopped stubbing the encoder and rendered at their 96x64 fixture size.
    # This is codec SELECTION, not a fallback: both encoders emit the same
    # bt709 / yuv420p H.264 clip, and the caller proves the contract either
    # way. The dimension test is first so a small canvas skips the probe.
    use_nvenc = (fw >= _NVENC_MIN_W and fh >= _NVENC_MIN_H
                 and _has_nvenc(fb))
    cmd = [fb, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", "%dx%d" % (fw, fh), "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "h264_nvenc" if use_nvenc else "libx264"]
    if use_nvenc:
        cmd += ["-preset", "p5", "-rc", "vbr", "-b:v", "8M"]
    else:
        cmd += ["-preset", "medium", "-crf", "20"]
    cmd += ["-pix_fmt", "yuv420p", "-vsync", "cfr", "-r", str(fps),
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart", out_path]
    # ffmpeg's stderr goes to a TEMP FILE, not a PIPE. Nothing reads stderr
    # while the frame loop is writing, so a pipe deadlocks the moment ffmpeg
    # emits more than one OS buffer (~64KB on Windows) of error text: ffmpeg
    # blocks writing its own stderr, stops draining stdin, and our next
    # stdin.write() blocks too. That state raises NOTHING, so neither except
    # clause below ever runs and the child is never reaped -- the one exit
    # path where the "ffmpeg left alive holding the output file" defect could
    # still happen. A file sink has no buffer limit. (The FIRST encoder does
    # not have this hazard: it uses proc.communicate(), which drains
    # concurrently. This one must stream, so it cannot.)
    err_sink = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=err_sink)
    written = 0
    try:
        frame = first
        while True:
            arr = np.ascontiguousarray(frame)
            if arr.shape != first.shape or arr.dtype != first.dtype:
                raise RuntimeError(
                    "scope_draw: frame %d of %r is %r/%s but the clip was "
                    "declared %r/%s by frame 0 -- one odd frame desynchronises "
                    "the rawvideo pipe for every frame after it. NO FALLBACK."
                    % (written, out_path, (arr.shape,), arr.dtype,
                       (first.shape,), first.dtype))
            proc.stdin.write(arr.tobytes())
            written += 1
            try:
                frame = next(frames)
            except StopIteration:
                break
    except BrokenPipeError:
        # ffmpeg died mid-stream. Without this the caller sees BrokenPipeError
        # and never learns WHY ffmpeg quit, because the stderr goes unread.
        err = _read_sink(err_sink)
        _reap(proc, err_sink)
        raise RuntimeError(
            "scope_draw: ffmpeg closed the pipe after %d frame(s) of %r: %s"
            % (written, out_path, err[-800:]))
    except BaseException:
        _reap(proc, err_sink)
        raise
    proc.stdin.close()
    proc.wait()
    err = _read_sink(err_sink)
    err_sink.close()
    if proc.returncode != 0:
        raise RuntimeError("scope_draw: ffmpeg failed: %s" % err[-800:])
    if written != int(total):
        raise RuntimeError(
            "scope_draw: %r declared %d frame(s) but the generator produced "
            "%d. That declared number is what an engine stamps as "
            "CanonicalClip.frame_count -- the integer timing authority -- so "
            "the beat would drift against its own audio behind a clean exit "
            "code. NO FALLBACK." % (out_path, int(total), written))
    return out_path
