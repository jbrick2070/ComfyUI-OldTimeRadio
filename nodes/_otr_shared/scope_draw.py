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


# --------------------------------------------------------------------------- #
# Silent ffmpeg encode (copied; SILENT -- only OTR_MasterAudioMux adds audio).
# --------------------------------------------------------------------------- #
def find_ffmpeg(ffmpeg):
    if ffmpeg and (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)):
        return ffmpeg
    return shutil.which("ffmpeg")


def _has_nvenc(ffmpeg):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-codecs"],
                             capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in (out.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def encode_silent_mp4(frames_iter, total, out_path, w, h, fps, ffmpeg):
    fb = find_ffmpeg(ffmpeg)
    if not fb:
        raise RuntimeError("scope_draw: ffmpeg not found.")
    use_nvenc = _has_nvenc(fb)
    cmd = [fb, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", "%dx%d" % (w, h), "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "h264_nvenc" if use_nvenc else "libx264"]
    if use_nvenc:
        cmd += ["-preset", "p5", "-rc", "vbr", "-b:v", "8M"]
    else:
        cmd += ["-preset", "medium", "-crf", "20"]
    cmd += ["-pix_fmt", "yuv420p", "-vsync", "cfr", "-r", str(fps),
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in frames_iter:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("scope_draw: ffmpeg failed: %s" % err[-800:])
    return out_path
