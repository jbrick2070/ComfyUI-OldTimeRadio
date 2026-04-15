r"""
OTR_SignalLostVideo - Procedural Audio-Reactive Video Engine
=============================================================

Generates a length-perfect MP4 from a finished SIGNAL LOST episode.
No AI video generation - pure math-driven procedural CRT aesthetic
rendered frame-by-frame and piped to ffmpeg.

Architecture:
  1. Master Clock: audio duration x FPS = exact frame count
  2. Audio Analysis: per-frame RMS volume + 32-bin FFT spectrum
  3. CRT Frame Renderer: title bar, procedural art centre,
     frequency bars, oscilloscope, scan lines, vignette
  4. MP4 Encoder: raw RGB frames piped to ffmpeg stdin + audio mux

The centre area is PURE PROCEDURAL ART - no script text, no lyrics,
no dialogue cards.  All visuals are driven by audio energy + math.

VRAM cost: zero. Runs entirely on CPU (numpy + PIL).

v2.0  2026-04-05  Jeffrey Brick
"""

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time as _time

import numpy as np
import torch

log = logging.getLogger("OTR")

# -----------------------------------------------------------------------------
# LAZY PIL IMPORT
# -----------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError(
        "Pillow is required for OTR_SignalLostVideo. "
        "Run: pip install Pillow"
    )

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
CRT_BG        = (8, 8, 16)         # deep navy-black
CRT_GREEN     = (0, 255, 65)       # phosphor green
CRT_AMBER     = (255, 176, 0)      # amber accent
CRT_DIM       = (0, 100, 28)       # dim green
CRT_DARK      = (0, 50, 14)        # very dim green
CRT_RED       = (255, 50, 50)      # alert red
CRT_WHITE     = (180, 200, 180)    # faded CRT white
CRT_CYAN      = (0, 200, 200)      # cyan accent
CRT_MAGENTA   = (200, 0, 200)      # magenta accent

# -----------------------------------------------------------------------------
# FONT LOADING
# -----------------------------------------------------------------------------
_FONT_CACHE = {}

def _load_font(size):
    """Load a monospace TTF font. Cached per size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = []
    if sys.platform == "win32":
        fd = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        candidates = [
            os.path.join(fd, "consola.ttf"),
            os.path.join(fd, "cour.ttf"),
            os.path.join(fd, "lucon.ttf"),
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except OSError:
                pass

    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# -----------------------------------------------------------------------------
# AUDIO ANALYSIS - per-frame RMS + FFT
# -----------------------------------------------------------------------------
def _analyze_audio(audio_np, sample_rate, total_frames, fps):
    """Return (volume_curve, freq_data, waveform_chunks).

    volume_curve    : list[float]       -- normalised 0-1 RMS per frame
    freq_data       : list[np.ndarray]  -- normalised 0-1 FFT bins per frame
    waveform_chunks : list[np.ndarray]  -- raw audio samples per frame
    """
    spf = sample_rate // fps  # samples per frame
    volume = []
    freqs = []
    waves = []

    for i in range(total_frames):
        s = i * spf
        e = min(s + spf, len(audio_np))
        chunk = audio_np[s:e] if s < len(audio_np) else np.zeros(spf)

        # RMS
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        volume.append(rms)

        # FFT - 32 bins
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

        # Waveform chunk (downsample to ~200 points for drawing)
        if len(chunk) > 200:
            idx = np.linspace(0, len(chunk) - 1, 200, dtype=int)
            waves.append(chunk[idx])
        else:
            waves.append(chunk)

    # Normalise
    vmax = max(volume) if volume and max(volume) > 0 else 1.0
    volume = [v / vmax for v in volume]

    fmax = max(np.max(f) for f in freqs) if freqs else 1.0
    if fmax > 0:
        freqs = [f / fmax for f in freqs]

    return volume, freqs, waves


# -----------------------------------------------------------------------------
# CRT FRAME RENDERER - Pure Procedural Art
# -----------------------------------------------------------------------------
class _CRTRenderer:
    """Generates audio-reactive procedural CRT frames.

    No script text, no dialogue, no lyrics.
    Centre area is pure math-driven generative art.
    """

    def __init__(self, w, h, title):
        self.w = w
        self.h = h
        self.title = title

        # Fonts
        self.f_title = _load_font(max(14, h // 30))
        self.f_sub   = _load_font(max(12, h // 38))
        self.f_small = _load_font(max(9, h // 72))

        # Pre-build scan-line overlay (RGBA)
        self._scanlines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(self._scanlines)
        step = max(2, h // 360)
        for y in range(0, h, step):
            sd.line([(0, y), (w, y)], fill=(0, 0, 0, 45))

        # Pre-build vignette multiplier
        cy, cx = h / 2.0, w / 2.0
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 / (cx ** 2) + (Y - cy) ** 2 / (cy ** 2))
        self._vignette = np.clip(1.0 - dist * 0.35, 0.45, 1.0).astype(np.float32)

        # Pre-compute ring geometry for the circular visualizer
        self._ring_cx = w // 2
        self._ring_cy = int(h * 0.42)
        self._ring_r = min(w, h) // 5

        # v1.5 Adaptive Brightness Gating - EMA smoothed volume tracker
        # Quiet scenes dim toward dark navy; loud scenes brighten to full phosphor.
        # EMA smoothing prevents flickering on transient spikes.
        self._brightness_ema = 0.5  # start at mid-brightness
        self._brightness_alpha = 0.08  # EMA smoothing factor (lower = smoother)

    # -- Public ----------------------------------------------------------

    def render(self, fi, total, fps, vol, freq, wave):
        """Render frame *fi* and return a PIL RGB Image."""
        img = Image.new("RGB", (self.w, self.h), CRT_BG)
        draw = ImageDraw.Draw(img)
        t = fi / fps
        dur = total / fps
        pad = self.w // 48

        # -- 1. Title bar ---------------------------------------------
        draw.text((pad, pad // 2), "=== SIGNAL LOST ===",
                  fill=CRT_GREEN, font=self.f_title)
        sub = f'"{self.title}"'
        draw.text((pad, pad // 2 + self.h // 18), sub,
                  fill=CRT_DIM, font=self.f_sub)
        # Divider
        ly = self.h // 10
        draw.line([(pad, ly), (self.w - pad, ly)], fill=CRT_DARK, width=1)

        # Timestamp (top-right)
        mm, ss = int(t // 60), int(t % 60)
        draw.text((self.w - pad - self.w // 10, pad // 2),
                  f"{mm:02d}:{ss:02d}", fill=CRT_DIM, font=self.f_sub)

        # -- 2. CIRCULAR FREQUENCY RING (centre) ---------------------
        cx, cy, base_r = self._ring_cx, self._ring_cy, self._ring_r
        r = base_r + int(vol * base_r * 0.3)
        n_bars = min(32, len(freq))
        for i in range(n_bars):
            angle = 2 * math.pi * i / n_bars - math.pi / 2
            bar_len = int(freq[i] * self.h * 0.18) + 2
            x0 = cx + int(r * math.cos(angle))
            y0 = cy + int(r * math.sin(angle))
            x1 = cx + int((r + bar_len) * math.cos(angle))
            y1 = cy + int((r + bar_len) * math.sin(angle))
            g = int(255 * (1.0 - freq[i] * 0.6))
            rb = int(freq[i] * 180)
            col = (rb, g, max(20, 65 - int(freq[i] * 50)))
            draw.line([(x0, y0), (x1, y1)], fill=col, width=max(2, self.w // 400))

        # Inner ring outline
        ring_bright = min(1.0, 0.3 + vol * 0.7)
        ring_col = tuple(min(255, int(c * ring_bright)) for c in CRT_GREEN)
        bbox = [(cx - r, cy - r), (cx + r, cy + r)]
        draw.ellipse(bbox, outline=ring_col, width=2)

        # -- 3. ORBITING PARTICLES ----------------------------------
        n_particles = 12
        for p in range(n_particles):
            phase = 2 * math.pi * p / n_particles
            orbit_r = r + int(self.h * 0.12) + int(vol * 30)
            speed = 0.3 + freq[p % len(freq)] * 2.0
            angle = phase + t * speed
            px = cx + int(orbit_r * math.cos(angle))
            py = cy + int(orbit_r * math.sin(angle) * 0.6)
            size = max(2, int(3 + freq[p % len(freq)] * 8))
            hue_shift = (p / n_particles + t * 0.05) % 1.0
            if hue_shift < 0.33:
                pcol = CRT_GREEN
            elif hue_shift < 0.66:
                pcol = CRT_CYAN
            else:
                pcol = CRT_AMBER
            bright = min(1.0, 0.3 + freq[p % len(freq)] * 0.7)
            pcol = tuple(min(255, int(c * bright)) for c in pcol)
            draw.ellipse([(px - size, py - size), (px + size, py + size)],
                         fill=pcol)

        # -- 4. GEOMETRIC GRID ----------------------------------------
        grid_step = max(40, self.w // 24)
        grid_alpha = max(8, int(15 + vol * 25))
        grid_col = (0, grid_alpha, int(grid_alpha * 0.4))
        for gx in range(pad, self.w - pad, grid_step):
            wobble = int(math.sin(gx * 0.01 + t * 2.0) * vol * 12)
            draw.line([(gx + wobble, ly + pad), (gx - wobble, self.h - pad * 2)],
                      fill=grid_col, width=1)
        for gy in range(ly + pad, self.h - pad * 2, grid_step):
            wobble = int(math.sin(gy * 0.01 + t * 1.5) * vol * 8)
            draw.line([(pad + wobble, gy), (self.w - pad - wobble, gy)],
                      fill=grid_col, width=1)

        # -- 5. MIRRORED WAVEFORM -------------------------------------
        wave_y = int(self.h * 0.72)
        wave_h = int(self.h * 0.12)
        if wave is not None and len(wave) > 1:
            self._waveform_mirror(draw, wave, pad, wave_y,
                                  self.w - pad * 2, wave_h, vol, t)

        # -- 6. FREQUENCY BARS ----------------------------------------
        bar_y = int(self.h * 0.86)
        bar_h = int(self.h * 0.06)
        if freq is not None:
            self._freq_bars_wide(draw, freq, pad, bar_y,
                                 self.w - pad * 2, bar_h, vol)

        # -- 7. Bottom bar --------------------------------------------
        by = self.h - pad
        draw.line([(pad, by - pad // 3), (self.w - pad, by - pad // 3)],
                  fill=CRT_DARK, width=1)
        draw.text((pad, by - pad // 6),
                  f"OTR v1.0  x  {self.w}x{self.h}  x  {fps}fps",
                  fill=CRT_DARK, font=self.f_small)
        draw.text((self.w - pad - self.w // 8, by - pad // 6),
                  f"frame {fi:05d}/{total:05d}",
                  fill=CRT_DARK, font=self.f_small)

        # -- 8. CRT post-processing ----------------------------------
        img = Image.alpha_composite(img.convert("RGBA"),
                                     self._scanlines).convert("RGB")

        arr = np.array(img, dtype=np.float32)
        arr *= self._vignette[:, :, np.newaxis]

        # -- 8b. Adaptive Brightness Gating - DISABLED --------------
        # Removed in v1.5.1: dimmed the CRT text to unreadable levels.
        # Full brightness preserved (matches v1.4 behavior).

        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        if vol > 0.3:
            arr = np.array(img, dtype=np.int16)
            intensity = int(vol * 12)
            noise = np.random.randint(-intensity, intensity + 1,
                                       arr.shape, dtype=np.int16)
            arr = np.clip(arr + noise, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        return img

    # -- Private drawing helpers -------------------------------------

    def _waveform_mirror(self, draw, wave, x, y, w, h, vol, t):
        mid = y + h // 2
        n = len(wave)
        pts_top = []
        pts_bot = []
        for i in range(min(n, w)):
            px = x + int(i * w / n)
            amp = wave[i] * h * 0.45
            pts_top.append((px, mid - int(amp)))
            pts_bot.append((px, mid + int(amp)))

        brightness = min(1.0, 0.3 + vol * 0.9)
        col_top = tuple(min(255, int(c * brightness)) for c in CRT_GREEN)
        col_bot = tuple(min(255, int(c * brightness * 0.5)) for c in CRT_CYAN)

        if len(pts_top) > 1:
            draw.line(pts_top, fill=col_top, width=2)
            draw.line(pts_bot, fill=col_bot, width=1)

        draw.line([(x, mid), (x + w, mid)], fill=CRT_DARK, width=1)

    def _freq_bars_wide(self, draw, freq, x, y, w, h, vol):
        n = min(32, len(freq))
        bw = w // n - 1
        for i in range(n):
            bh = max(1, int(freq[i] * h * 1.5))
            bx = x + i * (bw + 1)
            by = y + h - min(bh, h)
            ratio = i / max(1, n - 1)
            if ratio < 0.5:
                r = int(ratio * 2 * 255)
                g = 255
                b = 20
            else:
                r = 255
                g = int((1.0 - (ratio - 0.5) * 2) * 200)
                b = 20
            brightness = 0.25 + freq[i] * 0.75
            col = (min(255, int(r * brightness)),
                   min(255, int(g * brightness)),
                   min(255, int(b * brightness)))
            draw.rectangle([(bx, by), (bx + bw, y + h)], fill=col)


# -----------------------------------------------------------------------------
# FFMPEG ENCODER
# -----------------------------------------------------------------------------
def _find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


_NVENC_AVAILABLE = None  # cached after first check
_NVENC_LOCK = None       # lazy-init threading.Lock()

def _check_nvenc(ffmpeg_path):
    """Return True if ffmpeg supports h264_nvenc on this system.

    Thread-safe: uses a lock so parallel ComfyUI workers can't race on the
    first check. Safe when ffmpeg_path is None (returns False immediately).
    """
    global _NVENC_AVAILABLE, _NVENC_LOCK
    # Lazy-init the lock to avoid import-time threading overhead
    if _NVENC_LOCK is None:
        import threading
        _NVENC_LOCK = threading.Lock()

    with _NVENC_LOCK:
        if _NVENC_AVAILABLE is not None:
            return _NVENC_AVAILABLE
        if not ffmpeg_path:
            _NVENC_AVAILABLE = False
            log.info("[Video] Encoder: CPU libx264 (ffmpeg path unknown)")
            return False
        try:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-codecs"],
                capture_output=True, text=True, timeout=5,
            )
            _NVENC_AVAILABLE = "h264_nvenc" in result.stdout
        except Exception:
            _NVENC_AVAILABLE = False
        tag = "NVIDIA h264_nvenc" if _NVENC_AVAILABLE else "CPU libx264"
        log.info("[Video] Encoder: %s", tag)
        return _NVENC_AVAILABLE


def _encode_mp4(frames_iter, total_frames, audio_path, output_path,
                width, height, fps):
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install via: winget install ffmpeg  "
            "(or add ffmpeg to PATH)"
        )

    use_nvenc = _check_nvenc(ffmpeg)
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "h264_nvenc" if use_nvenc else "libx264",
    ]

    if use_nvenc:
        # NVIDIA hardware encoder: quality VBR, ~8 Mbps target
        cmd.extend(["-preset", "slow", "-rc", "vbr", "-b:v", "8M"])
    else:
        # CPU software encoder: slower but always available
        cmd.extend(["-preset", "medium", "-crf", "20"])

    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ])

    log.info("[Video] Launching ffmpeg: %s", " ".join(cmd[:6]) + " ...")

    import tempfile as _tf
    stderr_file = _tf.NamedTemporaryFile(
        mode="w+b", prefix="otr_ffmpeg_", suffix=".log", delete=False
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
    )

    t0 = _time.time()
    for i, frame in enumerate(frames_iter):
        proc.stdin.write(frame.tobytes())
        if i % max(1, fps * 10) == 0:
            elapsed = _time.time() - t0
            pct = int(100 * i / total_frames)
            log.info("[Video] Encoding: frame %d/%d (%d%%) -- %.1fs elapsed",
                     i, total_frames, pct, elapsed)

    proc.stdin.close()
    proc.wait()

    stderr_file.seek(0)
    stderr_text = stderr_file.read().decode(errors="replace")
    stderr_file.close()
    try:
        os.remove(stderr_file.name)
    except OSError:
        pass

    if proc.returncode != 0:
        log.error("[Video] ffmpeg failed:\n%s", stderr_text[-2000:])
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    elapsed = _time.time() - t0
    log.info("[Video] Encode complete: %d frames in %.1fs (%.1f fps)",
             total_frames, elapsed, total_frames / elapsed if elapsed > 0 else 0)

    return output_path


# -----------------------------------------------------------------------------
# TELEMETRY HUD - Post-roll treatment card
# Rendered AFTER the episode audio ends.  No spoilers during playback.
#
# Layout:
#   LEFT  (30 %) - static: title, genre, news seed, cast & voices
#   DIV         - 1 px phosphor-green divider
#   RIGHT (70 %) - scrolling classified transcript (scene arc + full script)
# -----------------------------------------------------------------------------

def _fh(font, pad=4):
    """Line height for *font* (PIL getbbox-safe)."""
    try:
        bb = font.getbbox("Mg")
        return bb[3] - bb[1] + pad
    except Exception:
        return 16 + pad


def _fw(text, font):
    """Text pixel width for *font* (PIL getbbox-safe)."""
    if not text:
        return 0
    try:
        bb = font.getbbox(str(text))
        return max(0, bb[2] - bb[0])
    except Exception:
        return len(str(text)) * 8


def _draw_wrapped(draw, text, x, y, max_w, font, fill, lh, indent=0):
    """Word-wrap *text* into *draw* starting at (x, y). Returns new y."""
    words = str(text).split()
    line_buf = []
    for word in words:
        test = " ".join(line_buf + [word])
        if _fw(test, font) > max_w and line_buf:
            draw.text((x + indent, y), " ".join(line_buf), fill=fill, font=font)
            y += lh
            line_buf = [word]
        else:
            line_buf.append(word)
    if line_buf:
        draw.text((x + indent, y), " ".join(line_buf), fill=fill, font=font)
        y += lh
    return y


def _get_latest_telemetry():
    """Parse the otr_runtime.log for the most recent VRAM and Speed stats."""
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "otr_runtime.log")
    
    # Defaults
    peak_gb = "???"
    speed = "???"
    model = "UNKNOWN CORE"
    
    if not os.path.exists(log_path):
        return peak_gb, speed, model
        
    try:
        import re
        re_vram = re.compile(r"VRAM_SNAPSHOT.*?peak_gb=([0-9.]+)")
        re_speed = re.compile(r"DONE:\s+.*?([0-9.]+)\s+tok/s")
        re_llm = re.compile(r"LLM loaded:\s+([^\s]+)")
        
        # Read last ~200 lines (enough for one generation cycle)
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            end_pos = f.tell()
            f.seek(max(0, end_pos - 15000))
            lines = f.readlines()
            
        for line in lines:
            m_vram = re_vram.search(line)
            if m_vram: peak_gb = m_vram.group(1)
            
            m_speed = re_speed.search(line)
            if m_speed: speed = m_speed.group(1)
            
            m_llm = re_llm.search(line)
            if m_llm: model = m_llm.group(1).split("/")[-1].upper()
            
    except Exception as e:
        log.warning(f"[Telemetry] Parser failed: {e}")
        
    return peak_gb, speed, model

def _parse_hud_data(episode_title, script_json_str, production_plan_json_str,
                    news_used, duration_s, W, H):
    """Return a clean data dict for *_TelemetryHUDRenderer*."""
    import time as _t
    try:
        script = (json.loads(script_json_str)
                  if isinstance(script_json_str, str) else (script_json_str or []))
        plan   = (json.loads(production_plan_json_str)
                  if isinstance(production_plan_json_str, str) else (production_plan_json_str or {}))
        if not isinstance(script, list):
            script = []
    except Exception:
        script, plan = [], {}

    # Voice assignments - same normalisation as _write_story_treatment
    voices = {}
    for k, v in (plan.get("voice_assignments", {}) or {}).items():
        if isinstance(v, dict):
            voices[str(k)] = str(v.get("voice_preset", v.get("preset", v.get("voice", ""))))
        else:
            voices[str(k)] = str(v)

    # News seeds (handle JSON-list format)
    news_seeds = []
    _nr = (news_used or "").strip()
    if _nr.startswith("["):
        try:
            news_seeds = [
                str(s.get("headline", s) if isinstance(s, dict) else s)
                for s in json.loads(_nr)[:2]
            ]
        except Exception:
            news_seeds = [_nr[:100]]
    elif _nr:
        news_seeds = [_nr.split("\n")[0][:100]]

    # Cast
    cast = [{"char": c, "preset": p, "desc": _PRESET_DESC.get(p, "")}
            for c, p in sorted(voices.items())]

    # Scenes from Canonical 1.0 item stream
    scenes, cur = [], {"scene_num": "1", "env": "", "items": []}
    for item in script:
        t = item.get("type", "")
        if t == "scene_break":
            if cur["items"] or cur["env"]:
                scenes.append(cur)
            cur = {"scene_num": str(item.get("scene", len(scenes) + 2)),
                   "env": "", "items": []}
        elif t == "environment":
            cur["env"] = str(item.get("description", ""))
        elif t == "dialogue":
            char = str(item.get("character_name", item.get("character", "?")))
            cur["items"].append({
                "type": "dialogue",
                "char": char,
                "text": str(item.get("line", item.get("text", ""))),
                "preset": voices.get(char, ""),
            })
        elif t == "sfx":
            cur["items"].append({
                "type": "sfx",
                "text": str(item.get("description", item.get("text", ""))),
            })
        elif t == "pause":
            cur["items"].append({"type": "pause"})
    if cur["items"] or cur["env"]:
        scenes.append(cur)

    peak_gb, speed, model = _get_latest_telemetry()
    
    return {
        "title":      episode_title,
        "genre":      plan.get("genre_flavor", plan.get("genre", "sci-fi")),
        "produced":   _t.strftime("%Y-%m-%d  %H:%M"),
        "duration_s": duration_s,
        "resolution": f"{W}x{H}",
        "news_seeds": news_seeds,
        "cast":       cast,
        "scenes":     scenes,
        "telemetry":  {"peak": peak_gb, "speed": speed, "model": model}
    }


class _TelemetryHUDRenderer:
    """Post-roll Telemetry HUD frame generator.

    Pre-renders both panels at init time; render() is cheap (crop + paste).
    """

    _SCROLL_PPS = 65  # pixels per second (comfortable reading speed)

    def __init__(self, w, h, fps, data):
        self.w, self.h, self.fps = w, h, fps
        self.data = data

        self.LEFT_W  = max(280, int(w * 0.36))
        self.DIV_X   = self.LEFT_W
        self.RIGHT_X = self.LEFT_W + 2
        self.RIGHT_W = w - self.LEFT_W - 2
        self.P       = max(8, w // 120)

        self.f_head  = _load_font(max(22, h // 28))
        self.f_label = _load_font(max(16, h // 42))
        self.f_body  = _load_font(max(14, h // 50))
        self.f_small = _load_font(max(13, h // 58))

        self._lhH = _fh(self.f_head)
        self._lhL = _fh(self.f_label)
        self._lhB = _fh(self.f_body)
        self._lhS = _fh(self.f_small)

        self._left  = self._build_left()
        self._right, self._right_h = self._build_right()

        # Scanline overlay (full canvas)
        self._scanlines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        _sl = ImageDraw.Draw(self._scanlines)
        for sy in range(0, h, max(2, h // 360)):
            _sl.line([(0, sy), (w, sy)], fill=(0, 0, 0, 40))

    # -- Public API --------------------------------------------------------

    def hud_frames(self):
        """Total frame count for the HUD post-roll (20-90 s)."""
        scroll_px = max(0, self._right_h - self.h)
        secs = scroll_px / self._SCROLL_PPS + 8.0
        return int(max(20.0, min(90.0, secs)) * self.fps)

    def render(self, fi, total_hud_frames):
        """Return a PIL RGB Image for HUD frame *fi*."""
        img = Image.new("RGB", (self.w, self.h), CRT_BG)

        # Static left panel
        img.paste(self._left, (0, 0))

        # Divider
        d = ImageDraw.Draw(img)
        d.line([(self.DIV_X, 0), (self.DIV_X, self.h)], fill=CRT_DIM, width=1)

        # Scrolling right panel
        s_start = int(total_hud_frames * 0.08)
        s_end   = int(total_hud_frames * 0.92)
        if fi <= s_start:
            frac = 0.0
        elif fi >= s_end:
            frac = 1.0
        else:
            frac = (fi - s_start) / max(1, s_end - s_start)

        sy  = int(frac * max(0, self._right_h - self.h))
        bot = min(sy + self.h, self._right.height)
        if bot > sy:
            crop = self._right.crop((0, sy, self.RIGHT_W, bot))
            if crop.height < self.h:
                pad_img = Image.new("RGB", (self.RIGHT_W, self.h), CRT_BG)
                pad_img.paste(crop, (0, 0))
                crop = pad_img
            img.paste(crop, (self.RIGHT_X, 0))

        # Fade in / out (6 % of total)
        fade_f = max(1, int(total_hud_frames * 0.06))
        if fi < fade_f:
            black = Image.new("RGB", (self.w, self.h), (0, 0, 0))
            img = Image.blend(black, img, fi / fade_f)
        elif fi > total_hud_frames - fade_f:
            black = Image.new("RGB", (self.w, self.h), (0, 0, 0))
            img = Image.blend(black, img, (total_hud_frames - fi) / fade_f)

        # Scanlines
        img = Image.alpha_composite(img.convert("RGBA"),
                                    self._scanlines).convert("RGB")
        return img

    # -- Panel builders ----------------------------------------------------

    def _build_left(self):
        img = Image.new("RGB", (self.LEFT_W, self.h), CRT_BG)
        d, P = ImageDraw.Draw(img), self.P
        y = P

        # Title
        d.text((P, y), "SIGNAL LOST", fill=CRT_GREEN, font=self.f_head)
        y += self._lhH
        d.text((P, y), "EPISODE TREATMENT", fill=CRT_DIM, font=self.f_body)
        y += self._lhB + P
        d.line([(P, y), (self.LEFT_W - P, y)], fill=CRT_DARK, width=1)
        y += P * 2

        # Metadata
        d.text((P, y), "METADATA", fill=CRT_AMBER, font=self.f_label)
        y += self._lhL
        lbl_w = _fw("DATE  ", self.f_body)
        for lbl, val in [
            ("TITLE", self.data.get("title", "?")),
            ("GENRE", self.data.get("genre", "?")),
            ("LEN",   f'{self.data.get("duration_s", 0) / 60:.1f} min'),
            ("RES",   self.data.get("resolution", "?")),
            ("DATE",  self.data.get("produced", "")[:10]),
        ]:
            d.text((P, y), lbl, fill=CRT_DIM, font=self.f_body)
            d.text((P + lbl_w, y), str(val), fill=CRT_WHITE, font=self.f_body)
            y += self._lhB
        y += P
        d.line([(P, y), (self.LEFT_W - P, y)], fill=CRT_DARK, width=1)
        y += P * 2

        # News seed
        d.text((P, y), "NEWS SEED", fill=CRT_AMBER, font=self.f_label)
        y += self._lhL
        for seed in self.data.get("news_seeds", [])[:2]:
            d.text((P, y), seed, fill=CRT_DIM, font=self.f_body)
            y += self._lhB + P // 2
        y += P
        d.line([(P, y), (self.LEFT_W - P, y)], fill=CRT_DARK, width=1)
        y += P * 2

        # Telemetry Block
        d.text((P, y), "SYSTEM TELEMETRY", fill=CRT_AMBER, font=self.f_label)
        y += self._lhL
        
        telemetry = self.data.get("telemetry", {})
        speed_raw = telemetry.get('speed', '???')
        # Ensure it fits elegantly
        speed_str = f"{float(speed_raw):.1f} T/s" if speed_raw.replace('.', '', 1).isdigit() else f"{speed_raw}"
        
        for lbl, val in [
            ("CORE",  telemetry.get("model", "UNKNOWN")),
            ("FLUX",  speed_str),
            ("MEM",   f"{telemetry.get('peak', '???')} GB")
        ]:
            d.text((P, y), lbl, fill=CRT_DARK, font=self.f_body)
            d.text((P + lbl_w, y), str(val), fill=CRT_GREEN, font=self.f_body)
            y += self._lhB
            
        y += P
        d.line([(P, y), (self.LEFT_W - P, y)], fill=CRT_DARK, width=1)
        y += P * 2

        # Cast & voices - stop rendering if we're close to the footer
        footer_y = self.h - self._lhS - P * 2  # reserve space for footer
        if y < footer_y:
            d.text((P, y), "CAST & VOICES", fill=CRT_AMBER, font=self.f_label)
            y += self._lhL
        for m in self.data.get("cast", []):
            if y + self._lhB >= footer_y:
                break  # no more room - don't draw over the footer
            d.text((P, y), m.get("char", "?"), fill=CRT_GREEN, font=self.f_body)
            y += self._lhB
            preset = m.get("preset", "")
            if preset and y + self._lhS < footer_y:
                d.text((P * 2, y), preset, fill=CRT_DIM, font=self.f_small)
                y += self._lhS
            desc = m.get("desc", "")
            if desc and y + self._lhS < footer_y:
                char_w = max(1, _fw("m", self.f_small))
                trunc = desc[:(self.LEFT_W - P * 3) // char_w]
                if len(trunc) < len(desc):
                    trunc = trunc.rstrip() + "-"
                d.text((P * 2, y), trunc, fill=CRT_DARK, font=self.f_small)
                y += self._lhS
            y += P // 2

        # Footer - always anchored to bottom regardless of cast overflow
        fy = self.h - self._lhS - P
        d.line([(P, fy - P), (self.LEFT_W - P, fy - P)], fill=CRT_DARK, width=1)
        d.text((P, fy), "OTR v1.0", fill=CRT_DARK, font=self.f_small)
        return img

    def _build_right(self):
        """Pre-render the full scrollable transcript. Returns (img, content_h)."""
        P, RW = self.P, self.RIGHT_W
        scenes = self.data.get("scenes", [])
        n_items = sum(len(s.get("items", [])) for s in scenes)
        est_h = self.h + (len(scenes) * 4 + n_items * 5) * self._lhB + self.h
        est_h = max(est_h, self.h * 3)

        img = Image.new("RGB", (RW, est_h), CRT_BG)
        d   = ImageDraw.Draw(img)
        y   = self.h // 4   # breathing room above first line

        # Transcript header
        d.text((P, y), "[ CLASSIFIED TRANSCRIPT ]", fill=CRT_GREEN, font=self.f_head)
        y += self._lhH
        d.text((P, y), f"EPISODE  //  {self.data.get('title','?').upper()}",
               fill=CRT_AMBER, font=self.f_label)
        y += self._lhL + P
        d.line([(P, y), (RW - P, y)], fill=CRT_DIM, width=1)
        y += P * 3

        for scene in scenes:
            sc_num = scene.get("scene_num", "1")
            env    = scene.get("env", "")

            d.text((P, y), f"--  SCENE {sc_num}", fill=CRT_AMBER, font=self.f_label)
            y += self._lhL
            if env:
                d.text((P * 3, y), f"ENV: {env}", fill=CRT_DIM, font=self.f_body)
                y += self._lhB
            y += P
            d.line([(P, y), (RW // 2, y)], fill=CRT_DARK, width=1)
            y += P * 2

            for item in scene.get("items", []):
                kind = item.get("type", "")
                if kind == "dialogue":
                    char   = item.get("char", "?")
                    preset = item.get("preset", "")
                    char_tag = f"{char}  [{preset}]" if preset else char
                    d.text((P, y), char_tag, fill=CRT_GREEN, font=self.f_label)
                    y += self._lhL
                    y = _draw_wrapped(d, item.get("text", ""),
                                      P * 3, y, RW - P * 4,
                                      self.f_body, CRT_WHITE, self._lhB)
                    y += P
                elif kind == "sfx":
                    d.text((P, y), f"[SFX]  {item.get('text','')[:80]}",
                           fill=CRT_CYAN, font=self.f_body)
                    y += self._lhB + P // 2
                elif kind == "pause":
                    d.text((P, y), "[ . . . ]", fill=CRT_DARK, font=self.f_body)
                    y += self._lhB
            y += P * 3

        # Sci-Fi Telemetry Easter Egg
        telemetry = self.data.get("telemetry", {})
        y += P * 2
        easter_egg = f">> DIAGNOSTIC MEMORY ALLOCATION PEAKED AT {telemetry.get('peak', '???')}GB. NEURAL FLUX MAINTAINED AT {telemetry.get('speed', '???')} T/S. CONTAINMENT FIELD HOLDING."
        y = _draw_wrapped(d, easter_egg, P, y, RW - P * 4, self.f_body, CRT_CYAN, self._lhB)
        
        # Footer
        y += P * 4
        d.line([(P, y), (RW - P, y)], fill=CRT_DIM, width=1)
        y += P * 2
        d.text((P, y), "[ END OF CLASSIFIED TRANSCRIPT ]",
               fill=CRT_DIM, font=self.f_label)
        y += self._lhL
        d.text((P, y), "SIGNAL LOST  //  ALL RIGHTS RESERVED",
               fill=CRT_DARK, font=self.f_small)
        y += self._lhS + P * 6

        return img, y


# -----------------------------------------------------------------------------
# COMFYUI NODE
# -----------------------------------------------------------------------------


# -- Story Treatment Writer --------------------------------------------------

_PRESET_DESC = {
    "v2/en_speaker_0": "male * authoritative * deep (ANNOUNCER voice)",
    "v2/en_speaker_1": "male * warm * conversational",
    "v2/en_speaker_2": "male * calm * measured",
    "v2/en_speaker_3": "male * gruff * weathered",
    "v2/en_speaker_4": "female * bright * energetic",
    "v2/en_speaker_5": "male * casual * warm",
    "v2/en_speaker_6": "male * deep * resonant",
    "v2/en_speaker_7": "male * sharp * anxious",
    "v2/en_speaker_8": "male * clipped * precise",
    "v2/en_speaker_9": "female * mature * authoritative",
    "v2/de_speaker_0": "male * German accent * precise * clipped",
    "v2/de_speaker_4": "female * German accent * clear * analytical",
    "v2/fr_speaker_0": "male * French accent * smooth * baritone",
    "v2/fr_speaker_4": "female * French accent * warm * elegant",
    "v2/es_speaker_0": "male * Spanish accent * warm * authoritative",
    "v2/es_speaker_9": "female * Spanish accent * mature * expressive",
    "v2/it_speaker_0": "male * Italian accent * dramatic * animated",
    "v2/it_speaker_4": "female * Italian accent * expressive * warm",
    "v2/pt_speaker_0": "male * Portuguese accent * soft * thoughtful",
    "v2/pt_speaker_4": "female * Portuguese accent * gentle * clear",
}


def _write_story_treatment(out_path, episode_title, script_json_str,
                            production_plan_json_str, news_used,
                            duration, W, H, fps, size_mb):
    """Save a complete episode treatment alongside the MP4.

    Includes full script in scene order, voice assignments with
    human-readable descriptions, scene arc, and production stats.
    Same information as the mission-control log but formatted as a
    permanent show-bible record.
    """
    try:
        import time as _t
        import json as _json

        script = _json.loads(script_json_str) if isinstance(script_json_str, str) else (script_json_str or [])
        plan   = _json.loads(production_plan_json_str) if isinstance(production_plan_json_str, str) else (production_plan_json_str or {})
        if not isinstance(script, list):
            script = []

        # Normalize voice_assignments: values may be dicts like {"voice_preset": "v2/en_speaker_0", ...}
        voices_raw = plan.get("voice_assignments", {})
        voices = {}
        if isinstance(voices_raw, dict):
            for k, v in voices_raw.items():
                if isinstance(v, dict):
                    voices[str(k)] = str(v.get("voice_preset", v.get("preset", v.get("voice", str(v)))))
                else:
                    voices[str(k)] = str(v)

        genre  = plan.get("genre_flavor", plan.get("genre", "sci-fi radio drama"))
        ts     = _t.strftime("%Y-%m-%d  %H:%M:%S")
        BAR    = "\u2500" * 64
        DBAR   = "\u2550" * 64

        out = []
        W_ = lambda s="": out.append(str(s))

        # Header
        W_(DBAR)
        W_("  SIGNAL LOST  \u00b7  EPISODE TREATMENT")
        W_(DBAR)
        W_()
        W_(f'  Title    :  "{episode_title}"')
        W_(f"  Genre    :  {genre}")
        W_(f"  Produced :  {ts}")
        W_()

        # News seed - may arrive as a JSON list ["headline 1", "headline 2", ...]
        W_("NEWS SEED")
        W_(BAR)
        _news_raw = (news_used or "").strip()
        if _news_raw.startswith("["):
            try:
                _seeds = _json.loads(_news_raw)
                news_clean = " | ".join(
                    str(s.get("headline", s) if isinstance(s, dict) else s)[:80]
                    for s in _seeds[:3]
                ) if _seeds else ""
            except Exception:
                news_clean = _news_raw[:120]
        else:
            news_clean = _news_raw.split("\n")[0][:120]
        W_(f"  {news_clean if news_clean else '(no news seed - custom premise used)'}")
        W_()

        # Cast & voices
        W_("CAST & VOICES")
        W_(BAR)
        if voices:
            pad_w = max((len(str(k)) for k in voices), default=10)
            for char in sorted(voices.keys()):
                preset = voices[char]
                desc   = _PRESET_DESC.get(preset, preset)
                W_(f"  {str(char):<{pad_w}}  \u2192  {preset:<24}  {desc}")
        else:
            W_("  (no voice assignments recorded)")
        W_()

        # Scene arc summary - build from scene_break / environment / dialogue items
        scenes = {}
        _cur_sc = "1"
        _cur_env = ""
        for item in script:
            t = item.get("type", "")
            if t == "scene_break":
                _cur_sc = str(item.get("scene", _cur_sc))
            elif t == "environment":
                _cur_env = str(item.get("description", ""))
            if _cur_sc not in scenes:
                scenes[_cur_sc] = {"env": _cur_env, "sfx": [], "d": 0}
            if t == "environment":
                scenes[_cur_sc]["env"] = _cur_env
            elif t == "sfx":
                scenes[_cur_sc]["sfx"].append(str(item.get("description", item.get("text", ""))))
            elif t == "dialogue":
                scenes[_cur_sc]["d"] += 1

        W_("SCENE ARC")
        W_(BAR)
        if scenes:
            for sc_num, sc in sorted(scenes.items()):
                W_(f"  Scene {sc_num}  \u00b7  {(sc['env'] or '').strip()}")
                for sfx in sc["sfx"]:
                    W_(f"    [SFX]  {sfx}")
                W_(f"    {sc['d']} dialogue lines")
                W_()
        else:
            W_("  (scene data unavailable)")
            W_()

        # Full script in scene order - Canonical 1.0 item types:
        #   scene_break - {"type":"scene_break","scene":"1"}
        #   environment - {"type":"environment","description":"..."}
        #   dialogue    - {"type":"dialogue","character_name":"...","line":"..."}
        #   sfx         - {"type":"sfx","description":"..."}
        #   pause       - {"type":"pause","kind":"beat","duration_ms":200}
        d_count = sum(1 for i in script if i.get("type") == "dialogue")
        s_count = sum(1 for i in script if i.get("type") == "sfx")
        W_(f"FULL SCRIPT  ({d_count} dialogue  \u00b7  {s_count} sfx cues)")
        W_(BAR)
        cur_scene = None
        cur_env = ""
        scene_header_written = False
        for item in script:
            kind = item.get("type", "")
            if kind == "scene_break":
                cur_scene = str(item.get("scene", cur_scene or "1"))
                scene_header_written = False
            elif kind == "environment":
                cur_env = str(item.get("description", "")).strip()
                if not scene_header_written:
                    if cur_scene is None:
                        cur_scene = "1"
                    W_()
                    W_(f"  \u2500\u2500 SCENE {cur_scene}  \u00b7  {cur_env}")
                    W_()
                    scene_header_written = True
            elif kind == "dialogue":
                if not scene_header_written:
                    if cur_scene is None:
                        cur_scene = "1"
                    W_()
                    W_(f"  \u2500\u2500 SCENE {cur_scene}  \u00b7  {cur_env}")
                    W_()
                    scene_header_written = True
                char   = str(item.get("character_name", item.get("character", "?")))
                text   = str(item.get("line", item.get("text", ""))).strip()
                preset = voices.get(char, "")
                desc   = _PRESET_DESC.get(preset, "")
                vtag   = f"  [{preset}]" if preset else ""
                W_(f"  {char}{vtag}")
                W_(f"    {text}")
                W_()
            elif kind == "sfx":
                sfx_text = str(item.get("description", item.get("text", ""))).strip()
                W_(f"  [SFX]  {sfx_text}")
                W_()
            elif kind == "pause":
                pause_kind = item.get("kind", "beat")
                W_(f"  [PAUSE/{pause_kind.upper()}]")
                W_()
        W_()

        # Production
        W_("PRODUCTION")
        W_(BAR)
        W_(f"  Duration    :  {duration/60:.1f} min  ({duration:.1f} s)")
        W_(f"  Resolution  :  {W}x{H} @ {fps} fps")
        W_(f"  File        :  {os.path.basename(out_path)}")
        W_(f"  Size        :  {size_mb:.1f} MB")
        W_()
        W_(DBAR)
        W_()
        
        peak_gb, speed, model = _get_latest_telemetry()
        W_(f">> DIAGNOSTIC MEMORY ALLOCATION PEAKED AT {peak_gb}GB. NEURAL FLUX MAINTAINED AT {speed} T/S. CONTAINMENT FIELD HOLDING.")
        W_()

        treatment_path = out_path.replace(".mp4", "_treatment.txt")
        with open(treatment_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out))

        log.info("[Video] Treatment saved: %s", os.path.basename(treatment_path))
        return treatment_path

    except Exception as exc:
        log.warning("[Video] Story treatment write failed: %s", exc)
        try:
            from .story_orchestrator import _runtime_log
            _runtime_log(f"Video: TREATMENT WRITE FAILED -- {exc}")
        except Exception:
            pass
        return None

class SignalLostVideoRenderer:
    """Generate a procedural CRT-aesthetic MP4 from an OTR episode."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "render_video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "script_json": ("STRING", {
                    "multiline": True, "default": "[]",
                    "tooltip": "Parsed script JSON (pipeline compat)"
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "Production plan JSON (pipeline compat)"
                }),
                "news_used": ("STRING", {
                    "multiline": True, "default": "[]",
                    "tooltip": "News JSON (pipeline compat)"
                }),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 24, "min": 12, "max": 60, "step": 1,
                    "tooltip": "Frames per second (24 = cinematic)"
                }),
                "resolution": (["1920x1080", "1280x720", "3840x2160"], {
                    "default": "1920x1080",
                    "tooltip": "Output video resolution"
                }),
                "episode_title": ("STRING", {
                    "default": "",
                    "tooltip": "Episode title for the title bar. Normally resolved from the script_json title token; widget acts as a last-resort override."
                }),
                "closing_audio": ("AUDIO", {
                    "tooltip": "Unique closing music from MusicGen for the credits post-roll. If not connected, a gentle decay from the episode audio is used instead of looping."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _time.time()

    def render_video(self, audio, script_json, production_plan_json,
                     news_used, fps=24, resolution="1920x1080",
                     episode_title="", closing_audio=None):

        from .story_orchestrator import _runtime_log

        # -- 1. Parse inputs & Extract Smart Title -------------------
        # BUG-LOCAL-035: Title resolution is now primary-source script_json,
        # fallback widget. Previously the list-format branch did nothing and
        # every run silently fell through to the widget default
        # "The Last Frequency", yielding 100% TITLE_STUCK filenames.
        # Resolution order:
        #   1. title token in list-format script_json  (canonical path)
        #   2. top-level 'title' key in dict-format script_json (legacy)
        #   3. widget episode_title (manual override / last resort)
        # Any value matching _STUCK_TITLE_DEFAULTS triggers TITLE_RESOLVE_FAIL
        # so the run fails loud instead of writing a stuck-default filename.
        _STUCK_TITLE_DEFAULTS = {
            "", "the last frequency", "untitled", "episode",
            "signal lost", "custom episode",
        }

        import json as _json
        _title_source = "widget"
        _widget_title = (episode_title or "").strip()
        _script_title = ""
        try:
            _script_data = _json.loads(script_json) if isinstance(script_json, str) else (script_json or [])
            if isinstance(_script_data, list):
                # Canonical 1.0: look for a {"type": "title", "value": "..."} token.
                # LLMScriptWriter.write_script prepends this in v2.0-alpha.
                for _tok in _script_data:
                    if isinstance(_tok, dict) and _tok.get("type") == "title":
                        _cand = (_tok.get("value") or "").strip()
                        if _cand:
                            _script_title = _cand
                            break
            elif isinstance(_script_data, dict) and "title" in _script_data:
                _script_title = (_script_data.get("title") or "").strip()
        except Exception as _te:
            log.warning("[Video] Title extraction from script_json failed: %s", _te)

        if _script_title and _script_title.lower() not in _STUCK_TITLE_DEFAULTS:
            episode_title = _script_title
            _title_source = "script_json"
        elif _widget_title and _widget_title.lower() not in _STUCK_TITLE_DEFAULTS:
            episode_title = _widget_title
            _title_source = "widget_override"
        else:
            # Fail loud: we'd write a stuck-default filename again. Surface it.
            _runtime_log(
                f"TITLE_TRACE | source=NONE | script_json_title='{_script_title}' "
                f"| widget_title='{_widget_title}' -> TITLE_RESOLVE_FAIL"
            )
            raise RuntimeError(
                f"TITLE_RESOLVE_FAIL: no usable episode title. "
                f"script_json_title='{_script_title}', widget='{_widget_title}'. "
                f"Writer must emit a {{\"type\":\"title\"}} token or the "
                f"widget must be set to a non-stuck value."
            )

        _runtime_log(
            f"TITLE_TRACE | source={_title_source} | resolved='{episode_title}' "
            f"| widget='{_widget_title}' | script_json='{_script_title}'"
        )

        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        if waveform.dim() == 3:
            audio_np = waveform[0].mean(dim=0).cpu().numpy()
        elif waveform.dim() == 2:
            audio_np = waveform.mean(dim=0).cpu().numpy()
        else:
            audio_np = waveform.cpu().numpy()

        duration = len(audio_np) / sr
        W, H = [int(x) for x in resolution.split("x")]
        total_frames = int(math.ceil(duration * fps))

        log.info("[Video] Starting render: %.1fs audio -> %d frames @ %dfps (%s)",
                 duration, total_frames, fps, resolution)
        _runtime_log(f"Video: Starting {total_frames} frames @ {fps}fps ({resolution})")

        # -- 2. Audio analysis ----------------------------------------
        _runtime_log("Video: Analysing audio (FFT + RMS)")
        volume, freqs, waves = _analyze_audio(audio_np, sr, total_frames, fps)

        # -- 2b. Build post-roll Telemetry HUD (no VRAM, pure PIL) ----
        try:
            _hud_data     = _parse_hud_data(episode_title, script_json,
                                            production_plan_json, news_used,
                                            duration, W, H)
            _hud_renderer = _TelemetryHUDRenderer(W, H, fps, _hud_data)
            _hud_frames   = _hud_renderer.hud_frames()
        except Exception as _he:
            log.warning("[Video] HUD build failed (post-roll skipped): %s", _he)
            _hud_renderer = None
            _hud_frames   = 0

        # -- 3. Save audio to temp WAV for ffmpeg ---------------------
        import tempfile
        import wave as wave_mod

        tmp_wav = os.path.join(tempfile.gettempdir(), "otr_video_audio.wav")
        pcm = (audio_np * 32767).astype(np.int16)
        # Use unique closing music for HUD post-roll instead of looping.
        # Priority: closing_audio from MusicGen > gentle decay from episode tail.
        if _hud_frames > 0:
            hud_samples = int(_hud_frames / fps * sr)
            if closing_audio is not None:
                # -- Unique closing music from MusicGen --------------------
                _cw = closing_audio["waveform"]
                _csr = closing_audio["sample_rate"]
                if _cw.dim() == 3:
                    _c_np = _cw[0].mean(dim=0).cpu().numpy()
                elif _cw.dim() == 2:
                    _c_np = _cw.mean(dim=0).cpu().numpy()
                else:
                    _c_np = _cw.cpu().numpy()
                # Resample to episode SR if needed
                if _csr != sr:
                    import scipy.signal as _sig
                    _c_np = _sig.resample(
                        _c_np, int(len(_c_np) * sr / _csr)
                    ).astype(np.float32)
                # If closing cue is shorter than HUD duration, pad with
                # a gentle tail decay (NOT a loop) then silence.
                if len(_c_np) < hud_samples:
                    # Fade out the last 2 seconds of the cue
                    fade_out = min(int(2 * sr), len(_c_np) // 2)
                    _c_np[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
                    # Pad remainder with silence
                    _c_np = np.pad(_c_np, (0, hud_samples - len(_c_np)))
                else:
                    _c_np = _c_np[:hud_samples]
                    # Fade out last 2 seconds
                    fade_out = min(int(2 * sr), hud_samples // 4)
                    _c_np[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
                hud_wave = _c_np * 0.45  # duck to 45% (louder than old 35% loop)
                # Smooth 1-second crossfade from episode audio
                join_in = min(sr, hud_samples)
                hud_wave[:join_in] *= np.linspace(0, 1, join_in, dtype=np.float32)
                log.info("[Video] Credits music: unique MusicGen closing cue (%.1fs)",
                         len(hud_wave) / sr)
            else:
                # -- Fallback: gentle one-shot decay from episode tail -----
                # Take last ~10s, apply a long exponential fade-out so it
                # decays naturally into silence. NO looping.
                tail_len = min(len(audio_np), int(10 * sr))
                tail_src = audio_np[-tail_len:].copy().astype(np.float32)
                # Apply exponential decay over the entire tail
                decay = np.exp(-np.linspace(0, 5, tail_len)).astype(np.float32)
                tail_src *= decay
                # Pad with silence to fill HUD duration
                if len(tail_src) < hud_samples:
                    tail_src = np.pad(tail_src, (0, hud_samples - len(tail_src)))
                else:
                    tail_src = tail_src[:hud_samples]
                hud_wave = tail_src * 0.35
                # Smooth join
                join_in = min(sr, hud_samples)
                hud_wave[:join_in] *= np.linspace(0, 1, join_in, dtype=np.float32)
                log.info("[Video] Credits music: episode tail decay (no loop, %.1fs)",
                         len(hud_wave) / sr)
            hud_pcm = np.clip(hud_wave * 32767, -32767, 32767).astype(np.int16)
            pcm_out = np.concatenate([pcm, hud_pcm])
        else:
            pcm_out = pcm
        with wave_mod.open(tmp_wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_out.tobytes())

        # -- 4. Determine output path ---------------------------------
        out_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "ComfyUI",
            "output", "old_time_radio"
        )
        os.makedirs(out_dir, exist_ok=True)

        ts = _time.strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in episode_title)
        safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
        out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")

        # -- 5. Build frame generator ---------------------------------
        renderer = _CRTRenderer(W, H, episode_title)
        total_encode_frames = total_frames + _hud_frames

        def _render_crt(fi):
            return renderer.render(fi, total_frames, fps, volume[fi], freqs[fi], waves[fi])

        def _render_hud(hi):
            return _hud_renderer.render(hi, _hud_frames)

        def _frame_gen():
            import concurrent.futures
            max_workers = min(32, (os.cpu_count() or 4) + 4)
            chunk_size = max_workers * 2  # Hard limit RAM footprint

            # Main audio-reactive content
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for start in range(0, total_frames, chunk_size):
                    end = min(start + chunk_size, total_frames)
                    for j, frame in enumerate(executor.map(_render_crt, range(start, end))):
                        fi = start + j
                        yield frame
                        if fi % (fps * 30) == 0 and fi > 0:
                            _runtime_log(f"Video: {fi}/{total_frames} frames rendered")

            # Post-roll Telemetry HUD (no spoilers - plays after audio ends)
            if _hud_renderer is not None and _hud_frames > 0:
                _runtime_log(f"Video: Treatment HUD - {_hud_frames} frames")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for start in range(0, _hud_frames, chunk_size):
                        end = min(start + chunk_size, _hud_frames)
                        for frame in executor.map(_render_hud, range(start, end)):
                            yield frame

        # -- 6. Encode ------------------------------------------------
        _runtime_log(f"Video: Encoding MP4 via ffmpeg -> {os.path.basename(out_path)}")
        _encode_mp4(_frame_gen(), total_encode_frames, tmp_wav, out_path, W, H, fps)

        try:
            os.remove(tmp_wav)
        except OSError:
            pass

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log.info("[Video] Saved: %s (%.1f MB, %.1fs, %d frames total)",
                 out_path, size_mb, duration + (_hud_frames / fps), total_encode_frames)
        _runtime_log(f"Video: DONE -- {os.path.basename(out_path)} ({size_mb:.1f} MB)")

        # Write story treatment companion file
        _write_story_treatment(
            out_path, episode_title, script_json,
            production_plan_json, news_used,
            duration, W, H, fps, size_mb
        )

        # Return using the standard ComfyUI 'gifs' key so the canvas
        # spawns an HTML5 video player widget automatically.
        return {
            "ui": {"gifs": [{"filename": os.path.basename(out_path),
                             "subfolder": "old_time_radio",
                             "type": "output"}]},
            "result": (out_path,),
        }
