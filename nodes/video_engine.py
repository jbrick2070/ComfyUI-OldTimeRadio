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
import re as _re
import shutil
import subprocess
import sys
import time as _time

import numpy as np
import torch

# Shared hero-title-card arithmetic. RELATIVE with a bare-name fallback, never
# `from nodes...`: an absolute package import resolves in the test suite (where
# the repo root is on sys.path) and RAISES on a live server, which is lesson L23
# -- three shared modules shipped that way and one of them silently made five
# lanes motion-exempt in production while every test stayed green.
try:
    from . import _otr_title_card as _OTRTC  # type: ignore
except ImportError:  # loaded with nodes/ on sys.path
    import _otr_title_card as _OTRTC  # type: ignore

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


def _draw_crt_overstrike_text(
    draw,
    xy,
    text,
    *,
    fill,
    font,
    stroke_width=0,
    stroke_fill=(0, 0, 0),
):
    """Draw CRT fake-bold text with an optional outline.

    Pillow's stroke support gives the hero title a real dark edge while the
    overstrike keeps the existing phosphor-bold look.
    """
    x, y = xy
    kwargs = {"fill": fill, "font": font}
    if int(stroke_width) > 0:
        kwargs.update({
            "stroke_width": int(stroke_width),
            "stroke_fill": stroke_fill,
        })
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        draw.text((x + ox, y + oy), text, **kwargs)


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
# TITLE-CARD TIMING -- resolve the b000 music-open window from the ledger
# -----------------------------------------------------------------------------
# The SceneSequencer stamps led["lines"] with speaker_role + start_s + dur_s
# (seconds). The opening music line is the carrier for the hero title card.
_MUSIC_OPEN_ROLES = ("music_open", "music_visual")
_SPEECH_ROLES_VIDEO = ("announcer", "character")


def _envelope_intro_end(volume, fps, cap):
    """Heuristic intro end-frame from the volume envelope: the first sustained
    quiet dip AFTER the opening swell, within ``cap`` frames. 0 = undetectable
    (the title card then stays disabled rather than guess)."""
    v = np.asarray(volume, dtype=np.float32)
    n = min(len(v), int(cap))
    if n < int(fps):  # need at least ~1s of envelope
        return 0
    seg = v[:n]
    thr = 0.25
    risen = False
    for i in range(n):
        if seg[i] > 0.45:
            risen = True
        elif risen and seg[i] < thr:
            return max(int(fps), i)
    return 0


def _resolve_title_timing(led, volume, fps, total_frames):
    """Resolve the hero-title-card window from the ledger lines.

    Returns a dict consumed by ``_CRTRenderer`` (``music_open_start_f`` /
    ``music_open_end_f`` / ``first_dialogue_f`` / ``kind``) or ``{}`` when
    nothing resolves (card disabled, no crash).

    Primary: the first line whose ``speaker_role`` is a music-open role, using
    its ``start_s``/``dur_s`` in SECONDS -> ``round(start_s*fps) ..
    round((start_s+dur_s)*fps)``. Fallback (wire ledger may carry
    ``start_s=None``): derive the intro window from the volume envelope (music
    from frame 0 to the first dialogue onset), capped."""
    try:
        lines = (led or {}).get("lines") or []
    except Exception:  # noqa: BLE001
        lines = []
    fps = int(fps) if fps else 24
    total = int(total_frames or 0)
    if total <= 0:
        return {}

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    music_open = None
    first_dialogue_f = None
    music_close = None
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        role = str(ln.get("speaker_role") or "")
        if music_open is None and role in _MUSIC_OPEN_ROLES:
            music_open = ln
        if role == "music_close":
            music_close = ln  # last wins
        if first_dialogue_f is None and role in _SPEECH_ROLES_VIDEO:
            s = _f(ln.get("start_s"))
            if s is not None:
                first_dialogue_f = int(round(s * fps))

    out = {"kind": "open"}

    # -- OPEN window --------------------------------------------------
    resolved_open = False
    if music_open is not None:
        s = _f(music_open.get("start_s"))
        d = _f(music_open.get("dur_s"))
        if s is not None and d is not None and d > 0:
            start_f = max(0, int(round(s * fps)))
            end_f = min(total, int(round((s + d) * fps)))
            if start_f < end_f:
                out["music_open_start_f"] = start_f
                out["music_open_end_f"] = end_f
                if first_dialogue_f is not None:
                    out["first_dialogue_f"] = first_dialogue_f
                resolved_open = True
    if not resolved_open:
        # Fallback: volume-envelope intro window (wire start_s unavailable).
        cap = min(total, int(fps * 12))  # <=12s intro ceiling
        if first_dialogue_f is not None:
            # A real onset at frame zero means there is no head gap for an
            # opening card. It is known timing, not a missing-overlay sentinel.
            end_f = min(max(0, first_dialogue_f), cap)
            out["first_dialogue_f"] = first_dialogue_f
        else:
            # FIX3 / BUG-LOCAL-404 guard: reaching the volume-envelope heuristic
            # means NO music_open line resolved AND no first-dialogue onset --
            # i.e. the lines carry no start_s (the audio-timing overlay is
            # missing). That previously collapsed the card to ~1s SILENTLY;
            # make it LOUD so a missing overlay can never quietly recur. The
            # render_video overlay_audio_timing() call should keep this
            # unreachable on a normal run.
            end_f = _envelope_intro_end(volume, fps, cap)
            log.warning(
                "[Video] title-card timing DEGRADE (BUG-404 guard): no "
                "music_open line and no first-dialogue onset -- lines carry no "
                "start_s, so the card window falls back to the volume-envelope "
                "heuristic (end_f=%s). The audio-timing overlay should make "
                "this unreachable; investigate the ledger timing for this run.",
                end_f)
        if end_f and end_f > 0:
            out["music_open_start_f"] = 0
            out["music_open_end_f"] = int(end_f)
            if first_dialogue_f is not None:
                out["first_dialogue_f"] = first_dialogue_f

    # -- CLOSE window (S5 conditional outro bookend) ------------------
    # Only resolves with real start_s/dur_s on the last music_close line and
    # only when it lands inside total_frames (the credits HUD post-roll runs
    # AFTER total). No envelope fallback for the close.
    if music_close is not None:
        s = _f(music_close.get("start_s"))
        d = _f(music_close.get("dur_s"))
        if s is not None and d is not None and d > 0:
            cstart = int(round(s * fps))
            cend = int(round((s + d) * fps))
            if 0 <= cstart < cend <= total:
                out["music_close_start_f"] = cstart
                out["music_close_end_f"] = cend

    # SYNTHESIZED end title (operator 2026-06-17: "flash a big title at the end
    # just like the start -- animate gibberish -> the actual episode title").
    # The closing theme lives in the credits post-roll (AFTER total), so a real
    # music_close window almost never lands inside total and the close hero card
    # never fired. Synthesize one in the last seconds of the MAIN video so the
    # SAME _draw_title_card decode-reveals the EPISODE TITLE before the credits
    # roll. Duration mirrors the open when known (else ~4s), clamped 2-6s and
    # never overrunning the open window.
    if "music_close_start_f" not in out and total > int(fps * 2):
        open_end = int(out.get("music_open_end_f", 0))
        if "music_open_start_f" in out:
            close_dur = int(out["music_open_end_f"]) - int(out["music_open_start_f"])
        else:
            close_dur = int(fps * 4)
        close_dur = max(int(fps * 2), min(int(fps * 6), close_dur))
        cstart = max(open_end, total - close_dur)
        if cstart < total:
            out["music_close_start_f"] = cstart
            out["music_close_end_f"] = total
            out.setdefault("close_synthesized", True)

    if "music_open_start_f" in out or "music_close_start_f" in out:
        return out
    return {}


def _hero_title_in_procgen():
    """Should the hero title be drawn INTO the procgen frame? Default: NO.

    The procgen layer is composited `screen` + `green_only`, which can only
    LIGHTEN, so a black outline there is the blend's identity and the title
    measured 1.13:1 over a lit monitor. The legible copy is emitted as ASS by
    OTR_CaptionBurn, downstream of that blend.

    ``OTR_HERO_TITLE_IN_PROCGEN=1`` selects the LEGACY arm as a whole -- the
    hero is drawn here again AND ``title_card_plan()`` returns None, so no ASS
    title is emitted either. Both halves matter: a flag that only restored the
    draw would render the title twice and the "before" picture it exists to
    produce would contain the "after" as well.

    The TITLE-FREE control the acceptance measurement needs is a different and
    cheaper thing -- with the suppression in place, node 93's own output is
    already title-free, so the brightest-underlying-source ROI is read from
    there rather than from a third render.
    """
    return str(os.environ.get("OTR_HERO_TITLE_IN_PROCGEN", "")).strip().lower() \
        in {"1", "true", "yes", "on"}


def _title_reveal_progress(fi, w0, me, in_dock, reveal_frac=0.4):
    """BUG-LOCAL-409: decode/reveal progress for the hero title card.

    The reveal COMPLETES in the first ``reveal_frac`` of the ``[w0, me)`` window
    and then HOLDS solid (p == 1.0) until the POP/dock -- previously it stretched
    across the WHOLE window, so the title only resolved on the final frame (the
    operator saw scramble for the entire duration). Returns p in [0.0, 1.0].

    The implementation moved to ``_otr_title_card`` when the title gained a
    SECOND consumer (the ASS emitter downstream of the procgen blend). This name
    stays because it is the public one the tests and the renderer already use;
    there is exactly one implementation behind it, which is the point.
    """
    return _OTRTC.title_reveal_progress(fi, w0, me, in_dock, reveal_frac)


# -----------------------------------------------------------------------------
# CRT FRAME RENDERER - Pure Procedural Art
# -----------------------------------------------------------------------------
class _CRTRenderer:
    """Generates audio-reactive procedural CRT frames.

    No script text, no dialogue, no lyrics.
    Centre area is pure math-driven generative art.
    """

    def __init__(self, w, h, title, volume, freqs, waves, fps, timing=None):
        self.w = w
        self.h = h
        self.title = title
        self.fps = int(fps) if fps else 24

        # -- Audio arrays (convert lists -> np FIRST; the EMA precompute and
        #    the bounded-lookback trails index them every frame) ----------
        self.volume = np.asarray(volume, dtype=np.float32) if len(volume) else \
            np.zeros(0, dtype=np.float32)
        self.freqs = list(freqs)   # list[np.ndarray] (32-bin)
        self.waves = list(waves)   # list[np.ndarray] (~200 pt)
        self.total = len(self.volume)
        self.timing = dict(timing or {})

        # Fonts
        self._title_size = max(14, h // 30)
        self._hero_size_max = max(28, h // 12)  # 2-3x f_title (#1 spec ceiling)
        self.f_title = _load_font(self._title_size)
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

        # -- Title-bar layout (geometry from w/h, never baked 1920) -------
        # These coords are the section-1 ident/subtitle/timestamp anchors AND
        # the dock target for the hero card (#1 D: the docked state IS the
        # normal section-1 draw).
        self._pad = w // 48
        self._ident_xy = (self._pad, self._pad // 2)
        self._sub_xy   = (self._pad, self._pad // 2 + h // 18)
        self._ts_x     = w - self._pad - w // 10
        self._divider_y = h // 10

        # -- DUAL signal-strength EMA (the conductor; #concept) -----------
        # signal: slow ambient brightness (alpha ~0.05); trig: fast lock /
        # glitch (alpha ~0.3); loss = 1 - signal. signal[0]=trig[0]=vol[0].
        # READ-ONLY downstream -- never multiplied into the whole frame
        # (that was the v1.5.1 unreadable-text bug).
        sig = np.zeros(self.total, dtype=np.float32)
        trg = np.zeros(self.total, dtype=np.float32)
        if self.total > 0:
            v0 = float(self.volume[0])
            sig[0] = trg[0] = v0
            a_s, a_t = 0.05, 0.30
            for i in range(1, self.total):
                vi = float(self.volume[i])
                sig[i] = sig[i - 1] + a_s * (vi - sig[i - 1])
                trg[i] = trg[i - 1] + a_t * (vi - trg[i - 1])
        self._signal = sig
        self._trig = trg
        self._loss = (1.0 - sig).astype(np.float32)

        # -- Title card windows (decode->reveal->POP->dock); [] disables --
        self._cards = self._resolve_card_windows(self.timing)

    # -- Deterministic per-frame RNG ------------------------------------
    def _rng(self, fi, salt):
        """A stable-hash-seeded Generator (fixes the v1 unseeded section-8
        noise). seed = blake2s(title|fi|salt); same inputs -> same frame.

        Shared with the title planner: the ASS emitter must be able to reproduce
        the hero scramble of a frame it never rendered.
        """
        return _OTRTC.decode_rng(self.title, fi, salt)

    # -- Title card window resolution -----------------------------------
    def _resolve_card_windows(self, timing):
        """Return a list of hero-title-card window dicts (open + optional
        close), each: {start_f, music_end_f, dock_frames, end_f, kind}.
        Computes dock_frames so a window never overruns the first dialogue or
        `total`. Missing fields -> the window is simply absent (no crash)."""
        if not timing or self.total <= 0:
            return []
        total = int(self.total)
        half = int(self.fps * 0.5)
        cards = []

        # OPEN: docks down into the section-1 ident before the first dialogue.
        w0 = timing.get("music_open_start_f")
        w1 = timing.get("music_open_end_f")
        if w0 is not None and w1 is not None:
            w0 = max(0, int(w0))
            w1 = min(total, int(w1))
            if w1 > w0:
                first_dialogue_f = timing.get("first_dialogue_f")
                if first_dialogue_f is not None:
                    dock = max(0, min(half, int(first_dialogue_f) - w1))
                else:
                    dock = max(0, min(half, total - w1))
                cards.append({
                    "start_f": w0, "music_end_f": w1,
                    "dock_frames": dock, "end_f": min(total, w1 + dock),
                    "kind": "open",
                })

        # CLOSE (S5 bookend): reveal + POP, no dock (it rides into the credits
        # tail); only when it resolved inside total_frames.
        c0 = timing.get("music_close_start_f")
        c1 = timing.get("music_close_end_f")
        if c0 is not None and c1 is not None:
            c0 = max(0, int(c0))
            c1 = min(total, int(c1))
            if c1 > c0:
                cards.append({
                    "start_f": c0, "music_end_f": c1,
                    "dock_frames": 0, "end_f": c1, "kind": "close",
                })
        return cards

    # -- Public ----------------------------------------------------------

    def render(self, fi, draw_scopes=True):
        """Render frame *fi* and return a PIL RGB Image.

        Draw order (S1): background (grid + gated scopes + bottom bar) ->
        section-8 CRT post (scanlines/vignette/noise, numpy) -> THEN the
        section-1 ident OR the hero title card, drawn AFTER the vignette so
        the always-on vignette can never dim the text (the v1.5.1 bug)."""
        total = self.total
        fps = self.fps
        if total <= 0:
            return Image.new("RGB", (self.w, self.h), CRT_BG)
        fi = max(0, min(int(fi), total - 1))
        vol = float(self.volume[fi])
        freq = self.freqs[fi]
        wave = self.waves[fi]
        signal = float(self._signal[fi])
        loss = float(self._loss[fi])
        t = fi / fps
        pad = self._pad
        ly = self._divider_y

        # Is a hero title card active this frame? Decide BEFORE drawing so
        # section-1 ident/subtitle/timestamp are suppressed while it owns the
        # screen (the docked state after the window IS the normal section-1).
        card = None
        for c in self._cards:
            if c["start_f"] <= fi < c["end_f"]:
                card = c
                break
        card_active = card is not None

        img = Image.new("RGB", (self.w, self.h), CRT_BG)
        draw = ImageDraw.Draw(img)

        # -- divider (background chrome) ------------------------------
        draw.line([(pad, ly), (self.w - pad, ly)], fill=CRT_DARK, width=1)

        # -- 2/3/5/6. GATED SCOPE SECTIONS ----------------------------
        # draw_scopes=False (v2 scene-aware path) skips {2,3,5,6} as a SET:
        # section 3 reads section 2's `r`, so 2+3 go together; 5+6 are
        # independent. {1,4,7,8} (title/grid/bottom/CRT) always draw.
        if draw_scopes:
            # -- 2. CIRCULAR FREQUENCY RING (centre) ------------------
            cx, cy, base_r = self._ring_cx, self._ring_cy, self._ring_r
            r = base_r + int(vol * base_r * 0.3)
            # S4 sync-drift: in weak signal (silent gaps) the receiver loses
            # lock -> a bounded horizontal coordinate offset (NOT np.roll, NOT
            # a hue shift). Clamped so the ring never leaves the frame.
            drift = int(round(loss * (self.w // 120)))
            cx = max(r, min(self.w - r, cx + drift))
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
                draw.line([(x0, y0), (x1, y1)], fill=col,
                          width=max(2, self.w // 400))

            # Inner ring outline
            ring_bright = min(1.0, 0.3 + vol * 0.7)
            ring_col = tuple(min(255, int(c * ring_bright)) for c in CRT_GREEN)
            bbox = [(cx - r, cy - r), (cx + r, cy + r)]
            draw.ellipse(bbox, outline=ring_col, width=2)

            # -- 3. ORBITING PARTICLES -------------------------------
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
        # S4 brightness hierarchy: the grid dims FIRST when signal weakens
        # (the silent inter-beat gaps) -- it scales down faster than the ident
        # (which is drawn after the vignette at full brightness). signal-keyed
        # so it is deterministic.
        grid_step = max(40, self.w // 24)
        grid_alpha = max(6, int((15 + vol * 25) * (0.35 + 0.65 * signal)))
        grid_col = (0, grid_alpha, int(grid_alpha * 0.4))
        for gx in range(pad, self.w - pad, grid_step):
            wobble = int(math.sin(gx * 0.01 + t * 2.0) * vol * 12)
            draw.line([(gx + wobble, ly + pad), (gx - wobble, self.h - pad * 2)],
                      fill=grid_col, width=1)
        for gy in range(ly + pad, self.h - pad * 2, grid_step):
            wobble = int(math.sin(gy * 0.01 + t * 1.5) * vol * 8)
            draw.line([(pad + wobble, gy), (self.w - pad - wobble, gy)],
                      fill=grid_col, width=1)

        if draw_scopes:
            # -- 5. MIRRORED WAVEFORM ---------------------------------
            wave_y = int(self.h * 0.72)
            wave_h = int(self.h * 0.12)
            if wave is not None and len(wave) > 1:
                self._waveform_mirror(draw, wave, pad, wave_y,
                                      self.w - pad * 2, wave_h, vol, t)

            # -- 6. FREQUENCY BARS ------------------------------------
            bar_y = int(self.h * 0.86)
            bar_h = int(self.h * 0.06)
            if freq is not None:
                self._freq_bars_wide(draw, freq, pad, bar_y,
                                     self.w - pad * 2, bar_h, vol)

        # -- 7. Bottom bar (dim chrome; stays pre-vignette) -----------
        by = self.h - pad
        draw.line([(pad, by - pad // 3), (self.w - pad, by - pad // 3)],
                  fill=CRT_DARK, width=1)
        draw.text((pad, by - pad // 6),
                  f"OTR v1.0  x  {self.w}x{self.h}  x  {fps}fps",
                  fill=CRT_DARK, font=self.f_small)
        draw.text((self.w - pad - self.w // 8, by - pad // 6),
                  f"frame {fi:05d}/{total:05d}",
                  fill=CRT_DARK, font=self.f_small)

        # -- 8. CRT post-processing (numpy; runs UNDER the text) ------
        img = Image.alpha_composite(img.convert("RGBA"),
                                     self._scanlines).convert("RGB")

        arr = np.array(img, dtype=np.float32)
        arr *= self._vignette[:, :, np.newaxis]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        if vol > 0.3:
            arr = np.array(img, dtype=np.int16)
            intensity = int(vol * 12)
            # Seeded (was unseeded np.random in v1) so the frame is
            # deterministic per (title, fi).
            noise = self._rng(fi, "noise").integers(
                -intensity, intensity + 1, size=arr.shape, dtype=np.int16)
            arr = np.clip(arr + noise, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        # -- 1. SECTION-1 TEXT / HERO CARD (drawn AFTER the vignette) -
        # Text-exemption: the always-on vignette multiply above can never dim
        # the ident or the hero card because they composite last.
        draw = ImageDraw.Draw(img)
        if card_active:
            self._draw_title_card(draw, fi, card, signal, loss)
        else:
            self._draw_ident(draw, fi, t)

        return img

    # -- Section-1 ident (normal / docked state) ------------------------
    def _draw_ident(self, draw, fi, t):
        pad = self._pad
        draw.text(self._ident_xy, "=== SIGNAL LOST ===",
                  fill=CRT_GREEN, font=self.f_title)
        draw.text(self._sub_xy, f'"{self.title}"',
                  fill=CRT_DIM, font=self.f_sub)
        mm, ss = int(t // 60), int(t % 60)
        draw.text((self._ts_x, pad // 2),
                  f"{mm:02d}:{ss:02d}", fill=CRT_DIM, font=self.f_sub)

    # -- Hero title card (#1: decode -> reveal -> POP -> dock) ----------
    # The scramble alphabet moved to _otr_title_card with the decode itself, so
    # this name is an ALIAS, not a second copy -- two alphabets would silently
    # desynchronise the drawn frame from the planned one.
    _DECODE_GLYPHS = _OTRTC.DECODE_GLYPHS

    def _tw(self, draw, s, font):
        b = draw.textbbox((0, 0), s, font=font)
        return b[2] - b[0]

    def _th(self, draw, s, font):
        b = draw.textbbox((0, 0), s, font=font)
        return b[3] - b[1]

    def _wrap_words(self, draw, text, font, max_w):
        """Greedy word-wrap so each line fits max_w (a single over-long word
        is left to overflow -- the floor font already caps the size)."""
        words = text.split()
        if not words:
            return [text]
        lines, cur = [], words[0]
        for wd in words[1:]:
            if self._tw(draw, cur + " " + wd, font) <= max_w:
                cur += " " + wd
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
        return lines

    def _fit_hero(self, draw, title):
        """Shrink the hero font (from f_hero ceiling to a floor) + word-wrap
        until the block fits max_w ~= 0.8w / max_h ~= 0.28h. Returns
        (font, lines, size)."""
        title = (title or "SIGNAL").upper()
        max_w = int(self.w * 0.8)
        max_h = int(self.h * 0.28)
        floor = max(16, self.h // 40)
        size = self._hero_size_max
        while True:
            font = _load_font(size)
            lines = self._wrap_words(draw, title, font, max_w)
            tw = max((self._tw(draw, ln, font) for ln in lines), default=0)
            th = sum(self._th(draw, ln, font) for ln in lines)
            if (tw <= max_w and th <= max_h) or size <= floor:
                return font, lines, size
            size -= max(2, size // 12)

    def _decode_line(self, text, reveal_n, rng):
        """Left-to-right decode: the first reveal_n chars are solid; the rest
        are seeded scramble (spaces stay spaces).

        Delegates to the shared planner so the frame this renderer draws and the
        frame the ASS emitter plans scramble IDENTICALLY -- two implementations
        of the same seeded shuffle would drift the moment either was touched.
        """
        return _OTRTC.decode_line(text, reveal_n, rng)

    # -- Title-card plan (the ASS emitter's input) -----------------------

    def _hero_measure(self, draw):
        """Return (measure_text, fit_hero) closures over THIS renderer's PIL
        state, in the shape the shared planner asks for.

        Measurement is the only part of title planning that needs fonts, so it
        is the only part that stays here. ``fit_hero(title, size=N)`` re-wraps at
        an exact size, which is what the dock needs as it shrinks the hero onto
        the ident.
        """
        def measure_text(text, size):
            font = _load_font(max(8, int(size)))
            return (self._tw(draw, text, font), self._th(draw, text or "M", font))

        def fit_hero(title, size=None):
            if size is None:
                font, lines, resolved = self._fit_hero(draw, title)
                return (lines, resolved)
            font = _load_font(max(8, int(size)))
            return (self._wrap_words(draw, title, font, int(self.w * 0.8)),
                    int(size))

        return measure_text, fit_hero

    def title_card_plan(self, main_frames=None):
        """Build the serializable per-frame hero-title plan for this episode.

        Handed downstream to ``OTR_CaptionBurn``, which renders it as ASS AFTER
        the procgen blend -- the only place an outline is not a no-op. Returns
        ``None`` when this episode has no card at all, which is a legitimate
        state (no resolved music window) and NOT a failure.

        Also ``None`` under ``OTR_HERO_TITLE_IN_PROCGEN=1``. That flag selects
        the LEGACY arm whole: the title goes back into the procgen layer and no
        ASS title is emitted. Without this the flag drew the hero TWICE -- baked
        in AND burned on top -- so the "control" render it exists to produce was
        never a clean before-picture and the contrast comparison it feeds would
        have measured the new title against itself.
        """
        if not self._cards or _hero_title_in_procgen():
            return None
        img = Image.new("RGB", (self.w, self.h), CRT_BG)
        draw = ImageDraw.Draw(img)
        measure_text, fit_hero = self._hero_measure(draw)
        try:
            reveal_frac = float(os.environ.get("OTR_TITLE_REVEAL_FRACTION", "0.4"))
        except (TypeError, ValueError):
            reveal_frac = 0.4
        return _OTRTC.build_title_plan(
            title=(self.title or "SIGNAL").upper(),
            # The draw seeds its scramble from the RAW title while drawing the
            # uppercased one. Reproducing the frames means reproducing that.
            rng_title=self.title,
            cards=self._cards,
            fps=self.fps,
            frame_size=(self.w, self.h),
            ident_xy=self._ident_xy,
            title_size=self._title_size,
            # MAIN frames, never the encoded total: the encode appends the HUD
            # post-roll, and a closing card clamped to the encoded length would
            # hang the title over the post-roll and into the credits.
            main_frames=int(self.total if main_frames is None else main_frames),
            measure_text=measure_text,
            fit_hero=fit_hero,
            reveal_frac=reveal_frac,
        )

    def _draw_carrier_meter(self, draw, level):
        """A broken-phosphor carrier-strength meter under the ident: blocks
        crawl to solid on the signal swell (level 0..1)."""
        pad = self._pad
        n = 16
        bw = max(3, self.w // 160)
        gap = max(2, bw // 2)
        x0 = self._sub_xy[0]
        y0 = self._sub_xy[1] + self.h // 22
        filled = int(round(max(0.0, min(1.0, level)) * n))
        for i in range(n):
            x = x0 + i * (bw + gap)
            on = i < filled
            # broken phosphor: a couple of dim cells even when "on"
            col = CRT_GREEN if (on and i % 5 != 4) else CRT_DARK
            draw.rectangle([(x, y0), (x + bw, y0 + bw)], fill=col)

    def _draw_title_card(self, draw, fi, card, signal, loss):
        """The b000 music-open hero card. Phases inside the window:
        reveal [start, music_end) -> POP (last 1-2 reveal frames) -> dock
        [music_end, end). The docked target IS the section-1 ident."""
        w0 = card["start_f"]
        me = card["music_end_f"]
        dock_frames = card["dock_frames"]
        in_dock = fi >= me
        # BUG-LOCAL-409: resolve the reveal in the first fraction of the window,
        # then hold the solid title until the POP/dock (env OTR_TITLE_REVEAL_FRACTION).
        try:
            _rfrac = float(os.environ.get("OTR_TITLE_REVEAL_FRACTION", "0.4"))
        except (TypeError, ValueError):
            _rfrac = 0.4
        p = _title_reveal_progress(fi, w0, me, in_dock, _rfrac)

        # -- carrier-lock line "SIGNAL LOST" (decode scramble -> solid) ---
        carrier_src = "=== SIGNAL LOST ==="
        # solid chars proportional to p; '=' framing stays as-is
        reveal_n = int(p * len(carrier_src))
        carrier = self._decode_line(carrier_src, reveal_n, self._rng(fi, "carrier"))
        draw.text(self._ident_xy, carrier, fill=CRT_GREEN, font=self.f_title)
        self._draw_carrier_meter(draw, signal if not in_dock else 1.0)

        # -- THE HERO STOPS HERE (2026-08-12) ------------------------------
        # Everything above this line still draws: the decode-SCRAMBLED carrier
        # and the carrier meter exist ONLY on card frames and ARE the Matrix
        # look, which is kept. Only the hero glyphs below are suppressed.
        #
        # The cut is HERE and not at the top of this method on purpose. An early
        # return would take the carrier decode and the meter with it, and forcing
        # `card_active` False would ALSO add the section-1 subtitle and timestamp
        # from frame 0 -- both change the picture rather than move the title.
        # Below this point the method draws exactly two things, the hero
        # overstrike text and the reveal cursor, and those are what now live in
        # the ASS layer downstream of the procgen blend, where an outline is not
        # a no-op (Bible 07.32).
        if not _hero_title_in_procgen():
            return

        # -- HERO title (big + bold, centre-anchored, EXEMPT from gutter) -
        title = (self.title or "SIGNAL").upper()
        font, lines, hero_size = self._fit_hero(draw, title)

        # POP: a 1-2 frame brightness bloom right before lock (music_end).
        pop = (not in_dock) and (1 <= (me - fi) <= 2)
        if pop:
            hero_col = (200, 255, 200)
        else:
            hero_col = CRT_GREEN

        # decoded-fragment reveal across the wrapped lines (integer-frame).
        total_chars = sum(len(ln) for ln in lines)
        revealed = total_chars if in_dock else int(p * total_chars)
        shown, consumed = [], 0
        for ln in lines:
            take = max(0, min(len(ln), revealed - consumed))
            shown.append(self._decode_line(ln, take, self._rng(fi, "hero")))
            consumed += len(ln)

        # Block metrics at the (possibly docked) size.
        if in_dock and dock_frames > 0:
            d = min(1.0, max(0.0, (fi - me) / dock_frames))
            cur_size = int(round(hero_size + (self._title_size - hero_size) * d))
            font = _load_font(max(8, cur_size))
            lines = self._wrap_words(draw, title, font, int(self.w * 0.8))
            shown = lines  # fully decoded while docking
        else:
            d = 0.0

        line_h = max(self._th(draw, s or "M", font) for s in shown)
        block_h = line_h * len(shown) + (len(shown) - 1) * (line_h // 4)
        # centre anchor -> dock target (the ident top-left) as d: 0 -> 1
        cx_centre, cy_centre = self.w // 2, self.h // 2
        top_centre = cy_centre - block_h // 2
        ident_x, ident_y = self._ident_xy
        # left/top of the block lerp from centred to the ident anchor.
        for li, s in enumerate(shown):
            tw = self._tw(draw, s, font)
            x_centre = cx_centre - tw // 2
            y_centre = top_centre + li * (line_h + line_h // 4)
            x = int(round(x_centre + (ident_x - x_centre) * d))
            y = int(round(y_centre + (ident_y - y_centre) * d))
            # clamp inside the frame (no black edge from any drift/dock)
            x = max(0, min(x, self.w - tw))
            y = max(0, min(y, self.h - line_h))
            # Fake-bold by overstrike, plus a real black edge so the large
            # episode title reads over the animated CRT field.
            _draw_crt_overstrike_text(
                draw, (x, y), s, fill=hero_col, font=font,
                stroke_width=max(1, min(6, line_h // 18)),
                stroke_fill=(0, 0, 0),
            )
            # block cursor after the last revealed char (during reveal only)
            if not in_dock and li == len(shown) - 1 and p < 1.0:
                cur_w = max(4, line_h // 2)
                draw.rectangle([(x + tw + 2, y), (x + tw + 2 + cur_w, y + line_h)],
                               fill=hero_col)

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
        # ONE OWNER FOR THIS QUESTION (2026-08-30). The probe lives in
        # `_otr_shared.encode_sink.has_nvenc`; this used to be a second,
        # independent copy of the same test, and when the first was fixed the
        # duplicate failed the very next render at OTR_SceneAwareScopes
        # eighteen minutes in. Two implementations of one capability check is
        # how a fix looks applied and is not.
        try:
            from ._otr_shared.encode_sink import has_nvenc as _has_nvenc
        except ImportError:  # pragma: no cover -- flat-import test harnesses
            from _otr_shared.encode_sink import has_nvenc as _has_nvenc  # type: ignore
        _NVENC_AVAILABLE = bool(_has_nvenc(ffmpeg_path))
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
        # No -shortest (CW-4 frozen-audio spine): the procedural video
        # includes the Telemetry HUD credits post-roll, which intentionally
        # runs LONGER than the episode audio. -shortest would truncate the
        # end credits to the audio length -- never what we want. The exact
        # frame count is piped from total_frames, so the output is bounded
        # without it.
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
    "v2/en_speaker_7": "female * sharp * anxious",  # FIX-3 sync: en_speaker_7 reclassified female in story_orchestrator
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


def _write_story_treatment(out_path, episode_title, led,
                            voice_assignments, style, genre,
                            news_used,
                            duration, W, H, fps, size_mb):
    """Save a complete episode treatment alongside the MP4.

    v2 ledger consumer (2026-05-09): takes parsed ``led`` (v2 ledger
    dict). Single parse at top of render_video.

    Voice-path-cleanbreak Sprint 6.3 (2026-05-12): signature changed
    from `plan` (director-shaped intermediate dict) to explicit
    `voice_assignments` + `style` + `genre` parameters. Caller derives
    voice_assignments from led["cast"] via
    _otr_ledger_consumers.voice_assignments_from_cast.

    Behavior change vs legacy: scene_break / environment / pause
    item-types don't exist in the v2 ledger schema. The treatment
    output loses scene-arc summary, scene headers, and environment
    descriptions in the FULL SCRIPT section. Replaces with a flat
    list of dialogue in ledger order. Cast block is enriched
    from led.cast (which carries name + voice_preset per entry).
    """
    try:
        import time as _t
        import json as _json
        from . import _otr_ledger_consumers as _OTRLC

        led = led if isinstance(led, dict) else {}
        voice_assignments = voice_assignments if isinstance(voice_assignments, dict) else {}

        # Sprint 6.3: voice_assignments arrives pre-derived from cast
        # (Sprint 6.2 helper output). Flatten {"name": {"voice_preset": "..."}}
        # to {"name": "preset"} for treatment-text use.
        voices = {}
        for k, v in voice_assignments.items():
            if isinstance(v, dict):
                voices[str(k)] = str(v.get("voice_preset", v.get("preset", v.get("voice", str(v)))))
            else:
                voices[str(k)] = str(v)

        # Also build a cast-name -> preset lookup from led.cast directly --
        # equivalent to voice_assignments_from_cast but kept here as a
        # belt-and-braces fallback for ledgers where the helper produced
        # an empty dict (e.g. all rows are ANNOUNCER-skipped).
        led_cast_lookup: dict[str, str] = {}
        for entry in (led.get("cast") or []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            preset = str(entry.get("voice_preset") or "").strip()
            if name and preset:
                led_cast_lookup[name] = preset

        def _preset_for(char_name: str) -> str:
            return voices.get(char_name) or led_cast_lookup.get(char_name, "")

        # Sprint 6.3: explicit `style` + `genre` parameters; no plan.get
        # chain. Sprint C C3 (2026-05-15): the `or genre` fall-through is
        # deleted alongside the retirement of the `meta.visual_plan.genre`
        # stamp; treatment text now uses style first, generic descriptor
        # otherwise. The `genre` parameter on this function is now
        # dead-but-harmless (caller passes ""); Sprint G's orphan-parameter
        # sweep will drop it.
        style = style or "audio drama"
        ts     = _t.strftime("%Y-%m-%d  %H:%M:%S")
        BAR    = "\u2500" * 64
        DBAR   = "\u2550" * 64

        # --- Forensic data sources (all best-effort; missing -> "(not recorded)")
        meta = led.get("meta") or {}
        gp   = meta.get("gen_params_initial") or {}
        NR   = "(not recorded)"

        def _g(d, k, default=NR):
            v = (d or {}).get(k)
            return v if (v is not None and v != "") else default

        def _engine_map_lines(by_role):
            """Format {role: {engine: count}} into aligned 'role  eng xN' lines."""
            lines = []
            for role in sorted(by_role or {}):
                engs = by_role[role] or {}
                joined = ", ".join("%s x%d" % (e, n) for e, n in sorted(engs.items()))
                lines.append("    %-20s %s" % (role, joined))
            return lines

        out = []
        W_ = lambda s="": out.append(str(s))

        # Header
        W_(DBAR)
        W_("  SIGNAL LOST  \u00b7  EPISODE TREATMENT")
        W_(DBAR)
        W_()
        W_(f'  Title    :  "{episode_title}"')
        W_(f"  Style    :  {style}")
        W_(f"  Produced :  {ts}")
        W_()

        # WRITER / LLM CONFIG -- which model wrote this, at what settings.
        # Slugs are kept verbatim so "openrouter:~anthropic/claude-opus-latest",
        # "comfy:..." or a local repo id all read truthfully. slot_transitions
        # > 0 proves the run actually split work across the A/B slots.
        W_("WRITER / LLM CONFIG")
        W_(BAR)
        W_(f"  Creative slot (A)  :  {_g(gp, 'creative_writing_model', _g(meta, 'creative_writing_model'))}")
        W_(f"  Technical slot (B) :  {_g(gp, 'technical_model', _g(meta, 'technical_model'))}")
        _transitions = meta.get("slot_transitions")
        if _transitions is not None:
            W_(f"  Slot routing       :  {_transitions} A<->B transition(s)")
        _calls = meta.get("slot_calls_by_slot") or {}
        if _calls:
            W_("  Calls by slot      :  "
               + ", ".join("%s=%s" % (k, v) for k, v in sorted(_calls.items())))
        W_(f"  Creativity         :  {_g(gp, 'creativity')}")
        W_(f"  Temperature        :  {_g(gp, 'temperature')}    top_p: {_g(gp, 'top_p')}")
        W_(f"  Optimization       :  {_g(gp, 'optimization_profile')}")
        W_(f"  Seed source        :  {_g(gp, 'seed_source')}")
        # 2026-08-14: the "Target words" half is gone with the word
        # authority; it was rendering "(not recorded)" into every
        # _treatment.txt sidecar.
        W_(f"  Words              :  {_g(meta, 'total_word_count')} "
           f"(char {_g(meta, 'character_word_count')} / "
           f"announcer {_g(meta, 'announcer_word_count')})")
        W_(f"  Characters         :  {_g(gp, 'num_characters')}")
        # Resolved OpenRouter models -- the CONCRETE model each `~latest` alias
        # actually resolved to server-side this run, with call count + USD cost.
        # Historical accuracy: ~latest drifts, so record what truly served it.
        try:
            from ._otr_openrouter_backend import resolved_models_snapshot as _rms
            _resolved = _rms()
        except Exception:  # noqa: BLE001 -- backend optional / not loaded
            _resolved = {}
        if _resolved:
            W_("  Resolved (OpenRouter):")
            _total_cost = 0.0
            for _slug in sorted(_resolved):
                _info = _resolved[_slug] or {}
                _rm = _info.get("resolved") or "(unreported)"
                _calls = _info.get("calls") or 0
                _cost = float(_info.get("cost_usd") or 0.0)
                _total_cost += _cost
                W_("    %s  ->  %s  (%d call(s), $%.4f)"
                   % (_slug, _rm, _calls, _cost))
            W_("    Total OpenRouter cost: $%.4f" % _total_cost)
        W_()

        # News seed - may arrive as a JSON list ["headline 1", ...]; the
        # first entry may carry a bank-aware origin_label (kibitz r2-r4;
        # original_radio stamps "STORY ORIGIN"). Legacy default holds.
        _news_raw = (news_used or "").strip()
        _origin_label = "NEWS SEED"
        # PARSED ONCE. This block used to call `_json.loads(_news_raw)` twice on
        # the identical string -- once for the label, once for the headlines --
        # so a malformed payload was decoded, discarded and re-decoded, and BOTH
        # failures were swallowed by a bare `except: pass`. One parse, one
        # log-continue: the card still renders on the legacy defaults, but the
        # reason it fell back is now on the record instead of being invisible.
        _seeds = None
        _parse_failed = False
        if _news_raw.startswith("["):
            try:
                _seeds = _json.loads(_news_raw)
            except Exception as news_err:  # noqa: BLE001 -- optional metadata
                _parse_failed = True
                log.info(
                    "[OTR.credits] news payload looked like JSON but did not "
                    "parse (%s: %s); using the raw text and the legacy %r "
                    "label. The episode is unaffected.",
                    type(news_err).__name__, news_err, _origin_label)
            else:
                if _seeds and isinstance(_seeds[0], dict):
                    _origin_label = str(
                        _seeds[0].get("origin_label") or "NEWS SEED")
        W_(_origin_label)
        W_(BAR)
        if _parse_failed:
            news_clean = _news_raw[:120]
        elif _seeds is not None:
            news_clean = " | ".join(
                str(s.get("headline", s) if isinstance(s, dict) else s)[:80]
                for s in _seeds[:3]
            ) if _seeds else ""
        else:
            news_clean = _news_raw.split("\n")[0][:120]
        W_(f"  {news_clean if news_clean else '(no news seed - custom premise used)'}")
        W_()

        # STORY SPINE -- the actual premise + the opposed wants the writer
        # derived from the news (B1 dramatic-state). This is what makes the
        # episode legible years later: what the story was ABOUT.
        W_("STORY SPINE")
        W_(BAR)
        # Meta split (2026-07-09): Premise = the PRODUCED story's logline
        # (what the composed episode is actually about); the interpreter's
        # pre-generation digest prints as "Source brief" when the logline
        # exists. Old ledgers keep the legacy labeling.
        _produced = (meta.get("produced_story")
                     if isinstance(meta.get("produced_story"), dict) else {})
        _newsd = meta.get("news") if isinstance(meta.get("news"), dict) else {}
        if _produced.get("logline"):
            W_(f"  Premise    :  {_produced['logline']}")
            if _produced.get("subject"):
                W_(f"  Subject    :  {_produced['subject']}")
        if _newsd:
            if _produced.get("logline"):
                W_(f"  Source brief: {_g(_newsd, 'script_brief')}")
            else:
                W_(f"  Premise    :  {_g(_newsd, 'script_brief')}")
            _kt = _newsd.get("key_terms") or []
            if _kt:
                W_("  Key terms  :  " + ", ".join(str(t) for t in _kt))
            W_(f"  Casting    :  {_g(_newsd, 'casting_brief')}")
            W_(f"  Sign-off   :  {_g(_newsd, 'news_close_brief')}")
        elif not _produced.get("logline"):
            W_("  (custom premise -- no structured news brief recorded)")
        _ds = meta.get("dramatic_state") if isinstance(meta.get("dramatic_state"), dict) else {}
        if _ds:
            W_(f"  Question   :  {_g(_ds, 'dramatic_question')}")
            W_(f"  A wants    :  {_g(_ds, 'character_a_wants')}")
            W_(f"  B wants    :  {_g(_ds, 'character_b_wants')}")
            W_(f"  Ending     :  {_g(_ds, 'ending_change')}")
            W_(f"  Costly beat:  {_g(_ds, 'costly_choice_beat')}")
        W_()

        # Cast & voices: prefer led.cast (v2 ledger has name + voice_preset
        # per entry); fall back to production_plan.voice_assignments.
        W_("CAST & VOICES")
        W_(BAR)
        cast_rows = []
        for entry in (led.get("cast") or []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("char_id") or "?")
            preset = str(
                entry.get("voice_preset") or voices.get(name, "")
            )
            engine = str(entry.get("voice_engine") or "")
            cast_rows.append((name, engine, preset))
        if not cast_rows and voices:
            cast_rows = [(c, "", voices[c]) for c in sorted(voices.keys())]
        if cast_rows:
            pad_w = max((len(name) for name, _, _ in cast_rows), default=10)
            for name, engine, preset in cast_rows:
                desc   = _PRESET_DESC.get(preset, preset)
                etag   = f"[{engine}] " if engine else ""
                W_(f"  {name:<{pad_w}}  \u2192  {etag}{preset:<24}  {desc}")
        else:
            W_("  (no voice assignments recorded)")
        W_()

        # RENDER ENGINES -- which video engine rendered each role (stamped by
        # the render batch into meta.render_engines) and which image engine
        # produced the stills (from ledger['images'], each still carries its
        # role + engine_id). Lets a viewer see e.g. character beats = z_image
        # stills while the bookends = ltx_av_music audio-in video.
        W_("RENDER ENGINES")
        W_(BAR)
        _re = meta.get("render_engines") or {}
        _vid_lines = _engine_map_lines(_re.get("by_role") or {})
        if _vid_lines:
            W_("  Video (per role):")
            for _l in _vid_lines:
                W_(_l)
        else:
            W_("  Video (per role):  " + NR)
        _img_rows = ((led.get("images") or {}).get("images")) or []
        _img_by_role: dict = {}
        for _r in _img_rows:
            if not isinstance(_r, dict):
                continue
            _role = str(_r.get("role") or "?")
            _eng = str(_r.get("engine_id") or "?")
            _img_by_role.setdefault(_role, {})
            _img_by_role[_role][_eng] = _img_by_role[_role].get(_eng, 0) + 1
        _img_lines = _engine_map_lines(_img_by_role)
        if _img_lines:
            W_("  Image (per role):")
            for _l in _img_lines:
                W_(_l)
        else:
            W_("  Image (per role):  " + NR)
        _hist = _re.get("histogram") or {}
        if _hist:
            W_("  Engine histogram:  "
               + ", ".join("%s=%s" % (k, v) for k, v in sorted(_hist.items())))
        if _re.get("video_revision") is not None:
            W_(f"  Video revision  :  {_re.get('video_revision')}")
        W_()

        # Scene arc / FULL SCRIPT under v2 ledger schema:
        # the ledger has no scene_break / environment / pause markers,
        # so we drop the scene-arc summary entirely and emit a flat
        # list of dialogue in ledger order. This is a deliberate
        # simplification (architect-approved 2026-05-09 Surprise #3).
        W_("SCENE ARC")
        W_(BAR)
        W_("  (single-scene ledger -- v2 schema does not carry scene_break "
           "/ environment / pause markers; arc summary collapsed to a flat "
           "dialogue list below)")
        W_()

        d_count = sum(
            1 for ln in _OTRLC.iter_lines(led, roles={"character", "announcer"})
        )
        W_(f"FULL SCRIPT  ({d_count} dialogue)")
        W_(BAR)
        for line in _OTRLC.iter_lines(led):
            role = line.get("speaker_role") or ""
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if role in ("character", "announcer"):
                char = _OTRLC.speaker_name(led, line)
                preset = _preset_for(char)
                vtag   = f"  [{preset}]" if preset else ""
                W_(f"  {char}{vtag}")
                W_(f"    {text}")
                W_()
            # music_* roles intentionally skipped from the printed
            # treatment (consistent with HUD's music skip behavior).
        W_()

        # Production
        W_("PRODUCTION")
        W_(BAR)
        W_(f"  Duration    :  {duration/60:.1f} min  ({duration:.1f} s)")
        W_(f"  Resolution  :  {W}x{H} @ {fps} fps")
        W_(f"  File        :  {os.path.basename(out_path)}")
        W_(f"  Size        :  {size_mb:.1f} MB")
        _vram_peak = (meta.get("render_engines") or {}).get("vram_peak_mb")
        if _vram_peak:
            W_(f"  VRAM peak   :  {_vram_peak} MB")
        W_()

        # SYSTEM -- the machine + software stack this episode was rendered on,
        # so anyone can look back and see exactly what produced it. Best-effort:
        # any unprobeable field degrades to "(unknown)" rather than failing.
        W_("SYSTEM")
        W_(BAR)
        try:
            from ._otr_sys_specs import collect_system_specs as _css
            _sys = _css()
        except Exception as _sys_exc:  # noqa: BLE001
            _sys = {}
            log.warning("[Video] system-spec collection skipped: %s", _sys_exc)
        _sg = lambda k: _sys.get(k) or "(unknown)"
        W_(f"  Host     :  {_sg('hostname')}")
        W_(f"  OS       :  {_sg('os')}")
        W_(f"  CPU      :  {_sg('cpu')}  ({_sg('cpu_cores')})")
        W_(f"  RAM      :  {_sg('ram')}  (peak {_sg('ram_peak')})")
        W_(f"  GPU      :  {_sg('gpu')}  ({_sg('vram')} VRAM)")
        W_(f"  CUDA     :  {_sg('cuda')}    torch {_sg('torch')}")
        W_(f"  Python   :  {_sg('python')}")
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
    # title_card_plan_json APPENDED, never inserted: output order is positional
    # in the saved graph exactly as widget order is, so a new socket goes last.
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "title_card_plan_json")
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
                # 2026-05-03 EVENING (BUG-LOCAL-030 Phase B, Jeffrey
                # final spec): default RAISED from 832x480 to 1920x1080
                # so procgen renders directly at delivery resolution.
                # Per Jeffrey: "proc gen 1920x1080 ... then a final
                # ffmpeg w/ the proc gen mix for final 1080p". Procgen
                # is now SEPARATE from the per-clip-mux composite path,
                # which reaches 1920x1080 itself via render.composite_w/h.
                # The OTR_PostUpscaleProcgenBlend node overlays this
                # 1920x1080 procgen on the composited mp4 at delivery
                # res. Result: procgen's CRT scanline / audio-reactive
                # flicker stays CRISP at 1080p (it would be smeared if
                # run through a super-resolution model -- synthetic
                # patterns + AI upscaler = ringing / softening, which is
                # why procgen renders at delivery res instead), and
                # fills the visible HuMo black pillarbox bars from
                # BUG-030 Phase A as the SIGNAL LOST visual signature.
                # 832x480 retained as legacy mode for the prior path
                # (composite everything small, then upscale it together).
                "resolution": (["1920x1080", "1280x720", "832x480", "854x480", "3840x2160"], {
                    "default": "1920x1080",
                    "tooltip": "Procgen output resolution. 1920x1080 = delivery res for the final procgen blend (BUG-030 Phase B default). 832x480 was the prior default (rendered cheap, then upscaled with everything else; legacy mode). 1280x720 / 854x480 / 3840x2160 retained for one-off needs."
                }),
                "episode_title": ("STRING", {
                    "default": "",
                    "tooltip": "Episode title for the title bar. Normally resolved from the script_json title token; widget acts as a last-resort override."
                }),
                # APPENDED LAST (widgets_values is positional -- only ever
                # append). v2 scene-aware path sets this False so the floor's
                # in-frame scopes {2,3,5,6} (centre ring / particles /
                # waveform / bars) turn OFF and OTR_SceneAwareScopes draws the
                # scene-aware scopes downstream instead. Default True keeps the
                # v1 floor visual for every existing saved workflow.
                "draw_scopes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw the floor's in-frame scopes (centre ring, orbit particles, waveform, freq bars). Default ON (v1). Set OFF for the v2 scene-aware-scopes pipeline, where OTR_SceneAwareScopes draws the scopes in the real per-beat gutters downstream."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _time.time()

    def render_video(self, audio, script_json, news_used,
                     fps=24, resolution="1920x1080",
                     episode_title="", draw_scopes=True):

        from .story_orchestrator import _runtime_log

        # -- 1. Parse inputs & Extract Smart Title -------------------
        # v2 ledger consumer (Pattern 1): parse the wire input as a v2
        # ledger dict. load_ledger raises ValueError on the legacy
        # parser-list shape; Video is in the loud-fail group. By the
        # time Video runs, every prior consumer (Bark/Kokoro/Sequencer)
        # has already validated ledger shape, so a
        # legacy-list at this point is upstream-wiring failure, not a
        # silent-degrade case.
        #
        # Voice-path-cleanbreak Sprint 2 + Sprint 6 (2026-05-12).
        # Sprint 2 deleted the production_plan_json socket and built
        # a legacy-shape `plan` dict from meta so downstream plan.get()
        # sites stayed source-stable. Sprint 6.3 deconstructs that
        # intermediate dict: voice_assignments now derives from
        # led["cast"] at render time (Sprint 6.2 helper); style + genre
        # are unpacked into local variables. Helpers (
        # _write_story_treatment) take voice_assignments/style/genre
        # as separate parameters instead of a director-shaped plan dict.
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)
        # FIX3 / BUG-LOCAL-404: the script_json from the freeze cascade is
        # PRE-audio -- its lines carry no start_s, so _resolve_title_timing
        # (below) saw first_dialogue_f=None and collapsed the hero title-card
        # window to the ~1s volume-envelope fallback. Overlay the real per-line
        # audio timing from the newest on-disk ledger (the same disk contract
        # ShotLock/SceneSequencer use; gated on audio_done by the time Video
        # runs, fail-soft, skipped under OTR_TEST_MODE) so the card spans
        # [0, first_dialogue). Import is lazy + side-effect-free.
        from . import otr_shot_lock as _OTRSL
        led = _OTRSL.overlay_audio_timing(led)
        _meta = led.get("meta") or {}
        voice_assignments = _OTRLC.voice_assignments_from_cast(led)
        # HUD + treatment show the LOOK, never the pre-story catalog draw
        # (operator ruling 2026-08-03). meta.style is the scaffold and lied on
        # screen twice in one day; meta.visual_style is the channel that
        # actually governs the picture and is stamped on every run. The
        # scaffold remains in the ledger as provenance only.
        style = _meta.get("visual_style") or ""
        genre = (_meta.get("visual_plan") or {}).get("genre") or ""

        # Title chain (Path B; slot 1 live since the J.5 title pass):
        #   1. led["meta"]["episode_title"]   (primary -- stamped on
        #      every run by OTR_LedgerScriptWriter's J.5 post-composition
        #      title pass, which LLM-regenerates the title from the
        #      finished script, or uses the typed widget value, or
        #      falls back to outline.title)
        #   2. led["meta"]["title"]           (forward-compat slot)
        #   3. led["title"]                   (top-level; not stamped
        #      by OTR_LedgerScriptWriter -- pre-LPL legacy slot kept
        #      in the chain for older ledgers loaded from disk)
        #   4. widget episode_title           (manual override)
        #   5. TIMESTAMP_LASTRESORT
        #
        # news_used[0].headline and meta.news_seed.headline are
        # intentionally NOT in this chain -- both surface news/outline
        # content as titles, neither is what we want on screen. With
        # the J.5 title pass live, slot 1 is populated on every writer
        # run; the timestamp last-resort fires only on a degenerate run
        # where the writer produced no title at all.
        def _is_clean(s):
            return bool(s)

        _meta = led.get("meta") or {}
        _meta_episode_title = (_meta.get("episode_title") or "").strip()
        _meta_title         = (_meta.get("title") or "").strip()
        _led_title          = (led.get("title") or "").strip()
        _widget_title       = (episode_title or "").strip()

        if _is_clean(_meta_episode_title):
            episode_title = _meta_episode_title
            _title_source = "led.meta.episode_title"
        elif _is_clean(_meta_title):
            episode_title = _meta_title
            _title_source = "led.meta.title"
        elif _is_clean(_led_title):
            episode_title = _led_title
            _title_source = "led.title (legacy stamp)"
        elif _is_clean(_widget_title):
            episode_title = _widget_title
            _title_source = "widget_override"
        else:
            # Final fallback: timestamp-based unique title. Guarantees
            # we never crash the run when every title source is empty.
            episode_title = f"Signal Lost {_time.strftime('%Y%m%d %H%M%S')}"
            _title_source = "timestamp_lastresort"
            log.warning(
                "[Video] TITLE LAST-RESORT: meta.episode_title='%s', "
                "meta.title='%s', led.title='%s', widget='%s' -- ALL "
                "EMPTY. Using timestamp fallback %r so the run "
                "can finish. This should not happen on a normal run: "
                "OTR_LedgerScriptWriter's J.5 title pass stamps "
                "led.meta.episode_title every time. Reaching here means "
                "the writer produced no title at all.",
                _meta_episode_title, _meta_title, _led_title,
                _widget_title, episode_title,
            )

        _runtime_log(
            f"TITLE_TRACE | source={_title_source} | resolved='{episode_title}' "
            f"| widget='{_widget_title}' | meta.episode_title='{_meta_episode_title}' "
            f"| meta.title='{_meta_title}' | led.title='{_led_title}'"
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

        # THE POST-ROLL TELEMETRY HUD WAS REMOVED HERE (2026-09-01).
        # This engine used to append its own credits card after the episode
        # audio -- METADATA / NEWS SEED / SYSTEM TELEMETRY / a scrolling
        # transcript. OTR_CreditsRoll (node 95 in the canonical workflow)
        # superseded it in a0224438 and this was never taken out, so every
        # episode carried TWO credits sequences back to back. It went
        # unnoticed for months because the lane that shows it most plainly is
        # rarely run.
        #
        # What this engine still owns is untouched: the procedural base video,
        # the title-card plan, and the procgen mp4 that PostUpscaleProcgenBlend
        # blends -- the `_silent_procgen_blended_` in every output filename.

        # -- 3. Save audio to temp WAV for ffmpeg ---------------------
        import tempfile
        import wave as wave_mod

        tmp_wav = os.path.join(tempfile.gettempdir(), "otr_video_audio.wav")
        pcm = (audio_np * 32767).astype(np.int16)
        # Credits-music loop RIPPED (credits enrichment 2026-07-03, silent-tail
        # model). This base mp4's OWN audio track is stripped downstream by
        # OTR_SilentComposite (V-1); the delivered episode audio is the frozen
        # master muxed on LAST, and the credits roll is now a SILENT tail
        # rendered late by OTR_CreditsRoll (which drops the closing-cue-under-
        # credits music by the operator's silent-tail decision). The HUD
        # post-roll frames still render (the reduced dossier / transcript
        # easter egg); its audio is simply silence so this base mp4's own audio
        # length matches its video length.
        #
        # THE `closing_audio` INPUT IS GONE (operator ruling 2026-08-19). It was
        # an "AUDIO" socket this node accepted and then discarded -- so wiring
        # it up, expecting a closing sting under the credits, would have got
        # silence and no explanation. Removing it also required renumbering two
        # links in `workflows/otr_canonical.json`: it sat at input slot 1, and
        # `script_json` / `news_used` followed it, so a naive deletion would
        # have shifted both onto the wrong sockets.
        #
        # THE ENDING AUDIO IS UNAFFECTED and that was checked before removal:
        # the credits tail is SILENT by the operator's own silent-tail ruling,
        # and the real closing music is `closing_theme_audio` on the sequencer
        # (a different input on a different node), while the delivered episode
        # audio is the frozen master muxed on last.
        pcm_out = pcm
        with wave_mod.open(tmp_wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_out.tobytes())

        # -- 4. Determine output path ---------------------------------
        # BUG-LOCAL-020 (Phase G, 2026-05-03): write the procgen mp4
        # INTO the per-episode pending workspace so it travels with
        # `Ledger.rename_episode` when the title is finalized below.
        # Prior to this fix, the mp4 landed at the legacy flat
        # `output/otr/audio/` which is OUTSIDE the per-episode tree --
        # rename couldn't move it, BatchHumoRender's stem-swap looked
        # for the ledger in the legacy dir, didn't find it (ledger had
        # already been moved into the per-episode workspace by Phase B),
        # and crashed with `derived ledger from .mp4 not found`.
        #
        # Strategy: read out_dir from the in-flight Ledger singleton
        # (which is `episodes/pending_<ts>/audio/` at this point).
        # After `Ledger.rename_episode(ep_id)` below moves that parent
        # dir to `episodes/<ep_id>/`, the mp4 moves with it. We then
        # recompute the final on-disk path before returning so the
        # downstream graph receives the post-rename location.
        try:
            from .production_ledger import get_ledger as _get_ledger
            _early_led = _get_ledger()
            out_dir = str(_early_led.out_dir)
        except Exception as _exc:  # noqa: BLE001
            # Fall back to legacy if the ledger is somehow unavailable
            # (test harness, headless invocation). This preserves the
            # pre-Phase-G behavior in those edge cases.
            log.warning(
                "[Video] ledger singleton unavailable for out_dir (%s); "
                "falling back to legacy output/otr/audio/", _exc,
            )
            out_dir = os.path.join(
                os.path.expanduser("~"), "Documents", "ComfyUI",
                "output", "otr", "audio"
            )
        os.makedirs(out_dir, exist_ok=True)

        ts = _time.strftime("%Y%m%d_%H%M%S")

        # BUG-LOCAL-110 Layer 3 -- SUPERSEDED 2026-05-09 by the v2 ledger
        # title chain at the top of render_video. The chain probes
        # led.meta.episode_title -> led.meta.title -> led.title -> widget
        # -> TIMESTAMP_LASTRESORT against the wire-input ledger, so the
        # singleton-side preference is no longer needed. Slot 3 of the
        # chain (`led.title`) covers the legacy writer's stamp.

        # BUG-LOCAL-110 Layer 1 (2026-05-05, round-robin verified): slug
        # cleanup. Pipeline:
        #   1. drop punctuation (alnum + "_" + " " kept)
        #   2. collapse runs of whitespace to a single space FIRST -- this
        #      prevents titles like "Signal Lost - The Crystal!" from
        #      becoming "signal_lost__the_crystal" (the strip-of-punct
        #      leaves a double space between "Lost" and "The" which then
        #      becomes a double underscore on `replace(" ", "_")`).
        #      Round-robin (Gemini + NVIDIA) caught this; ChatGPT missed it.
        #   3. underscore-ize, lowercase, truncate to 40
        #   4. collapse any remaining underscore runs and strip both ends
        #      again (truncation can leave a trailing "_" if char 40 was
        #      a separator)
        #   5. fall back to "untitled" if the result is empty so we never
        #      emit `signal_lost__<ts>.mp4` (double underscore).
        safe_title = "".join(
            c if c.isalnum() or c in "_ " else ""
            for c in str(episode_title or "")
        )
        safe_title = _re.sub(r"\s+", " ", safe_title).strip()
        safe_title = safe_title.replace(" ", "_").lower()[:40]
        safe_title = _re.sub(r"_+", "_", safe_title).strip("_") or "untitled"
        out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")

        # -- 5. Build frame generator ---------------------------------
        # Resolve the b000 music-open title-card window from the ledger lines
        # (start_s/dur_s/speaker_role), with a volume-envelope fallback.
        _title_timing = _resolve_title_timing(led, volume, fps, total_frames)
        renderer = _CRTRenderer(W, H, episode_title, volume, freqs, waves, fps,
                                timing=_title_timing)
        total_encode_frames = total_frames

        # -- Hero title-card plan for the DOWNSTREAM burn ------------------
        # The title is drawn here into a layer that is composited `screen` +
        # `green_only` -- lighten-only, where a black outline is a no-op and the
        # hero measured 1.13:1 over a lit monitor (Bible 07.32). So the legible
        # copy is emitted as ASS by OTR_CaptionBurn, which runs AFTER that
        # blend. This plan is what it draws from.
        #
        # total_frames, NOT total_encode_frames: the encode appends the HUD
        # post-roll, and a closing card clamped to the encoded length would hang
        # the title over the post-roll and into the credits.
        _title_plan_json = ""
        try:
            _title_plan = renderer.title_card_plan(main_frames=total_frames)
            if _title_plan is not None:
                _title_plan_json = json.dumps(_title_plan, separators=(",", ":"))
                _runtime_log(
                    f"Video: title-card plan -- {len(_title_plan['frames'])} frames "
                    f"across {len(_title_plan['cards'])} card(s)")
        except Exception as _e:  # noqa: BLE001
            # An episode with no resolvable card is a legitimate state and sends
            # an EMPTY plan. A plan that FAILED to build is not: it would reach
            # the burn as "no title wanted" and publish a title-less master, so
            # it stops the render here where the cause is still on screen.
            raise RuntimeError(
                f"OTR_SignalLostVideo: hero title-card planning failed "
                f"({type(_e).__name__}: {_e}). The downstream burn cannot tell "
                f"a failed plan from an episode with no card, so this refuses "
                f"rather than shipping an untitled episode."
            ) from _e

        def _render_crt(fi):
            return renderer.render(fi, draw_scopes=draw_scopes)

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


        # -- 6. Encode ------------------------------------------------
        _runtime_log(f"Video: Encoding MP4 via ffmpeg -> {os.path.basename(out_path)}")
        _encode_mp4(_frame_gen(), total_encode_frames, tmp_wav, out_path, W, H, fps)

        try:
            os.remove(tmp_wav)
        except OSError:
            pass

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log.info("[Video] Saved: %s (%.1f MB, %.1fs, %d frames total)",
                 out_path, size_mb, duration, total_encode_frames)
        _runtime_log(f"Video: DONE -- {os.path.basename(out_path)} ({size_mb:.1f} MB)")

        # Forensic treatment engine-enrich RIPPED (credits enrichment
        # 2026-07-03). This merge pulled render_engines / images from the
        # singleton into the wire `led` for the credits sheet -- but node 12
        # runs BEFORE the image dispatcher (91) and the video render batch (92),
        # so at this point the singleton has NO post-render receipts yet: the
        # merge only ever grafted stale/empty maps. The engine receipts are now
        # rendered LATE from the DURABLE ledger in OTR_CreditsRoll. The treatment
        # sidecar below keeps only the story facts that are true at node-12 time.

        # Write story treatment companion file. v2 ledger consumer:
        # pass parsed led + voice_assignments/style/genre (Sprint 6.3
        # rename of the legacy `plan` parameter).
        _write_story_treatment(
            out_path, episode_title, led,
            voice_assignments, style, genre,
            news_used,
            duration, W, H, fps, size_mb
        )

        # ------------------------------------------------------------------
        # Production Ledger (L1) -- finalize the episode. Rename the ledger
        # from its placeholder "pending_<ts>" to match the MP4 basename,
        # attach final audio + video paths, record total episode duration,
        # and stamp meta.procgen_path (Pattern 4 write-back; v2 ledger
        # consumer 2026-05-09).
        # ------------------------------------------------------------------
        # Note: `_final_led` here is the production_ledger singleton; the
        # outer `led` variable still holds the wire-input ledger dict
        # parsed at the top of render_video. Both refer to the same
        # on-disk file once rename_episode + save complete.
        try:
            from .production_ledger import get_ledger
            from . import _otr_ledger as _OTRL  # type: ignore
            _final_led = get_ledger()
            # Strip the ".mp4" extension and use the stem as the ledger id.
            ep_id = os.path.splitext(os.path.basename(out_path))[0]
            _final_led.rename_episode(ep_id)
            # BUG-LOCAL-020 (Phase G): the rename moved our parent dir
            # from `episodes/pending_<ts>/audio/` to `episodes/<ep_id>/audio/`.
            # The mp4 we wrote at `out_path` no longer exists at that
            # path string -- it's at `<new_audio_dir>/<basename>`.
            # Recompute and verify so downstream graph receives the
            # post-rename location, not the stale pending path.
            try:
                _final_out_path = os.path.join(_final_led.out_dir, os.path.basename(out_path))
                if os.path.exists(_final_out_path):
                    if _final_out_path != out_path:
                        log.info(
                            "[Video] post-rename mp4 path: %s -> %s",
                            out_path, _final_out_path,
                        )
                    out_path = _final_out_path
                else:
                    # rename_episode failed silently or fell back to
                    # the legacy out_dir; original out_path may still
                    # be valid. Log and continue with original.
                    log.warning(
                        "[Video] post-rename recompute: expected %s does "
                        "not exist; keeping pre-rename path %s",
                        _final_out_path, out_path,
                    )
            except Exception as _exc_recompute:  # noqa: BLE001
                log.warning(
                    "[Video] post-rename path recompute failed: %s",
                    _exc_recompute,
                )
            _final_led.set_final_paths(
                audio_path=None,           # standalone WAV not saved yet (L2)
                video_path=out_path,
                total_episode_dur_s=float(duration),
            )
            # Pattern 4 stamp (architect spec): meta.procgen_path =
            # <output mp4 path>. Mirrors the per-line write-back contract
            # from prior consumers but at meta level (no per-line stamping
            # for Video). Mutate the singleton's .data dict before save()
            # so the persisted file carries the stamp.
            try:
                _OTRL.set_meta(_final_led.data, "procgen_path", out_path)
            except Exception as _stamp_exc:  # noqa: BLE001
                log.warning(
                    "[Video] meta.procgen_path stamp failed (non-fatal): %s",
                    _stamp_exc,
                )
            _final_led.save()
        except Exception as _e:  # noqa: BLE001
            log.warning("[Ledger] SignalLostVideo finalize failed: %s", _e)

        # Build the ComfyUI 'gifs' subfolder hint from the actual
        # post-rename location. ComfyUI expects subfolder relative to
        # `output/`, so strip the leading `output/` prefix from the
        # absolute path's parent. Falls back to `otr/audio` if the
        # path doesn't have an `output/` segment (legacy / test).
        try:
            _abs_parent = os.path.abspath(os.path.dirname(out_path))
            _abs_parent_norm = _abs_parent.replace("\\", "/")
            if "/output/" in _abs_parent_norm:
                _subfolder = _abs_parent_norm.split("/output/", 1)[1]
            else:
                _subfolder = "otr/audio"
        except Exception:  # noqa: BLE001
            _subfolder = "otr/audio"

        # Return using the standard ComfyUI 'gifs' key so the canvas
        # spawns an HTML5 video player widget automatically.
        return {
            "ui": {"gifs": [{"filename": os.path.basename(out_path),
                             "subfolder": _subfolder,
                             "type": "output"}]},
            "result": (out_path, _title_plan_json),
        }
