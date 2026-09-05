"""OTR_SceneAwareScopes -- v2 scene-aware audio-reactive scopes (§4D).

The §4C-v1 in-floor scopes are SUPERSEDED by this late, additive node (the
operator's LOCKED new-node-only decision 2026-06-13). The procgen FLOOR
(`OTR_SignalLostVideo`) is NOT relocated: it keeps emitting its full v1 video
(CRT + title card + gap/credits fill) and is the composite's floor/gap/credits
+ green-blend base; its in-frame scopes are simply turned OFF via the new
`draw_scopes=False` flag. ALL the scene-aware scope visuals live HERE.

Why a late node (grounded): the floor `render_video` receives no clip manifest
and no per-beat engine plan, so it cannot be scene-aware. The per-beat
portrait(HuMo 480x832)-vs-landscape(LTX/Wan) aspect first exists at the clip
manifest (produced by `OTRVideoRenderBatch`, whose RETURN_NAMES include
`clip_manifest_json`). This node reads that manifest + (optionally) the master
audio analysis and draws the SAME two-asymmetric-circular-scope design into the
REAL per-beat gutters (edge/suppressed on landscape, centre on the signal-lost
gaps, suppressed on the credits tail), GREEN-ONLY, on BLACK -- so there is no
master decode (no generation loss). The result, `scopes_only.mp4`, is screened +
lightened over the upscaled video by `OTR_PostUpscaleProcgenBlend`'s 3rd input.

Invariants honored: 100% local/offline; GREEN-ONLY (CRT_GREEN/CRT_DIM/CRT_DARK
only -- the colored CRT constants are deliberately NOT imported); deterministic
(stable-hash seeded RNG -- also fixes the floor's old unseeded np.random);
silent -an encode (the floor's `_encode_mp4` HARD-REQUIRES audio, so this node
ships its own no-audio encoder); UTF-8 no BOM; SFW.
"""
from __future__ import annotations

import hashlib
import logging
import re as _re
import math
import os
import sys
import tempfile
import time as _time

import numpy as np
from PIL import Image, ImageDraw

_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)

try:  # ComfyUI loads these node modules flat as well as packaged
    from ._otr_shared import ffprobe as _ffp
except ImportError:  # pragma: no cover -- flat (sys.path) test import
    from _otr_shared import ffprobe as _ffp  # type: ignore

log = logging.getLogger("OTR.SceneAwareScopes")

# -- GREEN-ONLY palette (forbid CRT_CYAN/CRT_AMBER by simply not defining them) -
CRT_GREEN = (0, 255, 65)
CRT_DIM = (0, 100, 28)
CRT_DARK = (0, 50, 14)
CRT_BLACK = (0, 0, 0)

_FFT_SPOKES = 32
_TRAIL_N = 6  # bounded comet-tail / sweep lookback


# --------------------------------------------------------------------------- #
# Deterministic RNG (stable-hash; fixes the floor's old unseeded noise too)
# --------------------------------------------------------------------------- #
def _rng(key, fi, salt):
    seed = int.from_bytes(
        hashlib.blake2s(f"{key}|{int(fi)}|{salt}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Pure-numpy audio analysis (mirrors video_engine._analyze_audio EXACTLY so the
# scopes are frame-identical at 25fps; kept torch-free for testability).
# --------------------------------------------------------------------------- #
def _analyze_audio_np(audio_np, sample_rate, total_frames, fps):
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


def _dual_ema(volume):
    """signal (slow, ambient) + trig (fast, lock) + loss = 1 - signal."""
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
# GREEN-ONLY circular scope helpers (S-v2a). Geometry by params (no self), so
# the floor (v1) and this node (v2) can both call them. amp clamped to r*0.35.
# --------------------------------------------------------------------------- #
def _green(scale):
    s = max(0.0, min(1.0, scale))
    return tuple(min(255, int(c * s)) for c in CRT_GREEN)


def draw_fft_scope(draw, cx, cy, r, freq_window, env):
    """LEFT scope: 32 radial FFT spokes + per-spoke phosphor comet-tails
    (bounded lookback). Idle (low signal) -> a slow rotating radar sweep.
    GREEN-ONLY. ``freq_window`` is the bounded lookback list (oldest..newest)."""
    fi = int(env.get("fi", 0))
    fps = float(env.get("fps", 25) or 25)
    signal = float(env.get("signal", 0.0))
    amp_cap = r * 0.35
    cur = freq_window[-1] if freq_window else np.zeros(_FFT_SPOKES)
    n = min(_FFT_SPOKES, len(cur))
    # graticule crosshair (static, dim)
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=CRT_DARK, width=1)
    for i in range(n):
        ang = 2 * math.pi * i / n - math.pi / 2
        ca, sa = math.cos(ang), math.sin(ang)
        # comet-tail: faint older frames first, bright newest last
        for k, past in enumerate(freq_window):
            val = float(past[i]) if i < len(past) else 0.0
            blen = min(amp_cap, val * amp_cap * 2.2)
            x0 = cx + int(r * ca)
            y0 = cy + int(r * sa)
            x1 = cx + int((r + blen) * ca)
            y1 = cy + int((r + blen) * sa)
            tail = (k + 1) / max(1, len(freq_window))
            draw.line([(x0, y0), (x1, y1)], fill=_green(0.25 + 0.75 * tail * (0.3 + 0.7 * val)), width=1)
    if signal < 0.18:
        # idle radar sweep: a single bright spoke rotating deterministically.
        ang = (2 * math.pi * (fi / fps) * 0.5) % (2 * math.pi)
        x1 = cx + int(r * math.cos(ang))
        y1 = cy + int(r * math.sin(ang))
        draw.line([(cx, cy), (x1, y1)], fill=_green(0.5), width=1)


def draw_scope(draw, cx, cy, r, wave_window, env):
    """RIGHT scope: the waveform samples traced AROUND the circumference + a
    bright electron sweep dot with a short decaying trail. Idle -> a jittering
    baseline circle. GREEN-ONLY. ``wave_window`` is the bounded lookback."""
    fi = int(env.get("fi", 0))
    signal = float(env.get("signal", 0.0))
    amp_cap = r * 0.35
    wave = wave_window[-1] if wave_window else np.zeros(0)
    m = len(wave)
    # graticule
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=CRT_DARK, width=1)
    if m > 1 and signal >= 0.18:
        pts = []
        for j in range(m):
            ang = 2 * math.pi * j / m - math.pi / 2
            rad = r + max(-amp_cap, min(amp_cap, float(wave[j]) * amp_cap))
            pts.append((cx + int(rad * math.cos(ang)), cy + int(rad * math.sin(ang))))
        if len(pts) > 1:
            draw.line(pts + [pts[0]], fill=_green(0.85), width=2)
        # electron sweep dot + short decaying trail (lookback over waves)
        for k, past in enumerate(wave_window[-_TRAIL_N:]):
            pm = len(past)
            if pm < 1:
                continue
            idx = int((fi - (len(wave_window[-_TRAIL_N:]) - 1 - k)) % pm)
            ang = 2 * math.pi * idx / pm - math.pi / 2
            rad = r + max(-amp_cap, min(amp_cap, float(past[idx]) * amp_cap))
            dx = cx + int(rad * math.cos(ang))
            dy = cy + int(rad * math.sin(ang))
            tail = (k + 1) / _TRAIL_N
            sz = max(1, int(3 * tail))
            draw.ellipse([(dx - sz, dy - sz), (dx + sz, dy + sz)], fill=_green(0.3 + 0.7 * tail))
    else:
        # idle: a jittering baseline circle (deterministic jitter)
        rng = _rng(env.get("key", "scopes"), fi, "idle_scope")
        pts = []
        for j in range(0, 360, 12):
            ang = math.radians(j)
            jit = int(rng.integers(-2, 3))
            pts.append((cx + int((r + jit) * math.cos(ang)),
                        cy + int((r + jit) * math.sin(ang))))
        if len(pts) > 1:
            draw.line(pts + [pts[0]], fill=_green(0.4), width=1)


# --------------------------------------------------------------------------- #
# Geometry (from the DELIVERY size; never baked 1920)
# --------------------------------------------------------------------------- #
def _gutter_geom(out_w, out_h):
    portrait_w = round(480 * out_w / 1472)
    gutter = max(0, (out_w - portrait_w) // 2)
    left_cx = gutter // 2
    right_cx = out_w - gutter // 2
    cy = out_h // 2
    r = int(min(gutter * 0.36, out_h * 0.30))
    return gutter, left_cx, right_cx, cy, r


def _centre_geom(out_w, out_h):
    cx, cy = out_w // 2, out_h // 2
    r = int(min(out_w * 0.16, out_h * 0.30))
    return cx, cy, r


#: Caption safe-area = the lower fraction reserved for SDH subtitles; the bars
#: strip stays ABOVE it so captions are never occluded (defense-in-depth on top of
#: caption-layer-last ordering -- see CODER_KICKOFF_BARS_OVERLAY.md "Caption layering").
_BARS_CAPTION_SAFE_FRAC = 0.15


def _bars_geom(out_w, out_h):
    """(x, y, w, h) for the landscape bottom-bars strip. A wide green frequency
    strip seated just ABOVE the caption safe-area (the lower ~15%), with a small
    side margin -- the "old-school bottom-of-screen" accent over real video."""
    margin = int(out_w * 0.05)
    strip_h = max(8, int(out_h * 0.10))
    safe = int(out_h * _BARS_CAPTION_SAFE_FRAC)
    y = out_h - safe - strip_h
    return margin, y, out_w - 2 * margin, strip_h


# --------------------------------------------------------------------------- #
# Aspect probe (memoized; un-probeable -> suppress + log)
# --------------------------------------------------------------------------- #
def _probe_is_portrait(path, ffprobe, cache):
    """``True`` / ``False`` / ``None`` (un-probeable -> the beat is suppressed).

    The POLICY is this node's and does not move: an unreadable clip suppresses
    its scope rather than guessing an aspect and drawing gutters down the middle
    of a landscape shot. Only the launch is shared -- and ``ffprobe`` may now
    arrive as the bare default here, because the boundary reads that as "no
    preference" and consults OTR_FFPROBE for us."""
    if path in cache:
        return cache[path]
    res = None
    try:
        if path and os.path.isfile(path):
            out = _ffp.probe_raw(
                ["-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                 path],
                ffprobe=ffprobe, timeout=10)
            txt = (out.stdout or "").strip().splitlines()
            if out.returncode == 0 and txt:
                w, h = (int(x) for x in txt[0].split("x")[:2])
                res = h > w
    except Exception as exc:  # noqa: BLE001
        log.warning("[SceneAwareScopes] ffprobe failed for %r (%s) -> suppress",
                    path, exc)
        res = None
    cache[path] = res
    return res


# --------------------------------------------------------------------------- #
# Per-frame plan: classify each absolute frame range from the manifest segments
# --------------------------------------------------------------------------- #
def plan_scope_frames(manifest, out_w, out_h, ffprobe="ffprobe",
                      landscape_bars="off"):
    """Return (plan, total): plan[fi] is one of:
      ('gutters', left_cx, right_cx, cy, r)   -- clip + PORTRAIT
      ('centre',  cx, cy, r)                   -- head/inter gap (keep alive)
      ('bars',    x, y, w, h)                  -- clip + LANDSCAPE when
                                                  landscape_bars='bottom' (the
                                                  optional green bottom-strip)
      None                                     -- suppress (clip+landscape with
                                                  bars OFF / tail-credits /
                                                  un-probeable)
    Uses plan_timeline_segments for the frame-accurate beat ranges (no source
    probe for frame counts -- only an aspect probe per clip path).

    ``landscape_bars`` ('off' DEFAULT | 'bottom'): 'off' is byte-identical to the
    pre-overlay behavior (landscape clips suppress); 'bottom' paints a green
    frequency-bar strip over a REAL landscape clip (portrait is False, never an
    un-probeable None). Portrait gutters, gap centres and the tail-credits
    suppression are UNCHANGED in both modes."""
    try:
        from .otr_silent_composite import plan_timeline_segments  # type: ignore
    except ImportError:  # pragma: no cover -- flat test import
        from otr_silent_composite import plan_timeline_segments  # type: ignore

    fps = int((manifest or {}).get("fps") or 25)
    target = (manifest or {}).get("total_target_frames")
    segs, total = plan_timeline_segments(
        manifest, floor_available=True, target_total_frames=target, fps=fps)
    if int(total) <= 0:
        raise ValueError("OTR_SceneAwareScopes: empty/zero-length manifest "
                         "(total_target_frames<=0) -- nothing to render.")

    # absolute frame ranges + the last-beat-end (for tail/credits detection)
    ranges, cursor, last_beat_end = [], 0, 0
    for seg in segs:
        n = int(seg.get("n_frames") or 0)
        start, end = cursor, cursor + n
        ranges.append((start, end, seg))
        if seg.get("source") == "clip":
            last_beat_end = max(last_beat_end, end)
        cursor = end

    gut = _gutter_geom(out_w, out_h)
    cen = _centre_geom(out_w, out_h)
    bars = _bars_geom(out_w, out_h)
    want_bars = str(landscape_bars or "off").lower() == "bottom"
    cache = {}
    plan = [None] * total
    for (start, end, seg) in ranges:
        src = seg.get("source")
        mode = None
        if src == "clip":
            portrait = _probe_is_portrait(seg.get("path"), ffprobe, cache)
            if portrait is True:
                mode = ("gutters", gut[1], gut[2], gut[3], gut[4])
            elif want_bars and portrait is False:
                # REAL landscape clip + bars enabled -> the bottom green strip.
                # Un-probeable (portrait is None) still suppresses (unchanged).
                mode = ("bars", bars[0], bars[1], bars[2], bars[3])
            else:  # landscape with bars OFF, OR un-probeable -> suppress
                mode = None
        else:  # floor / black gap
            if start >= last_beat_end and last_beat_end > 0:
                mode = None  # tail credits post-roll -> do not cover
            else:
                mode = ("centre", cen[0], cen[1], cen[2])  # keep the gap alive
        for fi in range(max(0, start), min(total, end)):
            plan[fi] = mode
    return plan, total


# --------------------------------------------------------------------------- #
# Silent encoder (the floor's _encode_mp4 HARD-REQUIRES audio; this node needs
# the -an variant matching the blend's input contract: yuv420p / CFR / 25fps).
#
# THE COPY IS GONE (2026-07-28). This module used to carry its own
# ``_encode_silent_mp4`` -- the THIRD copy of the same encoder in the tree,
# assembling a byte-for-byte identical ffmpeg command, and carrying every
# defect the shared one was just fixed for: ``total`` accepted and never read,
# the rawvideo ``-s`` built from the caller's w/h while the pipe carried
# whatever the generator painted, no per-frame shape or dtype check, nvenc
# selected with no minimum-canvas floor, and stderr on a PIPE read only after
# the whole stream was written -- which deadlocks without raising, so the child
# was never reaped and kept the output file open.
#
# It now calls ``_otr_shared.scope_draw.encode_silent_mp4``, which is exactly
# the refactor that module's own docstring anticipated. The SEPARATION
# INVARIANT is unharmed and points the other way: scope_draw must not import
# the floor or the overlay NODE, and this node already imports
# ``freq_bars_green`` from it. Hardening a third dialect instead would have
# left three encoders to fix the next time one of them is wrong.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The node
# --------------------------------------------------------------------------- #
class SceneAwareScopes:
    """Render scene-aware GREEN-ONLY scopes on BLACK -> scopes_only.mp4."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "render_scopes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scopes_mp4_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_manifest_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "clip_manifest_json from OTRVideoRenderBatch "
                               "(per-beat engine_id + start_s + path)."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Master audio for analysis ONLY (never decoded "
                               "into the video). Absent -> silent idle scopes."}),
                "out_w": ("INT", {
                    "default": 1920, "min": 320, "max": 7680, "step": 2,
                    "tooltip": "Scopes output width (px, even values). "
                               "Profile/platform-owned; matches the final "
                               "master canvas, not the render-tier canvas.",
                }),
                "out_h": ("INT", {
                    "default": 1080, "min": 240, "max": 4320, "step": 2,
                    "tooltip": "Scopes output height (px, even values). "
                               "Profile/platform-owned -- see out_w.",
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg", "multiline": False,
                    "tooltip": "DEPRECATED and IGNORED (2026-09-04). A workflow value cannot name the binary this pack runs -- it arrives over an unauthenticated /prompt request. Set the OTR_FFMPEG environment variable to pin a build.",
                }),
                # APPEND-ONLY (BUG-LOCAL-097 positional rule -- keep LAST). 'off'
                # is byte-identical to today (landscape clips show nothing); 'bottom'
                # paints a green audio-reactive frequency strip along the bottom of
                # any LANDSCAPE clip (over any engine), above the caption safe-area.
                "landscape_bars": (["off", "bottom"], {
                    "default": "off",
                    "tooltip": "Optional old-school green audio bars along the "
                               "bottom of landscape video (any engine). off = "
                               "unchanged; captions always stay on top."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _time.time()

    def render_scopes(self, clip_manifest_json, audio=None,
                      out_w=1920, out_h=1080, ffmpeg="ffmpeg",
                      landscape_bars="off"):
        # B1 (2026-09-04): the widget is UNTRUSTED /prompt input, not
        # operator intent. Discarded HERE, at the node boundary, so no
        # helper underneath can be handed it.
        try:
            from ._otr_shared.ffmpeg import widget_ffmpeg_is_ignored
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_shared.ffmpeg import widget_ffmpeg_is_ignored  # type: ignore
        ffmpeg = widget_ffmpeg_is_ignored(ffmpeg, "OTR_SceneAwareScopes")
        import json
        try:
            manifest = json.loads(clip_manifest_json) if clip_manifest_json else {}
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"OTR_SceneAwareScopes: bad clip_manifest_json: {exc}")
        if not isinstance(manifest, dict) or not manifest.get("clips"):
            raise ValueError("OTR_SceneAwareScopes: empty manifest (no clips) "
                             "-- fail early rather than render an empty file.")

        out_w, out_h = int(out_w), int(out_h)
        fps = 25  # HARD-LOCK 25 across planner / analysis / encode
        # The seven lines that used to live here found ffmpeg, LOWER-CASED its
        # basename, swapped ffmpeg->ffprobe and fell back to a bare literal.
        # Lower-casing a path is fine on Windows and wrong anywhere else, and
        # the fallback never consulted OTR_FFPROBE at all.
        probe = _ffp.resolve_ffprobe(ffmpeg=ffmpeg) or "ffprobe"

        plan, total = plan_scope_frames(manifest, out_w, out_h, ffprobe=probe,
                                        landscape_bars=landscape_bars)
        # THE MANIFEST'S episode_id BECOMES A FILENAME, so it is reduced to a
        # safe token first (2026-09-05). `clip_manifest_json` is a workflow
        # STRING, so `episode_id` is caller-chosen; unsanitized it is joined
        # into `otr_scopes_{key}_{ts}.mp4` below and separators or `..` walk the
        # write out of the tmp tier -- on Windows the `otr_scopes_` prefix does
        # not stop it, because `..` collapses lexically before the filesystem
        # sees the path. A filename cannot hold a separator, so whitelist rather
        # than reject: this is a label, and a stripped one still renders.
        key = _re.sub(r"[^A-Za-z0-9_.-]", "_",
                      str(manifest.get("episode_id") or "scopes")).strip("._-") or "scopes"

        # -- audio analysis (optional; absent -> zero arrays, NOT _analyze) --
        if audio is not None:
            import torch  # lazy (ComfyUI runtime only)
            wf = audio["waveform"]
            sr = int(audio["sample_rate"])
            if wf.dim() == 3:
                a = wf[0].mean(dim=0).cpu().numpy()
            elif wf.dim() == 2:
                a = wf.mean(dim=0).cpu().numpy()
            else:
                a = wf.cpu().numpy()
            # BUG-LOCAL-406 FIX: the scopes track MUST span the full MASTER-audio
            # length (= the composite's master-extended length), NOT just the
            # beats-only total_target_frames. The downstream §4D 3-input blend
            # (otr_post_upscale_procgen_blend) uses blend filter shortest=1, so a
            # scopes input shorter than the master clamps the WHOLE blended video
            # below the master length -> OTR_MasterAudioMux then clone-holds the
            # last frame over the remaining closing-theme audio (the FREEZE +
            # "no HUD treatment" the operator saw; regressed when the scopes node
            # landed 2026-06-13). Pad the plan tail to the master length; tail
            # frames stay None (no scope drawn -- a black frame the lighten-blend
            # ignores), matching the suppressed credits/post-roll region.
            master_frames = int(np.ceil(len(a) / sr * fps)) if sr else total
            if master_frames > total:
                plan = plan + [None] * (master_frames - total)
                log.info("[SceneAwareScopes] BUG-406: extended scopes %d -> %d "
                         "frames to span the master audio (no §4D blend clamp)",
                         total, master_frames)
                total = master_frames
            volume, freqs, waves = _analyze_audio_np(a, sr, total, fps)
        else:
            volume = [0.0] * total
            freqs = [np.zeros(_FFT_SPOKES, dtype=np.float32)] * total
            waves = [np.zeros(200, dtype=np.float32)] * total
        signal, trig, loss = _dual_ema(volume)

        def _frame(fi):
            img = Image.new("RGB", (out_w, out_h), CRT_BLACK)
            mode = plan[fi]
            if mode is not None:
                draw = ImageDraw.Draw(img)
                lo = max(0, fi - _TRAIL_N + 1)
                fwin = freqs[lo:fi + 1]
                wwin = waves[lo:fi + 1]
                env = {"fi": fi, "fps": fps, "key": key,
                       "vol": float(volume[fi]), "signal": float(signal[fi]),
                       "loss": float(loss[fi]), "trig": float(trig[fi])}
                if mode[0] == "gutters":
                    _, lcx, rcx, cy, r = mode
                    draw_fft_scope(draw, lcx, cy, r, fwin, env)
                    draw_scope(draw, rcx, cy, r, wwin, env)
                elif mode[0] == "bars":
                    # GREEN-ONLY bottom strip over a landscape clip (DRY: the
                    # shared green freq-bar routine). Single-frame spectrum.
                    try:
                        from ._otr_shared.scope_draw import freq_bars_green
                    except ImportError:  # pragma: no cover -- flat test import
                        from _otr_shared.scope_draw import freq_bars_green
                    _, bx, by, bw, bh = mode
                    freq_bars_green(draw, freqs[fi], bx, by, bw, bh)
                else:  # centre (gap): FFT outer + oscilloscope inner, concentric
                    _, cx, cy, r = mode
                    draw_fft_scope(draw, cx, cy, r, fwin, env)
                    draw_scope(draw, cx, cy, int(r * 0.6), wwin, env)
            return np.asarray(img)

        def _gen():
            import concurrent.futures
            mw = min(16, (os.cpu_count() or 4) + 2)
            chunk = mw * 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=mw) as ex:
                for start in range(0, total, chunk):
                    end = min(start + chunk, total)
                    for frame in ex.map(_frame, range(start, end)):
                        yield frame

        ts = _time.strftime("%Y%m%d_%H%M%S")
        # OH-2 hygiene: write the scopes intermediate to the OTR-controlled scratch
        # tier (otr/episodes/_shared/tmp -- the OH-3 janitor sweeps it), NEVER the
        # ambient system temp dir. A server NOT booted via the soak launcher leaves
        # TEMP unrepointed, so gettempdir() lands otr_scopes_*.mp4 in
        # %LOCALAPPDATA%\Temp and trips the soak hygiene gate even though the render
        # is fine (2026-06-30). The downstream OTR_PostUpscaleProcgenBlend consumes
        # this path; cleanup is the janitor's -- never delete it in the producer.
        try:
            try:
                from ._otr_paths import otr_shared_tmp_dir
            except ImportError:                      # flat (sys.path) load
                from _otr_paths import otr_shared_tmp_dir  # type: ignore
            _tmp_root = str(otr_shared_tmp_dir())
            os.makedirs(_tmp_root, exist_ok=True)
        except Exception as _tmp_exc:                # noqa: BLE001
            _tmp_root = tempfile.gettempdir()
            log.warning("[SceneAwareScopes] OTR tmp tier unavailable (%s); "
                        "falling back to %s (test/headless only)", _tmp_exc, _tmp_root)
        out_path = os.path.join(_tmp_root, f"otr_scopes_{key}_{ts}.mp4")
        log.info("[SceneAwareScopes] %d frames @ %dx%d 25fps -> %s",
                 total, out_w, out_h, out_path)
        try:
            from ._otr_shared import scope_draw as _sd
        except ImportError:  # pragma: no cover -- flat (sys.path) test import
            from _otr_shared import scope_draw as _sd  # type: ignore
        _sd.encode_silent_mp4(_gen(), total, out_path, out_w, out_h, fps, ffmpeg)
        return {"result": (out_path,)}


__all__ = ["SceneAwareScopes", "draw_fft_scope", "draw_scope",
           "plan_scope_frames", "_analyze_audio_np", "_dual_ema"]
