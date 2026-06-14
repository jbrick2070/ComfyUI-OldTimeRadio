# GROUNDING -- real node IO for the v2 scene-aware overlay. Do not invent fields.

# === otr_silent_composite.py : the CLIP MANIFEST shape (plan_timeline_segments) ===
def plan_timeline_segments(manifest, *, floor_available=False, floor_frames=0,
                           target_total_frames=None, fps=None):
    """Pure: the frame-accurate per-beat segment plan from a clip manifest.

    POSITION mode (every beat carries ``start_s`` AND a ``target_total_frames``
    master length is given): place each beat at ``round(start_s*fps)`` and
    floor/black gap-fill the head (the +intro shift), the inter-beat silences,
    and the tail (the closing theme) so the assembled length == the master length
    (the pre-mux A/V-sync guard). SEQUENTIAL mode (no start_s): concat the beats,
    then tail-fill to ``target_total_frames`` when given. Each beat is EXACTLY
    ``target_frame_count`` frames; a real on-disk clip is used, else the floor
    (timeline-aligned slice in sequential mode) or black. Returns ``(segments,
    total_frames)``; each segment is ``{order, shot_id, source, path,
    src_start_frame, n_frames, engine_id}``. Frame counts only."""
    rows = [r for r in ((manifest or {}).get("clips") or [])
            if int((r or {}).get("target_frame_count") or 0) > 0]
    fps = int(fps or (manifest or {}).get("fps") or 25)
    ff = int(floor_frames or 0)
    gap_src = "floor" if floor_available else "black"
    segments = []
    cursor = 0

    def emit(source, path, n, src_start, shot_id=None, engine_id=None):
        if int(n) <= 0:
            return
        segments.append({
            "order": len(segments), "shot_id": shot_id, "source": source,
            "path": path or "", "src_start_frame": int(src_start),
            "n_frames": int(n), "engine_id": engine_id})

    def _floor_aligned(start_cursor, n):
        """Timeline-aligned floor slice start (production restore 2026-06-10):
        gaps used to slice the procgen from frame 0 (the open repeated); the
        OLD episodes ran the procgen CONTINUOUSLY underneath, so head gaps show
        the radio open, inter-beat gaps the matching mid-roll, and the TAIL the
        rolling-credits post-roll. Clamped so n frames remain. (The procgen may
        run at a different fps than the composite -- ~4% drift at 24v25 -- an
        acceptable skew for a background/credits roll.)"""
        if gap_src != "floor":
            return 0
        return min(int(start_cursor), max(0, ff - int(n)))

    positioned = (target_total_frames is not None and rows
                  and all(r.get("start_s") is not None for r in rows))
    if positioned:
        for r in sorted(rows, key=lambda x: float(x.get("start_s") or 0)):
            n = int(r.get("target_frame_count") or 0)
            start_frame = int(round(float(r.get("start_s") or 0) * fps))
            if start_frame > cursor:                       # head / inter-beat gap
                gap_n = start_frame - cursor
                emit(gap_src, "", gap_n, _floor_aligned(cursor, gap_n))
                cursor = start_frame
            if r.get("exists") and r.get("path"):
                emit("clip", r.get("path"), n, 0, r.get("shot_id"), r.get("engine_id"))
            else:
                emit(gap_src, "", n, _floor_aligned(cursor, n),
                     r.get("shot_id"), r.get("engine_id"))
            cursor += n
    else:                                                  # SEQUENTIAL (legacy)
        for r in rows:
            n = int(r.get("target_frame_count") or 0)
            if r.get("exists") and r.get("path"):
                emit("clip", r.get("path"), n, 0, r.get("shot_id"), r.get("engine_id"))
            elif floor_available:
                start = min(cursor, max(0, ff - n)) if ff else cursor
                emit("floor", "", n, start, r.get("shot_id"), r.get("engine_id"))
            else:
                emit("black", "", n, 0, r.get("shot_id"), r.get("engine_id"))
            cursor += n
    if target_total_frames is not None and int(target_total_frames) > cursor:
        # Tail to the master length. With a floor available this slice is the
        # END of the procgen -- the ROLLING-CREDITS post-roll riding under the
        # closing theme (production restore 2026-06-10).
        tail_n = int(target_total_frames) - cursor
        emit(gap_src, "", tail_n, _floor_aligned(cursor, tail_n))
        cursor = int(target_total_frames)
    return segments, cursor


# === otr_silent_composite.py : OTRSilentComposite INPUT_TYPES (clip_manifest_json) ===
class OTRSilentComposite:
    """Registered as ``OTR_SilentComposite``. Render output -> ONE always-silent
    canonical video (bt709/yuv420p/CFR, audio stripped). Mux happens downstream."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "composite"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("silent_video_path", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": (
                        "Base video to normalize (M1: the OTR_SignalLostVideo "
                        "radio-floor mp4). Any audio is STRIPPED here (V-1)."
                    ),
                }),
            },
            "optional": {
                "canvas_w": ("INT", {"default": 1472, "min": 16, "max": 7680}),
                "canvas_h": ("INT", {"default": 832, "min": 16, "max": 4320}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Silent composite path. Empty -> <output>/otr/episodes/<stem>_silent.mp4.",
                }),
                "gate_in": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
                "clip_manifest_json": ("STRING", {
                    "default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": (
                        "OTR_VideoRenderBatch(mode=episode) clip manifest. When "
                        "it carries per-beat clips the composite ASSEMBLES a "
                        "frame-accurate CFR timeline (each beat conformed to its "
                        "audio-derived frame count; gaps filled from "
                        "base_video_path). Empty -> the single-base floor path."
                    ),
                }),
            },
        }


# === otr_post_upscale_procgen_blend.py : green-only blend filter (R/B zeroed, screen) ===
    # BUG-LOCAL-104 green-only overlay + BUG-LOCAL-105 RGB planar pin:
    # when enabled, zero the procgen R and B channels via colorchannelmixer
    # BEFORE the blend so only the procgen G channel survives. THEN pin
    # both inputs to gbrp (planar RGB) before blend and back to yuv420p
    # after, so the per-channel `blend` math runs in RGB instead of YUV.
    # Without the format pin, ffmpeg auto-converts to yuv420p to match
    # libx264's pix_fmt and the blend filter ends up running 'lighten'
    # on Y/U/V planes -- which produces color-shifting garbage that
    # swallows the 1%-sparse green wireframe entirely. With the pin,
    # 'lighten/screen/addition' run as honest per-RGB-channel ops and
    # the green CRT overlays cleanly on top of any source color.
    #
    # Filter chain when green_only_overlay=True:
    #   [1:v] scale -> crop -> setpts -> lutrgb (crush) ->
    #         colorchannelmixer (zero R+B) -> format=gbrp [pgn]
    #   [0:v] format=gbrp [main]
    #   [main][pgn] blend=mode:opacity:shortest -> format=yuv420p [v]
    #
    # Filter chain when green_only_overlay=False (legacy path):
    #   [1:v] scale -> crop -> setpts -> lutrgb (crush) [pgn]
    #   [0:v][pgn] blend=mode:opacity:shortest [v]
    if green_only_overlay:
        green_only_step = (
            ",colorchannelmixer="
            "rr=0:rg=0:rb=0:"
            "gr=0:gg=1:gb=0:"
            "br=0:bg=0:bb=0"
            ",format=gbrp"
        )
        main_format_step = "[0:v]format=gbrp[main];"
        main_input_label = "[main]"
        post_blend_format = ",format=yuv420p"
    else:
        green_only_step = ""
        main_format_step = ""
        main_input_label = "[0:v]"
        post_blend_format = ""

# === otr_post_upscale_procgen_blend.py : PostUpscaleProcgenBlend INPUT_TYPES ===
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mp4_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "Path to RTXUpscale output mp4 "
                        "(1920x1080 HuMo + LTX composite)."
                    ),
                }),
                "procgen_mp4_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "Path to OTR_SignalLostVideo procgen mp4 "
                        "(1920x1080 native per BUG-030 Phase B)."
                    ),
                }),
            },
            "optional": {
                "blend_mode": (_BLEND_MODE_CHOICES, {"default": _DEFAULT_BLEND_MODE}),
                "blend_opacity": ("FLOAT", {
                    "default": _DEFAULT_BLEND_OPACITY,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Procgen overlay opacity. 0.0 = procgen invisible "
                        "(equivalent to bypass). 1.0 = procgen contributes "
                        "at full strength via the selected blend mode. "
                        "Default 1.0 + 'screen' mode (BUG-LOCAL-096) gives "
                        "the canonical bright-additive overlay -- procgen "
                        "colors at full intensity, upscale still visible "
                        "underneath. Drop to 0.5 for a moderate sheen "
                        "instead of full strength."
                    ),
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "multiline": False,
                    "tooltip": "ffmpeg binary path or PATH-resolvable name.",
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Skip the blend entirely and copy source -> output "
                        "verbatim. Useful for A/B comparison vs the procgen "
                        "blend, or when procgen visual is unwanted (clean "
                        "uplift for an external editor)."
                    ),
                }),
                "out_suffix": ("STRING", {
                    "default": "_procgen_blended",
                    "multiline": False,
                    "tooltip": (
                        "Filename suffix for the blended output. Final "
                        "filename: ``<source_stem><out_suffix>.mp4`` placed "
                        "in the same dir as ``source_mp4_path``."
                    ),
                }),

# === video_engine.py : _analyze_audio (RMS/FFT/wave per frame) ===
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


# === video_engine.py : SignalLostVideoRenderer INPUT_TYPES (NO manifest / NO engine plan) ===
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
                # is now SEPARATE from the per-clip-mux composite path
                # (which stays at 1472x832 native + RTXUpscaled to
                # 1920x1080). The OTR_PostUpscaleProcgenBlend node
                # overlays this 1920x1080 procgen on the upscaled mp4
                # at delivery res. Result: procgen's CRT scanline /
                # audio-reactive flicker stays CRISP at 1080p (would
                # have been smeared if upscaled by RTX VSR -- synthetic
                # patterns + AI upscaler = ringing / softening), and
                # fills the visible HuMo black pillarbox bars from
                # BUG-030 Phase A as the SIGNAL LOST visual signature.
                # 832x480 retained as legacy mode for the prior path
                # (composite-then-RTXUpscale-everything-together).
                "resolution": (["1920x1080", "1280x720", "832x480", "854x480", "3840x2160"], {
                    "default": "1920x1080",
                    "tooltip": "Procgen output resolution. 1920x1080 = delivery res for post-RTXUpscale blend (BUG-030 Phase B default). 832x480 was the prior default (rendered cheap, then upscaled with everything else; legacy mode). 1280x720 / 854x480 / 3840x2160 retained for one-off needs."
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
