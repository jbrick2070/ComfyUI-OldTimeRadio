"""
batch_ltx_render.py  --  OTR_BatchLTXRender ComfyUI node
========================================================

Render N LTX-2 video clips in lockstep from a Production Ledger.
Sister to ``batch_humo_render.py``: walks ``ledger.lines[]`` filtering
for non-character roles ({announcer, music_open, music_close,
music_inter, sfx}) and renders each as an LTX I2V clip with the
radio_bookend.png as the conditioning image.

Architecture (locked 2026-05-01 with Jeffrey, after BUG-LOCAL-129
settled that HuMo cannot animate non-face references):

    character lines         -> HuMo (existing batch_humo_render.py)
    announcer / music / sfx -> LTX-2 (this node)
                                I2V ref = radio_bookend.png
                                25 fps native, 480p, 8n+1 frames

Pattern adapted from Jeffrey's ComfyUI-Goofer ``GooferBatchVideo``
which is already proven on this exact Blackwell sm_120 hardware:
distilled LTX-2 sigma schedule, SamplerCustomAdvanced, video-only
decode (we discard LTX's optional audio output -- master_mix audio
is muxed at VideoComposite step, byte-identical to v1.5 baseline).

VRAM strategy:
  - Frame count cap: 257 (LTX VAE max). Per Jeffrey 2026-05-01 EVENING,
    his proven ComfyUI-Goofer node renders 257-frame clips at 768x512
    on this exact Blackwell hardware -- the trick is ``VAEDecodeTiled``
    at decode time. Gemini's 97-frame round-robin estimate assumed
    non-tiled decode; with tiled decode, 257 fits.
  - VAE decode uses ``VAEDecodeTiled`` (tile_size=512, overlap=64,
    temporal_size=4096, temporal_overlap=8) -- exactly the parameters
    Goofer ships with. This is the only reason 257 frames doesn't OOM.
  - Strict teardown after the loop (unload_all_models + gc +
    empty_cache + cuda.synchronize) per
    ``reference_chained_backend_teardown.md`` so HuMo (16.5 GB
    staged) loads cleanly in the next phase.

Per CLAUDE.md C5: weights stay at the model's native precision; we
do NOT re-quantize.

Per CLAUDE.md C7: this node generates VIDEO-ONLY clips. No audio is
written. Audio path stays untouched all the way through to
VideoComposite's final mux which uses ``-c:a copy`` from procgen.
"""
from __future__ import annotations

import gc
import logging
import math
import sys as _sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path-helper bootstrap (same pattern as batch_humo_render.py)
# ---------------------------------------------------------------------------
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))

# folder_paths is the ComfyUI canonical path resolver. The OTR helpers
# (otr_videos_dir, otr_stills_dir, etc.) wrap folder_paths internally for
# the broadcast tree under output/otr/, but referencing folder_paths here
# explicitly satisfies the Bug Bible BUG-01.02 contract that every
# OUTPUT_NODE file references the canonical resolver.
#
# 2026-05-04 BUG-LOCAL-091 follow-up: wrapped in try/except so headless
# pytest collection doesn't blow up (folder_paths is provided by the
# ComfyUI runtime; pytest doesn't have it in scope). The runtime path
# inside execute() goes through _otr_paths helpers which already have
# their own folder_paths fallback chain.
try:
    import folder_paths  # noqa: F401,E402
except ImportError:
    folder_paths = None  # type: ignore[assignment]

from _otr_paths import (  # noqa: E402
    otr_audio_dir,
    otr_legacy_audio_dir,
    otr_stills_dir,
    otr_videos_dir,
)
from _otr_speaker_role import (  # noqa: E402
    SPEAKER_ROLE_CHARACTER,
    is_never_humo_role,
    resolve_speaker_role,
)

log = logging.getLogger("OTR.batch_ltx_render")


# ---------------------------------------------------------------------------
# LTX constants
# ---------------------------------------------------------------------------

# Match HuMo's frame rate so concat-demuxer at VideoComposite stage
# doesn't see frame rate seams. Gemini round-robin 2026-05-01 caught
# this as a stutter-prevention requirement.
LTX_FPS = 25

# 8n+1 frame count rule for LTX VAE temporal compression.
# BUG-LOCAL-091 (2026-05-04 EVENING): bumped from 177 (~7.08s) to 353
# (~14.12s) to match the post-BUG-086 HuMo cap. Lines exceeding the
# user-configurable per-chunk ceiling (the new ``clip_length`` widget,
# default 7.0s) fall into the chunking dispatch below. ComfyUI-Goofer
# proved 257 native is fine; 353 is untested-but-likely-fine because
# VAEDecodeTiled handles temporal chunking at the VAE level.
# 353 = 8*44+1 satisfies LTX's 8n+1 constraint.
LTX_MAX_FRAMES = 353
LTX_MIN_FRAMES = 9    # 8*1+1, smallest valid LTX render
# BUG-LOCAL-091 (2026-05-04 EVENING): when a non-character audio line
# exceeds the per-chunk ceiling, split it into consecutive LTX chunks of
# this many frames each, then ffmpeg-concat into a single per-line mp4.
# 177 frames (7.08s) per chunk matches the historically-stable LTX render
# size and the pre-bump LTX_MAX_FRAMES.
LTX_CHUNK_FRAMES = 177

# VAEDecodeTiled parameters (from ``ComfyUI-Goofer`` line 472-476).
# These have been empirically verified on RTX 5080 Blackwell --
# don't change without re-validating on hardware.
LTX_TILE_SIZE = 512
LTX_TILE_OVERLAP = 64
LTX_TEMPORAL_SIZE = 4096
LTX_TEMPORAL_OVERLAP = 8

# 480p native -- LTX 2B optimal resolution. Width chosen to match
# HuMo's 832 (so VideoComposite pillarbox math is identical for both
# clip sources).
# 2026-05-03 EVENING (BUG-LOCAL-030 revised spec): kept LTX at native
# 832x480 landscape per Jeffrey's revised wording ("ltx native landscape
# render downscaled to 1472x832"). An earlier revision bumped to
# 1216x704 for higher pre-upscale detail, but Jeffrey reverted to
# "native" twice -- safer trained-distribution default. VideoComposite
# pillarboxes to the 1472x832 canvas: 832x480 -> scale to height 832 =
# 1442x832 -> pad to 1472x832 with ~15px black per side. Final
# RTXUpscale 1472x832 -> 1920x1080 then post-upscale procgen blend
# (Phase B) for the SIGNAL LOST CRT signature.
LTX_WIDTH = 832
LTX_HEIGHT = 480

# Distilled sigma schedule from Jeffrey's ComfyUI-Goofer, proven on
# RTX 5080 Blackwell. 8 sampling steps, last sigma 0.0 = full denoise.
LTX_DISTILLED_SIGMAS = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]

# CFG = 1.0 for distilled LTX (no classifier-free guidance needed).
LTX_CFG = 1.0

# I2V conditioning strength: 0.75 = strong reference, but leaves room
# for the model to add motion. Goofer's empirically tuned default.
LTX_I2V_STRENGTH = 0.75

# 2026-05-01 Jeffrey ORIGINAL INTENT: feed radio still as BOTH start
# and end keyframes for seamless-loop ping-pong in VideoComposite.
# RETIRED 2026-05-03 EVENING per BUG-LOCAL-032: two strong anchors
# clamped LTX into near-static ping-pong.
# 2026-05-04 EVENING per BUG-LOCAL-095: even the start-frame guide
# (LTXVAddGuide) was acting as a hard keyframe pin and producing
# "ltx looks like a still" output. Replaced with the canonical
# LTXVImgToVideoConditionOnly i2v init which encodes the image into
# the first frames of the latent + adds a noise mask for free motion.
# Same proven path comfyui-data-media-machine uses.
LTX_END_FRAME_STRENGTH = 0.6  # DEPRECATED -- see batch_ltx_render.py BUG-032 fix block

# Speaker_role -> LTX prompt template. Builds a per-cue prompt anchored
# to the radio bookend image so the model knows we want the radio set
# animating, not random scenes.
_PROMPT_BY_ROLE = {
    "announcer": (
        "Vintage 1940s radio broadcast set, glowing tuning dial pulses "
        "gently, copper vacuum tubes warm amber glow, brass speaker "
        "grille vibrates subtly with the music, dim studio lighting, "
        "slow dolly forward, no people in frame, cinematic 35mm film grain"
    ),
    "music_open": (
        "Vintage 1940s radio at the start of a broadcast, dial sweeping "
        "across frequency band, oscilloscope-style display animating, "
        "warm amber tube glow brightening, slow camera dolly forward, "
        "no people in frame, cinematic mood, 35mm film grain"
    ),
    "music_close": (
        "Vintage 1940s radio at the end of a broadcast, dial settling, "
        "tube glow dimming gently, scope display fading, slow camera "
        "pull back, dim studio lighting, no people in frame, "
        "cinematic 35mm film grain"
    ),
    "music_inter": (
        "Vintage 1940s radio playing instrumental music, dial steady, "
        "oscilloscope-style display animating to the rhythm, copper "
        "tubes glow warm, gentle camera drift, no people in frame, "
        "cinematic 35mm film grain"
    ),
    "sfx": (
        "Vintage 1940s radio reacting to a sound effect, tube glow "
        "flickers, scope display spikes briefly, dial trembles, dim "
        "studio lighting, no people in frame, cinematic 35mm film grain"
    ),
}

# BUG-LOCAL-025 (Phase H, 2026-05-03): _PROMPT_BY_ROLE alone produces
# the SAME LTX motion prompt for every episode regardless of story
# style or scene context. Jeffrey: "be sure story arc or better
# shot/scene arc is being fed into FLUX and LTX as well to match the
# short." This builder enriches each role base prompt with:
#   - the line's scene_env (lookup via line.shot_id -> shot.scene_id
#     -> scene.env / scene.description)
#   - the episode's style from gen_params_initial.style
# Result: each LTX clip matches the show's tone AND the specific scene
# it accompanies. Bounded length: scene_env capped at 60 chars to
# avoid overwhelming the role's motion intent (LTX is i2v -- the
# image carries visual identity; the prompt mostly drives motion).
def _build_ltx_role_prompt(role: str, line: dict, ledger: dict) -> str:
    base = _PROMPT_BY_ROLE.get(role, _PROMPT_BY_ROLE["sfx"])

    # Scene context from line.shot_id -> shot.scene_id -> scene.env/description
    scene_env = ""
    if isinstance(line, dict) and isinstance(ledger, dict):
        shot_id = line.get("shot_id")
        if shot_id:
            shots = ledger.get("shots") or []
            scene_id = next(
                (s.get("scene_id") for s in shots
                 if isinstance(s, dict) and s.get("shot_id") == shot_id),
                None,
            )
            if scene_id:
                scenes = ledger.get("scenes") or []
                scene_obj = next(
                    (sc for sc in scenes
                     if isinstance(sc, dict) and sc.get("scene_id") == scene_id),
                    None,
                )
                if isinstance(scene_obj, dict):
                    raw_env = (
                        scene_obj.get("env")
                        or scene_obj.get("description")
                        or ""
                    )
                    if isinstance(raw_env, str):
                        scene_env = raw_env.strip()[:60].rstrip(",").strip()

    # Style from gen_params_initial / gen_params (post-Phase-G ledger
    # discovery means this reads the CURRENT episode's style, not a
    # stale leftover).
    style = ""
    if isinstance(ledger, dict):
        meta = ledger.get("meta") if isinstance(ledger.get("meta"), dict) else {}
        gp = meta.get("gen_params_initial")
        if not isinstance(gp, dict):
            gp = meta.get("gen_params")
        if isinstance(gp, dict):
            raw = gp.get("style")
            if isinstance(raw, str):
                style = raw.strip()

    parts = [base]
    if scene_env:
        parts.append(f"scene context: {scene_env}")
    if style:
        parts.append(f"{style} broadcast tone")
    return ", ".join(parts)


# Negative prompt: aggressively suppress face hallucination. LTX is
# generic motion (not face-locked like HuMo), so without this it might
# wander off the radio still and try to add a person.
_LTX_NEGATIVE = (
    "person, human, face, woman, man, hands, fingers, body, "
    "people in frame, ugly, deformed, low quality, jpeg artifacts, "
    "blurry, motion blur, talking, mouth, lips, smile"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ltx_length_for_dur(dur_s: float, *, fps: int = LTX_FPS) -> int:
    """Pick the smallest valid LTX ``length`` >= round(dur_s * fps).

    LTX VAE temporal compression requires length = 8n + 1 (vs HuMo's
    4n + 1). Floored at LTX_MIN_FRAMES, capped at LTX_MAX_FRAMES.

    Post BUG-LOCAL-091 (2026-05-04): LTX_MAX_FRAMES = 353 = ~14.12 s
    @ 25 fps. Long announcer lines that exceed the per-chunk widget
    cap go through the chunking dispatch in execute() rather than
    relying on the legacy clamp.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 7) // 8
    frames = 8 * n + 1
    if frames < LTX_MIN_FRAMES:
        frames = LTX_MIN_FRAMES
    if frames > LTX_MAX_FRAMES:
        frames = LTX_MAX_FRAMES
    return frames


def ltx_length_for_dur_uncapped(dur_s: float, *, fps: int = LTX_FPS) -> int:
    """Same as ``ltx_length_for_dur`` but without the LTX_MAX_FRAMES cap.

    BUG-LOCAL-091: used by the chunking dispatch to decide whether a
    line's audio duration exceeds the per-chunk ceiling. If the
    uncapped value > the user-configured chunk cap, the line is split
    into multiple chunks before rendering.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 7) // 8
    frames = 8 * n + 1
    if frames < LTX_MIN_FRAMES:
        frames = LTX_MIN_FRAMES
    return frames


def _concat_clips_via_ffmpeg(
    chunk_paths: list,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """BUG-LOCAL-091: concat per-chunk LTX mp4 files into a single
    per-line mp4 via ffmpeg's concat demuxer.

    Same pattern as the BUG-LOCAL-086 helper in batch_humo_render.py;
    duplicated here rather than imported to keep the LTX render path
    self-contained. Each chunk goes through ``_save_video_mp4`` with the
    same fps + codec, so concat with ``-c copy`` is safe.

    Single-chunk case is a defensive copy if not already at out_path.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if not chunk_paths:
        raise RuntimeError("_concat_clips_via_ffmpeg: empty chunk list")
    if len(chunk_paths) == 1:
        if Path(chunk_paths[0]) == Path(out_path):
            return out_path
        shutil.copy2(str(chunk_paths[0]), str(out_path))
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_fd, list_path = tempfile.mkstemp(prefix="ltx_concat_", suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                abs_p = str(Path(p).resolve()).replace("'", "'\\''")
                f.write(f"file '{abs_p}'\n")
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[:500]}"
            )
        return out_path
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


_NODE_CACHE: dict[str, Any] = {}


def _node(name: str):
    """Fetch a ComfyUI node class by its registered name (lazy)."""
    if name not in _NODE_CACHE:
        from nodes import NODE_CLASS_MAPPINGS  # noqa: late import
        cls = NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            raise RuntimeError(
                f"OTR_BatchLTXRender: node '{name}' not found in "
                f"NODE_CLASS_MAPPINGS. Required nodes: CLIPTextEncode, "
                f"LTXVImgToVideoConditionOnly (BUG-LOCAL-095), "
                f"LTXVConditioning, RandomNoise, CFGGuider, "
                f"KSamplerSelect, SamplerCustomAdvanced, VAEDecodeTiled, "
                f"EmptyLTXVLatentVideo. "
                f"Are stock comfy_extras + ComfyUI-LTXVideo installed?"
            )
        _NODE_CACHE[name] = cls
    return _NODE_CACHE[name]


def _call(name: str, **kwargs):
    """Instantiate a ComfyUI node and call its FUNCTION method."""
    cls = _node(name)
    fn_name = getattr(cls, "FUNCTION", "execute")
    obj = cls()
    result = getattr(obj, fn_name)(**kwargs)
    # io.NodeOutput unwrap (modern ComfyUI nodes return io.NodeOutput).
    if hasattr(result, "args"):
        return result.args
    return result


def _png_to_image_tensor(png_path: Path):
    """Load a PNG file as a ComfyUI IMAGE tensor [1, H, W, C] in [0, 1] float32.

    Mirrors ``batch_humo_render._png_to_tensor`` so the loading
    semantics are identical between the two render nodes -- VideoComposite
    sees clips that were both fed the same image-shape at I2V time.
    """
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, C]
    return tensor


def _resolve_radio_still_path(ledger: dict | None) -> Path | None:
    """Locate the radio_bookend.png for this episode.

    Mirrors ``batch_humo_render._resolve_radio_still_path`` exactly so
    the two render nodes find the same file via the same fallback
    chain (top-level -> meta -> filesystem-by-episode_id).
    """
    if not isinstance(ledger, dict):
        return None

    def _is_valid(p: Path) -> bool:
        try:
            if not p.is_file():
                return False
            return p.stat().st_size >= 256
        except Exception:  # noqa: BLE001
            return False

    cand = ledger.get("radio_bookend_path")
    if not cand:
        meta = ledger.get("meta") or {}
        if isinstance(meta, dict):
            cand = meta.get("radio_bookend_path")
    if cand:
        try:
            p = Path(cand)
        except Exception:  # noqa: BLE001
            p = None
        if p is not None and _is_valid(p):
            return p

    # Filesystem fallback by episode_id.
    eid_raw = ledger.get("episode_id")
    if not isinstance(eid_raw, str):
        return None
    eid = eid_raw.strip()
    if not eid or any(t in eid for t in ("/", "\\", "..", "\x00")):
        return None
    try:
        # BUG-LOCAL-028 fix (2026-05-03): pass episode_id so the lookup
        # resolves to ``output/otr/episodes/<eid>/stills/radio_bookend_<eid>.png``
        # (the canonical Phase B per-episode workspace). Prior to this
        # fix, ``otr_stills_dir()`` with no arg fell back to
        # ``output/otr/_legacy_stills/`` per ``nodes/_otr_paths.py:208-218``;
        # after BUG-028's writer fix, the radio bookend now lands in the
        # per-episode dir and this READ site needs to point there too,
        # otherwise LTX can't find the radio still and falls back to a
        # generic motion clip with no scene continuity.
        fs_path = otr_stills_dir(eid) / f"radio_bookend_{eid}.png"
    except Exception as exc:  # noqa: BLE001
        log.warning("[BatchLTXRender] otr_stills_dir lookup failed: %s", exc)
        return None
    if _is_valid(fs_path):
        return fs_path
    return None


def _save_video_mp4(
    *,
    images,
    out_path: Path,
    fps: int,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Save an IMAGE tensor [N, H, W, C] in [0,1] float as a silent
    mp4 to ``out_path`` via ffmpeg. Uses h264 yuv420p libx264 to match
    the encoding profile HuMo clips use, so VideoComposite's
    concat-demuxer can ``-c copy`` cleanly across both sources.

    No audio -- LTX clips are video-only. VideoComposite muxes the
    master mix at the final stage with ``-c:a copy``.
    """
    import numpy as np  # type: ignore
    import subprocess
    import tempfile
    import torch  # type: ignore

    if hasattr(images, "detach"):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    n_frames, h, w, _ = arr.shape

    tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_clip_"))
    try:
        # Pipe raw frames to ffmpeg via stdin.
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(int(fps)),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            # Match HuMo container timebase (Jeffrey review 2026-05-01,
            # video_composite.py BUG-129a static-fill comment).
            "-video_track_timescale", "12800",
            "-an",
            str(out_path),
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg failed encoding LTX clip: {stderr[:500]}"
            )
    finally:
        # Clean up the tempdir even on success.
        try:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()
        except Exception:  # noqa: BLE001
            pass
    return out_path


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BatchLTXRender:
    """Render N LTX-2 video clips in one graph execution for non-character
    ledger lines (announcer / music_* / sfx).

    OUTPUT_NODE = True so ComfyUI doesn't prune this side-effect node
    when downstream consumers don't fully chain (lesson from
    BUG-LOCAL-077).
    """

    CATEGORY = "OTR/v2/Visual"
    OUTPUT_NODE = True
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("clips_dir", "clip_count", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX-2 model (CheckpointLoaderSimple loading "
                               "ltx-video-2b-v0.9.x.safetensors)",
                }),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "ledger_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Production ledger -- inline JSON or a path "
                               "to *_ledger.json. Walked for non-character "
                               "speaker_role lines.",
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "ffmpeg binary path or PATH-resolvable name",
                }),
                "humo_clips_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "DAG SEQUENCING EDGE -- not data. Wire this to "
                        "BatchHumoRender.clips_dir (or .report) so ComfyUI "
                        "waits for HuMo to finish writing its per-line .mp4 "
                        "files before LTX starts loading. HuMo's strict "
                        "teardown then releases the 16.5 GB MODEL before "
                        "LTX claims VRAM. The value is intentionally NOT "
                        "used by execute() -- if you remove this input "
                        "thinking it's dead code, ComfyUI may schedule "
                        "LTX too early and OOM. Round-robin consult "
                        "2026-05-02 endorsed this pattern as acceptable "
                        "ComfyUI sequencing."
                    ),
                }),
                # BUG-LOCAL-097 (2026-05-04 EVENING): clip_length appended
                # at the END of optional. BUG-091 originally put it FIRST,
                # which inserted a new widget at position [3] of
                # widgets_values and shifted every subsequent saved value
                # by one slot -- existing workflow JSONs (otr_scifi_16gb_full.json
                # and any user-saved variants) read "ffmpeg" as FLOAT
                # clip_length and the workflow validation crashed at
                # `Failed to convert an input value to a FLOAT value:
                # clip_length, ffmpeg, could not convert string to float:
                # 'ffmpeg'` BEFORE the LTX node could even run. Moving
                # clip_length to the END means new widget appears at
                # position [5], past the existing 5 saved values, so old
                # workflows fall through to the FLOAT default (7.0)
                # cleanly. Backward-compat preserved.
                "clip_length": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.32,
                    "max": 14.12,
                    "step": 0.04,
                    "tooltip": (
                        "Max per-CHUNK duration in seconds (BUG-LOCAL-091, "
                        "matches BatchHumoRender behaviour). Lines whose "
                        "audio exceeds this are split into N consecutive "
                        "chunks rendered against the radio bookend, then "
                        "ffmpeg-concat into the final per-line mp4. Default "
                        "7.0 -> 175 frames -> 177 (LTX 8n+1 = 7.08s, the "
                        "historically-stable LTX render size). Bump up to "
                        "14.12 (353 frames) if VRAM holds, to single-pass "
                        "typical announcer monologues."
                    ),
                }),
            },
        }

    def execute(self, model, clip, vae, ledger_json, seed=1,
                ffmpeg="ffmpeg", humo_clips_dir="",
                clip_length=7.0):
        # NOTE: ``humo_clips_dir`` is intentionally consumed but unused.
        # See INPUT_TYPES tooltip -- it is a pure DAG sequencing edge so
        # ComfyUI schedules this node after BatchHumoRender finishes its
        # render + teardown. Do NOT remove it; doing so will let the LTX
        # checkpoint load race HuMo's 16.5 GB MODEL and OOM on 16 GB.
        del humo_clips_dir  # explicit: value ignored, edge is the contract
        t_start = time.time()
        report_lines: list[str] = []

        # ----------------------------------------------------------------
        # 1. Load the ledger + resolve the radio bookend png
        # ----------------------------------------------------------------
        ledger, ledger_path = self._load_ledger(ledger_json)
        if ledger is None:
            raise RuntimeError(
                "BatchLTXRender: ledger could not be loaded from inline "
                "JSON or path"
            )

        episode_id = str(ledger.get("episode_id") or "episode")
        report_lines.append(
            f"BatchLTXRender: episode={episode_id}"
        )

        radio_png = _resolve_radio_still_path(ledger)
        if radio_png is None:
            report_lines.append(
                "  WARNING: no radio_bookend.png on disk -- "
                "non-character lines will fall through to "
                "VideoComposite static-radio fallback (BUG-129a) "
                "with no I2V animation. Proceed anyway."
            )
            log.warning(
                "[BatchLTXRender] no radio bookend resolved for episode "
                "%s -- skipping LTX render entirely",
                episode_id,
            )
            return (
                str(otr_videos_dir(episode_id)), 0, "\n".join(report_lines)
            )

        try:
            ref_image = _png_to_image_tensor(radio_png)
        except Exception as exc:
            raise RuntimeError(
                f"BatchLTXRender: failed to load radio bookend png "
                f"{radio_png}: {exc}"
            )

        report_lines.append(
            f"  radio_bookend: {radio_png.name} "
            f"({radio_png.stat().st_size} bytes)"
        )

        # ----------------------------------------------------------------
        # 2. Walk ledger.lines[] for non-character roles
        # ----------------------------------------------------------------
        lines = ledger.get("lines") or []
        plan: list[dict] = []
        for ln in lines:
            line_id = str(ln.get("line_id") or "")
            if not line_id:
                continue
            speaker_role = resolve_speaker_role(ln)
            if speaker_role == SPEAKER_ROLE_CHARACTER:
                continue  # HuMo will render this in the next phase
            if not is_never_humo_role(speaker_role):
                continue  # safety: only render roles we're sure about
            dur_s = ln.get("dur_s")
            if not isinstance(dur_s, (int, float)) or float(dur_s) <= 0.0:
                continue
            dur_s = float(dur_s)

            # BUG-LOCAL-091 (2026-05-04): chunking dispatch. ``clip_length``
            # is the max single-pass duration (workflow widget, default 7.0s).
            # Lines whose audio exceeds it are split into N consecutive
            # chunks of <= clip_length each, rendered independently against
            # the radio bookend, then ffmpeg-concat into the final per-line
            # mp4. Pre-091 the cap silently truncated the back half of any
            # >7s line, leaving the radio scene frozen while the announcer
            # audio kept playing.
            chunk_max_dur_s = max(0.04, float(clip_length))
            if dur_s <= chunk_max_dur_s:
                chunk_specs = [{"dur_s": dur_s}]
            else:
                n_chunks = max(2, math.ceil(dur_s / chunk_max_dur_s))
                chunk_dur_s = dur_s / n_chunks
                chunk_specs = [{"dur_s": chunk_dur_s} for _ in range(n_chunks)]
                log.info(
                    "[BatchLTXRender] BUG-LOCAL-091: line %s dur_s=%.2fs > "
                    "clip_length=%.2fs -- splitting into %d chunks of "
                    "%.2fs each",
                    line_id, dur_s, float(clip_length),
                    n_chunks, chunk_dur_s,
                )
            ltx_length = ltx_length_for_dur(chunk_specs[0]["dur_s"])
            # BUG-LOCAL-025 (Phase H): enrich per-role base with line's
            # scene context + episode style instead of using the bare
            # hardcoded role prompt. Same role across two episodes
            # now produces visibly different motion intent.
            prompt_text = _build_ltx_role_prompt(speaker_role, ln, ledger)
            plan.append({
                "line_id": line_id,
                "speaker_role": speaker_role,
                "dur_s": dur_s,
                "ltx_length": ltx_length,
                "chunks": chunk_specs,
                "prompt_text": prompt_text,
            })

        if not plan:
            report_lines.append(
                "  no non-character lines to render -- LTX phase skipped"
            )
            log.info(
                "[BatchLTXRender] no non-character lines in ledger; "
                "skipping render entirely"
            )
            return (
                str(otr_videos_dir(episode_id)), 0, "\n".join(report_lines)
            )

        report_lines.append(
            f"  plan: {len(plan)} clip(s) "
            f"({sum(1 for e in plan if e['speaker_role']=='announcer')} announcer + "
            f"{sum(1 for e in plan if e['speaker_role'].startswith('music'))} music + "
            f"{sum(1 for e in plan if e['speaker_role']=='sfx')} sfx)"
        )

        # ----------------------------------------------------------------
        # 3. Pre-bake shared LTX objects (sampler, sigmas, base negative)
        # ----------------------------------------------------------------
        import torch  # type: ignore

        sampler_obj = _call("KSamplerSelect", sampler_name="euler")[0]
        sigmas = torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)

        neg_tokens = clip.tokenize(_LTX_NEGATIVE)
        base_negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # ----------------------------------------------------------------
        # 4. Per-line render loop
        # ----------------------------------------------------------------
        clips_dir = otr_videos_dir(episode_id)
        clips_dir.mkdir(parents=True, exist_ok=True)
        rendered_clips: list[dict] = []

        try:
            import comfy.model_management as mm  # type: ignore
            mm.load_models_gpu([model])
        except Exception as exc:
            log.debug("[BatchLTXRender] pre-pin model skipped: %s", exc)

        with torch.inference_mode():
            for idx, entry in enumerate(plan):
                line_id = entry["line_id"]
                prompt_text = entry["prompt_text"]
                chunks = entry.get("chunks") or [{"dur_s": entry["dur_s"]}]
                n_chunks = len(chunks)
                shot_t0 = time.time()

                # Anti-clobber (consult 2026-05-02 ChatGPT + Gemini):
                # HuMo writes <line_id>.mp4 for character lines; LTX writes
                # <line_id>.mp4 for non-character lines into the SAME dir.
                # Role filter (is_never_humo_role) should keep them disjoint,
                # but if a future ledger has a duplicate or drifted role and
                # LTX overwrites a HuMo character clip, the failure would be
                # invisible until final composite. Skip pre-existing files.
                pre_existing = clips_dir / f"{line_id}.mp4"
                if pre_existing.exists() and pre_existing.stat().st_size > 0:
                    log.warning(
                        "[BatchLTXRender] %s: skip -- existing clip on disk "
                        "(%s, %d bytes). Refusing to overwrite.",
                        line_id, pre_existing.name, pre_existing.stat().st_size,
                    )
                    report_lines.append(
                        f"  {line_id} ({entry['speaker_role']}): "
                        f"SKIP existing {pre_existing.name}"
                    )
                    continue

                try:
                    pos_tokens = clip.tokenize(prompt_text)
                    positive = clip.encode_from_tokens_scheduled(pos_tokens)

                    # Set frame_rate via LTXVConditioning (must be 25 to
                    # match HuMo at concat time, Gemini round-robin
                    # 2026-05-01 catch).
                    cond_pos, cond_neg = _call(
                        "LTXVConditioning",
                        positive=positive, negative=base_negative,
                        frame_rate=LTX_FPS,
                    )

                    # BUG-LOCAL-091: per-chunk render loop. Single-chunk
                    # path (n_chunks==1) renders one pass and writes
                    # directly to <line_id>.mp4 -- matches pre-091 layout.
                    # Multi-chunk renders each chunk to a part file then
                    # ffmpeg-concats. All chunks share the same prompt +
                    # ref_image (radio bookend) so the visual snap at the
                    # chunk boundary is the radio scene resetting; the
                    # alternative (carry last frame to next chunk) is a
                    # future upgrade documented as BUG-LOCAL-091a.
                    chunk_mp4_paths: list = []
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_dur_s = float(chunk["dur_s"])
                        chunk_ltx_length = ltx_length_for_dur(chunk_dur_s)

                        # Per-chunk shot_seed: stride-shifted so chunks
                        # 1+2 of the same line don't render with identical
                        # seed (would look quasi-identical at the join).
                        shot_seed = (
                            seed
                            + idx * 1009
                            + chunk_idx * 7919
                        ) & 0x7FFFFFFFFFFFFFFF

                        # Empty video latent at the chunk's requested length.
                        empty_latent = _call(
                            "EmptyLTXVLatentVideo",
                            width=LTX_WIDTH, height=LTX_HEIGHT,
                            length=chunk_ltx_length, batch_size=1,
                        )[0]

                        # BUG-LOCAL-095 (2026-05-04 EVENING): swapped from
                        # LTXVAddGuide to LTXVImgToVideoConditionOnly.
                        #
                        # LTXVAddGuide is for KEYFRAME PINNING inside an
                        # existing pipeline -- it attaches the image to
                        # positive/negative cond as a hard anchor at
                        # frame_idx and clamps motion away from that
                        # frame. Even at strength=0.75, frame 0 stays
                        # rigidly locked. This is what produced the
                        # "ltx looks like a still" artefact Jeffrey
                        # reported AFTER BUG-032 removed the end guide
                        # (the start guide was still pinning).
                        #
                        # LTXVImgToVideoConditionOnly is the canonical
                        # I2V INIT node -- it encodes the image into
                        # the FIRST FRAMES of the latent and creates a
                        # noise mask for strength control. The model
                        # sees "start with this image, then evolve
                        # freely" rather than "stay anchored to this
                        # frame". This matches the exact pattern in
                        # comfyui-data-media-machine's DMMBatchVideoGenerator
                        # (nodes/dmm_batch_video.py::_apply_i2v_conditioning,
                        # called via _call("LTXVImgToVideoConditionOnly",
                        # vae=vae, image=image, latent=latent,
                        # strength=strength)) which Jeffrey confirms
                        # produces visibly animated radio still output.
                        #
                        # Returns a single conditioned LATENT; the
                        # cond_pos / cond_neg from LTXVConditioning go
                        # straight to CFGGuider unchanged.
                        latent_chunk = _call(
                            "LTXVImgToVideoConditionOnly",
                            vae=vae,
                            image=ref_image,
                            latent=empty_latent,
                            strength=LTX_I2V_STRENGTH,
                        )[0]

                        # Sample with distilled schedule. Cond pos/neg
                        # come straight from LTXVConditioning (post-BUG-095);
                        # LTXVImgToVideoConditionOnly only modifies the
                        # latent, not the conditioning.
                        noise = _call("RandomNoise", noise_seed=shot_seed)[0]
                        guider = _call(
                            "CFGGuider",
                            model=model,
                            positive=cond_pos,
                            negative=cond_neg,
                            cfg=LTX_CFG,
                        )[0]
                        samples_out, _denoised = _call(
                            "SamplerCustomAdvanced",
                            noise=noise, guider=guider,
                            sampler=sampler_obj, sigmas=sigmas,
                            latent_image=latent_chunk,
                        )

                        # VAEDecodeTiled per Goofer's proven pattern --
                        # required for safe decoding at higher frame counts
                        # on this Blackwell hardware.
                        frames = _call(
                            "VAEDecodeTiled",
                            samples=samples_out, vae=vae,
                            tile_size=LTX_TILE_SIZE,
                            overlap=LTX_TILE_OVERLAP,
                            temporal_size=LTX_TEMPORAL_SIZE,
                            temporal_overlap=LTX_TEMPORAL_OVERLAP,
                        )[0]

                        # Per-chunk destination. Single-chunk writes to
                        # canonical <line_id>.mp4; multi-chunk writes to
                        # per-chunk part files for the concat below.
                        if n_chunks == 1:
                            chunk_dest = clips_dir / f"{line_id}.mp4"
                        else:
                            chunk_dest = clips_dir / (
                                f"{line_id}__chunk{chunk_idx + 1:02d}.mp4"
                            )
                        _save_video_mp4(
                            images=frames,
                            out_path=chunk_dest,
                            fps=LTX_FPS,
                            ffmpeg=ffmpeg,
                        )
                        chunk_mp4_paths.append(chunk_dest)

                    # BUG-LOCAL-091: concat per-chunk part files into the
                    # canonical per-line mp4 if multi-chunk. Same codec /
                    # sample rate / fps across all chunks (same
                    # _save_video_mp4 invocation), so concat-demuxer with
                    # -c copy is safe.
                    out_path = clips_dir / f"{line_id}.mp4"
                    if n_chunks > 1:
                        _concat_clips_via_ffmpeg(
                            chunk_mp4_paths, out_path, ffmpeg=ffmpeg,
                        )
                        for _cp in chunk_mp4_paths:
                            try:
                                _cp.unlink()
                            except OSError:
                                pass
                    # Use the per-line ltx_length for log/report -- it's
                    # the per-chunk frame count which the user expects to
                    # see for sizing context.
                    ltx_length = entry["ltx_length"]

                    shot_ms = int((time.time() - shot_t0) * 1000)
                    if n_chunks > 1:
                        log.info(
                            "[BatchLTXRender] %s done: role=%s "
                            "dur_s=%.3f -> %s (%d ms, %d chunks, "
                            "BUG-LOCAL-091 chunked + concat)",
                            line_id, entry["speaker_role"],
                            entry["dur_s"], out_path.name, shot_ms, n_chunks,
                        )
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}, "
                            f"{ltx_length}f x {n_chunks} chunks, "
                            f"{entry['dur_s']:.2f}s): {shot_ms} ms"
                        )
                    else:
                        log.info(
                            "[BatchLTXRender] %s done: role=%s length=%d "
                            "dur_s=%.3f -> %s (%d ms)",
                            line_id, entry["speaker_role"], ltx_length,
                            entry["dur_s"], out_path.name, shot_ms,
                        )
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}, "
                            f"{ltx_length}f, {entry['dur_s']:.2f}s): "
                            f"{shot_ms} ms"
                        )

                    # BUG-LOCAL-084 fix: stamp start_s + REAL on-disk dur_s
                    # so downstream consumers (audit, debug, future composite
                    # paths) have ground truth instead of the audio-target
                    # placeholder that BUG-LOCAL-033 surfaced.
                    _start_s_truth = None
                    _dur_s_truth = float(entry["dur_s"])  # fallback to audio target
                    try:
                        # Pull start_s from the matching ledger.lines[] entry
                        # by line_id. Lines are stamped by SceneSequencer with
                        # the canonical audio timeline start_s.
                        _ll = ledger.get("lines") or []
                        for _line in _ll:
                            if str(_line.get("line_id") or "") == line_id:
                                _ss = _line.get("start_s")
                                if isinstance(_ss, (int, float)):
                                    _start_s_truth = float(_ss)
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        # ffprobe the actual rendered file for real dur_s
                        # (BUG-LOCAL-033: never trust ledger.clips[].dur_s
                        # for LTX -- it's the AUDIO TARGET, not video real)
                        from . import _otr_probe as _PROBE  # type: ignore
                        _real = _PROBE.probe_duration_s(out_path)
                        if _real and _real > 0.0:
                            _dur_s_truth = float(_real)
                    except Exception:  # noqa: BLE001
                        pass
                    rendered_clips.append({
                        "line_id": line_id,
                        "speaker_role": entry["speaker_role"],
                        "mp4_path": str(out_path),
                        "ltx_length": ltx_length,
                        "start_s": _start_s_truth,
                        "dur_s": _dur_s_truth,
                        "audio_target_dur_s": float(entry["dur_s"]),
                        "ltx_render_ms": shot_ms,
                        "ref_png_name": radio_png.name,
                        "ref_source": "ltx-radio-bookend",
                        "source_kind": "ltx",
                        # BUG-LOCAL-091 traceability: how many chunks
                        # were rendered + concatenated for this line.
                        # 1 = single pass (pre-091 layout); >1 = chunked
                        # because audio dur exceeded clip_length.
                        "n_chunks": int(n_chunks),
                    })

                except Exception as exc:
                    log.exception(
                        "[BatchLTXRender] %s failed: %s", line_id, exc
                    )
                    report_lines.append(
                        f"  {line_id}: FAILED ({exc})"
                    )

        # ----------------------------------------------------------------
        # 5. Strict teardown so HuMo can load cleanly in the next phase
        #    (per reference_chained_backend_teardown.md)
        # ----------------------------------------------------------------
        try:
            import comfy.model_management as mm  # type: ignore
            mm.unload_all_models()
            log.info("[BatchLTXRender] teardown: unload_all_models")
        except Exception as exc:
            log.warning("[BatchLTXRender] teardown unload failed: %s", exc)
        gc.collect()
        try:
            import torch as _t  # type: ignore
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
            log.info("[BatchLTXRender] teardown: cuda empty_cache + sync")
        except Exception:  # noqa: BLE001
            pass

        # ----------------------------------------------------------------
        # 6. Stamp ledger.clips[] (additive -- HuMo will append its
        #    clips later in the next phase under the same clips_dir)
        # ----------------------------------------------------------------
        if ledger_path is not None and rendered_clips:
            try:
                from . import _otr_ledger as _OTRL  # type: ignore
                existing = ledger.get("clips") or []
                ledger["clips"] = list(existing) + rendered_clips
                _OTRL.save_ledger_safe(ledger_path, ledger)
                log.info(
                    "[BatchLTXRender] ledger updated: %d clip records -> %s",
                    len(rendered_clips), ledger_path.name,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[BatchLTXRender] ledger write-back failed: %s", exc
                )

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"BatchLTXRender: {len(rendered_clips)}/{len(plan)} clips in "
            f"{total_ms} ms (avg "
            f"{total_ms // max(len(rendered_clips), 1)} ms/clip)"
        )
        log.info(
            "[BatchLTXRender] complete: %d/%d clips in %d ms",
            len(rendered_clips), len(plan), total_ms,
        )

        return (str(clips_dir), len(rendered_clips), "\n".join(report_lines))

    # ---------------------------------------------------------------------
    # Ledger loading helpers (mirror batch_humo_render.py shape)
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_ledger(arg: str) -> tuple[dict | None, Path | None]:
        """Accept either:
          - inline JSON string (starts with '{' or '[') -- returns (dict, None)
          - path to *_ledger.json -- returns (dict, Path)
          - path to *.mp4 (audio episode); ledger inferred via
            suffix swap (.mp4 -> _ledger.json), since that's the
            convention OTR_SignalLostVideo / EpisodeAssembler write.
            Lets us wire BatchLTXRender's ledger_json input directly
            from SignalLostVideo.video_path -- no separate ledger
            output node required.
          - empty -> auto-pick newest non-pending in the canonical
            audio dirs (BUG-LOCAL-076 fallback chain).

        Returns (ledger_dict_or_None, ledger_path_or_None). Mirrors the
        contract of BatchHumoRender._load_ledger_with_path so the two
        sister nodes resolve the same ledger from the same SignalLostVideo
        STRING output.

        Resolution policy (post-2026-05-02 round-robin hardening on
        BUG-LOCAL-011):
          1. Use ``_otr_ledger.load_ledger_safe(path)`` for path loads
             so the read goes through the canonical loader (consistent
             ``[OTR_Ledger]`` log prefix; future-proof against any
             hardening added there).
          2. Tier 3 fuzzy directory scan from BatchHumoRender is
             intentionally NOT ported. Round-robin (gpt-5.5 + gemini-3.1
             + nemotron-49b) converged on rejecting it for LTX:
             non-deterministic, could plausibly bind to a wrong neighbour
             ledger, and silent fallback would burn ~1 hour rendering
             against bad metadata.
          3. If a Tier 1 or Tier 2 candidate file EXISTS but fails to
             parse / read, raise loud so the run halts immediately
             rather than fall through. Windows file-locking
             ``PermissionError`` falling silently to Tier 2/3 was the
             specific concern Gemini flagged; we honour it by failing
             fast.
        """
        import json as _json
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
        except Exception:  # noqa: BLE001
            _OTRL = None  # type: ignore

        def _read(p: Path) -> dict:
            """Load a JSON file via _OTRL when available; raise on failure.

            ``_OTRL.load_ledger_safe`` returns None on any error; we
            convert that to a RuntimeError so an existing-but-unreadable
            ledger does NOT silently fall through to a fuzzy / wrong
            candidate. Direct ``json.load`` is the fallback when
            ``_otr_ledger`` is unimportable (test contexts).
            """
            if _OTRL is not None:
                led = _OTRL.load_ledger_safe(p)
                if led is None:
                    raise RuntimeError(
                        f"BatchLTXRender: _OTRL.load_ledger_safe returned "
                        f"None for {p} (file exists; check WARNING log "
                        f"line above for the underlying parse / OS error)"
                    )
                return led
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f)

        s = (arg or "").strip()

        # Layer 0: empty input -> auto-pick newest non-pending ledger.
        if not s:
            audio_dirs = [otr_audio_dir(), otr_legacy_audio_dir()]
            cands = []
            for d in audio_dirs:
                if d.exists():
                    cands.extend(
                        p for p in d.glob("*_ledger.json")
                        if not p.name.startswith("pending_")
                    )
            if not cands:
                log.warning(
                    "[BatchLTXRender] ledger_json empty and auto-pick "
                    "found no ledger in audio dirs"
                )
                return None, None
            p = max(cands, key=lambda x: x.stat().st_mtime)
            return _read(p), p

        # Layer 1: inline JSON object.
        if s.startswith("{") or s.startswith("["):
            try:
                return _json.loads(s), None
            except _json.JSONDecodeError as exc:
                log.warning(
                    "[BatchLTXRender] inline JSON parse failed: %s", exc,
                )
                return None, None

        # Layer 2: filesystem path -- could be .mp4 or _ledger.json.
        try:
            p = Path(s)
        except Exception:  # noqa: BLE001
            return None, None

        # Layer 2a: .mp4 path -> swap suffix to _ledger.json
        # (SignalLostVideo / EpisodeAssembler convention). Two-tier
        # resolution per round-robin verdict:
        #   Tier 1 -- exact stem match.
        #   Tier 2 -- collapsed-underscore variant (BUG-LOCAL-118 carry-over).
        # If either candidate file exists but fails to load, _read
        # raises -- intentional fail-loud, no fuzzy fallback.
        if p.suffix.lower() == ".mp4":
            audio_dir = p.parent
            stem = p.stem

            ledger_p = audio_dir / f"{stem}_ledger.json"
            if ledger_p.exists():
                return _read(ledger_p), ledger_p

            collapsed = stem
            while "__" in collapsed:
                collapsed = collapsed.replace("__", "_")
            if collapsed != stem:
                cand = audio_dir / f"{collapsed}_ledger.json"
                if cand.exists():
                    log.warning(
                        "[BatchLTXRender] BUG-LOCAL-118 underscore-mismatch "
                        "fallback: .mp4 stem %r had double underscores; "
                        "loaded matching ledger %r instead.",
                        stem, cand.name,
                    )
                    return _read(cand), cand

            log.warning(
                "[BatchLTXRender] derived ledger from .mp4 not found: "
                "%s (tried Tier 1 exact + Tier 2 collapsed-underscore; "
                "Tier 3 fuzzy scan removed by 2026-05-02 round-robin)",
                ledger_p,
            )
            return None, None

        # Layer 2b: plain _ledger.json path.
        if not p.is_file():
            return None, None
        return _read(p), p


__all__ = ["BatchLTXRender"]
