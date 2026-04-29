"""
batch_humo_render.py  --  OTR_BatchHumoRender ComfyUI node
==========================================================

Render N HuMo lip-sync clips in lockstep from a Production Ledger,
in a single graph run, sharing one HuMo + Lora + CLIP + VAE +
Whisper load.

This is the in-graph counterpart to ``scripts/render_humo_batch.py``
(the subprocess orchestrator). The CLI script remains for ad-hoc
smoke tests; production runs use this node so progress is visible
in ComfyUI's UI and the FULL workflow is a single self-contained
JSON with no hidden subprocess.

Pattern mirrors OTR_BatchFluxRender:
  - take pre-loaded MODEL/CLIP/VAE/AUDIO_ENCODER as INPUT_TYPES so
    Comfy's model-management owns the lifecycle
  - load core ComfyUI nodes lazily at execute() time
  - loop N ledger lines, calling node objects directly (NOT via
    /prompt HTTP API)
  - each line: slice audio tensor, load portrait from disk, encode
    text + audio + ref_image, run WanHuMoImageToVideo + KSampler +
    VAEDecode + CreateVideo + SaveVideo, write ``<line_id>.mp4``
  - return clips_dir + count + report

Per CLAUDE.md C5: HuMo runs at fp8_e4m3fn weights staged via
ComfyUI's dynamic VRAM loading. UnloadAll wires after this node to
free HuMo before VideoComposite reads the proc gen + clips.

Ported logic from render_humo_batch.py:
  - HUMO_FPS / HUMO_MIN_FRAMES / HUMO_MAX_FRAMES constants
  - humo_length_for_dur(): 4n+1 frame snap with floor + ceiling
  - find_portrait_for_speaker(): cast-position fallback
  - find_composite_for_shot_speaker(): per-(shot,speaker) pass3
    composite picker
  - cid_to_name speaker resolution (BUG-LOCAL-074)
  - Save filename = ``<line_id>.mp4`` so OTR_VideoComposite (or
    render_episode_concat.py) finds clips by ledger line_id
"""
from __future__ import annotations

import json
import logging
import re
import sys as _sys
import time
from pathlib import Path
from typing import Any

# Path-helper bootstrap: ``_otr_paths`` is a sibling module. In
# production ComfyUI imports this file as ``nodes.batch_humo_render``
# so a relative import works. In tests + ad-hoc scripts the module is
# loaded directly via ``importlib.util.spec_from_file_location`` with
# no parent package, breaking relative imports. The sys.path prepend
# below makes ``_otr_paths`` reachable as a top-level module in both
# contexts; the import is idempotent across multiple node loads.
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))
from _otr_paths import (  # noqa: E402
    comfy_output_dir,
    otr_audio_dir,
    otr_legacy_audio_dir,
    otr_videos_dir,
)

log = logging.getLogger("OTR.batch_humo_render")


# ---------------------------------------------------------------------------
# HuMo constants (ported from render_humo_batch.py)
# ---------------------------------------------------------------------------

HUMO_FPS = 25
HUMO_MIN_FRAMES = 33   # smaller frame counts have hung this hardware
HUMO_MAX_FRAMES = 177  # last empirically verified value on RTX 5080 Laptop 16GB

# ByteDance Chinese negative prompt -- empirically the best HuMo neg
# on the official template.
_CHINESE_NEGATIVE = (
    "色调艳丽，过曝，静态，"
    "细节模糊不清，字幕，风格"
    "，作品，画作，画面，静止"
    "，整体发灰，最差质量，低"
    "质量，JPEG压缩残留，丑陋的"
    "，残缺的，多余的手指，画"
    "得不好的手部，画得不好的"
    "脸部，畸形的，毁容的，形"
    "态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背"
    "景，三条腿，背景人很多，"
    "倒着走"
)

_DEFAULT_POS_SUFFIX = (
    "dimly lit interior, ambient cinematic lighting, "
    "35mm film grain, shallow depth of field"
)

# Slug rule must stay in lockstep with render_flux_batch.slugify so
# pass3 composite filenames written by FLUX get found by HuMo.
_SLUG_RE_CONSUME = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def humo_length_for_dur(dur_s: float, *, fps: int = HUMO_FPS) -> int:
    """Pick the smallest valid HuMo `length` >= round(dur_s * fps).

    Wan 2.1 VAE temporal compression requires length = 4n + 1.
    Floored at HUMO_MIN_FRAMES. Capped at HUMO_MAX_FRAMES.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 3) // 4
    frames = 4 * n + 1
    if frames < HUMO_MIN_FRAMES:
        frames = HUMO_MIN_FRAMES
    if frames > HUMO_MAX_FRAMES:
        frames = HUMO_MAX_FRAMES
    return frames


def _composite_slug(s: str, limit: int = 40) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE_CONSUME.sub("_", s).strip("_")
    return (s[:limit] or "unnamed")


def _find_portrait(
    speaker: str,
    cast: list[dict],
    portraits_dir: Path,
) -> Path | None:
    """Find a portrait image for a speaker.

    Strategy (extends render_humo_batch.find_portrait_for_speaker
    with a final FLUX env-still fallback so the workflow runs even
    when no per-cast PASS1 portraits have been pre-rendered):

      1. cast[].portrait_path if populated and exists
      2. otr_humo_pass1_portrait_*.png by index in cast list
      3. any otr_humo_pass1_portrait_*.png
      4. **NEW**: full_env_*.png by index in cast list (uses FLUX
         environment stills as face stand-ins — visually wrong per
         line but the workflow runs end-to-end without a separate
         portrait-render pass; BUG-LOCAL-078 stopgap)
      5. any full_env_*.png
      6. None

    BUG-LOCAL-087 ordering: every glob result is sorted by mtime
    descending (newest first). Without this, a freshly-rendered
    full_env_00059_.png would lose to a days-old full_env_00001_.png
    under default alphabetic sort -- meaning HuMo would always pick
    the OLDEST visual reference even when FLUX just rendered fresh
    per-episode stills (BUG-086). Sorting by mtime ensures the
    newest stills line up with cast position 0 first.
    """
    speaker_norm = (speaker or "").upper().strip()

    # 1. Direct path from ledger cast entry
    for c in cast:
        if (c.get("name") or "").upper().strip() == speaker_norm:
            p = c.get("portrait_path")
            if p and Path(p).exists():
                return Path(p)

    cast_names = [(c.get("name") or "").upper().strip() for c in cast]

    # 2. PASS1 humo portraits indexed by cast position (canonical naming)
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_humo_pass1_portrait_*.png")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[idx % len(candidates)]

    # 3. Any HuMo portrait (no cast match) -- newest first
    candidates = sorted(
        portraits_dir.glob("otr_humo_pass1_portrait_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # 4. FLUX env-stills indexed by cast position (BUG-078 stopgap).
    #    These are typically full_env_NNNNN_.png from BatchFluxRender's
    #    environment-token output. Visually wrong per-line (each cast
    #    member maps to a random env still) but produces a runnable
    #    end-to-end pipeline. Replace with proper PASS1 portraits when
    #    a portrait render path is wired into the workflow.
    #    BUG-LOCAL-087: sort by mtime descending so the newest stills
    #    (this episode's FLUX output) come first -- cast index 0 maps
    #    to the freshest still.
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/full_env_*.png"))
            + list(portraits_dir.glob("otr_stills/full_env_*.png")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[idx % len(candidates)]

    # 5. Any FLUX env still (last resort) -- newest first
    candidates = sorted(
        list(portraits_dir.glob("otr/stills/full_env_*.png"))
        + list(portraits_dir.glob("otr_stills/full_env_*.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return None


def _find_composite(
    shot_id: str | None,
    speaker: str | None,
    portraits_dir: Path,
) -> Path | None:
    """Per-(shot, speaker) FLUX pass3 composite picker."""
    if not shot_id or not speaker:
        return None
    shot_slug = _composite_slug(shot_id, limit=24)
    speaker_slug = _composite_slug(speaker, limit=40)
    new_pat = f"otr/stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    mid_pat = f"otr_stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    legacy_pat = f"otr_humo_pass3_{shot_slug}_{speaker_slug}_*.png"
    candidates = sorted(
        list(portraits_dir.glob(new_pat))
        + list(portraits_dir.glob(mid_pat))
        + list(portraits_dir.glob(legacy_pat)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _png_to_tensor(png_path: Path):
    """Load a PNG file as a ComfyUI IMAGE tensor [1, H, W, C] in
    [0, 1] float32. Bypasses ComfyUI's LoadImage which requires the
    file to be in input/ -- we read directly from the otr/ tree."""
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, C]
    return tensor


def _slice_audio_tensor(audio_dict: dict, start_s: float, dur_s: float) -> dict:
    """Slice an AUDIO dict by start_s and dur_s. The AUDIO type in
    ComfyUI is ``{"waveform": Tensor[B, C, samples], "sample_rate":
    int}``. Returns a new dict with the sliced waveform; sample_rate
    unchanged."""
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    start_sample = max(0, int(start_s * sr))
    end_sample = max(start_sample + 1, int((start_s + dur_s) * sr))
    if waveform.dim() == 2:
        # Some callers pass [C, samples] instead of [B, C, samples]
        sliced = waveform[..., start_sample:end_sample]
    else:
        sliced = waveform[..., start_sample:end_sample]
    return {"waveform": sliced.contiguous(), "sample_rate": sr}


def _pad_audio_silence_lead(audio_dict: dict, pad_samples: int) -> dict:
    """BUG-LOCAL-102: prepend ``pad_samples`` of digital silence
    (zero samples) to a sliced AUDIO dict. Used to give HuMo's audio
    cross-attention some leading silence so the model burns its
    intrinsic ~3-6-frame motion-onset freeze on silence rather than
    on the first dialogue phoneme. ChatGPT + Gemini round-robin
    2026-04-28 PM both confirm zeros are sufficient (Whisper / HuMo
    audio encoders treat zeros, low-noise, and room tone identically
    for the purpose of "no speech here"). Returns a new audio dict;
    sample_rate is preserved.
    """
    if pad_samples <= 0:
        return audio_dict
    import torch  # type: ignore  # lazy: this module is imported headless by tests
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    pad_shape = list(waveform.shape)
    pad_shape[-1] = int(pad_samples)
    silence = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)
    padded = torch.cat([silence, waveform], dim=-1).contiguous()
    return {"waveform": padded, "sample_rate": sr}


def _trim_audio_lead(audio_dict: dict, pad_samples: int) -> dict:
    """Inverse of ``_pad_audio_silence_lead``: drop the leading
    ``pad_samples`` from an AUDIO dict. Used at HuMo clip save time
    so the on-disk mp4 contains only the dialogue audio (no leading
    silence), keeping VideoComposite placement math unchanged.
    """
    if pad_samples <= 0:
        return audio_dict
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    pad_samples = int(pad_samples)
    if waveform.shape[-1] <= pad_samples:
        # Nothing left after trim -- caller should never hit this with
        # a positive pad_ms; degrade by returning a minimal 1-sample slice.
        trimmed = waveform[..., :1]
    else:
        trimmed = waveform[..., pad_samples:]
    return {"waveform": trimmed.contiguous(), "sample_rate": sr}


def _resolve_cast_stills_from_ledger(
    cast: list[dict],
    portraits_dir: Path,
    ledger_mtime: float | None,
    *,
    slack_seconds: float = 60.0,
) -> tuple[dict[str, Path], list[Path]]:
    """BUG-LOCAL-088: build a cast_char_id -> still_path map using
    the ledger's mtime as the freshness floor.

    Strategy:
      1. Glob `output/otr/stills/full_env_*.png` (both nested layouts).
      2. Filter to mtime >= (ledger_mtime - slack_seconds). If no
         ledger mtime is available (inline JSON ledger), fall back
         to "last 30 minutes" so we still favour fresh stills.
      3. Sort surviving candidates by mtime descending (newest first).
      4. Walk the cast list in order; assign candidates[i] to
         cast[i]['char_id']. If candidates run out, leave that
         char_id unmapped (caller falls back to the existing
         _find_portrait/_find_composite tiers).

    Returns ``(char_id_to_path, fresh_candidates_sorted_newest_first)``.
    The caller logs which assignment came from "fresh" vs "stale" so
    the runtime log shows the freshness state per line.
    """
    import time as _time

    fresh_floor: float
    if ledger_mtime is not None:
        fresh_floor = ledger_mtime - slack_seconds
    else:
        # No on-disk ledger source; conservatively accept anything
        # written in the last 30 minutes.
        fresh_floor = _time.time() - 30 * 60

    candidates_all: list[Path] = []
    for pattern in ("otr/stills/full_env_*.png", "otr_stills/full_env_*.png"):
        for p in portraits_dir.glob(pattern):
            try:
                if p.stat().st_mtime >= fresh_floor:
                    candidates_all.append(p)
            except OSError:
                continue

    candidates_all.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    char_id_to_path: dict[str, Path] = {}
    for i, c in enumerate(cast):
        char_id = (c.get("char_id") or "").strip()
        if not char_id:
            continue
        if i < len(candidates_all):
            char_id_to_path[char_id] = candidates_all[i]

    return char_id_to_path, candidates_all


def _save_clip_via_ffmpeg(
    *,
    images,
    audio_dict: dict,
    out_path: Path,
    fps: int = 25,
) -> Path:
    """Write a HuMo clip .mp4 directly via ffmpeg, bypassing
    CreateVideo + SaveVideo.

    BUG-LOCAL-083 workaround: ComfyUI v0.20.1's CreateVideo +
    SaveVideo are V3-style nodes that read ``cls.hidden`` (set by
    the executor with extra_pnginfo + prompt metadata) at execute
    time. Direct-calling those nodes from inside another node's
    execute() leaves ``cls.hidden = None`` and they crash at::

        File "comfy_extras/nodes_video.py", line 100, in execute
            if cls.hidden.extra_pnginfo is not None:
        AttributeError: 'NoneType' object has no attribute
        'extra_pnginfo'

    HuMo renders the frames + audio segment fine, but the save step
    fails -- 12 clips x 9 minutes each = 108 minutes of GPU work
    discarded. Same family as BUG-LOCAL-074 (executor state assumed
    by node code).

    Fix: write the decoded image tensor as a PNG sequence + the
    audio waveform as raw f32 PCM into a temp dir, then mux into
    the final mp4 with ffmpeg. Same on-disk artifact (mp4 with
    embedded audio at canonical ``<line_id>.mp4`` filename), no
    executor coupling.

    Args:
        images: IMAGE tensor ``[N_frames, H, W, C]`` float in [0, 1]
            (or 3-D ``[H, W, C]`` for a single frame).
        audio_dict: ``{"waveform": Tensor[..., samples],
                       "sample_rate": int}`` (ComfyUI AUDIO type).
        out_path: final mp4 path, e.g.
            ``output/otr/videos/<ep_id>/<line_id>.mp4``.
        fps: frame rate (HuMo = 25).

    Returns:
        ``out_path`` on success.

    Raises:
        RuntimeError on ffmpeg failure.
    """
    import shutil
    import subprocess
    import tempfile
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Image tensor -> uint8 numpy [N, H, W, C] ----
    img = images
    if isinstance(img, torch.Tensor):
        if img.device.type != "cpu":
            img = img.detach().to("cpu", copy=False)
        arr = img.numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise RuntimeError(
            f"_save_clip_via_ffmpeg: unexpected image shape {arr.shape}"
        )
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    n_frames = int(arr.shape[0])
    if n_frames < 1:
        raise RuntimeError("_save_clip_via_ffmpeg: zero frames decoded")

    # ---- 2. Audio waveform -> raw f32 interleaved ----
    waveform = audio_dict.get("waveform") if audio_dict else None
    sr = int(audio_dict.get("sample_rate", 16000)) if audio_dict else 16000
    have_audio = waveform is not None
    if have_audio:
        if isinstance(waveform, torch.Tensor):
            if waveform.device.type != "cpu":
                waveform = waveform.detach().to("cpu", copy=False)
            wf = waveform.numpy()
        else:
            wf = np.asarray(waveform)
        # Strip leading batch dims down to [C, samples]
        while wf.ndim > 2:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[None, :]
        # ffmpeg f32le wants samples-major interleaved for multi-ch
        wf_il = np.ascontiguousarray(wf.T.astype(np.float32, copy=False))
        n_channels = int(wf.shape[0])
        if wf_il.size == 0:
            have_audio = False

    tmp = Path(tempfile.mkdtemp(prefix="otr_humo_clip_"))
    try:
        # ---- 3. Write PNG frame sequence ----
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        for i in range(n_frames):
            Image.fromarray(arr[i]).save(frames_dir / f"f_{i:05d}.png")

        # ---- 4. Write audio as raw f32le ----
        audio_raw = tmp / "audio.f32"
        if have_audio:
            wf_il.tofile(audio_raw)

        # ---- 5. ffmpeg mux ----
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-framerate", str(int(fps)),
            "-i", str(frames_dir / "f_%05d.png"),
        ]
        if have_audio:
            cmd += [
                "-f", "f32le",
                "-ar", str(sr),
                "-ac", str(n_channels),
                "-i", str(audio_raw),
            ]
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
        ]
        if have_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[:500]}"
            )
        return out_path
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def _build_pos_prompt(speaker: str, ln: dict, cast: list[dict]) -> str:
    """Build a HuMo positive prompt from the ledger line + cast desc."""
    speaker_desc = ""
    for c in cast:
        if (c.get("name") or "").upper().strip() == (speaker or "").upper().strip():
            speaker_desc = (c.get("description") or "").strip()
            break
    if not speaker_desc:
        speaker_desc = (
            f"A {speaker.lower()} character speaks calmly with subtle facial "
            f"expressions"
        ) if speaker else "A character speaks calmly with subtle facial expressions"
    return f"{speaker_desc}, {_DEFAULT_POS_SUFFIX}"


def _lazy_humo_nodes() -> dict[str, Any]:
    """Resolve the ComfyUI / extension node classes BatchHumoRender
    needs. Lazy import so the module loads even if Comfy isn't fully
    initialized (e.g., during static analysis or pytest)."""
    refs: dict[str, Any] = {}
    try:
        import nodes  # type: ignore
        refs["CLIPTextEncode"] = getattr(nodes, "CLIPTextEncode", None)
        refs["KSampler"] = getattr(nodes, "KSampler", None)
        refs["VAEDecode"] = getattr(nodes, "VAEDecode", None)
        node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    except Exception:
        node_mappings = {}

    # Wan 2.1 / HuMo extension nodes (community / Kijai pack)
    refs["WanHuMoImageToVideo"] = node_mappings.get("WanHuMoImageToVideo")
    refs["AudioEncoderEncode"] = node_mappings.get("AudioEncoderEncode")
    refs["CreateVideo"] = node_mappings.get("CreateVideo")
    refs["SaveVideo"] = node_mappings.get("SaveVideo")
    return refs


def _call(node_instance, **kwargs):
    """Call a ComfyUI node by its declared FUNCTION method name with
    keyword arguments matching INPUT_TYPES.

    BUG-LOCAL-080 fix: ComfyUI node method names are NOT consistent.
    AudioEncoderEncode does not have ``.encode()`` -- the actual
    method is named by the class's ``FUNCTION`` attribute. Hardcoding
    method names (``audio_enc.encode(...)``) blows up at runtime.

    Using ``getattr(instance, instance.FUNCTION)`` resolves whatever
    name the node author declared. Passing keyword arguments avoids
    positional-order mismatches (CLIPTextEncode expects ``clip``
    first in some signatures, ``text`` first in others).

    Returns whatever the underlying method returns (typically a
    tuple of outputs).
    """
    fn_name = getattr(node_instance, "FUNCTION", "execute")
    fn = getattr(node_instance, fn_name, None)
    if fn is None:
        raise AttributeError(
            f"node {type(node_instance).__name__} has no method "
            f"{fn_name!r} (declared by FUNCTION attribute)"
        )
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BatchHumoRender:
    """Render N HuMo lip-sync clips in one graph execution.

    OUTPUT_NODE = True (BUG-LOCAL-077 lesson): this node has
    side-effects (writes per-line .mp4 files to output/otr/videos/
    <ep_id>/) AND its output STRINGs are diagnostic. Without
    OUTPUT_NODE the executor would prune it when downstream consumers
    don't fully chain.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("clips_dir", "clip_count", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers  # type: ignore
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["uni_pc"]
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "HuMo model with lightx2v Lora applied (post ModelSamplingSD3).",
                }),
                "clip": ("CLIP", {
                    "tooltip": "umt5_xxl text encoder loaded via CLIPLoader.",
                }),
                "vae": ("VAE", {
                    "tooltip": "wan_2.1 VAE loaded via VAELoader.",
                }),
                "audio_encoder": ("AUDIO_ENCODER", {
                    "tooltip": "Whisper Large v3 fp16 loaded via AudioEncoderLoader.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Full episode audio from EpisodeAssembler.",
                }),
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Production Ledger JSON. Either the JSON string "
                        "itself or a path to *_ledger.json on disk."
                    ),
                }),
                "portraits_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Directory holding PASS1 character portraits "
                        "(otr_humo_pass1_portrait_*.png) and optional "
                        "PASS3 (shot,speaker) composites. Empty -> "
                        "auto-resolve to output/otr/portraits/<ep_id>/."
                    ),
                }),
                "clip_length": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.32,
                    "max": 7.08,
                    "step": 0.04,
                    "tooltip": (
                        "Per-clip duration in seconds. Default 7.0 -> "
                        "175 frames -> 177 (Wan 2.1 4n+1 = 7.08s)."
                    ),
                }),
                "max_clips": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "0 = render every line; smoke-test caps with positive int.",
                }),
                "seed": ("INT", {"default": 7, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "uni_pc"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "width": ("INT", {"default": 480, "min": 256, "max": 1280, "step": 8}),
                "height": ("INT", {"default": 832, "min": 256, "max": 1280, "step": 8}),
            },
            "optional": {
                # BUG-LOCAL-086: dependency-only gate input. Wire any
                # IMAGE output from the FLUX-render branch (typically
                # OTR_UnloadAll's IMAGE passthrough, which sits between
                # OTR_BatchFluxRender and SaveImage) to this socket so
                # ComfyUI's topological scheduler runs FLUX env stills
                # FIRST, then UnloadAll frees FLUX VRAM, THEN this
                # batch-HuMo loop starts with fresh per-episode
                # `full_env_*.png` stand-ins on disk. Without the wire,
                # HuMo would run before FLUX (both being independent
                # graph branches downstream of SignalLostVideo) and
                # `_find_portrait` would fall back to whatever stale
                # env stills happened to be on disk from prior runs.
                # The value is intentionally ignored at runtime; the
                # wire is purely a graph-edge for ordering.
                "flux_done_gate": ("IMAGE", {
                    "tooltip": (
                        "Optional FLUX->HuMo dependency gate. Wire "
                        "OTR_UnloadAll's IMAGE output here to force "
                        "FLUX env stills to render BEFORE this HuMo "
                        "loop. Value is ignored; only the dependency "
                        "edge matters."
                    ),
                }),
                "humo_warmup_pad_ms": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "tooltip": (
                        "BUG-LOCAL-102: leading silence (in ms) padded "
                        "onto each line's audio before HuMo's audio "
                        "cross-attention. Burns HuMo's intrinsic ~3-6 "
                        "frame motion-onset freeze on silence rather "
                        "than on the first dialogue word, eliminating "
                        "the constant audio-leads-lips lag the listener "
                        "hears in the rendered episode. The pad is "
                        "trimmed back off the on-disk clip so timeline "
                        "placement math is unchanged. Set to 0 to "
                        "disable (reverts to pre-BUG-102 behavior)."
                    ),
                }),
            },
        }

    def execute(
        self,
        model, clip, vae, audio_encoder, audio,
        ledger_json: str,
        portraits_dir: str,
        clip_length: float,
        max_clips: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        width: int,
        height: int,
        flux_done_gate=None,  # BUG-LOCAL-086: dependency gate, value ignored
        humo_warmup_pad_ms: int = 200,  # BUG-LOCAL-102: see INPUT_TYPES tooltip
    ):
        t_start = time.time()
        # BUG-LOCAL-086: flux_done_gate is intentionally not consumed.
        # Its sole purpose is to insert a graph edge so ComfyUI's
        # topological scheduler runs the FLUX -> UnloadAll branch
        # BEFORE this node starts.
        del flux_done_gate

        # BUG-LOCAL-102: clamp the pad to sane bounds. 0 disables.
        warmup_pad_ms = max(0, min(500, int(humo_warmup_pad_ms)))

        # ---- 1. Parse ledger ----
        ledger, ledger_path = self._load_ledger_with_path(ledger_json)
        cast = ledger.get("cast") or []
        lines = ledger.get("lines") or []
        episode_id = ledger.get("episode_id", "episode")
        cid_to_name = {
            (c.get("char_id") or ""): (c.get("name") or "")
            for c in cast if c.get("char_id")
        }
        ledger_mtime: float | None = None
        if ledger_path is not None:
            try:
                ledger_mtime = ledger_path.stat().st_mtime
            except OSError:
                ledger_mtime = None
        log.info(
            "[BatchHumoRender] episode_id=%s cast=%d lines=%d ledger=%s",
            episode_id, len(cast), len(lines),
            (ledger_path.name if ledger_path is not None else "<inline>"),
        )

        # ---- 2. Resolve portraits dir ----
        # Default = ComfyUI output root so _find_portrait's globs hit
        # both `otr/portraits/<ep_id>/otr_humo_pass1_portrait_*.png`
        # AND `otr/stills/full_env_*.png` AND `otr/stills/pass1_*.png`.
        # User-supplied input wins if non-empty.
        if not portraits_dir or not portraits_dir.strip():
            portraits_dir_path = comfy_output_dir()
        else:
            portraits_dir_path = Path(portraits_dir)
        log.info("[BatchHumoRender] portraits_dir=%s", portraits_dir_path)

        # ---- 3. Resolve clips output_dir (canonical OTR tree) ----
        # output_dir = ComfyUI/output/otr/videos/<episode_id>/
        clips_dir_path = otr_videos_dir(episode_id)
        clips_dir_path.mkdir(parents=True, exist_ok=True)

        # ---- 4. Lazy-load Comfy nodes ----
        # BUG-LOCAL-083: CreateVideo + SaveVideo dropped from the
        # required list. Their V3 ``cls.hidden`` coupling crashes
        # when called outside the executor; we mux clip mp4s with
        # ``_save_clip_via_ffmpeg`` instead.
        refs = _lazy_humo_nodes()
        missing = [k for k in (
            "CLIPTextEncode", "KSampler", "VAEDecode",
            "WanHuMoImageToVideo", "AudioEncoderEncode",
        ) if refs.get(k) is None]
        if missing:
            raise RuntimeError(
                f"BatchHumoRender: missing required ComfyUI node classes: {missing}"
            )

        text_enc = refs["CLIPTextEncode"]()
        sampler = refs["KSampler"]()
        vae_decoder = refs["VAEDecode"]()
        humo_node = refs["WanHuMoImageToVideo"]()
        audio_enc_node = refs["AudioEncoderEncode"]()

        # ---- 5. Pin model on GPU ----
        try:
            import comfy.model_management as mm  # type: ignore
            try:
                mm.load_models_gpu([model], force_full_load=True)
            except TypeError:
                mm.load_models_gpu([model])
            log.info("[BatchHumoRender] pinned MODEL via load_models_gpu")
        except Exception as exc:
            log.debug("[BatchHumoRender] pre-pin skipped: %s", exc)

        # ---- 5.5. BUG-LOCAL-088: ledger-driven cast->still binding ----
        # Build a per-cast-member still resolver using the LEDGER as
        # the source of truth for "this episode". After BUG-086 the
        # FLUX env stills are guaranteed to land before HuMo starts;
        # after BUG-087 they sort newest-first; this step adds the
        # explicit freshness floor so STALE stills from prior runs
        # are never selected even when fresh-count < cast-count.
        cast_still_map, fresh_candidates = _resolve_cast_stills_from_ledger(
            cast=cast,
            portraits_dir=portraits_dir_path,
            ledger_mtime=ledger_mtime,
        )
        log.info(
            "[BatchHumoRender] cast-still binding: %d/%d cast members "
            "matched to fresh stills (%d fresh stills available)",
            len(cast_still_map), len(cast), len(fresh_candidates),
        )
        if cast and not fresh_candidates:
            log.warning(
                "[BatchHumoRender] no fresh stills found above ledger "
                "mtime cutoff -- portrait selection will fall back to "
                "global glob (stale stills possible)"
            )
        elif len(cast_still_map) < sum(1 for c in cast if c.get("line_count", 0) > 0):
            log.warning(
                "[BatchHumoRender] only %d fresh stills for %d voiced "
                "cast members; trailing cast positions will fall back "
                "to legacy _find_portrait tiers",
                len(fresh_candidates),
                sum(1 for c in cast if c.get("line_count", 0) > 0),
            )
        for c in cast:
            cid = c.get("char_id") or ""
            cname = c.get("name") or ""
            if cid in cast_still_map:
                log.info(
                    "[BatchHumoRender]   cast %s (%s) -> %s (FRESH)",
                    cid, cname, cast_still_map[cid].name,
                )

        # ---- 6. Build per-line plan (no GPU work yet) ----
        if max_clips and max_clips > 0:
            lines = lines[:max_clips]

        report_lines: list[str] = [
            f"BatchHumoRender: episode={episode_id} target_lines={len(lines)}",
        ]
        if max_clips and max_clips > 0:
            report_lines.append(f"  capped to {max_clips} clips")

        # ---- 5.7. BUG-LOCAL-094: distribute clip timings across full episode ----
        # When upstream nodes (SceneSequencer / Bark / EpisodeAssembler)
        # don't write `lines[i].start_s` and `lines[i].dur_s` back to
        # the ledger -- which is the current state -- the legacy
        # fallback was `start_s = idx * clip_length` (idx * 7s) for
        # every line. That clumped all HuMo clips into the first
        # `N * 7s` of the episode, leaving the back half as
        # procgen-only. Worse: with episodes longer than 7s/line
        # average, the audio slice for each clip drifted progressively
        # out of sync with the actual Bark TTS playback in the
        # composite.
        #
        # Fix (heuristic, until proper Bark write-back lands):
        # estimate per-line timings from the line's `word_count`
        # share of the total episode duration. This:
        #   * spreads HuMo clips evenly across the whole timeline
        #   * scales the slice positions roughly with where each
        #     line actually plays in the assembled audio
        #   * still respects the `clip_length` ceiling for HuMo
        #     render budget (BUG-082 VRAM cap)
        # The estimate is overridden line-by-line whenever the
        # ledger DOES carry an explicit `start_s` / `dur_s` (so a
        # future SceneSequencer write-back upgrade replaces this
        # automatically with no further code changes here).
        estimated_timings: dict[int, tuple[float, float]] = {}
        try:
            total_episode_dur = ledger.get("total_episode_dur_s")
            if not total_episode_dur:
                # Fallback: estimate from the audio tensor itself.
                wf = audio.get("waveform") if isinstance(audio, dict) else None
                sr = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
                if wf is not None and sr > 0:
                    try:
                        total_episode_dur = float(wf.shape[-1]) / float(sr)
                    except Exception:
                        total_episode_dur = None
            if total_episode_dur and total_episode_dur > 0:
                voiced = [ln for ln in lines if (ln.get("char_id") or "").strip()]
                # word-share denominator: prefer word_count, fall back
                # to char_count (so a 0-word ledger does not divide by 0)
                total_weight = sum(
                    max(int(ln.get("word_count") or 0), 1)
                    for ln in voiced
                ) or len(voiced) or 1
                cursor = 0.0
                # Map indices in `lines` (not just voiced) so the
                # per-line loop's `idx` lookup still works
                for idx, ln in enumerate(lines):
                    if not (ln.get("char_id") or "").strip():
                        # non-voiced line (announcer-only ledger
                        # entry, or filtered-out parse artefact);
                        # zero-dur slice that won't render.
                        estimated_timings[idx] = (cursor, 0.0)
                        continue
                    weight = max(int(ln.get("word_count") or 0), 1)
                    proportional = float(total_episode_dur) * (weight / total_weight)
                    estimated_timings[idx] = (cursor, proportional)
                    cursor += proportional
                log.info(
                    "[BatchHumoRender] BUG-094 estimated %d line timings "
                    "across %.1fs episode (avg %.2fs/line)",
                    len(estimated_timings), total_episode_dur,
                    total_episode_dur / max(len(voiced), 1),
                )
        except Exception as exc:
            log.warning("[BatchHumoRender] timing estimation failed: %s", exc)
            estimated_timings = {}

        # Per-line plan: (line_id, speaker, start_s, dur_s, humo_length,
        #                 pos_text, ref_image_tensor, audio_dict)
        plan: list[dict] = []
        for idx, ln in enumerate(lines):
            line_id = str(ln.get("line_id") or f"l{idx + 1:03d}")
            speaker = (
                ln.get("speaker")
                or cid_to_name.get(ln.get("char_id") or "", "")
                or ""
            ).strip()
            char_id = (ln.get("char_id") or "").strip()

            # Per-line timing priority (BUG-094):
            #   1. ledger.lines[i].start_s/dur_s (real values from a
            #      future SceneSequencer write-back -- best)
            #   2. estimated_timings[idx] (this run's word-share
            #      heuristic computed in section 5.7 above -- spreads
            #      clips across the full episode duration)
            #   3. legacy fallback: idx * clip_length (clumped at
            #      front of timeline; only triggers when neither the
            #      ledger nor the heuristic produced timings, e.g.
            #      total_episode_dur_s missing AND audio tensor
            #      unreadable)
            start_s = ln.get("start_s")
            dur_s = ln.get("dur_s")
            if (start_s is None or start_s == "") and idx in estimated_timings:
                start_s = estimated_timings[idx][0]
            if (dur_s is None or dur_s == "" or dur_s <= 0) and idx in estimated_timings:
                dur_s = estimated_timings[idx][1] or clip_length
            if start_s is None or start_s == "":
                start_s = idx * clip_length
            if dur_s is None or dur_s == "" or dur_s <= 0:
                dur_s = clip_length
            start_s = float(start_s)
            dur_s = float(dur_s)

            # Cap to clip_length
            dur_s = min(dur_s, float(clip_length))
            # BUG-LOCAL-102: extend humo_length to cover the leading
            # silence pad so HuMo has frames to render BOTH the
            # warm-up freeze AND the dialogue. The pad frames are
            # trimmed off the on-disk clip below so timeline math
            # stays in terms of dur_s.
            pad_s = warmup_pad_ms / 1000.0
            humo_length = humo_length_for_dur(dur_s + pad_s)

            # Resolve portrait. BUG-LOCAL-088 priority:
            #   1. ledger-driven cast_still_map (this-episode FLUX)
            #   2. _find_composite (per-shot pass3 composite)
            #   3. _find_portrait (legacy tiers; mtime-sorted post BUG-087)
            shot_id = ln.get("shot_id")
            ref_png: Path | None = None
            ref_source = "none"
            if char_id and char_id in cast_still_map:
                ref_png = cast_still_map[char_id]
                ref_source = "ledger-cast-fresh"
            if not ref_png:
                ref_png = _find_composite(shot_id, speaker, portraits_dir_path)
                if ref_png:
                    ref_source = "find_composite"
            if not ref_png:
                ref_png = _find_portrait(speaker, cast, portraits_dir_path)
                if ref_png:
                    ref_source = "find_portrait"
            if not ref_png:
                report_lines.append(f"  {line_id}: SKIP no portrait")
                log.warning("[BatchHumoRender] line %s speaker=%r: no portrait",
                            line_id, speaker)
                continue

            try:
                ref_image = _png_to_tensor(ref_png)
            except Exception as exc:
                report_lines.append(f"  {line_id}: SKIP portrait load failed: {exc}")
                continue

            line_audio = _slice_audio_tensor(audio, start_s, dur_s)
            # BUG-LOCAL-102: pad the line's audio with leading silence
            # so HuMo's first few frames (motion-onset freeze) happen
            # during silence, not during the first phoneme. The pad is
            # at the slice's own sample rate.
            pad_samples = (
                int(warmup_pad_ms * int(line_audio.get("sample_rate", 0)) / 1000)
                if warmup_pad_ms > 0 else 0
            )
            line_audio = _pad_audio_silence_lead(line_audio, pad_samples)
            pos_text = _build_pos_prompt(speaker, ln, cast)

            # BUG-LOCAL-088: log which still each line consumed + source
            # tier + freshness so post-mortem can verify ledger
            # accuracy without re-globbing the disk.
            try:
                ref_mtime = ref_png.stat().st_mtime
                ref_age = (time.time() - ref_mtime) if ref_mtime else None
                age_str = (
                    f"{ref_age:.0f}s ago" if ref_age is not None and ref_age < 3600
                    else (f"{ref_age/60:.0f}min ago" if ref_age is not None and ref_age < 86400
                          else (f"{ref_age/3600:.0f}h ago" if ref_age is not None
                                else "unknown"))
                )
            except OSError:
                age_str = "unknown"
            log.info(
                "[BatchHumoRender] line %s speaker=%s char_id=%s "
                "ref=%s source=%s age=%s",
                line_id, speaker or "?", char_id or "?",
                ref_png.name, ref_source, age_str,
            )

            plan.append({
                "idx": idx, "line_id": line_id, "speaker": speaker,
                "start_s": start_s, "dur_s": dur_s, "humo_length": humo_length,
                "pos_text": pos_text, "ref_image": ref_image,
                "audio": line_audio, "ref_png_name": ref_png.name,
                "ref_source": ref_source,
                # BUG-LOCAL-102: how many leading samples / frames the
                # save step must trim back off so the on-disk clip is
                # aligned (audio = dialogue from sample 0; video =
                # articulated motion from frame 0, no warm-up freeze).
                "warmup_pad_samples": pad_samples,
                "warmup_pad_frames": (
                    int(warmup_pad_ms * HUMO_FPS / 1000)
                    if warmup_pad_ms > 0 else 0
                ),
            })

        if not plan:
            report_lines.append("FATAL: empty plan -- no lines had portraits")
            return (str(clips_dir_path), 0, "\n".join(report_lines))

        # ---- 7. Phase A: encode all text prompts up front ----
        # Encode the negative once, then every positive in sequence.
        # This avoids reloading WanTEModel between every line during
        # the HuMo render loop (BUG-LOCAL-080 architectural complaint:
        # "audio was trying encode at the same time as humo was
        # running" -- the per-line encode + render mix forced ComfyUI
        # to swap models in and out of VRAM constantly).
        log.info("[BatchHumoRender] Phase A: encoding %d positive + 1 negative text prompts",
                 len(plan))
        try:
            negative = _call(text_enc, clip=clip, text=_CHINESE_NEGATIVE)[0]
        except Exception as exc:
            raise RuntimeError(f"BatchHumoRender: negative encode failed: {exc}")

        for entry in plan:
            try:
                entry["positive"] = _call(text_enc, clip=clip, text=entry["pos_text"])[0]
            except Exception as exc:
                log.warning("[BatchHumoRender] %s: text encode failed: %s",
                            entry["line_id"], exc)
                entry["positive"] = None

        # ---- 8. Phase B: encode all per-line audio up front ----
        log.info("[BatchHumoRender] Phase B: encoding %d audio segments via Whisper",
                 len(plan))
        for entry in plan:
            try:
                entry["audio_emb"] = _call(
                    audio_enc_node,
                    audio_encoder=audio_encoder,
                    audio=entry["audio"],
                )[0]
            except Exception as exc:
                log.warning("[BatchHumoRender] %s: audio encode failed: %s",
                            entry["line_id"], exc)
                entry["audio_emb"] = None

        # ---- 8.5. VRAM cleanup between Phase B and Phase C ----
        # BUG-LOCAL-081: at 30+ lines, Phase B reloaded Whisper for
        # every line (each AudioEncoderEncode call triggered a fresh
        # "WhisperLargeV3 prepared" log). At Phase B end, GPU still
        # holds: Whisper weights (1.2 GB), umt5_xxl text encoder
        # (6.4 GB from Phase A), 30 audio embedding tensors (~10 MB
        # each = 300 MB), 30 positive cond tensors (~50 MB each =
        # 1.5 GB). Total ~9.4 GB pinned before HuMo even starts to
        # load.  Phase C asks for HuMo (16.5 GB staged) on a 16 GB
        # card -- ComfyUI's dynamic VRAM loader thrashes pages
        # perpetually and never converges to forward progress.
        # Symptom: KSampler stuck at "0/6 [?it/s]" for 20+ minutes.
        #
        # Fix is two-step:
        #   1. Move every Phase A/B output tensor to CPU. That
        #      releases GPU pages backing positive/negative cond and
        #      audio embeddings -- Phase C's WanHuMoImageToVideo
        #      moves them back to GPU when it actually needs them.
        #   2. unload_all_models + soft_empty_cache to evict Whisper
        #      and umt5_xxl from GPU and return pages to the CUDA
        #      allocator pool.
        try:
            import torch  # type: ignore
            def _to_cpu(obj):
                """Best-effort move a Comfy CONDITIONING / AUDIO_EMB
                payload to CPU. Conditioning is a list of [tensor,
                meta_dict] pairs; meta_dict can carry pooled_output
                also as a tensor. AUDIO_EMB shape is unknown but
                likely tensor or dict."""
                if isinstance(obj, torch.Tensor):
                    return obj.detach().to("cpu", copy=False) if obj.device.type == "cuda" else obj
                if isinstance(obj, list):
                    return [_to_cpu(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple(_to_cpu(x) for x in obj)
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                return obj
            negative = _to_cpu(negative)
            for entry in plan:
                if entry.get("positive") is not None:
                    entry["positive"] = _to_cpu(entry["positive"])
                if entry.get("audio_emb") is not None:
                    entry["audio_emb"] = _to_cpu(entry["audio_emb"])
            log.info("[BatchHumoRender] Phase A/B tensors moved to CPU")
        except Exception as exc:
            log.warning("[BatchHumoRender] CPU offload failed: %s", exc)

        try:
            import comfy.model_management as mm  # type: ignore
            log.info("[BatchHumoRender] Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache")
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
        except Exception as exc:
            log.warning("[BatchHumoRender] inter-phase VRAM cleanup failed: %s", exc)

        # ---- 9. Phase C: HuMo render loop (model stays warm) ----
        log.info("[BatchHumoRender] Phase C: HuMo render loop, %d lines",
                 sum(1 for e in plan if e.get("positive") is not None
                     and e.get("audio_emb") is not None))
        rendered = 0
        # BUG-LOCAL-089: track per-line clip records for ledger write-back.
        # The `clips[]` ledger field lets downstream tools (concat,
        # post-mortem, OBS scheduler) find every rendered clip by
        # ledger lookup instead of filesystem glob.
        clip_records: list[dict] = []
        for entry in plan:
            line_id = entry["line_id"]
            if entry.get("positive") is None or entry.get("audio_emb") is None:
                report_lines.append(f"  {line_id}: SKIP encode failed in earlier phase")
                continue

            shot_t0 = time.time()
            shot_seed = (seed + entry["idx"] * 1009) & 0x7FFFFFFFFFFFFFFF
            try:
                # WanHuMoImageToVideo returns (positive_with_humo_inputs,
                # negative_with_humo_inputs, latent). Use kwargs matching
                # the API-format prompt's input names (BUG-LOCAL-080 fix).
                humo_out = _call(
                    humo_node,
                    width=width,
                    height=height,
                    length=entry["humo_length"],
                    batch_size=1,
                    positive=entry["positive"],
                    negative=negative,
                    vae=vae,
                    audio_encoder_output=entry["audio_emb"],
                    ref_image=entry["ref_image"],
                )
                humo_pos, humo_neg, humo_latent = humo_out[:3]

                samples = _call(
                    sampler,
                    model=model,
                    seed=shot_seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=humo_pos,
                    negative=humo_neg,
                    latent_image=humo_latent,
                    denoise=1.0,
                )[0]

                images_out = _call(vae_decoder, samples=samples, vae=vae)[0]

                # BUG-LOCAL-102: trim the leading warmup window. HuMo
                # rendered N extra frames against the silence we
                # padded onto the audio so the model's intrinsic
                # motion-onset freeze (~3-6 frames) burned down before
                # the first dialogue phoneme. Drop those leading frames
                # from the decoded image tensor and the matching
                # leading silence from the audio so the on-disk clip
                # starts at the first articulated motion / first
                # dialogue word. Timeline placement (clips[].start_s)
                # therefore stays in terms of the original dur_s.
                pad_frames = int(entry.get("warmup_pad_frames", 0) or 0)
                pad_samples_save = int(entry.get("warmup_pad_samples", 0) or 0)
                if pad_frames > 0 and isinstance(images_out, torch.Tensor):
                    if images_out.shape[0] > pad_frames:
                        images_out = images_out[pad_frames:].contiguous()
                    else:
                        log.warning(
                            "[BatchHumoRender] BUG-102: line %s has %d frames "
                            "but pad_frames=%d -- skipping trim (degraded)",
                            line_id, images_out.shape[0], pad_frames,
                        )
                audio_for_save = (
                    _trim_audio_lead(entry["audio"], pad_samples_save)
                    if pad_samples_save > 0 else entry["audio"]
                )

                # BUG-LOCAL-083: ComfyUI v0.20.1's SaveVideo (and the
                # paired CreateVideo) is a V3-style node that expects
                # `cls.hidden` to be populated by the executor with
                # `extra_pnginfo` + `prompt` metadata. Direct-calling
                # those nodes from inside another node's execute()
                # leaves `cls.hidden = None` and crashes at line 100
                # of comfy_extras/nodes_video.py.
                #
                # Workaround: skip CreateVideo + SaveVideo entirely.
                # Write the clip mp4 directly via ffmpeg from the
                # decoded image tensor + per-line audio tensor. Same
                # output (a .mp4 with audio embedded) without the
                # broken executor-coupling. Output filename is
                # canonical `<line_id>.mp4` so concat finds it.
                clip_mp4_path = clips_dir_path / f"{line_id}.mp4"
                _save_clip_via_ffmpeg(
                    images=images_out,
                    audio_dict=audio_for_save,
                    out_path=clip_mp4_path,
                    fps=25,
                )

                # BUG-LOCAL-089: record this clip in the per-run
                # ledger.clips[] array. Includes line_id, char_id,
                # mp4_path, start_s, dur_s, source-tier, ref_png so
                # concat (VideoComposite) can resolve every clip via
                # ledger lookup instead of glob heuristics.
                clip_records.append({
                    "line_id": line_id,
                    "char_id": (lines[entry["idx"]].get("char_id") or "").strip() or None,
                    "speaker": entry["speaker"] or None,
                    "mp4_path": str(clip_mp4_path),
                    "start_s": float(entry["start_s"]),
                    "dur_s": float(entry["dur_s"]),
                    "humo_length": int(entry["humo_length"]),
                    "ref_png_name": entry.get("ref_png_name"),
                    "ref_source": entry.get("ref_source"),
                    # BUG-LOCAL-102: how many ms of leading silence the
                    # model rendered against (and that the save trim
                    # removed). Recorded for post-mortem traceability.
                    "warmup_pad_ms": int(warmup_pad_ms),
                })

                shot_ms = int((time.time() - shot_t0) * 1000)
                report_lines.append(
                    f"  {line_id} ({entry['speaker']}): {shot_ms} ms "
                    f"(length={entry['humo_length']} ref={entry['ref_png_name']})"
                )
                log.info("[BatchHumoRender] %s done in %d ms", line_id, shot_ms)
                rendered += 1
            except Exception as exc:
                log.exception("[BatchHumoRender] line %s failed: %s", line_id, exc)
                report_lines.append(f"  {line_id}: FAILED ({exc})")

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"Total: {rendered}/{len(plan)} clip(s) in {total_ms} ms"
        )
        log.info("[BatchHumoRender] complete: %d/%d clips in %d ms",
                 rendered, len(plan), total_ms)

        # ---- 10. BUG-LOCAL-089: write clips[] back to the ledger ----
        # Persist the per-clip records so VideoComposite + post-mortem
        # can resolve every clip via ledger lookup. Skipped silently
        # when the ledger came from an inline JSON blob (no source
        # path to write back to).
        if ledger_path is not None and clip_records:
            try:
                ledger["clips"] = clip_records
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, indent=2, ensure_ascii=False)
                log.info(
                    "[BatchHumoRender] ledger updated: %d clip records -> %s",
                    len(clip_records), ledger_path.name,
                )
                report_lines.append(
                    f"  ledger updated: {len(clip_records)} clip records"
                )
            except Exception as exc:
                log.warning(
                    "[BatchHumoRender] ledger clips write-back failed: %s", exc
                )
                report_lines.append(f"  ledger write-back FAILED: {exc}")
        elif ledger_path is None:
            log.info(
                "[BatchHumoRender] inline ledger (no path) -- "
                "skipping clips[] write-back"
            )

        return (str(clips_dir_path), rendered, "\n".join(report_lines))

    @staticmethod
    def _load_ledger(ledger_arg: str) -> dict:
        """Compatibility shim around ``_load_ledger_with_path``: returns
        only the parsed dict for callers that don't need the source
        path. New code should prefer ``_load_ledger_with_path`` so
        ledger-mtime-based freshness checks (BUG-LOCAL-088) can use
        the file's mtime as a cutoff.
        """
        ledger, _ = BatchHumoRender._load_ledger_with_path(ledger_arg)
        return ledger

    @staticmethod
    def _load_ledger_with_path(ledger_arg: str) -> tuple[dict, Path | None]:
        """Accept either:
          - inline JSON string (starts with '{') -- returns (dict, None)
          - path to *_ledger.json -- returns (dict, Path)
          - path to *.mp4 (audio episode); ledger inferred via
            suffix swap (.mp4 -> _ledger.json), since that's the
            convention OTR_SignalLostVideo / EpisodeAssembler write.
            Lets us wire BatchHumoRender's ledger_json input directly
            from SignalLostVideo.video_path -- no separate ledger
            output node required.
          - empty -> auto-pick newest non-pending in the canonical
            audio dirs (BUG-LOCAL-076 fallback chain).

        Returns (ledger_dict, ledger_path_or_None). When the input is
        an inline JSON blob the path is None (no on-disk source
        exists; BUG-088 freshness check falls back to a wall-clock
        cutoff in that case).
        """
        s = (ledger_arg or "").strip()

        # Auto-pick fallback when input empty
        if not s:
            audio_dirs = [
                otr_audio_dir(),
                otr_legacy_audio_dir(),
            ]
            cands = []
            for d in audio_dirs:
                if d.exists():
                    cands.extend(
                        p for p in d.glob("*_ledger.json")
                        if not p.name.startswith("pending_")
                    )
            if not cands:
                raise RuntimeError("BatchHumoRender: ledger_json empty and auto-pick found no ledger")
            p = max(cands, key=lambda x: x.stat().st_mtime)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p

        if s.startswith("{"):
            return json.loads(s), None

        p = Path(s)
        # .mp4 path -> swap suffix to _ledger.json (SignalLostVideo
        # convention)
        if p.suffix.lower() == ".mp4":
            ledger_p = p.with_suffix("").parent / f"{p.stem}_ledger.json"
            if ledger_p.exists():
                with open(ledger_p, "r", encoding="utf-8") as f:
                    return json.load(f), ledger_p
            raise RuntimeError(
                f"BatchHumoRender: derived ledger from .mp4 not found: {ledger_p}"
            )

        # Plain ledger.json path
        if not p.exists():
            raise RuntimeError(f"BatchHumoRender: ledger path not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p

    @staticmethod
    def _rename_savevideo_output(
        clips_dir: Path,
        line_id: str,
        report_lines: list[str],
    ) -> None:
        """SaveVideo writes ``humo_<line_id>_NNNNN_.mp4``. Rename
        the newest match to ``<line_id>.mp4`` so concat finds it
        without globbing for raw SaveVideo prefixes."""
        import shutil as _shutil
        candidates = sorted(
            clips_dir.glob(f"humo_{line_id}_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            report_lines.append(f"  {line_id}: WARN SaveVideo output not found in {clips_dir}")
            return
        src = candidates[0]
        dst = clips_dir / f"{line_id}.mp4"
        try:
            _shutil.move(str(src), str(dst))
        except Exception as exc:
            report_lines.append(f"  {line_id}: WARN rename failed: {exc}")


__all__ = ["BatchHumoRender"]
