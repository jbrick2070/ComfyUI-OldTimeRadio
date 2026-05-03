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
# the broadcast tree under output/otr/, but importing folder_paths here
# explicitly satisfies the Bug Bible BUG-01.02 contract that every
# OUTPUT_NODE file references the canonical resolver.
import folder_paths  # noqa: F401,E402

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

# 8n+1 frame count rule for LTX VAE temporal compression. 177 =
# 8*22+1 = ~7.08 s @ 25 fps. This intentionally matches HuMo's
# HUMO_MAX_FRAMES=177 cap (HuMo uses 4n+1; 177 is also 4n+1=4*44+1)
# so the chunking math in scene_sequencer.py stays the same for both
# renderers -- no per-renderer math branching needed.
# Per Jeffrey 2026-05-01 EVENING ("to make it easy you can keep LTX
# to the same clip duration max as HuMo"). LTX could go to 257
# native (proven in ComfyUI-Goofer with VAEDecodeTiled), but keeping
# 177 here simplifies the timing contract.
LTX_MAX_FRAMES = 177
LTX_MIN_FRAMES = 9    # 8*1+1, smallest valid LTX render

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

# 2026-05-01 Jeffrey: feed the radio still as BOTH start and end
# keyframes via LTXVAddGuide so each clip seamlessly loops back to
# its starting composition. This makes ping-pong / continuous loop
# in VideoComposite trivially clean (no jarring cut at clip boundary)
# and matches the OTR aesthetic of "the radio is the visual anchor;
# motion happens around it, then settles." End-frame strength is
# slightly lower than start so the model has more freedom in the
# middle of the clip.
LTX_END_FRAME_STRENGTH = 0.6

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
    LTX_MAX_FRAMES = 257 = ~10.28 s @ 25 fps -- covers full music_open
    (~10 s) and music_close (~8 s) cues natively. For lines longer
    than 10.28 s (e.g. a long announcer monologue), VideoComposite
    downstream can ping-pong-loop or freeze-frame extend.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 7) // 8
    frames = 8 * n + 1
    if frames < LTX_MIN_FRAMES:
        frames = LTX_MIN_FRAMES
    if frames > LTX_MAX_FRAMES:
        frames = LTX_MAX_FRAMES
    return frames


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
                f"LTXVImgToVideo, LTXVConditioning, RandomNoise, "
                f"CFGGuider, KSamplerSelect, SamplerCustomAdvanced, "
                f"VAEDecode, EmptyLTXVLatentVideo. "
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
        fs_path = otr_stills_dir() / f"radio_bookend_{eid}.png"
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
                        "Optional sequencing edge from BatchHumoRender.clips_dir. "
                        "When wired, ComfyUI waits for HuMo to finish writing its "
                        "per-line .mp4 files before LTX starts loading -- so HuMo "
                        "fully unloads before LTX claims VRAM. Pure DAG dependency; "
                        "the value is not used by execute()."
                    ),
                }),
            },
        }

    def execute(self, model, clip, vae, ledger_json, seed=1, ffmpeg="ffmpeg",
                humo_clips_dir=""):
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
            ltx_length = ltx_length_for_dur(float(dur_s))
            prompt_text = _PROMPT_BY_ROLE.get(
                speaker_role, _PROMPT_BY_ROLE["sfx"]
            )
            plan.append({
                "line_id": line_id,
                "speaker_role": speaker_role,
                "dur_s": float(dur_s),
                "ltx_length": ltx_length,
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
                ltx_length = entry["ltx_length"]
                prompt_text = entry["prompt_text"]
                shot_seed = (seed + idx * 1009) & 0x7FFFFFFFFFFFFFFF
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

                    # Empty video latent at the requested length.
                    empty_latent = _call(
                        "EmptyLTXVLatentVideo",
                        width=LTX_WIDTH, height=LTX_HEIGHT,
                        length=ltx_length, batch_size=1,
                    )[0]

                    # 2026-05-01 EVENING (Jeffrey): feed the radio
                    # still as BOTH the start (frame_idx=0) AND the
                    # end (frame_idx=-1) keyframe via LTXVAddGuide.
                    # This makes each clip seamlessly loop -- whatever
                    # motion happens in the middle settles back to
                    # the same composition at the end. Ping-pong /
                    # continuous looping in VideoComposite gets a
                    # clean cut. Negative frame_idx is supported
                    # natively by LTXVAddGuide ("counted from end of
                    # video", per nodes_lt.py:224).
                    cond_pos, cond_neg, guided_latent = _call(
                        "LTXVAddGuide",
                        positive=cond_pos,
                        negative=cond_neg,
                        vae=vae,
                        latent=empty_latent,
                        image=ref_image,
                        frame_idx=0,
                        strength=LTX_I2V_STRENGTH,
                    )
                    cond_pos, cond_neg, latent = _call(
                        "LTXVAddGuide",
                        positive=cond_pos,
                        negative=cond_neg,
                        vae=vae,
                        latent=guided_latent,
                        image=ref_image,
                        frame_idx=-1,
                        strength=LTX_END_FRAME_STRENGTH,
                    )

                    # Sample with distilled schedule
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
                        latent_image=latent,
                    )

                    # VAEDecodeTiled per Goofer's proven pattern --
                    # required for safe decoding at higher frame counts
                    # on this Blackwell hardware. Parameters from
                    # ComfyUI-Goofer line 472-476 (empirically verified).
                    frames = _call(
                        "VAEDecodeTiled",
                        samples=samples_out, vae=vae,
                        tile_size=LTX_TILE_SIZE,
                        overlap=LTX_TILE_OVERLAP,
                        temporal_size=LTX_TEMPORAL_SIZE,
                        temporal_overlap=LTX_TEMPORAL_OVERLAP,
                    )[0]

                    # Save mp4 (silent, libx264 yuv420p, fixed timebase
                    # to match HuMo clips for VideoComposite concat).
                    out_path = clips_dir / f"{line_id}.mp4"
                    _save_video_mp4(
                        images=frames,
                        out_path=out_path,
                        fps=LTX_FPS,
                        ffmpeg=ffmpeg,
                    )

                    shot_ms = int((time.time() - shot_t0) * 1000)
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

                    rendered_clips.append({
                        "line_id": line_id,
                        "speaker_role": entry["speaker_role"],
                        "mp4_path": str(out_path),
                        "ltx_length": ltx_length,
                        "dur_s": entry["dur_s"],
                        "ltx_render_ms": shot_ms,
                        "ref_png_name": radio_png.name,
                        "ref_source": "ltx-radio-bookend",
                        "source_kind": "ltx",
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
          - inline JSON string (starts with '{') -- returns (dict, None)
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
        STRING output. BUG-LOCAL-011 (2026-05-02): originally this
        method only handled exact .json paths and failed when wired to
        SignalLostVideo's .mp4 output -- ported the multi-tier stem
        fallback from batch_humo_render.py.
        """
        import json as _json

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
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return _json.load(f), p
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[BatchLTXRender] auto-pick ledger %s failed to load: %s",
                    p, exc,
                )
                return None, None

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
        # (SignalLostVideo / EpisodeAssembler convention). Same multi-tier
        # fallback chain as BatchHumoRender (BUG-LOCAL-118 hardening):
        #   (1) exact match,
        #   (2) collapsed-underscore variant,
        #   (3) directory scan for newest <1h old fuzzy match.
        if p.suffix.lower() == ".mp4":
            audio_dir = p.parent
            stem = p.stem

            # Tier 1: direct match.
            ledger_p = audio_dir / f"{stem}_ledger.json"
            if ledger_p.exists():
                try:
                    with open(ledger_p, "r", encoding="utf-8") as f:
                        return _json.load(f), ledger_p
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[BatchLTXRender] tier-1 ledger %s failed to "
                        "load: %s", ledger_p, exc,
                    )

            # Tier 2: underscore-collapse variant.
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
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            return _json.load(f), cand
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "[BatchLTXRender] tier-2 ledger %s failed to "
                            "load: %s", cand, exc,
                        )

            # Tier 3: directory scan for fuzzy-match ledger <1h old.
            try:
                cands = list(audio_dir.glob("*_ledger.json"))
                cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                stem_norm = collapsed
                for cand in cands[:10]:
                    cand_eid = cand.stem
                    if cand_eid.endswith("_ledger"):
                        cand_eid = cand_eid[: -len("_ledger")]
                    cand_norm = cand_eid
                    while "__" in cand_norm:
                        cand_norm = cand_norm.replace("__", "_")
                    if (cand_norm == stem_norm
                            or cand_norm in stem_norm
                            or stem_norm in cand_norm):
                        age_s = time.time() - cand.stat().st_mtime
                        if age_s > 3600:
                            continue
                        log.warning(
                            "[BatchLTXRender] BUG-LOCAL-118 fuzzy fallback: "
                            "binding to %r (age %.0fs) for .mp4 stem %r.",
                            cand.name, age_s, stem,
                        )
                        try:
                            with open(cand, "r", encoding="utf-8") as f:
                                return _json.load(f), cand
                        except Exception as exc:  # noqa: BLE001
                            log.warning(
                                "[BatchLTXRender] tier-3 ledger %s failed "
                                "to load: %s", cand, exc,
                            )
            except Exception as scan_exc:  # noqa: BLE001
                log.warning(
                    "[BatchLTXRender] BUG-LOCAL-118 directory-scan "
                    "fallback failed (%s) - falling through.", scan_exc,
                )

            log.warning(
                "[BatchLTXRender] derived ledger from .mp4 not found: "
                "%s (tried direct, collapsed-underscore, fuzzy scan)",
                ledger_p,
            )
            return None, None

        # Layer 2b: plain _ledger.json path.
        if not p.is_file():
            return None, None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f), p
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[BatchLTXRender] ledger path %s failed to load: %s",
                p, exc,
            )
            return None, None


__all__ = ["BatchLTXRender"]
