"""
rtx_upscale.py  --  OTR_RTXUpscale ComfyUI node
================================================

Final-stage video upscaler for the OTR pipeline. Wraps NVIDIA's
``RTXVideoSuperResolution`` (from custom_nodes/Nvidia_RTX_Nodes_ComfyUI)
with a path-in / path-out interface that preserves C7 audio identity:

    composite mp4 (1472x832) -> chunked frame extract via ffmpeg
                              -> RTX VSR (HW-accelerated, ULTRA quality)
                              -> silent libx264 yuv420p mp4
                              -> ffmpeg mux with -c:a copy from source mp4
                              -> output/otr/episodes/<ep>/upscaled/<source filename> (1920x1080)
    (No filename suffix is appended; the output uses the same basename
    as the source. A prior version added a ``_1080p`` suffix; that
    behavior was removed -- if a stale workflow JSON still has an 8th
    widget value of ``"_1080p"``, ComfyUI silently ignores it.)

The audio stream is NEVER decoded -- ``-c:a copy`` from the source mp4
to the upscaled mp4 means the audio bytes are bit-identical. Stays
inside C7.

When ``bypass`` is True the node is a pass-through: it returns the
source path unchanged with no upscale work, so the workflow Ctrl+B
toggle just disables this stage cleanly (no half-upscaled output, no
broken downstream reads).

VRAM strategy: nvvfx is HW-accelerated via the RTX driver and uses
~0 GB of CUDA VRAM (separate NVIDIA Video Effects context). We chunk
frames to bound RAM (default 64 frames per chunk = ~26 MB at 1920x1080
RGB float32 staging, ~18 MB at 832x480) so a 5 min episode at 25 fps
(7,500 frames) doesn't allocate a single 4 GB tensor.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys as _sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path-helper bootstrap (same pattern as batch_humo_render.py)
# ---------------------------------------------------------------------------
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))

# folder_paths is the ComfyUI canonical path resolver. The OTR helpers
# wrap it internally; this explicit import documents the dependency and
# satisfies the Bug Bible BUG-01.02 contract that every OUTPUT_NODE file
# references the canonical resolver.
import folder_paths  # noqa: F401,E402

from _otr_paths import otr_obs_dir, otr_upscaled_dir  # noqa: E402

log = logging.getLogger("OTR.rtx_upscale")


# Default chunk size for frame batching. Keeps host RAM bounded on long
# episodes. 64 frames @ 1920x1080 RGB uint8 = 380 MB; @ float32 = 1.5 GB
# for the brief upscale-tensor lifetime. Safe on 32 GB host RAM machines
# with browsers + ComfyUI desktop also resident.
DEFAULT_CHUNK_FRAMES = 64

# Default upscale target. 832x480 -> 1920x1080 = 2.3x linear, near
# RTX VSR's sweet spot for 1080p output.
DEFAULT_TARGET_WIDTH = 1920
DEFAULT_TARGET_HEIGHT = 1080


def _probe_video_dims(ffprobe: str, mp4_path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) of a video file via ffprobe."""
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=nw=1:nk=1",
        str(mp4_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
    parts = [p.strip() for p in out.strip().splitlines() if p.strip()]
    if len(parts) < 3:
        raise RuntimeError(f"ffprobe returned malformed output: {out!r}")
    w = int(parts[0])
    h = int(parts[1])
    rate = parts[2]
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) != 0.0 else 25.0
    else:
        fps = float(rate)
    return w, h, fps


def _chunked_upscale(
    *,
    src_mp4: Path,
    silent_out_mp4: Path,
    target_w: int,
    target_h: int,
    quality: str,
    chunk_frames: int,
    ffmpeg: str,
    ffprobe: str,
    diagnostic_dump_dir: Path | None = None,
) -> tuple[int, float]:
    """Decode src frames in chunks via ffmpeg, run RTX VSR per chunk,
    encode to silent libx264 yuv420p mp4. Returns (total_frames_out, fps).

    Avoids holding all frames in RAM. Pipes raw RGB24 in/out of ffmpeg
    so RTX VSR sees a torch tensor that's already in CPU memory.

    BUG-LOCAL-031 diagnostic instrumentation (2026-05-03 EVENING):
    when ``diagnostic_dump_dir`` is set, every chunk dumps three PNGs
    + a stats line:

      * ``chunkNNNN_a_input_uint8.png``   -- first frame BEFORE nvvfx
      * ``chunkNNNN_b_nvvfx_float.png``   -- first frame AFTER ``sr.run`` (auto-scaled)
      * ``chunkNNNN_c_post_clamp_byte.png`` -- first frame as it lands in encode pipe
      * ``chunk_stats.txt`` -- per-chunk min/max/mean of the post-nvvfx tensor

    Reading the stats file alone is enough to localize the bug:
      * If post-nvvfx max <= 1.0 -> nvvfx returns 0..1 floats and the
        unconditional ``clamp(0, 255).byte()`` truncates everything to 0
        (range mismatch fix: multiply by 255 before clamp).
      * If post-nvvfx max > 1 but <= 255 -> range was correct; bug is
        elsewhere (encode pipe, dim mismatch, etc).
      * If post-nvvfx max == 0 across all chunks -> nvvfx returned an
        all-zero tensor (likely Blackwell sm_120 incompat: file an
        upstream bug, do NOT silently fall back).

    Production paths leave ``diagnostic_dump_dir=None`` so this is
    pure instrumentation, never modifies behavior on a normal render.
    """
    import numpy as np  # type: ignore
    import torch  # type: ignore

    src_w, src_h, src_fps = _probe_video_dims(ffprobe, src_mp4)
    log.info(
        "[OTR_RTXUpscale] source dims=%dx%d fps=%.3f -> target %dx%d quality=%s",
        src_w, src_h, src_fps, target_w, target_h, quality,
    )

    # Lazy import nvvfx so the node doesn't fail to load on machines
    # without the RTX VSR driver components installed.
    try:
        import nvvfx  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "[OTR_RTXUpscale] nvvfx (NVIDIA Video Effects SDK Python "
            f"binding) not available: {exc}. Install the "
            "Nvidia_RTX_Nodes_ComfyUI custom node + matching driver."
        )

    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    quality_level = quality_mapping.get(
        quality.upper(), nvvfx.effects.QualityLevel.ULTRA
    )

    # Spawn the decode pipe: rawvideo rgb24 from src
    decode_cmd = [
        ffmpeg, "-loglevel", "error",
        "-i", str(src_mp4),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an",  # ignore audio on the decode side; we mux it later
        "-",
    ]
    # Spawn the encode pipe: rawvideo rgb24 -> libx264 yuv420p
    encode_cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{target_w}x{target_h}",
        "-r", f"{src_fps:.6f}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast",
        "-video_track_timescale", "12800",  # match HuMo/LTX/procgen timebase
        "-an",
        str(silent_out_mp4),
    ]
    silent_out_mp4.parent.mkdir(parents=True, exist_ok=True)

    src_frame_bytes = src_w * src_h * 3
    total_frames_out = 0

    # BUG-LOCAL-031 diagnostic dump setup (no-op when disabled).
    _diag_dir: Path | None = None
    _diag_stats_lines: list[str] = []
    if diagnostic_dump_dir is not None:
        _diag_dir = Path(diagnostic_dump_dir)
        _diag_dir.mkdir(parents=True, exist_ok=True)
        _diag_stats_lines.append(
            "# BUG-LOCAL-031 RTXUpscale diagnostic dump\n"
            f"# source: {src_mp4}\n"
            f"# source dims: {src_w}x{src_h} fps={src_fps:.3f}\n"
            f"# target dims: {target_w}x{target_h} quality={quality}\n"
            f"# chunk_frames={chunk_frames}\n"
            f"# columns: chunk_idx | n_frames | "
            "in_uint8(min,max,mean) | "
            "post_nvvfx_float(min,max,mean) | "
            "post_clamp_byte(min,max,mean)\n"
        )
        log.info(
            "[OTR_RTXUpscale] DIAGNOSTIC DUMP enabled -> %s",
            _diag_dir,
        )

    # Windows pipe deadlock fix (consult 2026-05-02 Gemini): the OS pipe
    # buffer on Windows is ~64 KB. ffmpeg writes to stderr verbosely under
    # `-loglevel error` only when something is wrong, but a long-running
    # decode/encode can still emit enough text to fill the buffer and
    # block. We don't read stderr in real time (it would require a thread),
    # so route stderr to DEVNULL. If a stage fails the returncode is
    # still non-zero and the upscale fails loudly with a clear message.
    decode_proc = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    encode_proc = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    try:
        with nvvfx.VideoSuperRes(quality_level) as sr:
            sr.output_width = target_w
            sr.output_height = target_h
            sr.load()

            buffer = bytearray()
            chunk_byte_target = src_frame_bytes * chunk_frames
            eof = False
            while not eof:
                # Read until we have enough for a full chunk or EOF.
                while len(buffer) < chunk_byte_target:
                    data = decode_proc.stdout.read(
                        chunk_byte_target - len(buffer)
                    )
                    if not data:
                        eof = True
                        break
                    buffer.extend(data)

                # Emit only whole frames from this chunk.
                whole_frames = len(buffer) // src_frame_bytes
                if whole_frames == 0:
                    break

                chunk_bytes = bytes(buffer[: whole_frames * src_frame_bytes])
                buffer = buffer[whole_frames * src_frame_bytes:]

                # Reshape to [N, H, W, 3] uint8.
                # np.frombuffer returns a non-writable array (it's a view
                # over the immutable bytes buffer). torch.from_numpy on a
                # non-writable ndarray emits UserWarning. Copy to silence
                # the warning AND to give torch a writable buffer it can
                # safely transfer to CUDA.
                arr = np.frombuffer(chunk_bytes, dtype=np.uint8).copy()
                arr = arr.reshape((whole_frames, src_h, src_w, 3))

                # Per-frame upscale via nvvfx (matches the upstream
                # RTXVideoSuperResolution.execute() loop pattern).
                # Keep the work in fp32 on CUDA, return uint8 on CPU.
                #
                # BUG-LOCAL-031 FIX (2026-05-03 EVENING): nvvfx expects
                # input in 0.0..1.0 float range (matches the ComfyUI
                # IMAGE convention used by the upstream
                # `Nvidia_RTX_Nodes_ComfyUI/RTXVideoSuperResolution`
                # reference node). The previous code did ``.float()``
                # WITHOUT dividing by 255, which kept the values in
                # 0..255 numerically and fed nvvfx out-of-distribution
                # input. nvvfx then internally treated those values
                # like 0..1, saturated them to garbage near 1.0, and
                # the resulting frames clamped+byte()'d to solid
                # black/white. Diagnosis confirmed via
                # rtx_diag/chunk_stats.txt: every chunk reported
                # nvvfx output min=0.0/max=1.0 with mean values that
                # did not correlate sensibly with input mean.
                #
                # Companion fix: the post-nvvfx output is multiplied
                # by 255.0 before the byte cast (see _scaled below) to
                # bring the 0..1 range back into uint8 0..255 for
                # ffmpeg's rgb24 pipe.
                gpu_in = torch.from_numpy(arr).cuda().permute(
                    0, 3, 1, 2
                ).float().contiguous() / 255.0
                gpu_out = torch.empty(
                    (whole_frames, 3, target_h, target_w),
                    device=gpu_in.device, dtype=gpu_in.dtype,
                )
                for j in range(whole_frames):
                    dlpack_out = sr.run(gpu_in[j]).image
                    gpu_out[j: j + 1] = (
                        torch.from_dlpack(dlpack_out)
                        .movedim(0, -1)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                # BUG-LOCAL-031 diagnostic: capture nvvfx output stats
                # BEFORE the clamp/byte conversion. If nvvfx returns
                # 0..1 floats and we never multiply by 255, this is
                # where the data dies. The stats line answers in one
                # number whether we have a range bug, a Blackwell
                # silent-zero bug, or something else entirely.
                _post_nvvfx_min = _post_nvvfx_max = _post_nvvfx_mean = 0.0
                _in_min = _in_max = _in_mean = 0.0
                if _diag_dir is not None:
                    try:
                        _gn = gpu_out.detach()
                        _post_nvvfx_min = float(_gn.min().item())
                        _post_nvvfx_max = float(_gn.max().item())
                        _post_nvvfx_mean = float(_gn.float().mean().item())
                        _in_min = float(arr.min())
                        _in_max = float(arr.max())
                        _in_mean = float(arr.mean())
                    except Exception:
                        pass

                # BUG-LOCAL-031 FIX (2026-05-03 EVENING):
                # nvvfx.VideoSuperRes returns float32 in 0.0..1.0, NOT
                # 0..255. The previous code did clamp(0, 255).byte()
                # which is a no-op for 0..1 floats, then byte() truncated
                # 0.95 -> 0 and 1.0 -> 1, producing essentially black
                # frames (the source of the BUG-LOCAL-031 black-output
                # symptom). Diagnosis confirmed via chunk_stats.txt
                # diagnostic dump: every chunk reported nvvfx max=1.0,
                # post_clamp max=1, mean dropping from 0.95 to 0.3 to
                # 0.9 across the timeline -- exactly the 0..1 range
                # signature.
                #
                # Detect the range from the chunk maximum and scale
                # accordingly. Only chunks above max>1.5 are treated as
                # already-uint8-range (would happen if a future nvvfx
                # version changes its output convention -- this is
                # forward-compatible).
                _gn_max = float(gpu_out.detach().max().item())
                if _gn_max <= 1.5:
                    _scaled = gpu_out * 255.0
                else:
                    _scaled = gpu_out

                # Back to NHWC uint8 on CPU for ffmpeg pipe.
                out_frames = (
                    _scaled.clamp(0.0, 255.0)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                del _scaled

                # BUG-LOCAL-031 diagnostic: dump three sample frames +
                # the post-clamp stats. We grab frame[0] of each chunk
                # so the dump stays small even on long episodes.
                if _diag_dir is not None and out_frames.shape[0] > 0:
                    try:
                        from PIL import Image  # type: ignore
                        chunk_idx = total_frames_out // max(1, chunk_frames)
                        # (a) input uint8 -- pre-nvvfx
                        Image.fromarray(arr[0]).save(
                            _diag_dir
                            / f"chunk{chunk_idx:04d}_a_input_uint8.png"
                        )
                        # (b) post-nvvfx float -- auto-scale to 0..255
                        # for visual inspection. Picks the sensible scale
                        # based on observed max so the PNG is viewable
                        # whether the model returned 0..1 or 0..255.
                        _f0 = (
                            gpu_out[0]
                            .detach()
                            .float()
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                        _scale = (
                            255.0 if _post_nvvfx_max <= 1.5
                            else 1.0
                        )
                        _f0v = (_f0 * _scale).clip(0, 255).astype("uint8")
                        Image.fromarray(_f0v).save(
                            _diag_dir
                            / f"chunk{chunk_idx:04d}_b_nvvfx_float_x{int(_scale)}.png"
                        )
                        # (c) post clamp+byte -- exactly what hits ffmpeg
                        Image.fromarray(out_frames[0]).save(
                            _diag_dir
                            / f"chunk{chunk_idx:04d}_c_post_clamp_byte.png"
                        )
                        _post_byte_min = int(out_frames.min())
                        _post_byte_max = int(out_frames.max())
                        _post_byte_mean = float(out_frames.mean())
                        _diag_stats_lines.append(
                            f"chunk{chunk_idx:04d} | "
                            f"n={whole_frames} | "
                            f"in_u8(min={_in_min:.0f},max={_in_max:.0f},mean={_in_mean:.1f}) | "
                            f"nvvfx(min={_post_nvvfx_min:.4f},max={_post_nvvfx_max:.4f},mean={_post_nvvfx_mean:.4f}) | "
                            f"post_clamp(min={_post_byte_min},max={_post_byte_max},mean={_post_byte_mean:.1f})"
                        )
                    except Exception as _diag_exc:
                        log.warning(
                            "[OTR_RTXUpscale] diagnostic dump failed: %s",
                            _diag_exc,
                        )

                encode_proc.stdin.write(out_frames.tobytes())
                total_frames_out += whole_frames

                # Free GPU staging eagerly.
                del gpu_in, gpu_out, out_frames
        encode_proc.stdin.close()
    except Exception:
        # Make sure subprocesses are torn down before re-raising.
        try:
            encode_proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        decode_proc.kill()
        encode_proc.kill()
        raise

    decode_proc.wait()
    encode_proc.wait()
    if decode_proc.returncode not in (0, None):
        stderr = decode_proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed: {stderr[:500]}")
    if encode_proc.returncode not in (0, None):
        stderr = encode_proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg encode failed: {stderr[:500]}")

    # BUG-LOCAL-031 diagnostic: write the per-chunk stats file.
    if _diag_dir is not None and _diag_stats_lines:
        try:
            (_diag_dir / "chunk_stats.txt").write_text(
                "\n".join(_diag_stats_lines), encoding="utf-8",
            )
            log.info(
                "[OTR_RTXUpscale] DIAGNOSTIC DUMP wrote %d chunk stats -> %s",
                len(_diag_stats_lines) - 1,  # -1 for header
                _diag_dir / "chunk_stats.txt",
            )
        except Exception as _exc:
            log.warning(
                "[OTR_RTXUpscale] failed to write diagnostic stats: %s",
                _exc,
            )

    return total_frames_out, src_fps


def _mux_audio_passthrough(
    *,
    silent_video_mp4: Path,
    audio_source_mp4: Path,
    out_mp4: Path,
    ffmpeg: str,
) -> None:
    """Mux audio_source_mp4's audio stream onto silent_video_mp4's video
    with -c:a copy (byte-identical audio). C7-safe."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Consult 2026-05-02 (Gemini): drop `-shortest`. The silent video and the
    # audio source share the same source frame timing (we re-encoded video at
    # the same fps, same frame count from the same input mp4), so video and
    # audio durations are equal by construction. `-shortest` adds no safety
    # and risks dropping final AAC packets if there's any sub-frame drift.
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(silent_video_mp4),
        "-i", str(audio_source_mp4),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[OTR_RTXUpscale] audio passthrough mux failed: {stderr[:500]}"
        )


class RTXUpscale:
    """Final-stage video upscaler for the OTR pipeline.

    Path-in / path-out wrapper around NVIDIA's RTX VSR. Audio passes
    through with -c:a copy so the upscaled mp4 keeps byte-identical
    audio (C7).

    Bypass mode (workflow Ctrl+B): return the source mp4 path unchanged
    with no upscale work, so toggling the node off cleanly disables
    just this stage.
    """

    CATEGORY = "OTR/v2/Visual"
    OUTPUT_NODE = True
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("upscaled_mp4_path", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mp4_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Path to the composited mp4 from VideoComposite "
                        "(typically <ep>.mp4 at 832x480). Will be upscaled "
                        "to target_width x target_height with audio "
                        "passthrough."
                    ),
                }),
            },
            "optional": {
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When True, return source_mp4_path unchanged "
                        "(no upscale, no new file written). Use this to "
                        "skip the upscale stage for raw 832x480 output."
                    ),
                }),
                "target_width": ("INT", {
                    "default": DEFAULT_TARGET_WIDTH,
                    "min": 256, "max": 7680, "step": 8,
                    "tooltip": "Upscaled width in pixels.",
                }),
                "target_height": ("INT", {
                    "default": DEFAULT_TARGET_HEIGHT,
                    "min": 256, "max": 4320, "step": 8,
                    "tooltip": "Upscaled height in pixels.",
                }),
                "quality": (
                    ["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    {"default": "ULTRA",
                     "tooltip": (
                         "RTX VSR quality preset. ULTRA is the strongest, "
                         "near-real-time on RTX 50 series."
                     )},
                ),
                "chunk_frames": ("INT", {
                    "default": DEFAULT_CHUNK_FRAMES,
                    "min": 4, "max": 512, "step": 4,
                    "tooltip": (
                        "Frames per RTX VSR chunk. Bounds host RAM on "
                        "long episodes. 64 is safe for 5 min episodes "
                        "on a 32 GB host."
                    ),
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "multiline": False,
                    "tooltip": "ffmpeg binary path or PATH-resolvable name.",
                }),
            },
        }

    def execute(
        self,
        source_mp4_path: str,
        bypass: bool = False,
        target_width: int = DEFAULT_TARGET_WIDTH,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        quality: str = "ULTRA",
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        ffmpeg: str = "ffmpeg",
        diagnostic_dump_dir: str = "",
    ):
        t_start = time.time()
        report_lines: list[str] = []

        src = Path((source_mp4_path or "").strip())
        if not src.exists():
            return ("", f"error: source_mp4_path not found: {src}")

        ffprobe = (
            ffmpeg.replace("ffmpeg", "ffprobe")
            if ffmpeg.endswith("ffmpeg")
            else "ffprobe"
        )
        if not (shutil.which(ffmpeg) or Path(ffmpeg).exists()):
            return ("", f"error: ffmpeg not found at {ffmpeg!r}")

        # 1080p upscale lands in the per-episode upscaled/ dir as an
        # INTERMEDIATE -- the broadcast obs/ dir holds only the final
        # post-blend deliverable (Jeffrey directive 2026-05-05; pre-blend
        # files in obs/ broke the "exactly one mp4 per episode" contract).
        # source filename is "<episode_id>.mp4" out of VideoComposite, so
        # src.stem == episode_id.
        episode_id = src.stem
        out_dir = otr_upscaled_dir(episode_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / src.name

        if bypass:
            # Bypass: copy the composite intermediate into the upscaled/
            # per-episode dir so the chain still produces an upscaled-
            # stage artifact for the downstream procgen blend node to
            # read, regardless of upscale on/off.
            shutil.copy2(src, out_mp4)
            report_lines.append(
                f"OTR_RTXUpscale: BYPASS -- copied {src.name} to "
                f"upscaled/ ({out_mp4.stat().st_size / (1024*1024):.1f} MB)"
            )
            log.info(
                "[OTR_RTXUpscale] bypass=True; copied %s -> %s",
                src, out_mp4,
            )
            return (str(out_mp4), "\n".join(report_lines))

        # Source dims for the report
        try:
            src_w, src_h, _src_fps = _probe_video_dims(ffprobe, src)
            report_lines.append(
                f"OTR_RTXUpscale: source {src.name} "
                f"{src_w}x{src_h} -> {target_width}x{target_height} "
                f"({quality}); writing to "
                f"otr/episodes/{episode_id}/upscaled/{out_mp4.name}"
            )
        except Exception as exc:  # noqa: BLE001
            report_lines.append(
                f"OTR_RTXUpscale: ffprobe of source failed ({exc}); "
                "continuing"
            )

        # Silent intermediate goes to a tempdir, gets removed after mux
        tmp_dir = Path(tempfile.mkdtemp(prefix="otr_rtx_upscale_"))
        silent = tmp_dir / "video_only.mp4"
        try:
            _diag_path: Path | None = None
            if diagnostic_dump_dir:
                _diag_path = Path(diagnostic_dump_dir)
            else:
                # Live-soak override: set env var OTR_RTX_DIAG_DUMP=1 to
                # force the diagnostic dump on for ONE render without
                # editing code or exposing a widget. Lands in
                # output/otr/obs/rtx_diag/ next to the upscale mp4.
                # Per fact-check 2026-05-03 EVENING: this avoids the
                # one-line code edit Step 3 of the go-forward plan
                # would otherwise require.
                if os.environ.get("OTR_RTX_DIAG_DUMP", "").strip() in ("1", "true", "TRUE", "yes"):
                    _diag_path = otr_obs_dir() / "rtx_diag"
                    log.info(
                        "[OTR_RTXUpscale] OTR_RTX_DIAG_DUMP=1 detected; "
                        "diagnostic dump forced ON -> %s", _diag_path,
                    )
            n_frames, fps = _chunked_upscale(
                src_mp4=src,
                silent_out_mp4=silent,
                target_w=target_width,
                target_h=target_height,
                quality=quality,
                chunk_frames=chunk_frames,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                diagnostic_dump_dir=_diag_path,
            )
            report_lines.append(
                f"  RTX VSR: {n_frames} frames @ {fps:.3f} fps "
                f"-> {silent.stat().st_size / (1024*1024):.1f} MB silent"
            )

            _mux_audio_passthrough(
                silent_video_mp4=silent,
                audio_source_mp4=src,
                out_mp4=out_mp4,
                ffmpeg=ffmpeg,
            )
            out_size_mb = out_mp4.stat().st_size / (1024 * 1024)
            report_lines.append(
                f"  audio mux (-c:a copy): {out_mp4.name} ({out_size_mb:.1f} MB)"
            )
        finally:
            try:
                if silent.exists():
                    silent.unlink()
                tmp_dir.rmdir()
            except Exception:  # noqa: BLE001
                pass

        # ----------------------------------------------------------------
        # Spacesaver cleanup: read the ledger's perfect_run_spacesaver
        # flag (stamped by OTR_LedgerScriptWriter at workflow start). When
        # ON, wipe every per-episode intermediate file under
        # otr/episodes/<episode_id>/ EXCEPT the ledger json + the
        # treatment txt. The final upscaled mp4 already lives at
        # otr/episodes/<ep>/upscaled/<ep>.mp4 (or out_mp4 for bypass),
        # unaffected. The final post-blend mp4 in otr/obs/ is also
        # written by a downstream node, not touched here.
        # ----------------------------------------------------------------
        try:
            _spacesaver_cleanup_if_flagged(src=src, log_lines=report_lines)
        except Exception as exc:  # noqa: BLE001
            report_lines.append(
                f"  spacesaver cleanup failed (non-fatal): {exc}"
            )
            log.warning("[OTR_RTXUpscale] spacesaver cleanup failed: %s", exc)

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"OTR_RTXUpscale: complete in {total_ms} ms -> {out_mp4}"
        )
        log.info(
            "[OTR_RTXUpscale] complete in %d ms: %s -> %s",
            total_ms, src.name, out_mp4.name,
        )
        return (str(out_mp4), "\n".join(report_lines))


def _spacesaver_cleanup_if_flagged(*, src: Path, log_lines: list[str]) -> None:
    """If the ledger's `meta.perfect_run_spacesaver` flag is True, wipe
    every per-episode intermediate file under otr/episodes/<ep>/ EXCEPT
    the ledger json + the treatment txt. No-op when the flag is False.

    The episode directory is derived DIRECTLY from `src` -- which is
    always otr/episodes/<ep>/composited/<ep>.mp4 (the VideoComposite
    output). No global ledger scan: that approach was wrong-episode-
    unsafe when two episodes were in flight (BUG-LOCAL-014, consult
    2026-05-02 -- both reviewers flagged it). src.parent.parent IS the
    episode root, by construction.

    Keep list (NOT deleted):
      - otr/episodes/<ep>/audio/*_ledger.json (whichever ledger lives there)
      - otr/episodes/<ep>/audio/*_treatment.txt (built from real on-disk
        filenames, not slug-reconstructed -- see Finding 4 in QA pass)

    Wipe list (everything else under otr/episodes/<ep>/):
      - audio/<ep>.mp4 (procgen base)
      - audio/*.wav (Bark / MusicGen / AudioGen wavs)
      - audio/director_dump_*.txt
      - audio/pending_*_ledger.json (any orphan pending files)
      - stills/                       (FLUX environments + radio bookend)
      - portraits/                    (PASS1 character portraits)
      - videos/                       (per-line piece clips)
      - composited/                   (832x480 intermediate)

    Hard-stops (no destructive action taken):
      - src is not under otr/episodes/                  (legacy flat / bypass)
      - ep_dir is not a direct child of otr/episodes/   (depth-1 invariant)
      - no *_ledger.json found in <ep>/audio/           (nothing to read flag from)
      - perfect_run_spacesaver flag is False            (not opted in)
      - obs/<ep>.mp4 not on disk                        (final hasn't landed yet)
    """
    # Derive the episode dir from src. src is the composited mp4:
    #   otr/episodes/<ep>/composited/<ep>.mp4
    # so src.parent.parent is the episode root. No mtime guessing.
    try:
        from _otr_paths import otr_episodes_root
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "[OTR_RTXUpscale] spacesaver: path import failed (%s); skipping",
            exc,
        )
        return

    ep_dir = src.resolve().parent.parent
    audio_dir = ep_dir / "audio"

    # Sanity guard: ep_dir MUST be a direct child of otr/episodes/.
    # Anything else (legacy flat layout, bypass with a non-OTR source,
    # user-typed path outside the workspace) bails with no destructive
    # action. Replaces the prior substring-only check that could be
    # tricked by user-named folders.
    try:
        rel = ep_dir.relative_to(otr_episodes_root().resolve())
    except ValueError:
        log.warning(
            "[OTR_RTXUpscale] spacesaver: src %s is not under otr/episodes/; "
            "refusing destructive cleanup",
            src,
        )
        return
    if len(rel.parts) != 1:
        log.warning(
            "[OTR_RTXUpscale] spacesaver: ep_dir %s is not a direct child of "
            "otr/episodes/ (got depth %d); refusing cleanup",
            ep_dir, len(rel.parts),
        )
        return

    # Load the ledger from THIS episode's audio dir, not a global scan.
    # Multiple ledgers in one episode dir means a stale pending_ wasn't
    # cleaned up -- prefer the finalized (non-pending) one; if all are
    # pending, use newest.
    ledger_candidates = sorted(audio_dir.glob("*_ledger.json"))
    if not ledger_candidates:
        log.debug(
            "[OTR_RTXUpscale] spacesaver: no ledger in %s; skipping",
            audio_dir,
        )
        return
    finalized = [p for p in ledger_candidates if not p.name.startswith("pending_")]
    ledger_path = finalized[-1] if finalized else ledger_candidates[-1]

    import json as _json
    try:
        led = _json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_RTXUpscale] spacesaver: ledger parse failed (%s); skipping",
            exc,
        )
        return
    meta = led.get("meta") or {}
    flag = bool(meta.get("perfect_run_spacesaver"))
    if not flag:
        log.debug("[OTR_RTXUpscale] spacesaver flag OFF; no cleanup")
        return

    ep_id = led.get("episode_id") or ep_dir.name

    # OBS-existence precondition: spacesaver MUST NOT wipe intermediates
    # if the final deliverable hasn't landed on disk yet. The OBS final
    # is now ``<ep>_procgen_blended.mp4`` (BUG-LOCAL-108 path cleanup
    # 2026-05-05); written by OTR_PostUpscaleProcgenBlend which runs
    # AFTER RTXUpscale. While this cleanup is invoked from RTXUpscale
    # the procgen blend hasn't happened yet, so this guard will fail-
    # closed and spacesaver becomes a silent no-op until the call site
    # is moved to OTR_PostUpscaleProcgenBlend (planned followup).
    # Keeping the path string accurate so when the move lands the
    # guard already points at the right artifact.
    obs_final = ep_dir.parent.parent / "obs" / f"{ep_id}_procgen_blended.mp4"
    if not obs_final.exists():
        log.warning(
            "[OTR_RTXUpscale] spacesaver: OBS deliverable %s not on disk; "
            "refusing cleanup until final lands (expected -- spacesaver "
            "now lives in the wrong stage; will no-op until call site "
            "moves to PostUpscaleProcgenBlend)",
            obs_final,
        )
        return

    # Keep-list built from REAL filenames on disk, not slug-reconstructed
    # from ep_id. Avoids the slug-mismatch trap where ep_id was slugged
    # differently than the on-disk file name (Finding 4 in QA pass).
    keep_files = {ledger_path}
    for tx in audio_dir.glob("*_treatment.txt"):
        keep_files.add(tx)
    deleted_files = 0
    deleted_bytes = 0
    deleted_dirs = 0

    # Walk audio/: keep only the ledger + treatment.
    if audio_dir.exists():
        for child in audio_dir.iterdir():
            if child in keep_files:
                continue
            try:
                if child.is_file():
                    deleted_bytes += child.stat().st_size
                    child.unlink()
                    deleted_files += 1
                elif child.is_dir():
                    import shutil as _shutil
                    deleted_bytes += sum(
                        f.stat().st_size for f in child.rglob('*') if f.is_file()
                    )
                    _shutil.rmtree(child, ignore_errors=True)
                    deleted_dirs += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_RTXUpscale] spacesaver: delete %s failed: %s",
                    child, exc,
                )

    # Wipe sibling subdirs entirely: stills/, portraits/, videos/,
    # composited/, upscaled/. ``upscaled/`` added 2026-05-05 (BUG-LOCAL-108)
    # alongside the obs/ contract cleanup -- it's an intermediate too
    # once the final blend is in obs/.
    for sub_name in ("stills", "portraits", "videos", "composited", "upscaled"):
        sub = ep_dir / sub_name
        if not sub.exists():
            continue
        try:
            import shutil as _shutil
            deleted_bytes += sum(
                f.stat().st_size for f in sub.rglob('*') if f.is_file()
            )
            _shutil.rmtree(sub, ignore_errors=True)
            deleted_dirs += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[OTR_RTXUpscale] spacesaver: rmtree %s failed: %s",
                sub, exc,
            )

    mb = deleted_bytes / (1024 * 1024)
    msg = (
        f"  spacesaver: wiped {deleted_files} file(s) + {deleted_dirs} dir(s) "
        f"({mb:.1f} MB freed) under {ep_dir.name}/ (kept ledger + treatment)"
    )
    log_lines.append(msg)
    log.info("[OTR_RTXUpscale] %s", msg.lstrip())


__all__ = ["RTXUpscale"]
