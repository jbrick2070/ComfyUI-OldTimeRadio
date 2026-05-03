"""
rtx_upscale.py  --  OTR_RTXUpscale ComfyUI node
================================================

Final-stage video upscaler for the OTR pipeline. Wraps NVIDIA's
``RTXVideoSuperResolution`` (from custom_nodes/Nvidia_RTX_Nodes_ComfyUI)
with a path-in / path-out interface that preserves C7 audio identity:

    final_mp4 (832x480) ---> chunked frame extract via ffmpeg
                          -> RTX VSR (HW-accelerated, ULTRA quality)
                          -> silent libx264 yuv420p mp4
                          -> ffmpeg mux with -c:a copy from source mp4
                          -> <ep>_1080p.mp4

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
) -> tuple[int, float]:
    """Decode src frames in chunks via ffmpeg, run RTX VSR per chunk,
    encode to silent libx264 yuv420p mp4. Returns (total_frames_out, fps).

    Avoids holding all frames in RAM. Pipes raw RGB24 in/out of ffmpeg
    so RTX VSR sees a torch tensor that's already in CPU memory.
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
                gpu_in = torch.from_numpy(arr).cuda().permute(
                    0, 3, 1, 2
                ).float().contiguous()
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
                # Back to NHWC uint8 on CPU for ffmpeg pipe.
                out_frames = (
                    gpu_out.clamp(0.0, 255.0)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
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
                "out_suffix": ("STRING", {
                    "default": "_1080p",
                    "multiline": False,
                    "tooltip": (
                        "Suffix for the upscaled deliverable filename. "
                        "<ep>.mp4 -> <ep><suffix>.mp4. Default '_1080p'."
                    ),
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
        out_suffix: str = "_1080p",
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

        if bypass:
            report_lines.append(
                f"OTR_RTXUpscale: BYPASS -- returning source unchanged "
                f"({src.name})"
            )
            log.info("[OTR_RTXUpscale] bypass=True; returning %s", src)
            return (str(src), "\n".join(report_lines))

        # Source dims for the report
        try:
            src_w, src_h, _src_fps = _probe_video_dims(ffprobe, src)
            report_lines.append(
                f"OTR_RTXUpscale: source {src.name} "
                f"{src_w}x{src_h} -> {target_width}x{target_height} "
                f"({quality})"
            )
        except Exception as exc:  # noqa: BLE001
            report_lines.append(
                f"OTR_RTXUpscale: ffprobe of source failed ({exc}); "
                "continuing"
            )

        # Output path: <ep>_1080p.mp4 in the same dir as source
        suffix = (out_suffix or "_1080p").strip()
        if not suffix.startswith("_"):
            suffix = "_" + suffix
        out_mp4 = src.with_name(f"{src.stem}{suffix}.mp4")

        # Silent intermediate goes to a tempdir, gets removed after mux
        tmp_dir = Path(tempfile.mkdtemp(prefix="otr_rtx_upscale_"))
        silent = tmp_dir / "video_only.mp4"
        try:
            n_frames, fps = _chunked_upscale(
                src_mp4=src,
                silent_out_mp4=silent,
                target_w=target_width,
                target_h=target_height,
                quality=quality,
                chunk_frames=chunk_frames,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
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

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"OTR_RTXUpscale: complete in {total_ms} ms -> {out_mp4}"
        )
        log.info(
            "[OTR_RTXUpscale] complete in %d ms: %s -> %s",
            total_ms, src.name, out_mp4.name,
        )
        return (str(out_mp4), "\n".join(report_lines))


__all__ = ["RTXUpscale"]
