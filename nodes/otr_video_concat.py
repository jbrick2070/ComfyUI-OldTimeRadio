"""
nodes.otr_video_concat  --  seamless video-clip concatenation for OTR
=====================================================================

MIT-licensed, OTR-original.  Replaces the need to vendor
``Kosinkadink/ComfyUI-VideoHelperSuite`` (GPL-3.0) just for the
``VHS_VideoCombine`` node.

Purpose
-------
Concat N .mp4 clips into a single .mp4 whose duration matches the
sum of its parts.  Designed to live downstream of a batched video
backend (LTX I2V, Wan 2.2 FLF2V, etc.) that produces one clip per
shot.  When all input clips share the same codec / resolution /
framerate (the common case when they come from the same backend),
ffmpeg's concat demuxer stitches them without re-encoding -- fast
and lossless.

C7 invariant
------------
Audio streams pass through byte-identical via ``-c:a copy -map
0:a?``.  This node does not touch audio.  The OTR pipeline muxes
the Bark episode audio downstream; this stage usually operates on
video-only inputs anyway, but the C7 guarantee holds either way.

First-last-frame compatibility
------------------------------
When upstream clips come from an FLF video engine (each clip ends
on a frame that the next clip begins on), hard-cut concat is
visually seamless by construction -- no crossfade needed.  A
``crossfade_ms`` parameter exists for non-FLF sources but defaults
to 0.

Node inputs
-----------
* ``input_paths`` (STRING, multiline): newline-separated absolute
  paths to input .mp4 files, in concat order.  Empty lines and
  lines starting with ``#`` are ignored.
* ``output_path`` (STRING): absolute path for the final .mp4.
* ``reencode`` (BOOLEAN, default False): when False, use
  ``-c copy`` (fast, lossless, requires matching codecs). When
  True, force re-encode to h264 yuv420p + AAC (safe for mixed
  sources, slight quality loss).
* ``crossfade_ms`` (INT, default 0): reserved for future xfade
  work.  Ignored by the MVP implementation.

Returns
-------
* ``video_path`` (STRING): absolute path to the produced .mp4
  (same as ``output_path`` on success).

Stub mode
---------
When ffmpeg is not locatable (``OTR_VIDEO_CONCAT_STUB=1`` or
``find_ffmpeg()`` returns None), the node copies the first input
to the output as a passthrough.  Lets CI and weight-missing dev
machines exercise the pipeline without ffmpeg.

License
-------
MIT.  Part of ComfyUI-OldTimeRadio.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("OTR.nodes.otr_video_concat")


# ---------------------------------------------------------------------------
# ffmpeg discovery (shared pattern with visual/postproc/vhs.py)
# ---------------------------------------------------------------------------

_FFMPEG_CANDIDATES: tuple[str, ...] = ("ffmpeg", "ffmpeg.exe")


def find_ffmpeg() -> str | None:
    """Locate ffmpeg on PATH.  Returns the resolved path or None."""
    for candidate in _FFMPEG_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without ComfyUI)
# ---------------------------------------------------------------------------


def parse_input_paths(raw: str) -> list[Path]:
    """Turn a multiline STRING widget value into an ordered list of Paths.

    Empty lines and lines starting with ``#`` are skipped so users can
    annotate the input list.  Leading/trailing whitespace on each
    line is trimmed.  Windows backslashes and forward slashes are both
    accepted.
    """
    if not raw:
        return []
    paths: list[Path] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(Path(stripped))
    return paths


def write_concat_filelist(paths: list[Path], list_path: Path) -> None:
    """Write an ffmpeg concat-demuxer filelist at ``list_path``.

    Format:
        file 'C:/abs/path/to/clip_00.mp4'
        file 'C:/abs/path/to/clip_01.mp4'

    Uses forward slashes (ffmpeg accepts them on Windows) and
    single-quotes to survive spaces.  Each input path is shell-escaped
    minimally by doubling single-quotes inside the path (ffmpeg concat
    demuxer convention).
    """
    lines: list[str] = []
    for p in paths:
        # ffmpeg concat demuxer escaping: double any literal single quote
        escaped = str(p).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_concat_argv(
    ffmpeg_path: str,
    filelist_path: Path,
    output_path: Path,
    *,
    reencode: bool,
) -> list[str]:
    """Build the ffmpeg argv for concat.  Pure; unit-testable.

    ``-f concat -safe 0`` uses the concat demuxer with absolute paths
    allowed.  ``-c copy`` = lossless stream copy (requires uniform
    codec/res/fps across inputs).  ``-map 0:a?`` is a safety net so a
    missing audio stream doesn't break the mux.
    """
    argv: list[str] = [
        ffmpeg_path,
        "-y",                       # overwrite output
        "-f", "concat",
        "-safe", "0",
        "-i", str(filelist_path),
    ]
    if reencode:
        argv += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
        ]
    else:
        argv += ["-c", "copy"]
    argv += ["-map", "0:v?", "-map", "0:a?", str(output_path)]
    return argv


# ---------------------------------------------------------------------------
# Core concat function (unit-testable without ComfyUI)
# ---------------------------------------------------------------------------


def concat_videos(
    input_paths: list[Path],
    output_path: Path,
    *,
    reencode: bool = False,
    force_stub: bool = False,
    ffmpeg_path: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Concat the given clips into ``output_path``.  Returns a result dict.

    Result keys:
        ``mode``        -- "real" | "stub"
        ``stub_reason`` -- str or None
        ``output``      -- str absolute path
        ``argv``        -- list[str] ffmpeg argv used (real mode only)
        ``clip_count``  -- int
        ``returncode``  -- int (0 on success)
    """
    if not input_paths:
        raise ValueError("otr_video_concat: input_paths is empty")
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"otr_video_concat: missing input {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stub-mode resolution
    stub_reason: str | None = None
    if force_stub:
        stub_reason = "force_stub=True"
    elif os.environ.get("OTR_VIDEO_CONCAT_STUB") == "1":
        stub_reason = "env OTR_VIDEO_CONCAT_STUB=1"
    else:
        ff = ffmpeg_path or find_ffmpeg()
        if ff is None:
            stub_reason = "ffmpeg not found"
        else:
            ffmpeg_path = ff

    if stub_reason is not None:
        shutil.copyfile(input_paths[0], output_path)
        return {
            "mode": "stub",
            "stub_reason": stub_reason,
            "output": str(output_path),
            "argv": None,
            "clip_count": len(input_paths),
            "returncode": 0,
        }

    # Real mode: build filelist, run ffmpeg
    filelist_path = output_path.with_suffix(".concat.txt")
    write_concat_filelist(input_paths, filelist_path)
    argv = build_concat_argv(
        ffmpeg_path,                # type: ignore[arg-type]
        filelist_path,
        output_path,
        reencode=reencode,
    )
    log.info("otr_video_concat: running %s", " ".join(argv))
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    try:
        filelist_path.unlink()
    except OSError:
        pass

    if proc.returncode != 0:
        log.error(
            "otr_video_concat FAILED (%d): %s",
            proc.returncode,
            proc.stderr[-2000:] if proc.stderr else "(no stderr)",
        )
        # Graceful fallback: if we were in fast-copy mode and it
        # failed (most likely cause: codec mismatch), retry with
        # re-encode so the user gets output rather than an error.
        if not reencode:
            log.warning(
                "otr_video_concat: retrying with reencode=True"
            )
            return concat_videos(
                input_paths,
                output_path,
                reencode=True,
                force_stub=False,
                ffmpeg_path=ffmpeg_path,
                timeout_s=timeout_s,
            )
        raise RuntimeError(
            f"ffmpeg concat failed (rc={proc.returncode}): "
            f"{proc.stderr[-500:] if proc.stderr else 'no stderr'}"
        )

    return {
        "mode": "real",
        "stub_reason": None,
        "output": str(output_path),
        "argv": argv,
        "clip_count": len(input_paths),
        "returncode": 0,
    }


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------


class OTRVideoConcat:
    """OTR_VideoConcat  --  ffmpeg-based seamless clip concatenation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_paths": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# One .mp4 path per line, in concat order.\n"
                            "# Lines starting with '#' are ignored.\n"
                        ),
                    },
                ),
                "output_path": (
                    "STRING",
                    {"default": ""},
                ),
                "reencode": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
            "optional": {
                "crossfade_ms": (
                    "INT",
                    {"default": 0, "min": 0, "max": 5000, "step": 50},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "concat"
    CATEGORY = "OldTimeRadio/video"

    def concat(
        self,
        input_paths: str,
        output_path: str,
        reencode: bool,
        crossfade_ms: int = 0,
    ):
        if crossfade_ms:
            log.warning(
                "OTR_VideoConcat: crossfade_ms=%d ignored in MVP "
                "(FLF chains do not need crossfade)",
                crossfade_ms,
            )
        paths = parse_input_paths(input_paths)
        if not paths:
            raise ValueError(
                "OTR_VideoConcat: no input paths provided (check "
                "'input_paths' widget -- one path per line, # for "
                "comments)"
            )
        if not output_path:
            # Default to first-input-dir / "concat.mp4"
            output = paths[0].parent / "concat.mp4"
        else:
            output = Path(output_path)

        result = concat_videos(paths, output, reencode=reencode)
        log.info(
            "OTR_VideoConcat READY: %s (%d clips, mode=%s)",
            result["output"],
            result["clip_count"],
            result["mode"],
        )
        return (result["output"],)


# ---------------------------------------------------------------------------
# Module-level registration hook (re-exported by package __init__.py)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "OTR_VideoConcat": OTRVideoConcat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_VideoConcat": " OTR Video Concat",
}

__all__ = [
    "OTRVideoConcat",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "concat_videos",
    "parse_input_paths",
    "write_concat_filelist",
    "build_concat_argv",
    "find_ffmpeg",
]
