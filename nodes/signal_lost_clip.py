"""
SignalLostClip — FFmpeg-loop the pre-baked fallback clip to a target duration.

Never loads any model. This is the fallback path when OOM or generation
failure prevents a real animated clip from being produced.

Preflight: if the asset file is missing, the run must hard-fail at startup.
"""

import logging
import os
import subprocess

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def signal_lost_clip(duration_s: float, out_path: str,
                     asset_path: str = None) -> str:
    """Loop the pre-baked signal-lost clip to the requested duration.

    Args:
        duration_s: Target duration in seconds.
        out_path: Destination .mp4 path.
        asset_path: Path to the source signal_lost_prerender.mp4.
                    Defaults to assets/signal_lost_prerender.mp4 in repo root.

    Returns:
        out_path on success.

    Raises:
        FileNotFoundError: If the asset file does not exist.
        RuntimeError: If FFmpeg fails.
    """
    if asset_path is None:
        asset_path = os.path.join(_REPO_ROOT, "assets", "signal_lost_prerender.mp4")

    if not os.path.isfile(asset_path):
        raise FileNotFoundError(
            f"Fallback asset missing: {asset_path}. "
            f"Run preflight.py or place the file manually."
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", asset_path,
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        out_path,
    ]

    log.info("[SignalLostClip] Generating %.1fs fallback clip -> %s",
             duration_s, os.path.basename(out_path))

    result = subprocess.run(
        ffmpeg_cmd, capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (rc={result.returncode}): {result.stderr[-300:]}"
        )

    # Verify output exists and is reasonable size
    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError(f"FFmpeg produced empty/missing output: {out_path}")

    log.info("[SignalLostClip] Fallback clip ready: %s (%.0f KB)",
             os.path.basename(out_path), os.path.getsize(out_path) / 1024)

    return out_path
