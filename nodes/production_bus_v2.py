"""
ProductionBus v2 — Scene-level orchestrator for keyframes and animated modes.

Keyframes mode: FFmpeg static clip from anchor, length = audio_duration_s.
Animated mode: delegate to SceneAnimator, then reconcile duration per config.

Drift handling: if abs(actual - target) > av_drift_fail_ms -> fallback clip.
Fallback budget: if fallback_count > max_per_episode -> EpisodeFallbackBudgetExceeded.
"""

import json
import logging
import os
import subprocess
from typing import Optional

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EpisodeFallbackBudgetExceeded(RuntimeError):
    """Raised when too many scenes fell back to static clips."""
    pass


class ProductionBusV2:
    """Scene-level render orchestrator.

    Args:
        config: Parsed rtx5080.yaml dict.
        mode: 'keyframes' or 'animated'.
    """

    def __init__(self, config: dict, mode: str = "keyframes"):
        self.config = config
        self.mode = mode
        self.fallback_count = 0
        self.max_fallbacks = config.get("fallback", {}).get("max_per_episode", 2)
        self.av_tolerance_ms = config.get("reconcile", {}).get("av_tolerance_ms", 150)
        self.av_drift_fail_ms = config.get("reconcile", {}).get("av_drift_fail_ms", 500)
        self.av_policy = config.get("reconcile", {}).get(
            "av_duration_policy", "hold_last_frame")
        self.ffmpeg_timeout = config.get("ffmpeg", {}).get("concat_timeout_s", 120)
        self.static_timeout = config.get("ffmpeg", {}).get("static_clip_timeout_s", 30)

    def render_scene(self, prompt, anchor_path: str,
                     audio_duration_s: float) -> str:
        """Render one scene. Returns path to the clip MP4.

        Args:
            prompt: ScenePrompt dataclass instance.
            anchor_path: Path to the SD3.5 anchor image for this scene.
            audio_duration_s: Target duration from the audio timeline.

        Returns:
            Path to the rendered MP4 clip.
        """
        scene_id = prompt.scene_id
        out_dir = os.path.join(_REPO_ROOT, "output", "v2_clips")
        os.makedirs(out_dir, exist_ok=True)
        clip_path = os.path.join(out_dir, f"scene_{scene_id}.mp4")

        if self.mode == "keyframes":
            return self._render_keyframe(anchor_path, clip_path, audio_duration_s)

        elif self.mode == "animated":
            # Animated mode delegates to SceneAnimator (called externally).
            # This method handles duration reconciliation after the clip exists.
            if os.path.isfile(clip_path):
                actual_s = _probe_duration(clip_path)
                drift_ms = abs(actual_s - audio_duration_s) * 1000

                if drift_ms > self.av_drift_fail_ms:
                    log.warning(
                        "[ProductionBus] Scene %s drift %.0fms > %dms — fallback",
                        scene_id, drift_ms, self.av_drift_fail_ms
                    )
                    return self._fallback(clip_path, audio_duration_s, scene_id)

                if drift_ms > self.av_tolerance_ms:
                    clip_path = self._reconcile_duration(
                        clip_path, audio_duration_s, scene_id
                    )

            return clip_path

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def concat(self, clip_paths: list, out_path: str) -> str:
        """Concatenate scene clips into final video.

        Args:
            clip_paths: Ordered list of MP4 clip paths.
            out_path: Destination path for the concatenated video.

        Returns:
            out_path on success.
        """
        if not clip_paths:
            raise ValueError("No clip paths to concatenate")

        valid = [p for p in clip_paths if os.path.isfile(p)]
        if not valid:
            raise FileNotFoundError("None of the clip paths exist on disk")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Build concat file
        concat_path = out_path + ".concat.txt"
        with open(concat_path, "w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for clip in valid:
                safe = clip.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            out_path,
        ]

        result = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True,
            timeout=self.ffmpeg_timeout
        )

        # Cleanup concat file
        try:
            os.remove(concat_path)
        except OSError:
            pass

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed (rc={result.returncode}): "
                f"{result.stderr[-300:]}"
            )

        log.info("[ProductionBus] Concatenated %d clips -> %s",
                 len(valid), os.path.basename(out_path))
        return out_path

    def _render_keyframe(self, anchor_path: str, clip_path: str,
                         duration_s: float) -> str:
        """Create a static clip from an anchor image held for duration_s."""
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", anchor_path,
            "-t", f"{duration_s:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-r", "24",
            "-an",
            clip_path,
        ]

        result = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True,
            timeout=self.static_timeout
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Keyframe render failed: {result.stderr[-300:]}"
            )

        return clip_path

    def _reconcile_duration(self, clip_path: str, target_s: float,
                            scene_id: str) -> str:
        """Adjust clip duration to match audio timeline."""
        policy = self.av_policy

        if policy == "hold_last_frame":
            # Re-encode with -t to extend/trim
            tmp = clip_path + ".reconciled.mp4"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", clip_path,
                "-t", f"{target_s:.3f}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                tmp,
            ]
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                os.replace(tmp, clip_path)
                log.info("[ProductionBus] Scene %s reconciled to %.1fs (hold_last_frame)",
                         scene_id, target_s)
            else:
                log.warning("[ProductionBus] Reconcile failed for %s, keeping original",
                            scene_id)
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        # Other policies (loop, crossfade_black, timestretch) are v2.1+
        return clip_path

    def _fallback(self, clip_path: str, duration_s: float,
                  scene_id: str) -> str:
        """Replace a failed scene with the fallback clip."""
        from .signal_lost_clip import signal_lost_clip

        self.fallback_count += 1
        if self.fallback_count > self.max_fallbacks:
            raise EpisodeFallbackBudgetExceeded(
                f"Scene {scene_id}: {self.fallback_count} fallbacks > "
                f"max {self.max_fallbacks}"
            )

        log.warning("[ProductionBus] Scene %s -> fallback (%d/%d)",
                    scene_id, self.fallback_count, self.max_fallbacks)

        return signal_lost_clip(duration_s, clip_path)


def _probe_duration(path: str) -> float:
    """Get duration of an MP4 in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
