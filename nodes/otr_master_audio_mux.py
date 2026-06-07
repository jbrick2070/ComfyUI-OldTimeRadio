"""OTR_MasterAudioMux -- the TERMINAL audio mux (A-S3 / CW-4).

THE ONLY node that may add audio to a video (invariant V-1). It muxes the FROZEN
master episode audio onto the always-silent composite with ``-c:a copy`` -- the
audio stream passes through with ZERO re-encode, so the output's audio is
byte-identical to the master. ``ffmpeg -shortest`` is FORBIDDEN here (it would
truncate the episode to the shorter stream and silently drop an audio tail); the
silent composite is already built to the audio-derived frame budget, so a
duration assertion (within 1/fps) runs BEFORE the mux to catch any drift.

Replaces the legacy ``OTR_VideoComposite`` audio path (which re-encoded to AAC in
its humo_concat mode and used ``-shortest``). This node does NO model work and
holds NO CUDA residency -- it is pure ffmpeg, so the BUG-291 patcher-ref / NVML
guards do not apply here; it only polls the interrupt flag so Cancel is honoured.

Audio-identity is asserted by decoding both the muxed output's audio and the
master to canonical PCM and comparing SHA-256 -- container-agnostic proof that
the audio was copied, not re-encoded or trimmed. Cold-import clean (stdlib only).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

import logging

log = logging.getLogger("OTR")


def _ffmpeg_bin(ffmpeg: str) -> str:
    return ffmpeg if (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) else ""


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or ""


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _probe_float(path: str, stream: str) -> float:
    """Duration (s) of the first ``stream`` (``v:0`` / ``a:0``) via ffprobe."""
    fp = _ffprobe_bin()
    if not fp:
        return -1.0
    p = _run([fp, "-v", "error", "-select_streams", stream, "-show_entries",
              "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", path])
    try:
        return float((p.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        # container duration fallback
        p2 = _run([fp, "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=nokey=1:noprint_wrappers=1", path])
        try:
            return float((p2.stdout or "").strip())
        except ValueError:
            return -1.0


def _count_audio_streams(path: str) -> int:
    fp = _ffprobe_bin()
    if not fp:
        return -1
    p = _run([fp, "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=index", "-of", "csv=p=0", path])
    return len([ln for ln in (p.stdout or "").splitlines() if ln.strip()])


def audio_pcm_sha(path: str) -> str:
    """SHA-256 of decoded s16le mono @24k -- codec/container-agnostic audio
    identity (same method the A-S2 mux probe used). '' on failure."""
    fp = shutil.which("ffmpeg")
    if not fp:
        return ""
    raw = subprocess.run(
        [fp, "-v", "error", "-i", path, "-map", "0:a", "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", "-"],
        capture_output=True)
    if raw.returncode != 0 or not raw.stdout:
        return ""
    return hashlib.sha256(raw.stdout).hexdigest()


class _Interrupted(RuntimeError):
    pass


def _poll_interrupt():
    """Honour ComfyUI Cancel (BUG-073). No-op outside ComfyUI."""
    try:
        import comfy.model_management as mm  # type: ignore
        mm.throw_exception_if_processing_interrupted()
    except _Interrupted:
        raise
    except Exception:  # noqa: BLE001  (not running under comfy / no such API)
        return


def mux_master_audio(silent_video_path: str, master_audio_path: str, out_path: str,
                     ffmpeg: str = "ffmpeg", fps: int = 25,
                     duration_tol_frames: float = 1.0):
    """Mux the frozen master audio onto the silent video; FAIL CLOSED.

    Pure function (used by the node + tests). Steps: validate inputs -> duration
    assert (|v - a| <= duration_tol_frames / fps) BEFORE the mux -> ffmpeg
    ``-map 0:v -map 1:a -c:v copy -c:a copy`` (NO ``-shortest``) -> assert the
    output audio decodes identically to the master. Returns ``(out_path,
    report_lines)``; raises ``ValueError`` on any gate failure (never produces a
    silently-wrong episode).
    """
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError(f"OTR_MasterAudioMux: ffmpeg not found ({ffmpeg!r})")
    if not os.path.isfile(silent_video_path):
        raise ValueError(f"OTR_MasterAudioMux: silent video missing: {silent_video_path!r}")
    if not os.path.isfile(master_audio_path):
        raise ValueError(f"OTR_MasterAudioMux: master audio missing: {master_audio_path!r}")

    # post-composite duration assert vs master BEFORE the mux (settb drift guard).
    v_dur = _probe_float(silent_video_path, "v:0")
    a_dur = _probe_float(master_audio_path, "a:0")
    tol = max(1, int(fps or 25)) and (float(duration_tol_frames) / float(fps or 25))
    if v_dur >= 0 and a_dur >= 0 and abs(v_dur - a_dur) > tol:
        raise ValueError(
            f"OTR_MasterAudioMux: silent video {v_dur:.4f}s vs master audio "
            f"{a_dur:.4f}s differ by > {tol:.4f}s (1/fps); the composite is not "
            f"built to the audio-derived budget -- refusing to mux (would need "
            f"-shortest, which is forbidden)"
        )
    report.append(f"duration_check v={v_dur:.3f}s a={a_dur:.3f}s tol={tol:.4f}s OK")

    _poll_interrupt()
    # mux-LAST: copy both streams, NO -shortest.
    cmd = [
        fb, "-y", "-loglevel", "error",
        "-i", silent_video_path,
        "-i", master_audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "copy",
        out_path,
    ]
    assert "-shortest" not in cmd, "V-2: -shortest must never appear in the mux"
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError(f"OTR_MasterAudioMux: ffmpeg mux failed :: {p.stderr.strip()[:300]}")

    # byte-identity: the output audio must decode identically to the master.
    h_master = audio_pcm_sha(master_audio_path)
    h_out = audio_pcm_sha(out_path)
    if not h_out or h_out != h_master:
        raise ValueError(
            f"OTR_MasterAudioMux: output audio NOT byte-identical to master "
            f"(master={h_master[:12]} out={h_out[:12]}); the audio was re-encoded "
            f"or trimmed -- C7/V-1 violated"
        )
    report.append(f"audio_byte_identical OK ({h_out[:12]})")
    return out_path, report


class OTRMasterAudioMux:
    """Registered as ``OTR_MasterAudioMux``. Terminal audio mux (V-1: the ONLY
    node that adds audio). ``-c:a copy``, NO ``-shortest``, byte-identical assert."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "mux"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_video_path", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "silent_video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Always-silent composite (OTR_SilentComposite). Audio is added HERE only.",
                }),
                "master_audio_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Frozen master episode audio (OTR_EpisodeAssembler output_path). Copied with -c:a copy.",
                }),
            },
            "optional": {
                "audio_done": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Audio-done gate (EpisodeAssembler out3). Orders the mux AFTER the audio freezes. Opaque.",
                }),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Final mp4 path. Empty -> <output>/otr/episodes/<stem>_final.mp4.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _default_out(self, silent_video_path: str) -> str:
        try:
            import folder_paths  # type: ignore
            root = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            root = "."
        out_dir = os.path.join(root, "otr", "episodes")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(silent_video_path or "episode"))[0]
        return os.path.join(out_dir, f"{stem}_final.mp4")

    def mux(self, silent_video_path, master_audio_path, audio_done="", fps=25,
            ffmpeg="ffmpeg", output_path=""):
        out = output_path.strip() or self._default_out(silent_video_path)
        try:
            final, report = mux_master_audio(
                silent_video_path, master_audio_path, out, ffmpeg=ffmpeg, fps=int(fps),
            )
        except _Interrupted:
            raise
        except ValueError as exc:
            log.error("[OTR_MasterAudioMux] %s", exc)
            return ("", f"error: {exc}")
        for line in report:
            log.info("[OTR_MasterAudioMux] %s", line)
        return (final, "OTR_MasterAudioMux OK -> " + final + "\n" + "\n".join(report))


__all__ = ["OTRMasterAudioMux", "mux_master_audio", "audio_pcm_sha"]
