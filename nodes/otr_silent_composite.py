"""OTR_SilentComposite -- normalize render output into ONE always-silent video (A-S3 / CW-4).

Produces the canonical, ALWAYS-SILENT composite the terminal ``OTR_MasterAudioMux``
then muxes the frozen master audio onto. For M1 (first watchable episode) the base
is the radio-floor video (``OTR_SignalLostVideo``); later windows composite real
engine clips here too. Whatever the source, the output is guaranteed:

* **silent** -- ``-an`` strips any audio (invariant V-1: only MasterAudioMux adds audio),
* **CFR** at the canonical fps (``fps`` filter + ``-vsync cfr``; no VFR drift),
* **yuv420p**, even (mod-2) dimensions, padded to the canonical canvas,
* **bt709 IDENTITY-tagged** -- untagged input is TAGGED bt709, NEVER matrix-converted
  (no silent BT.601->709 shift); the scale/pad filters do not touch the color matrix,
* duration preserved from the (audio-derived) base, so the mux duration assert passes.

Pure ffmpeg, cold-import clean (stdlib only) -- no torch, no CUDA residency.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import logging

log = logging.getLogger("OTR")


def _ffmpeg_bin(ffmpeg: str) -> str:
    return ffmpeg if (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) else ""


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or ""


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def count_audio_streams(path: str) -> int:
    fp = _ffprobe_bin()
    if not fp:
        return -1
    p = _run([fp, "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=index", "-of", "csv=p=0", path])
    return len([ln for ln in (p.stdout or "").splitlines() if ln.strip()])


def probe_video(path: str) -> dict:
    """w,h,pix_fmt,avg_frame_rate of v:0 (for the CFR / color / mod-2 asserts)."""
    fp = _ffprobe_bin()
    if not fp:
        return {}
    p = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate",
              "-of", "default=noprint_wrappers=1", path])
    out: dict = {}
    for line in (p.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _even(n: int) -> int:
    n = int(n)
    return n if n % 2 == 0 else n - 1


def normalize_to_silent_canonical(in_path: str, out_path: str, *, w: int = 1472,
                                  h: int = 832, fps: int = 25, ffmpeg: str = "ffmpeg"):
    """Re-encode ``in_path`` into the canonical ALWAYS-SILENT clip; FAIL CLOSED.

    Returns ``(out_path, report)``; raises ``ValueError`` on a gate failure.
    """
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError(f"OTR_SilentComposite: ffmpeg not found ({ffmpeg!r})")
    if not os.path.isfile(in_path):
        raise ValueError(f"OTR_SilentComposite: input missing: {in_path!r}")
    w, h, fps = _even(w), _even(h), max(1, int(fps))
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps}"
    )
    cmd = [
        fb, "-y", "-loglevel", "error",
        "-i", in_path,
        "-an",                                  # V-1: strip ALL audio
        "-vf", vf,
        "-vsync", "cfr",                        # constant frame rate (no VFR drift)
        "-pix_fmt", "yuv420p",
        # TAG bt709 (identity) -- do NOT matrix-convert an untagged source.
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        out_path,
    ]
    assert "-shortest" not in cmd, "V-2: -shortest must not appear in the composite"
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError(f"OTR_SilentComposite: ffmpeg failed :: {p.stderr.strip()[:300]}")

    # gate: the composite MUST be silent (V-1) + yuv420p + even dims.
    na = count_audio_streams(out_path)
    if na != 0:
        raise ValueError(f"OTR_SilentComposite: output has {na} audio stream(s); must be 0 (V-1)")
    info = probe_video(out_path)
    if info.get("pix_fmt") and info["pix_fmt"] != "yuv420p":
        raise ValueError(f"OTR_SilentComposite: pix_fmt {info['pix_fmt']} != yuv420p")
    try:
        ow, oh = int(info.get("width", w)), int(info.get("height", h))
        if ow % 2 or oh % 2:
            raise ValueError(f"OTR_SilentComposite: non-mod-2 output dims {ow}x{oh}")
    except ValueError:
        raise
    report.append(f"silent_canonical {info.get('width', w)}x{info.get('height', h)} "
                  f"yuv420p bt709 cfr@{fps} audio_streams=0 OK")
    return out_path, report


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
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _default_out(self, base_video_path: str) -> str:
        try:
            import folder_paths  # type: ignore
            root = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            root = "."
        out_dir = os.path.join(root, "otr", "episodes")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(base_video_path or "episode"))[0]
        return os.path.join(out_dir, f"{stem}_silent.mp4")

    def composite(self, base_video_path, canvas_w=1472, canvas_h=832, fps=25,
                  ffmpeg="ffmpeg", output_path="", gate_in=""):
        out = output_path.strip() or self._default_out(base_video_path)
        try:
            silent, report = normalize_to_silent_canonical(
                base_video_path, out, w=int(canvas_w), h=int(canvas_h),
                fps=int(fps), ffmpeg=ffmpeg,
            )
        except ValueError as exc:
            log.error("[OTR_SilentComposite] %s", exc)
            return ("", f"error: {exc}")
        for line in report:
            log.info("[OTR_SilentComposite] %s", line)
        return (silent, "OTR_SilentComposite OK -> " + silent + "\n" + "\n".join(report))


__all__ = ["OTRSilentComposite", "normalize_to_silent_canonical",
           "count_audio_streams", "probe_video"]
