"""OTR_CaptionBurn -- SDH open-caption burn for the NEW render path (CW-4 migration).

The dedicated home for the SDH caption burn that used to live inside the legacy
``OTR_PostUpscaleProcgenBlend`` (now being torn out). Sits BETWEEN
``OTR_SilentComposite`` and the terminal ``OTR_MasterAudioMux``:

    SignalLostVideo -> SilentComposite -> [OTR_CaptionBurn] -> MasterAudioMux

It burns the SDH ``.ass`` (built by the surviving ``nodes/_otr_captions.py``
``build_ass_from_ledger``) onto the SILENT video stream only -- it NEVER touches
audio (audio is added LAST by MasterAudioMux with ``-c:a copy``), so the
byte-identical audio spine is untouched. Default-OFF ("clean master" is the
default); enable via the ``burn_captions`` widget (set by the capability
profiles -- the env-only enable path was CUT in the 2026-07-04 widget-audit).
When OFF (or on any caption-build failure) it PASSES THE INPUT THROUGH
unchanged -- captions never block the deliverable.

The burn re-encodes the video to the SAME canonical shape SilentComposite
produced (yuv420p / CFR / bt709 identity), so the downstream mux duration assert
still holds. Cold-import clean: ffmpeg/_otr_captions/_otr_paths are touched only
inside the burn path. NO ``-shortest``. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("OTR")

_CAPTION_STYLE_CHOICES = ["sdh_standard", "otr_crt"]
_DEFAULT_CAPTION_STYLE = "sdh_standard"


def _ffmpeg_bin(ffmpeg: str) -> str:
    return ffmpeg if (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) else ""


def _ass_filter_arg(ass_path: str) -> tuple[str, str]:
    """(basename, cwd) for the ffmpeg ``ass=`` filter -- reference the subtitle
    file by BASENAME with ffmpeg's cwd set to its folder, so a Windows
    drive-letter colon never reaches the filtergraph parser (mirrors the legacy
    blend node's proven trick)."""
    p = Path(ass_path)
    return (p.name, str(p.parent))


def _build_ass(ledger_path: str, style: str, margin_v: Optional[int]):
    """Build the SDH .ass via the surviving _otr_captions builder (lazy import).
    Returns (ass_path|None, report). Best-effort -- never raises."""
    try:
        try:
            from ._otr_captions import build_ass_from_ledger  # type: ignore
        except ImportError:  # loaded with nodes/ on sys.path
            from _otr_captions import build_ass_from_ledger  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, f"caption builder unavailable: {type(exc).__name__}: {exc}")
    try:
        return build_ass_from_ledger(ledger_path, style=style, margin_v=margin_v)
    except Exception as exc:  # noqa: BLE001
        return (None, f"build_ass_from_ledger raised: {type(exc).__name__}: {exc}")


def _resolve_ledger_path(video_path: str) -> Optional[str]:
    """Resolve this episode's TIMED ledger (start_s/dur_s) from disk by the video
    stem -- otr_audio_dir(stem)/<stem>_ledger.json, falling back to the in-flight
    ledger singleton (mirrors the legacy _resolve_captions_ass). Lazy imports."""
    stem = Path(video_path).stem
    # strip our pipeline suffixes so the stem matches the episode id. The
    # 86-owner migration (2026-07-04 widget-audit) moved CaptionBurn to AFTER the
    # procgen blend (node 93), so the incoming video is now
    # "<slug>_procgen_blended.mp4" -- strip that suffix too (ported from the
    # legacy blend node's resolver).
    for suf in ("_procgen_blended", "_silent", "_captioned", "_final", "_blend"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    try:
        try:
            from ._otr_paths import otr_audio_dir  # type: ignore
        except ImportError:
            from _otr_paths import otr_audio_dir  # type: ignore
        cand = Path(otr_audio_dir(stem)) / f"{stem}_ledger.json"
        if cand.is_file():
            return str(cand)
    except Exception:  # noqa: BLE001
        pass
    # Layout-relative fallback (ported from otr_post_upscale_procgen_blend): the
    # video lives INSIDE the per-episode folder (otr/episodes/<slug>/...), so its
    # sibling audio/ dir carries the ledger no matter which output root the
    # server pinned.
    try:
        sib = Path(video_path).resolve().parent / "audio"
        cands = sorted(sib.glob("*_ledger.json")) if sib.is_dir() else []
        if cands:
            return str(cands[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
        except ImportError:
            import _otr_ledger as _OTRL  # type: ignore
        alt = _OTRL.in_flight_ledger_path()
        if alt and Path(alt).is_file():
            return str(alt)
    except Exception:  # noqa: BLE001
        pass
    return None


def burn_captions_on_video(video_path: str, ledger_path: str, out_path: str, *,
                           style: str = _DEFAULT_CAPTION_STYLE, fps: int = 25,
                           ffmpeg: str = "ffmpeg", margin_v: Optional[int] = None):
    """Burn the SDH .ass onto the SILENT ``video_path`` -> ``out_path``.

    Pure function (node + tests). Builds the .ass from ``ledger_path`` then burns
    it with the libass ``ass`` filter, re-encoding video to canonical yuv420p /
    CFR / bt709 and keeping the clip SILENT (``-an``; audio is added downstream
    by MasterAudioMux). Returns ``(out_path, report)``; raises ``ValueError`` only
    on a hard ffmpeg/input error (the CALLER decides passthrough on failure)."""
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError(f"OTR_CaptionBurn: ffmpeg not found ({ffmpeg!r})")
    if not os.path.isfile(video_path):
        raise ValueError(f"OTR_CaptionBurn: input video missing: {video_path!r}")
    ass_path, report = _build_ass(ledger_path, style, margin_v)
    if not ass_path:
        raise ValueError(f"OTR_CaptionBurn: no captions ({report})")
    ass_name, ass_cwd = _ass_filter_arg(ass_path)
    cmd = [
        fb, "-y", "-loglevel", "error",
        "-i", os.path.abspath(video_path),
        "-an",                                   # silent in, silent out (V-1)
        "-vf", f"ass={ass_name},fps={max(1, int(fps))}",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        os.path.abspath(out_path),
    ]
    assert "-shortest" not in cmd, "V-2: -shortest must not appear in the caption burn"
    p = subprocess.run(cmd, cwd=ass_cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise ValueError(f"OTR_CaptionBurn: ffmpeg ass-burn failed :: {p.stderr.strip()[:300]}")
    return out_path, f"captions burned (style={style}) :: {report.splitlines()[0] if report else ''}"


class OTRCaptionBurn:
    """Registered as ``OTR_CaptionBurn``. SDH open-caption burn on the silent
    video (default-OFF passthrough; audio never touched)."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "burn"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Silent video from OTR_SilentComposite. Captions burn here; audio is added later by MasterAudioMux.",
                }),
            },
            "optional": {
                "burn_captions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Burn SDH open captions into the video. Default OFF (clean master); enabled by the capability profile. Node 86 is the single caption owner.",
                }),
                "caption_style": (_CAPTION_STYLE_CHOICES, {"default": _DEFAULT_CAPTION_STYLE}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "ledger_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional explicit timed-ledger path. Empty -> resolved from the video stem (otr_audio_dir / in-flight ledger).",
                }),
                "output_path": ("STRING", {"default": ""}),
                "gate_in": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _default_out(self, video_path: str) -> str:
        # Write the captioned mp4 BESIDE the input video. The 86-owner input is
        # node 93's "<slug>_procgen_blended.mp4" inside otr/episodes/<ep>/, so the
        # captioned twin lands in that same per-episode folder -- never the flat
        # episodes root (2026-07-04 widget-audit). Fall back to otr/episodes only
        # when the input path carries no resolvable directory.
        stem = os.path.splitext(os.path.basename(video_path or "episode"))[0]
        in_dir = os.path.dirname(os.path.abspath(video_path)) if video_path else ""
        if in_dir and os.path.isdir(in_dir):
            return os.path.join(in_dir, f"{stem}_captioned.mp4")
        try:
            import folder_paths  # type: ignore
            root = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            root = "."
        out_dir = os.path.join(root, "otr", "episodes")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{stem}_captioned.mp4")

    def burn(self, video_path, burn_captions=False, caption_style=_DEFAULT_CAPTION_STYLE,
             fps=25, ffmpeg="ffmpeg", ledger_path="", output_path="", gate_in=""):
        # Default-OFF clean master: pass the input through untouched. Enablement is
        # the burn_captions widget (set by capability profiles) ONLY -- the
        # env-only path was CUT (2026-07-04 widget-audit: captions are a clean
        # widget + profile on/off feature; node 86 is the single owner).
        if not bool(burn_captions):
            return (video_path, "OTR_CaptionBurn: captions OFF (clean master) -- passthrough")
        style = str(os.environ.get("OTR_CAPTION_STYLE", "") or caption_style).strip() or _DEFAULT_CAPTION_STYLE
        led = ledger_path.strip() or _resolve_ledger_path(video_path) or ""
        if not led:
            log.warning("[OTR_CaptionBurn] no timed ledger found; passthrough (no captions)")
            return (video_path, "OTR_CaptionBurn: no timed ledger; passthrough")
        out = output_path.strip() or self._default_out(video_path)
        try:
            final, report = burn_captions_on_video(
                video_path, led, out, style=style, fps=int(fps), ffmpeg=ffmpeg,
            )
        except ValueError as exc:
            # Best-effort: captions never block the deliverable -- pass through.
            log.warning("[OTR_CaptionBurn] %s; passthrough", exc)
            return (video_path, f"OTR_CaptionBurn: {exc}; passthrough (clean master)")
        log.info("[OTR_CaptionBurn] %s -> %s", report, final)
        return (final, "OTR_CaptionBurn OK -> " + final + "\n" + report)


__all__ = ["OTRCaptionBurn", "burn_captions_on_video", "_resolve_ledger_path", "_ass_filter_arg"]
