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
unchanged -- CAPTIONS never block the deliverable.

**THE HERO TITLE IS THE EXCEPTION, and it is deliberate (2026-08-12).** This
node also burns the episode's title card, because the procgen layer it used to
be drawn in composites lighten-only and cannot carry an outline (Bible 07.32).
So when ``title_card_plan_json`` is supplied this node is NO LONGER a
best-effort passthrough: it burns REGARDLESS of ``burn_captions``, and any
failure RAISES instead of passing through. That is the whole point -- there are
five no-error exits here, and once the title exists only in this .ass, each one
would publish an untitled episode with a clean log. With no plan supplied every
historical passthrough above is unchanged.

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
    """The ffmpeg this box should run, or ``""`` when it has none.

    HONOURS ``OTR_FFMPEG`` BEFORE PATH (2026-08-28). It did not, and that was
    the mirror image of a bug the pack had already fixed once: the shared
    ``_otr_shared/ffprobe.py`` resolver exists because only `otr_credits_roll`
    honoured ``OTR_FFPROBE`` while every other caller trusted PATH. That
    consolidation was scoped to the PROBE; the ENCODER kept the same hole here,
    in `otr_caption_burn`, `otr_master_audio_mux` and `otr_silent_composite` --
    which are the caption burn, the terminal audio mux and the silent-video
    normalize, i.e. the LAST three stages of an episode.

    So on a box where ffmpeg is reachable only through ``OTR_FFMPEG`` -- the
    AMD/Mac/alternate-box case the variant workflows exist for -- every earlier
    stage would succeed (the video engines all honour the variable) and the
    episode would die at the end, having spent the whole render.

    The explicit widget argument still wins when it resolves: an operator who
    typed a path meant it. The env var is consulted only when the passed value
    does not resolve, and PATH remains the last resort.
    """
    if ffmpeg and (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)):
        return ffmpeg
    cand = (os.environ.get("OTR_FFMPEG") or "").strip()
    if cand and (os.path.isfile(cand) or shutil.which(cand)):
        return cand
    return shutil.which("ffmpeg") or ""


def _ass_filter_arg(ass_path: str) -> tuple[str, str]:
    """(basename, cwd) for the ffmpeg ``ass=`` filter -- reference the subtitle
    file by BASENAME with ffmpeg's cwd set to its folder, so a Windows
    drive-letter colon never reaches the filtergraph parser (mirrors the legacy
    blend node's proven trick)."""
    p = Path(ass_path)
    return (p.name, str(p.parent))


def _build_ass(ledger_path: str, style: str, margin_v: Optional[int],
               title_plan=None, ass_out: Optional[str] = None):
    """Build the SDH .ass via the surviving _otr_captions builder (lazy import).
    Returns (ass_path|None, report).

    Best-effort for CAPTIONS -- never raises for them. A ``TitlePlanError``
    propagates, deliberately: a hero title that was planned and then cannot be
    drawn must reach the caller as a failure, not as a reason string that gets
    logged and passed through.
    """
    try:
        try:
            from ._otr_captions import build_ass_from_ledger, TitlePlanError  # type: ignore
        except ImportError:  # loaded with nodes/ on sys.path
            from _otr_captions import build_ass_from_ledger, TitlePlanError  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if title_plan:
            raise RuntimeError(
                f"OTR_CaptionBurn: caption builder unavailable and a hero title "
                f"card was planned: {type(exc).__name__}: {exc}"
            ) from exc
        return (None, f"caption builder unavailable: {type(exc).__name__}: {exc}")
    try:
        return build_ass_from_ledger(ledger_path, style=style, margin_v=margin_v,
                                     title_plan=title_plan, out_path=ass_out)
    except TitlePlanError:
        raise
    except Exception as exc:  # noqa: BLE001
        if title_plan:
            raise RuntimeError(
                f"OTR_CaptionBurn: .ass build failed while a hero title card was "
                f"planned: {type(exc).__name__}: {exc}"
            ) from exc
        return (None, f"build_ass_from_ledger raised: {type(exc).__name__}: {exc}")


def _parse_title_plan(raw: str):
    """Parse the wire title plan. Returns ``(plan|None, report)``.

    An EMPTY string means "this graph does not send a title plan" -- the honest
    default, and the node keeps its historical best-effort caption behaviour.
    Anything else is a promise that a title exists, so a malformed or empty-of-
    frames plan RAISES rather than degrading to None: silently reading a broken
    plan as "no title wanted" is precisely the failure this whole change exists
    to remove.
    """
    import json

    text = (raw or "").strip()
    if not text:
        return (None, "no title plan supplied")
    try:
        plan = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(
            f"OTR_CaptionBurn: title_card_plan_json is not valid JSON "
            f"({exc}); refusing to render a title-less master"
        ) from exc
    if not isinstance(plan, dict):
        raise RuntimeError(
            f"OTR_CaptionBurn: title_card_plan_json parsed as "
            f"{type(plan).__name__}, expected an object"
        )
    frames = plan.get("frames") or []
    if not frames:
        raise RuntimeError(
            "OTR_CaptionBurn: title_card_plan_json carries no frames. An "
            "episode with no title card sends an EMPTY string, not an empty plan."
        )
    return (plan, f"title plan v{plan.get('version')}: {len(frames)} planned "
                  f"frames across {len(plan.get('cards') or [])} card(s)")


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
                           ffmpeg: str = "ffmpeg", margin_v: Optional[int] = None,
                           title_plan=None):
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
    # Say WHERE the .ass goes rather than letting the builder derive it from the
    # ledger path. A title-only burn legitimately has NO ledger (that is the
    # point of the fail-closed work), and the derivation then falls back to a
    # bare relative name -- which writes a stray "episode_captions.ass" into the
    # SERVER'S working directory instead of the episode folder. The destination
    # mp4 is already resolved and already lives beside the episode, so it is the
    # honest source for this. Found in QA.
    ass_out = os.path.splitext(os.path.abspath(out_path))[0] + ".ass"
    ass_path, report = _build_ass(ledger_path, style, margin_v,
                                  title_plan=title_plan, ass_out=ass_out)
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
    video (default-OFF passthrough; audio never touched).

    ALSO the hero title card's burn site, and there it is NOT a passthrough: a
    supplied ``title_card_plan_json`` makes this node burn regardless of
    ``burn_captions`` and REFUSE on any failure. See the module docstring.
    """

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
                "caption_style": (_CAPTION_STYLE_CHOICES, {
                    "default": _DEFAULT_CAPTION_STYLE,
                    "tooltip": "Visual style of the burned SDH captions. The "
                               "OTR_CAPTION_STYLE env var may override the "
                               "STYLE only -- no env var can turn burning on; "
                               "only the burn_captions widget (or a supplied "
                               "title-card plan, which forces the hero-title "
                               "burn alone) does that.",
                }),
                "fps": ("INT", {
                    "default": 25, "min": 1, "max": 120,
                    "tooltip": "Frame rate used for caption timing math. Must "
                               "match the incoming silent video's real rate or "
                               "captions drift against speech.",
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "ffmpeg binary for the caption burn. Resolution "
                               "order: this widget's value if it runs, then "
                               "the OTR_FFMPEG env var, then PATH.",
                }),
                "ledger_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional explicit timed-ledger path. Empty -> resolved from the video stem (otr_audio_dir / in-flight ledger).",
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Captioned mp4 path. Empty (the shipped "
                               "default) -> <stem>_captioned.mp4 beside the "
                               "input video. With burn OFF and no title plan "
                               "the input passes through untouched.",
                }),
                "gate_in": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
                # APPENDED at the END of `optional` on purpose: widgets_values is
                # POSITIONAL, so inserting mid-list shifts every saved value in
                # every workflow (BUG-LOCAL-097). New optional widgets only ever
                # go last.
                "title_card_plan_json": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Hero title-card plan from OTR_SignalLostVideo. When present the title burns REGARDLESS of burn_captions and any failure REFUSES instead of passing through -- the title is the only copy left once the procgen draw is suppressed.",
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
             fps=25, ffmpeg="ffmpeg", ledger_path="", output_path="", gate_in="",
             title_card_plan_json=""):
        # THE HERO TITLE IS NOT A CAPTION, and this node has FIVE no-error
        # passthrough exits. That was harmless while the title was baked into
        # procgen pixels; it is not once the title lives ONLY in this .ass,
        # because every one of them would ship an episode with NO TITLE and a
        # cheerful log line. So a planned title makes this node FAIL CLOSED:
        #   1. burn_captions OFF        -> still burns (below)
        #   2. no timed ledger          -> still burns, title-only .ass
        #   3. unknown caption style    -> raises
        #   4. no speech lines          -> still burns, title-only .ass
        #   5. any ValueError from the burn (ffmpeg missing, input missing,
        #      ledger unreadable, ffmpeg failure) -> re-raised
        title_plan, plan_report = _parse_title_plan(title_card_plan_json)
        title_required = title_plan is not None

        # Default-OFF clean master: pass the input through untouched. Enablement is
        # the burn_captions widget (set by capability profiles) ONLY -- the
        # env-only path was CUT (2026-07-04 widget-audit: captions are a clean
        # widget + profile on/off feature; node 86 is the single owner).
        if not bool(burn_captions) and not title_required:
            return (video_path, "OTR_CaptionBurn: captions OFF (clean master) -- passthrough")
        style = str(os.environ.get("OTR_CAPTION_STYLE", "") or caption_style).strip() or _DEFAULT_CAPTION_STYLE
        led = ledger_path.strip() or _resolve_ledger_path(video_path) or ""
        if not led and not title_required:
            log.warning("[OTR_CaptionBurn] no timed ledger found; passthrough (no captions)")
            return (video_path, "OTR_CaptionBurn: no timed ledger; passthrough")
        out = output_path.strip() or self._default_out(video_path)
        try:
            final, report = burn_captions_on_video(
                video_path, led, out, style=style, fps=int(fps), ffmpeg=ffmpeg,
                title_plan=title_plan,
            )
        except ValueError as exc:
            if title_required:
                # REFUSE. The alternative is a published episode with no title
                # and nothing in the log that looks like a failure.
                raise RuntimeError(
                    f"OTR_CaptionBurn: a hero title card was planned "
                    f"({plan_report}) and the burn failed: {exc}. Refusing to "
                    f"pass through a title-less master."
                ) from exc
            # Best-effort: captions never block the deliverable -- pass through.
            log.warning("[OTR_CaptionBurn] %s; passthrough", exc)
            return (video_path, f"OTR_CaptionBurn: {exc}; passthrough (clean master)")
        log.info("[OTR_CaptionBurn] %s -> %s", report, final)
        head = "OTR_CaptionBurn OK -> " + final
        if title_required:
            head += f"\n{plan_report}"
        return (final, head + "\n" + report)


__all__ = ["OTRCaptionBurn", "burn_captions_on_video", "_resolve_ledger_path", "_ass_filter_arg"]
