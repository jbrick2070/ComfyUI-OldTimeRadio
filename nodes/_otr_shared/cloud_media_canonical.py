"""Cloud media canonicalization contract -- S0 skeleton (pass04 sec 6).

S0 ships the TYPES + validation guard + dispatch signatures; the real
per-modality canonicalizers land with their lane sprints (S1 stills,
S2 voice/music, S3 video). Everything here is fail-closed: partial or
invalid media never reaches an episode path.

PartnerResult (DS R4 #2): the exact shape invoke_partner_node returns
and the ONLY input shape canonicalizers accept. Downloads stream to a
temp path -- adapters never hold whole media in memory.

Loudness: the reference constant is resolved at S2 from the EXISTING
local lane (verify-at-build #11) -- do not invent a value here.
LOUDNESS_REFERENCE_SOURCE documents where it must come from.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

from .cloud_media_backend import CloudErrorCode, CloudMediaError
from .ffmpeg import resolve_ffmpeg

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import env as otr_env  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import env as otr_env  # type: ignore  # _otr_shared/ on sys.path

try:
    from . import proc as otr_proc
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import proc as otr_proc  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import proc as otr_proc  # type: ignore  # _otr_shared/ on sys.path

__all__ = [
    "PartnerResult",
    "CanonicalAsset",
    "CANONICALIZER_VERSION",
    "LOUDNESS_REFERENCE_SOURCE",
    "validate_partner_result",
    "canonicalize_audio",
    "canonicalize_image",
    "canonicalize_video",
    "canonical_clip_frame_count",
    "cloud_delivery_wh",
    "engine_request_target_frames",
]


def cloud_delivery_wh(request_w, request_h, *, land_env, port_env,
                      land_default="1920x1080", port_default="1080x1920"):
    """The TRUE 1080p cloud DELIVERY canvas (orientation-preserving), env-overridable.

    The provider clip/still is conformed to THIS canvas by ``canonicalize_*`` -- NOT
    the smaller per-family request canvas (which would DOWNSCALE a 1080p provider
    output, e.g. word_razzle's 1472x832 request -> a 1472x832 clip). CLOUD-LANE ONLY:
    no local engine reads this, so locals keep their own render resolution
    (kibitz-grounded 2026-07-03). Portrait is chosen when the request canvas is
    TALLER than wide (rh > rw) -- the cloud analogue of "HuMo portrait excepted from
    the landscape bump". Unknown / zero request -> landscape default.
    """
    try:
        rw, rh = int(request_w or 0), int(request_h or 0)
    except (TypeError, ValueError):
        rw = rh = 0
    portrait = rh > rw > 0
    spec = (otr_env.get(port_env, port_default) if portrait
            else otr_env.get(land_env, land_default))
    try:
        parts = str(spec).lower().split("x", 1)
        return (max(1, int(parts[0])), max(1, int(parts[1])))
    except (ValueError, IndexError, AttributeError):
        return (1080, 1920) if portrait else (1920, 1080)


def engine_request_target_frames(request) -> int:
    """The coverage plan's render length for this clip, or 0 if absent.

    ``timing.target_frame_count`` is ``segment.render_frames`` -- the length
    the assembler will demand -- not the later visible count after
    ``trim_tail``. A zero means the canonicalizer must not cap.
    """
    if request is None:
        return 0
    if isinstance(request, dict):
        timing = request.get("timing") or {}
    else:
        timing = getattr(request, "timing", None) or {}
    if isinstance(timing, dict):
        raw = timing.get("target_frame_count", 0)
    else:
        raw = getattr(timing, "target_frame_count", 0)
    try:
        n = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def canonical_clip_frame_count(asset: CanonicalAsset) -> int:
    """Pictures this canonical file carries. Prefer a counted field."""
    raw = getattr(asset, "frame_count", None)
    try:
        counted = int(raw)
    except (TypeError, ValueError):
        counted = 0
    if counted > 0:
        return counted
    try:
        fps = float(asset.fps or 0.0)
        dur = float(asset.duration_s or 0.0)
    except (TypeError, ValueError):
        return 0
    if fps <= 0.0 or dur <= 0.0:
        return 0
    return int(round(dur * fps))


#: bumped on ANY output-contract change (DS R3 S-2: simple integers).
CANONICALIZER_VERSION = 3

#: RESOLVED (cloud-audio S0/C8, 2026-07-03): the local lane's real loudness
#: handling is scene_sequencer's per-segment RMS leveling (NOT a LUFS
#: convention): every dialogue/cue clip is single-gain leveled toward
#: OTR_SEGMENT_TARGET_RMS_DBFS (default -16.0 dBFS), peak-safe, via
#: ``scene_sequencer._loudness_normalize_clip`` with ``_loudnorm_params()``.
#: canonicalize_audio reuses that EXACT gain so a cloud WAV sits at the same
#: perceived loudness as a local line going into OTR_EpisodeAssembler.
LOUDNESS_REFERENCE_SOURCE = (
    "nodes.scene_sequencer._loudness_normalize_clip / _loudnorm_params "
    "(per-segment RMS leveling, target OTR_SEGMENT_TARGET_RMS_DBFS=-16.0 dBFS)")


class PartnerResult(TypedDict):
    """Return shape of invoke_partner_node (pass04 sec 3)."""

    path: str                 # temp file the download streamed to
    content_type: str         # e.g. "video/mp4", "audio/wav", "image/png"
    duration_s: Optional[float]
    provider_job_id: Optional[str]
    raw_meta: dict


@dataclass(frozen=True)
class CanonicalAsset:
    """Output of every canonicalizer (pass04 sec 6 / GPT R2 #9)."""

    path: Path
    sha256: str
    media_type: str  # "audio" | "image" | "video"
    duration_s: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    container: Optional[str]
    provider_job_id: Optional[str]
    validation_warnings: tuple = ()
    #: Counted pictures in the canonical file when known. Optional so older
    #: still/audio constructors stay valid. Video adapters prefer this over
    #: ``round(duration_s * fps)``, which is how a 5.08 s provider clip became
    #: a 127-frame receipt against a 125-frame plan (PBUG-20260915-01).
    frame_count: Optional[int] = None


def validate_partner_result(raw: dict) -> PartnerResult:
    """Fail-closed shape check before any canonicalization work."""
    if not isinstance(raw, dict):
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              "partner result is not a mapping")
    missing = [k for k in ("path", "content_type") if not raw.get(k)]
    if missing:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"partner result missing {missing}")
    path = Path(raw["path"])
    if not path.is_file() or path.stat().st_size == 0:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"partner result file missing or empty: {path}")
    return PartnerResult(
        path=str(path),
        content_type=str(raw["content_type"]),
        duration_s=raw.get("duration_s"),
        provider_job_id=raw.get("provider_job_id"),
        raw_meta=dict(raw.get("raw_meta") or {}),
    )


def canonicalize_audio(raw: PartnerResult, request: dict, session=None) -> CanonicalAsset:
    """S0/C8 (cloud-audio 2026-07-03). Conform a provider audio result to the
    local lane's contract so a cloud WAV drops into OTR_EpisodeAssembler exactly
    like a local line:

    - decode (any provider container) -> resample to ``sample_rate`` (default
      44100) -> ``stereo_policy`` channels, via ffmpeg (robust to mp3/wav/etc);
    - **loudness matched by REUSING the existing local RMS leveler** (NOT a fresh
      LUFS convention -- operator chose RMS): the SAME single peak-safe gain
      ``scene_sequencer._loudness_normalize_clip`` applies toward
      ``_loudnorm_params()`` (target -16 dBFS). No re-implementation.
    - +/-250ms per-line tolerance: when ``request['target_duration_s']`` is given
      and the clip is shorter within tolerance, pad TAIL silence to the slot
      (never trims within tolerance);
    - emit ``actual_duration_s`` (as ``duration_s``) for the line metadata.

    ``request`` keys: ``sample_rate`` (default 44100), ``stereo_policy``
    ("stereo"|"mono", default stereo), optional ``target_duration_s`` +
    ``out_path``. ``session`` is accepted for signature parity (unused here)."""
    import hashlib
    import numpy as np

    validated = validate_partner_result(dict(raw))
    src = validated["path"]
    sr = int(request.get("sample_rate") or 44100)
    ch = 1 if str(request.get("stereo_policy") or "stereo").lower() == "mono" else 2

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              "ffmpeg not found -- cannot canonicalize audio")

    dec = otr_proc.run(
        [ffmpeg, "-v", "error", "-i", str(src), "-ar", str(sr), "-ac", str(ch),
         "-f", "f32le", "-"], capture_output=True)
    if dec.returncode != 0 or not dec.stdout:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            "audio decode failed: %r" % (dec.stderr[-300:],))
    flat = np.frombuffer(dec.stdout, dtype=np.float32).copy()
    audio = flat.reshape(-1, ch) if ch == 2 else flat

    # --- loudness: reuse the EXISTING local RMS leveler (no re-code) ---
    warnings: list = []
    try:
        from ..scene_sequencer import (_loudness_normalize_clip,  # type: ignore
                                        _loudnorm_params)
    except ImportError:  # pragma: no cover -- flat test imports
        from scene_sequencer import (_loudness_normalize_clip,  # type: ignore
                                     _loudnorm_params)
    mono = audio if ch == 1 else audio.mean(axis=1)
    if mono.size and float(np.abs(mono).max()) > 0.0:
        leveled = _loudness_normalize_clip(mono, **_loudnorm_params())
        j = int(np.argmax(np.abs(mono)))
        gain = float(leveled[j] / mono[j]) if mono[j] != 0 else 1.0
        audio = (audio * gain).astype(np.float32)

    # --- +/-250ms tolerance: pad TAIL silence up to the slot (never trim) ---
    target = request.get("target_duration_s")
    if target is not None:
        cur = audio.shape[0] / float(sr)
        deficit = float(target) - cur
        if 0.0 < deficit <= 0.25 + 1e-6:
            pad_n = int(round(deficit * sr))
            sil = (np.zeros((pad_n, ch), np.float32) if ch == 2
                   else np.zeros(pad_n, np.float32))
            audio = np.concatenate([audio, sil], axis=0)
        elif abs(deficit) > 0.25:
            warnings.append(
                "duration %.3fs vs target %.3fs exceeds +/-250ms tolerance"
                % (cur, float(target)))

    out_path = Path(request.get("out_path") or (str(src) + ".canon.wav"))
    enc = otr_proc.run(
        [ffmpeg, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr),
         "-ac", str(ch), "-i", "-", "-c:a", "pcm_s16le", str(out_path)],
        input=audio.astype(np.float32).tobytes(), capture_output=True)
    if enc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            "audio encode failed: %r" % (enc.stderr[-300:],))

    actual = audio.shape[0] / float(sr)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return CanonicalAsset(
        path=out_path, sha256=sha, media_type="audio",
        duration_s=round(actual, 3), width=None, height=None, fps=None,
        container="wav", provider_job_id=validated.get("provider_job_id"),
        validation_warnings=tuple(warnings))


def canonicalize_image(raw: PartnerResult, request: dict, session=None) -> CanonicalAsset:
    """S1 (2026-07-03). Conform a provider still to the ROLE canvas:

    - decode the provider image (PNG/JPEG/WEBP -- the bridge already streamed
      it to a temp file) and force sRGB RGB (drop alpha / ICC quirks);
    - resize to the EXACT role canvas WITHOUT distortion: scale-to-COVER then
      centre-crop to ``w x h`` (stills get Ken-Burns panned by still_pan /
      still_flat downstream, so a fill -- never letterbox bars -- is the right
      conform for a background/portrait plate);
    - re-encode a real sRGB PNG on disk (the dispatcher's ``_coerce_pixels``
      reads a ``.png`` PATH and enforces a minimum byte floor);
    - sha256 of the canonical bytes.

    ``request`` supplies ``{"w", "h"}`` (both required, fail-closed) and
    optionally ``format`` (default ``"PNG"``) + ``out_path`` (default: a fresh
    png beside the input with a ``.canon.png`` suffix). ``session`` is accepted
    for signature parity with the audio/video canonicalizers but is unused here
    (image conform is a pure pixel op)."""
    import hashlib
    validated = validate_partner_result(dict(raw))
    src = Path(validated["path"])
    try:
        w = int(request["w"])
        h = int(request["h"])
    except (KeyError, TypeError, ValueError):
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            "canonicalize_image request must carry integer w/h "
            f"(got {request!r})")
    if w <= 0 or h <= 0:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"canonicalize_image needs positive w/h (got {w}x{h})")
    fmt = str(request.get("format") or "PNG").upper()
    if fmt != "PNG":
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"canonicalize_image only emits PNG (got format={fmt!r})")
    out_path = Path(request.get("out_path") or
                    src.with_suffix("")).with_suffix(".canon.png")
    try:
        from PIL import Image
        with Image.open(str(src)) as im:
            im = im.convert("RGB")          # sRGB, drop alpha/palette/ICC
            src_w, src_h = im.size
            if src_w <= 0 or src_h <= 0:
                raise CloudMediaError(
                    CloudErrorCode.CORRUPT_OUTPUT,
                    f"provider image {src} decoded to a zero dimension")
            # scale-to-COVER: the larger ratio fills the canvas, centre-crop
            # the overflow -> exact w x h, no distortion, no bars.
            scale = max(w / src_w, h / src_h)
            new_w = max(w, int(round(src_w * scale)))
            new_h = max(h, int(round(src_h * scale)))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            im = im.crop((left, top, left + w, top + h))
            im.save(str(out_path), format="PNG")
    except CloudMediaError:
        raise
    except Exception as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"image conform failed for {src}: {exc}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"canonical image not written or empty: {out_path}")
    sha = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    warnings = ()
    if (src_w, src_h) != (w, h):
        warnings = (f"provider still {src_w}x{src_h} conformed to "
                    f"{w}x{h} (cover+crop)",)
    return CanonicalAsset(
        path=out_path,
        sha256=sha.hexdigest(),
        media_type="image",
        duration_s=None,
        width=w, height=h, fps=None,
        container="png",
        provider_job_id=validated.get("provider_job_id"),
        validation_warnings=warnings,
    )


def _ffprobe_streams(path: str) -> dict:
    """``{"video": [...], "audio": [...], "duration_s": float}`` via ffprobe.
    Fail-closed CORRUPT_OUTPUT on any probe failure -- partial media never
    proceeds. THAT VERDICT IS THIS MODULE'S and does not move; the shared
    boundary only finds and launches the tool, which is how a provider clip is
    finally measured with the same ffprobe the local engines use."""
    from . import ffprobe as _ffp
    try:
        doc = _ffp.probe_json(path, extra_args=("-show_streams", "-show_format"),
                              timeout=120)
    except _ffp.FFprobeError as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"ffprobe failed on {path}: {exc}")
    streams = doc.get("streams") or []
    dur_raw = ((doc.get("format") or {}).get("duration"))
    try:
        duration = float(dur_raw)
    except (TypeError, ValueError):
        # pass04 sec 6: actual_duration_s validators fail with a NAMED
        # missing-field error on cloud runs -- never a silent None.
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"provider clip {path} has no format.duration "
            f"(actual_duration_s unresolvable)")
    return {
        "video": [s for s in streams if s.get("codec_type") == "video"],
        "audio": [s for s in streams if s.get("codec_type") == "audio"],
        "duration_s": duration,
    }


def _count_output_frames(path: str) -> int | None:
    """Decoded picture count when the container header has no nb_frames.

    ``-frames:v`` is supposed to be exact; a missing header is not proof
    it held. Decode-count the output the same way the assembly boundary
    does, and let the caller fail closed on a surplus.
    """
    from . import ffprobe as _ffp
    try:
        doc = _ffp.probe_json(
            path, "stream=nb_read_frames", select_streams="v:0",
            extra_args=("-count_frames",), timeout=120)
    except _ffp.FFprobeError:
        return None
    streams = doc.get("streams") or []
    if not streams:
        return None
    try:
        n = int(streams[0].get("nb_read_frames"))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def canonicalize_video(raw: PartnerResult, request: dict, session=None) -> CanonicalAsset:
    """S3 (2026-07-02). Conform a provider clip to the ROLE contract:

    - provider audio ALWAYS stripped (``-an``; must_strip_audio=True across
      shipped rows -- master audio is frozen upstream, mux is LAST), with a
      POST-STRIP PROOF (re-probe: zero audio streams) recorded on the asset;
    - role canvas (fit + pad, never distort) + role fps + h264/yuv420p/bt709
      mp4 (the CanonicalClip container contract every local engine ships);
    - ``actual_duration_s`` measured from the OUTPUT (named error when the
      provider clip carries no duration);
    - sha256 of the canonical bytes.

    ``request`` supplies ``{"w", "h", "fps"}`` (all required, fail-closed) and
    optionally ``out_path`` (default: a fresh mp4 beside the input with a
    ``.canon.mp4`` suffix) and ``target_frames`` (the coverage plan's
    ``render_frames``). When ``target_frames`` is a positive integer, the
    encoder keeps that many pictures after the duration-preserving ``fps``
    filter -- the ``allow_tail_trim`` the cloud adapters already declare.
    A billed N-second provider clip (and the fps resample onto the 25 fps
    canvas) routinely emits two extra frames; without the cap those frames
    fail the plan-vs-output proof at assembly."""
    import hashlib
    validated = validate_partner_result(dict(raw))
    src = Path(validated["path"])
    try:
        w = int(request["w"])
        h = int(request["h"])
        fps = int(request["fps"])
    except (KeyError, TypeError, ValueError):
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            "canonicalize_video request must carry integer w/h/fps "
            f"(got {request!r})")
    raw_target = request.get("target_frames")
    target_frames = 0
    if raw_target not in (None, ""):
        try:
            target_frames = int(raw_target)
        except (TypeError, ValueError):
            raise CloudMediaError(
                CloudErrorCode.MALFORMED_CONFIG,
                "canonicalize_video target_frames must be an integer "
                f"(got {raw_target!r})")
        if target_frames < 0:
            raise CloudMediaError(
                CloudErrorCode.MALFORMED_CONFIG,
                "canonicalize_video target_frames must be >= 0 "
                f"(got {target_frames})")
    probe = _ffprobe_streams(str(src))
    if not probe["video"]:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"provider clip {src} has no video stream")
    out_path = Path(request.get("out_path") or
                    src.with_suffix("")).with_suffix(".canon.mp4")
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p")
    # The pack's ONE ffmpeg answer (the pin, then PATH) -- the same one the
    # AUDIO canonicalizer above and every episode-stage encoder use.
    ffmpeg_bin = resolve_ffmpeg()
    if not ffmpeg_bin:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              "ffmpeg not found -- cannot canonicalize video")
    cmd = [ffmpeg_bin, "-v", "error", "-y", "-i", str(src), "-an",
           "-vf", vf]
    if target_frames > 0:
        cmd.extend(["-frames:v", str(target_frames)])
    cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-movflags", "+faststart",
                str(out_path)])
    try:
        res = otr_proc.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[-300:])
    except CloudMediaError:
        raise
    except Exception as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"ffmpeg conform failed for {src}: {exc}")
    post = _ffprobe_streams(str(out_path))
    if post["audio"]:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"audio strip FAILED: canonical {out_path} still carries "
            f"{len(post['audio'])} audio stream(s)")
    sha = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    warnings = ()
    if probe["audio"]:
        warnings = (f"provider audio stripped ({len(probe['audio'])} "
                    f"stream(s); strip proof: 0 in output)",)
    v0 = (post.get("video") or [{}])[0]
    counted = None
    raw_nb = v0.get("nb_frames")
    try:
        parsed_nb = int(raw_nb)
    except (TypeError, ValueError):
        parsed_nb = 0
    if parsed_nb > 0:
        counted = parsed_nb
    duration_s = post["duration_s"]
    frame_count = counted
    if target_frames > 0:
        if counted is None:
            counted = _count_output_frames(str(out_path))
        if counted is not None and counted > target_frames:
            raise CloudMediaError(
                CloudErrorCode.CORRUPT_OUTPUT,
                f"canonical {out_path} kept {counted} frame(s) after "
                f"-frames:v {target_frames}")
        kept = counted if counted is not None else target_frames
        frame_count = kept
        if counted is None or counted == target_frames:
            # Stamp the plan's duration so round(duration*fps) cannot
            # revive a container-header surplus of a couple of frames.
            duration_s = float(kept) / float(fps)
    return CanonicalAsset(
        path=out_path,
        sha256=sha.hexdigest(),
        media_type="video",
        duration_s=duration_s,
        width=w, height=h, fps=float(fps),
        container="mp4",
        provider_job_id=validated.get("provider_job_id"),
        validation_warnings=warnings,
        frame_count=frame_count,
    )


# The SFX-bed extraction chain (extract_sfx_bed_from_provider_video ->
# _normalize_sfx_stem_audio -> _sfx_loudnorm_params -> _env_float/_rms_dbfs,
# plus _provider_video_path) was deleted with the SFX bed producers
# (rip-sfx 2026-08-06). No surviving engine keeps provider audio.
