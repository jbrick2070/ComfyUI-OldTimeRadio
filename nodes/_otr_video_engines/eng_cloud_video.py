"""Cloud partner VIDEO adapters -- S3 core (pass04 secs 5+7, operator GO
2026-07-02 evening: "code the cloud video plan").

Four rows from the S0 pin table, invoked through the S0 bridge
(``invoke_partner_node``) and conformed by ``canonicalize_video``:

    cloud_kling_avatar   required_audio_ref   (init_image, audio_ref)
    cloud_seedance_2     required_audio_ref   (init_image, audio_ref)
    cloud_wan_i2v        mute_only            (init_image, text_prompt)
    cloud_wan_i2v_audio  required_audio_ref   (init_image, audio_ref)
    cloud_vidu_q2_pro_fast_720p mute_only     (init_image, text_prompt)

S3-CORE SCOPE: rows REGISTER unconditionally (registry-IS-the-menu C6) with
empty ``default_roles`` -- selectable, NEVER automatic. Operator directive
2026-07-02: the DROPDOWN PICK is the enable (no OTR_ENABLE_COMFY_CLOUD_MEDIA
hidden switch -- same clean break as the OpenRouter C6 flag removal); a pick
without credentials fails LOUD at auth resolution. ``assert_usable`` fails
CLOSED (EngineUnusable) unless ffmpeg is present (the canonicalizer strips
provider audio) and the pin row is OK. The reactive
auto-default policy + ShotLock audit stamps + fallback chains land with S3
FULL, after the operator's live smokes prove the bridge.

ALL provider audio is stripped unconditionally (must_strip_audio=True; the
master mix is frozen upstream, mux is LAST) -- clips return has_audio=False
like every local engine. Money: per-clip estimate rides
``OTR_CLOUD_VIDEO_EST_USD`` (default 0.50) against the session budget
ceiling; timeout ``OTR_CLOUD_VIDEO_TIMEOUT_S`` (default 900).

Cold-import-clean: stdlib + registry only at module scope; torch / PIL /
soundfile / the bridge import lazily inside the render lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

from .registry import EngineUnusable, EngineUsabilityReason, register
from . import frame_contract as _fc
from .frame_contract import CONTINUITY_SOFT_REFERENCE, FrameContract
from .._otr_shared.still_plan_helpers import StillPlanRow
from .._otr_story_brief_helpers import (
    append_visual_safety_clause,
    visual_safety_negative,
)

_LOG = logging.getLogger("OTR.video.eng_cloud_video")

#: The canvas rate word_razzle's discrete menu is expressed at -- the same 25
#: its contract declares as ``native_fps``. Named so the seconds<->frames
#: conversion in ``_duration_seconds`` reads off one number, not a literal.
_RAZZLE_FPS = 25


#: S1 (2026-07-25) per-model still plans for the cloud video engines
#: registered in this file (spec section 3, Shape A -- scene spine).
#: FILE-LOCAL, fully declared: the plan tuple is the whole authority; no
#: cross-module or cross-file sharing (per the spec's "no inheritance or
#: shared defaults" rule -- reusing a Python variable within THIS module
#: is not shared-defaults, the plan is fully declared by name here). Nothing
#: reads the plan at S1 -- S2 wires it in.
_CLOUD_VIDEO_SHAPE_A_BASE_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)


#: Same as ``_CLOUD_VIDEO_SHAPE_A_BASE_PLAN`` but portrait REQUIRED=always,
#: for the audio-driven-face lane (``cloud_kling_avatar``): the provider
#: consumes an init face image every beat, so the portrait is structural.
_CLOUD_KLING_AVATAR_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="always",
                 framing_geometry=(
                     ("in-character cinematic medium shot, head and "
                      "shoulders, face clearly visible, subject centred with "
                      "natural headroom above the head (never crop the top of "
                      "the head)")),
                 style_tail_policy="full"),
)

#: kling avatar mode COMBO -- the pin excludes combo options (S0), so the
#: adapter ships the provider's documented std tier; env-overridable.
_KLING_MODE_ENV = "OTR_CLOUD_KLING_MODE"
_KLING_MODE_DEFAULT = "std"
_KLING_MODES = ("std", "pro")
_KLING_MODE_ALIASES = {
    "standard": "std",
    "standard mode": "std",
    "professional": "pro",
    "professional mode": "pro",
}
_KLING_AVATAR_MARKER = (
    "Natural broadcast avatar delivery; lip sync leads all motion.")
_KLING_AVATAR_BASE_CLAUSE = (
    "Broadcast-style digital human delivery. Natural lip sync follows the "
    "audio exactly. Keep the head and shoulders stable in frame with subtle "
    "facial expression, gentle eye contact, and small natural head movement. "
    "No exaggerated gestures, no camera shake, no sudden reframing. "
    f"{_KLING_AVATAR_MARKER}")

_SEEDANCE_MODEL_ALIASES = {
    # The installed ByteDance2ReferenceNode indexes SEEDANCE_MODELS by UI label.
    # Accept provider ids too so older operator env overrides keep failing loud
    # only when the value is genuinely unknown.
    "dreamina-seedance-2-0-260128": "Seedance 2.0",
    "dreamina-seedance-2-0-fast-260128": "Seedance 2.0 Fast",
    "dreamina-seedance-2-0-mini": "Seedance 2.0 Mini",
}
_SEEDANCE_RESOLUTIONS = {
    "Seedance 2.0": ("480p", "720p", "1080p", "4k"),
    "Seedance 2.0 Fast": ("480p", "720p"),
    "Seedance 2.0 Mini": ("480p", "720p"),
}
_SEEDANCE_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")
_SEEDANCE_SMOOTH_MARKER = (
    "Gentle parallax only; all motion gradual and physically continuous.")
_SEEDANCE_SMOOTH_MOTION_CLAUSE = (
    "One continuous uncut shot. Smooth stabilized camera on a slow dolly "
    "with gentle ease-in and ease-out. Motion begins immediately in the first "
    "frame and remains gentle and continuous throughout. Preserve the "
    "reference-image composition and framing. No whip pans, handheld shake, "
    "sudden reframing, jump cuts, or rapid zooms. "
    f"{_SEEDANCE_SMOOTH_MARKER}")
_SEEDANCE_PROMPT_VARIANT = "seedance_smooth_v1"
_SEEDANCE_PROMPT_SOFTENERS = (
    ("dynamic_dolly_push",
     re.compile(r"\bdynamic\s+dolly\s+push\b", re.IGNORECASE),
     "slow controlled dolly push"),
    ("handheld_dolly",
     re.compile(r"\bhandheld\s+dolly\b", re.IGNORECASE),
     "stabilized dolly"),
    ("whip_pans",
     re.compile(r"\bwhip[- ]pans?\b", re.IGNORECASE),
     "slowly sweeps"),
    ("white_hot",
     re.compile(r"\bwhite[- ]hot\b", re.IGNORECASE),
     "bright warm glow"),
    ("rapid_zooms",
     re.compile(r"\brapid\s+zooms?\b", re.IGNORECASE),
     "slow controlled push"),
    ("aggressively",
     re.compile(r"\baggressively\b", re.IGNORECASE),
     "subtly"),
    ("standalone_handheld",
     re.compile(r"\bhandheld\b", re.IGNORECASE),
     "stabilized"),
)

_WAN_MODELS = ("wan2.7-i2v",)
_WAN_MODEL_ALIASES = {
    # The cloud Partner node now exposes the Wan 2.7 selector. Older saved env
    # overrides and local-Wan docs used 2.2-style ids; normalize those at the
    # adapter boundary so the live request still carries a schema-valid model.
    "wan-2.2-i2v": "wan2.7-i2v",
    "wan2.2-i2v": "wan2.7-i2v",
    "wan-2.7-i2v": "wan2.7-i2v",
}
_WAN_RESOLUTIONS = ("720P", "1080P")
_WAN_SMOOTH_MARKER = (
    "Stable first-frame motion; preserve composition and move continuously.")
_WAN_SMOOTH_MOTION_CLAUSE = (
    "Generate one continuous shot from the first frame. Preserve the "
    "first-frame subject, composition, aspect ratio, lighting, and visual "
    "style. Use slow, physically continuous motion with gentle parallax and "
    "no cuts. No whip pans, handheld shake, sudden reframing, jump cuts, "
    "rapid zooms, melting geometry, warped faces, drifting text, black frames, "
    "or pillarbox bars. "
    f"{_WAN_SMOOTH_MARKER}")
_WAN_PROMPT_VARIANT = "wan_i2v_smooth_v1"
_WAN_NEGATIVE_DEFAULT = visual_safety_negative(
    "jump cuts, whip pans, rapid zooms, handheld shake, jitter, flicker, "
    "melting geometry, warped face, distorted hands, drifting text, unreadable "
    "text, black frame, pillarbox bars")

_VIDU_Q2_MODEL = "viduq2-pro-fast"
_VIDU_Q2_RESOLUTION = "720p"
_VIDU_Q2_MOVEMENT_AMPLITUDES = ("auto", "small", "medium", "large")
_VIDU_Q2_SMOOTH_MARKER = (
    "Vidu Q2 i2v motion; preserve the still and move gently.")
_VIDU_Q2_SMOOTH_MOTION_CLAUSE = (
    "Generate one continuous image-to-video shot from the supplied start "
    "frame. Preserve the subject identity, composition, period-radio visual "
    "style, lighting, and aspect ratio. Use gentle physically continuous "
    "motion with subtle parallax. No sudden reframing, jump cuts, rapid zooms, "
    "melting geometry, warped faces, drifting text, black frames, or bars. "
    f"{_VIDU_Q2_SMOOTH_MARKER}")
_VIDU_Q2_PROMPT_VARIANT = "vidu_q2_pro_fast_720p_smooth_v1"


def _condition_kling_avatar_prompt(prompt: str) -> "tuple[str, dict]":
    original = str(prompt or "")
    if _KLING_AVATAR_MARKER in original:
        conditioned = original
    elif original.strip():
        conditioned = original.rstrip() + "\n\n" + _KLING_AVATAR_BASE_CLAUSE
    else:
        conditioned = _KLING_AVATAR_BASE_CLAUSE
    conditioned = append_visual_safety_clause(conditioned)
    return conditioned, {
        "changed": conditioned != original,
        "original_sha8": _sha8(original),
        "conditioned_sha8": _sha8(conditioned),
        "original_excerpt": _log_excerpt(original),
        "conditioned_excerpt": _log_excerpt(conditioned),
    }


def _condition_wan_prompt(prompt: str) -> "tuple[str, dict]":
    original = str(prompt)
    if not original.strip():
        raise ValueError("Wan prompt conditioner requires non-empty prompt")
    if _WAN_SMOOTH_MARKER in original:
        conditioned = original
    else:
        conditioned = original.rstrip() + "\n\n" + _WAN_SMOOTH_MOTION_CLAUSE
    return conditioned, {
        "changed": conditioned != original,
        "original_sha8": _sha8(original),
        "conditioned_sha8": _sha8(conditioned),
        "original_excerpt": _log_excerpt(original),
        "conditioned_excerpt": _log_excerpt(conditioned),
    }


def _condition_vidu_q2_prompt(prompt: str) -> "tuple[str, dict]":
    original = str(prompt)
    if not original.strip():
        raise ValueError("Vidu Q2 prompt conditioner requires non-empty prompt")
    if _VIDU_Q2_SMOOTH_MARKER in original:
        conditioned = original
    else:
        conditioned = original.rstrip() + "\n\n" + _VIDU_Q2_SMOOTH_MOTION_CLAUSE
    return conditioned, {
        "changed": conditioned != original,
        "original_sha8": _sha8(original),
        "conditioned_sha8": _sha8(conditioned),
        "original_excerpt": _log_excerpt(original),
        "conditioned_excerpt": _log_excerpt(conditioned),
    }


def _est_usd() -> float:
    try:
        return float(os.environ.get("OTR_CLOUD_VIDEO_EST_USD", "0.50"))
    except ValueError:
        return 0.50


def _timeout_s() -> float:
    try:
        return float(os.environ.get("OTR_CLOUD_VIDEO_TIMEOUT_S", "900"))
    except ValueError:
        return 900.0


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be boolean-like (true/false, 1/0)")


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _log_excerpt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:160]


def _condition_seedance_prompt(prompt: str) -> "tuple[str, dict]":
    """Seedance 2 stabilizer: keep source style dynamic, provider prompt smooth.

    The marker check runs before softening so repeated calls do not rewrite the
    appended negative-motion clause ("No whip pans..." etc.).
    """
    original = str(prompt)
    if not original.strip():
        raise ValueError("Seedance prompt conditioner requires non-empty prompt")

    def _meta(conditioned: str, changed: bool,
              softeners_applied: "list[str]") -> dict:
        return {
            "changed": bool(changed),
            "original_sha8": _sha8(original),
            "conditioned_sha8": _sha8(conditioned),
            "original_excerpt": _log_excerpt(original),
            "conditioned_excerpt": _log_excerpt(conditioned),
            "softeners_applied": list(softeners_applied),
        }

    if _SEEDANCE_SMOOTH_MARKER in original:
        return original, _meta(original, False, [])

    softened = original
    applied: "list[str]" = []
    for softener_id, pattern, replacement in _SEEDANCE_PROMPT_SOFTENERS:
        softened_next, count = pattern.subn(replacement, softened)
        if count:
            applied.append(softener_id)
            softened = softened_next

    conditioned = softened.rstrip() + "\n\n" + _SEEDANCE_SMOOTH_MOTION_CLAUSE
    return conditioned, _meta(conditioned, conditioned != original, applied)


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _ref_path(ref) -> str:
    """A request asset ref -> filesystem path (mirrors eng_visualizer)."""
    if not ref:
        return ""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        return str(ref.get("path") or ref.get("wav_path") or "")
    return str(getattr(ref, "path", "") or "")


def _load_image_tensor(path: str):
    """PNG/JPG -> comfy IMAGE tensor [1,H,W,C] float32 0-1 (lazy imports)."""
    import numpy as np
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


def _load_audio_dict(path: str):
    """WAV -> comfy AUDIO dict {waveform [1,C,T], sample_rate} (soundfile --
    torchaudio save/load is torchcodec-broken on this box)."""
    import soundfile as sf
    import torch
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return {"waveform": torch.from_numpy(data.T)[None, ...],
            "sample_rate": int(sr)}


class _CloudVideoBase:
    """Shared S3 adapter mechanics; subclasses pin the row identity."""

    # --- registry-facing core ---
    roles: tuple = ()
    default_roles: tuple = ()            # NEVER automatic in S3-core
    commercial_clean = True              # partner API rows; ToS audit rides S0 docs
    # C2-C6 registry-IS-the-menu: NO registered engine declares a flag.
    # Operator directive 2026-07-02: no hidden enable switch either -- the
    # dropdown pick IS the enable; auth fails LOUD at invoke if missing.
    requires_flag = None
    invocable = True
    invocability_reason = ""

    # --- row identity (subclasses) ---
    name = ""
    node_key = ""                        # partner_nodes.yaml row key
    family = ""
    required_inputs: tuple = ()
    reactivity = ""                      # required_audio_ref|lipsync_overlay|mute_only
    must_strip_audio = True
    render_aspect = "wide"

    def load(self) -> None:              # no local weights
        return None

    def unload(self) -> None:
        return None

    # ---- render lifecycle -------------------------------------------------
    def assert_usable(self, host_caps, profile, request_template=None):
        # NO enable-flag check (operator directive 2026-07-02): the dropdown
        # pick is the enable. Credentials resolve fail-closed at invoke time
        # (resolve_auth names OTR_COMFY_API_KEY / logged-in Comfy hidden
        # inputs) -- hidden auth only exists in the prompt context, so a
        # resolve-time env check would wrongly block logged-in desktop users.
        import shutil

        from .._otr_shared.ffprobe import resolve_ffprobe
        # The gate asks the SAME question the canonicalizer will: ffmpeg is
        # still PATH-only (it runs a literal "ffmpeg"), but ffprobe is now
        # found the way the probe itself finds it, so an OTR_FFPROBE box is no
        # longer refused over a tool it has.
        if not shutil.which("ffmpeg") or not resolve_ffprobe():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg not on PATH, or no ffprobe (OTR_FFPROBE / PATH / "
                "ffmpeg sibling) -- the cloud video canonicalizer strips "
                "provider audio via ffmpeg (must_strip_audio)",
                kind="video")
        from .._otr_shared.cloud_media_invoke import partner_rows
        row = partner_rows().get(self.node_key)
        if not isinstance(row, dict) or str(row.get("status")) != "OK":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                f"partner pin row {self.node_key!r} missing or not OK -- "
                f"re-pin via scripts/otr_pin_partner_nodes.py", kind="video")

    def prepare(self, host_caps, profile, session_ctx):
        return {}

    def _partner_inputs(self, request) -> dict:
        raise NotImplementedError

    def _canonical_video_asset(self, raw, request):
        from .._otr_shared.cloud_media_canonical import (
            canonicalize_video, cloud_delivery_wh)
        canvas = _req_get(request, "canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        rw = int(c_get("w", 0) or 0)
        rh = int(c_get("h", 0) or 0)
        # TRUE 1080p cloud delivery (operator 2026-07-03): conform the provider
        # clip to a real 1080p canvas, NOT the smaller per-family request canvas
        # (canonicalize_video otherwise downscales the provider's 1080p output to
        # e.g. 1472x832). Orientation-preserving + CLOUD-LANE ONLY -- locals keep
        # their own render res. Env OTR_CLOUD_VIDEO_CANVAS[_PORTRAIT].
        tw, th = cloud_delivery_wh(
            rw, rh, land_env="OTR_CLOUD_VIDEO_CANVAS",
            port_env="OTR_CLOUD_VIDEO_CANVAS_PORTRAIT")
        return canonicalize_video(raw, {
            "w": tw, "h": th, "fps": int(c_get("fps", 25) or 25),
        })

    def render_clip(self, request, prepared):
        from .._otr_shared.cloud_media_invoke import invoke_partner_node
        inputs = self._partner_inputs(request)
        _LOG.warning(
            "[OTR video] CLOUD render: %s -> partner %s (est<=$%.2f, "
            "timeout %.0fs, shot %s)", self.name, self.node_key, _est_usd(),
            _timeout_s(), _req_get(request, "shot_id"))
        return invoke_partner_node(
            self.node_key, inputs,
            timeout_s=_timeout_s(), estimated_usd=_est_usd())

    def canonicalize(self, raw, request, profile):
        asset = self._canonical_video_asset(raw, request)
        frame_count = int(round((asset.duration_s or 0.0) * (asset.fps or 0.0)))
        return {
            "clip_id": _req_get(request, "shot_id") or f"{self.name}_clip",
            "type": "video", "path": str(asset.path),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(asset.fps or 25), "frame_count": frame_count,
            "has_audio": False,          # strip PROVEN in canonicalize_video
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            "provider_job_id": asset.provider_job_id,
            "content_sha256": asset.sha256,
            "actual_duration_s": asset.duration_s,
            # THE HONESTY RECEIPTS (2026-08-06). Every PROVIDER surface carried
            # ZERO references to these two fields before today, which is the
            # dormancy the step-3 v1 contract closes: a cloud lane that answers
            # nothing looks exactly like a local lane that pads without saying
            # so, and no rule could tell them apart.
            #
            # A provider clip is native BY CONSTRUCTION, and the reason is
            # structural rather than a claim about the vendor: the delivered
            # asset is downloaded and re-containered whole, and OTR owns no code
            # on this path that could lengthen it. ``frame_count`` is derived
            # from the asset's OWN measured duration and fps just above, so the
            # native count is the same number by the same derivation -- not a
            # second measurement that could disagree with the first.
            "native_frame_count": frame_count,
            "extension_mode": "none",
        }

    def teardown(self, prepared) -> None:
        return None

    # ---- shared input builders --------------------------------------------
    def _seed(self, request) -> int:
        seeds = _req_get(request, "seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return int(s_get("request_seed", 0) or 0)

    def _seed_i32(self, request) -> int:
        """Partner V3 video nodes declare seed max=2147483647."""
        return self._seed(request) & 0x7FFFFFFF

    def _text_prompt_input(self, request) -> str:
        prompt = str(_req_get(request, "text_prompt")
                     or _req_get(request, "prompt") or "").strip()
        if not prompt:
            raise RuntimeError(
                f"{self.name}: text_prompt missing/blank -- the partner V3 "
                f"model schema requires model['prompt']; NO FALLBACK")
        return append_visual_safety_clause(prompt)

    def _duration_seconds(self, request, *, env: str, default: int,
                          min_s: int, max_s: int) -> int:
        raw = os.environ.get(env, "").strip()
        if raw:
            try:
                secs = int(raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"{self.name}: {env} must be an integer number of seconds"
                ) from exc
        else:
            canvas = _req_get(request, "canvas") or {}
            c_get = canvas.get if isinstance(canvas, dict) else (
                lambda k, d=None: getattr(canvas, k, d))
            fps = int(c_get("fps", 25) or 25) or 25
            timing = _req_get(request, "timing") or {}
            t_get = timing.get if isinstance(timing, dict) else (
                lambda k, d=None: getattr(timing, k, d))
            n = int(t_get("target_frame_count", 0) or 0)
            # Provider clips can always be trimmed downstream, but a too-short
            # provider clip becomes a freeze-hold. Ceil the audio-derived frame
            # target so cloud engines render enough motion for fractional beats.
            secs = int((n + fps - 1) // fps) if n else default
        return max(min_s, min(max_s, secs))

    def _choice(self, env: str, default: str, allowed: tuple[str, ...],
                *, transform=None) -> str:
        value = os.environ.get(env, "").strip() or default
        if transform is not None:
            value = transform(value)
        if value not in allowed:
            raise RuntimeError(
                f"{self.name}: {env}={value!r} is unsupported; expected one "
                f"of {allowed}")
        return value

    def _init_image_ref(self, request):
        """Resolve the init-image ref. Real ``render_driver.build_request()``
        output carries it under ``asset_refs["init_image"]`` (the scene/word
        still); older hand-built dict requests may put it TOP-LEVEL. Try
        asset_refs FIRST, then top-level (the eng_humo resolution order). Pure."""
        assets = _req_get(request, "asset_refs") or {}
        a_get = assets.get if isinstance(assets, dict) else (
            lambda k, d=None: getattr(assets, k, d))
        return a_get("init_image") or _req_get(request, "init_image")

    def _init_image_input(self, request):
        path = _ref_path(self._init_image_ref(request))
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                f"{self.name}: init_image missing/absent on disk ({path!r}) "
                f"-- NO FALLBACK (required_inputs={self.required_inputs}; "
                f"checked asset_refs['init_image'] + top-level init_image)")
        return _load_image_tensor(path)

    def _audio_input(self, request, *, min_duration_s: float | None = None,
                     max_duration_s: float | None = None,
                     pad_to_min: bool = False):
        path = _ref_path(_req_get(request, "audio_ref"))
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                f"{self.name}: audio_ref missing/absent on disk ({path!r}) "
                f"-- reactivity={self.reactivity}, NO FALLBACK")
        audio = _load_audio_dict(path)
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        samples = int(waveform.shape[-1])
        duration = samples / float(sr) if sr > 0 else 0.0
        if min_duration_s is not None and duration < float(min_duration_s):
            if not pad_to_min:
                raise RuntimeError(
                    f"{self.name}: audio_ref duration {duration:.3f}s is below "
                    f"provider minimum {float(min_duration_s):.3f}s -- "
                    f"NO FALLBACK")
            import torch
            target_samples = int(round(float(min_duration_s) * sr))
            pad_n = max(0, target_samples - samples)
            if pad_n:
                pad = torch.zeros(
                    *waveform.shape[:-1], pad_n,
                    dtype=waveform.dtype, device=waveform.device)
                audio = dict(audio)
                audio["waveform"] = torch.cat([waveform, pad], dim=-1)
                _LOG.warning(
                    "[OTR.cloud.%s] padded request audio %.3fs -> %.3fs "
                    "for provider minimum; episode timeline still trims to "
                    "target frames", self.name, duration, float(min_duration_s))
        if max_duration_s is not None and duration > float(max_duration_s):
            raise RuntimeError(
                f"{self.name}: audio_ref duration {duration:.3f}s exceeds "
                f"provider maximum {float(max_duration_s):.3f}s -- "
                f"split the shot before selecting this engine")
        return audio


class CloudKlingAvatarEngine(_CloudVideoBase):
    """Kling avatar: TALKING default row (audio CONDITIONS the clip)."""

    name = "cloud_kling_avatar"
    node_key = "cloud_kling_avatar"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). Kling sends NO duration parameter at all --
    #: the clip is as long as the sound_file, which _audio_input
    #: bounds at min_duration_s=2.0 / max_duration_s=300.0.
    #: 2-300 s at the 25 fps canvas rate = 50-7500 frames.
    frame_contract = FrameContract(
        min_frames=50,
        max_frames=7500,
        quantum=1,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    family = "audio_driven_face"
    required_inputs = ("init_image", "audio_ref")
    reactivity = "required_audio_ref"
    #: S1 per-model still plan -- portrait REQUIRED because Kling consumes an
    #: init face every beat (audio-driven-face family; spec section 3).
    still_plan = _CLOUD_KLING_AVATAR_PLAN

    def _mode(self) -> str:
        raw = os.environ.get(_KLING_MODE_ENV, _KLING_MODE_DEFAULT).strip()
        folded = raw.lower()
        mode = _KLING_MODE_ALIASES.get(folded, folded)
        if mode not in _KLING_MODES:
            raise RuntimeError(
                f"{self.name}: {_KLING_MODE_ENV}={raw!r} is unsupported; "
                f"expected one of {_KLING_MODES} or known aliases")
        return mode

    def _partner_inputs(self, request):
        prompt, prompt_meta = _condition_kling_avatar_prompt(
            str(_req_get(request, "text_prompt") or ""))
        log_fields = dict(prompt_meta)
        log_fields.update({
            "engine": self.name,
            "prompt_variant": "kling_avatar_broadcast_v1",
        })
        _LOG.info("[OTR.cloud.kling_avatar] prompt_conditioner %s",
                  json.dumps(log_fields, sort_keys=True))
        return {
            "image": self._init_image_input(request),
            "sound_file": self._audio_input(
                request, min_duration_s=2.0, max_duration_s=300.0,
                pad_to_min=True),
            "mode": self._mode(),
            "seed": self._seed_i32(request),
            "prompt": prompt,
        }


class CloudSeedance2Engine(_CloudVideoBase):
    """ByteDance Seedance 2 reference row: music/b-roll reactive default."""

    name = "cloud_seedance_2"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). OTR_CLOUD_SEEDANCE_DURATION, default 7 s,
    #: clamped to 4-15 s at the call site. That clamp is 7c's to
    #: delete; this ladder is what refuses in its place.
    #: 4-15 s at the 25 fps canvas rate = 100-375 frames.
    frame_contract = FrameContract(
        min_frames=100,
        max_frames=375,
        quantum=25,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    node_key = "cloud_seedance_2"
    family = "audio_conditioned_video"
    required_inputs = ("init_image", "audio_ref", "text_prompt")
    reactivity = "required_audio_ref"
    #: S1 per-model still plan (Shape A base).
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN

    def _model_label(self) -> str:
        from .._otr_shared.cloud_model_ids import resolve_model_id
        value = resolve_model_id(self.node_key)
        label = _SEEDANCE_MODEL_ALIASES.get(value, value)
        if label not in _SEEDANCE_RESOLUTIONS:
            raise RuntimeError(
                f"{self.name}: unsupported Seedance model selector {value!r}; "
                f"expected one of {tuple(_SEEDANCE_RESOLUTIONS)} or known "
                f"provider-id aliases")
        return label

    def _partner_inputs(self, request):
        model_label = self._model_label()
        prompt, prompt_meta = _condition_seedance_prompt(
            self._text_prompt_input(request))
        duration = self._duration_seconds(
            request, env="OTR_CLOUD_SEEDANCE_DURATION",
            default=7, min_s=4, max_s=15)
        log_fields = dict(prompt_meta)
        log_fields.update({
            "engine": self.name,
            "prompt_variant": _SEEDANCE_PROMPT_VARIANT,
            "seedance_requested_duration_s": duration,
        })
        _LOG.info("[OTR.cloud.seedance] prompt_conditioner %s",
                  json.dumps(log_fields, sort_keys=True))
        return {
            "model": {
                "model": model_label,
                "prompt": prompt,
                "resolution": self._choice(
                    "OTR_CLOUD_SEEDANCE_RESOLUTION", "720p",
                    _SEEDANCE_RESOLUTIONS[model_label], transform=str.lower),
                "ratio": self._choice(
                    "OTR_CLOUD_SEEDANCE_RATIO", "adaptive",
                    _SEEDANCE_RATIOS),
                "duration": duration,
                # OTR always strips provider audio at canonicalize; do not ask
                # Seedance to synthesize a second mix just to discard it.
                "generate_audio": False,
                "reference_images": {"image_1": self._init_image_input(request)},
                "reference_audios": {"audio_1": self._audio_input(request)},
            },
            "seed": self._seed_i32(request),
            "watermark": False,
        }


class CloudWanI2VEngine(_CloudVideoBase):
    """Wan image-to-video: the MUTE opt-down row (explicit picks only)."""

    name = "cloud_wan_i2v"
    node_key = "cloud_wan_i2v"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). OTR_CLOUD_WAN_DURATION, default 5 s, clamped
    #: to 2-15 s at the call site. Inherited by cloud_wan_i2v_audio,
    #: which sends the same provider duration.
    #: 2-15 s at the 25 fps canvas rate = 50-375 frames.
    frame_contract = FrameContract(
        min_frames=50,
        max_frames=375,
        quantum=25,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"
    #: S1 per-model still plan (Shape A base).
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN

    def _model_selector(self) -> str:
        from .._otr_shared.cloud_model_ids import resolve_model_id
        value = resolve_model_id(self.node_key)
        folded = re.sub(r"[\s_]+", "-", value).lower()
        model = _WAN_MODEL_ALIASES.get(value, _WAN_MODEL_ALIASES.get(folded, value))
        if model != value:
            _LOG.warning("[OTR.cloud.wan] normalized model selector %r -> %r",
                         value, model)
        if model not in _WAN_MODELS:
            raise RuntimeError(
                f"{self.name}: unsupported Wan model selector {value!r}; "
                f"expected one of {_WAN_MODELS} or known legacy aliases")
        return model

    def _partner_inputs(self, request):
        model = self._model_selector()
        prompt, prompt_meta = _condition_wan_prompt(
            self._text_prompt_input(request))
        log_fields = dict(prompt_meta)
        log_fields.update({
            "engine": self.name,
            "prompt_variant": _WAN_PROMPT_VARIANT,
        })
        _LOG.info("[OTR.cloud.wan] prompt_conditioner %s",
                  json.dumps(log_fields, sort_keys=True))
        return {
            "first_frame": self._init_image_input(request),
            "model": {
                "model": model,
                "prompt": prompt,
                "negative_prompt": visual_safety_negative(os.environ.get(
                    "OTR_CLOUD_WAN_NEGATIVE_PROMPT", "").strip()
                    or _WAN_NEGATIVE_DEFAULT),
                "resolution": self._choice(
                    "OTR_CLOUD_WAN_RESOLUTION", "720P",
                    _WAN_RESOLUTIONS, transform=str.upper),
                "duration": self._duration_seconds(
                    request, env="OTR_CLOUD_WAN_DURATION", default=5,
                    min_s=2, max_s=15),
            },
            "prompt_extend": _bool_env("OTR_CLOUD_WAN_PROMPT_EXTEND", False),
            "seed": self._seed_i32(request),
            "watermark": False,
        }


class CloudWanI2VAudioEngine(CloudWanI2VEngine):
    """Wan image-to-video with its installed optional driving-audio input."""

    name = "cloud_wan_i2v_audio"
    node_key = "cloud_wan_i2v"
    family = "audio_conditioned_video"
    required_inputs = ("init_image", "audio_ref", "text_prompt")
    reactivity = "required_audio_ref"
    #: S1 per-model still plan (Shape A base) -- declared explicitly rather
    #: than inherited from CloudWanI2VEngine so the audit sees each adapter
    #: owning its own plan (spec section 6, "on each adapter").
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN

    def _partner_inputs(self, request):
        inputs = super()._partner_inputs(request)
        inputs["audio"] = self._audio_input(
            request, min_duration_s=2.0, max_duration_s=30.0,
            pad_to_min=True)
        return inputs


class CloudViduQ2ProFast720pEngine(_CloudVideoBase):
    """Vidu Q2 pro-fast image-to-video, fixed to the cheap 720p tier."""

    name = "cloud_vidu_q2_pro_fast_720p"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). OTR_CLOUD_VIDU_Q2_DURATION, default 5 s,
    #: clamped to 1-10 s at the call site.
    #: 1-10 s at the 25 fps canvas rate = 25-250 frames.
    frame_contract = FrameContract(
        min_frames=25,
        max_frames=250,
        quantum=25,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    node_key = "cloud_vidu_q2_i2v"
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"
    #: S1 per-model still plan (Shape A base).
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN

    def _movement_amplitude(self) -> str:
        return self._choice(
            "OTR_CLOUD_VIDU_Q2_MOVEMENT", "auto",
            _VIDU_Q2_MOVEMENT_AMPLITUDES, transform=str.lower)

    def _partner_inputs(self, request):
        prompt, prompt_meta = _condition_vidu_q2_prompt(
            self._text_prompt_input(request))
        log_fields = dict(prompt_meta)
        duration = self._duration_seconds(
            request, env="OTR_CLOUD_VIDU_Q2_DURATION",
            default=5, min_s=1, max_s=10)
        log_fields.update({
            "engine": self.name,
            "prompt_variant": _VIDU_Q2_PROMPT_VARIANT,
            "vidu_requested_duration_s": duration,
            "vidu_model": _VIDU_Q2_MODEL,
            "vidu_resolution": _VIDU_Q2_RESOLUTION,
        })
        _LOG.info("[OTR.cloud.vidu_q2] prompt_conditioner %s",
                  json.dumps(log_fields, sort_keys=True))
        return {
            "model": _VIDU_Q2_MODEL,
            "image": self._init_image_input(request),
            "prompt": prompt,
            "duration": duration,
            "seed": self._seed_i32(request),
            "resolution": _VIDU_Q2_RESOLUTION,
            "movement_amplitude": self._movement_amplitude(),
        }


# word_razzle Phase 1 (2026-07-03): the ANIMATED word-card cloud i2v engine.
# Pixverse is the --audit-i2v Phase-0 pick (promptable, non-V3, required image
# init + prompt + seed + duration_seconds + motion_mode). A word_razzle beat's
# base still (a still_word / word-card still, or any scene still) is fed as the
# init image; the engine adds a LIVING-POSTER world-motion prompt whose whole
# job is to keep the lettering readable EVERY frame (the operator acceptance
# bar). motion_mode / quality / duration_seconds are provider COMBOs (the pin
# excludes option lists) so the adapter ships documented defaults, all
# env-overridable -- the same discipline as the kling mode default. mute_only:
# the provider audio is stripped (must_strip_audio); the master mix is frozen
# upstream and muxed LAST. NO FALLBACK: a missing init still fails LOUD.
_RAZZLE_MOTION_ENV = "OTR_CLOUD_RAZZLE_MOTION_PROMPT"
_RAZZLE_MOTION_DEFAULT = (
    "living period poster, gentle atmospheric motion around the lettering -- "
    "drifting mist, soft neon flicker, subtle parallax depth -- the words stay "
    "crisp, sharp and fully legible in every frame, letterforms never warp, "
    "melt or distort")
_RAZZLE_NEG_ENV = "OTR_CLOUD_RAZZLE_NEG"
_RAZZLE_NEG_DEFAULT = visual_safety_negative(
    "warped text, melting letters, distorted typography, "
    "illegible words, garbled text, flickering letters")


class CloudWordRazzleEngine(_CloudVideoBase):
    """word_razzle: animate a word-card still into a living period poster."""

    name = "word_razzle"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). Pixverse serves a fixed
    #: TWO-ENTRY menu, 5 s or 8 s -- `_duration_seconds` ends
    #: `return 8 if secs > 5 else 5`, so a 12-second beat silently
    #: becomes an 8-second clip today. As frames at the 25 fps canvas
    #: rate: 5 s = 125, 8 s = 200.
    frame_contract = FrameContract(
        discrete_frames=(125, 200),
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    node_key = "cloud_pixverse_i2v"
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"
    #: S1 per-model still plan (Shape A base) -- the word-card init still is
    #: minted by an image engine upstream (still_word / any scene still), so
    #: the plan matches the standard scene-spine shape.
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN

    def _razzle_prompt(self, request) -> str:
        """The world-motion + text-preservation prompt. The base motion clause
        (env-overridable) LEADS; the beat's own text_prompt (scene/subject) is
        appended so the animation matches the beat, never overriding the
        readability directive."""
        motion = os.environ.get(_RAZZLE_MOTION_ENV, "").strip() or _RAZZLE_MOTION_DEFAULT
        beat = str(_req_get(request, "text_prompt") or "").strip()
        prompt = f"{motion}. {beat}".strip().rstrip(".") if beat else motion
        return append_visual_safety_clause(prompt)

    def _duration_seconds(self, request) -> int:
        """The provider duration (seconds). Derived from the beat's frame
        target (timing.target_frame_count / fps) and clamped to Pixverse's
        supported 5s / 8s tiers; env OTR_CLOUD_PIXVERSE_DURATION overrides."""
        env = os.environ.get("OTR_CLOUD_PIXVERSE_DURATION", "").strip()
        if env:
            # THE ENV MAY NOT LEAVE THE MENU (chunk 7b, 2026-07-26). This
            # branch used to `return int(env)` outright, skipping the 5/8 bucket
            # ladder every other path goes through -- so an operator could send
            # Pixverse a 20-second ask that this adapter's own declared
            # ``discrete_frames=(125, 200)`` says is impossible, and the
            # coverage plan had already partitioned the beat over 5s/8s clips.
            # A non-integer was silently swallowed here too, the only duration
            # env in this file that failed quietly rather than loud.
            legal = sorted(
                int(d) // _RAZZLE_FPS for d in type(self).frame_contract.discrete_frames)
            try:
                secs = int(env)
            except ValueError as exc:
                raise _fc.ContractEnvConflict(
                    "%s: OTR_CLOUD_PIXVERSE_DURATION=%r is not a whole number "
                    "of seconds. Pixverse serves %s s and nothing between them."
                    % (self.name, env, legal)) from exc
            if secs not in legal:
                raise _fc.ContractEnvConflict(
                    "%s: OTR_CLOUD_PIXVERSE_DURATION=%d is not on Pixverse's "
                    "menu %s s, which this adapter declares as "
                    "discrete_frames=%r at %d fps. The beat was already "
                    "partitioned over those lengths. NO FALLBACK."
                    % (self.name, secs, legal,
                       type(self).frame_contract.discrete_frames, _RAZZLE_FPS))
            return secs
        canvas = _req_get(request, "canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        fps = int(c_get("fps", 25) or 25) or 25
        timing = _req_get(request, "timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        n = int(t_get("target_frame_count", 0) or 0)
        secs = int(round(n / fps)) if n else 5
        return 8 if secs > 5 else 5

    def _partner_inputs(self, request):
        return {
            "image": self._init_image_input(request),
            "prompt": self._razzle_prompt(request),
            "negative_prompt": visual_safety_negative(
                os.environ.get(_RAZZLE_NEG_ENV, "").strip()
                or _RAZZLE_NEG_DEFAULT),
            "motion_mode": os.environ.get("OTR_CLOUD_PIXVERSE_MOTION", "normal"),
            "quality": os.environ.get("OTR_CLOUD_PIXVERSE_QUALITY", "1080p"),
            "duration_seconds": self._duration_seconds(request),
            "seed": self._seed(request),
        }


KlingAvatar = CloudKlingAvatarEngine()
Seedance2 = CloudSeedance2Engine()
WanI2V = CloudWanI2VEngine()
WanI2VAudio = CloudWanI2VAudioEngine()
ViduQ2ProFast720p = CloudViduQ2ProFast720pEngine()
WordRazzle = CloudWordRazzleEngine()

for _eng in (
        KlingAvatar, Seedance2, WanI2V, WanI2VAudio,
        ViduQ2ProFast720p, WordRazzle):
    register(_eng)

__all__ = [
    "CloudKlingAvatarEngine", "CloudSeedance2Engine",
    "CloudWanI2VEngine", "CloudWanI2VAudioEngine",
    "CloudViduQ2ProFast720pEngine",
    "CloudWordRazzleEngine",
]
