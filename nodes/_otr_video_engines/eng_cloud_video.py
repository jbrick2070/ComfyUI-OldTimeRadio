"""Cloud partner VIDEO adapters -- S3 core (pass04 secs 5+7, operator GO
2026-07-02 evening: "code the cloud video plan").

Four rows from the S0 pin table, invoked through the S0 bridge
(``invoke_partner_node``) and conformed by ``canonicalize_video``:

    cloud_kling_avatar   required_audio_ref   (init_image, audio_ref)
    cloud_seedance_2     required_audio_ref   (init_image, audio_ref)
    cloud_wan_i2v        mute_only            (init_image, text_prompt)
    cloud_wan_i2v_audio  required_audio_ref   (init_image, audio_ref)
    cloud_vidu_q2_pro_fast_720p mute_only     (init_image, text_prompt)
    cloud_ltx25_foley_plus mute_only          (init_image, text_prompt;
                                              generate_audio harvested as foley)
    cloud_ltx25_audio_in   required_audio_ref (init_image, audio_ref)

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

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_LOG = logging.getLogger("OTR.video.eng_cloud_video")


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
                     ("cinematic three-quarter framing, the subject shown "
                      "whole with clear space around it inside frame, "
                      "balanced composition")),
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


#: Kling Avatar (Comfy ``KlingAvatarNode``) is audio-in, not I2V. The
#: canonical node takes ONE reference photo + ``sound_file`` + an optional
#: prompt for actions / emotions / camera -- it does not take a scene still.
#: Minting Shape-A wide scenes here spent image credits on stills the adapter
#: never sends, and ``aspect="inherit_engine"`` against ``render_aspect=wide``
#: minted 16:9 "portraits" instead of a face photo.
#: https://docs.comfy.org/built-in-nodes/KlingAvatarNode
_CLOUD_KLING_AVATAR_PLAN = (
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="portrait",
                 required="always",
                 framing_geometry=(
                     ("single avatar reference photo, head and shoulders, "
                      "face clearly visible, mouth unobstructed, subject "
                      "centred with natural headroom above the head "
                      "(never crop the top of the head); fill the frame "
                      "with the person -- this image is Kling Avatar's "
                      "only visual input")),
                 style_tail_policy="full"),
)

#: Comfy KlingAvatarNode image gate: min 300px on both edges, aspect
#: between 1:2.5 and 2.5:1.
_KLING_AVATAR_MIN_PX = 300
_KLING_AVATAR_ASPECT_MIN = 1.0 / 2.5
_KLING_AVATAR_ASPECT_MAX = 2.5

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
    "Kling avatar audio-in; lip sync leads, action is full and sustained.")
#: SAME ACTION BUDGET AS WAN / VIDU / SEEDANCE, plus the audio-in contract.
#: Operator 2026-09-15: no silly subtle prompts -- keep the shot's action
#: prompt and append the lane clause, exactly as the other cloud lanes do.
#: Operator 2026-08-27 still stands for the extra sentence: audio-in must
#: name lip-sync and that we feed both the beat audio and the spoken line.
#: Artifact guards (whip pans, jump cuts, warped faces) stay; "subtle" /
#: "small natural head movement" / "no exaggerated gestures" stay gone.
_KLING_AVATAR_BASE_CLAUSE = (
    "Generate one continuous audio-driven shot from the reference photo and "
    "the supplied audio. Natural lip sync follows the supplied audio exactly "
    "and matches the spoken line word for word -- feed both the beat audio "
    "and the dialogue; the sync leads identity. The subject performs a full, "
    "decisive action that develops across the shot -- turning, reaching, "
    "rising, gesturing, crossing the space -- and lands on a clear final "
    "position, with a purposeful camera move that follows the action. Motion "
    "begins immediately in the first frame and is sustained throughout. Keep "
    "the face unobstructed so the mouth stays readable. Preserve the "
    "reference-image subject and style. No whip pans, jump cuts, melting "
    "geometry, warped faces, drifting text, black frames, or pillarbox bars. "
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
#: MOTION RAISED, ARTIFACT GUARDS KEPT (2026-08-27, Option B).
#: What changed: "slow dolly with gentle ease-in and ease-out", "remains gentle
#: and continuous throughout" and "Gentle parallax only" were DAMPING -- the
#: 08-17 antipattern that authored the silly pan. What did NOT change: every
#: artifact guard. Whip pans, handheld shake, sudden reframing, jump cuts and
#: rapid zooms are documented FAILURE MODES on this family, not stillness, and
#: the reference-image preservation is what holds continuity.
#: Budget (operator matrix): one purposeful audio-reactive subject action plus
#: one stabilized camera move. Audio-CONDITIONED -- never promise phoneme sync.
_SEEDANCE_SMOOTH_MARKER = (
    "Full sustained action; motion physically continuous and uncut.")
_SEEDANCE_SMOOTH_MOTION_CLAUSE = (
    "One continuous uncut shot with strong, decisive movement. The subject "
    "performs a full action that responds to the audio and develops across "
    "the shot -- turning, reaching, rising, crossing the space -- and lands on "
    "a clear final position, with a purposeful camera move that follows the "
    "action. Motion begins immediately in the first frame and is sustained "
    "throughout. Preserve the reference-image subject and style. No whip "
    "pans, handheld shake, sudden reframing, jump cuts, or rapid zooms. "
    f"{_SEEDANCE_SMOOTH_MARKER}")
_SEEDANCE_PROMPT_VARIANT = "seedance_action_v2"
#: ARTIFACT GUARDS ONLY (2026-08-27). Two entries were removed because they
#: were DAMPING rather than protection: "aggressively"->"subtly" (and "subtly"
#: is precisely the word PBUG-20260827-04 banned) and "dynamic dolly
#: push"->"slow controlled dolly push", which would have quietly undone the new
#: motion envelope on every Seedance render. What remains guards against real,
#: documented failure modes on this family -- whip pans, rapid zooms, handheld
#: shake and blown highlights -- and those are kept verbatim.
_SEEDANCE_PROMPT_SOFTENERS = (
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
#: MOTION RAISED, ARTIFACT GUARDS KEPT (2026-08-27). Only one word changed in
#: substance: "slow" became purposeful subject movement. The long exclusion
#: list stays exactly as it is -- whip pans, melting geometry, warped faces,
#: drifting text and pillarbox bars are ARTIFACT guards on a cloud i2v model,
#: not motion damping, and stripping them to chase movement would trade a
#: still-looking render for a broken one. The distinction matters: "slow"
#: describes how the SUBJECT moves; the exclusions describe how the CAMERA and
#: the geometry must not fail.
#: MOTION RAISED, ARTIFACT GUARDS KEPT (2026-08-27, Option B). "gentle
#: parallax" was the only damping phrase here and is replaced by the budget the
#: operator's matrix names: one purposeful action arc, an endpoint, an optional
#: reaction, one camera behaviour. Every geometry and negative guard below is
#: preserved verbatim -- melting geometry and warped faces are failure modes.
_WAN_SMOOTH_MOTION_CLAUSE = (
    "Generate one continuous shot from the first frame. Preserve the "
    "first-frame subject, composition, aspect ratio, lighting, and visual "
    "style. The subject carries out a full, decisive action that develops "
    "across the shot -- turning, reaching, rising, or crossing the space -- and "
    "finishes on a clear final position, with a purposeful camera move "
    "described in its own right; keep the motion physically continuous and "
    "uncut and sustained from the first frame. No whip pans, "
    "handheld shake, sudden reframing, jump cuts, "
    "rapid zooms, melting geometry, warped faces, drifting text, black frames, "
    "or pillarbox bars. "
    f"{_WAN_SMOOTH_MARKER}")
_WAN_PROMPT_VARIANT = "wan_i2v_action_v2"
_WAN_NEGATIVE_DEFAULT = visual_safety_negative(
    "jump cuts, whip pans, rapid zooms, handheld shake, jitter, flicker, "
    "melting geometry, warped face, distorted hands, drifting text, unreadable "
    "text, black frame, pillarbox bars")

_VIDU_Q2_MODEL = "viduq2-pro-fast"
_VIDU_Q2_RESOLUTION = "720p"
_VIDU_Q2_MOVEMENT_AMPLITUDES = ("auto", "small", "medium", "large")

#: SET FROM THE VENDOR SPEC, NOT FROM A RENDER (operator, 2026-08-28: "reach
#: the requirements spec and prompt knob the best we can, we can test later,
#: it's an uncommon option"). Vidu documents `auto` as best for PORTRAIT-style
#: animation and `medium`/`large` as the recommendation for full-body or
#: ACTION scenes; `small` is for subtle motion.
#:
#: This lane asks for an action scene in as many words -- the motion clause
#: below reads "performs a full, decisive action that develops across the shot
#: and completes on a clear final position, with a purposeful camera move".
#: Pairing that request with `auto` was asking for an action and paying for a
#: portrait. The note directly under this block already recorded `medium` as
#: the intent on 2026-08-27; the default simply had not moved.
#:
#: NOT YET PROVEN ON A RENDER, deliberately -- Vidu is a PAID cloud lane and an
#: uncommon option, so the comparison is deferred rather than bought.
_VIDU_Q2_MOVEMENT = "medium"
#: THE CONTRADICTION THIS RESOLVES (2026-08-27). The clause said "move
#: gently" and "subtle parallax" while Option B raises this lane's
#: `movement_amplitude` to `medium` -- paying for more amplitude and asking for
#: less in the same request. The damping goes; every artifact guard stays.
_VIDU_Q2_SMOOTH_MARKER = (
    "Vidu Q2 i2v motion; preserve the still and carry a full action.")
_VIDU_Q2_SMOOTH_MOTION_CLAUSE = (
    "Generate one continuous image-to-video shot from the supplied start "
    "frame. Preserve the subject identity, composition, period-radio visual "
    "style, lighting, and aspect ratio. The subject performs a full, decisive "
    "action that develops across the shot and completes on a clear final "
    "position, with a purposeful camera move, physically continuous and "
    "sustained throughout. No sudden reframing, jump cuts, rapid "
    "zooms, melting geometry, warped faces, drifting text, black frames, or "
    "bars. "
    f"{_VIDU_Q2_SMOOTH_MARKER}")
_VIDU_Q2_PROMPT_VARIANT = "vidu_q2_pro_fast_720p_action_v2"


def _condition_kling_avatar_prompt(prompt: str) -> "tuple[str, dict]":
    """Keep the shot's ACTION prompt, then append the audio-in lip-sync
    clause -- the same append shape as Wan / Vidu / Seedance. Never replace
    the action with a damped talking-head blurb.
    """
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
        return float(otr_env.get("OTR_CLOUD_VIDEO_EST_USD", "0.50"))
    except ValueError:
        return 0.50


#: Live 2026-09-16 Comfy invoice: ltx-2-5-fast 2s billed 78.45 credits
#: (~$0.7845 at $0.01/credit) => $0.392/s. A flat $0.50 reserve on an
#: 8s closer under-counts real spend by ~6x and lets a 5-act blow the
#: wallet while the local cap still thinks it has headroom.
_LTX25_EST_USD_PER_S = 0.40


def ltx25_estimated_usd(duration_s: int, *, floor_usd: float | None = None) -> float:
    """Reserve for one LTX 2.5 Fast/Pro partner clip."""
    try:
        per_s = float(otr_env.get(
            "OTR_CLOUD_LTX25_EST_USD_PER_S", str(_LTX25_EST_USD_PER_S)))
    except ValueError:
        per_s = _LTX25_EST_USD_PER_S
    if floor_usd is None:
        floor_usd = _est_usd()
    return max(float(floor_usd), float(per_s) * max(int(duration_s), 1))


def _timeout_s() -> float:
    try:
        return float(otr_env.get("OTR_CLOUD_VIDEO_TIMEOUT_S", "900"))
    except ValueError:
        return 900.0


def _bool_env(name: str, default: bool) -> bool:
    raw = otr_env.get(name, "").strip().lower()
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


def _assert_kling_avatar_image(tensor) -> None:
    """Fail closed on the Comfy KlingAvatarNode image contract.

    Width and height must be at least 300px; aspect must sit between
    1:2.5 and 2.5:1. A too-small or ultra-wide still is a provider 400,
    not a render we can salvage.
    """
    if tensor is None or not hasattr(tensor, "shape") or tensor.ndim != 4:
        raise RuntimeError(
            "cloud_kling_avatar: init_image is not an IMAGE tensor "
            "[1,H,W,C] -- NO FALLBACK")
    _n, height, width, _c = (int(v) for v in tensor.shape)
    if height < _KLING_AVATAR_MIN_PX or width < _KLING_AVATAR_MIN_PX:
        raise RuntimeError(
            "cloud_kling_avatar: init_image %dx%d is below KlingAvatarNode "
            "minimum %dpx on both edges -- mint a face photo, NO FALLBACK"
            % (width, height, _KLING_AVATAR_MIN_PX))
    aspect = float(width) / float(height) if height else 0.0
    if not (_KLING_AVATAR_ASPECT_MIN <= aspect <= _KLING_AVATAR_ASPECT_MAX):
        raise RuntimeError(
            "cloud_kling_avatar: init_image %dx%d aspect %.4f is outside "
            "KlingAvatarNode 1:2.5 .. 2.5:1 -- NO FALLBACK"
            % (width, height, aspect))


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
    # No local handles: every render_clip is an independent provider call, so
    # BeatSession does not ask this family to name one (registry.VideoEngine).
    session_residency = "remote"

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

    def cloud_selectors(self):
        from .._otr_shared.cloud_slug_preflight import default_partner_selectors
        return default_partner_selectors(self.node_key)

    # ---- render lifecycle -------------------------------------------------
    def assert_usable(self, host_caps, profile, request_template=None):
        # NO enable-flag check (operator directive 2026-07-02): the dropdown
        # pick is the enable. Credentials resolve fail-closed at invoke time
        # (resolve_auth reads the queue's api_key_comfy_org hidden input,
        # which only exists in the prompt context) -- so a resolve-time
        # check here would wrongly block signed-in desktop users.
        from .._otr_shared.ffmpeg import resolve_ffmpeg
        from .._otr_shared.ffprobe import resolve_ffprobe
        # The gate asks the SAME question the canonicalizer will, through the
        # same two owners, so a box configured only through OTR_FFMPEG /
        # OTR_FFPROBE is not refused over tools it has.
        if not resolve_ffmpeg() or not resolve_ffprobe():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg not found (OTR_FFMPEG / PATH), or no ffprobe (OTR_FFPROBE / PATH / "
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
            canonicalize_video, cloud_delivery_wh,
            engine_request_target_frames)
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
        spec = {
            "w": tw, "h": th, "fps": int(c_get("fps", 25) or 25),
        }
        n = engine_request_target_frames(request)
        if n:
            spec["target_frames"] = n
        return canonicalize_video(raw, spec)

    def _estimated_usd(self, request) -> float:
        return _est_usd()

    def render_clip(self, request, prepared):
        from .._otr_shared.cloud_media_invoke import invoke_partner_node
        inputs = self._partner_inputs(request)
        est = self._estimated_usd(request)
        _LOG.warning(
            "[OTR video] CLOUD render: %s -> partner %s (est<=$%.2f, "
            "timeout %.0fs, shot %s)", self.name, self.node_key, est,
            _timeout_s(), _req_get(request, "shot_id"))
        return invoke_partner_node(
            self.node_key, inputs,
            timeout_s=_timeout_s(), estimated_usd=est)

    def canonicalize(self, raw, request, profile):
        from .._otr_shared.cloud_media_canonical import (
            canonical_clip_frame_count)
        asset = self._canonical_video_asset(raw, request)
        frame_count = canonical_clip_frame_count(asset)
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
            # on this path that could lengthen it. ``frame_count`` is the
            # counted length of THAT file after the fps-resample cap to
            # ``segment.render_frames``. Assembly ``trim_tail`` is later.
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
        raw = otr_env.get(env, "").strip()
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
        value = otr_env.get(env, "").strip() or default
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

    def wants_talking_prompt(self):
        """Stills must be minted face-forward with a readable mouth.

        KlingAvatarNode lip-syncs the reference photo to ``sound_file``.
        Without this hook the director stamps talking=false and the image
        phase writes I2V scene stills instead of an avatar photo.
        """
        return True

    def _init_image_input(self, request):
        tensor = super()._init_image_input(request)
        _assert_kling_avatar_image(tensor)
        return tensor

    def _mode(self) -> str:
        raw = otr_env.get(_KLING_MODE_ENV, _KLING_MODE_DEFAULT).strip()
        folded = raw.lower()
        mode = _KLING_MODE_ALIASES.get(folded, folded)
        if mode not in _KLING_MODES:
            raise RuntimeError(
                f"{self.name}: {_KLING_MODE_ENV}={raw!r} is unsupported; "
                f"expected one of {_KLING_MODES} or known aliases")
        return mode

    def cloud_selectors(self):
        raw = otr_env.get(_KLING_MODE_ENV, _KLING_MODE_DEFAULT).strip()
        folded = raw.lower()
        mode = _KLING_MODE_ALIASES.get(folded, folded)
        return {self.node_key: {"mode": (mode,)}}

    def _partner_inputs(self, request):
        prompt, prompt_meta = _condition_kling_avatar_prompt(
            str(_req_get(request, "text_prompt") or ""))
        log_fields = dict(prompt_meta)
        log_fields.update({
            "engine": self.name,
            "prompt_variant": "kling_avatar_action_v2",
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
                "negative_prompt": visual_safety_negative(otr_env.get(
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

    def cloud_selectors(self):
        movement = self._movement_amplitude()
        return {self.node_key: {
            "model": (_VIDU_Q2_MODEL,),
            "resolution": (_VIDU_Q2_RESOLUTION,),
            "movement_amplitude": (movement,),
        }}

    def _movement_amplitude(self) -> str:
        """MEDIUM, on the vendor's own guidance -- see `_VIDU_Q2_MOVEMENT`."""
        return self._choice(
            "OTR_CLOUD_VIDU_Q2_MOVEMENT", _VIDU_Q2_MOVEMENT,
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


#: LTX 2.5 partner I2V duration menu (Fast). Pro stops at 10. Snap, never send
#: a second the combo does not list (7 and 9 are missing on purpose).
_LTX25_FAST_DURATIONS = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)
_LTX25_PRO_DURATIONS = (2, 3, 4, 5, 6, 8, 10)
_LTX25_FAST_LABEL = "LTX-2.5 (Fast)"
_LTX25_PRO_LABEL = "LTX-2.5 (Pro)"
_LTX25_I2V_RESOLUTIONS = (
    "1280x720", "720x1280", "1920x1080", "1080x1920",
    "2560x1440", "1440x2560", "3840x2160", "2160x3840",
)
_LTX25_A2V_RESOLUTIONS = ("1920x1080", "1080x1920")
#: Comfy canonical API template ``api_ltx2_5_i2v`` widgets_values:
#: Fast, duration "8", 1920x1080, fps "25", generate_audio true.
_LTX25_FOLEY_RES_DEFAULT = "1920x1080"
_LTX25_A2V_RES_DEFAULT = "1920x1080"
_LTX25_I2V_FPS_DEFAULT = "25"


def _snap_ltx25_duration(secs, legal):
    secs = max(int(secs), int(legal[0]))
    for item in legal:
        if item >= secs:
            return item
    return int(legal[-1])


class CloudLtx25FoleyPlusEngine(_CloudVideoBase):
    """Cloud analogue of local ``ltx25_foley_plus``.

    NOT audio-in. Partner node ``LtxApi25ImageToVideo`` (Comfy template
    ``api_ltx2_5_i2v``): Fast, 1920x1080, 25 fps, ``generate_audio=True``.
    The native bed is harvested BEFORE ``canonicalize_video`` strips the
    picture, then mixed 0.50/0.50 under the episode master. Family stays
    ``image_to_video`` so ShotLock never asks who owns the lips.
    """

    name = "cloud_ltx25_foley_plus"
    node_key = "cloud_ltx25_i2v"
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN
    frame_contract = FrameContract(
        min_frames=50,
        max_frames=500,
        quantum=25,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )

    def _model_label(self) -> str:
        from .._otr_shared.cloud_model_ids import resolve_model_id
        label = resolve_model_id(self.node_key)
        if label not in (_LTX25_FAST_LABEL, _LTX25_PRO_LABEL):
            raise RuntimeError(
                "%s: unsupported LTX 2.5 model %r; expected %r or %r"
                % (self.name, label, _LTX25_FAST_LABEL, _LTX25_PRO_LABEL))
        return label

    def _duration_menu(self, label):
        return (_LTX25_PRO_DURATIONS if label == _LTX25_PRO_LABEL
                else _LTX25_FAST_DURATIONS)

    def _fps_menu(self, label):
        return (("24", "25", "50") if label == _LTX25_PRO_LABEL
                else ("24", "25", "48", "50"))

    def _i2v_duration(self, request) -> int:
        label = self._model_label()
        legal = self._duration_menu(label)
        return _snap_ltx25_duration(
            self._duration_seconds(
                request, env="OTR_CLOUD_LTX25_DURATION",
                default=8, min_s=legal[0], max_s=legal[-1]),
            legal)

    def _estimated_usd(self, request) -> float:
        return ltx25_estimated_usd(self._i2v_duration(request))

    def _partner_inputs(self, request):
        label = self._model_label()
        duration = self._i2v_duration(request)
        resolution = self._choice(
            "OTR_CLOUD_LTX25_RESOLUTION", _LTX25_FOLEY_RES_DEFAULT,
            _LTX25_I2V_RESOLUTIONS)
        fps = self._choice(
            "OTR_CLOUD_LTX25_FPS", _LTX25_I2V_FPS_DEFAULT, self._fps_menu(label))
        if duration > 10 and (
                int(fps) > 25 or resolution in (
                    "2560x1440", "1440x2560", "3840x2160", "2160x3840")):
            raise RuntimeError(
                "%s: LTX 2.5 durations over 10s require 720p/1080p and 24/25 "
                "fps (got resolution=%s fps=%s)" % (
                    self.name, resolution, fps))
        prompt = self._text_prompt_input(request)
        _LOG.info("[OTR.cloud.ltx25_foley] %s", json.dumps({
            "engine": self.name,
            "ltx25_model": label,
            "ltx25_duration_s": duration,
            "ltx25_resolution": resolution,
            "ltx25_fps": fps,
            "generate_audio": True,
        }, sort_keys=True))
        return {
            "image": self._init_image_input(request),
            "model": {
                "model": label,
                # Combo options in api_ltx2_5_i2v / LtxApi25ImageToVideo are
                # strings ("8", "25"), not ints.
                "duration": str(duration),
                "resolution": resolution,
                "fps": fps,
                "generate_audio": True,
            },
            "prompt": prompt,
            "seed": self._seed(request),
        }

    def canonicalize(self, raw, request, profile):
        from .._otr_shared.cloud_media_backend import CloudErrorCode
        from .._otr_shared.cloud_media_canonical import (
            validate_partner_result)
        from .foley_stems import (
            FoleyStemError, conform_stem_to_frame_count, durable_foley_dir,
            extract_pcm16_wav_from_video, read_pcm16_wav, sha256_of_file,
            write_pcm16_wav,
        )

        raw_path = str(validate_partner_result(dict(raw))["path"])
        dest_dir = durable_foley_dir()
        # JUMP/CHAIN segments share shot_id. Naming the harvest or the
        # durable stem from shot_id made segment 1 overwrite segment 0
        # (live 2026-09-16 FoleyStemError on shot_b004.wav: 10s leftover
        # against a 20s first picture). Local LTX already names from the
        # unique video basename. Harvest runs BEFORE parent canonicalize,
        # so it keys off the partner tmp path; the durable stem keys off
        # clip["path"] after that call. clip_id stays shot_id for trace.
        raw_stem = os.path.splitext(os.path.basename(raw_path))[0]
        if not raw_stem:
            stem_exc = FoleyStemError(
                "cloud Foley harvest has no basename from %r" % raw_path)
            stem_exc.code = CloudErrorCode.CORRUPT_OUTPUT
            raise stem_exc
        harvest_tmp = os.path.join(str(dest_dir), raw_stem + ".src.wav")
        # Harvest the provider bed FIRST. Parent canonicalize_video writes a
        # sibling .canon.mp4 with -an; if that ever became an in-place strip
        # the bed would already be gone. Fail closed before the picture
        # conform so a silent provider file never becomes a mute foley clip.
        try:
            extract_pcm16_wav_from_video(raw_path, harvest_tmp)
            arr, rate = read_pcm16_wav(harvest_tmp)
            clip = super().canonicalize(raw, request, profile)
            video_stem = os.path.splitext(
                os.path.basename(str(clip.get("path") or "")))[0]
            if not video_stem:
                raise FoleyStemError(
                    "cloud Foley cannot name a stem from empty clip path "
                    "(shot %s)" % (_req_get(request, "shot_id") or self.name))
            stem_path = os.path.join(str(dest_dir), video_stem + "_foley.wav")
            matched = conform_stem_to_frame_count(
                arr, rate, int(clip["frame_count"]), int(clip["fps"] or 25))
            n_samples, n_ch = write_pcm16_wav(stem_path, matched, rate)
        except FoleyStemError as stem_exc:
            # THE PROVIDER WAS ALREADY PAID WHEN THIS FIRES. The clip came
            # back, the harvest ran, and the bed turned out unusable -- so
            # this is the provider delivering something this pipeline cannot
            # use, which is exactly CORRUPT_OUTPUT. Stamping it is what lets
            # the render driver's cloud floor SEE it: unstamped, it surfaced
            # as a bare RenderError, the floor looked past it, and the episode
            # died with every paid beat in it. Live 2026-09-16 on this engine
            # (see the shot_b004.wav note above), which is why it is stamped
            # here and not left to a prose guess one layer up.
            stem_exc.code = CloudErrorCode.CORRUPT_OUTPUT
            raise
        finally:
            try:
                if os.path.isfile(harvest_tmp):
                    os.remove(harvest_tmp)
            except OSError:
                pass
        duration_s = n_samples / float(rate) if rate else 0.0
        clip.update({
            "foley_path": stem_path,
            "foley_sha256": sha256_of_file(stem_path),
            "foley_samples": int(n_samples),
            "foley_sample_rate": int(rate),
            "foley_channels": int(n_ch),
            "foley_duration_s": float(duration_s),
        })
        return clip


class CloudLtx25AudioInEngine(_CloudVideoBase):
    """Cloud analogue of local ``ltx_audio_in`` on LTX 2.5 Audio-to-Video.

    There is no Comfy ``api_ltx2_5_a2v`` template. The live node
    ``LtxApi25AudioToVideo`` is the contract: audio 2-20s SETS duration,
    Fast/Pro with resolution 1920x1080 or 1080x1920 only, prompt, seed,
    optional first-frame image. No duration/fps/generate_audio widgets --
    those belong to I2V Foley. Audio DRIVES the picture. Not Foley. Joins
    ``_AUDIO_IN_CHARACTER_ENGINES`` so a character beat owns a mouth.
    """

    name = "cloud_ltx25_audio_in"
    node_key = "cloud_ltx25_a2v"
    family = "audio_conditioned_video"
    required_inputs = ("init_image", "audio_ref", "text_prompt")
    reactivity = "required_audio_ref"
    still_plan = _CLOUD_VIDEO_SHAPE_A_BASE_PLAN
    frame_contract = FrameContract(
        min_frames=50,
        max_frames=500,
        quantum=25,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )

    def _model_label(self) -> str:
        from .._otr_shared.cloud_model_ids import resolve_model_id
        label = resolve_model_id(self.node_key)
        if label not in (_LTX25_FAST_LABEL, _LTX25_PRO_LABEL):
            raise RuntimeError(
                "%s: unsupported LTX 2.5 A2V model %r; expected %r or %r"
                % (self.name, label, _LTX25_FAST_LABEL, _LTX25_PRO_LABEL))
        return label

    def _estimated_usd(self, request) -> float:
        secs = self._duration_seconds(
            request, env="OTR_CLOUD_LTX25_DURATION",
            default=8, min_s=2, max_s=20)
        return ltx25_estimated_usd(secs)

    def _partner_inputs(self, request):
        label = self._model_label()
        prompt = self._text_prompt_input(request)
        audio = self._audio_input(
            request, min_duration_s=2.0, max_duration_s=20.0,
            pad_to_min=True)
        resolution = self._choice(
            "OTR_CLOUD_LTX25_A2V_RESOLUTION", _LTX25_A2V_RES_DEFAULT,
            _LTX25_A2V_RESOLUTIONS)
        _LOG.info("[OTR.cloud.ltx25_a2v] %s", json.dumps({
            "engine": self.name,
            "ltx25_model": label,
            "ltx25_resolution": resolution,
        }, sort_keys=True))
        return {
            "audio": audio,
            "model": {
                "model": label,
                "resolution": resolution,
            },
            "prompt": prompt,
            "seed": self._seed(request),
            "image": self._init_image_input(request),
        }


KlingAvatar = CloudKlingAvatarEngine()
Seedance2 = CloudSeedance2Engine()
WanI2V = CloudWanI2VEngine()
WanI2VAudio = CloudWanI2VAudioEngine()
ViduQ2ProFast720p = CloudViduQ2ProFast720pEngine()
Ltx25FoleyPlus = CloudLtx25FoleyPlusEngine()
Ltx25AudioIn = CloudLtx25AudioInEngine()

for _eng in (
        KlingAvatar, Seedance2, WanI2V, WanI2VAudio,
        ViduQ2ProFast720p, Ltx25FoleyPlus, Ltx25AudioIn):
    register(_eng)

__all__ = [
    "CloudKlingAvatarEngine", "CloudSeedance2Engine",
    "CloudWanI2VEngine", "CloudWanI2VAudioEngine",
    "CloudViduQ2ProFast720pEngine",
    "CloudLtx25FoleyPlusEngine", "CloudLtx25AudioInEngine",
]
