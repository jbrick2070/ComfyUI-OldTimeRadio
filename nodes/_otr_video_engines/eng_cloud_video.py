"""Cloud partner VIDEO adapters -- S3 core (pass04 secs 5+7, operator GO
2026-07-02 evening: "code the cloud video plan").

Four rows from the S0 pin table, invoked through the S0 bridge
(``invoke_partner_node``) and conformed by ``canonicalize_video``:

    cloud_kling_avatar   required_audio_ref   (init_image, audio_ref)
    cloud_kling_lipsync  lipsync_overlay      (base_clip_ref, audio_ref)
    cloud_seedance_2     required_audio_ref   (init_image, audio_ref)
    cloud_wan_i2v        mute_only            (init_image, text_prompt)

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

import logging
import os

from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.eng_cloud_video")

#: kling avatar mode COMBO -- the pin excludes combo options (S0), so the
#: adapter ships the provider's documented std tier; env-overridable.
_KLING_MODE_ENV = "OTR_CLOUD_KLING_MODE"
_KLING_MODE_DEFAULT = "std"


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
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg/ffprobe not on PATH -- the cloud video canonicalizer "
                "strips provider audio via ffmpeg (must_strip_audio)",
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
        from .._otr_shared.cloud_media_canonical import canonicalize_video
        canvas = _req_get(request, "canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        asset = canonicalize_video(raw, {
            "w": int(c_get("w", 0) or 0),
            "h": int(c_get("h", 0) or 0),
            "fps": int(c_get("fps", 25) or 25),
        })
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
        }

    def teardown(self, prepared) -> None:
        return None

    # ---- shared input builders --------------------------------------------
    def _seed(self, request) -> int:
        seeds = _req_get(request, "seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return int(s_get("request_seed", 0) or 0)

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

    def _audio_input(self, request):
        path = _ref_path(_req_get(request, "audio_ref"))
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                f"{self.name}: audio_ref missing/absent on disk ({path!r}) "
                f"-- reactivity={self.reactivity}, NO FALLBACK")
        return _load_audio_dict(path)


class CloudKlingAvatarEngine(_CloudVideoBase):
    """Kling avatar: TALKING default row (audio CONDITIONS the clip)."""

    name = "cloud_kling_avatar"
    node_key = "cloud_kling_avatar"
    family = "audio_driven_face"
    required_inputs = ("init_image", "audio_ref")
    reactivity = "required_audio_ref"

    def _partner_inputs(self, request):
        return {
            "image": self._init_image_input(request),
            "sound_file": self._audio_input(request),
            "mode": os.environ.get(_KLING_MODE_ENV, _KLING_MODE_DEFAULT),
            "seed": self._seed(request),
            "prompt": str(_req_get(request, "text_prompt") or ""),
        }


class CloudKlingLipsyncEngine(_CloudVideoBase):
    """Kling lip-sync overlay: drives a BASE clip's mouth from audio."""

    name = "cloud_kling_lipsync"
    node_key = "cloud_kling_lipsync"
    family = "lipsync_overlay"
    required_inputs = ("base_clip_ref", "audio_ref")
    reactivity = "lipsync_overlay"

    def _partner_inputs(self, request):
        base = _ref_path(_req_get(request, "base_clip_ref"))
        if not base or not os.path.isfile(base):
            raise RuntimeError(
                f"{self.name}: base_clip_ref missing/absent ({base!r}) -- "
                f"NO FALLBACK (provide a base clip or OTR_LSYNC_BASE_ENGINE)")
        try:
            from comfy_api.input_impl import VideoFromFile  # type: ignore
        except Exception as exc:  # pragma: no cover -- core drift
            raise RuntimeError(
                f"{self.name}: comfy_api VideoFromFile unavailable ({exc}) "
                f"-- the pinned KlingLipSync row needs a VIDEO input")
        return {
            "video": VideoFromFile(base),
            "audio": self._audio_input(request),
            "voice_language": os.environ.get(
                "OTR_CLOUD_KLING_LIPSYNC_LANG", "en"),
        }


class CloudSeedance2Engine(_CloudVideoBase):
    """ByteDance Seedance 2 reference row: music/b-roll reactive default."""

    name = "cloud_seedance_2"
    node_key = "cloud_seedance_2"
    family = "audio_conditioned_video"
    required_inputs = ("init_image", "audio_ref")
    reactivity = "required_audio_ref"

    def _partner_inputs(self, request):
        # PINNED static inputs are ONLY (model COMBO, seed, watermark) -- the
        # reference image + audio + prompt hide inside the DYNAMICCOMBO_V3
        # model schema (docs-window gotcha 2026-07-02: the static pin is
        # SHALLOW for V3 rows). Sending guessed kwargs would TypeError at the
        # partner node, so this row stays an HONEST dark row until the
        # V3-expansion pin (ships with S1) names the real dynamic inputs.
        raise RuntimeError(
            f"{self.name}: the pinned row's media inputs ride the "
            f"DYNAMICCOMBO_V3 model schema (static pin is shallow) -- "
            f"V3-expansion pin lands with cloud S1; row not yet invocable")


class CloudWanI2VEngine(_CloudVideoBase):
    """Wan image-to-video: the MUTE opt-down row (explicit picks only)."""

    name = "cloud_wan_i2v"
    node_key = "cloud_wan_i2v"
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"

    def _partner_inputs(self, request):
        # Pinned STATIC inputs: first_frame IMAGE, model DYNAMICCOMBO_V3,
        # prompt_extend BOOL, seed INT, watermark BOOL (+ optional audio,
        # NOT sent -- this is the mute opt-down row). There is NO top-level
        # prompt input (docs-window catch 2026-07-02): the text prompt rides
        # the dynamic model schema and lands with the S1 V3-expansion pin.
        model = os.environ.get("OTR_CLOUD_WAN_MODEL", "")
        if not model:
            raise RuntimeError(
                f"{self.name}: OTR_CLOUD_WAN_MODEL unset -- the pinned row's "
                f"model COMBO is dynamic (pin excludes options); set the env "
                f"to the provider's model id")
        return {
            "first_frame": self._init_image_input(request),
            "model": model,
            "prompt_extend": False,
            "seed": self._seed(request),
            "watermark": False,
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
    "drifting smoke, soft neon flicker, subtle parallax depth -- the words stay "
    "crisp, sharp and fully legible in every frame, letterforms never warp, "
    "melt or distort")
_RAZZLE_NEG_ENV = "OTR_CLOUD_RAZZLE_NEG"
_RAZZLE_NEG_DEFAULT = ("warped text, melting letters, distorted typography, "
                       "illegible words, garbled text, flickering letters")


class CloudWordRazzleEngine(_CloudVideoBase):
    """word_razzle: animate a word-card still into a living period poster."""

    name = "word_razzle"
    node_key = "cloud_pixverse_i2v"
    family = "image_to_video"
    required_inputs = ("init_image", "text_prompt")
    reactivity = "mute_only"

    def _razzle_prompt(self, request) -> str:
        """The world-motion + text-preservation prompt. The base motion clause
        (env-overridable) LEADS; the beat's own text_prompt (scene/subject) is
        appended so the animation matches the beat, never overriding the
        readability directive."""
        motion = os.environ.get(_RAZZLE_MOTION_ENV, "").strip() or _RAZZLE_MOTION_DEFAULT
        beat = str(_req_get(request, "text_prompt") or "").strip()
        return f"{motion}. {beat}".strip().rstrip(".") if beat else motion

    def _duration_seconds(self, request) -> int:
        """The provider duration (seconds). Derived from the beat's frame
        target (timing.target_frame_count / fps) and clamped to Pixverse's
        supported 5s / 8s tiers; env OTR_CLOUD_PIXVERSE_DURATION overrides."""
        env = os.environ.get("OTR_CLOUD_PIXVERSE_DURATION", "").strip()
        if env:
            try:
                return int(env)
            except ValueError:
                pass
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
            "negative_prompt": os.environ.get(_RAZZLE_NEG_ENV, "").strip()
            or _RAZZLE_NEG_DEFAULT,
            "motion_mode": os.environ.get("OTR_CLOUD_PIXVERSE_MOTION", "normal"),
            "quality": os.environ.get("OTR_CLOUD_PIXVERSE_QUALITY", "540p"),
            "duration_seconds": self._duration_seconds(request),
            "seed": self._seed(request),
        }


KlingAvatar = CloudKlingAvatarEngine()
KlingLipsync = CloudKlingLipsyncEngine()
Seedance2 = CloudSeedance2Engine()
WanI2V = CloudWanI2VEngine()
WordRazzle = CloudWordRazzleEngine()

for _eng in (KlingAvatar, KlingLipsync, Seedance2, WanI2V, WordRazzle):
    register(_eng)

__all__ = [
    "CloudKlingAvatarEngine", "CloudKlingLipsyncEngine",
    "CloudSeedance2Engine", "CloudWanI2VEngine", "CloudWordRazzleEngine",
]
