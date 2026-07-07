"""Cloud partner IMAGE (stills) adapters -- S1 stills lane (pass04 + remaining-
sprints PLAN.md Sprint B; operator GO 2026-07-03 "S1").

Four rows from the S0 pin table, invoked through the S0 bridge
(``invoke_partner_node``) and conformed by ``canonicalize_image``:

    cloud_flux_pro       text2img   BEST (prompt continuity with flux_gen1)
    cloud_nano_banana_2  text2img   BEST (V3 model row; reference consistency)
    cloud_seedream_2     text2img   cheapest stylization tier (V3 model row)

(``cloud_ideogram_v4`` ships as ``ideo`` for ordinary scene stills; a future
``ideo_word`` specialist can share the same Partner row.)

Each adapter implements the REDUCED ImageEngine protocol (assert_usable /
prepare / render_image / teardown -- registry.py): ``render_image`` builds the
per-row PINNED kwargs, calls the bridge, conforms the provider still to the exact
role canvas via ``canonicalize_image``, and returns the PNG PATH string (the
dispatcher's ``_coerce_pixels`` reads a ``.png`` path directly).

S1 SCOPE (registry-IS-the-menu C6): rows REGISTER unconditionally with EMPTY
``default_roles`` -- selectable, NEVER automatic. Operator directive 2026-07-02:
the DROPDOWN PICK is the enable (no hidden switch); a pick without credentials
fails LOUD at invoke-time auth. NO local weights, NO VRAM. Money: per-row
estimate against the session budget ceiling (env-overridable); the budget machine
is inert unless a nonzero estimate is passed (cloud_media_invoke.py).

Cold-import-clean (V-12): stdlib + registry + role vocab at module scope; the
bridge / canonicalizer / model-id resolver / PIL import lazily inside the render
lifecycle.
"""
from __future__ import annotations

import logging
import os

from .registry import EngineUnusable, EngineUsabilityReason, register
from .._otr_shared.role_compat import ROLES

_LOG = logging.getLogger("OTR.image.eng_cloud_image")

#: default still canvas when the request carries no dims (env-overridable).
_DEF_W_ENV, _DEF_H_ENV = "OTR_CLOUD_IMAGE_WIDTH", "OTR_CLOUD_IMAGE_HEIGHT"
_DEF_W, _DEF_H = 832, 1216
_U64_MAX = 0xFFFFFFFFFFFFFFFF
_I32_MAX = 2147483647

_NANO_MODELS = ("Nano Banana 2 (Gemini 3.1 Flash Image)", "Nano Banana 2 Lite")
_NANO_ASPECTS = (
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "8:1", "1:8",
)
_NANO_MODALITIES = ("IMAGE", "IMAGE+TEXT")
_NANO_THINKING_LEVELS = ("MINIMAL", "HIGH")
_NANO_RESOLUTIONS = {
    "Nano Banana 2 (Gemini 3.1 Flash Image)": ("1K", "2K", "4K"),
    "Nano Banana 2 Lite": ("1K",),
}

_SEEDREAM_MODELS = (
    "seedream 5.0 lite",
    "seedream-4-5-251128",
    "seedream-4-0-250828",
)
_SEEDREAM_PRESETS = {
    "seedream 5.0 lite": {
        "1:1": "(2K) 2048x2048 (1:1)",
        "16:9": "(2K) 2848x1600 (16:9)",
        "9:16": "(2K) 1600x2848 (9:16)",
    },
    "seedream-4-5-251128": {
        "1:1": "(2K) 2048x2048 (1:1)",
        "16:9": "(2K) 2848x1600 (16:9)",
        "9:16": "(2K) 1600x2848 (9:16)",
    },
    "seedream-4-0-250828": {
        "1:1": "(2K) 2048x2048 (1:1)",
        "16:9": "(2K) 2848x1600 (16:9)",
        "9:16": "(2K) 1600x2848 (9:16)",
    },
}

_IDEOGRAM_RESOLUTIONS = (
    "Auto",
    "2048x2048 (1:1)",
    "1440x2880 (1:2)",
    "2880x1440 (2:1)",
    "1664x2496 (2:3)",
    "2496x1664 (3:2)",
    "1792x2240 (4:5)",
    "2240x1792 (5:4)",
    "1440x2560 (9:16)",
    "2560x1440 (16:9)",
    "1600x2560 (5:8)",
    "2560x1600 (8:5)",
    "1728x2304 (3:4)",
    "2304x1728 (4:3)",
    "1296x3168 (9:22)",
    "3168x1296 (22:9)",
    "1152x2944 (9:23)",
    "2944x1152 (23:9)",
    "1248x3328 (3:8)",
    "3328x1248 (8:3)",
    "1280x3072 (5:12)",
    "3072x1280 (12:5)",
)
_IDEOGRAM_RESOLUTION_BY_PIXELS = {
    r.split(" ")[0]: r for r in _IDEOGRAM_RESOLUTIONS if r != "Auto"
}


def _efloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _eint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _timeout_s() -> float:
    return _efloat("OTR_CLOUD_IMAGE_TIMEOUT_S", 300.0)


#: Ideogram rendering_speed -> per-image USD estimate (the pin carries no
#: structured pricing; adapters must pass a numeric estimate). Values from
#: installed Partner-node pricing constants; v1 default speed = TURBO
#: (cheapest). Env-overridable via OTR_CLOUD_IDEOGRAM_SPEED.
_IDEOGRAM_SPEED_PRICE = {"TURBO": 0.0429, "DEFAULT": 0.0858, "QUALITY": 0.143}
_IDEOGRAM_DEFAULT_SPEED = "TURBO"


def _ideogram_speed() -> str:
    return _choice_env(
        "OTR_CLOUD_IDEOGRAM_SPEED", _IDEOGRAM_DEFAULT_SPEED,
        tuple(_IDEOGRAM_SPEED_PRICE), normalize=str.upper)


def _ideogram_est_usd() -> float:
    env = os.environ.get("OTR_CLOUD_IDEOGRAM_EST_USD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _IDEOGRAM_SPEED_PRICE.get(_ideogram_speed(),
                                     _IDEOGRAM_SPEED_PRICE["DEFAULT"])


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _first_present(request, *keys, default=None):
    for key in keys:
        value = _req_get(request, key, None)
        if value is not None:
            return value
    return default


def _shape_error(detail: str):
    from .._otr_shared.cloud_media_backend import CloudErrorCode, CloudMediaError
    raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG, detail)


def _choice(value: str, allowed: tuple, *, name: str, normalize=None,
            aliases: dict | None = None) -> str:
    raw = "" if value is None else str(value).strip()
    if normalize is not None:
        raw = normalize(raw)
    if aliases and raw in aliases:
        raw = aliases[raw]
    if raw not in allowed:
        _shape_error(
            f"{name}={value!r} is not supported by the installed Partner "
            f"node; allowed: {list(allowed)}")
    return raw


def _choice_env(env_name: str, default: str, allowed: tuple, *,
                normalize=None, aliases: dict | None = None) -> str:
    return _choice(os.environ.get(env_name, "") or default, allowed,
                   name=env_name, normalize=normalize, aliases=aliases)


def _clamp_int(value, *, name: str, lo: int, hi: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        from .._otr_shared.cloud_media_backend import CloudErrorCode, CloudMediaError
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"{name}={value!r} is not an integer") from exc
    return max(int(lo), min(int(hi), ivalue))


def _ceil_step_clamped(value: int, *, lo: int, hi: int, step: int) -> int:
    value = max(int(lo), min(int(hi), int(value)))
    return max(int(lo), min(int(hi), ((value + step - 1) // step) * step))


class _CloudImageBase:
    """Shared S1 stills-adapter mechanics; subclasses pin the row identity."""

    # --- registry-facing core (mirrors flux_gen1 / AudioEngine core) ---
    roles: tuple = ROLES
    default_roles: tuple = ()            # NEVER automatic in S1 (S3 rule)
    commercial_clean = True              # partner API rows; ToS audit in S0 docs
    requires_flag = None                 # dropdown pick IS the enable (C6)
    #: an image engine needs a text prompt (reduced prompt -> image contract).
    required_inputs: tuple = ("text_prompt",)

    # --- row identity (subclasses) ---
    name = ""
    node_key = ""                        # partner_nodes.yaml row key
    est_usd_env = ""
    est_usd_default = 0.05

    def load(self) -> None:              # no local weights
        return None

    def unload(self) -> None:
        return None

    def teardown(self, prepared) -> None:
        return None

    # ---- render lifecycle -------------------------------------------------
    def assert_usable(self, host_caps, profile, request_template=None):
        # NO enable-flag check (operator directive 2026-07-02): the dropdown
        # pick is the enable; credentials resolve fail-closed at invoke time.
        # Only structural checks here: PIL (canonicalize_image transcodes to
        # sRGB PNG) + a healthy pin row.
        try:
            import PIL  # noqa: F401
        except Exception as exc:  # pragma: no cover -- PIL is a comfy hard dep
            raise EngineUnusable(
                self.name, "", EngineUsabilityReason.MALFORMED_CONFIG,
                f"Pillow unavailable ({exc}) -- the cloud stills canonicalizer "
                f"transcodes provider output to sRGB PNG", kind="image")
        from .._otr_shared.cloud_media_invoke import partner_rows
        row = partner_rows().get(self.node_key)
        if not isinstance(row, dict) or str(row.get("status")) != "OK":
            raise EngineUnusable(
                self.name, "", EngineUsabilityReason.MALFORMED_CONFIG,
                f"partner pin row {self.node_key!r} missing or not OK -- "
                f"re-pin via scripts/otr_pin_partner_nodes.py", kind="image")
        return self.name

    def prepare(self, host_caps, profile, session_ctx):
        # The dispatcher calls prepare(None, None, None); auth/budget live in
        # the invoke bridge, not here -- must never crash on the None triple.
        return {}

    # ---- request helpers --------------------------------------------------
    def _prompt(self, request) -> str:
        prompt = str(_req_get(request, "prompt")
                     or _req_get(request, "text_prompt") or "").strip()
        if not prompt:
            _shape_error(
                f"{self.name}: blank prompt; installed Partner image nodes "
                "require a non-empty prompt")
        return prompt

    def _seed(self, request) -> int:
        return int(_first_present(request, "seed", "request_seed", default=0))

    def _seed_u64(self, request) -> int:
        return _clamp_int(
            _first_present(request, "seed", "request_seed", default=0),
            name=f"{self.name}.seed", lo=0, hi=_U64_MAX)

    def _seed_i32(self, request) -> int:
        """Provider-safe seed for APIs capped at signed int32."""
        return _clamp_int(
            _first_present(request, "seed", "request_seed", default=0),
            name=f"{self.name}.seed", lo=0, hi=_I32_MAX)

    def _canvas_wh(self, request):
        w = int(_req_get(request, "width") or _req_get(request, "w")
                or _eint(_DEF_W_ENV, _DEF_W))
        h = int(_req_get(request, "height") or _req_get(request, "h")
                or _eint(_DEF_H_ENV, _DEF_H))
        # TRUE 1080p cloud stills (operator 2026-07-03): conform to a real 1080p
        # canvas (orientation taken from the role's request canvas), NOT the
        # ~832x480 / 1472x832 role canvas. CLOUD-LANE ONLY -- a LOCAL video engine
        # that later consumes this still remaps it into its own render canvas
        # (no VRAM impact). Env OTR_CLOUD_STILL_CANVAS[_PORTRAIT].
        from .._otr_shared.cloud_media_canonical import cloud_delivery_wh
        return cloud_delivery_wh(w, h, land_env="OTR_CLOUD_STILL_CANVAS",
                                 port_env="OTR_CLOUD_STILL_CANVAS_PORTRAIT")

    def _est_usd(self) -> float:
        return _efloat(self.est_usd_env, self.est_usd_default)

    def _partner_inputs(self, request) -> dict:
        raise NotImplementedError

    def render_image(self, request, prepared=None):
        """Mint ONE still via the partner node + conform to the cloud 1080p
        delivery canvas (orientation-preserving, cloud-lane only; see
        _canvas_wh); return the canonical PNG PATH (dispatcher reads a .png
        path). Fails LOUD -- NO fallback (directive)."""
        from .._otr_shared.cloud_media_invoke import invoke_partner_node
        from .._otr_shared.cloud_media_canonical import canonicalize_image
        inputs = self._partner_inputs(request)
        w, h = self._canvas_wh(request)
        _LOG.warning(
            "[OTR image] CLOUD still: %s -> partner %s (est<=$%.3f, "
            "timeout %.0fs, %dx%d)", self.name, self.node_key, self._est_usd(),
            _timeout_s(), w, h)
        raw = invoke_partner_node(
            self.node_key, inputs,
            timeout_s=_timeout_s(), estimated_usd=self._est_usd())
        asset = canonicalize_image(raw, {"w": w, "h": h, "format": "PNG"})
        return str(asset.path)


class CloudFluxProImageEngine(_CloudImageBase):
    """BFL Flux.2 Pro: the BEST stills row (prompt continuity w/ flux_gen1)."""

    name = "cloud_flux_pro"
    node_key = "cloud_flux_pro"
    est_usd_env = "OTR_CLOUD_FLUX_PRO_EST_USD"
    est_usd_default = 0.05

    def _partner_inputs(self, request):
        # pinned required: height INT, prompt STRING, prompt_upsampling BOOL,
        # seed INT, width INT.
        w, h = self._canvas_wh(request)
        # BFL requires /32-aligned request dims in the live schema's 256..2048
        # range. Snap upward when possible (1920x1080 -> 1920x1088), then the
        # canonical PNG cover-crops back to the exact delivery canvas.
        w32 = _ceil_step_clamped(int(w), lo=256, hi=2048, step=32)
        h32 = _ceil_step_clamped(int(h), lo=256, hi=2048, step=32)
        return {
            "prompt": self._prompt(request),
            "width": w32,
            "height": h32,
            "prompt_upsampling": False,
            "seed": self._seed_u64(request),
        }


class CloudNanoBanana2ImageEngine(_CloudImageBase):
    """Gemini Nano-Banana 2: BEST stills, reference consistency (V3 model row)."""

    name = "cloud_nano_banana_2"
    node_key = "cloud_nano_banana_2"
    est_usd_env = "OTR_CLOUD_NANO_BANANA_EST_USD"
    est_usd_default = 0.04

    def _partner_inputs(self, request):
        # pinned required: model DYNAMICCOMBO_V3, prompt STRING,
        # response_modalities COMBO, seed INT. The DYNAMICCOMBO_V3 value is a
        # DICT -- NOT a bare slug (a string raises "string indices must be
        # integers"). The live V2 partner row needs model/resolution/aspect_ratio,
        # but its own execute() always adds a thinkingConfig that Vertex rejects
        # for this image model. The OTR invoke bridge therefore uses a scoped
        # GeminiNanoBanana2V2 override that preserves the pinned auth/session path
        # while omitting thinkingConfig. resolution options ["1K","2K","4K"]
        # (pro price 1K=$0.134/2K/4K=$0.24); aspect_ratio "auto".
        # response_modalities is compared `== "IMAGE"` (:1024) -> MUST be
        # uppercase "IMAGE" for image-only; anything else silently requests
        # IMAGE+TEXT. All env-overridable.
        from .._otr_shared.cloud_model_ids import resolve_model_id
        model_name = _choice(
            resolve_model_id(self.node_key), _NANO_MODELS,
            name="OTR_CLOUD_NANO_BANANA_MODEL")
        resolution = _choice_env(
            "OTR_CLOUD_NANO_RESOLUTION", "1K",
            _NANO_RESOLUTIONS[model_name])
        return {
            "model": {
                "model": model_name,
                "resolution": resolution,
                "aspect_ratio": _choice_env(
                    "OTR_CLOUD_NANO_ASPECT", "auto", _NANO_ASPECTS),
                "thinking_level": _choice_env(
                    "OTR_CLOUD_NANO_THINKING_LEVEL", "MINIMAL",
                    _NANO_THINKING_LEVELS, normalize=str.upper),
            },
            "prompt": self._prompt(request),
            "response_modalities": _choice_env(
                "OTR_CLOUD_NANO_MODALITIES", "IMAGE", _NANO_MODALITIES,
                normalize=str.upper),
            "seed": self._seed_u64(request),
        }


class CloudSeedream2ImageEngine(_CloudImageBase):
    """ByteDance Seedream 2: cheapest stylization tier (V3 model row)."""

    name = "cloud_seedream_2"
    node_key = "cloud_seedream_2"
    est_usd_env = "OTR_CLOUD_SEEDREAM_EST_USD"
    est_usd_default = 0.02

    def _partner_inputs(self, request):
        # pinned required: model DYNAMICCOMBO_V3, prompt STRING, seed INT,
        # watermark BOOLEAN. The DYNAMICCOMBO_V3 value is a DICT -- NOT a bare
        # slug: ByteDanceSeedreamNodeV2.execute (nodes_bytedance.py:790+) reads
        # model["model"] (KeyError/"string indices must be integers" on a bare
        # string), while size_preset/width/height use model.get(...) with safe
        # defaults (2048/2048). OTR includes an installed size preset so the
        # provider request matches the delivery orientation without custom dims.
        # The ByteDance schema caps seed at signed int32; OTR clamps before
        # invoke so the partner node never sees an out-of-range value.
        from .._otr_shared.cloud_model_ids import resolve_model_id
        model_name = _choice(
            resolve_model_id(self.node_key), _SEEDREAM_MODELS,
            name="OTR_CLOUD_SEEDREAM_MODEL")
        return {
            "model": {
                "model": model_name,
                "size_preset": self._size_preset(model_name, request),
                "max_images": 1,
            },
            "prompt": self._prompt(request),
            "seed": self._seed_i32(request),
            "watermark": False,
        }

    def _size_preset(self, model_name: str, request) -> str:
        presets = _SEEDREAM_PRESETS[model_name]
        allowed = tuple(presets.values())
        env = os.environ.get("OTR_CLOUD_SEEDREAM_SIZE_PRESET", "").strip()
        if env:
            return _choice(env, allowed, name="OTR_CLOUD_SEEDREAM_SIZE_PRESET")
        w, h = self._canvas_wh(request)
        if h > w:
            ratio = "9:16"
        elif w > h:
            ratio = "16:9"
        else:
            ratio = "1:1"
        return presets[ratio]


class CloudIdeoImageEngine(_CloudImageBase):
    """Ideogram v4: PLAIN scene-still option for ANY slot (S1+1 `ideo`).

    Ordinary scene stills through the existing scene-prompt path (the request
    prompt already carries compose_still_prompt + NO_TEXT_CLAUSE) -- exactly
    like the other S1 rows but Ideogram-flavored. The words-specialist
    ``ideo_word`` (lyric_text / title_mood modes) is a SEPARATE engine; this one
    adds no new prompt path. estimated_usd follows the rendering_speed price
    map. (docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md, `ideo`.)"""

    name = "ideo"
    node_key = "cloud_ideogram_v4"

    def _est_usd(self) -> float:
        return _ideogram_est_usd()

    def _partner_inputs(self, request):
        # pinned required: prompt STRING, rendering_speed COMBO,
        # resolution COMBO, seed INT.
        return {
            "prompt": self._prompt(request),
            "rendering_speed": _ideogram_speed(),
            "resolution": self._resolution(request),
            "seed": self._seed_i32(request),
        }

    def _resolution(self, request) -> str:
        env = os.environ.get("OTR_CLOUD_IDEOGRAM_RESOLUTION", "").strip()
        aliases = dict(_IDEOGRAM_RESOLUTION_BY_PIXELS)
        if env:
            return _choice(env, _IDEOGRAM_RESOLUTIONS,
                           name="OTR_CLOUD_IDEOGRAM_RESOLUTION",
                           aliases=aliases)
        w, h = self._canvas_wh(request)
        if h > w:
            return "1440x2560 (9:16)"
        if w > h:
            return "2560x1440 (16:9)"
        return "2048x2048 (1:1)"


FluxPro = CloudFluxProImageEngine()
NanoBanana2 = CloudNanoBanana2ImageEngine()
Seedream2 = CloudSeedream2ImageEngine()
Ideo = CloudIdeoImageEngine()

for _eng in (FluxPro, NanoBanana2, Seedream2, Ideo):
    register(_eng)

__all__ = [
    "CloudFluxProImageEngine",
    "CloudNanoBanana2ImageEngine", "CloudSeedream2ImageEngine",
    "CloudIdeoImageEngine",
]
