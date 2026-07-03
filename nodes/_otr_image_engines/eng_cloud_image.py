"""Cloud partner IMAGE (stills) adapters -- S1 stills lane (pass04 + remaining-
sprints PLAN.md Sprint B; operator GO 2026-07-03 "S1").

Four rows from the S0 pin table, invoked through the S0 bridge
(``invoke_partner_node``) and conformed by ``canonicalize_image``:

    cloud_recraft        text2img   CHEAP
    cloud_flux_pro       text2img   BEST (prompt continuity with flux_gen1)
    cloud_nano_banana_2  text2img   BEST (V3 model row; reference consistency)
    cloud_seedream_2     text2img   cheapest stylization tier (V3 model row)

(``cloud_ideogram_v4`` -- the text-render row -- ships as ``ideo`` / ``ideo_word``
with its own dedicated doc, docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md
= S1+1; until then it carries an explicit xfail in the conformance test.)

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
#: PRICING.md / partner_nodes.yaml notes (TURBO 9.05cr -> QUALITY 27.16cr,
#: 211cr = $1). v1 default speed = TURBO (cheapest). Env-overridable via
#: OTR_CLOUD_IDEOGRAM_SPEED; unknown speeds fall back to the DEFAULT price.
_IDEOGRAM_SPEED_PRICE = {"TURBO": 0.043, "DEFAULT": 0.086, "QUALITY": 0.13}
_IDEOGRAM_DEFAULT_SPEED = "TURBO"


def _ideogram_speed() -> str:
    return (os.environ.get("OTR_CLOUD_IDEOGRAM_SPEED", "")
            or _IDEOGRAM_DEFAULT_SPEED).upper()


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
        return str(_req_get(request, "prompt")
                   or _req_get(request, "text_prompt") or "")

    def _seed(self, request) -> int:
        return int(_req_get(request, "seed")
                   or _req_get(request, "request_seed") or 0)

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


class CloudRecraftImageEngine(_CloudImageBase):
    """Recraft text-to-image: the CHEAP stills row."""

    name = "cloud_recraft"
    node_key = "cloud_recraft"
    est_usd_env = "OTR_CLOUD_RECRAFT_EST_USD"
    est_usd_default = 0.04

    def _partner_inputs(self, request):
        # pinned required: n INT, prompt STRING, seed INT, size COMBO.
        return {
            "prompt": self._prompt(request),
            "n": 1,
            "seed": self._seed(request),
            "size": os.environ.get("OTR_CLOUD_RECRAFT_SIZE", "1024x1024"),
        }


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
        return {
            "prompt": self._prompt(request),
            "width": w,
            "height": h,
            "prompt_upsampling": False,
            "seed": self._seed(request),
        }


class CloudNanoBanana2ImageEngine(_CloudImageBase):
    """Gemini Nano-Banana 2: BEST stills, reference consistency (V3 model row)."""

    name = "cloud_nano_banana_2"
    node_key = "cloud_nano_banana_2"
    est_usd_env = "OTR_CLOUD_NANO_BANANA_EST_USD"
    est_usd_default = 0.04

    def _partner_inputs(self, request):
        # pinned required: model DYNAMICCOMBO_V3, prompt STRING,
        # response_modalities COMBO, seed INT. model resolves via the single
        # source of truth (never the placeholder).
        from .._otr_shared.cloud_model_ids import resolve_model_id
        return {
            "model": resolve_model_id(self.node_key),
            "prompt": self._prompt(request),
            "response_modalities": os.environ.get(
                "OTR_CLOUD_NANO_MODALITIES", "Image"),
            "seed": self._seed(request),
        }


class CloudSeedream2ImageEngine(_CloudImageBase):
    """ByteDance Seedream 2: cheapest stylization tier (V3 model row)."""

    name = "cloud_seedream_2"
    node_key = "cloud_seedream_2"
    est_usd_env = "OTR_CLOUD_SEEDREAM_EST_USD"
    est_usd_default = 0.02

    def _partner_inputs(self, request):
        # pinned required: model DYNAMICCOMBO_V3, prompt STRING, seed INT,
        # watermark BOOLEAN.
        from .._otr_shared.cloud_model_ids import resolve_model_id
        return {
            "model": resolve_model_id(self.node_key),
            "prompt": self._prompt(request),
            "seed": self._seed(request),
            "watermark": False,
        }


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
            "resolution": os.environ.get(
                "OTR_CLOUD_IDEOGRAM_RESOLUTION", "1024x1024"),
            "seed": self._seed(request),
        }


Recraft = CloudRecraftImageEngine()
FluxPro = CloudFluxProImageEngine()
NanoBanana2 = CloudNanoBanana2ImageEngine()
Seedream2 = CloudSeedream2ImageEngine()
Ideo = CloudIdeoImageEngine()

for _eng in (Recraft, FluxPro, NanoBanana2, Seedream2, Ideo):
    register(_eng)

__all__ = [
    "CloudRecraftImageEngine", "CloudFluxProImageEngine",
    "CloudNanoBanana2ImageEngine", "CloudSeedream2ImageEngine",
    "CloudIdeoImageEngine",
]
