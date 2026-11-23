"""Qwen-Image 2.1 image adapter -- an OPT-IN local still engine (in-process).

Alibaba Qwen's 7B single-checkpoint model (released 2026-09-20). It runs on stock
ComfyUI CORE (>= v0.37.0, commit 6bfaacc6 "feat: Qwen-image 2.1 support", which
added the ``Text Encode Qwen Image 2.1`` and ``Qwen Image 2.1 Cache`` nodes), so it
is an ordinary ``wrapper_bridge`` split-file graph exactly like ``moss_image``:
UNETLoader -> KSampler -> VAEDecode, with the Qwen3-VL-8B text encoder and the
Qwen-Image 2.1 VAE.

LICENCE: the Qwen RESEARCH LICENSE AGREEMENT (2026-09-20) grants the Materials
"FOR NON-COMMERCIAL PURPOSES ONLY" ("research or evaluation purposes only"); any
commercial use needs a separate licence from Qwen. So ``commercial_clean = False``
-- the same shelf as ``flux3_klein`` and ``flux_gen1``. The pack ships no weights:
they are fetched on first use from Comfy-Org's public, ungated repackage at an exact
pinned revision. See ``_PINNED_SOURCES`` in ``nodes/_otr_visual_assets.py`` and
apple/QWEN_IMAGE_21_LICENSE_ATTESTATION.md.

THE RECIPE is Comfy's own template (``image_qwen_image_2_1_t2i.json``): the int8
ConvRot checkpoint, 25 steps, cfg 1.0 (the negative is inert at cfg 1), euler /
simple, denoise 1, ``EmptyLatentImage``, and the cache node left at its defaults.
The template's prompt rewriter (a ``TextGenerate`` pass on a second 9 GB encoder) is
NOT used: the pack composes its own full scene caption.

TEXT-TO-IMAGE ONLY (``accepts_reference_image = False``). The model can also edit
from up to 10 reference pictures, but its edit canvas follows its FIRST image, which
does not match how the dispatcher hands a character portrait to a scene still. That
is a design question for a separate engine row, not a switch on this one.

TRANSPARENCY: the VAE decodes FOUR channels. An ordinary prompt gives an alpha that
is opaque to within ~1.6% (measured 0.984-1.0 on plain stills), and the engine
contract is a plain RGB array, so ``_flatten_alpha`` composites over white (the
colour the model's own vision tower assumes). Native transparent stills are a
measured capability of the model (planning/qwen_image_21/probe_2026-11-22.md) but the
dispatcher does not carry alpha, so this engine does not request it.

Cold-import clean: only the dep-free registry, role vocabulary and stdlib at
module scope. torch / comfy / numpy are lazy inside the functions that need them.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.image.qwen_image_21")

MODEL_ENV = "OTR_QWEN21_UNET"
CLIP_ENV = "OTR_QWEN21_CLIP"
VAE_ENV = "OTR_QWEN21_VAE"

#: The int8 ConvRot build (7.26 GB), Comfy's own default pick in its template. The
#: bf16 build is picked up if installed (see ``_installed_unets``) but never fetched.
_DEFAULT_UNET = "qwen_image_2.1_int8_convrot.safetensors"
_DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
_DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"
#: Core's ``CLIPLoader`` picks the 2.1 tokenizer by DETECTING the Qwen3-VL-8B file
#: under this existing type (comfy/sd.py), so there is no ``qwen_image_21`` type.
_CLIP_TYPE = "qwen_image"

#: Stock ComfyUI node classes the graph needs. ``TextEncodeQwenImage21`` is the
#: version test: it first shipped in v0.37.0 (the older ``qwen_image`` CLIP type
#: alone proves nothing about 2.1).
_REQUIRED_CORE_NODES = ("TextEncodeQwenImage21", "QwenImage21Cache",
                        "EmptyLatentImage", "CLIPLoader")
_CORE_UPDATE_MESSAGE = (
    "Update ComfyUI to v0.37.0 or newer: qwen_image_21 needs the built-in "
    "'Text Encode Qwen Image 2.1' node, which this version does not have")

#: The Qwen3-VL text encoder's vision tower is not used in text-to-image, and the
#: negative is inert at cfg 1.0 -- so there is nothing to carry. The operator's
#: standing directive (2026-08-03) forbids content guardrails on the generation path.
_HYGIENE_NEGATIVE = ""

_CANVAS_MULTIPLE = 32           # the template: "Prefer multiples of 32"
_CANVAS_FLOOR = 256

#: Alpha above this is the model's opaque noise floor (0.984-0.988 measured on plain
#: stills), not transparency; below it the pixel is composited over white.
_OPAQUE_ALPHA = 0.97


def _snap(value: int) -> int:
    """Nearest multiple of 32, floored at 256 (the template's canvas rule)."""
    v = max(_CANVAS_FLOOR, int(value))
    return max(_CANVAS_FLOOR, int(round(v / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)


def _flatten_alpha(frames):
    """A uint8 ``(B,H,W,C)`` batch -> uint8 ``(B,H,W,3)`` RGB.

    C == 3 comes back untouched. C == 4 (the Qwen VAE decodes alpha) is composited
    over white wherever alpha is below ``_OPAQUE_ALPHA``; everything at or above it
    is treated as opaque so a plain still keeps its exact RGB bytes."""
    import numpy as np
    arr = np.asarray(frames)
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] != 4:
        raise ValueError("expected 3 or 4 channels, got %d" % arr.shape[-1])
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    alpha = np.where(alpha >= _OPAQUE_ALPHA, 1.0, alpha)
    flat = rgb * alpha + 255.0 * (1.0 - alpha)
    return np.ascontiguousarray(np.clip(flat + 0.5, 0.0, 255.0).astype(np.uint8))


def _installed_unets():
    """Installed ``qwen_image_2.1*`` diffusion models, ranked int8 > bf16 > other.
    The TURBO builds (``qwen_image_2.1_turbo*``) are excluded: they are a few-step
    recipe and this one is the 25-step recipe. ``[]`` when none (or ``folder_paths``
    absent)."""
    try:
        import folder_paths  # ComfyUI runtime; stubbed in tests
        names = folder_paths.get_filename_list("diffusion_models") or []
    except Exception:  # noqa: BLE001
        return []
    cands = [os.path.basename(n) for n in names
             if os.path.basename(n).lower().startswith("qwen_image_2.1")
             and "turbo" not in os.path.basename(n).lower()
             and str(n).lower().endswith(".safetensors")]

    def _rank(n):
        low = n.lower()
        return (0 if "int8" in low else 1 if "bf16" in low else 2, low)

    return sorted(set(cands), key=_rank)


def _resolve_unet_name():
    """``(basename, verified)`` -- ONE truth shared by the params and
    ``assert_usable``. Env override (verified iff it resolves) > best installed
    > the fetch default (unverified)."""
    env = otr_env.get(MODEL_ENV, "").strip()
    installed = _installed_unets()
    if env:
        base = os.path.basename(env)
        verified = (base in installed) or os.path.isfile(env)
        if not verified:
            try:
                import folder_paths
                verified = bool(folder_paths.get_full_path("diffusion_models", base))
            except Exception:  # noqa: BLE001
                pass
        return base, bool(verified)
    if installed:
        return installed[0], True
    return _DEFAULT_UNET, False


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


def _core_problem(mapping=None):
    """The reason this ComfyUI core cannot run Qwen-Image 2.1, or ``None``.

    ``mapping`` is ComfyUI's NODE_CLASS_MAPPINGS (the queue-time gate passes the
    one it already holds); ``None`` reads it from the running server. A probe
    that cannot run answers ``None`` -- probes never fail a workflow; the
    loader still raises at render if the core really is too old."""
    if mapping is None:
        try:
            import nodes as _comfy_nodes
            mapping = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", None)
        except Exception:  # noqa: BLE001
            return None
    if not mapping:
        return None
    if not all(n in mapping for n in _REQUIRED_CORE_NODES):
        return _CORE_UPDATE_MESSAGE
    return None


@register
class QwenImage21Engine:
    """The Qwen-Image 2.1 adapter (reduced ``prompt -> image`` protocol)."""

    name = "qwen_image_21"
    roles = ROLES
    default_roles = ()               # opt-in; never automatic
    commercial_clean = False         # Qwen Research License: non-commercial only
    requires_flag = None
    required_inputs = ("text_prompt",)
    engine_version = "1"
    accepts_reference_image = False  # text-to-image only

    _TERMINAL = "decode"

    def _params(self, request):
        """Pure: sampler params from the request + env. Seed, prompt and canvas
        come from the request (deterministic re-gen); the rest is Comfy's
        template recipe."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))

        def _eint(name, default):
            try:
                return int(otr_env.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        def _efloat(name, default):
            try:
                return float(otr_env.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        unet, _ok = _resolve_unet_name()
        width = _snap(get("width") or get("w") or _eint("OTR_QWEN21_WIDTH", 1024))
        height = _snap(get("height") or get("h") or _eint("OTR_QWEN21_HEIGHT", 1024))
        return {
            "unet_name": unet,
            "clip_name": os.path.basename(otr_env.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(otr_env.get(VAE_ENV, "") or _DEFAULT_VAE),
            "clip_type": _CLIP_TYPE,
            "unet_dtype": otr_env.get("OTR_QWEN21_UNET_DTYPE", "default"),
            "prompt": str(get("prompt") or ""),
            "negative": str(get("negative_prompt") or _HYGIENE_NEGATIVE),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_QWEN21_STEPS", 25),
            "cfg": _efloat("OTR_QWEN21_CFG", 1.0),
            "sampler_name": otr_env.get("OTR_QWEN21_SAMPLER", "euler"),
            "scheduler": otr_env.get("OTR_QWEN21_SCHEDULER", "simple"),
            "width": width,
            "height": height,
        }

    def _node_candidates(self, params=None):  # noqa: ARG002
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "cache": ("QwenImage21Cache",),
            "encode": ("TextEncodeQwenImage21",),
            "latent": ("EmptyLatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_graph(self, params, wire):
        """Pure: Comfy's Qwen-Image 2.1 t2i graph in ``wrapper_bridge.run_graph``
        form (no prompt rewriter, no reference images)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"],
                                "weight_dtype": params["unet_dtype"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": params["clip_type"]}},
            "vae": {"class": "vae", "inputs": {"vae_name": params["vae_name"]}},
            # Left at the template's own defaults ("auto" device, lossless dtype).
            "cache": {"class": "cache",
                      "inputs": {"model": W("unet", 0), "device": "auto",
                                 "dtype": "default"}},
            # One node encodes both prompts; ``resolution`` only sizes reference
            # pictures, and there are none here. Output slots: 0 positive,
            # 1 negative, 2 an empty latent that this graph does not use.
            "encode": {"class": "encode",
                       "inputs": {"clip": W("clip", 0), "prompt": params["prompt"],
                                  "negative_prompt": params["negative"],
                                  "resolution": 1024}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(params["width"]),
                                  "height": int(params["height"]),
                                  "batch_size": 1}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(params["seed"]),
                                    "steps": int(params["steps"]),
                                    "cfg": float(params["cfg"]),
                                    "sampler_name": params["sampler_name"],
                                    "scheduler": params["scheduler"],
                                    "denoise": 1.0,
                                    "model": W("cache", 0),
                                    "positive": W("encode", 0),
                                    "negative": W("encode", 1),
                                    "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("ksampler", 0),
                                  "vae": W("vae", 0)}},
        }

    def load(self):  # pragma: no cover -- resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def core_problem(self, mapping=None):
        """Read by the queue-time gate (``_refuse_missing_node_packs``) so an
        old core is refused BEFORE the 17 GB fetch, not at the dispatcher."""
        return _core_problem(mapping)

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        """Fail closed until the diffusion model is INSTALLED and the running
        ComfyUI core can load Qwen-Image 2.1 (>= v0.37.0)."""
        problem = _core_problem()
        if problem:
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL, problem, kind="image",
            )
        _name, verified = _resolve_unet_name()
        if not verified:
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"qwen_image_21 diffusion model not found (resolved {_name!r}, not "
                f"installed): it is fetched on first use, or point {MODEL_ENV} at "
                f"a qwen_image_2.1*.safetensors (+ {CLIP_ENV} Qwen3-VL-8B "
                f"encoder / {VAE_ENV} qwen_image_2.1_vae)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU
        """Mint ONE still and return a decoded uint8 (H,W,3) RGB array."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates(params))
        self._classes = classes
        graph = self._build_graph(params, _wb.Wire)
        try:
            # Drop the 9 GB Qwen3-VL encoder the moment its consumers have run so
            # the sampler starts with it off the card.
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL,
                                   free_after_use=True, keep={"unet"},
                                   evict_after_use={"clip"})[0]
            frames = _flatten_alpha(_wb.images_to_uint8(images))
        finally:
            _wb.reclaim_idle_models(reason="qwen_image_21 post-decode")
        log.info("[OTR.image.qwen_image_21] minted still %dx%d seed=%d steps=%d "
                 "cfg=%.2f", params["width"], params["height"],
                 params["seed"], params["steps"], params["cfg"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["QwenImage21Engine", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
