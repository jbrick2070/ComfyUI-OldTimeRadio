"""MOSS-Image image adapter -- an OPT-IN local still engine (in-process).

OpenMOSS-Team's ~5B single-stream DiT, the instruction-tuned SFT checkpoint. It
runs on stock ComfyUI CORE (>= v0.42.0: ``comfy/ldm/moss_image`` and the
``CLIPLoader`` type ``moss_image``, added by commit e087d8f6 "feat: Add
MOSS-image support (#17095)", first tagged v0.42.0), so it is an ordinary
``wrapper_bridge`` split-file graph exactly like ``flux3_klein``: UNETLoader ->
KSampler -> VAEDecode, with the Qwen3.5-9B text encoder and the small Flux 2 VAE.

LICENCE: Apache-2.0 for the model, and Comfy-Org's repackage
(``Comfy-Org/MOSS-Image``) is public and ungated. So ``commercial_clean = True``
and there is no attestation file -- nothing here restricts the outputs. The pack
ships no weights: they are fetched on first use at an exact pinned revision. See
``_PINNED_SOURCES`` in ``nodes/_otr_visual_assets.py``.

THE RECIPE is Comfy's own template (``image_moss_image_t2i.json``): 25 steps,
cfg 2.5, er_sde / simple, denoise 1, ``EmptyFlux2LatentImage``. There is NO
ModelSampling node -- shift 5 is the model's own default, and the template does
not override it. Skip-layer guidance is OFF, which is the template's default
switch position. The template's prompt upsampler (a ``TextGenerate`` pass on the
same 9B encoder) is NOT used: the pack composes its own full scene caption, and
that choice is unmeasured -- override nothing, A/B it.

REFERENCE IMAGES ARE OFF (``accepts_reference_image = False``): the graph is
text-to-image only. The layout-prompt mode (``MOSSImageLayoutPrompt``) is not
wired either.

Cold-import clean: only the dep-free registry, role vocabulary and stdlib at
module scope. torch / comfy are lazy inside ``render_image`` (via wrapper_bridge).
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

log = logging.getLogger("OTR.image.moss_image")

MODEL_ENV = "OTR_MOSS_UNET"
CLIP_ENV = "OTR_MOSS_CLIP"
VAE_ENV = "OTR_MOSS_VAE"

#: The int8 ConvRot SFT build (5.4 GB), Comfy's own default pick. The bf16 SFT
#: build is picked up if installed (see ``_installed_unets``) but never fetched.
_DEFAULT_UNET = "moss_image_sft_int8_convrot.safetensors"
_DEFAULT_CLIP = "qwen3.5_9b_w4a8_gptq.safetensors"
_DEFAULT_VAE = "flux2-vae_small.safetensors"
_CLIP_TYPE = "moss_image"

#: Stock ComfyUI node classes the graph needs. EmptyFlux2LatentImage predates
#: v0.42.0, so it is necessary but NOT the version test: that is the
#: ``moss_image`` entry in CLIPLoader's type list, first tagged v0.42.0.
_REQUIRED_CORE_NODES = ("EmptyFlux2LatentImage", "CLIPLoader")
_CORE_UPDATE_MESSAGE = (
    "Update ComfyUI to v0.42.0 or newer: moss_image needs the built-in "
    "CLIPLoader type 'moss_image', which this version does not have")

#: Comfy's recommended MOSS-Image negative WITHOUT its leading "nsfw" term. The
#: operator's standing directive (2026-08-03) forbids content guardrails on the
#: generation path, so only the quality terms are carried.
_HYGIENE_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"

_CANVAS_MULTIPLE = 16
_CANVAS_FLOOR = 256


def _snap(value: int) -> int:
    """Nearest multiple of 16, floored at 256 (the model needs multiples of 16)."""
    v = max(_CANVAS_FLOOR, int(value))
    return max(_CANVAS_FLOOR, int(round(v / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)


def _installed_unets():
    """Installed ``moss_image*`` diffusion models, ranked: ``moss_image_sft*``
    first (int8 > bf16 > other), then any other ``moss_image*`` build. The BASE
    checkpoints (``moss_image_base*``, meant for training) are excluded: this
    recipe is the SFT one. ``[]`` when none (or ``folder_paths`` absent)."""
    try:
        import folder_paths  # ComfyUI runtime; stubbed in tests
        names = folder_paths.get_filename_list("diffusion_models") or []
    except Exception:  # noqa: BLE001
        return []
    cands = [os.path.basename(n) for n in names
             if os.path.basename(n).lower().startswith("moss_image")
             and not os.path.basename(n).lower().startswith("moss_image_base")
             and str(n).lower().endswith(".safetensors")]

    def _rank(n):
        low = n.lower()
        return (0 if low.startswith("moss_image_sft") else 1,
                0 if "int8" in low else 1 if "bf16" in low else 2, low)

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
    """The reason this ComfyUI core cannot run MOSS-Image, or ``None``.

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
    try:
        types = mapping["CLIPLoader"].INPUT_TYPES()["required"]["type"][0]
    except Exception:  # noqa: BLE001 -- an unreadable schema is not a refusal
        return None
    if isinstance(types, (list, tuple)) and "moss_image" not in types:
        return _CORE_UPDATE_MESSAGE
    return None


@register
class MossImageEngine:
    """The MOSS-Image SFT adapter (reduced ``prompt -> image`` protocol)."""

    name = "moss_image"
    roles = ROLES
    default_roles = ()               # opt-in; never automatic
    commercial_clean = True          # Apache-2.0, ungated Comfy-Org repackage
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
        width = _snap(get("width") or get("w") or _eint("OTR_MOSS_WIDTH", 1024))
        height = _snap(get("height") or get("h") or _eint("OTR_MOSS_HEIGHT", 1024))
        return {
            "unet_name": unet,
            "clip_name": os.path.basename(otr_env.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(otr_env.get(VAE_ENV, "") or _DEFAULT_VAE),
            "clip_type": _CLIP_TYPE,
            "unet_dtype": otr_env.get("OTR_MOSS_UNET_DTYPE", "default"),
            "prompt": str(get("prompt") or ""),
            "negative": str(get("negative_prompt") or _HYGIENE_NEGATIVE),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_MOSS_STEPS", 25),
            "cfg": _efloat("OTR_MOSS_CFG", 2.5),
            "sampler_name": otr_env.get("OTR_MOSS_SAMPLER", "er_sde"),
            "scheduler": otr_env.get("OTR_MOSS_SCHEDULER", "simple"),
            "width": width,
            "height": height,
        }

    def _node_candidates(self, params=None):  # noqa: ARG002
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptyFlux2LatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_graph(self, params, wire):
        """Pure: Comfy's MOSS-Image t2i graph in ``wrapper_bridge.run_graph``
        form (no upsampler, no skip-layer guidance, no ModelSampling node)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"],
                                "weight_dtype": params["unet_dtype"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": params["clip_type"]}},
            "vae": {"class": "vae", "inputs": {"vae_name": params["vae_name"]}},
            "pos": {"class": "pos",
                    "inputs": {"text": params["prompt"], "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": params["negative"], "clip": W("clip", 0)}},
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
                                    "model": W("unet", 0),
                                    "positive": W("pos", 0),
                                    "negative": W("neg", 0),
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
        old core is refused BEFORE the 13 GB fetch, not at the dispatcher."""
        return _core_problem(mapping)

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        """Fail closed until the diffusion model is INSTALLED and the running
        ComfyUI core can load MOSS-Image (>= v0.42.0)."""
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
                f"moss_image diffusion model not found (resolved {_name!r}, not "
                f"installed): it is fetched on first use, or point {MODEL_ENV} at "
                f"a moss_image_sft*.safetensors (+ {CLIP_ENV} Qwen3.5-9B "
                f"encoder / {VAE_ENV} flux2-vae_small)",
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
            # Drop the 9B Qwen3.5 encoder the moment its consumers have run so
            # the sampler starts with it off the card.
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL,
                                   free_after_use=True, keep={"unet"},
                                   evict_after_use={"clip"})[0]
            frames = _wb.images_to_uint8(images)
        finally:
            _wb.reclaim_idle_models(reason="moss_image post-decode")
        log.info("[OTR.image.moss_image] minted still %dx%d seed=%d steps=%d "
                 "cfg=%.2f", params["width"], params["height"],
                 params["seed"], params["steps"], params["cfg"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["MossImageEngine", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
