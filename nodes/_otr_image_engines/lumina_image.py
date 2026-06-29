"""Lumina-Image 2.0 image adapter -- a model-agnostic image peer (C6), BUILT.

A 6th image engine from ``C2_DEP_LICENSE_MATRIX.md`` (the lightweight Apache-2.0
candidate). It registers EXACTLY like the other image peers, growing the open
per-role registry. Lumina-Image 2.0 is a NATIVE flow model loaded through
ComfyUI's own loaders on the protected main venv (not GGUF, not a sidecar): the
split-file recipe is UNETLoader (the 2.6B bf16 diffusion model) + CLIPLoader with
``type="lumina2"`` (the Gemma-2 2B text encoder) + VAELoader (the Flux ``ae`` VAE)
-> ModelSamplingAuraFlow (the AuraFlow/Lumina-2 sigma shift) -> KSampler ->
VAEDecode. Lightweight (~7 GB working set, comfortably under the 14.5 GB single-
resident ceiling) and Apache-2.0 (commercial-clean).

Flux stays gen 1; Lumina is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"), greyed until ``OTR_ENABLE_LUMINA=1`` AND its checkpoint exists. The
fail-closed gate is the WEIGHTS FILE (``OTR_LUMINA_CKPT``): ``assert_usable``
raises MISSING_MODEL until it points at the downloaded diffusion model (ABSENT/
greyed, never a silent stub -- BUG-046). The TE + VAE loaders fail LOUD at render
if their files are absent (the dispatcher catches it fail-closed -> radio floor).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the
heavy path is lazy, inside ``render_image`` (via wrapper_bridge), mirroring
``flux_gen1``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.lumina_image")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_LUMINA"

#: Env var pointing at the downloaded Lumina-Image 2.0 diffusion model. Absent /
#: not a file -> ``assert_usable`` fails closed. Native in-stack load (ComfyUI
#: loaders), so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_LUMINA_CKPT"

#: The split-file companions: the Gemma-2 2B text encoder (CLIPLoader type
#: lumina2) and the Flux ``ae`` VAE. Default to the Comfy-Org repackaged
#: filenames in the standard model dirs; the loaders resolve a basename via
#: ComfyUI folder_paths, so either an absolute path or a bare filename works.
CLIP_ENV = "OTR_LUMINA_CLIP"
VAE_ENV = "OTR_LUMINA_VAE"
_DEFAULT_CKPT = "lumina_2_model_bf16.safetensors"
_DEFAULT_CLIP = "gemma_2_2b_fp16.safetensors"
_DEFAULT_VAE = "lumina2_ae.safetensors"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class LuminaImage2Engine:
    """The Lumina-Image 2.0 image adapter (reduced ``prompt -> image`` protocol)."""

    name = "lumina_image"
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # Apache-2.0 (C2 matrix; operator confirms weights provenance)
    requires_flag = None             # vestigial (registry IS the menu; no flag gate)
    required_inputs = ("text_prompt",)
    engine_version = "1"

    #: Terminal graph node (its IMAGE output is the still).
    _TERMINAL = "decode"

    # ---- params / graph (pure; CPU-testable) ----------------------------
    def _lumina_params(self, request):
        """Pure: resolve the Lumina-2 sampler params from the request + env. The
        model / TE / VAE / steps / cfg / shift / sampler are env-overridable (the
        operator points at the installed files without editing code); the seed +
        prompt + dims come from the request so a re-gen is deterministic (V-7).
        The loaders take a basename (ComfyUI folder_paths resolves it), so an
        absolute path is reduced to its filename. Official sampling is shift 6 /
        36 steps; the defaults here (shift 6 / 30 steps / cfg 4) are a balanced
        starting point, all env-tunable."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))

        def _eint(name, default):
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        def _efloat(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        return {
            "unet_name": os.path.basename(os.environ.get(MODEL_ENV, "") or _DEFAULT_CKPT),
            "clip_name": os.path.basename(os.environ.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(os.environ.get(VAE_ENV, "") or _DEFAULT_VAE),
            "prompt": str(get("prompt") or ""),
            "negative": os.environ.get("OTR_LUMINA_NEGATIVE", ""),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_LUMINA_STEPS", 30),
            "cfg": _efloat("OTR_LUMINA_CFG", 4.0),
            "shift": _efloat("OTR_LUMINA_SHIFT", 6.0),
            "sampler_name": os.environ.get("OTR_LUMINA_SAMPLER", "euler"),
            "scheduler": os.environ.get("OTR_LUMINA_SCHEDULER", "normal"),
            # Request dims take precedence (still-spine: w/h plumbed end-to-end so
            # landscape SCENE stills are real); env knobs are the no-request default.
            "width": int(get("width") or get("w") or _eint("OTR_LUMINA_WIDTH", 1024)),
            "height": int(get("height") or get("h") or _eint("OTR_LUMINA_HEIGHT", 1024)),
        }

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node -- stock core
        classes (the native Lumina-2 recipe, verified on /object_info)."""
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "sampling": ("ModelSamplingAuraFlow",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptySD3LatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_lumina_graph(self, params, wire):
        """Pure: the declarative Lumina-2 txt2img graph (wrapper_bridge.run_graph
        format). UNETLoader out 0=MODEL; CLIPLoader out 0=CLIP; VAELoader out
        0=VAE; ModelSamplingAuraFlow out 0=MODEL (shifted)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"],
                                "weight_dtype": "default"}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": "lumina2"}},
            "vae": {"class": "vae",
                    "inputs": {"vae_name": params["vae_name"]}},
            "sampling": {"class": "sampling",
                         "inputs": {"model": W("unet", 0),
                                    "shift": float(params["shift"])}},
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
                                    "model": W("sampling", 0),
                                    "positive": W("pos", 0),
                                    "negative": W("neg", 0),
                                    "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("ksampler", 0),
                                  "vae": W("vae", 0)}},
        }

    # ---- residency (classes resolve lazily; loader nodes own the weights) ----
    def load(self):  # pragma: no cover - resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the Lumina-Image 2.0 diffusion model exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file). The
        TE + VAE loaders fail LOUD at render if their files are absent."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"lumina_image diffusion model not found; set {MODEL_ENV} to the "
                f"downloaded lumina_2_model_bf16.safetensors path (and {CLIP_ENV}"
                f"/{VAE_ENV} for the Gemma-2 TE + ae VAE) after enabling "
                f"{ENABLE_FLAG}=1",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU/operator
        """Mint ONE still via the in-stack ComfyUI Lumina-2 graph and return it as
        a decoded uint8 (H,W,3) RGB array (the dispatcher content-addresses +
        stamps it). Drives the native split-file recipe through wrapper_bridge,
        then reclaims the resident model (BUG-291 detach) so VRAM drops back under
        the single-resident ceiling. Raises a NAMED wrapper error on a missing
        node / file / failed render -- the dispatcher catches it fail-closed and
        the episode falls to the radio floor (LOUD)."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._lumina_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        self._classes = classes
        graph = self._build_lumina_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: free the model so the next heavy engine
            # can take the lease (LOUD; detach only, never unload_all_models).
            _wb.reclaim_idle_models(reason="lumina_image post-decode")
        log.info(
            "[OTR.image.lumina_image] minted still %dx%d seed=%d steps=%d "
            "cfg=%.2f shift=%.2f", params["width"], params["height"],
            params["seed"], params["steps"], params["cfg"], params["shift"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["LuminaImage2Engine", "ENABLE_FLAG", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
