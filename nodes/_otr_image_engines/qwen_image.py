"""Qwen-Image (GGUF) image adapter -- the THIRD image engine (C3, generic peer), BUILT.

Carries the model-agnostic image layer from "it holds two engines" (C2) to "it
holds an OPEN, growable set": Qwen-Image registers EXACTLY like ``flux_gen1`` /
``z_image_turbo`` / ``lumina_image``, so the registry holds >=N peers the user
picks per role. Per ``C2_DEP_LICENSE_MATRIX.md`` Qwen-Image is the all-rounder:
20B, GGUF (Q-quant), Apache-2.0 (commercial-clean), strong in-image text.

The C3 demonstration -- a DIFFERENT integration mode behind the SAME protocol.
Qwen-Image GGUF rides the ALREADY-INSTALLED in-stack ComfyUI-GGUF loader on the
protected cu130 main venv: a GGUF checkpoint is dtype-isolated and does NOT swap
torch or drag a banned dep onto the stack, so it needs no cu128 sidecar. The
native ComfyUI recipe (grounded against the official
``image_qwen_image_2512_with_2steps_lora`` template, minus the lightning LoRA):
UnetLoaderGGUF (the 20B diffusion GGUF) + CLIPLoader ``type="qwen_image"`` (the
Qwen-2.5-VL 7B text encoder) + VAELoader (the Qwen-Image VAE) ->
ModelSamplingAuraFlow (the AuraFlow sigma shift) -> CLIPTextEncode pos/neg ->
EmptySD3LatentImage -> KSampler -> VAEDecode. Same shape as ``lumina_image``;
the diffusion loader is GGUF (UnetLoaderGGUF) instead of UNETLoader, and the CLIP
type is ``qwen_image``.

Flux stays gen 1; Qwen is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"), greyed until ``OTR_ENABLE_QWEN_IMAGE=1`` AND its GGUF exists. The
fail-closed gate is the WEIGHTS FILE (``OTR_QWEN_IMAGE_GGUF``): ``assert_usable``
raises MISSING_MODEL until it points at the downloaded ``.gguf`` (ABSENT/greyed,
never a silent stub -- BUG-046). The TE + VAE loaders fail LOUD at render if their
files are absent (the dispatcher catches it fail-closed -> radio floor).

VRAM (build-or-NO-GO, 2026-06-19): the 20B core is TIGHT under the 14.5 GB
single-resident ceiling -- the Qwen-2.5-VL TE (~5 GB fp8) offloads before
sampling, leaving the GGUF diffusion (Q3/Q4 ~8-11 GB) + VAE resident. The
per-quant resident peak MUST be measured on the 5080 before promotion to
``VALIDATED_ENGINES`` (it stays HIDDEN until that GPU smoke passes the ceiling).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / the ComfyUI-GGUF loader / the model are NEVER
imported here -- the heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.qwen_image")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1"
#: (build-or-NO-GO: promoted only after the operator's verify-on-5080 GO).
ENABLE_FLAG = "OTR_ENABLE_QWEN_IMAGE"

#: Env var pointing at the downloaded Qwen-Image GGUF (Q-quant) diffusion file.
#: Absent / not a file -> ``assert_usable`` fails closed (the deeper disk check
#: beyond the flag). In-stack GGUF (the installed ComfyUI-GGUF loader), no sidecar.
MODEL_ENV = "OTR_QWEN_IMAGE_GGUF"

#: The split-file companions: the Qwen-2.5-VL text encoder (CLIPLoader type
#: qwen_image) and the Qwen-Image VAE. Default to the official Comfy-Org
#: filenames in the standard model dirs; the loaders resolve a basename via
#: ComfyUI folder_paths, so either an absolute path or a bare filename works.
CLIP_ENV = "OTR_QWEN_IMAGE_CLIP"
VAE_ENV = "OTR_QWEN_IMAGE_VAE"
_DEFAULT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
_DEFAULT_VAE = "qwen_image_vae.safetensors"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class QwenImageEngine:
    """The Qwen-Image (GGUF) image adapter (reduced ``prompt -> image`` protocol)."""

    name = "qwen_image"
    #: Can serve any image role; OPT-IN peer (Flux remains the in-stack default
    #: everywhere, so default_roles is empty here -- no model is "primary").
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # Apache-2.0 (C2 matrix; operator confirms weights provenance)
    requires_flag = ENABLE_FLAG      # default-OFF (build-or-NO-GO)
    required_inputs = ("text_prompt",)
    engine_version = "1"

    #: Terminal graph node (its IMAGE output is the still).
    _TERMINAL = "decode"

    # ---- params / graph (pure; CPU-testable) ----------------------------
    def _qwen_params(self, request):
        """Pure: resolve the Qwen-Image sampler params from the request + env. The
        diffusion GGUF / TE / VAE / steps / cfg / shift / sampler are env-
        overridable (the operator points at the installed files without editing
        code); the seed + prompt + dims come from the request so a re-gen is
        deterministic (V-7). The loaders take a basename (ComfyUI folder_paths
        resolves it), so an absolute path is reduced to its filename. Defaults
        (shift 3 / 20 steps / cfg 2.5 / euler-simple) are the standard non-
        lightning Qwen-Image settings, all env-tunable."""
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
            "unet_name": os.path.basename(os.environ.get(MODEL_ENV, "") or ""),
            "clip_name": os.path.basename(os.environ.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(os.environ.get(VAE_ENV, "") or _DEFAULT_VAE),
            "prompt": str(get("prompt") or ""),
            "negative": os.environ.get("OTR_QWEN_IMAGE_NEGATIVE", ""),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_QWEN_IMAGE_STEPS", 20),
            "cfg": _efloat("OTR_QWEN_IMAGE_CFG", 2.5),
            "shift": _efloat("OTR_QWEN_IMAGE_SHIFT", 3.0),
            "sampler_name": os.environ.get("OTR_QWEN_IMAGE_SAMPLER", "euler"),
            "scheduler": os.environ.get("OTR_QWEN_IMAGE_SCHEDULER", "simple"),
            # Request dims take precedence (still-spine: w/h plumbed end-to-end so
            # landscape SCENE stills are real); env knobs are the no-request default.
            "width": int(get("width") or get("w") or _eint("OTR_QWEN_IMAGE_WIDTH", 1024)),
            "height": int(get("height") or get("h") or _eint("OTR_QWEN_IMAGE_HEIGHT", 1024)),
        }

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node -- the GGUF
        diffusion loader + stock core classes (the native Qwen-Image recipe,
        verified against the installed ComfyUI-GGUF + the official template)."""
        return {
            "unet": ("UnetLoaderGGUF",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "sampling": ("ModelSamplingAuraFlow",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptySD3LatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_qwen_graph(self, params, wire):
        """Pure: the declarative Qwen-Image txt2img graph (wrapper_bridge.run_graph
        format). UnetLoaderGGUF out 0=MODEL; CLIPLoader out 0=CLIP; VAELoader out
        0=VAE; ModelSamplingAuraFlow out 0=MODEL (shifted)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": "qwen_image"}},
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
        """FAIL CLOSED until the Qwen-Image GGUF diffusion file exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file). The
        TE + VAE loaders fail LOUD at render if their files are absent."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"qwen_image GGUF diffusion not found; set {MODEL_ENV} to the "
                f"downloaded Qwen-Image .gguf path (and {CLIP_ENV}/{VAE_ENV} for "
                f"the Qwen-2.5-VL TE + VAE) after enabling {ENABLE_FLAG}=1 -- "
                f"per-quant VRAM must be confirmed on the 5080 first (20B is tight)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU/operator
        """Mint ONE still via the in-stack ComfyUI Qwen-Image GGUF graph and return
        it as a decoded uint8 (H,W,3) RGB array (the dispatcher content-addresses +
        stamps it). Drives the native GGUF recipe through wrapper_bridge, then
        reclaims the resident model (BUG-291 detach) so VRAM drops back under the
        single-resident ceiling. Raises a NAMED wrapper error on a missing node /
        file / failed render -- the dispatcher catches it fail-closed and the
        episode falls to the radio floor (LOUD)."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._qwen_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        self._classes = classes
        graph = self._build_qwen_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: free the model so the next heavy engine
            # can take the lease (LOUD; detach only, never unload_all_models).
            _wb.reclaim_idle_models(reason="qwen_image post-decode")
        log.info(
            "[OTR.image.qwen_image] minted still %dx%d seed=%d steps=%d "
            "cfg=%.2f shift=%.2f", params["width"], params["height"],
            params["seed"], params["steps"], params["cfg"], params["shift"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["QwenImageEngine", "ENABLE_FLAG", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
