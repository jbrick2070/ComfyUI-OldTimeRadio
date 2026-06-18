"""Z-Image-Turbo image adapter -- the LOW-VRAM image option (in-process).

The commercial-clean (Apache-2.0) low-VRAM peer of ``flux_gen1``: a distilled 6B
S3-DiT (Alibaba Tongyi) that renders a still in 8 steps. It registers EXACTLY like
``flux_gen1`` / ``lumina_image`` so the operator picks it per role; Flux stays the
in-stack default (``default_roles=()`` here -- no model is "primary").

ARCHITECTURE (2026-06-18 roundtable-converged; SUPERSEDES the old cu128 sidecar
stub): Z-Image is now a ComfyUI-CORE split-file model, so it runs IN-PROCESS via
``wrapper_bridge`` exactly like ``lumina_image`` -- UNETLoader (diffusion) +
CLIPLoader (the Qwen3-4B text encoder) + VAELoader (the Flux ``ae`` VAE) ->
ModelSamplingAuraFlow (sigma shift) -> KSampler -> VAEDecode. No sidecar.

QWEN3 IS MANDATORY (operator asked): Z-Image's S3-DiT was trained on Qwen3-4B
text embeddings -- there is no CLIP-only Z-Image. It does NOT break the low-VRAM
goal: the DEFAULTS here point at the **fp8** diffusion model + **fp8** Qwen3 TE
(fp8 ~= half size, negligible quality loss; GGUF smaller), and ComfyUI EVICTS the
TE before the diffusion sampling peak, so the resident peak is the diffusion
model, not TE+diffusion co-resident. Point the env knobs at bf16/GGUF to taste.

LOW-VRAM DEFAULTS: 8 steps / cfg 2.0 (keeps the NEGATIVE live -- a lever Flux@cfg
1.0 lacks) / ModelSamplingAuraFlow shift 3.0 / euler / normal. The composed
prompt is REUSED AS-IS from ``compose_still_prompt`` (same grade tails as Flux ->
same filmic look); the live negative pushes Z-Image off its clean-digital default
toward the muted Flux grade.

VERIFY-AT-BUILD (capture from a live ``/object_info`` on the installed Z-Image
nodes before the first real render -- these cannot be confirmed from the repo and
are env-overridable so the operator never edits code):
  * ``OTR_ZIMAGE_CLIP_TYPE`` -- the exact CLIPLoader ``type`` for Qwen3 (default
    ``"z_image"``; could be ``"qwen3"`` etc.).
  * ``OTR_ZIMAGE_LATENT_NODE`` -- ``EmptySD3LatentImage`` (16-ch, matches the Flux
    ae VAE) vs ``EmptyLatentImage`` (4-ch).
  * ModelSamplingAuraFlow is the right shift node (S3-DiT != guaranteed AuraFlow).
  * ``OTR_ZIMAGE_UNET_DTYPE`` -- UNETLoader weight_dtype (default ``"default"``).

Fail-closed: ``assert_usable`` raises MISSING_MODEL until the diffusion-model file
exists (the TE+VAE loaders fail LOUD at render -> dispatcher floor). Greyed until
``OTR_ENABLE_ZIMAGE=1``. Kept OUT of ``registry.VALIDATED_ENGINES`` (hidden from
the tested-only dropdown) until the GPU A/B look-match passes.

Cold-import clean (V-12): module scope imports only the dep-free registry + role
vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the heavy
path is lazy, inside ``render_image`` (via wrapper_bridge), mirroring flux/lumina.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.z_image_turbo")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_ZIMAGE"

#: Split-file weights. Defaults target the LOW-VRAM fp8 variants (the whole point
#: of this engine); point them at bf16/GGUF to taste. The loaders take a basename
#: (ComfyUI folder_paths resolves it), so an absolute path is reduced to filename.
MODEL_ENV = "OTR_ZIMAGE_UNET"
CLIP_ENV = "OTR_ZIMAGE_CLIP"
VAE_ENV = "OTR_ZIMAGE_VAE"
_DEFAULT_UNET = "z_image_turbo_fp8_scaled.safetensors"
_DEFAULT_CLIP = "qwen3_4b_fp8_scaled.safetensors"
_DEFAULT_VAE = "ae.safetensors"

#: VERIFY-AT-BUILD knobs (env-overridable; confirm against the installed node).
_DEFAULT_CLIP_TYPE = "z_image"          # CLIPLoader type for the Qwen3-4B TE
_DEFAULT_LATENT_NODE = "EmptySD3LatentImage"   # 16-ch (matches the Flux ae VAE)


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class ZImageTurboEngine:
    """The Z-Image-Turbo image adapter (reduced ``prompt -> image`` protocol)."""

    name = "z_image_turbo"
    roles = ROLES
    default_roles = ()               # opt-in peer (Flux remains the default)
    commercial_clean = True          # Apache-2.0
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    #: Terminal graph node (its IMAGE output is the still).
    _TERMINAL = "decode"

    # ---- params / graph (pure; CPU-testable) ----------------------------
    def _zimage_params(self, request):
        """Pure: resolve the Z-Image sampler params from the request + env. The
        model / TE / VAE / steps / cfg / shift / sampler / clip-type / latent-node
        are env-overridable; the seed + prompt + dims come from the request so a
        re-gen is deterministic (V-7). Low-VRAM fp8 defaults; 8 steps / cfg 2.0 /
        shift 3.0 / euler / normal (the roundtable-converged starting config)."""
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
            "unet_name": os.path.basename(os.environ.get(MODEL_ENV, "") or _DEFAULT_UNET),
            "clip_name": os.path.basename(os.environ.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(os.environ.get(VAE_ENV, "") or _DEFAULT_VAE),
            "clip_type": os.environ.get("OTR_ZIMAGE_CLIP_TYPE", _DEFAULT_CLIP_TYPE),
            "latent_node": os.environ.get("OTR_ZIMAGE_LATENT_NODE", _DEFAULT_LATENT_NODE),
            "unet_dtype": os.environ.get("OTR_ZIMAGE_UNET_DTYPE", "default"),
            "prompt": str(get("prompt") or ""),
            "negative": os.environ.get(
                "OTR_ZIMAGE_NEGATIVE",
                "oversaturated, glossy, clean digital, plastic skin, waxy skin, "
                "sterile studio lighting, cartoon, illustration, text, watermark"),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_ZIMAGE_STEPS", 8),       # distilled design point
            "cfg": _efloat("OTR_ZIMAGE_CFG", 2.0),       # keeps the negative live
            "shift": _efloat("OTR_ZIMAGE_SHIFT", 3.0),   # ModelSamplingAuraFlow
            "sampler_name": os.environ.get("OTR_ZIMAGE_SAMPLER", "euler"),
            "scheduler": os.environ.get("OTR_ZIMAGE_SCHEDULER", "normal"),
            # Request dims win (aspect-aware still spine); env knobs are the
            # no-request default. Honor dims EXACTLY -- no snapping/upscale here.
            "width": int(get("width") or get("w") or _eint("OTR_ZIMAGE_WIDTH", 1024)),
            "height": int(get("height") or get("h") or _eint("OTR_ZIMAGE_HEIGHT", 1024)),
        }

    def _node_candidates(self, params=None):
        """Ordered ComfyUI node-class candidates per graph node. The latent node
        is VERIFY-AT-BUILD (16-ch SD3 vs 4-ch) -> env-selected, candidate-ordered."""
        latent = (params or {}).get("latent_node", _DEFAULT_LATENT_NODE)
        latent_candidates = (latent, "EmptySD3LatentImage", "EmptyLatentImage")
        # de-dup preserving order
        seen = []
        for c in latent_candidates:
            if c not in seen:
                seen.append(c)
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "sampling": ("ModelSamplingAuraFlow",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": tuple(seen),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_zimage_graph(self, params, wire):
        """Pure: the declarative Z-Image txt2img graph (wrapper_bridge.run_graph
        format), mirroring lumina's split-file AuraFlow recipe. UNETLoader out
        0=MODEL; CLIPLoader out 0=CLIP; VAELoader out 0=VAE; ModelSamplingAuraFlow
        out 0=MODEL (shifted)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"],
                                "weight_dtype": params["unet_dtype"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": params["clip_type"]}},
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
        """FAIL CLOSED until the Z-Image diffusion model exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the diffusion WEIGHTS
        file). The TE + VAE loaders fail LOUD at render if their files are
        absent. The model path is the env value if set, else the default basename
        resolved via ComfyUI folder_paths at render."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if ckpt and not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"z_image_turbo diffusion model not found at {MODEL_ENV}={ckpt!r}; "
                f"point it at the downloaded fp8/bf16 z_image_turbo diffusion "
                f"model (and {CLIP_ENV}/{VAE_ENV} for the Qwen3-4B TE + ae VAE)",
                kind="image",
            )
        if not ckpt:
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"z_image_turbo diffusion model not configured; set {MODEL_ENV} to "
                f"the downloaded z_image_turbo diffusion model (fp8 for low VRAM) "
                f"after enabling {ENABLE_FLAG}=1, plus {CLIP_ENV} (Qwen3-4B TE) "
                f"and {VAE_ENV} (Flux ae)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU/operator
        """Mint ONE still via the in-stack ComfyUI Z-Image graph and return it as a
        decoded uint8 (H,W,3) RGB array (the dispatcher content-addresses + stamps
        it). Drives the native split-file recipe through wrapper_bridge, then
        reclaims the resident model (BUG-291 detach) so VRAM drops back under the
        single-resident ceiling. Raises a NAMED wrapper error on a missing node /
        file / failed render -- the dispatcher catches it fail-closed -> radio
        floor (LOUD)."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._zimage_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates(params))
        self._classes = classes
        graph = self._build_zimage_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: free the model so the next heavy engine
            # can take the lease (LOUD; detach only, never unload_all_models).
            _wb.reclaim_idle_models(reason="z_image_turbo post-decode")
        log.info(
            "[OTR.image.z_image_turbo] minted still %dx%d seed=%d steps=%d "
            "cfg=%.2f shift=%.2f sampler=%s/%s", params["width"], params["height"],
            params["seed"], params["steps"], params["cfg"], params["shift"],
            params["sampler_name"], params["scheduler"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["ZImageTurboEngine", "ENABLE_FLAG", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
