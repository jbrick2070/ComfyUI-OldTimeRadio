"""FLUX.2 [klein] 4B image adapter -- a model-agnostic image peer (C8), BUILT.

A matrix peer from ``C2_DEP_LICENSE_MATRIX.md`` (the GO #2 candidate: strong pose +
multi-reference character consistency). FLUX.2 [klein] 4B is **Apache-2.0**
(confirmed against the Black Forest Labs / unsloth Hugging Face cards, 2026-06-18),
so it is commercial-clean -- the earlier "verify (FLUX family)" caveat is resolved.

The smallest comfy-compatible set for a 16 GB Blackwell laptop (the operator's
"balance of VRAM and simplicity"):

* diffusion: ``flux-2-klein-4b-Q4_K_M.gguf`` (~2.6 GB) via ComfyUI-GGUF's
  ``UnetLoaderGGUF`` (already installed) -- small resident VRAM.
* text encoder: ``qwen_3_4b.safetensors`` (~8.0 GB) via the stock ``CLIPLoader``
  typed ``flux2``. klein-4B is paired with the **Qwen-3-4B** encoder (7680-wide
  conditioning), NOT flux2-dev's Mistral-3 (15360-wide) -- that exact 2x mismatch
  caused the ``mat1/mat2 (512x15360 @ 7680x3072)`` sampler error. Verified against
  the official ComfyUI ``image_flux2_klein_text_to_image`` template, whose
  CLIPLoader loads ``qwen_3_4b.safetensors`` (type ``flux2``). Source repo:
  ``Comfy-Org/flux2-klein`` (split_files/text_encoders). 63 GB system RAM absorbs
  the offload; an fp4 variant (``qwen_3_4b_fp4_flux2.safetensors``, ~3.85 GB) is a
  later Blackwell-native VRAM optimization.
* VAE: ``flux2-vae.safetensors`` (~0.34 GB) via ``VAELoader``.

FLUX.2's sampling path is the CUSTOM-sampler route (NOT plain KSampler), verified
against the official ComfyUI flux2 template + the live node schemas:
``UnetLoaderGGUF`` -> ``CLIPLoader[type=flux2]`` -> ``CLIPTextEncode`` ->
``FluxGuidance`` -> ``BasicGuider``; ``EmptyFlux2LatentImage`` (128-ch latent) +
``Flux2Scheduler`` (sigmas) + ``KSamplerSelect`` (euler) + ``RandomNoise`` ->
``SamplerCustomAdvanced`` -> ``VAEDecode``.

Flux gen-1 stays the in-stack default; klein is an OPT-IN peer
(``default_roles=()`` -- no model is "primary"), greyed until
``OTR_ENABLE_FLUX2_KLEIN=1`` AND its checkpoint exists. The fail-closed gate is the
WEIGHTS FILE (``OTR_FLUX2_KLEIN_CKPT``): ``assert_usable`` raises MISSING_MODEL
until it points at the downloaded GGUF (ABSENT/greyed, never a stub -- BUG-046).
The TE + VAE loaders fail LOUD at render if their files are absent (the dispatcher
catches it fail-closed). The verify-on-5080 must also confirm SageAttention is NOT
patched onto the FLUX-style attention before the first forward (BUG-070, sm_120).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the
heavy path is lazy, inside ``render_image`` (via wrapper_bridge), mirroring
``flux_gen1`` / ``lumina_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.flux2_klein")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_FLUX2_KLEIN"

#: Env var pointing at the downloaded FLUX.2 klein diffusion GGUF. Absent / not a
#: file -> ``assert_usable`` fails closed. The GGUF UnetLoader resolves a basename
#: via ComfyUI folder_paths, so set this to the full path (gate) -- the loader uses
#: its basename.
MODEL_ENV = "OTR_FLUX2_KLEIN_CKPT"

#: Split-file companions: the Qwen-3-4B flux2 text encoder (CLIPLoader type flux2 --
#: klein-4B's matched encoder, 7680-wide; NOT Mistral) and the flux2 VAE. Default to
#: the Comfy-Org repackaged filenames in the standard model dirs; the loaders resolve
#: a basename via folder_paths, so either an absolute path or a bare filename works.
CLIP_ENV = "OTR_FLUX2_KLEIN_TE"
VAE_ENV = "OTR_FLUX2_KLEIN_VAE"
_DEFAULT_CKPT = "flux-2-klein-4b-Q4_K_M.gguf"
_DEFAULT_CLIP = "qwen_3_4b.safetensors"
_DEFAULT_VAE = "flux2-vae.safetensors"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class Flux2KleinEngine:
    """The FLUX.2 [klein] 4B image adapter (reduced ``prompt -> image`` protocol)."""

    name = "flux2_klein"
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # FLUX.2 [klein] 4B = Apache-2.0 (confirmed 2026-06-18)
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    #: Terminal graph node (its IMAGE output is the still).
    _TERMINAL = "decode"

    # ---- params / graph (pure; CPU-testable) ----------------------------
    def _klein_params(self, request):
        """Pure: resolve the FLUX.2 klein sampler params from the request + env.
        The diffusion GGUF / TE / VAE / steps / guidance / sampler are
        env-overridable (the operator points at the installed files without editing
        code); the seed + prompt + dims come from the request so a re-gen is
        deterministic (V-7). The loaders take a basename (ComfyUI folder_paths
        resolves it), so an absolute path is reduced to its filename. Official
        flux2 sampling is ~20 steps, guidance 4, euler -- env-tunable. Dims are
        snapped to a multiple of 16 (the flux2 latent is width//16)."""
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

        def _snap16(v):
            v = max(16, int(v))
            return v - (v % 16)

        return {
            "unet_name": os.path.basename(os.environ.get(MODEL_ENV, "") or _DEFAULT_CKPT),
            "clip_name": os.path.basename(os.environ.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(os.environ.get(VAE_ENV, "") or _DEFAULT_VAE),
            "prompt": str(get("prompt") or ""),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_FLUX2_KLEIN_STEPS", 20),
            "guidance": _efloat("OTR_FLUX2_KLEIN_GUIDANCE", 4.0),
            "sampler_name": os.environ.get("OTR_FLUX2_KLEIN_SAMPLER", "euler"),
            # Request dims take precedence (still-spine: w/h plumbed end-to-end so
            # landscape SCENE stills are real); env knobs are the no-request default.
            "width": _snap16(get("width") or get("w") or _eint("OTR_FLUX2_KLEIN_WIDTH", 1024)),
            "height": _snap16(get("height") or get("h") or _eint("OTR_FLUX2_KLEIN_HEIGHT", 1024)),
        }

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node -- the GGUF UNet
        loader (ComfyUI-GGUF) + stock flux2 custom-sampler classes (verified on
        /object_info + the official flux2 template)."""
        return {
            "unet": ("UnetLoaderGGUF",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "pos": ("CLIPTextEncode",),
            "guidance": ("FluxGuidance",),
            "latent": ("EmptyFlux2LatentImage",),
            "scheduler": ("Flux2Scheduler",),
            "sampler": ("KSamplerSelect",),
            "noise": ("RandomNoise",),
            "guider": ("BasicGuider",),
            "sample": ("SamplerCustomAdvanced",),
            "decode": ("VAEDecode",),
        }

    def _build_klein_graph(self, params, wire):
        """Pure: the declarative FLUX.2 klein txt2img graph (wrapper_bridge.run_graph
        format). UnetLoaderGGUF out 0=MODEL; CLIPLoader out 0=CLIP; VAELoader out
        0=VAE; FluxGuidance out 0=CONDITIONING; EmptyFlux2LatentImage out 0=LATENT;
        Flux2Scheduler out 0=SIGMAS; KSamplerSelect out 0=SAMPLER; RandomNoise out
        0=NOISE; BasicGuider out 0=GUIDER; SamplerCustomAdvanced out 0=output LATENT
        (1=denoised)."""
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": "flux2"}},
            "vae": {"class": "vae",
                    "inputs": {"vae_name": params["vae_name"]}},
            "pos": {"class": "pos",
                    "inputs": {"text": params["prompt"], "clip": W("clip", 0)}},
            # FluxGuidance bakes the guidance value into the conditioning (flux2's
            # richness/adherence lever); BasicGuider then reads the guided positive.
            "guidance": {"class": "guidance",
                         "inputs": {"conditioning": W("pos", 0),
                                    "guidance": float(params["guidance"])}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(params["width"]),
                                  "height": int(params["height"]),
                                  "batch_size": 1}},
            "scheduler": {"class": "scheduler",
                          "inputs": {"steps": int(params["steps"]),
                                     "width": int(params["width"]),
                                     "height": int(params["height"])}},
            "sampler": {"class": "sampler",
                        "inputs": {"sampler_name": params["sampler_name"]}},
            "noise": {"class": "noise",
                      "inputs": {"noise_seed": int(params["seed"])}},
            "guider": {"class": "guider",
                       "inputs": {"model": W("unet", 0),
                                  "conditioning": W("guidance", 0)}},
            "sample": {"class": "sample",
                       "inputs": {"noise": W("noise", 0),
                                  "guider": W("guider", 0),
                                  "sampler": W("sampler", 0),
                                  "sigmas": W("scheduler", 0),
                                  "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("sample", 0),
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
        """FAIL CLOSED until the FLUX.2 klein diffusion GGUF exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file). The
        TE + VAE loaders fail LOUD at render if their files are absent. The
        verify-on-5080 must also confirm SageAttention is not patched (BUG-070,
        FLUX-style attn)."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"flux2_klein diffusion GGUF not found; set {MODEL_ENV} to the "
                f"downloaded flux-2-klein-4b-Q4_K_M.gguf path (and {CLIP_ENV}"
                f"/{VAE_ENV} for the Qwen-3-4B flux2 TE + flux2 VAE) after enabling "
                f"{ENABLE_FLAG}=1 (confirm SageAttention not patched -- BUG-070)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU/operator
        """Mint ONE still via the in-stack ComfyUI FLUX.2 klein graph and return it
        as a decoded uint8 (H,W,3) RGB array (the dispatcher content-addresses +
        stamps it). Drives the GGUF + custom-sampler recipe through wrapper_bridge,
        then reclaims the resident model (BUG-291 detach) so VRAM drops back under
        the single-resident ceiling. Raises a NAMED wrapper error on a missing
        node / file / failed render -- the dispatcher catches it fail-closed and
        the episode falls to the radio floor (LOUD)."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._klein_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        self._classes = classes
        graph = self._build_klein_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: free the model so the next heavy engine
            # can take the lease (LOUD; detach only, never unload_all_models).
            _wb.reclaim_idle_models(reason="flux2_klein post-decode")
        log.info(
            "[OTR.image.flux2_klein] minted still %dx%d seed=%d steps=%d "
            "guidance=%.2f", params["width"], params["height"],
            params["seed"], params["steps"], params["guidance"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["Flux2KleinEngine", "ENABLE_FLAG", "MODEL_ENV", "CLIP_ENV", "VAE_ENV"]
