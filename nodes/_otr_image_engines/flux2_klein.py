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

Flux gen-1 stays the in-stack default; klein is a selectable peer
(``default_roles=()`` -- no model is "primary"). The registry is the menu, so
there is no environment flag gate. The fail-closed gate is the WEIGHTS FILE:
``assert_usable`` resolves the default GGUF through ComfyUI ``folder_paths``;
``OTR_FLUX2_KLEIN_CKPT`` is only an explicit full-path override (ABSENT/greyed,
never a stub -- BUG-046).
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

#: Historical compatibility name. ``requires_flag`` is None; the registry is
#: the menu and this variable does not gate selection.
ENABLE_FLAG = "OTR_ENABLE_FLUX2_KLEIN"

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "One present-tense sentence in natural prose, under 24 words. Begin with the "
    "subject and action. Preserve every required beat fact. Camera direction and "
    "speed only from Camera; if NONE, no camera wording. No tags, weights, or "
    "negatives."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
PROVENANCE FIRST, because this one is different from the other ten: **the operator
supplied this directive himself on 2026-08-17, drafted from PUBLIC DOCS and
explicitly labelled "NOT yet validated".** It is stored verbatim, not rewritten --
the standing rule is to fix a colliding token and never the surrounding writing.
The ten engines the RESEARCH doc enumerated carry driver-derived directives; this
engine and its two siblings (``hidream_i1``, ``sd35_large``) carry the operator's
drafts. Neither set is measured.

THIS ENGINE IS NOT FLAG-GATED IN PRACTICE. ``ENABLE_FLAG`` exists above, but the
class sets ``requires_flag = None`` and calls it "vestigial (registry IS the menu;
no flag gate)". So unlike its two default-OFF siblings, flux2_klein is SELECTABLE
as shipped -- which is why it was the one worth flagging when the audit found three
registered image engines outside the RESEARCH doc's ten.

FLUX.2 TAKES NO NEGATIVE AT ALL, so the directive's closing "no tags, weights, or
negatives" is a prohibition on the WRITER emitting any -- which is the strongest
possible form of the 2026-08-17 strike, not a violation of it. (It did trip the
first version of the test guard, which flagged any mention of "negative" without an
approved has-no-effect hedge; the guard was widened to recognise a PROHIBITION as
compliant. The directive was not touched.)

REGISTRY NOTES -- v2, VALIDATED by the operator against public docs 2026-08-17.
Recorded as authored; the probe gate still rules and none of this is built.

IDENTITY: released **2026-01-15**. Two tiers, 4B and 9B, and **both are step- AND
guidance-distilled at 4 inference steps**. Undistilled Base tiers (~50 steps) exist
for LoRA / fine-tune work ONLY -- not a render option.

**RECORD tier=4B, and the reason is licence as much as VRAM.** The 4B is Apache-2.0
at ~13GB bf16, with official FP8/NVFP4 quants cutting VRAM 40-55% and benchmarked on
the RTX 5080 -- this box. The 9B is **non-commercial licence**, ~29GB, with a
Qwen3-8B embedder: out of budget AND out of licence. Two independent disqualifiers,
either of which settles it.

NEGATIVES: **CONFIRMED none.** BFL states FLUX.2 takes no negative prompts
family-wide, and distilled klein runs guidance 1.0 in the official pipeline -- inert
twice over. **Third-party pages quoting "guidance 3.0-4.0" are describing the
EMBEDDED-GUIDANCE parameter on hosted/turbo variants, not a CFG branch**, so a
reader who finds one has not found a live negative. This is why the directive's "no
tags, weights, or negatives" is a prohibition on the writer rather than advice.

ENCODER: an LLM embedder. Qwen3-8B is confirmed on the 9B; **the 4B's exact Qwen3
size is UNCONFIRMED -- read the checkpoint's `text_encoder` config at install.**
Prose only: `(word:1.5)` and `++` are SILENTLY IGNORED, and tag lists parse as
broken English. Klein applies the prompt as-is: there is no auto-enhancement layer
to rescue a sloppy prompt.

style_token_position: **APPEND, and it is now evidence-backed rather than
drafted.** BFL documents front-to-back token weighting with priority
subject > action > style > context, and style-first demonstrably makes style
dominate while the subject drifts. That is the same mechanism recorded on
``eng_wan_ti2v.PROMPT_STYLE_NOTES`` (Wan's subject-first training) and the same
reason the LTX camera-first guidance was rejected there -- three engines now, one
mechanism. Keep the A/B anyway, per the gate.

effective_cap: the diffusers hard limit is **512 tokens**, and BFL's bands are
10-30 / 30-80 / 80+ words with 30-80 called ideal. A 188-char beat is ~27-30 words,
sitting at the FLOOR of the ideal band. **So our cap here is pipeline POLICY, not a
model limit** -- the clearest case in the set of "the budgets are ours" -- and the
24-word target from `cap/7` is safe.

THE GATE, endorsed unchanged and still NOT built: "engine selectable AND directive
present, else hard refuse at selection -- no bare-writer run, no borrowed
directive." Nothing reads these constants yet, so a refusal would gate on a value no
code consults. Build it in the change that wires them.

INSTALL-DAY VERIFY, for this engine: the **klein-4B embedder identity** (read the
checkpoint config) and the **local negative surface** (expect none).
"""

#: Optional full-path override for the FLUX.2 klein diffusion GGUF. When unset,
#: the standard filename resolves through ComfyUI ``folder_paths`` and then the
#: configured models-root environment fallback.
MODEL_ENV = "OTR_FLUX2_KLEIN_CKPT"

#: Split-file companions: the Qwen-3-4B flux2 text encoder (CLIPLoader type flux2 --
#: klein-4B's matched encoder, 7680-wide; NOT Mistral) and the flux2 VAE. Default to
#: the Comfy-Org repackaged filenames in the standard model dirs. Absolute overrides
#: are registered with ``folder_paths`` before their basename reaches the loader.
CLIP_ENV = "OTR_FLUX2_KLEIN_TE"
VAE_ENV = "OTR_FLUX2_KLEIN_VAE"
_DEFAULT_CKPT = "flux-2-klein-4b-Q4_K_M.gguf"
_DEFAULT_CLIP = "qwen_3_4b.safetensors"
_DEFAULT_VAE = "flux2-vae.safetensors"


def _register_loader_parent(category: str, path: str) -> str:
    """Make an explicit file visible to the basename-only ComfyUI loader.

    GGUF and stock loaders accept a model name, not an arbitrary path. Merely
    checking an absolute override and then stripping it to a basename creates a
    false-green preflight. Register its parent at highest priority and return
    the exact path the loader now resolves.
    """
    path = os.path.abspath(path)
    try:
        import folder_paths
        folder_paths.add_model_folder_path(
            category, os.path.dirname(path), is_default=True)
        found = folder_paths.get_full_path(category, os.path.basename(path))
        if found and os.path.realpath(found) == os.path.realpath(path):
            return os.path.abspath(found)
    except Exception:
        # ``assert_usable`` will still fail if the path itself is absent. A real
        # ComfyUI process always supplies folder_paths; keeping import lazy makes
        # cold module discovery side-effect free.
        pass
    return path


def _loader_name(env_name: str, category: str, default: str) -> str:
    selected = os.environ.get(env_name, "").strip() or default
    expanded = os.path.abspath(os.path.expanduser(selected))
    if os.path.isabs(os.path.expanduser(selected)) and os.path.isfile(expanded):
        _register_loader_parent(category, expanded)
    return os.path.basename(selected)


def _resolve_unet_path() -> str:
    """Resolve the selected GGUF through the same lazy path used by its loader."""
    explicit = os.environ.get(MODEL_ENV, "").strip()
    if explicit:
        return _register_loader_parent(
            "unet", os.path.abspath(os.path.expanduser(explicit)))

    try:
        import folder_paths
        found = folder_paths.get_full_path("unet", _DEFAULT_CKPT)
        if found:
            return os.path.abspath(found)
    except Exception:
        pass

    for env_name in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        root = os.environ.get(env_name, "").strip()
        if root:
            candidate = os.path.abspath(os.path.join(
                os.path.expanduser(root), "diffusion_models", _DEFAULT_CKPT))
            return _register_loader_parent("unet", candidate)
    return _DEFAULT_CKPT


def _resolve_unet_name() -> str:
    return os.path.basename(_resolve_unet_path())


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
    requires_flag = None             # vestigial (registry IS the menu; no flag gate)
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
            "unet_name": _resolve_unet_name(),
            "clip_name": _loader_name(
                CLIP_ENV, "text_encoders", _DEFAULT_CLIP),
            "vae_name": _loader_name(VAE_ENV, "vae", _DEFAULT_VAE),
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
        ABSENT/greyed, never a stub. The registry is the menu; this is the disk
        gate. The TE + VAE loaders fail LOUD at render if their files are
        absent. The verify-on-5080 must also confirm SageAttention is not
        patched (BUG-070, FLUX-style attn)."""
        ckpt = _resolve_unet_path()
        if not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"flux2_klein diffusion GGUF not found at {ckpt!r}; install "
                f"{_DEFAULT_CKPT} under the configured diffusion_models folder "
                f"or set {MODEL_ENV} to its full path (and {CLIP_ENV}"
                f"/{VAE_ENV} for the Qwen-3-4B flux2 TE + flux2 VAE; "
                f"confirm SageAttention is not patched -- BUG-070)",
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
            # free_after_use (the video-engine pattern, eng_wan_ti2v / eng_ltx_8gb):
            # the text encoder is dropped the moment its only consumer has run, so
            # the sampler starts with the encoder OFF the card. Without it an 8 GB
            # card kept the 7.7 GB Qwen3-4B encoder resident and loaded the DiT with
            # "0.00 MB usable" (~2 min per step; 4060 clean room, 2026-09-02). The
            # MODEL node stays in ``keep`` so the patcher the sampler holds is never
            # dropped under it; the terminal is kept by run_graph itself.
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL,
                                   free_after_use=True, keep={"unet"})[0]
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
