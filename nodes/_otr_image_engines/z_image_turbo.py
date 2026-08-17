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
# Defaults = the OFFICIAL Comfy-Org/z_image_turbo split files. For LOW VRAM point
# OTR_ZIMAGE_UNET at z_image_turbo_nvfp4.safetensors (Blackwell-native fp4) and
# OTR_ZIMAGE_CLIP at qwen_3_4b_fp8_mixed.safetensors (fp8 TE); both are in the
# same repo. The TE is offloaded before the diffusion peak either way.
_DEFAULT_UNET = "z_image_turbo_bf16.safetensors"
_DEFAULT_CLIP = "qwen_3_4b.safetensors"
_DEFAULT_VAE = "ae.safetensors"

# GPU-VERIFIED 2026-06-18 on the RTX 5080 (headless :8000, nvfp4 + qwen3 fp8 TE):
# the CLIPLoader type for Z-Image's Qwen3-4B encoder is "qwen_image" (the live
# /object_info enum has no "z_image"; "qwen_image" renders a clean on-prompt
# image), the latent node is EmptySD3LatentImage, ModelSamplingAuraFlow shift 3.0
# works, and the per-clip peak was ~10 GB (well under the 14.5 GB ceiling). All
# still env-overridable.
_DEFAULT_CLIP_TYPE = "qwen_image"       # CLIPLoader type for the Qwen3-4B TE (verified)
_DEFAULT_LATENT_NODE = "EmptySD3LatentImage"   # 16-ch (matches the Flux ae VAE; verified)

#: ANTI-ARTIFACT ONLY -- the hygiene floor, with NO style opinion in it.
#:
#: PBUG-20260817-01: this constant used to also carry "clean digital, cartoon,
#: illustration", which made it a STYLE authority living engine-side. On a
#: `cartoon` episode every still was minted with the positive "bright cartoon
#: illustration" AND the negative "cartoon, illustration" -- the engine vetoed
#: the style the episode had selected. By literal phrase match it fought FOUR
#: of the nine packs (anime, cartoon, sci_fi_radio, storybook_engraving).
#: The style half now lives in each pack's `negative_tail`, which the
#: dispatcher composes into the request; `sci_fi_radio` carries the historical
#: string verbatim so the default lane is unchanged.
#:
#: This remains the FALLBACK for a mint that arrives with no composed negative
#: at all -- a legacy frozen pack, or a direct/standalone engine call. Falling
#: back to nothing would strip legacy ledgers of their anti-artifact terms.
_HYGIENE_NEGATIVE = (
    "oversaturated, glossy, plastic skin, waxy skin, "
    "sterile studio lighting, text, watermark"
)


def _resolve_negative(request_negative):
    """The negative conditioning for THIS mint, and the one place that decides.

    Order: an explicit ``OTR_ZIMAGE_NEGATIVE`` override wins (dev convenience,
    deliberately OUT of the production chain -- the B6 lesson from
    ``eng_ltx_8gb`` is that a boot-time env channel makes two boxes render
    differently while stamping identical receipts). Otherwise the composed
    request negative -- pack style + any per-object negative, already merged
    and de-duplicated by the dispatcher. Empty falls back to hygiene, never to
    nothing.
    """
    override = os.environ.get("OTR_ZIMAGE_NEGATIVE")
    if override is not None:
        return override
    return str(request_negative or "").strip() or _HYGIENE_NEGATIVE


def _installed_unets():
    """The installed ``z_image_turbo*.safetensors`` basenames in the
    ``diffusion_models`` folder, RANKED nvfp4 > fp8 > bf16 > other and sorted
    deterministically WITHIN each tier. ``[]`` when none are installed (or
    ``folder_paths`` is unavailable). Blackwell-native nvfp4 is preferred (this
    box + every boot script uses it); the LOUD log at the call site names the
    pick, so a non-Blackwell mirror that only has nvfp4 is self-diagnosing."""
    try:
        import folder_paths  # ComfyUI runtime; stubbed in tests (returns [])
        names = folder_paths.get_filename_list("diffusion_models") or []
    except Exception:  # noqa: BLE001 -- absent folder_paths -> nothing discoverable
        return []
    cands = [os.path.basename(n) for n in names
             if os.path.basename(n).lower().startswith("z_image_turbo")
             and str(n).lower().endswith(".safetensors")]

    def _rank(n):
        low = n.lower()
        tier = (0 if "nvfp4" in low else 1 if "fp8" in low
                else 2 if "bf16" in low else 3)
        return (tier, low)                       # deterministic within a tier

    return sorted(set(cands), key=_rank)


def _resolve_unet_name():
    """Resolve the Z-Image UNET basename + whether it is VERIFIED installed.

    ONE truth shared by ``_zimage_params`` (needs a name to hand the loader) and
    ``assert_usable`` (needs a real installed/not-installed answer), so the two
    can never diverge again (the 2026-07-05 landmine: assert_usable required the
    env while render fell back to an absent bf16 default). Order:
      1. ``OTR_ZIMAGE_UNET`` override -> validated against folder_paths /
         os.path.isfile; verified iff the file actually resolves.
      2. else the best-ranked INSTALLED ``z_image_turbo*.safetensors`` (verified).
      3. else ``_DEFAULT_UNET`` (UNVERIFIED -> assert_usable greys the engine, and
         the loader still raises a CLEAR error if it is ever reached).
    Returns ``(basename, verified: bool)``."""
    env = os.environ.get(MODEL_ENV, "").strip()
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


@register
class ZImageTurboEngine:
    """The Z-Image-Turbo image adapter (reduced ``prompt -> image`` protocol)."""

    name = "z_image_turbo"
    roles = ROLES
    default_roles = ()               # opt-in peer (Flux remains the default)
    commercial_clean = True          # Apache-2.0
    requires_flag = None             # vestigial (registry IS the menu; no flag gate)
    required_inputs = ("text_prompt",)
    engine_version = "1"
    #: This adapter can condition a mint on the character's canonical portrait,
    #: so the dispatcher may hand it a reference_image. Lumina2/Z-Image carries
    #: the ref_latents path (model_base.Lumina2.extra_conds), and the installed
    #: checkpoint's header shows cap_pad_token and x_pad_token present with no
    #: siglip module, so the reference embeds through the shared x_embedder with
    #: no missing weights. Whether the TURBO checkpoint was trained to attend to
    #: it is a separate question that only the live A/B answers.
    accepts_reference_image = True

    #: Resolved SEPARATELY from _node_candidates and merged per call. These must
    #: never join the main candidate map: render_image caches the resolved map on
    #: the registry SINGLETON, and the episode's first mint is an unreferenced
    #: portrait -- so a params-gated map would be cached WITHOUT these keys and
    #: every later referenced mint would die with 'class unresolved'.
    #: ImageScale, never FluxKontextImageScale: resolve_graph_classes binds the
    #: first installed name, and FluxKontextImageScale.execute takes `image`
    #: ONLY, so it would receive upscale_method/width/height/crop and TypeError.
    _REF_CANDIDATES = {
        "load_ref": ("LoadImage",),
        "scale_ref": ("ImageScale",),
        "encode_ref": ("VAEEncode",),
        "ref_pos": ("ReferenceLatent",),
        "ref_neg": ("ReferenceLatent",),
    }

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

        _unet_name, _unet_ok = _resolve_unet_name()
        log.info("[OTR.image.z_image_turbo] unet resolved -> %s (verified=%s; "
                 "installed=%s)", _unet_name, _unet_ok, _installed_unets())
        return {
            "unet_name": _unet_name,
            "clip_name": os.path.basename(os.environ.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(os.environ.get(VAE_ENV, "") or _DEFAULT_VAE),
            "clip_type": os.environ.get("OTR_ZIMAGE_CLIP_TYPE", _DEFAULT_CLIP_TYPE),
            "latent_node": os.environ.get("OTR_ZIMAGE_LATENT_NODE", _DEFAULT_LATENT_NODE),
            "unet_dtype": os.environ.get("OTR_ZIMAGE_UNET_DTYPE", "default"),
            "prompt": str(get("prompt") or ""),
            "negative": _resolve_negative(get("negative_prompt")),
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
            "reference_image": str(get("reference_image") or ""),
            # Capped well under the portrait's native 1216 so the reference
            # costs roughly 1.5k latent tokens instead of 4k on an 8-step mint.
            "reference_height": _eint("OTR_PORTRAIT_REF_HEIGHT", 768),
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
        ref = str(params.get("reference_image") or "")
        graph = {
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
        if not ref:
            return graph
        # ImageScale.upscale takes all FIVE arguments -- image, upscale_method,
        # width, height, crop -- and run_graph calls fn(**kwargs) straight off
        # this inputs dict with no fallback, so a short-form node is a dead
        # episode, not a warning. width=0 lets ImageScale derive the missing
        # side from the actual image, keeping the 832x1216 portrait's aspect so
        # no face is cropped into a 16:9 band; VAE.encode crops to a multiple of
        # 8 internally, so a derived odd width is safe.
        graph["load_ref"] = {"class": "load_ref", "inputs": {"image": ref}}
        graph["scale_ref"] = {
            "class": "scale_ref",
            "inputs": {"image": W("load_ref", 0), "upscale_method": "lanczos",
                       "width": 0, "height": int(params["reference_height"]),
                       "crop": "disabled"}}
        graph["encode_ref"] = {
            "class": "encode_ref",
            "inputs": {"pixels": W("scale_ref", 0), "vae": W("vae", 0)}}
        # BOTH conditionings. z_image runs cfg=2.0 with a live negative, and the
        # model doubles its timesteps only on the omni path -- referencing the
        # positive alone would take the CFG delta between two structurally
        # different forward passes.
        graph["ref_pos"] = {
            "class": "ref_pos",
            "inputs": {"conditioning": W("pos", 0), "latent": W("encode_ref", 0)}}
        graph["ref_neg"] = {
            "class": "ref_neg",
            "inputs": {"conditioning": W("neg", 0), "latent": W("encode_ref", 0)}}
        # The rewire is the point. A graph that builds the chain and forgets to
        # consume it passes a node-count check and renders nothing different.
        graph["ksampler"]["inputs"]["positive"] = W("ref_pos", 0)
        graph["ksampler"]["inputs"]["negative"] = W("ref_neg", 0)
        return graph

    # ---- residency (classes resolve lazily; loader nodes own the weights) ----
    def load(self):  # pragma: no cover - resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until a Z-Image diffusion model is INSTALLED (BUG-046):
        ABSENT/greyed, never a stub. Shares ``_resolve_unet_name()`` with
        ``_zimage_params`` so the usability gate and the render path can NEVER
        disagree (the 2026-07-05 landmine: the gate required ``OTR_ZIMAGE_UNET``
        while render fell back to an absent bf16 default -> deep FileNotFoundError
        instead of an early grey-out). Usable iff the resolver VERIFIES an
        installed model (env override that resolves, OR an auto-discovered
        ``z_image_turbo*.safetensors``). The CLIP + VAE loaders still fail LOUD at
        render if their files are absent (their defaults are present on this box)."""
        _name, verified = _resolve_unet_name()
        if not verified:
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"z_image_turbo diffusion model not found (resolved {_name!r}, not "
                f"installed): install a z_image_turbo*.safetensors in "
                f"diffusion_models (nvfp4 for Blackwell) or point {MODEL_ENV} at "
                f"one (+ {CLIP_ENV} Qwen3-4B TE / {VAE_ENV} Flux ae)",
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
        if params.get("reference_image"):
            # Stage the PNG into ComfyUI's input dir OUTSIDE the graph builder,
            # which stays pure -- LoadImage resolves the returned basename.
            params = dict(
                params,
                reference_image=_wb.stage_into_comfy_input(
                    params["reference_image"]),
            )
            ref_classes = getattr(self, "_ref_classes", None) \
                or _wb.resolve_graph_classes(self._REF_CANDIDATES)
            self._ref_classes = ref_classes
            # Merged into a LOCAL map for this call only, so the cached
            # self._classes never gains reference keys and an unreferenced mint
            # keeps resolving exactly what it always did.
            classes = {**classes, **ref_classes}
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
