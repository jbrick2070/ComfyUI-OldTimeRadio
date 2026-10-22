"""FLUX 3 Image Klein image adapter -- an OPT-IN local still engine (in-process).

Black Forest Labs' 5.6B open-weights still model (released 2026-10-15). It runs
on stock ComfyUI CORE (>= v0.41.0: ``CLIPLoader`` type ``flux3``,
``EmptyFlux2LatentImage``), so it is an ordinary ``wrapper_bridge`` split-file
graph exactly like ``z_image_turbo``: UNETLoader -> ModelSamplingFlux ->
KSampler -> VAEDecode, with the Qwen3-VL-4B text encoder and the Flux 2 VAE.

LICENCE (apple/FLUX3_KLEIN_LICENSE_ATTESTATION.md): FLUX Kommunity License
v1.0. Non-commercial use is free; commercial use of the OUTPUTS is allowed only
for a Qualifying User (< US$5M gross annual revenue with affiliates) who runs a
content filter or output review. So ``commercial_clean = False`` -- the same
shelf as ``flux_gen1``. The pack ships no weights: they are fetched on first use
at an exact pinned revision from an UNGATED third-party mirror
(``ml-resources/flux-3-image-klein-comfy``) whose file hashes are byte-identical
to Black Forest Labs' gated repository. See ``_PINNED_SOURCES`` in
``nodes/_otr_visual_assets.py``.

THE RECIPE is BFL's own distilled text-to-image workflow
(``flux3_klein_int8_t2i.json``): 16 steps, cfg 1.0 (no guidance; the negative is
inert), euler / simple, ``ModelSamplingFlux`` with max_shift = base_shift =
ln(alpha), alpha 16 (the model's reference is 6.93). The prompt upsampler BFL
ships (a 9.7 GB Qwen3.5-9B) is NOT used: the pack composes its own full scene
caption, and that choice is unmeasured -- override nothing, A/B it.

REFERENCE IMAGES ARE OPT-IN (``OTR_FLUX3_REFERENCE=1``; default off). With it
on, the adapter advertises ``accepts_reference_image`` and the dispatcher hands
every character still the character's canonical portrait, which conditions the
mint through core's ``ReferenceLatent`` (LoadImage -> ImageScale -> VAEEncode ->
ReferenceLatent on both conditionings; the negative is inert at cfg 1.0 but is
referenced too so a raised cfg stays coherent). BFL says references "anchor
content, they do not lock structure"; whether that holds a face between beats is
being measured on the stills workflow, and ``z_image_turbo`` shows what an
unmeasured generic reference path can do (a grid artefact, 2026-08-20). The
default flips only on that measurement.

Cold-import clean: only the dep-free registry, role vocabulary and stdlib at
module scope. torch / comfy are lazy inside ``render_image`` (via wrapper_bridge).
"""
from __future__ import annotations

import logging
import math
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.image.flux3_klein")

MODEL_ENV = "OTR_FLUX3_UNET"
CLIP_ENV = "OTR_FLUX3_CLIP"
VAE_ENV = "OTR_FLUX3_VAE"

#: The int8 ConvRot build: the smallest universal precision (5.8 GB), runs on any
#: NVIDIA card and Apple Silicon. nvfp4 is Blackwell-only and bf16 is 11 GB; both
#: are picked up if installed (see ``_installed_unets``) but are never fetched.
_DEFAULT_UNET = "flux3-image-klein-int8-convrot.safetensors"
_DEFAULT_CLIP = "qwen3vl_4b_int8_convrot.safetensors"
_DEFAULT_VAE = "flux2-vae.safetensors"
_CLIP_TYPE = "flux3"

#: Stock ComfyUI node classes the graph needs. Both predate v0.41.0 (Ideogram 4
#: uses EmptyFlux2LatentImage), so they are necessary but NOT the version test:
#: that is the ``flux3`` entry in CLIPLoader's type list, added to core by
#: commit 20b46065 "Support Flux3-image (#17064)", first tagged v0.41.0.
_REQUIRED_CORE_NODES = ("EmptyFlux2LatentImage", "ModelSamplingFlux", "CLIPLoader")
_CORE_UPDATE_MESSAGE = (
    "Update ComfyUI to v0.41.0 or newer: flux3_klein needs the built-in "
    "CLIPLoader type 'flux3', which this version does not have")

_HYGIENE_NEGATIVE = ""  # the distilled model runs at cfg 1.0: the negative is inert

_CANVAS_MULTIPLE = 16
_CANVAS_FLOOR = 256


def _snap(value: int) -> int:
    """Nearest multiple of 16, floored at 256 (the Flux 2 latent is 16x down)."""
    v = max(_CANVAS_FLOOR, int(value))
    return max(_CANVAS_FLOOR, int(round(v / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)


#: At most this many reference pictures a still: the speaker's portrait and two
#: more. Measured on an RTX 5080 Laptop 16 GB by the 3-act My Story demo of
#: 2026-10-20 (both leads from reference pictures, three a still at most); a
#: fourth is unmeasured, so it is not sent (Codex r1 on flux3_klein_ref).
MAX_REFERENCES = 3


def _reference_enabled() -> bool:
    """True when the operator switched the portrait-reference path on."""
    return str(otr_env.get("OTR_FLUX3_REFERENCE", "0")).strip() == "1"


def _stage_reference(wb, src):
    """Copy a reference into ComfyUI's input dir under a content-addressed name,
    so two characters never overwrite each other's file there."""
    import hashlib
    try:
        from .._otr_paths import confine_to_read_roots
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_paths import confine_to_read_roots  # type: ignore
    # Proven inside the approved folders, resolved, before a byte is read
    # (registry policy review, 2026-10-22); every caller already confines, and
    # this is the one seat that opens the file.
    src = confine_to_read_roots(src, "reference image")
    with open(src, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return wb.stage_into_comfy_input(
        src, "otr_ref_%s%s" % (digest, os.path.splitext(src)[1] or ".png"))


def _installed_unets():
    """Installed ``flux3-image-klein*`` diffusion models, ranked int8 > bf16 >
    nvfp4 > other. The BASE (undistilled, 50-step) checkpoints are excluded: this
    recipe is the distilled one. ``[]`` when none (or ``folder_paths`` absent)."""
    try:
        import folder_paths  # ComfyUI runtime; stubbed in tests
        names = folder_paths.get_filename_list("diffusion_models") or []
    except Exception:  # noqa: BLE001
        return []
    cands = [os.path.basename(n) for n in names
             if os.path.basename(n).lower().startswith("flux3-image-klein")
             and "base" not in os.path.basename(n).lower()
             and str(n).lower().endswith(".safetensors")]

    def _rank(n):
        low = n.lower()
        return (0 if "int8" in low else 1 if "bf16" in low
                else 2 if "nvfp4" in low else 3, low)

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
    """The reason this ComfyUI core cannot run Klein, or ``None``.

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
    if isinstance(types, (list, tuple)) and "flux3" not in types:
        return _CORE_UPDATE_MESSAGE
    return None


@register
class Flux3KleinEngine:
    """The FLUX 3 Image Klein adapter (reduced ``prompt -> image`` protocol)."""

    name = "flux3_klein"
    roles = ROLES
    default_roles = ()               # opt-in; never automatic
    commercial_clean = False         # FLUX Kommunity License: Qualifying Users only
    requires_flag = None
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def _references_on(self) -> bool:
        """Whether this engine paints from portrait references: the operator's
        switch, read per call so it needs no re-registration."""
        return _reference_enabled()

    @property
    def accepts_reference_image(self) -> bool:
        return self._references_on()

    #: Resolved SEPARATELY and merged only when a reference is present, so the
    #: base graph and its cached class map stay exactly the nine-node t2i graph.
    #: ImageScale, never FluxKontextImageScale: resolve_graph_classes binds the
    #: first installed name, and FluxKontextImageScale.execute takes ``image`` only.
    _REF_CANDIDATES = {
        "load_ref": ("LoadImage",),
        "scale_ref": ("ImageScale",),
        "encode_ref": ("VAEEncode",),
        "ref_pos": ("ReferenceLatent",),
        "ref_neg": ("ReferenceLatent",),
    }

    _TERMINAL = "decode"

    def _params(self, request):
        """Pure: sampler params from the request + env. Seed, prompt and canvas
        come from the request (deterministic re-gen); the rest is BFL's recipe."""
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
        alpha = max(1.0001, _efloat("OTR_FLUX3_ALPHA", 16.0))
        width = _snap(get("width") or get("w") or _eint("OTR_FLUX3_WIDTH", 1024))
        height = _snap(get("height") or get("h") or _eint("OTR_FLUX3_HEIGHT", 1024))
        return {
            "unet_name": unet,
            "clip_name": os.path.basename(otr_env.get(CLIP_ENV, "") or _DEFAULT_CLIP),
            "vae_name": os.path.basename(otr_env.get(VAE_ENV, "") or _DEFAULT_VAE),
            "clip_type": _CLIP_TYPE,
            "unet_dtype": otr_env.get("OTR_FLUX3_UNET_DTYPE", "default"),
            "prompt": str(get("prompt") or ""),
            "negative": str(get("negative_prompt") or _HYGIENE_NEGATIVE),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_FLUX3_STEPS", 16),
            "cfg": _efloat("OTR_FLUX3_CFG", 1.0),
            "shift": math.log(alpha),          # max_shift == base_shift == ln(alpha)
            "sampler_name": otr_env.get("OTR_FLUX3_SAMPLER", "euler"),
            "scheduler": otr_env.get("OTR_FLUX3_SCHEDULER", "simple"),
            "width": width,
            "height": height,
            # Admission is enforced HERE too, not only by the dispatcher's
            # capability read: with the switch off, a request that carries a
            # reference still renders the plain t2i graph.
            "reference_image": (str(get("reference_image") or "")
                                if self._references_on() else ""),
            # The other characters in the frame (the dispatcher's co-references),
            # each one more ReferenceLatent after the speaker's. Read only with
            # a primary reference: they are its companions, never a substitute.
            "reference_images": ([str(r) for r in (get("reference_images") or []) if r]
                                 [:MAX_REFERENCES - 1]
                                 if self._references_on() and get("reference_image")
                                 else []),
            # Capped under the portrait's native height so the reference adds
            # about 1.5k latent tokens, not 4k.
            "reference_height": _eint("OTR_FLUX3_REFERENCE_HEIGHT", 768),
        }

    def _node_candidates(self, params=None):  # noqa: ARG002
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "sampling": ("ModelSamplingFlux",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptyFlux2LatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_graph(self, params, wire):
        """Pure: BFL's distilled t2i graph in ``wrapper_bridge.run_graph`` form,
        plus the reference chain when ``params["reference_image"]`` names a
        staged input file."""
        W = wire
        graph = self._build_base_graph(params, W)
        ref = str(params.get("reference_image") or "")
        if not ref:
            return graph
        refs = [ref] + [str(r) for r in params.get("reference_images") or []]
        pos, neg = W("pos", 0), W("neg", 0)
        for i, image in enumerate(refs):
            # The speaker's chain keeps its original node ids; each further
            # reference gets a numbered copy of the same five nodes, chained so
            # the conditioning carries every latent (core appends them).
            sfx = "" if i == 0 else "_%d" % i
            # ImageScale takes all five arguments and run_graph calls it straight
            # off this dict. width=0 derives the width from the image, keeping the
            # portrait's aspect so no face is cropped.
            graph["load_ref" + sfx] = {"class": "load_ref", "inputs": {"image": image}}
            graph["scale_ref" + sfx] = {
                "class": "scale_ref",
                "inputs": {"image": W("load_ref" + sfx, 0), "upscale_method": "lanczos",
                           "width": 0, "height": int(params["reference_height"]),
                           "crop": "disabled"}}
            graph["encode_ref" + sfx] = {
                "class": "encode_ref",
                "inputs": {"pixels": W("scale_ref" + sfx, 0), "vae": W("vae", 0)}}
            graph["ref_pos" + sfx] = {
                "class": "ref_pos",
                "inputs": {"conditioning": pos, "latent": W("encode_ref" + sfx, 0)}}
            graph["ref_neg" + sfx] = {
                "class": "ref_neg",
                "inputs": {"conditioning": neg, "latent": W("encode_ref" + sfx, 0)}}
            pos, neg = W("ref_pos" + sfx, 0), W("ref_neg" + sfx, 0)
        graph["ksampler"]["inputs"]["positive"] = pos
        graph["ksampler"]["inputs"]["negative"] = neg
        return graph

    def _build_base_graph(self, params, wire):
        W = wire
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": params["unet_name"],
                                "weight_dtype": params["unet_dtype"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": params["clip_type"]}},
            "vae": {"class": "vae", "inputs": {"vae_name": params["vae_name"]}},
            "sampling": {"class": "sampling",
                         "inputs": {"model": W("unet", 0),
                                    "max_shift": float(params["shift"]),
                                    "base_shift": float(params["shift"]),
                                    "width": int(params["width"]),
                                    "height": int(params["height"])}},
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

    def load(self):  # pragma: no cover -- resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def core_problem(self, mapping=None):
        """Read by the queue-time gate (``_refuse_missing_node_packs``) so an
        old core is refused BEFORE the 10 GB fetch, not at the dispatcher."""
        return _core_problem(mapping)

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        """Fail closed until the diffusion model is INSTALLED and the running
        ComfyUI core can load Klein (>= v0.41.0)."""
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
                f"flux3_klein diffusion model not found (resolved {_name!r}, not "
                f"installed): it is fetched on first use, or point {MODEL_ENV} at "
                f"a flux3-image-klein*.safetensors (+ {CLIP_ENV} Qwen3-VL-4B "
                f"encoder / {VAE_ENV} flux2-vae)",
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
        if params["reference_image"]:
            params = dict(
                params, reference_image=_stage_reference(_wb, params["reference_image"]),
                reference_images=[_stage_reference(_wb, r)
                                  for r in params["reference_images"]])
            classes = dict(classes, **_wb.resolve_graph_classes(self._REF_CANDIDATES))
        graph = self._build_graph(params, _wb.Wire)
        try:
            # Drop the Qwen3-VL encoder the moment its consumers have run so the
            # sampler starts with it off the card (the z_image_turbo 8 GB lesson).
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL,
                                   free_after_use=True, keep={"unet"},
                                   evict_after_use={"clip"})[0]
            frames = _wb.images_to_uint8(images)
        finally:
            _wb.reclaim_idle_models(reason="flux3_klein post-decode")
        log.info("[OTR.image.flux3_klein] minted still %dx%d seed=%d steps=%d "
                 "cfg=%.2f shift=%.4f reference=%s also=%s", params["width"],
                 params["height"], params["seed"], params["steps"], params["cfg"],
                 params["shift"], params["reference_image"] or "none",
                 ",".join(params["reference_images"]) or "none")
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


@register
class Flux3KleinReferenceEngine(Flux3KleinEngine):
    """FLUX 3 Klein with portrait references ALWAYS on -- a stills choice, not
    an environment switch (operator, 2026-10-20: "use flux 3 ... we want ref
    image for my story as an option for this contest").

    The same model files, recipe and graph as flux3_klein; the one difference
    is that every character still is painted from that character's portrait
    (and the photos a person gave), so faces hold from beat to beat. Picking it
    is the option; plain flux3_klein keeps its own behaviour on every graph
    that ships it (the 8 GB and 24 GB lanes), so nothing changes for them.
    """

    name = "flux3_klein_ref"
    #: Chosen on purpose from a dropdown, never by a random roll: references
    #: only mean something for a story whose characters have pictures.
    rollable = False

    def _references_on(self) -> bool:
        return True


__all__ = ["Flux3KleinEngine", "Flux3KleinReferenceEngine", "MODEL_ENV", "CLIP_ENV",
           "VAE_ENV"]
