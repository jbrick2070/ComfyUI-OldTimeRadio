r"""``sd15`` -- Stable Diffusion 1.5, the SMALL local image engine.

WHY THIS EXISTS: it is the only local image engine that fits a 16 GB Apple
Silicon box. Measured 2026-09-07 on a Mac mini M4, every Z-Image variant failed:

  z_image_turbo_bf16          12.31 GB  -> MPS OOM (needed ~20.4 GiB of a
                                          20.13 GiB ceiling; both models DID
                                          load, it died in the KSampler)
  z_image_turbo_int8_convrot   5.78 GB  -> NotImplementedError: the operator
                                          'aten::_int_mm' is not implemented
                                          for MPS. int8 matmul does not exist
                                          on Metal.
  z_image_turbo_nvfp4          4.51 GB  -> Blackwell-native fp4, NVIDIA only.

SD 1.5 fp16 is 1.99 GB from the UNGATED ``Comfy-Org/stable-diffusion-v1-5-archive``,
loads through the ordinary ``CheckpointLoaderSimple`` (MODEL + CLIP + VAE in one
file -- no split loaders, no GGUF pack, no text-encoder download), and needs no
licence acceptance. That combination is what makes it installable on the
press-Run path this pack is built around.

RESOLUTION, AND IT MATTERS: SD 1.5 is 512x512 NATIVE. Past roughly 768 on the
long side it duplicates subjects -- two heads, mirrored torsos -- which is
CONFIDENTLY WRONG OUTPUT rather than an error, so nothing downstream reports it.

The engine therefore FITS every request down to ``OTR_SD15_MAX_SIDE`` (default
768), preserving aspect and snapping to a multiple of 8, and lets the still lane
frame what it is given. The canonical's 832x480 mints as 768x440.

It does NOT simply "default to 512": an earlier draft of this file said so, and
that was dead on every real request, because the composer always stamps w/h and
request dims win. ``OTR_SD15_WIDTH`` / ``OTR_SD15_HEIGHT`` are only the
no-request default; ``OTR_SD15_MAX_SIDE`` is the knob that actually governs
production output. Resolution risk raised by the 5080 window and the dead-default
bug caught by the cursor review lane, both 2026-09-07, before the first mint.

Cold-import clean (invariant V-12): this module imports only the registry
vocabulary + stdlib. torch / comfy / the model are NEVER imported here; the
heavy path is lazy, inside ``render_image`` via wrapper_bridge, mirroring
z_image_turbo / flux / lumina.
"""
from __future__ import annotations

import logging

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.image.sd15")

#: VESTIGIAL (no runtime reader; the registry IS the menu). Kept for pattern
#: parity with the sibling engines and the no-gate regression test.
ENABLE_FLAG = "OTR_ENABLE_SD15"

#: One ordinary checkpoint: MODEL + CLIP + VAE together. Env-overridable so a
#: user can point this at any SD-1.5-architecture checkpoint they already have
#: (see docs/ADDING_IMAGE_AND_VIDEO_LANES.md, level 1) without touching code.
CKPT_ENV = "OTR_SD15_CKPT"
_DEFAULT_CKPT = "v1-5-pruned-emaonly-fp16.safetensors"

#: 512 NATIVE. See the resolution note in the module docstring before changing.
_DEFAULT_W, _DEFAULT_H = 512, 512
_DEFAULT_STEPS, _DEFAULT_CFG = 20, 7.0
_DEFAULT_SAMPLER, _DEFAULT_SCHEDULER = "dpmpp_2m", "karras"

#: ANTI-ARTIFACT ONLY -- the hygiene floor, with NO style opinion in it.
#: PBUG-20260817-01 is why: a style opinion living engine-side vetoed the
#: episode's own chosen style. Style belongs to the composer, never here.
_NEGATIVE_FLOOR = ("lowres, bad anatomy, extra limbs, extra heads, duplicate, "
                   "watermark, text, signature, jpeg artifacts, blurry")


def _role_of(profile) -> str:
    """The role the dispatcher is asking about, or "". Same shape as the peers."""
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


#: SD 1.5's safe upper bound. Past roughly this, on EITHER axis, SD 1.5 starts
#: duplicating subjects -- two heads, mirrored torsos. Override deliberately.
MAX_SIDE_ENV = "OTR_SD15_MAX_SIDE"
_DEFAULT_MAX_SIDE = 768


def _fit_native(width: int, height: int, max_side: int):
    """Scale (w,h) down so the LONG side is at most ``max_side``, preserving
    aspect and snapping to a multiple of 8.

    THIS IS THE WHOLE POINT OF THE ENGINE'S SIZE HANDLING, and an earlier
    version got it wrong in a way that only showed at render. That version set a
    512 default and let request dims win -- which reads as "mints at native size
    by default", except the composer ALWAYS stamps w/h
    (otr_meta_brief_image_prompt.py), so the default was dead on every real
    request and the canonical's ~832-wide canvas went straight through. SD 1.5
    at 832 wide duplicates subjects, and that is confidently-wrong output with no
    error anywhere: the gate passes, the render succeeds, the still is nonsense.
    Caught by the cursor review lane before the first mint, 2026-09-07, after
    the 5080 window had flagged the resolution risk in the abstract.

    Downscaling rather than refusing is deliberate: the still lane frames what
    it is given, so a correctly-composed 768x440 is strictly better than either
    a two-headed 832x480 or a hard failure mid-episode.
    """
    w, h = max(8, int(width)), max(8, int(height))
    longest = max(w, h)
    if longest <= max_side:
        return (w - w % 8) or 8, (h - h % 8) or 8
    scale = max_side / float(longest)
    w2 = max(8, int(w * scale)) // 8 * 8
    h2 = max(8, int(h * scale)) // 8 * 8
    return max(8, w2), max(8, h2)


def _resolve_ckpt_name() -> str:
    """The checkpoint filename this engine will load. ONE resolver, shared by
    ``assert_usable`` and ``_sd15_params`` so the usability gate and the render
    path can never disagree -- the 2026-07-05 landmine that cost z_image_turbo a
    deep FileNotFoundError instead of an early grey-out."""
    return (otr_env.get(CKPT_ENV) or "").strip() or _DEFAULT_CKPT


def _installed(name: str) -> bool:
    """True iff ComfyUI can actually see ``name`` as a checkpoint. Asks
    folder_paths, never the filesystem, so extra_model_paths.yaml is honoured."""
    try:
        import folder_paths  # noqa: PLC0415 -- ComfyUI runtime only
        return name in set(folder_paths.get_filename_list("checkpoints"))
    except Exception:  # noqa: BLE001 -- absent outside ComfyUI (tests)
        return False


@register
class SD15Engine:
    """Stable Diffusion 1.5 (reduced ``prompt -> image`` protocol)."""

    name = "sd15"
    roles = ROLES
    default_roles = ()               # opt-in peer; nothing is displaced
    #: CreativeML OpenRAIL-M. Commercial use IS permitted, so True -- but the
    #: licence carries USE-BASED restrictions (the Attachment A prohibitions),
    #: which "commercial_clean" does not model. Declared True honestly, with
    #: the caveat written down rather than implied by omission (Gate IG1.2).
    commercial_clean = True
    requires_flag = None             # vestigial (registry IS the menu)
    required_inputs = ("text_prompt",)
    #: DECLARED, never left to the dispatcher's getattr fallback (Gate IG1.1).
    #: Bump this whenever output should stop being reused from cache.
    engine_version = "1"
    #: SD 1.5 has no native reference-latent path in this recipe.
    accepts_reference_image = False

    #: Terminal graph node (its IMAGE output is the still).
    _TERMINAL = "decode"

    # ---- params / graph (pure; CPU-testable) ----------------------------
    def _sd15_params(self, request):
        """Pure: resolve sampler params from the request + env."""
        g = (lambda k, d: (otr_env.get(k) or "").strip() or d)

        def _i(k, d):
            try:
                return int(g(k, str(d)))
            except (TypeError, ValueError):
                return d

        def _f(k, d):
            try:
                return float(g(k, str(d)))
            except (TypeError, ValueError):
                return d

        # THE FIELD IS "prompt", NOT "text_prompt". The dispatcher supplies
        # `prompt` (otr_image_gen_dispatcher.py:1686) and z_image_turbo.py:348
        # reads exactly that. An earlier version of this method read
        # `text_prompt` -- the name from `required_inputs`, which describes the
        # CONTRACT, not the request key -- so every still would have been minted
        # from an EMPTY prompt. Silently: a valid PNG of nothing in particular,
        # with no error anywhere. Caught by the codex review lane before the
        # first render, 2026-09-07.
        #
        # One getter for both request shapes, copied from z_image_turbo rather
        # than reinvented; the hand-rolled conditional this replaces also turned
        # a dict {"text_prompt": None} into the literal string "None".
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        neg = str(get("negative_prompt") or "").strip().strip(",").strip()
        return {
            "ckpt_name": _resolve_ckpt_name(),
            "prompt": str(get("prompt") or ""),
            "negative": (neg + ", " + _NEGATIVE_FLOOR) if neg else _NEGATIVE_FLOOR,
            **dict(zip(("width", "height"), _fit_native(
                int(get("width") or get("w") or _i("OTR_SD15_WIDTH", _DEFAULT_W)),
                int(get("height") or get("h") or _i("OTR_SD15_HEIGHT", _DEFAULT_H)),
                _i(MAX_SIDE_ENV, _DEFAULT_MAX_SIDE)))),
            # SAME fallback chain as the fitted pair above. Using the bare
            # defaults here made the downscale log claim "asked 512, minting
            # 640" whenever only the env knobs were set (cursor, round 3).
            "asked_width": int(get("width") or get("w") or _i("OTR_SD15_WIDTH", _DEFAULT_W)),
            "asked_height": int(get("height") or get("h") or _i("OTR_SD15_HEIGHT", _DEFAULT_H)),
            "max_side": _i(MAX_SIDE_ENV, _DEFAULT_MAX_SIDE),
            "steps": _i("OTR_SD15_STEPS", _DEFAULT_STEPS),
            "cfg": _f("OTR_SD15_CFG", _DEFAULT_CFG),
            "sampler_name": g("OTR_SD15_SAMPLER", _DEFAULT_SAMPLER),
            "scheduler": g("OTR_SD15_SCHEDULER", _DEFAULT_SCHEDULER),
            "seed": int(get("seed") or 0),
            # Request dims are FITTED, not taken raw -- _fit_native caps the
            # long side. The env knobs below are only the no-request default;
            # raising them without also raising OTR_SD15_MAX_SIDE does nothing.
        }

    def _node_candidates(self, params=None):
        """Ordered ComfyUI node-class candidates per graph node. SD 1.5 uses the
        4-channel EmptyLatentImage, NOT the 16-ch SD3 latent z_image needs."""
        return {
            "ckpt": ("CheckpointLoaderSimple",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptyLatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_sd15_graph(self, params, wire):
        """Pure: the declarative SD 1.5 txt2img graph (wrapper_bridge.run_graph
        format). CheckpointLoaderSimple out 0=MODEL, 1=CLIP, 2=VAE -- one loader,
        unlike z_image's split unet/clip/vae."""
        W = wire
        return {
            "ckpt": {"class": "ckpt",
                     "inputs": {"ckpt_name": params["ckpt_name"]}},
            "pos": {"class": "pos",
                    "inputs": {"text": params["prompt"], "clip": W("ckpt", 1)}},
            "neg": {"class": "neg",
                    "inputs": {"text": params["negative"], "clip": W("ckpt", 1)}},
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
                                    "model": W("ckpt", 0),
                                    "positive": W("pos", 0),
                                    "negative": W("neg", 0),
                                    "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("ksampler", 0),
                                  "vae": W("ckpt", 2)}},
        }

    # ---- residency (classes resolve lazily; the loader node owns the weights) --
    def load(self):  # pragma: no cover - resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED, by NAME, when the checkpoint is not installed -- never a
        stub and never a deep FileNotFoundError at render. Shares
        ``_resolve_ckpt_name()`` with ``_sd15_params`` so the gate and the render
        path cannot disagree."""
        ckpt = _resolve_ckpt_name()
        if not _installed(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile), EngineUsabilityReason.MISSING_MODEL,
                f"sd15 needs {ckpt!r} in models/checkpoints. Fetch the ungated "
                f"Comfy-Org/stable-diffusion-v1-5-archive copy (1.99 GB): "
                f"python -c \"from huggingface_hub import hf_hub_download; "
                f"print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive',"
                f"'v1-5-pruned-emaonly-fp16.safetensors'))\" then copy it there. "
                f"Or set {CKPT_ENV} to an SD-1.5-architecture checkpoint you have.",
                kind="image")
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU
        """Mint ONE still and return it as a decoded uint8 (H,W,3) RGB array."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._sd15_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates(params))
        self._classes = classes
        graph = self._build_sd15_graph(params, _wb.Wire)
        try:
            # No free_after_use/evict_after_use split here: SD 1.5's text encoder
            # is part of the SAME checkpoint as the UNet, so dropping the "clip"
            # output would evict the model the sampler is holding. That split is
            # correct for z_image (separate 7.7 GB encoder file) and WRONG here.
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            _wb.reclaim_idle_models(reason="sd15 post-decode")
        if (params["asked_width"], params["asked_height"]) != (params["width"], params["height"]):
            # TWO different reasons live in _fit_native, and saying the wrong one
            # is its own small lie: the clamp branch avoids SD 1.5 duplicating
            # subjects, while the snap branch only rounds to a multiple of 8 and
            # has nothing to do with duplication. A 750x750 request mints at
            # 744x744 purely from the snap, and the old single message claimed
            # it had exceeded a 768 ceiling it never approached.
            if max(params["asked_width"], params["asked_height"]) > params["max_side"]:
                log.info(
                    "[OTR.image.sd15] request asked %dx%d; minting %dx%d -- SD 1.5 "
                    "duplicates subjects past %d on the long side, and the still "
                    "lane frames what it is given. Raise %s deliberately if you "
                    "want the larger canvas.",
                    params["asked_width"], params["asked_height"],
                    params["width"], params["height"],
                    params["max_side"], MAX_SIDE_ENV)
            else:
                log.info(
                    "[OTR.image.sd15] request asked %dx%d; minting %dx%d -- "
                    "rounded down to a multiple of 8 (latent grid). No clamp: "
                    "the long side was already within %d.",
                    params["asked_width"], params["asked_height"],
                    params["width"], params["height"], params["max_side"])
        log.info(
            "[OTR.image.sd15] minted still %dx%d seed=%d steps=%d cfg=%.2f "
            "sampler=%s/%s ckpt=%s", params["width"], params["height"],
            params["seed"], params["steps"], params["cfg"],
            params["sampler_name"], params["scheduler"], params["ckpt_name"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None
