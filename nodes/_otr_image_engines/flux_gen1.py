"""Flux gen-1 image adapter -- the FIRST image engine (C-now / C1).

"Flux" is just gen 1, not a hardcoded default: it registers exactly like any
future image engine (Z-Image / Klein / Qwen / "+ Add Custom Model") so swapping
it touches nothing downstream. It wraps the SHIPPED Flux still/portrait
generation (``visual/batch_flux_render.py`` + ``visual/batch_flux_portrait_render.py``)
as the platform's gen-1 ``prompt -> .png`` engine.

Cold-import clean (V-12): module scope imports only the dep-free registry +
role vocabulary + stdlib. torch / comfy / the heavy Flux pipeline are imported
LAZILY inside ``load`` / ``render_image`` / ``assert_usable`` -- importing this
module (and thus the image namespace) never pulls a model framework.

Licensing: Flux.1-dev is BFL NON-commercial, so ``commercial_clean = False``.
It is still the in-stack gen-1 default (``default_roles`` = every image role), so
``assert_usable`` returns it for any role without requiring an opt-in flag; the
commercial-clean flag is metadata the UI/license gate reads, not a usability gate.
"""
from __future__ import annotations

import logging

from .registry import register
from .._otr_shared.role_compat import ROLES

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.image.flux_gen1")

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "State every requirement positively; exclusions have no effect at this "
    "guidance. One flowing sentence, subject first. Prefer specific nouns over "
    "stacked modifiers. Name lighting and lens facts plainly. No tag lists, no "
    "weight syntax."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
CONFIG AS SHIPPED: FLUX.1-dev fp8, 20 steps, cfg 1.0 with a FluxGuidance
embedding of 3.5. Guidance-distilled.

WHY THE DIRECTIVE LEADS WITH "POSITIVELY". At cfg 1.0 there is no
classifier-free-guidance branch, so the negative prompt is INERT -- it is not
weak, it is not consulted. That is the single most consequential fact about
phrasing for this engine, because the habit it breaks is universal: a writer who
has learned to push a look away by naming what to avoid gets literally nothing
here, silently, with no error and no receipt. The replacement is to name the
desired condition affirmatively -- "even overcast light" rather than "not harsh
light" -- which is why the directive spends its first clause on it and its last
on the CLIP-lineage syntax that also buys nothing.

The FluxGuidance embedding at 3.5 is a learned conditioning input, not a cfg
scale, so raising it does not restore a negative channel. Nothing a prompt can
say re-enables one.

EXTERNAL RESEARCH (2026-08-17, web lookup -- allowed per the operator's
2026-08-15 ruling, the RSS precedent). This lane came back CONFIRMED on every
clause, which is worth recording precisely because it is the least interesting
outcome and the easiest to leave unwritten:
  * Negatives: FLUX.1-dev is guidance-distilled and does not support them.
    Published guidance is to replace them with affirmative description -- write
    "clean, sharp background" rather than trying to exclude "blurry background".
    That is the directive's first clause, arrived at independently.
  * Subject-first: guides say FLUX wants natural language WITH THE SUBJECT FIRST.
  * Weight syntax: FLUX ignores parenthetical weight modifiers entirely, so the
    "no weight syntax" clause is not stylistic tidiness -- those characters buy
    literally nothing and cost budget.
  * Our FluxGuidance of 3.5 sits exactly on the published sweet spot (3.5-4.0), and
    guides warn that SD-habit values of 7.0+ produce oversaturated, artifact-heavy
    output on this architecture. Nothing to change; good to know the number is not
    accidental.

PROVENANCE: the DIRECTIVE was authored by the driver from this engine's shipped
configuration plus the five directive rules in the RESEARCH doc, then checked
against the external research above and found to agree on all four clauses. Still
NOT a measured finding on OUR lane. Treat the string as a hypothesis until the
probe A/B runs at a fixed seed.
"""


@register
class FluxGen1ImageEngine:
    """The gen-1 Flux image adapter (reduced ``prompt -> image`` protocol)."""

    name = "flux_gen1"
    #: Flux can produce an image for any role (announcer / music / character /
    #: scene / background); it is the in-stack default everywhere (gen 1).
    roles = ROLES
    default_roles = ROLES
    commercial_clean = False        # Flux.1-dev = BFL non-commercial
    requires_flag = None            # default engine -> no opt-in flag needed
    #: An image engine needs a text prompt; the edit/img2img path may also take
    #: an init_image, but gen-1 portrait/still generation is text -> image.
    required_inputs = ("text_prompt",)
    engine_version = "1"

    # --- in-process Flux gen-1 pipeline (the IMAGE GATE) -----------------
    # Mints a portrait from the MetaBrief prompt by driving the SHIPPED ComfyUI
    # Flux nodes in-process via wrapper_bridge -- the recipe proven on this RTX
    # 5080 in scripts/render_flux_batch.py: CheckpointLoaderSimple -> CLIPTextEncode
    # (pos/neg) -> EmptyLatentImage -> KSampler(euler/simple) -> VAEDecode. On this
    # box Flux ships as the all-in-one fp8 checkpoint "flux1-dev-fp8.safetensors"
    # (~13 GiB, under the 14.5 GB single-resident ceiling), NOT a UNET+CLIP+VAE
    # triplet -- so one CheckpointLoaderSimple is the loader. torch / comfy / numpy
    # are imported LAZILY (via wrapper_bridge, inside render_image) so importing
    # this module stays cold-import clean (V-12).

    #: Terminal graph node (its IMAGE output is the portrait).
    _TERMINAL = "decode"

    def _flux_params(self, request):
        """Pure: resolve the Flux sampler params from the request + env. The
        checkpoint / dims / steps / cfg / sampler are env-overridable (the
        operator points at the installed model without editing code); the seed +
        prompt come from the request so a re-gen is deterministic (V-7)."""
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

        return {
            "ckpt_name": otr_env.get(
                "OTR_FLUX_CKPT", "flux1-dev-fp8.safetensors"),
            "prompt": str(get("prompt") or ""),
            # Same request-negative channel as the other two negative-capable
            # engines (PBUG-20260817-01). INERT at the shipped cfg 1.0 -- Flux
            # takes adherence from the FluxGuidance embedding, not from CFG --
            # but OTR_FLUX_CFG is env-overridable, so leaving this one lane
            # reading env-only would be a latent inconsistency, not a saved line.
            # `is not None`, not `or` -- see lumina_image: an explicitly-empty
            # override means "no negative", not "fall through to the request".
            "negative": (_env_neg if (_env_neg := otr_env.get(
                "OTR_FLUX_NEGATIVE")) is not None
                else str(get("negative_prompt") or "")),
            "seed": int(get("seed") or 0),
            "steps": _eint("OTR_FLUX_STEPS", 20),
            "cfg": _efloat("OTR_FLUX_CFG", 1.0),
            # FluxGuidance embedding (BUG-411 restore): Flux.1-dev runs cfg=1.0
            # and takes its richness/adherence from the guidance value baked into
            # the conditioning by a FluxGuidance node -- the 6/5 pipeline applied
            # 3.5 and the rewrite dropped it, flattening the look. Env-overridable.
            "guidance": _efloat("OTR_FLUX_GUIDANCE", 3.5),
            "sampler_name": otr_env.get("OTR_FLUX_SAMPLER", "euler"),
            "scheduler": otr_env.get("OTR_FLUX_SCHEDULER", "simple"),
            # Request dims take precedence (still-spine ST-3 / pass-02 Gem-1:
            # w/h plumbed end-to-end so landscape SCENE stills are real);
            # the env knobs remain the no-request default.
            "width": int(get("width") or get("w")
                         or _eint("OTR_FLUX_WIDTH", 832)),
            "height": int(get("height") or get("h")
                          or _eint("OTR_FLUX_HEIGHT", 1216)),
        }

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node -- stock core
        classes (the proven render_flux_batch recipe)."""
        return {
            "ckpt": ("CheckpointLoaderSimple",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            # FluxGuidance lives in comfy_extras (nodes_flux); both the core and
            # the model-advanced registrations expose the same class name.
            "guidance": ("FluxGuidance",),
            "latent": ("EmptyLatentImage",),
            "ksampler": ("KSampler",),
            "decode": ("VAEDecode",),
        }

    def _build_flux_graph(self, params, wire):
        """Pure: the declarative Flux txt2img graph (wrapper_bridge.run_graph
        format). CheckpointLoaderSimple out slots are 0=MODEL, 1=CLIP, 2=VAE."""
        W = wire
        return {
            "ckpt": {"class": "ckpt",
                     "inputs": {"ckpt_name": params["ckpt_name"]}},
            "pos": {"class": "pos",
                    "inputs": {"text": params["prompt"], "clip": W("ckpt", 1)}},
            "neg": {"class": "neg",
                    "inputs": {"text": params["negative"], "clip": W("ckpt", 1)}},
            # FluxGuidance bakes the guidance value into the POSITIVE conditioning
            # (Flux.1-dev's real richness lever at cfg=1.0); the KSampler then
            # reads the guided positive. BUG-411 restore (6/5 applied guidance 3.5).
            "guidance": {"class": "guidance",
                         "inputs": {"conditioning": W("pos", 0),
                                    "guidance": float(params["guidance"])}},
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
                                    "positive": W("guidance", 0),
                                    "negative": W("neg", 0),
                                    "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("ksampler", 0),
                                  "vae": W("ckpt", 2)}},
        }

    # --- residency (classes resolve lazily; the loader node loads weights via
    #     ComfyUI's own model management when it executes) ---
    def load(self):  # pragma: no cover - resolved lazily in render_image
        from .._otr_video_engines import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def unload(self) -> None:  # pragma: no cover
        self._classes = None
        self._loaded = False

    def assert_usable(self, host_caps, profile, request_template=None):
        """Gen-1 Flux is usable wherever ComfyUI itself can run Flux. The real
        weight/dep check happens in the render (the loader node fails LOUD and the
        dispatcher degrades to the radio floor); this adapter returns its name
        (default engine) so the static dropdown + role filter resolve on CPU."""
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared=None):  # pragma: no cover - GPU/operator
        """Mint ONE portrait via the in-process ComfyUI Flux graph and return it
        as a decoded uint8 (H,W,3) RGB array (the dispatcher content-addresses +
        stamps it). Drives the proven render_flux_batch recipe through
        wrapper_bridge, then reclaims the resident checkpoint (BUG-291 detach) so
        VRAM drops back under the single-resident ceiling before HuMo. Raises a
        NAMED wrapper error on a missing node / failed render -- the dispatcher
        catches it fail-closed and the episode falls to the radio floor."""
        from .._otr_video_engines import wrapper_bridge as _wb
        params = self._flux_params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        self._classes = classes
        graph = self._build_flux_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)          # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: free the ~13 GiB checkpoint so HuMo can
            # take the lease next (LOUD; detach only, never unload_all_models).
            _wb.reclaim_idle_models(reason="flux_gen1 post-decode")
        log.info(
            "[OTR.image.flux_gen1] minted portrait %dx%d seed=%d steps=%d "
            "cfg=%.2f guidance=%.2f",
            params["width"], params["height"], params["seed"],
            params["steps"], params["cfg"], params["guidance"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover - GPU
        return None
