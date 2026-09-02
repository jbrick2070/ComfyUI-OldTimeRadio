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

THE TRAINED INPUT CONVENTION (2026-08-17 conformance audit). Lumina-2 is trained
on an instruction-style input: ComfyUI's own ``CLIPTextEncodeLumina2`` builds
``f'{system_prompt} <Prompt Start> {user_prompt}'``
(``comfy_extras/nodes_lumina2.py:113``). That prefix is NOT optional decoration
and it is NOT applied for us: ``comfy/text_encoders/lumina2.py``'s
``LuminaTokenizer`` is a plain ``SD1Tokenizer`` with no template of its own. This
is the exact opposite of Z-Image, whose ``qwen_image`` tokenizer wraps the text
INTERNALLY (``comfy/text_encoders/qwen_image.py:32-36``) -- which is why
``z_image_turbo`` is conformant on plain ``CLIPTextEncode`` and this engine was
not. Until the audit, every lumina mint ran out-of-distribution. We compose the
string here rather than swapping to ``CLIPTextEncodeLumina2`` because the graph
keeps ONE shape (the two classes take different input names), the composition
stays CPU-testable with no comfy import (V-12), and ComfyUI's own shipped
Lumina-family blueprint composes it as a string into a plain ``CLIPTextEncode``.

Flux stays gen 1; Lumina is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"). ``requires_flag`` is None -- the registry IS the menu, and
``OTR_ENABLE_LUMINA`` is vestigial, NOT a gate (pinned by
``tests/test_lumina_image_engine.py``, which deletes the var and still expects
the engine usable). The fail-closed gate is the WEIGHTS FILE
(``OTR_LUMINA_CKPT``): ``assert_usable``
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

#: The tag Lumina-2 was trained to split its system line from the user prompt on.
PROMPT_START_TAG = "<Prompt Start>"

#: The two system lines Lumina-2 ships with, copied VERBATIM from
#: ``CLIPTextEncodeLumina2.SYSTEM_PROMPT`` (``comfy_extras/nodes_lumina2.py``).
#: Held here rather than imported because this module is cold-import clean --
#: importing ``comfy_extras`` at module scope would drag in torch (V-12).
SYSTEM_PROMPTS = {
    "superior": (
        "You are an assistant designed to generate superior images with the "
        "superior degree of image-text alignment based on textual prompts or "
        "user prompts."
    ),
    "alignment": (
        "You are an assistant designed to generate high-quality images with "
        "the highest degree of image-text alignment based on textual prompts."
    ),
}

#: Which system line to use. ``superior`` is ComfyUI's own first/default option.
SYSTEM_ENV = "OTR_LUMINA_SYSTEM_PROMPT"
_DEFAULT_SYSTEM = "superior"

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "Write one declarative sentence, subject first. Adherence is literal at "
    "this guidance, so state only what must appear. No preamble or instruction "
    "wording; a system line is prepended. Full grammar, not tag lists."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
CONFIG AS SHIPPED: Lumina-Image 2.0, split-file, 30 steps, cfg 4.0, shift 6.0.
The text encoder is Gemma-2 2B (CLIPLoader type ``lumina2``), and every mint is
composed as ``{system line} <Prompt Start> {user text}`` by
``compose_encoder_text`` above.

WHY "NO PREAMBLE OR INSTRUCTION WORDING" IS THE UNIQUE RULE HERE. This is the
only engine in the stack whose conditioning text already opens with an
instruction addressed to an assistant ("You are an assistant designed to
generate superior images..."). Anything instruction-shaped the writer emits
lands AFTER that line and inside the same turn, so it reads as further
instruction rather than as scene content -- a failure mode no other engine has.
Gemma-2 is an LLM, so grammar is read: a full sentence conditions better than a
tag list, same as the Qwen3 encoder on z_image.

cfg 4.0 IS THE HIGHEST STILL GUIDANCE IN THE STACK (z_image runs 2.0), so
adherence is literal and over-specification is the expensive mistake, not
under-specification. Every clause the writer adds will be honoured.

THE NEGATIVE IS LIVE AT cfg 4.0, AND THE DIRECTIVE SAYS NOTHING ABOUT IT -- for
the ownership reason recorded on ``z_image_turbo.PROMPT_STYLE_NOTES``: the pack
owns the style negative, the engine owns hygiene, and a writer-authored negative
would be a third authority reintroducing PBUG-20260817-01 one layer up.
SEPARATELY AND STILL OPEN: this engine has no hygiene floor at all -- queue item
H. There is no ``_resolve_negative`` helper here at all (that name belongs to
``z_image_turbo``); the negative is resolved INLINE inside ``_lumina_params`` as a
bare ``str(get("negative_prompt") or "")``, with neither z_image's ``.strip()`` nor
its ``_HYGIENE_NEGATIVE`` fallback, and no hygiene constant exists in this file.
So an empty request negative reaches the encoder as ``""``.

**THE RECEIPT HALF OF ITEM H IS NOW FIXED, AND IT WAS NEVER A LUMINA BUG.** The
dispatcher's fourth ``_neg_source`` arm used to read ``"engine_hygiene"`` -- a
claim about what the ENGINE would do -- and it was computed BEFORE
``resolve_engine_for_role`` picks one, so it asserted a property of an engine not
yet chosen. True of z_image by COINCIDENCE (that engine really does have a floor),
false here, and consulted in neither case. It also mixed two authorities in one
value: composition is the dispatcher's business, the hygiene floor is the
engine's. That arm now reads ``"none_contributed"``, which is what the dispatcher
actually knows, so the value is engine-independent and this engine's missing floor
is no longer misreported by it.

**WHAT REMAINS OPEN IS ONLY THE FLOOR ITSELF, and it is the operator's call.**
Giving lumina a hygiene floor changes conditioning at cfg 4.0 on a live engine, so
the recipes directive applies and a render is owed. Three options are parked for
him: no floor, copy z_image's string, or a lumina-specific one. Do NOT copy
z_image's by reflex while editing nearby -- z_image runs cfg 1.0 and is a different
model with its own artifact profile.

EXTERNAL RESEARCH (2026-08-17, web lookup -- allowed per the operator's
2026-08-15 ruling, the RSS precedent):
  * Confirmed: the encoder is Gemma-2-2B, and because Gemma-2 was trained
    predominantly on English, English prompts perform best. Moot for this pipeline
    (it is English throughout) but it is the reason grammar reads well here.
  * **A TRAP IN THE PUBLIC MATERIAL, and it is the biggest one of the ten.** Most
    findable "Lumina prompting" writing is about **NETA LUMINA**, an ANIME FINETUNE,
    not base Lumina-Image 2.0. Neta's own guidance says it treats Danbooru-style
    TAGS and natural language as equal-level inputs -- so a reader who follows it
    will conclude tag soup is fine here. It is not: that tolerance was TRAINED INTO
    the finetune, and we load base Lumina-2 weights. The directive's "full grammar,
    not tag lists" stands, and this is exactly the "is this about the local weights
    or a variant" caution the RESEARCH doc's own prompt asks the researcher to
    honour.
  * Confirmed and already in the directive: detailed, specific prompts produce
    better embeddings here -- but read that alongside cfg 4.0 above, where the
    expensive mistake is over-specification rather than under-specification. The
    two are not in conflict: be specific about what must appear, not exhaustive.

PROVENANCE: the DIRECTIVE was authored by the driver from this engine's shipped
configuration plus the five directive rules in the RESEARCH doc, then checked
against the external research above. It is NOT a measured finding on OUR lane, and
most public guidance for this model name describes a finetune we do not load.
Treat the string as a hypothesis until the probe A/B runs at a fixed seed.
"""


def _resolve_system_key() -> str:
    """The system-prompt key for this mint, and the one place that decides.

    An unknown value falls back to the default LOUDLY rather than raising: a
    typo in an env var must not kill a render (BUG-046 degrades, never dies),
    but it must not be silent either or the operator never learns the knob
    missed."""
    key = (os.environ.get(SYSTEM_ENV, "") or "").strip() or _DEFAULT_SYSTEM
    if key not in SYSTEM_PROMPTS:
        log.warning(
            "[OTR.image.lumina_image] %s=%r is not one of %s -- falling back "
            "to %r", SYSTEM_ENV, key, sorted(SYSTEM_PROMPTS), _DEFAULT_SYSTEM)
        key = _DEFAULT_SYSTEM
    return key


def compose_encoder_text(user_text, system_key=None) -> str:
    """Pure: the string Lumina-2 was trained to receive.

    ``f'{system} <Prompt Start> {user}'`` -- byte-identical to what
    ``CLIPTextEncodeLumina2.execute`` hands the tokenizer, so a plain
    ``CLIPTextEncode`` fed this string and that node are the same two calls
    (``clip.tokenize`` then ``encode_from_tokens_scheduled``).

    IDEMPOTENT: text that already carries the tag is returned untouched, so a
    caller that composes twice cannot double-prefix (the precedent is
    ``_prefix_video_style_cue``'s already-prefixed check on the video side).
    An EMPTY user prompt still gets the system line -- that is exactly what the
    ComfyUI node emits for an empty ``user_prompt``, and conformance is the
    whole point of composing it."""
    text = str(user_text or "")
    if PROMPT_START_TAG in text:
        return text
    # None means "resolve from the env" so an out-of-engine caller (the live
    # smoke) composes the SAME string without duplicating the resolution rule.
    key = system_key or _resolve_system_key()
    system = SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS[_DEFAULT_SYSTEM])
    return "%s %s %s" % (system, PROMPT_START_TAG, text)


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
    # BUMPED 1 -> 2 by the 2026-08-17 input-convention fix. The dispatch cache
    # key is (role, object_id, prompt_hash, seed, engine_id, engine_version,
    # kind, w, h) -- and the prompt TEXT is unchanged by that fix, so without
    # this bump a resumed episode holding a pre-fix lumina cache entry would
    # keep re-serving the out-of-distribution still forever instead of
    # regenerating. No persisted ledger references lumina today, so this costs
    # nothing now; it is what makes the fix retroactive for any future resume.
    engine_version = "2"

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
            # The request's composed negative (pack style + per-object) wins;
            # the env override stays for dev only. Before 2026-08-17 this read
            # env ONLY, so the dispatcher's negative was computed and silently
            # discarded on this lane (PBUG-20260817-01). Live here: cfg 4.0.
            # `is not None`, not `or`: an explicitly-empty override means
            # "render with no negative", and must not fall through to the
            # request. That override-precedence rule matches
            # z_image_turbo._resolve_negative -- but the EDGES DELIBERATELY DO
            # NOT MATCH, and an earlier version of this comment wrongly claimed
            # they did. `z_image_turbo._resolve_negative` ends
            # `.strip() or _HYGIENE_NEGATIVE` (cited by SYMBOL: this comment used
            # to say `z_image_turbo.py:117` and the 2026-08-17 overlay commit
            # inserted lines above it, so the address was wrong within the hour);
            # lumina has no hygiene floor and no strip,
            # so an empty request negative reaches the encoder as "" and a
            # whitespace-only one is passed verbatim. That gap is REACHABLE
            # (`VISUAL_SAFETY_NEGATIVE_PROMPT` is "" and a pack may ship an
            # empty negative_tail). The dispatcher USED to label exactly that case
            # `_neg_source="engine_hygiene"`, which lumina does not honour; as of
            # 2026-08-17 that arm reads `none_contributed` and describes
            # composition only, so the receipt no longer claims a floor here.
            # Whether this engine should grow its own floor is a RENDER
            # decision on a different model, not a comment fix; it is logged,
            # not folded in here.
            "negative": (_env_neg if (_env_neg := os.environ.get(
                "OTR_LUMINA_NEGATIVE")) is not None
                else str(get("negative_prompt") or "")),
            # The system line is resolved here but applied in the GRAPH, so
            # `prompt` / `negative` stay the RAW request text: the dispatcher's
            # prompt_hash, cache key and seed derivation are computed upstream
            # from the request, and the composed string must never leak back
            # into them.
            "system_prompt": _resolve_system_key(),
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
        # Lumina-2's trained convention, applied to BOTH branches -- which is
        # what wiring a CLIPTextEncodeLumina2 into each side of the KSampler
        # does, and that node has no negative-specific mode. cfg defaults to
        # 4.0 here, so the uncond branch is genuinely sampled and an
        # out-of-distribution negative is not a free pass.
        _sys = params.get("system_prompt") or _DEFAULT_SYSTEM
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
                    "inputs": {"text": compose_encoder_text(params["prompt"], _sys),
                               "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": compose_encoder_text(params["negative"], _sys),
                               "clip": W("clip", 0)}},
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
                f"/{VAE_ENV} for the Gemma-2 TE + ae VAE)",
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
            _wb.reclaim_idle_models(reason="lumina_image post-decode")
        log.info(
            "[OTR.image.lumina_image] minted still %dx%d seed=%d steps=%d "
            "cfg=%.2f shift=%.2f system=%s", params["width"], params["height"],
            params["seed"], params["steps"], params["cfg"], params["shift"],
            params["system_prompt"])
        return frames[0]

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["LuminaImage2Engine", "ENABLE_FLAG", "MODEL_ENV", "CLIP_ENV", "VAE_ENV",
           "SYSTEM_ENV", "SYSTEM_PROMPTS", "PROMPT_START_TAG",
           "compose_encoder_text"]
