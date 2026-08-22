"""``animatediff15_video`` -- AnimateDiff Ghost Signal, the prompt-only lane.

THE SHAPE OF THIS LANE IN ONE PARAGRAPH. SD1.5 plus the v2 motion module
``mm-p_0.5.pth``, sampled at a fixed 512x288 through AnimateDiff-Evolved's
NON-LOOPED Standard Static context, then delivered by clean Lanczos straight to
1920x1080 at 25 fps. It owns its subject entirely: ``accepts_still = False``, so
the image phase mints NOTHING for it and every pixel traces to text this repo
composed. That is the whole design -- there is no still to inherit a look from,
no IPAdapter, no ControlNet, no reference folder.

WHAT IS DELIBERATELY ABSENT, so nobody adds it as a missing piece: looped
context, ping-pong, mirror, loop-fill, RIFE, a second KSampler, a hires fix, an
ESRGAN/SeedVR2 upscaler, ``ADE_AnimateDiffSamplingSettings``, any multival
source, Motion LoRA, keyframes, per-block, and view options. The optional ADE
sockets are OMITTED from the runtime input dictionaries rather than passed an
invented ``None`` widget, which is what leaves the pinned implementation owning
its own normal full-strength behaviour.

CADENCE IS A RESAMPLE, NOT AN EXTENSION. Ghost generates ``U = ceil(T/2)`` fresh
source positions for the beat's authoritative delivered target ``T`` and holds
each twice, except the once-shown odd tail. Every delivered frame selects a real
decoded frame FROM THE SAME BEAT; nothing is invented, no tail is mirrored, and
the duration is exact. ``extension_mode`` is ``"none"`` and it is true.

"FRESH" AND "UNIQUE" ARE PROVENANCE LAWS. Every source index is generated for
that beat and never borrowed from another. They do not claim a diffusion model
can be forced to produce pixelwise-different pictures.

NO VRAM CLAIM IS MADE ANYWHERE IN THIS FILE. The operator declined a measurement
campaign, so "very-low-VRAM-targeted" is a design target and admission is
recorded UNENFORCED in the evidence manifest. This lane may OOM; if it does, the
failure is named and the profile stays draft.

Cold-import clean (V-12): torch, numpy and every ComfyUI class are resolved
lazily inside the lifecycle methods, never at module scope.
"""
from __future__ import annotations

import logging
import os

from . import ghost_signal_prompt as _gsp
from . import motion_common as _MC
from .frame_contract import CONTINUITY_NONE, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.video.ghost_signal")

# --------------------------------------------------------------------------- #
# The frozen recipe. Every value here is pinned by the R3 contract in
# docs/2026-08-22-GHOST-SIGNAL-CODING-PLAN.md section 4.3 and by tests.
# --------------------------------------------------------------------------- #

#: The recipe receipt string threaded into the manifest. The version lives IN
#: the string: bumping a recipe means repointing this, never editing a versioned
#: value in place, or receipts already on disk stop being interpretable.
GHOST_RECIPE_RECEIPT = "animatediff_sd15_mmp05_static16_512x288_v1"

#: The two artifacts, by the BARE TOKEN their loader nodes receive. ComfyUI
#: resolves a bare token through ``folder_paths``, which is what makes
#: ``extra_model_paths.yaml`` work.
GHOST_CHECKPOINT_NAME = "v1-5-pruned-emaonly-fp16.safetensors"
GHOST_MOTION_MODULE_NAME = "mm-p_0.5.pth"

#: ``folder_paths`` categories the two tokens resolve through.
GHOST_CHECKPOINT_CATEGORY = "checkpoints"
GHOST_MOTION_CATEGORY = "animatediff_models"

#: Byte floors, so a truncated fetch is NAMED rather than traced through a
#: loader stack. These are the exact sizes in the dependency lock, minus a
#: small margin -- a file materially under them is not the pinned artifact.
GHOST_CHECKPOINT_MIN_BYTES = 2_000_000_000
GHOST_MOTION_MIN_BYTES = 1_700_000_000

#: THE NATIVE CANVAS. Fixed, full-frame 16:9, and legal on the live OTR /32
#: canvas law on BOTH axes -- which the retired 384x216 was not (216 % 32 != 0).
GHOST_CANVAS_W = 512
GHOST_CANVAS_H = 288

#: DELIVERY. Exact, and the mode is DECLARED DATA the composite reads -- never
#: an engine-name check downstream.
GHOST_DELIVERY_W = 1920
GHOST_DELIVERY_H = 1080
GHOST_DELIVERY_SCALE_MODE = "lanczos_clean_full_frame"

#: THE SAMPLER RECIPE. One KSampler, and the negative is LIVE at this cfg --
#: which is the whole reason an AnimateLCM checkpoint was refused: the Ghost
#: lettering defense needs real unconditional conditioning, and moving the
#: negative into the positive prompt would be an unreviewed workaround.
GHOST_STEPS = 20
GHOST_CFG = 8.0
GHOST_SAMPLER_NAME = "euler"
GHOST_SCHEDULER = "normal"
GHOST_DENOISE = 1.0
GHOST_BETA_SCHEDULE = "autoselect"

#: THE CONTEXT. Non-circular Standard Static: it slides length-16 windows with
#: overlap 4 across a longer timeline and does NOT close the loop.
GHOST_CONTEXT_LENGTH = 16
GHOST_CONTEXT_OVERLAP = 4
GHOST_CONTEXT_FUSE_METHOD = "pyramid"
GHOST_CONTEXT_USE_ON_EQUAL_LENGTH = False
GHOST_CONTEXT_START_PERCENT = 0.0
GHOST_CONTEXT_GUARANTEE_STEPS = 1

#: THE STRUCTURAL SOURCE FLOOR. Sixteen is the pinned upstream SD1.5 AnimateDiff
#: context length this fixed recipe uses, NOT a measured Ghost quality claim.
#: With ``use_on_equal_length=False`` an exact 16-frame request uses the motion
#: model directly; anything longer activates the static sliding context.
GHOST_SOURCE_FLOOR = 16

#: The source rate. 12.5 into 25 is exactly hold-2, which is why this lane can
#: promise an EXACT duration with no resampling arithmetic at all.
GHOST_SOURCE_FPS = 12.5
GHOST_TARGET_FPS = 25

#: THE EIGHT NODE INSTANCES. Ids are stable strings because the tests, the
#: executor aliases and the audit records all name them.
NODE_CKPT = "ckpt"
NODE_POSITIVE = "positive"
NODE_NEGATIVE = "negative"
NODE_CONTEXT = "context"
NODE_ADE = "ade"
NODE_LATENT = "latent"
NODE_SAMPLER = "sampler"
NODE_DECODE = "decode"

#: The SEVEN unique class ids the eight instances resolve to (the repeated text
#: class is resolved once). ONE NAME PER ALIAS -- never alternative spellings.
#: A miss stops the build; it does not activate runtime probing or a fallback
#: node. Every one of these was confirmed live in the Phase-0 capture at
#: docs/2026-08-22-ghost-signal-object-info.json.
GHOST_NODE_CANDIDATES = {
    "checkpoint": ("CheckpointLoaderSimple",),
    "text_encode": ("CLIPTextEncode",),
    "context": ("ADE_StandardStaticContextOptions",),
    "ade": ("ADE_AnimateDiffLoaderGen1",),
    "latent": ("EmptyLatentImage",),
    "sampler": ("KSampler",),
    "decode": ("VAEDecode",),
}

#: The ADE optional sockets this lane deliberately leaves UNCONNECTED. Pinned as
#: a set so a test can prove the runtime input dictionary contains none of them
#: -- omission is the contract, and an invented ``None`` would not be omission.
GHOST_ADE_OMITTED_SOCKETS = frozenset({
    "motion_lora", "ad_settings", "ad_keyframes", "sample_settings",
    "scale_multival", "effect_multival", "per_block",
})

#: The context node's two optional sockets, likewise unconnected.
GHOST_CONTEXT_OMITTED_SOCKETS = frozenset({"prev_context", "view_opts"})


class GhostCadenceError(ValueError):
    """The cadence arithmetic was handed something it cannot deliver."""


# --------------------------------------------------------------------------- #
# Cadence. PURE, module-level, and CPU-testable with nothing installed.
# --------------------------------------------------------------------------- #

def ghost_unique_source_count(target_frame_count) -> int:
    """``U = ceil(T / 2)`` -- how many FRESH source positions a beat needs.

    Derived from the integer target, never from a float duration: the two
    disagree at half-second banker's-rounding boundaries and the delivered count
    is the only duration authority this lane recognises.
    """
    target = int(target_frame_count or 0)
    if target < 1:
        raise GhostCadenceError(
            "Ghost Signal cadence needs a delivered target of at least 1 "
            "frame, got %r -- a 0-frame beat is a planning bug, not something "
            "to paper over here" % (target_frame_count,))
    return (target + 1) // 2


def ghost_source_request(target_frame_count) -> int:
    """How many source frames to actually ASK the model for.

    ``max(U, 16)``: the structural floor is the pinned context length, so a very
    short beat still hands the sampler a full window. The surplus is discarded
    BEFORE cadence conversion and is reported separately -- it is model work that
    happened, not a delivered frame.
    """
    return max(ghost_unique_source_count(target_frame_count), GHOST_SOURCE_FLOOR)


def ghost_hold2_selector(target_frame_count) -> list:
    """The delivered-index -> source-index map: ``[0, 0, 1, 1, ...][:T]``.

    Source frames ``0..U-2`` appear exactly twice. Source frame ``U-1`` appears
    twice when ``T`` is even and once when ``T`` is odd. Monotonically
    nondecreasing, every entry in ``[0, U)``, and exactly ``T`` long.
    """
    target = int(target_frame_count or 0)
    unique = ghost_unique_source_count(target)
    selector = []
    for index in range(unique):
        selector.append(index)
        selector.append(index)
    return selector[:target]


def ghost_cadence_receipts(target_frame_count, source_request=None) -> dict:
    """The MiniMax-H3-precedent rate-conversion accounting, as plain data.

    The scopes are separate on purpose. ``native_frame_count`` is the EMITTED
    count, so it can never exceed the clip; ``model_frame_count`` is every frame
    actually requested from and decoded by the model; and
    ``cadence_source_frame_count`` is the unique prefix the selector retained.
    ``model_frame_count - cadence_source_frame_count`` therefore exposes,
    truthfully, any source frames generated only to satisfy the live node's
    structural minimum, while ``cadence_tail_trim`` covers ONLY the final
    hold-2 surplus and is always 0 or 1.

    Keeping them apart is what stops ``native_frame_count`` being misread as
    "T distinct diffusion samples". It is not.
    """
    target = int(target_frame_count or 0)
    unique = ghost_unique_source_count(target)
    requested = int(source_request if source_request is not None
                    else ghost_source_request(target))
    return {
        "native_frame_count": target,
        "extension_mode": "none",
        "model_frame_count": requested,
        "cadence_mode": "hold_2",
        "cadence_source_frame_count": unique,
        "cadence_delivered_frame_count": target,
        "cadence_tail_trim": (2 * unique) - target,
        "delivery_scale_mode": GHOST_DELIVERY_SCALE_MODE,
    }


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #

@register
class GhostSignalEngine(_MC.MotionEngineBase):
    """``animatediff15_video`` -- AnimateDiff Ghost Signal."""

    name = "animatediff15_video"
    family = "text_to_video"

    #: THE PEER-LANE SEAM (2026-08-22). CLASS attributes, not module constants
    #: read from inside methods, and the distinction is the whole point:
    #: `eng_fastwan_8gb` records that a parent reading module-level constants
    #: means "a subclass declaring its own recipe would have SILENTLY rendered
    #: with the parent's and stamped its own receipt on the result". A peer lane
    #: on a different motion module must actually LOAD that module, and its
    #: receipt must actually NAME it.
    #:
    #: The module-level constants remain the default and the frozen artifact;
    #: only the read path moved, so this lane is byte-identical.
    motion_module_name = GHOST_MOTION_MODULE_NAME
    recipe_receipt_id = GHOST_RECIPE_RECEIPT
    required_inputs = ("text_prompt",)
    optional_inputs = ()
    roles = ("announcer_visual", "music_visual", "character_video")
    #: EMPTY. A new lane is opt-in through the director menu; it never seizes a
    #: role by declaring itself the default for one.
    default_roles = ()
    requires_flag = None

    render_aspect = "wide"
    render_canvas = (GHOST_CANVAS_W, GHOST_CANVAS_H)
    target_fps = GHOST_TARGET_FPS

    #: THE G3.7 DECLARATION, AND IT IS TRUE. This lane consumes no still of any
    #: kind, so it declares none -- and the portrait-free role set reads THIS,
    #: never an engine name, which is why no portrait is minted for a Ghost
    #: character beat with no edit anywhere in the image phase.
    accepts_still = False
    still_plan: tuple = ()
    subject_ownership = "prompt"

    #: The prompt capability the render driver branches on. A CAPABILITY, so the
    #: driver never compares an engine id.
    prompt_profile = _gsp.GHOST_PROMPT_PROFILE
    prompt_budget_chars = _gsp.GHOST_PROMPT_MAX_CHARS
    style_join = "compose"
    delivery_scale_mode = GHOST_DELIVERY_SCALE_MODE

    #: WHERE THE BEAT'S MOTION COMES FROM. A prompt-owned lane has no still to
    #: inherit movement from, so it has to name its motion authority -- and this
    #: one is the ledger's own `motion_clause`, with a deterministic pack/intent
    #: fallback when the optional motion pass is off. Declared rather than
    #: inferred because G3.7 asks the lane, not the reader.
    motion_source = "ledger_motion_clause"

    #: WHERE THE NEGATIVE ACTUALLY LANDS. Not decoration: it names the real graph
    #: binding, and G3.7 checks the CODE against it -- a lane whose adapter
    #: substitutes an engine-side constant for a missing request negative fails
    #: that check no matter what it declares here.
    negative_prompt_binding = "clip_text_encode_negative"

    #: LICENSE TRUTH. The ADE code is Apache-2.0 and the checkpoint is
    #: CreativeML Open RAIL-M, but the selected motion module's host publishes
    #: no model license grant -- so this lane is NOT commercially clean, and it
    #: says so. ``CAPABILITIES`` has no such key; this is the adapter's.
    commercial_clean = False

    #: DELIVERED-FRAME UNITS, unbounded, quantum 1. A beat is ONE timeline even
    #: when it spans several internal context windows -- the 16 is a context
    #: window, never an OTR clip duration -- so there is no ceiling to split on
    #: and continuity is explicitly NONE rather than inherited.
    frame_contract = FrameContract(
        min_frames=1,
        max_frames=0,
        quantum=1,
        native_fps=GHOST_TARGET_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_NONE,
    )

    def __init__(self):
        super().__init__()
        self._classes = None
        self._artifacts = None

    # ---- identity ------------------------------------------------------ #
    def _recipe_receipt(self):
        return self.recipe_receipt_id

    @staticmethod
    def _file_receipt(path):
        """``(size, mtime_ns)`` for a resolved artifact, or ``None``.

        Re-stat'ed on every ask rather than cached: the job is to NOTICE a
        weight that moved, and two stats are cheap where a hash of 3.9 GB is
        not."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def session_identity(self):
        """What this adapter's HANDLES are: engine, recipe, both artifacts.

        PRE-LOAD STABLE, like its siblings -- it may only describe what does not
        change across the load, so per-shot state (prompt, seed, frame count) is
        deliberately absent. That state lives in :meth:`shot_cache_identity`,
        which is a different question with a different answer.
        """
        parts = [self.name, self._recipe_receipt()]
        for token, path in (
                (GHOST_CHECKPOINT_NAME, self._ckpt_path()),
                (self.motion_module_name, self._motion_path())):
            parts.append(token)
            parts.append(repr(self._file_receipt(path or "")))
        return tuple(parts)

    def shot_cache_identity(self, request):
        """The PER-SHOT identity, and it is deliberately not the session one.

        A hit may replay only that same shot's deterministic artifact. Two beats
        with identical composed text are still two beats: the shot id and the
        resolved seed are in here precisely so a prompt-only cache key can never
        collapse them into one, and so no beat can be handed another beat's
        source batch or output path.
        """
        plan = self._build_render_request(request)
        return (
            self.name,
            self._recipe_receipt(),
            str(plan["shot_id"]),
            int(plan["seed"]),
            int(plan["unique_source_count"]),
            "%dx%d" % (GHOST_CANVAS_W, GHOST_CANVAS_H),
            GHOST_CHECKPOINT_NAME,
            self.motion_module_name,
            _sha8(plan["text_prompt"]),
            _sha8(plan["negative_prompt"]),
        )

    # ---- artifact resolution ------------------------------------------- #
    @staticmethod
    def _resolve_model_file_by_token(categories, name):
        """Where the LOADER would find ``name`` -- through the live
        ``folder_paths`` categories, so ``extra_model_paths.yaml`` works.

        The graphs hand their loader nodes a BARE TOKEN and ComfyUI resolves it
        through ``folder_paths``; asking the same way is the only check that
        answers the question the loader will actually ask. A directory override
        that satisfies a preflight while being invisible to the loader is the
        `wan_i2v` killer (lesson L1), and this name is the established one for
        exactly that reason -- it is the same question `wan_shared` asks.

        Returns ``None`` when absent everywhere. The offline invariant means NO
        runtime fetch: a missing file is fail-closed, never downloaded.
        """
        try:
            import folder_paths  # type: ignore
        except Exception:  # noqa: BLE001 -- not inside ComfyUI
            return None
        for category in ((categories,) if isinstance(categories, str)
                         else tuple(categories or ())):
            try:
                found = folder_paths.get_full_path(category, name)
            except Exception:  # noqa: BLE001 -- unregistered category -> None
                found = None
            if found:
                return found
        return None

    def _ckpt_path(self):
        """The SD1.5 checkpoint, by the loader's own resolution."""
        return self._resolve_model_file_by_token(
            (GHOST_CHECKPOINT_CATEGORY,), GHOST_CHECKPOINT_NAME)

    def _motion_path(self):
        """The v2 motion module, by the loader's own resolution."""
        return self._resolve_model_file_by_token(
            (GHOST_MOTION_CATEGORY,), self.motion_module_name)

    def _installed(self):
        """A PREDICATE -- it answers, it never raises."""
        try:
            return bool(self._ckpt_path()) and bool(self._motion_path())
        except Exception:  # noqa: BLE001
            return False

    def _node_candidates(self):
        return dict(GHOST_NODE_CANDIDATES)

    # ---- preflight ------------------------------------------------------ #
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail CLOSED, cheapest refusal first, and NEVER a download.

        Order is the contract: Sage costs nothing to check; then both artifacts
        by the loader's own resolution, then their byte floors so a truncated
        fetch is named rather than traced; then every node class at once, so a
        fresh install is one refusal naming everything instead of a sequence of
        failed renders.

        The VAE requirement is the checkpoint's output slot 2, not another file,
        so there is deliberately no third artifact to look for.
        """
        _MC.assert_sage_not_patched(self.name, self.family)

        rows = (
            ("checkpoint", GHOST_CHECKPOINT_NAME, GHOST_CHECKPOINT_CATEGORY,
             self._ckpt_path(), GHOST_CHECKPOINT_MIN_BYTES),
            ("motion_module", self.motion_module_name, GHOST_MOTION_CATEGORY,
             self._motion_path(), GHOST_MOTION_MIN_BYTES),
        )
        missing = ["%s=%s (folder_paths category %r)" % (label, token, category)
                   for label, token, category, path, _floor in rows if not path]
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s artifact(s) not found: %s -- drop them in the matching "
                "folder or register it in extra_model_paths.yaml. There is no "
                "fallback list and nothing is downloaded at render time."
                % (self.name, ", ".join(missing)), kind="video")

        for label, token, _category, path, floor in rows:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s %r is only %d bytes (< the %d floor) -- that is a "
                    "truncated or wrong file, not the pinned artifact; "
                    "re-fetch it" % (self.name, label, token, size, floor),
                    kind="video")

        from . import wrapper_bridge as _wb
        mapping = _wb.node_class_mappings()
        absent = []
        for _logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 -- collect EVERY miss before raising
                absent.append("/".join(candidates))
        if absent:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s -- Ghost Signal "
                "needs the canonical Kosinkadink ComfyUI-AnimateDiff-Evolved "
                "installed (see docs/2026-08-22-ghost-signal-dependency-lock"
                ".json for the pinned commit) and a server restart"
                % (self.name, ", ".join(absent)), kind="video")
        return self.name

    # ---- residency ------------------------------------------------------ #
    def load(self):
        """Resolve CLASSES and artifact tokens only. No node runs here.

        THIS OVERRIDE IS MANDATORY, not stylistic. ``MotionEngineBase.prepare``
        acquires the AS-3 lease and then unconditionally calls ``self.load()``;
        inheriting the base implementation would raise ``NotImplementedError``
        on every single beat.
        """
        if not self._installed():
            missing = ", ".join(
                token for token, path in (
                    (GHOST_CHECKPOINT_NAME, self._ckpt_path()),
                    (self.motion_module_name, self._motion_path()))
                if not path)
            raise RuntimeError(
                "%s not installed: missing %s" % (self.name, missing))
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._artifacts = {
            "checkpoint": GHOST_CHECKPOINT_NAME,
            "motion_module": self.motion_module_name,
        }
        self._loaded = True

    def unload(self):
        """Clear GHOST-OWNED references only, then restore the base flags.

        It never clears a global model registry and never calls
        ``unload_all_models`` (V-4 / V-5). The staged reclaim seam may detach
        loaded patchers; that is a different act from evicting the world.
        """
        self._classes = None
        self._artifacts = None
        super().unload()

    def prepare(self, host_caps, profile, session_ctx):
        """Base prepare (lease + load), then execute ONLY the ``ckpt`` node.

        The three outputs are split into SEPARATE one-slot handles and the
        combined result tuple is discarded, so a released family can never be
        retained by a tuple nobody remembered was holding it.
        """
        prepared = super().prepare(host_caps, profile, session_ctx)
        from . import wrapper_bridge as _wb

        classes = self._classes or _wb.resolve_graph_classes(
            self._node_candidates())

        def _register_base(node_id, out_tuple):
            """Register the base MODEL patcher THE MOMENT IT EXISTS.

            A post-``run_graph`` scan is too late: the executor can evict
            between the return and the scan, and a patcher we never registered
            is a patcher teardown cannot detach."""
            if node_id != NODE_CKPT:
                return
            if len(out_tuple) != 3:
                raise RuntimeError(
                    "%s: CheckpointLoaderSimple returned %d outputs, expected "
                    "exactly 3 (MODEL, CLIP, VAE)"
                    % (self.name, len(out_tuple)))
            model = out_tuple[0]
            if not callable(getattr(model, "detach", None)):
                raise RuntimeError(
                    "%s: the checkpoint MODEL has no callable detach() -- this "
                    "adapter cannot take ownership of a patcher it cannot "
                    "release, and proceeding would strand it resident"
                    % self.name)
            self._patchers.append(model)

        ckpt_out = _wb.run_graph(
            {NODE_CKPT: {"class": classes["checkpoint"],
                         "inputs": {"ckpt_name": GHOST_CHECKPOINT_NAME}}},
            terminal=NODE_CKPT, on_result=_register_base)

        if len(ckpt_out) != 3:
            raise RuntimeError(
                "%s: expected MODEL/CLIP/VAE from the checkpoint, got %d "
                "output(s)" % (self.name, len(ckpt_out)))

        # SEPARATE ONE-SLOT HANDLES, and the three-slot tuple goes away here.
        prepared["base_model"] = (ckpt_out[0],)
        prepared["clip"] = (ckpt_out[1],)
        prepared["vae"] = (ckpt_out[2],)
        prepared["recipe"] = self._recipe_receipt()
        del ckpt_out
        return prepared

    # ---- request -------------------------------------------------------- #
    def _build_render_request(self, request):
        """PURE: the normalized plan this adapter renders from.

        The negative is read from the REQUEST FIELD by name and a blank one is
        refused. There is deliberately no ``request.get("negative_prompt") or
        <engine constant>`` here: an engine-side negative constant would let the
        lane render with lettering hygiene the composer never authored, and the
        receipt would describe a negative that did not come from the pack.
        """
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))

        target = int(t_get("target_frame_count", 0) or 0)
        text_prompt = str(get("text_prompt") or "").strip()
        negative_prompt = str(get("negative_prompt") or "").strip()
        unique = ghost_unique_source_count(target)
        return {
            "shot_id": str(get("shot_id") or get("request_id") or ""),
            "text_prompt": text_prompt,
            "negative_prompt": negative_prompt,
            "seed": int(s_get("request_seed", 0) or 0),
            "target_frame_count": target,
            "unique_source_count": unique,
            "source_request": ghost_source_request(target),
            "fps": int(self.target_fps),
        }

    def _assert_required_inputs(self, plan):
        if not plan["text_prompt"]:
            raise RuntimeError(
                "%s requires a composed text_prompt and the request carries "
                "none -- Ghost owns its whole subject, so an empty prompt is a "
                "blank picture, not a default" % self.name)
        if not plan["negative_prompt"]:
            raise RuntimeError(
                "%s requires a composed negative_prompt and the request "
                "carries none. NO ENGINE-SIDE CONSTANT SUBSTITUTES FOR IT: the "
                "lane hygiene head and the pack's effective_negative are the "
                "composer's to author, and a negative invented here would be a "
                "receipt that lies about where it came from" % self.name)

    # ---- render --------------------------------------------------------- #
    def render_clip(self, request, prepared):
        """One Ghost beat: encode, sample, decode, hold-2, encode a silent MP4.

        Four bounded executor calls so the large families are not artificially
        pinned by one result dictionary. The cross-stage release law is the
        explicit owner clearing plus the public reclaim seam -- ``free_after_use``
        alone cannot release an external, because ``run_graph`` keeps every
        external id for the duration of that call.
        """
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4

        plan = self._build_render_request(request)
        self._assert_required_inputs(plan)
        classes = self._classes or _wb.resolve_graph_classes(
            self._node_candidates())
        owners = prepared if isinstance(prepared, dict) else {}

        # ---- STAGE 2: ENCODE ------------------------------------------- #
        encode_graph = {
            NODE_POSITIVE: {
                "class": classes["text_encode"],
                "inputs": {"text": plan["text_prompt"],
                           "clip": _wb.Wire("clip", 0)}},
            NODE_NEGATIVE: {
                "class": classes["text_encode"],
                "inputs": {"text": plan["negative_prompt"],
                           "clip": _wb.Wire("clip", 0)}},
        }
        encode_results = _wb.run_graph(
            encode_graph, external_results={"clip": owners["clip"]})
        positive_cond = (encode_results[NODE_POSITIVE][0],)
        negative_cond = (encode_results[NODE_NEGATIVE][0],)

        # The CLIP family goes now: both conditionings are extracted, so every
        # strong reference to it must be gone BEFORE the reclaim seam runs or
        # the seam has nothing to move.
        encode_results.clear()
        del encode_graph
        owners.pop("clip", None)
        _wb.reclaim_idle_models("ghost-signal post-encode")

        # ---- STAGE 3: SAMPLE ------------------------------------------- #
        sample_graph = {
            NODE_CONTEXT: {
                "class": classes["context"],
                # The exact six frozen literals. prev_context and view_opts are
                # OMITTED, not passed None.
                "inputs": {
                    "context_length": GHOST_CONTEXT_LENGTH,
                    "context_overlap": GHOST_CONTEXT_OVERLAP,
                    "fuse_method": GHOST_CONTEXT_FUSE_METHOD,
                    "use_on_equal_length": GHOST_CONTEXT_USE_ON_EQUAL_LENGTH,
                    "start_percent": GHOST_CONTEXT_START_PERCENT,
                    "guarantee_steps": GHOST_CONTEXT_GUARANTEE_STEPS,
                }},
            NODE_ADE: {
                "class": classes["ade"],
                # motion_lora / ad_settings / ad_keyframes / sample_settings /
                # scale_multival / effect_multival / per_block are OMITTED, which
                # is what leaves the pinned ADE implementation owning its own
                # normal full-strength behaviour.
                "inputs": {
                    "model": _wb.Wire("base_model", 0),
                    "model_name": self.motion_module_name,
                    "beta_schedule": GHOST_BETA_SCHEDULE,
                    "context_options": _wb.Wire(NODE_CONTEXT, 0),
                }},
            NODE_LATENT: {
                "class": classes["latent"],
                "inputs": {"width": GHOST_CANVAS_W,
                           "height": GHOST_CANVAS_H,
                           "batch_size": plan["source_request"]}},
            NODE_SAMPLER: {
                "class": classes["sampler"],
                "inputs": {
                    "model": _wb.Wire(NODE_ADE, 0),
                    "seed": plan["seed"],
                    "steps": GHOST_STEPS,
                    "cfg": GHOST_CFG,
                    "sampler_name": GHOST_SAMPLER_NAME,
                    "scheduler": GHOST_SCHEDULER,
                    "positive": _wb.Wire("positive_cond", 0),
                    "negative": _wb.Wire("negative_cond", 0),
                    "latent_image": _wb.Wire(NODE_LATENT, 0),
                    "denoise": GHOST_DENOISE,
                }},
        }

        def _register_ade(node_id, out_tuple):
            """Register the PATCHED model immediately -- before the executor
            can evict it."""
            if node_id != NODE_ADE:
                return
            if not out_tuple:
                raise RuntimeError(
                    "%s: the ADE loader returned no MODEL" % self.name)
            patched = out_tuple[0]
            if not callable(getattr(patched, "detach", None)):
                raise RuntimeError(
                    "%s: the patched ADE MODEL has no callable detach() -- "
                    "this adapter cannot take ownership of a patcher it cannot "
                    "release" % self.name)
            owners["ade_model"] = patched
            self._patchers.append(patched)

        sampled = _wb.run_graph(
            sample_graph,
            external_results={"base_model": owners["base_model"],
                              "positive_cond": positive_cond,
                              "negative_cond": negative_cond},
            terminal=NODE_SAMPLER, on_result=_register_ade)
        sampled_latent = (sampled[0],)

        # BOTH SAMPLING PATCHERS GO BEFORE DECODE, in ADE-then-base order.
        self._release_sampling_patchers_before_decode(owners)

        del sample_graph
        positive_cond = None
        negative_cond = None
        del sampled
        _wb.reclaim_idle_models("ghost-signal post-sample")

        # ---- STAGE 4: DECODE ------------------------------------------- #
        decode_graph = {
            NODE_DECODE: {
                "class": classes["decode"],
                "inputs": {"samples": _wb.Wire("sampled_latent", 0),
                           "vae": _wb.Wire("vae", 0)}},
        }
        images = _wb.run_graph(
            decode_graph,
            external_results={"sampled_latent": sampled_latent,
                              "vae": owners["vae"]},
            terminal=NODE_DECODE)[0]

        frames = _wb.images_to_uint8(images)
        decoded = int(frames.shape[0])
        if decoded != plan["source_request"]:
            raise RuntimeError(
                "%s decoded %d frame(s) but asked for exactly %d -- a count "
                "mismatch is a graph fault, and this lane pads nothing, "
                "substitutes no alternate graph and falls back to no still"
                % (self.name, decoded, plan["source_request"]))

        # ---- CADENCE ---------------------------------------------------- #
        # Retain the first U source frames in order; the rest is the structural
        # surplus and is discarded BEFORE conversion, never delivered.
        selector = ghost_hold2_selector(plan["target_frame_count"])
        unique = plan["unique_source_count"]
        delivered = frames[[min(i, unique - 1) for i in selector]]

        out_path = otr_engine_tmp_mp4("ghost_signal")
        out_path, proven = _wb.encode_frames_to_silent_mp4(
            delivered, out_path, int(self.target_fps))
        if int(proven) != int(plan["target_frame_count"]):
            raise RuntimeError(
                "%s encoded %s frame(s) where the beat's delivered target is "
                "%d" % (self.name, proven, plan["target_frame_count"]))

        raw = {
            "out_path": out_path,
            "frame_count": int(proven),
            "recipe": self._recipe_receipt(),
            "render_canvas": "%dx%d" % (GHOST_CANVAS_W, GHOST_CANVAS_H),
            "vram_peak_mb": None,        # no measurement campaign was authorized
        }
        raw.update(ghost_cadence_receipts(
            plan["target_frame_count"], plan["source_request"]))
        _LOG.info(
            "[OTR video] %s beat %s: T=%d U=%d requested=%d decoded=%d "
            "tail_trim=%d @ %dx%d -> %dx%d %s",
            self.name, plan["shot_id"], plan["target_frame_count"], unique,
            plan["source_request"], decoded, raw["cadence_tail_trim"],
            GHOST_CANVAS_W, GHOST_CANVAS_H, GHOST_DELIVERY_W,
            GHOST_DELIVERY_H, GHOST_DELIVERY_SCALE_MODE)
        return raw

    def _release_sampling_patchers_before_decode(self, owners):
        """Detach and identity-remove the ADE then base patchers, in that order.

        A GENERIC COMFY ``ModelPatcher`` CONTRACT, not copied LTX/Wan loader
        behaviour: ``detach(unpatch_all=True)`` is what actually moves weights to
        the offload device.

        Ordered, identity-deduplicated candidates. For each: detach first, and
        only AFTER it succeeds remove that exact identity, in place, from every
        distinct patcher list, and clear the Ghost-owned reference. If detach
        raises or is not callable the candidate STAYS TRACKED, a named error is
        raised, ``VAEDecode`` never runs, and outer ``finally`` teardown retries
        it best-effort and releases the lease. Retaining a patcher we failed to
        release is the point -- dropping it would leave it resident with nobody
        holding a handle to try again.
        """
        from . import wrapper_bridge as _wb

        candidates = []
        seen = set()
        for candidate in (owners.get("ade_model"),
                          (owners.get("base_model") or (None,))[0]):
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            candidates.append(candidate)

        # Every DISTINCT list that tracks patchers. `prepare` hands back
        # `self._patchers` itself, so these are usually one object -- but a
        # caller that copied it must not keep a released identity alive.
        lists = []
        for tracked in (owners.get("patchers"), self._patchers):
            if isinstance(tracked, list) and not any(
                    tracked is existing for existing in lists):
                lists.append(tracked)

        for candidate in candidates:
            detach = getattr(candidate, "detach", None)
            if not callable(detach):
                raise _wb.GraphExecutionError(
                    "%s: a sampling patcher (%s) has no callable detach() -- "
                    "refusing to decode with it still resident. It stays "
                    "tracked so teardown can try again."
                    % (self.name, type(candidate).__name__))
            try:
                detach(unpatch_all=True)
            except Exception as exc:  # noqa: BLE001 -- surfaced NAMED
                raise _wb.GraphExecutionError(
                    "%s: detaching a sampling patcher (%s) raised %s: %s -- "
                    "refusing to decode. It stays tracked so teardown can try "
                    "again." % (self.name, type(candidate).__name__,
                                type(exc).__name__, exc))
            for tracked in lists:
                for index in range(len(tracked) - 1, -1, -1):
                    if tracked[index] is candidate:
                        del tracked[index]

        owners.pop("ade_model", None)
        owners.pop("base_model", None)

    # ---- canonicalize ---------------------------------------------------- #
    def canonicalize(self, raw, request, profile):
        """The silent CanonicalClip dict, with silence and CANVAS both proved.

        ONE probe, fed to both checks. The exact-512x288 refusal is what earns
        the composite's right to trust the declared delivery mode and launch no
        redundant input probe of its own: the clean-Lanczos path does no
        aspect-ratio pad or crop, so a source that is not exactly 16:9 would be
        silently distorted on the way to 1920x1080. Asserting it HERE, at the
        seam where the mode is stamped, is what makes that safe.
        """
        raw = raw or {}
        path = raw.get("out_path", "")
        if path:
            fields = ffprobe_clip_fields(path)
            validate_silent_clip_contract(fields, self.target_fps)
            width = int(fields.get("width") or 0)
            height = int(fields.get("height") or 0)
            if width != GHOST_CANVAS_W or height != GHOST_CANVAS_H:
                raise RuntimeError(
                    "%s produced a %dx%d clip; this lane delivers by clean "
                    "full-frame Lanczos with no pad or crop, so only an exact "
                    "%dx%d source can be enlarged to %dx%d without distorting "
                    "it" % (self.name, width, height, GHOST_CANVAS_W,
                            GHOST_CANVAS_H, GHOST_DELIVERY_W, GHOST_DELIVERY_H))

        clip = self._clip_from_raw(raw, request)
        for key in ("model_frame_count", "cadence_mode",
                    "cadence_source_frame_count",
                    "cadence_delivered_frame_count", "cadence_tail_trim",
                    "delivery_scale_mode"):
            if key in raw:
                clip[key] = raw[key]
        return clip

    def _clip_from_raw(self, raw, request):
        """The self-declared clip dict :meth:`canonicalize` returns."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": (get("shot_id") or get("request_id")
                        or ("%s_clip" % self.name)),
            "type": "video",
            "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,     # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            "vram_peak_mb": raw.get("vram_peak_mb"),
            "recipe": raw.get("recipe"),
            "native_frame_count": raw.get("native_frame_count"),
            "extension_mode": raw.get("extension_mode"),
            "render_canvas": raw.get("render_canvas"),
        }


def _sha8(text) -> str:
    import hashlib
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:8]


__all__ = [
    "GhostSignalEngine", "GhostCadenceError",
    "GHOST_RECIPE_RECEIPT", "GHOST_CHECKPOINT_NAME", "GHOST_MOTION_MODULE_NAME",
    "GHOST_CHECKPOINT_CATEGORY", "GHOST_MOTION_CATEGORY",
    "GHOST_CANVAS_W", "GHOST_CANVAS_H",
    "GHOST_DELIVERY_W", "GHOST_DELIVERY_H", "GHOST_DELIVERY_SCALE_MODE",
    "GHOST_STEPS", "GHOST_CFG", "GHOST_SAMPLER_NAME", "GHOST_SCHEDULER",
    "GHOST_DENOISE", "GHOST_BETA_SCHEDULE",
    "GHOST_CONTEXT_LENGTH", "GHOST_CONTEXT_OVERLAP",
    "GHOST_CONTEXT_FUSE_METHOD", "GHOST_CONTEXT_USE_ON_EQUAL_LENGTH",
    "GHOST_CONTEXT_START_PERCENT", "GHOST_CONTEXT_GUARANTEE_STEPS",
    "GHOST_SOURCE_FLOOR", "GHOST_SOURCE_FPS", "GHOST_TARGET_FPS",
    "GHOST_NODE_CANDIDATES", "GHOST_ADE_OMITTED_SOCKETS",
    "GHOST_CONTEXT_OMITTED_SOCKETS",
    "NODE_CKPT", "NODE_POSITIVE", "NODE_NEGATIVE", "NODE_CONTEXT", "NODE_ADE",
    "NODE_LATENT", "NODE_SAMPLER", "NODE_DECODE",
    "ghost_unique_source_count", "ghost_source_request",
    "ghost_hold2_selector", "ghost_cadence_receipts",
]
