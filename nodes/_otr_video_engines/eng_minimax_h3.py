"""MiniMax H3 (33.1B packed AV DiT) -- the SHARED implementation module.

TWO ADAPTERS, TWO INTERNAL IDS, ONE IMPLEMENTATION. This file registers both
halves of the MiniMax H3 stack, and they were built one lane apart:

* ``minimax_h3_video`` (public ``h3_low_video``) -- LANE 19. The FL2VA route:
  a still pinned as the FIRST FRAME, no audio anywhere in its graph.
* ``minimax_h3_audio_in`` (public ``h3_low_audio_in``) -- LANE 20. The REF2VA
  route: a reference portrait plus the beat's own audio, on a DIFFERENT 21 GB
  DiT, loading an audio VAE it uses only to ENCODE that reference.

They share :class:`_MiniMaxH3Base` and differ in five places -- the DiT, the
conditioner node, the graph, the family/inputs, and CONTINUITY. **Two internal
ids, never one.** One internal engine cannot carry both modes, and two public
ids mapping to ONE internal id collapses ``public_engines._INTERNAL_TO_PUBLIC``
and trips its module-scope bijection assert at IMPORT time, which empties most
of the ComfyUI menu rather than failing one lane cleanly (lesson L5).

WHAT THIS ENGINE IS. H3 is the first LOCAL engine that natively produces AUDIO:
its latent is a NestedTensor PAIR -- video ``[B,24,T,H/16,W/16]`` and audio
``[B,32,2,T40]`` -- and the installed core nodes decode each half with its own
VAE. **NEITHER lane decodes the audio half.** No ``VAEDecodeAudio`` exists in
either graph, so both emitted clips are silent BY CONSTRUCTION as well as by
proof (V-1: only ``OTR_MasterAudioMux`` ever adds audio). Note the distinction
lane 20 makes easy to get wrong: it LOADS an audio VAE and still emits silence,
because that VAE encodes the reference audio into the conditioning and decodes
nothing. Neither construction is taken on trust; ``canonicalize`` ffprobes the
lane's OWN emitted file before writing ``has_audio: False``, because on this
engine that literal is the one field in the roster that could be a lie.

THE TWO NUMBERS THAT MAKE THIS LANE DIFFERENT
---------------------------------------------
**1. The frame grid is the model's, and it is not the canvas's.** The installed
node (``comfy_extras/nodes_minimax_h3.py``) snaps every request with::

    FPS = 24
    def align_frame_count(n):
        while n % 17 != 5:
            n += 1

so the legal MODEL lengths are ``17k + 5`` and the node's own ``length`` widget
documents the trained band as ~124-362. :data:`H3_MODEL_RUNGS` is DERIVED from
that rule here rather than transcribed from a document -- ``tests/
test_minimax_h3_video.py`` re-derives it against a copy of the node's own
function, so a model that moves its grid fails a test instead of silently
rendering a different length than the ledger planned (lesson L3).

**2. The model runs at 24 fps and the canvas is 25.** A 24 fps model DECLARES
the canvas rate and CONVERTS at delivery; it never relabels, because the local
encoder can only LABEL a rate (``wrapper_bridge`` puts ``-r`` before
``-i pipe:0``) and never resample. Relabelling 24 fps frames as 25 makes every
beat ~4% short -- the mouth slides ~320 ms over 8 s. So the contract is
published in CANVAS frames (:data:`H3_CANVAS_RUNGS`), and the decoded uint8
batch is remapped through a NEAREST-SOURCE-FRAME integer index map immediately
before ``encode_frames_to_silent_mp4``. 192 model frames = exactly 8.000 s =
200 canvas frames, and that identity is asserted rather than assumed.

THE CANVAS, AND WHY IT IS NOT THE TRAINED ONE. The module encodes a trained
shape -- ``BASE_SHORT_EDGE = 768``, ``MAX_PIXELS = 768*1344`` -- but
``adapt_canvas()`` is called at exactly ONE place, inside
``MiniMaxH3ReferenceToVideo``, to size REFERENCE media. It does not rewrite the
generation canvas, so the generation canvas is genuinely the caller's and those
numbers describe what the model was TRAINED at, not what the node enforces.
Both lanes declare **864x480** and the reasoning is in :attr:`render_canvas`.

BOOT. Sage silently turns H3 output into noise, video and audio, with no error
(Comfy-Org/ComfyUI#15263; the per-model KJ probe FAILED on sm_120), so BOTH
lanes are in the preflight matrix's ``SAGE_SENSITIVE`` set and refuse by name
before any weight resolves. Both also REQUIRE the named ``h3`` boot contract,
which they may do because neither has ever shipped under ``default``.

Cold-import clean (V-12): module scope imports the stdlib and the dep-free
shared helpers only; torch / numpy / PIL / ffmpeg / the ComfyUI node registry
are imported LAZILY inside ``load`` / ``render_clip``. UTF-8, no BOM, ASCII-only.

Config (env). The RECIPE is frozen in code (:data:`H3_RECIPE`) -- a production
episode is submitted to an already-booted server, so a knob exported at launch
cannot bind the work (PBUG-20260723-02). Only the loader tokens and the
deprecated-style directory pins are environment-readable:

* ``OTR_MINIMAX_H3_UNET_NAME`` / ``_CLIP_NAME`` / ``_VAE_NAME`` -- loader
  basenames, resolved through ComfyUI ``folder_paths`` exactly as the loader
  nodes resolve them.
* ``OTR_MINIMAX_H3_UNET_DIR`` / ``_CLIP_DIR`` / ``_VAE_DIR`` -- preflight-only
  directory probes, inherited from ``WanInitImageMixin._resolve_model_file``.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from . import wan_shared as _WS
from .._otr_shared.role_compat import ROLES
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import (
    CONTINUITY_SOFT_REFERENCE, CONTINUITY_STRICT_FIRST_FRAME, FrameContract)
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_LOG = logging.getLogger("OTR.video.minimax_h3")

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "This engine takes no negative prompt at all, so state every requirement "
    "positively. Name the subject, then one action and its speed as flowing prose, "
    "never a keyword list. Do not restate the set."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
CONFIG AS SHIPPED: MiniMax H3 video. It accepts NO negative prompt -- not an
inert one, none at all. Frame counts sit on the installed node's own grid
(``n % 17 == 5``), the MODEL runs at 24 fps against the OTR timeline's 25 fps
canvas, and the node's ``length`` tooltip declares a trained band of ~124-362.

"NO NEGATIVE PROMPT AT ALL" IS A DIFFERENT FACT FROM "THE NEGATIVE IS INERT", and
this is the one engine where the distinction matters. On ``flux_gen1``,
``ltx_video``, ``ltx_av`` and ``fastwan_8gb`` a negative is accepted and then not
consulted, because cfg 1.0 has no unconditional branch to evaluate it against;
here there is no input field to populate in the first place. For a writer the
practical instruction is identical, which is why the directive's remaining clauses
match those lanes. For a LEDGER it is not identical: a per-row record claiming a
negative conditioned a MiniMax pixel is not merely misleading, it is describing a
parameter that does not exist on this engine. That is worth remembering against
D-BIS finding 4, which is about the visual ledger not yet being able to prove
which negative conditioned which pixel -- on this lane the honest value is
"absent", never an empty string.

THE 24/25 FPS SPLIT IS NOT A PHRASING CONCERN and deliberately stays out of the
directive: it is resolved by the frame-grid code above, and rule 4 of the schema
is that a directive describes HOW to phrase, never what the scene contains or how
the clip is assembled. Recorded here only so a reader does not go looking for it
in the 240 chars.

EXTERNAL RESEARCH (2026-08-17, web lookup -- allowed per the operator's
2026-08-15 ruling, the RSS precedent):
  * **HOSTED IS NOT LOCAL, AND THIS IS THE ENGINE WHERE IT BITES.** Published
    MiniMax/Hailuo guides say the hosted UI DOES accept a negative prompt (to be
    used sparingly). Our node does NOT -- verified twice by reading this file:
    there is no negative field, parameter or payload key anywhere in the
    conditioner nodes, the graph build, or the request body. So "takes no negative
    prompt at all" is a true statement ABOUT THIS ADAPTER, and a reader who checks
    it against a hosted-product guide will think it is wrong. It is not; they are
    describing a different surface. If this adapter ever gains a negative input,
    this directive is the first thing to revisit.
  * ADOPTED INTO THE DIRECTIVE: guides describe H3 as wanting "a script, not a
    checklist" -- its LLM backbone rewards narrative flow, and camera direction
    should be written as part of the action rather than as a keyword list. That is
    a genuine phrasing rule, it is checkable by reading a prompt, and it is why the
    directive now says "as flowing prose, never a keyword list".
  * Also upstream: do not stack camera moves in one short shot. The directive's
    "one action and its speed" covers it.

PROVENANCE: the DIRECTIVE was authored by the driver from this engine's shipped
configuration plus the five directive rules in the RESEARCH doc, then checked
against the external research above -- and the "flowing prose" clause was added
BECAUSE of it, so this is the one directive of the ten that the research changed.
It is still NOT a measured finding on OUR lane. Treat the string as a hypothesis
until the probe A/B runs at a fixed seed.
"""


# ---------------------------------------------------------------------------
# THE MODEL GRID -- derived from the installed node's own rule, never typed
# ---------------------------------------------------------------------------

#: The MODEL's frame rate (``nodes_minimax_h3.FPS``). NOT the canvas rate.
H3_MODEL_FPS = 24

#: The canvas rate the whole OTR timeline is expressed at.
H3_CANVAS_FPS = 25

#: ``align_frame_count`` walks UP to the next ``n % 17 == 5``. Both numbers are
#: read off the installed node; the step is also the ``length`` widget's step.
H3_GRID_STEP = 17
H3_GRID_OFFSET = 5

#: The trained band, from the node's own ``length`` tooltip ("trained range is
#: ~124-362, longer is untested"). It is a REAL floor, not advice: the lab's
#: only sub-floor render, ``h3_i2v_canonical_832x480_f107``, is recorded in
#: ``docs/evidence/video_evidence_manifest.json`` as
#: ``MEASURED_BELOW_RANGE_FAILURE``.
H3_TRAINED_MIN_MODEL_FRAMES = 124
H3_TRAINED_MAX_MODEL_FRAMES = 362


def align_frame_count(n: int) -> int:
    """The installed node's grid rule: walk UP to the next ``17k + 5``.

    A COPY of ``comfy_extras/nodes_minimax_h3.align_frame_count``, deliberately
    duplicated rather than imported: this module is cold-import clean and must
    answer the grid question on a CPU box with no ComfyUI at all (the planner
    and the preflight matrix both ask it). ``tests/test_minimax_h3_video.py``
    pins this against the node's own arithmetic so the copy cannot drift.
    """
    n = int(n)
    while n % H3_GRID_STEP != H3_GRID_OFFSET:
        n += 1
    return n


def model_rungs() -> tuple:
    """Every legal MODEL length inside the trained band, ascending.

    DERIVED from :func:`align_frame_count`, so the tuple is a consequence of the
    grid rule rather than a transcription of it.
    """
    first = align_frame_count(H3_TRAINED_MIN_MODEL_FRAMES)
    return tuple(range(first, H3_TRAINED_MAX_MODEL_FRAMES + 1, H3_GRID_STEP))


def canvas_frames_for_model(model_frames: int) -> int:
    """How many whole 25 fps frames fit in ``model_frames`` at 24 fps.

    FLOOR, not round, and the direction is the point: a clip lasts
    ``model_frames / 24`` seconds, and the canvas may only publish frames that
    exist inside that duration. Rounding up would invent a frame past the end of
    the render, which is the manufactured-frame failure the whole
    ``native_frame_count`` receipt exists to make visible.
    """
    return (int(model_frames) * H3_CANVAS_FPS) // H3_MODEL_FPS


def model_frames_for_canvas(canvas_frames: int):
    """The MODEL rung whose delivery is exactly ``canvas_frames``, or ``None``.

    The inverse of :func:`canvas_frames_for_model` over the legal rungs. Exact
    lookup, never arithmetic: the map is injective on this grid but not onto,
    so computing a model length back from a canvas length would be able to
    produce a number the node would then silently snap somewhere else.
    """
    for model in model_rungs():
        if canvas_frames_for_model(model) == int(canvas_frames):
            return model
    return None


def canvas_index_map(model_frames: int, canvas_frames: int) -> tuple:
    """Nearest-source-frame index map: canvas frame ``j`` -> model frame index.

    ``j`` sits at ``j / 25`` seconds, which is model frame ``j * 24 / 25``. The
    nearest integer source frame is taken, clamped to the last rendered frame.

    INTEGER ARITHMETIC ON PURPOSE. ``(48*j + 25) // 50`` is exactly
    ``round(j * 24 / 25)`` with no float in it. Two reasons, and the second is
    the one that bites: a float path would make the map depend on binary
    rounding at the tail of a 377-frame clip, and Python's ``round`` is
    half-to-EVEN, so a tie would resolve differently at even and odd indices.
    (There are no ties here -- ``j * 24 / 25`` is never an exact half for
    integer ``j`` -- but a map whose correctness rests on that is a map that
    breaks the day the rates change.)
    """
    last = int(model_frames) - 1
    return tuple(min(last, (48 * j + 25) // 50) for j in range(int(canvas_frames)))


#: The legal render lengths this adapter PUBLISHES, in CANVAS frames. Derived,
#: so the contract and the grid can never disagree: 124..362 model frames on the
#: 17k+5 grid become 129..377 canvas frames at 25 fps.
H3_MODEL_RUNGS = model_rungs()
H3_CANVAS_RUNGS = tuple(canvas_frames_for_model(m) for m in H3_MODEL_RUNGS)


# ---------------------------------------------------------------------------
# THE FROZEN RECIPE + the loader tokens
# ---------------------------------------------------------------------------

#: The FL2VA (first-frame) DiT. The pruned int8 convrot repack is the whole
#: reason a 33.1B model has anything to do with a 16 GB card.
_H3_DEFAULT_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
#: The Qwen3-VL-32B multimodal encoder H3 conditions on, NVFP4-AWQ repack.
_H3_DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
#: The VIDEO VAE. The audio VAE is deliberately absent from this lane -- see the
#: module docstring; it belongs to lane 20, which encodes reference audio.
_H3_DEFAULT_VAE = "minimax_h3_video_vae_fp16.safetensors"

#: The REF2VA DiT -- lane 20's, and a DIFFERENT 21 GB repack from lane 19's.
#: The two are the same size and differ by one token in the name, which is
#: exactly the kind of pair a single hardcoded default gets wrong silently, so
#: each adapter names its own through ``_UNET_DEFAULT``.
_H3_DEFAULT_UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
#: The AUDIO VAE. Lane 20 only: ``MiniMaxH3ReferenceToVideo`` requires it to
#: encode reference audio. Lane 19 has no audio in its graph at all.
_H3_DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

#: ``(env_name_suffix, folder_paths categories, default token, min bytes)`` for
#: the weights BOTH lanes load. The byte floors distinguish the real artifact
#: from a truncated download: a partial 21 GB fetch otherwise reaches the loader
#: and the operator gets a deep safetensors traceback instead of the named
#: refusal that tells them to re-fetch. Each floor is ~80% of the real file, so
#: it catches truncation without pinning a repack's exact size.
#:
#: The UNET row's token is a PLACEHOLDER replaced per adapter from
#: ``_UNET_DEFAULT`` -- see :meth:`_MiniMaxH3Base._weight_rows`.
_H3_WEIGHTS = (
    ("UNET", ("diffusion_models", "unet"), None, 16 * 1024 ** 3),
    ("CLIP", ("text_encoders", "clip"), _H3_DEFAULT_CLIP, 12 * 1024 ** 3),
    ("VAE", ("vae",), _H3_DEFAULT_VAE, 4 * 1024 ** 3),
)

#: Lane 20's extra row, appended by the audio-in adapter only. Listed here
#: rather than inside that class so every weight this module can load is in one
#: place with its floor beside it.
_H3_AUDIO_VAE_ROW = ("AUDIO_VAE", ("vae",), _H3_DEFAULT_AUDIO_VAE,
                     256 * 1024 ** 2)

#: FROZEN IN CODE, and it is the lab's hash-pinned topology values, not a
#: preference: ``res_multistep`` / ``simple`` / 20 steps / denoise 1.0 /
#: ``weight_dtype=default`` are the settings every passing MiniMax H3 receipt on
#: this box was measured under. No env channel binds them -- a production
#: episode is submitted to an already-booted server, so a launch-time export
#: cannot reach the work anyway (PBUG-20260723-02), and the operator's standing
#: ruling is that a measured recipe is not on the table.
H3_RECIPE = {
    "sampler": "res_multistep",
    "scheduler": "simple",
    "steps": 20,
    "denoise": 1.0,
    "weight_dtype": "default",
    "clip_type": "minimax",
    "clip_device": "default",
}

#: The recipe receipt stamped into every clip this lane emits.
H3_RECIPE_RECEIPT = "minimax_h3_fl2va_int8_res_multistep_20step_v1"

#: Written in THIS ENGINE'S OWN DIALECT (2026-08-27, motion bake-in): the
#: directive above demands the subject, then ONE action and its speed, as
#: flowing prose -- never a keyword list. The old value ("subtle natural
#: motion, cinematic light") was a keyword list, carried the damping word
#: "subtle", and named no action at all -- it violated the directive it sits
#: forty lines under. No speech words: this default serves the SILENT lane.
_H3_DEFAULT_PROMPT = ("the subject crosses the frame with steady purpose, "
                      "one hand raised in emphasis")


# ---------------------------------------------------------------------------
# The S1 per-model still plan
# ---------------------------------------------------------------------------
_H3_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry="full-frame macro, centered subject",
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "cinematic three-quarter framing, the subject shown "
                     "whole with clear space around it inside frame, "
                     "balanced composition"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 # VERBATIM the producer's layer-2 geometry constant. These
                 # strings are not prose to be improved: an adapter that
                 # paraphrases one silently asks the image producer for a
                 # different shot than the plan claims, and
                 # tests/test_still_plan_layer2_parity.py compares them for
                 # EQUALITY for exactly that reason. (The first draft of this
                 # row dropped "16:9" and was caught by it.)
                 framing_geometry=(
                     "cinematic medium shot, the character framed within a "
                     "wide 16:9 environment, full head and shoulders with "
                     "clear headroom inside frame, face unobstructed, "
                     "balanced landscape composition"),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=(
                     "in-character cinematic medium shot, head and shoulders, "
                     "face clearly visible, subject centred with natural "
                     "headroom above the head (never crop the top of the head)"),
                 style_tail_policy="full"),
)


class _MiniMaxH3Base(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """Everything the two H3 adapters share -- NOT registered, never selectable.

    LANE 20 (2026-08-12) split this out of the lane-19 class. The two adapters
    are the same 33.1B stack at the same canvas on the same boot with the same
    grid, and they differ in exactly five places: the DiT they load, the node
    they condition through, the graph they build, the family/inputs they declare
    and -- the one that is a real decision rather than a detail -- their
    CONTINUITY.

    The split is a refactor with no behaviour change for lane 19: every method
    below moved verbatim, and ``tests/test_minimax_h3_video.py`` still holds the
    whole lane-19 contract against the registered class through its MRO.

    Deliberately NOT a place for a per-lane value. Anything that differs between
    the two adapters belongs in the subclass, where the difference is visible; a
    base-class value with a subclass override is how one lane silently inherits
    the other's answer (which is the exact shape of L19's "copy the reasoning,
    not the shape").
    """

    #: Wide: the director mints a landscape init still for both lanes.
    render_aspect = "wide"

    #: THE RENDER CANVAS, DECLARED -- and this lane DECLARES one where all eight
    #: cheap lanes before it documented the channel INERT. The premise really
    #: does invert here, and the value was derived rather than inherited.
    #:
    #: WHY DECLARE AT ALL. Lanes 11-18 are procedural: they paint at exactly the
    #: size the request carries, so they have no canvas of their own and a
    #: declaration would only overrule ``OTR_VIDEO_LANDSCAPE_CANVAS`` (lesson
    #: L19). H3 is not like that. Its width/height inputs are ``step=32`` on
    #: every node, its latent is ``H/16 x W/16``, and its cost is superlinear in
    #: pixels on a card this render does not fit in twice -- a size handed in by
    #: an operator lever is a size that can silently take this lane over the
    #: 14.5 GiB gate. That is a canvas-dependent property, which is exactly the
    #: condition L19 names for declaring.
    #:
    #: WHY 864x480 AND NOT THE TRAINED SHAPE. State the surface (L7): 864x480 is
    #: NOT what H3 was trained at. ``adapt_canvas`` maps a 16:9 ask to 1344x768
    #: (short edge 768 under the 768*1344 area cap), and that IS the trained
    #: shape -- but it is called at one place only, to size REFERENCE media
    #: inside ``MiniMaxH3ReferenceToVideo``, and never rewrites the generation
    #: canvas. So the trained numbers describe training, not enforcement, and
    #: the choice is genuinely ours.
    #:
    #: The controlling receipts use H3's legal floor, not the older 90-frame
    #: probe: FL2VA at 124 model / 129 canvas frames, 864x480, measured 6,315 MB
    #: cold absolute on the local RTX 5080; the REF2VA sibling measured 6,678 MB
    #: at the same legal floor. ``low`` therefore describes the measured recipe
    #: allocation on that machine. It does NOT qualify a physical 8 GB GPU, and
    #: host RAM was not measured in those receipts. The canvas is /32-legal on
    #: both axes (27 x 15); tests pin it and the legal-frame conversion.
    render_canvas = (864, 480)

    still_plan = _H3_STILL_PLAN
    roles = ROLES
    default_roles = ()

    #: The DiT each lane loads -- the ONE weight that differs between them.
    #: Subclasses set it; there is deliberately no default, so a future third
    #: adapter cannot silently inherit whichever repack happened to be first.
    _UNET_DEFAULT = None

    #: MiniMax H3 Community License, plus written authorization from MiniMax
    #: dated 2026-08-07 (docs/H3_LICENSE_ATTESTATION.md). NOT OSI, so this is
    #: False: ``commercial_clean`` drives the release-gate warning and the
    #: release filename tag, never selection, and a conditional grant to one
    #: licensee is not a clean commercial license for a published open-source
    #: pipeline. The operator confirms at license review.
    commercial_clean = False
    requires_flag = None                  # registry IS the menu; no flag gate
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = H3_CANVAS_FPS

    #: BOTH lanes REQUIRE a named H3 boot contract, which they are allowed to do
    #: because neither has ever shipped under ``default`` (boot_contracts' own
    #: rule -- requiring a contract on a lane that already ships would regress
    #: it). ``h3`` is the measured 16 GB streaming boot; ``h3_8gb_lab`` is the
    #: physical-8-GB lab launch that emits no reserve. Sage-free is non-negotiable in both:
    #: Sage does not degrade H3, it replaces the output with noise and reports
    #: success.
    compatible_boot_contracts = ("h3", "h3_8gb_lab")

    #: Terminal node of the graph (its IMAGE output becomes the clip).
    _TERMINAL = "decode"

    #: The graph spine both lanes share. The CONDITIONER is per-lane and lives
    #: in ``_EXTRA_NODE_CANDIDATES`` beside the inputs only that lane stages.
    _CORE_NODE_CANDIDATES = {
        "unet": ("UNETLoader",),
        "clip": ("CLIPLoader",),
        "vae": ("VAELoader",),
        "guider": ("BasicGuider",),
        "noise": ("RandomNoise",),
        "sampler": ("KSamplerSelect",),
        "sched": ("BasicScheduler",),
        "sample": ("SamplerCustomAdvanced",),
        "decode": ("VAEDecode",),
    }
    _EXTRA_NODE_CANDIDATES: dict = {}

    #: The recipe receipt stamped into this lane's clips.
    _RECIPE_RECEIPT = ""

    #: Fallback prompt when the beat carries none.
    _DEFAULT_PROMPT = _H3_DEFAULT_PROMPT

    # ---- loader tokens ------------------------------------------------- #
    def _weight_rows(self):
        """The weight table for THIS adapter: the shared rows with the UNET's
        token filled in from ``_UNET_DEFAULT``, plus whatever the lane adds.

        One seam, so preflight, the byte floors, the identity and the graph all
        read the same list and a lane cannot check one set of weights while
        loading another.
        """
        if not self._UNET_DEFAULT:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "%s declares no _UNET_DEFAULT, so it does not say which H3 DiT "
                "it loads" % self.name, kind="video")
        rows = []
        for label, cats, default, floor in _H3_WEIGHTS:
            if label == "UNET":
                default = self._UNET_DEFAULT
            rows.append((label, cats, default, floor))
        return tuple(rows)

    def _token_for(self, label, default):
        return otr_env.get("OTR_MINIMAX_H3_%s_NAME" % label) or default

    def _weight_paths(self):
        """``{label: (token, resolved path or None)}`` for all three weights.

        Resolution goes through ``WanInitImageMixin._resolve_model_file``, which
        asks ComfyUI ``folder_paths`` for the bare loader token -- the same
        question ``UNETLoader`` / ``CLIPLoader`` / ``VAELoader`` will ask, so a
        weight registered through ``extra_model_paths.yaml`` resolves here and a
        preflight cannot pass on a file the loader will not find (lesson L1).
        """
        out = {}
        for label, categories, default, _floor in self._weight_rows():
            token = self._token_for(label, default)
            out[label] = (token, self._resolve_model_file(
                categories, token, "OTR_MINIMAX_H3_%s_DIR" % label))
        return out

    def _ckpt_path(self):
        """The PRIMARY weight (the DiT), or ``None``. Resolved via
        ``folder_paths`` through :meth:`_weight_paths`."""
        return self._weight_paths()["UNET"][1]

    def _installed(self):
        """Are all three weights present? A PREDICATE -- it may not raise."""
        try:
            return all(path for _token, path in self._weight_paths().values())
        except Exception:  # noqa: BLE001 -- a predicate answers, never raises
            return False

    def session_identity(self):
        """What this adapter's HANDLES are -- engine, recipe, the three weights.

        REQUIRED, not optional, and the roster gate says exactly why: this lane
        SPLITS (a discrete menu topping out at 15 s cannot cover a 30 s beat in
        one call) and it HOLDS local handles, so ``BeatSession.open()`` would
        refuse every multi-segment beat with ``SessionIdentityUnavailable`` --
        at render time, after the script, the stills and the audio were paid
        for. On a lane where one segment is minutes of GPU, that is the most
        expensive possible place to learn it.

        PRE-LOAD STABLE, like its siblings: read once before the weights land,
        again after ``prepare()``, and before every segment, so it may only
        describe what does not change across the load. Per-segment state --
        prompt, seed, frame count, canvas -- is deliberately absent.

        Not cached: the job is to notice a weight that MOVED, so the receipts
        are re-stat'ed on every ask (three stats, never a hash of 21 GB).
        """
        # ``self._recipe_receipt()``, never the module constant. This read
        # ``H3_RECIPE_RECEIPT`` directly, which is lane 19's receipt -- so the
        # moment a second adapter shared this method, lane 20's identity
        # described lane 19's recipe. It stayed DISTINCT between the lanes only
        # because the DiT token below differs, so nothing would have reused the
        # wrong session; it would simply have been a receipt that lies about
        # which recipe produced the handles. Exactly the hardcoded-per-lane
        # value in a shared base that this class's own docstring warns about.
        parts = [self.name, self._recipe_receipt()]
        for label, _cats, _default, _floor in self._weight_rows():
            token, path = self._weight_paths()[label]
            parts.append(token)
            parts.append(repr(self._wan_file_receipt(path or "")))
        return tuple(parts)

    # ---- graph --------------------------------------------------------- #
    def _node_candidates(self):
        """The ComfyUI classes this lane's graph needs, by logical id.

        ``MiniMaxH3ImageToVideo`` is the only H3-specific one; the rest are
        core. Gating them at PREFLIGHT rather than at ``load()`` is lane 8's
        lesson -- a missing class must not surface mid-render, after a 21 GB
        checkpoint has already been paid for.

        **`MiniMaxH3SigmaShift` IS DELIBERATELY ABSENT, and this note exists so
        nobody adds it as a missing piece.** It is one of the four H3 classes the
        install ships, which makes its absence look like an oversight; it is not.
        The lab topology that produced every passing H3 receipt on this box names
        it in `forbidden_class_types` and records "No MiniMaxH3SigmaShift or
        SageAttention patch is inserted" as an intentional divergence. Omitting
        it means the flow shifts come from the checkpoint's own
        `sampling_settings`, which is what those receipts measured. Adding the
        node would silently re-tune the sigma schedule away from the only
        configuration this lane has evidence for -- and the operator's standing
        rule is that a measured recipe is not on the table.
        """
        out = dict(self._CORE_NODE_CANDIDATES)
        out.update(self._EXTRA_NODE_CANDIDATES)
        return out

    def _recipe_receipt(self):
        return self._RECIPE_RECEIPT

    def _sampler_spine(self, plan):
        """The loaders + sampler both H3 graphs share, in run_graph format.

        Everything here is IDENTICAL between the two lanes -- the same three
        loaders, the same frozen recipe, the same guider/noise/scheduler/sampler
        chain, the same single ``VAEDecode`` terminal. What differs is the
        CONDITIONER, which each lane supplies from ``_conditioner_nodes`` and
        which this spine reads through the ``h3`` node's two outputs
        (conditioning at slot 0, the AV latent at slot 1).

        NEITHER lane decodes the audio half. Both graphs are the lab's pinned
        topology minus ``VAEDecodeAudio`` / ``CreateVideo`` / ``SaveVideo``: OTR
        takes the decoded IMAGE batch and writes its own always-silent bt709
        mp4, which is what makes V-1 a property of the GRAPH and not only of a
        check. Lane 20 loads the audio VAE because its CONDITIONER needs it to
        encode reference audio -- never to decode any.
        """
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._weight_paths()
        return {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": names["UNET"][0],
                                "weight_dtype": H3_RECIPE["weight_dtype"]}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["CLIP"][0],
                                "type": H3_RECIPE["clip_type"],
                                "device": H3_RECIPE["clip_device"]}},
            "vae": {"class": "vae",
                    "inputs": {"vae_name": names["VAE"][0]}},
            "guider": {"class": "guider",
                       "inputs": {"model": W("unet", 0),
                                  "conditioning": W("h3", 0)}},
            "noise": {"class": "noise",
                      "inputs": {"noise_seed": int(plan.get("seed", 0))}},
            "sampler": {"class": "sampler",
                        "inputs": {"sampler_name": H3_RECIPE["sampler"]}},
            "sched": {"class": "sched",
                      "inputs": {"model": W("unet", 0),
                                 "scheduler": H3_RECIPE["scheduler"],
                                 "steps": H3_RECIPE["steps"],
                                 "denoise": H3_RECIPE["denoise"]}},
            "sample": {"class": "sample",
                       "inputs": {"noise": W("noise", 0),
                                  "guider": W("guider", 0),
                                  "sampler": W("sampler", 0),
                                  "sigmas": W("sched", 0),
                                  "latent_image": W("h3", 1)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("sample", 0), "vae": W("vae", 0)}},
        }

    def _build_graph(self, request, staged, plan, model_length, width, height):
        """Spine + this lane's conditioner. ``staged`` is whatever the lane's
        ``_stage_inputs`` produced (a basename, or a dict of them)."""
        graph = self._sampler_spine(plan)
        graph.update(self._conditioner_nodes(
            request, staged, plan, model_length, width, height))
        return graph

    # ---- per-lane seams (subclasses implement) ------------------------- #
    def _assert_required_inputs(self, plan):
        """Refuse a request missing what this lane needs, BEFORE anything is
        staged and long before the GPU. Subclasses implement."""
        raise NotImplementedError

    def _stage_inputs(self, request, plan, width, height):
        """Put this lane's inputs where ComfyUI's loaders can read them and
        return whatever ``_conditioner_nodes`` needs to wire them."""
        raise NotImplementedError

    def _conditioner_nodes(self, request, staged, plan, model_length,
                           width, height):
        """The ``h3`` conditioner node plus any input-loader nodes it wires.
        Must expose conditioning at ``h3`` slot 0 and the AV latent at slot 1."""
        raise NotImplementedError

    def _prompt_for(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        return get("text_prompt") or self._DEFAULT_PROMPT

    @staticmethod
    def _ref_path(ref):
        """A filesystem path out of an ``audio_ref`` that may be a bare string
        OR a mapping carrying a ``path`` key (the AudioRef shape). Same helper
        shape ``eng_ltx_av`` uses for the same field."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _build_render_request(self, request):
        """The shared Wan-style request PLUS ``audio_path``.

        ``WanInitImageMixin._build_render_request`` has no audio field -- the
        WAN lanes have no audio input -- so the audio-in lane would silently see
        ``None`` and refuse every beat. Added on the BASE rather than only on
        lane 20 so the two lanes' plans have one shape, and it is inert for lane
        19 (a request with no ``audio_ref`` yields "").
        """
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        plan = super()._build_render_request(request)
        plan["audio_path"] = self._ref_path(get("audio_ref"))
        plan["text_prompt"] = get("text_prompt") or ""
        return plan

    # ---- preflight ----------------------------------------------------- #
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail CLOSED, cheapest refusal first.

        ORDER IS THE CONTRACT, and each step is here for a reason:

        1. **Sage.** It costs nothing to check and everything to miss: Sage
           routes H3's DiT into an int8 QK path that emits pure noise, video and
           audio, with no error at all. A refusal before any weight resolves
           beats one after 21 GB has loaded.
        2. **The boot contract**, for the same reason one level up -- this is
           not fixable at render time, so learning it here saves a whole beat.
        3. **The weights**, by the loader's own resolution, then their byte
           floors so a truncated fetch is named rather than traced.
        4. **The node classes**, collecting EVERY miss before raising: naming
           them one at a time turns a fresh install into a sequence of failed
           renders.
        """
        _MC.assert_sage_not_patched(self.name, self.family)
        self._assert_boot_contract(profile)
        paths = self._weight_paths()
        missing = ["%s=%s" % (label, paths[label][0])
                   for label, _cats, _default, _floor in self._weight_rows()
                   if not paths[label][1]]
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s weight(s) not found: %s -- drop them in the "
                "matching models/ folder or register it in "
                "extra_model_paths.yaml (the OTR_MINIMAX_H3_*_NAME variables "
                "only name a file, they cannot make the loader find one)"
                % (self.name, ", ".join(missing)), kind="video")
        for label, _cats, _default, floor in self._weight_rows():
            token, path = paths[label]
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s weight %r is only %d bytes (< the %d floor) -- that is "
                    "a truncated or wrong file, not the H3 artifact; re-fetch "
                    "it" % (self.name, label, token, size, floor),
                    kind="video")
        from . import wrapper_bridge as _wb
        mapping = _wb.node_class_mappings()
        absent = []
        for _logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 -- collect every missing class
                absent.append("/".join(candidates))
        if absent:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s (MiniMax H3 "
                "needs a ComfyUI carrying comfy_extras/nodes_minimax_h3.py -- "
                "update ComfyUI and restart)"
                % (self.name, ", ".join(absent)), kind="video")
        return self.name

    def _assert_boot_contract(self, profile):
        """The profile's contract must be one this lane is proven on, AND the
        running server must actually be in that state.

        Two checks, deliberately separate (the lane-2 pattern). The first is
        pure and always runs. The second probes ``comfy.cli_args`` -- what this
        process was really started with -- because a check that reads the same
        config the launcher was supposed to honour cannot tell "applied" from
        "written down" (lesson L6).

        THE SECOND CHECK ASSERTS THE EFFECTIVE H3 CONTRACT, not the stripped
        policy's ``default``. On a real episode leg the policy every adapter sees
        is rebuilt from the ledger's ``video`` section and carries NO ``launch``
        key, so the live server is matched against only this engine's compatible
        H3 contracts. The chosen contract is then checked knob by knob.
        """
        from .._otr_shared import boot_contracts as _bc
        problems = _bc.check_engine_against_profile(self, profile or {})
        if problems:
            raise EngineUnusable(
                self.name, self.family,
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "; ".join(problems), kind="video")
        _bc.assert_running_server(
            _bc.contract_for_engine_runtime(self, profile or {}))

    # ---- residency ----------------------------------------------------- #
    def load(self):
        """Fail CLOSED until installed, then resolve node CLASSES only.

        No weights load here. The 21 GB DiT, the 15 GB encoder and the VAE are
        loaded by their loader nodes inside the segment graph, which is what
        lets ``free_after_use`` drop the encoder before the sampler runs -- the
        single biggest reason this stack fits at all.
        """
        if not self._installed():
            missing = ", ".join(
                "%s=%s" % (label, token)
                for label, (token, path) in sorted(self._weight_paths().items())
                if not path)
            raise RuntimeError(
                "%s not installed: missing %s" % (self.name, missing))
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    # ---- render -------------------------------------------------------- #
    def render_clip(self, request, prepared):
        """One FL2VA clip: stage the init still, sample, convert 24 -> 25, and
        encode a SILENT bt709 mp4 whose silence is then PROVEN."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)
        self._assert_required_inputs(plan)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        # THE CANVAS IS CHECKED BEFORE ANYTHING IS STAGED. Every H3 node takes
        # width/height at step=32 and the latent grid is H/16 x W/16, so an
        # off-grid canvas is not a slightly wrong render -- it is a latent the
        # model was never given. This is knowable from pure numbers, so it is
        # refused here rather than inside the graph.
        if int(width) % 32 or int(height) % 32:
            raise _wb.GraphExecutionError(
                "%s was handed a %dx%d canvas; both axes must be /32-legal "
                "(every H3 node declares width/height at step=32 and the latent "
                "is H/16 x W/16). This lane declares %dx%d."
                % (self.name, width, height,
                   self.render_canvas[0], self.render_canvas[1]))
        # THE LENGTH IS DECIDED BEFORE ANYTHING IS STAGED TOO (B4). The ask
        # arrives in CANVAS frames; the graph is driven in MODEL frames.
        target_frames = int(plan.get("target_frame_count") or 0)
        contract = self.frame_contract
        canvas_rung = contract.smallest_legal_at_least(target_frames)
        model_length = (None if canvas_rung is None
                        else model_frames_for_canvas(canvas_rung))
        if canvas_rung is None or model_length is None:
            raise _wb.GraphExecutionError(
                "%s was asked for %d canvas frame(s); its legal menu is "
                "%d..%d at 25 fps (the model's 17k+5 grid over its trained "
                "%d..%d band, converted from 24 fps). NO FALLBACK -- rendering "
                "a shorter clip and padding it back up is not a render of the "
                "ask, and on a chained lane the next segment would begin on a "
                "manufactured frame."
                % (self.name, target_frames,
                   H3_CANVAS_RUNGS[0], H3_CANVAS_RUNGS[-1],
                   H3_MODEL_RUNGS[0], H3_MODEL_RUNGS[-1]))
        staged = self._stage_inputs(request, plan, width, height)
        graph = self._build_graph(request, staged, plan, model_length,
                                  width, height)
        # free_after_use: the 15 GB Qwen3-VL encoder is freed the moment the
        # conditioning exists, before the sampler ever runs. `keep` holds the
        # DiT (retained for V-4 teardown) and the terminal.
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(
                graph, classes, free_after_use=True,
                keep={"unet", self._TERMINAL})
            images = results[self._TERMINAL][0]
        finally:
            render_peak = probe.stop()
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        out = results.get("unet") or ()
        model = out[0] if out else None
        if (model is not None and id(model) not in seen
                and callable(getattr(model, "detach", None))):
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        # THE PIPELINE INVARIANT, before any conversion. The graph was asked for
        # exactly `model_length` frames on the grid, so anything else means the
        # node snapped somewhere we did not plan for -- and a count mismatch
        # discovered after the 24 -> 25 remap would be indistinguishable from a
        # remap bug.
        if len(frames) != model_length:
            raise _wb.GraphExecutionError(
                "%s asked its graph for %d model frame(s) and decoded %d. NO "
                "FALLBACK -- padding the difference is how a render that did "
                "not happen gets counted as one."
                % (self.name, model_length, len(frames)))
        # THE 24 -> 25 CONVERSION, and it is this lane's real deliverable.
        # Nearest-source-frame resampling of the uint8 batch, immediately before
        # the encoder. Relabelling instead would make every beat ~4% short: the
        # local encoder can only LABEL a rate (`-r` precedes `-i pipe:0`), it
        # can never resample, so a 192-frame render tagged 25 fps would play in
        # 7.68 s while the ledger and the audio both expect 8.000 s.
        frames = frames[list(canvas_index_map(model_length, canvas_rung))]
        _LOG.info(
            "[OTR video] %s 24->25 conversion: %d model frame(s) at %d fps "
            "(%.3f s) -> %d canvas frame(s) at %d fps (%.3f s)",
            self.name, model_length, H3_MODEL_FPS, model_length / float(H3_MODEL_FPS),
            canvas_rung, H3_CANVAS_FPS, canvas_rung / float(H3_CANVAS_FPS))
        # THE TAIL TRIM, in REAL delivered frames. An ask off the menu renders
        # the next legal rung UP and drops the surplus in order -- no mirror, no
        # loop, no held frame. An ask below the shortest legal clip is left
        # alone: cutting under the declared minimum would invent a length the
        # contract forbids.
        native_frames = canvas_rung
        if canvas_rung > target_frames >= H3_CANVAS_RUNGS[0]:
            frames = frames[:target_frames]
            _LOG.info(
                "[OTR video] %s tail trim: delivered %d of %d canvas frame(s) "
                "(nearest legal rung at or above the %d-frame ask) @ %dx%d",
                self.name, len(frames), canvas_rung, target_frames,
                width, height)
        out_path = otr_engine_tmp_mp4("otr_%s_" % self.name)
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path,
                                                  self.target_fps)
        # M7: PROVE the silent bt709/yuv420p contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not otr_env.get("OTR_TEST_MODE"):
            _LOG.info(
                "[OTR video] %s VRAM render-phase peak %s MB @ %dx%d "
                "model_len=%d canvas_len=%d",
                self.name, render_peak, width, height, model_length, n)
        return {"out_path": path, "frame_count": n,
                "vram_peak_mb": render_peak,
                "recipe": self._recipe_receipt(),
                # NATIVE = EMITTED, AND `none`, AND HERE IS THE EXACT SURFACE
                # (L7), because 25/24 is not 1 and the number deserves saying.
                #
                # The index map SELECTS; it never synthesises. But 129 canvas
                # frames drawn from 124 source frames must show about one frame
                # in 24 twice -- 5 repeats at the base rung, 15 at f377. So the
                # clip does NOT carry 129 distinct source frames, and a reader
                # of this receipt should not conclude that it does.
                #
                # `native_frame_count = n` is nonetheless the only stamp that is
                # true here, and `acceptance.py` is what decides it: its
                # producer contract is that MANUFACTURED FRAMES ARE APPENDED AT
                # THE TAIL, so real frames are the prefix `[0, native)`. A rate
                # conversion's repeats are spread evenly through the clip, not
                # appended -- so stamping native=124 would tell the grader that
                # frames 124..128 are the manufactured ones, which is false, and
                # that same contract says an engine whose extension is not
                # tail-appended must not stamp these fields at all.
                #
                # Nothing here is manufactured under the definition the field
                # exists for: no frame is invented, no tail is mirrored, no beat
                # is padded to fake a length, and the DURATION is exact. A frame
                # held for one extra 1/25 s is a resample, which is the one
                # honest way to put 24 fps on a 25 fps timeline.
                "native_frame_count": n,
                "extension_mode": "none",
                "model_frame_count": model_length,
                "render_canvas": "%dx%d" % (int(width), int(height))}

    def canonicalize(self, raw, request, profile):
        """The silent CanonicalClip dict -- with silence PROVED here, on this
        lane's own emitted file, before the literal is written.

        Every other adapter writes ``has_audio: False`` because its model has no
        audio to emit. H3 does: its latent is a video+audio pair and the
        installed nodes ship a decoder for each half. NEITHER H3 lane wires
        ``VAEDecodeAudio``, so both files are silent by construction -- and note
        that lane 20 LOADS an audio VAE without decoding anything with it, which
        is exactly the sort of distinction a reader gets wrong. "By
        construction" is precisely the kind of claim that survives a graph edit
        it should not survive, and G5 exists because a declaration checking a
        declaration proves nothing (lesson L4). So the container is re-probed
        HERE, at the seam where the field is written, on BOTH lanes.
        """
        raw = raw or {}
        path = raw.get("out_path", "")
        if path:
            validate_silent_clip_contract(ffprobe_clip_fields(path),
                                          self.target_fps)
        clip = self._clip_from_raw(raw, request)
        clip["model_frame_count"] = raw.get("model_frame_count")
        return clip


@register
class MiniMaxH3VideoEngine(_MiniMaxH3Base):
    """``minimax_h3_video`` -- MiniMax H3 FL2VA, first frame to silent video.

    LANE 19. Conditions on a still pinned as the FIRST FRAME, on the fl2va DiT.
    No audio anywhere in its graph -- not decoded and not even loaded.
    """

    name = "minimax_h3_video"
    family = "image_to_video"
    _UNET_DEFAULT = _H3_DEFAULT_UNET
    _RECIPE_RECEIPT = H3_RECIPE_RECEIPT
    required_inputs = ("init_image",)

    _EXTRA_NODE_CANDIDATES = {
        "loadimage": ("LoadImage",),
        "h3": ("MiniMaxH3ImageToVideo",),
    }

    #: THE FRAME LADDER, PUBLISHED IN CANVAS FRAMES. ``discrete_frames`` is the
    #: 17k+5 model grid over the trained band, converted to 25 fps: 129 .. 377.
    #: Both tuples are derived from ``align_frame_count`` at import, so the menu
    #: cannot drift from the grid the node will actually snap to.
    #:
    #: ``allow_tail_trim`` is mandatory for a discrete menu (a fixed set of
    #: lengths cannot sum to an arbitrary beat) and it is honest here: the
    #: surplus is dropped in REAL delivered frames, in order, never mirrored.
    #:
    #: CONTINUITY -- decided, not defaulted, and NOT what lane 20 declares.
    #: ``MiniMaxH3ImageToVideo`` pins the supplied still as a keyframe at
    #: ``resolved_frame_index 0`` and the node's own docstring records that
    #: keyframe condition latents are "re-injected every step (never denoised)".
    #: That is a first-frame lock of the same class as
    #: ``LTXVImgToVideo(strength=1.0)`` on the incumbent chained lanes: a
    #: LATENT-space pin through a VAE round trip, not a byte copy of the source
    #: pixels. The lab's continuation experiment is the measurement --
    #: prior-clip-last to next-clip-first scored SSIM 0.900 / PSNR 33.4 dB and
    #: its seam gate passed. Declaring NONE instead would refuse chaining and
    #: turn every multi-segment H3 beat into jump cuts it does not need.
    frame_contract = FrameContract(
        discrete_frames=H3_CANVAS_RUNGS,
        native_fps=H3_CANVAS_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )

    def _assert_required_inputs(self, plan):
        from . import wrapper_bridge as _wb
        if not plan.get("init_image"):
            raise _wb.GraphExecutionError(
                "%s requires init_image (got %r)"
                % (self.name, plan.get("init_image")))

    def _stage_inputs(self, request, plan, width, height):
        return self._materialize_init_image(
            request, plan["init_image"], width, height)

    def _conditioner_nodes(self, request, staged, plan, model_length,
                           width, height):
        """``MiniMaxH3ImageToVideo`` with the still as ``first_frame``.

        ``last_frame`` is deliberately never wired. It is H3's first/LAST
        interpolation endpoint, which is a different capability from the
        first-frame chaining this lane declares -- supplying it would make the
        render interpolate toward a frame the coverage planner never chose.
        """
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        return {
            "loadimage": {"class": "loadimage", "inputs": {"image": staged}},
            "h3": {"class": "h3",
                   "inputs": {"clip": W("clip", 0), "vae": W("vae", 0),
                              "prompt": self._prompt_for(request),
                              "width": int(width), "height": int(height),
                              "length": int(model_length),
                              "first_frame": W("loadimage", 0)}},
        }


@register
class MiniMaxH3AudioInEngine(_MiniMaxH3Base):
    """``minimax_h3_audio_in`` -- MiniMax H3 REF2VA, audio-conditioned video.

    LANE 20. The sibling of :class:`MiniMaxH3VideoEngine` and a genuinely
    different capability, not a variant: it conditions on REFERENCE media --
    a portrait and the beat's own audio -- through
    ``MiniMaxH3ReferenceToVideo``, on the **ref2va** DiT rather than fl2va.

    THE AUDIO VAE IS LOADED, AND IT STILL DECODES NOTHING. ``audio_vae`` is a
    required input of the reference node, which uses it to ENCODE the reference
    audio into the conditioning. No ``VAEDecodeAudio`` exists in this graph
    either, so the emitted clip is silent by construction exactly like lane 19's
    and ``canonicalize`` re-probes it (V-1). "Loads an audio VAE" and "emits
    audio" are different claims and only the first is true here.
    """

    name = "minimax_h3_audio_in"
    #: The family the mouth policy and the ambient-audio router both read.
    #: ``audio_conditioned_video`` is the EXISTING family ``ltx_audio_in`` uses;
    #: minting a new one would put this lane outside the motion families the
    #: campaign's held-frame invariant checks (``scripts/otr_w45_campaign.py``)
    #: and make its frozen clips exempt, the trap the transplant plan names.
    family = "audio_conditioned_video"
    _UNET_DEFAULT = _H3_DEFAULT_UNET_REF2VA
    _RECIPE_RECEIPT = "minimax_h3_ref2va_int8_res_multistep_20step_v1"

    #: HARD-requires both: the reference node takes a portrait AND the audio it
    #: is meant to move to. A beat missing either is refused before staging.
    required_inputs = ("audio_ref", "init_image")

    _EXTRA_NODE_CANDIDATES = {
        "audiovae": ("VAELoader",),
        "loadimage": ("LoadImage",),
        "loadaudio": ("LoadAudio",),
        "h3": ("MiniMaxH3ReferenceToVideo",),
    }

    #: THE LAB-PROVEN ENVELOPE (2026-08-27), corrected from a first cut that
    #: said "shoulders and hands moving with the words". Broad hand acting is
    #: OUTSIDE what has been reviewed on this lane: H3 Ref2VA passed a human
    #: check at 5.17 s on seed 43 while the IDENTICAL prompt on seed 42 failed
    #: (`vram-recipe-lab/docs/ENVELOPE_LADDERS.md:30`), and the failure mode is
    #: the jaw, chin and collar RESHAPING once the head approaches yaw. So the
    #: demonstrated ceiling is one weight shift or forward lean plus a slight
    #: head tilt -- and hands, which cross the face, are the first danger.
    #:
    #: The lip anchor stays first and untouched: `<Audio 1>` and `<Picture 1>`
    #: are this model's own reference tokens, and the sync is the product.
    #: Seed sensitivity is real here -- a single failed take is not a verdict.
    _DEFAULT_PROMPT = (
        "a medium close shot of <Picture 1> speaking directly to camera, lip "
        "movements matching <Audio 1> precisely, one slight forward weight "
        "shift and a small head tilt, staying frontal, warm cinematic light")

    #: SAME ladder as lane 19 -- same model grid, same 24->25 conversion -- and
    #: a DIFFERENT continuity, which is the one contract decision this lane does
    #: not inherit.
    #:
    #: CONTINUITY IS ``soft_reference``, DECIDED. Lane 19 wires ``first_frame``,
    #: which ``MiniMaxH3ImageToVideo`` pins as a keyframe at
    #: ``resolved_frame_index 0`` and re-injects every step -- a real first-frame
    #: lock, so it earns STRICT. ``MiniMaxH3ReferenceToVideo`` has no
    #: ``first_frame`` input at all: its ``ref_images`` are IDENTITY references
    #: presented to the tokenizer as ``<Picture i>``, and nothing pins frame 0 to
    #: any of them. Claiming STRICT here would promise the coverage planner a
    #: seam this node cannot deliver, and the visible cost of a wrong strict
    #: claim is a jump at a join the plan said was seamless. A jump cut is
    #: honest; that is what SOFT means (same reasoning ``google_veo_video``
    #: records for its own endpoints).
    frame_contract = FrameContract(
        discrete_frames=H3_CANVAS_RUNGS,
        native_fps=H3_CANVAS_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )

    def _assert_required_inputs(self, plan):
        from . import wrapper_bridge as _wb
        if not plan.get("init_image"):
            raise _wb.GraphExecutionError(
                "%s requires init_image -- the reference node needs a portrait "
                "to present as <Picture 1> (got %r)"
                % (self.name, plan.get("init_image")))
        if not plan.get("audio_path"):
            raise _wb.GraphExecutionError(
                "%s requires audio_ref -- this is the AUDIO-CONDITIONED lane "
                "and without the beat's audio it would render an unconditioned "
                "clip while its receipt claimed lip-sync (got %r)"
                % (self.name, plan.get("audio_path")))

    def _stage_inputs(self, request, plan, width, height):
        """Stage the portrait and the audio where ComfyUI's loaders read them.

        The portrait goes through ``_materialize_init_image`` for the same
        reason every other lane does -- ONE uniform scale, never a silent
        stretch (N9). The audio is staged BYTE-FOR-BYTE: resampling or trimming
        it here would change the thing the model is being asked to lip-sync to,
        and the node's own encoder already resamples to the audio VAE's rate.
        """
        from . import wrapper_bridge as _wb
        return {
            "image": self._materialize_init_image(
                request, plan["init_image"], width, height),
            "audio": _wb.stage_into_comfy_input(plan["audio_path"]),
        }

    def _conditioner_nodes(self, request, staged, plan, model_length,
                           width, height):
        """``MiniMaxH3ReferenceToVideo`` with one picture and one audio.

        THE REFERENCE SOCKETS ARE ``COMFY_AUTOGROW_V3`` and take PLAIN DICTS
        here. In the lab's API graph they serialize dotted
        (``"ref_images.ref_image_0"``) because ComfyUI's prompt EXECUTOR
        flattens and reassembles them; this adapter calls the node class
        directly through ``wrapper_bridge``, whose ``run_graph`` invokes
        ``FUNCTION`` -- and a V3 node's ``FUNCTION`` is ``EXECUTE_NORMALIZED``,
        a plain passthrough to ``execute(**kwargs)``. So ``execute`` receives
        exactly the dict it iterates (``(ref_images or {}).values()``).
        ``_iter_wires`` recurses dicts, so the Wires inside resolve normally.

        ``ref_videos`` / ``ref_video_audios`` are deliberately never wired: this
        lane conditions a still portrait on a voice, and a reference VIDEO is a
        different capability with its own cost (reference tokens ride every
        sampling step).
        """
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._weight_paths()
        return {
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": names["AUDIO_VAE"][0]}},
            "loadimage": {"class": "loadimage",
                          "inputs": {"image": staged["image"]}},
            "loadaudio": {"class": "loadaudio",
                          "inputs": {"audio": staged["audio"]}},
            "h3": {"class": "h3",
                   "inputs": {"clip": W("clip", 0), "vae": W("vae", 0),
                              "audio_vae": W("audiovae", 0),
                              "prompt": self._prompt_for(request),
                              "width": int(width), "height": int(height),
                              "length": int(model_length),
                              # 'match' scales each reference DOWN to the
                              # generation's pixel area. 'max' uses a 2048 short
                              # edge for identity fidelity and the node's own
                              # tooltip warns it can be several times slower,
                              # because reference tokens ride through every
                              # sampling step. The lab's passing lip-sync leg
                              # used 'match'; this lane ships what was measured.
                              "ref_image_size": "match",
                              "ref_images": {"ref_image_0": W("loadimage", 0)},
                              "ref_audios": {"ref_audio_0": W("loadaudio", 0)}}},
        }

    def _weight_rows(self):
        """The shared three plus the audio VAE this lane's conditioner needs."""
        return super()._weight_rows() + (_H3_AUDIO_VAE_ROW,)


__all__ = [
    "MiniMaxH3VideoEngine", "MiniMaxH3AudioInEngine",
    "align_frame_count", "model_rungs",
    "canvas_frames_for_model", "model_frames_for_canvas", "canvas_index_map",
    "H3_MODEL_RUNGS", "H3_CANVAS_RUNGS", "H3_MODEL_FPS", "H3_CANVAS_FPS",
    "H3_RECIPE", "H3_RECIPE_RECEIPT",
]


# =========================================================================== #
# PER-LANE MOTION PROMPT -- EDIT HERE (Option B, operator ruling 2026-08-27)
# =========================================================================== #
# Bound to the class ITSELF at the bottom of this block. Dispatch reads
# ``type(engine).__dict__``, so an inherited formatter is invisible -- that is
# what keeps this lane independent of its siblings. Editing this changes THIS
# lane and nothing else.
try:
    from .motion_common import compose_parts as _parts, compose_legacy as _legacy
except ImportError:  # pragma: no cover -- flat test imports
    from motion_common import compose_parts as _parts, compose_legacy as _legacy


#: THE OFFICIAL ONE-IMAGE I2VA REFERENCE LINE, required verbatim as the FIRST
#: thing in an H3 prompt. Pinned here because it was not previously in this
#: file at all. See also: `render_driver._style_cue_after_pinned_opener` seats
#: the style cue AFTER this opener instead of prefixing it, which would
#: otherwise push text in front of the opener and make "must begin exactly"
#: false on any non-default style pack. That was a live defect until
#: 2026-08-28: this comment claimed an exemption that nothing implemented, and
#: the anime pack demonstrably produced "anime style. For the target video, ...".
#: The cue is seated rather than dropped so the pack's look is not lost here.
H3_REFERENCE_OPENER = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

def compose_minimax_h3_video(self, inputs):
    """`minimax_h3_video` -- silent. The official one-image I2VA grammar.

    The opener is REQUIRED VERBATIM and is pinned here because it did not exist
    in this file before: the old default said "the subject crosses the frame".
    Exactly one <Picture 1>, and NO dialogue, <Audio, overall_soundscape or
    non_diegetic_music -- omitting the audio fields is the correct silent-lane
    behaviour; emitting "N/A" is not.

    One flowing action with explicit TEMPO, then one camera type/amplitude/speed
    described separately. Prompt is the only exposed semantic motion control on
    this lane; speed wording is a request, not proof of displacement.
    """
    core = _parts(inputs)
    if not core:
        return _legacy(inputs)
    return ("%s %s, the movement carried at a clear steady tempo to a visible "
            "endpoint" % (H3_REFERENCE_OPENER, core.rstrip(" ,.")))


def compose_minimax_h3_audio_in(self, inputs):
    """`minimax_h3_audio_in` -- lip-sync plus JUST ENOUGH movement.

    Human-reviewed lab result: PASS at 5.17s on seed 43, and the IDENTICAL
    prompt FAILED on seed 42 -- so this lane is seed-sensitive and its recorded
    failure mode is the jaw, chin and collar reshaping as the head approaches
    yaw. Hence: ONE weight shift or forward lean plus a small head tilt, staying
    frontal, and STOP BEFORE PROFILE.

    Dialogue is retained here (this lane preserves it) and referenced against
    <Audio 1>, which is what the upstream grammar asks for.
    """
    core = _parts(inputs, include_camera=False)
    if not core:
        return _legacy(inputs)
    line = str((inputs or {}).get("dialogue") or "").strip()
    spoken = (" <d>[English]%s</d>, synchronised to <Audio 1>" % line) if line else ""
    return ("%s %s, one slight forward weight shift and a small head tilt, "
            "staying frontal with the mouth and jaw clearly visible, camera "
            "locked%s" % (H3_REFERENCE_OPENER, core.rstrip(" ,."), spoken))


MiniMaxH3VideoEngine.compose_prompt = compose_minimax_h3_video
MiniMaxH3AudioInEngine.compose_prompt = compose_minimax_h3_audio_in
