"""MiniMax H3 (33.1B packed AV DiT) -- the SHARED implementation module.

LANE 19 registers exactly ONE adapter out of this file: ``minimax_h3_video``
(public ``h3_low_video``), the FL2VA first-frame-to-video route. The second
adapter, ``minimax_h3_audio_in`` (ref2va, public ``h3_low_audio_in``), is LANE
20 and is deliberately NOT registered here yet -- one internal engine cannot
carry both modes, and two public ids mapping to one internal id collapses
``public_engines._INTERNAL_TO_PUBLIC`` and trips its module-scope bijection
assert at IMPORT time, which empties most of the ComfyUI menu rather than
failing one lane cleanly (lesson L5).

WHAT THIS ENGINE IS. H3 is the first LOCAL engine that natively produces AUDIO:
its latent is a NestedTensor PAIR -- video ``[B,24,T,H/16,W/16]`` and audio
``[B,32,2,T40]`` -- and the installed core nodes decode each half with its own
VAE. **This lane decodes the VIDEO half only.** There is no audio VAE in its
graph, so the emitted clip is silent BY CONSTRUCTION as well as by proof (V-1:
only ``OTR_MasterAudioMux`` ever adds audio). That construction is not taken on
trust; ``canonicalize`` ffprobes this lane's OWN emitted file before writing
``has_audio: False``, because on this engine that literal is the one field in
the roster that could be a lie.

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
This lane declares **864x480** and the reasoning is in :attr:`render_canvas`.

BOOT. Sage silently turns H3 output into noise, video and audio, with no error
(Comfy-Org/ComfyUI#15263; the per-model KJ probe FAILED on sm_120), so this lane
is in the preflight matrix's ``SAGE_SENSITIVE`` set and refuses by name before
any weight resolves. It also REQUIRES the named ``h3`` boot contract, which it
may do because it has never shipped under ``default``.

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
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.video.minimax_h3")


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

#: ``(env_name_suffix, folder_paths categories, default token, min bytes)`` for
#: every weight this lane loads. The byte floors distinguish the real artifact
#: from a truncated download: a partial 21 GB fetch otherwise reaches the loader
#: and the operator gets a deep safetensors traceback instead of the named
#: refusal that tells them to re-fetch. Each floor is ~80% of the real file, so
#: it catches truncation without pinning a repack's exact size.
_H3_WEIGHTS = (
    ("UNET", ("diffusion_models", "unet"), _H3_DEFAULT_UNET, 16 * 1024 ** 3),
    ("CLIP", ("text_encoders", "clip"), _H3_DEFAULT_CLIP, 12 * 1024 ** 3),
    ("VAE", ("vae",), _H3_DEFAULT_VAE, 4 * 1024 ** 3),
)

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

_H3_DEFAULT_PROMPT = "subtle natural motion, cinematic light"


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
                     "cinematic three-quarter framing, people shown with full "
                     "heads and clear headroom inside frame, faces "
                     "unobstructed, balanced composition"),
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


@register
class MiniMaxH3VideoEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """``minimax_h3_video`` -- MiniMax H3 FL2VA, first frame to silent video."""

    name = "minimax_h3_video"
    family = "image_to_video"
    #: Wide: the director mints a landscape init still for this lane.
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
    #: It is decided by the PUBLIC ID this lane was assigned. The naming
    #: convention is ``<model><version>_<low|high>_<capability>`` where ``low``
    #: means measured under ~8 GiB, and the corpus named this lane
    #: ``h3_low_video``. Against the lab receipts on THIS box, under this lane's
    #: own boot contract:
    #:
    #:   * 864x480 -- ``h3_mime_i2v`` model f90, **7.28 GiB** absolute, 178.9 s;
    #:     the ref2va sibling at model f124 measures 7.20 GiB. Both ``low``.
    #:   * 1344x768 (the trained shape) -- ``h3_i2v_best`` model f124,
    #:     **9.15 GiB** absolute, 1182.5 s. Not ``low``, and 6.6x the wall clock
    #:     for one 5 s beat.
    #:   * 832x480 -- the ONE canvas with no passing receipt: model f107 peaked
    #:     **15.39 GiB** and FAILED the 14.5 GiB gate. That leg was also below
    #:     the trained floor AND on a boot lane without the reserve clamp, so it
    #:     is evidence of a failure, never evidence for a number.
    #:
    #: 864x480 is the only value with a passing I2V receipt at which the id's
    #: own ``low`` token is true, and it is /32-legal on both axes (27 x 15).
    #: ``tests/test_minimax_h3_video.py`` pins it, and its profile carries the
    #: same pair as a drift guard.
    render_canvas = (864, 480)

    still_plan = _H3_STILL_PLAN
    roles = ROLES
    default_roles = ()
    required_inputs = ("init_image",)

    #: THE FRAME LADDER, PUBLISHED IN CANVAS FRAMES. ``discrete_frames`` is the
    #: 17k+5 model grid over the trained band, converted to 25 fps: 129 .. 377.
    #: Both tuples are derived from ``align_frame_count`` at import, so the menu
    #: cannot drift from the grid the node will actually snap to.
    #:
    #: ``allow_tail_trim`` is mandatory for a discrete menu (a fixed set of
    #: lengths cannot sum to an arbitrary beat) and it is honest here: the
    #: surplus is dropped in REAL delivered frames, in order, never mirrored.
    #:
    #: CONTINUITY -- decided, not defaulted. ``MiniMaxH3ImageToVideo`` pins the
    #: supplied still as a keyframe at ``resolved_frame_index 0`` and the node's
    #: own docstring records that keyframe condition latents are "re-injected
    #: every step (never denoised)". That is a first-frame lock of the same
    #: class as ``LTXVImgToVideo(strength=1.0)`` on the incumbent chained lanes:
    #: a LATENT-space pin through a VAE round trip, not a byte copy of the
    #: source pixels. The lab's continuation experiment is the measurement --
    #: prior-clip-last to next-clip-first scored SSIM 0.900 / PSNR 33.4 dB and
    #: its seam gate passed. Declaring NONE instead would refuse chaining and
    #: turn every multi-segment H3 beat into jump cuts it does not need.
    frame_contract = FrameContract(
        discrete_frames=H3_CANVAS_RUNGS,
        native_fps=H3_CANVAS_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )

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

    #: This lane REQUIRES the named ``h3`` boot contract, which it is allowed to
    #: do because it has never shipped under ``default`` (boot_contracts' own
    #: rule -- requiring a contract on a lane that already ships would regress
    #: it). Sage-free is non-negotiable here: Sage does not degrade H3, it
    #: replaces the output with noise and reports success.
    compatible_boot_contracts = ("h3",)

    #: Terminal node of the graph (its IMAGE output becomes the clip).
    _TERMINAL = "decode"

    # ---- loader tokens ------------------------------------------------- #
    def _token_for(self, label, default):
        return os.environ.get("OTR_MINIMAX_H3_%s_NAME" % label) or default

    def _weight_paths(self):
        """``{label: (token, resolved path or None)}`` for all three weights.

        Resolution goes through ``WanInitImageMixin._resolve_model_file``, which
        asks ComfyUI ``folder_paths`` for the bare loader token -- the same
        question ``UNETLoader`` / ``CLIPLoader`` / ``VAELoader`` will ask, so a
        weight registered through ``extra_model_paths.yaml`` resolves here and a
        preflight cannot pass on a file the loader will not find (lesson L1).
        """
        out = {}
        for label, categories, default, _floor in _H3_WEIGHTS:
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
        parts = [self.name, H3_RECIPE_RECEIPT]
        for label, _cats, _default, _floor in _H3_WEIGHTS:
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
        return {
            "unet": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "h3": ("MiniMaxH3ImageToVideo",),
            "guider": ("BasicGuider",),
            "noise": ("RandomNoise",),
            "sampler": ("KSamplerSelect",),
            "sched": ("BasicScheduler",),
            "sample": ("SamplerCustomAdvanced",),
            "decode": ("VAEDecode",),
        }

    def _recipe_receipt(self):
        return H3_RECIPE_RECEIPT

    def _build_graph(self, request, image_name, plan, model_length, width, height):
        """The declarative FL2VA graph (``wrapper_bridge.run_graph`` format).

        It is the lab's hash-pinned topology MINUS the audio half: no
        ``VAELoader`` for the audio VAE, no ``VAEDecodeAudio``, no
        ``CreateVideo`` / ``SaveVideo``. OTR takes the decoded IMAGE batch and
        writes its own always-silent bt709 mp4, which is what makes V-1 a
        property of the graph and not only of a check.

        ``last_frame`` is deliberately never wired. It is H3's first/LAST
        interpolation endpoint, which is a different capability from the
        first-frame chaining this lane declares -- supplying it would make the
        render interpolate toward a frame the coverage planner never chose.
        """
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        names = self._weight_paths()
        prompt = get("text_prompt") or _H3_DEFAULT_PROMPT
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
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "h3": {"class": "h3",
                   "inputs": {"clip": W("clip", 0), "vae": W("vae", 0),
                              "prompt": prompt,
                              "width": int(width), "height": int(height),
                              "length": int(model_length),
                              "first_frame": W("loadimage", 0)}},
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
                   for label, _cats, _default, _floor in _H3_WEIGHTS
                   if not paths[label][1]]
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "minimax_h3_video weight(s) not found: %s -- drop them in the "
                "matching models/ folder or register it in "
                "extra_model_paths.yaml (the OTR_MINIMAX_H3_*_NAME variables "
                "only name a file, they cannot make the loader find one)"
                % ", ".join(missing), kind="video")
        for label, _cats, _default, floor in _H3_WEIGHTS:
            token, path = paths[label]
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "minimax_h3_video %s weight %r is only %d bytes (< the %d "
                    "floor) -- that is a truncated or wrong file, not the H3 "
                    "artifact; re-fetch it" % (label, token, size, floor),
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
                "minimax_h3_video missing required ComfyUI node class(es): %s "
                "(MiniMax H3 needs a ComfyUI carrying comfy_extras/"
                "nodes_minimax_h3.py -- update ComfyUI and restart)"
                % ", ".join(absent), kind="video")
        return self.name

    def _assert_boot_contract(self, profile):
        """The profile's contract must be one this lane is proven on, AND the
        running server must actually be in that state.

        Two checks, deliberately separate (the lane-2 pattern). The first is
        pure and always runs. The second probes ``comfy.cli_args`` -- what this
        process was really started with -- because a check that reads the same
        config the launcher was supposed to honour cannot tell "applied" from
        "written down" (lesson L6).
        """
        from .._otr_shared import boot_contracts as _bc
        problems = _bc.check_engine_against_profile(self, profile or {})
        if problems:
            raise EngineUnusable(
                self.name, self.family,
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "; ".join(problems), kind="video")
        _bc.assert_running_server(_bc.contract_for_profile(profile or {}))

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
                "minimax_h3_video not installed: missing %s" % missing)
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
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "minimax_h3_video requires init_image (got %r)"
                % plan["init_image"])
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
                "minimax_h3_video was handed a %dx%d canvas; both axes must be "
                "/32-legal (every H3 node declares width/height at step=32 and "
                "the latent is H/16 x W/16). This lane declares %dx%d."
                % (width, height, self.render_canvas[0], self.render_canvas[1]))
        # THE LENGTH IS DECIDED BEFORE ANYTHING IS STAGED TOO (B4). The ask
        # arrives in CANVAS frames; the graph is driven in MODEL frames.
        target_frames = int(plan.get("target_frame_count") or 0)
        contract = self.frame_contract
        canvas_rung = contract.smallest_legal_at_least(target_frames)
        model_length = (None if canvas_rung is None
                        else model_frames_for_canvas(canvas_rung))
        if canvas_rung is None or model_length is None:
            raise _wb.GraphExecutionError(
                "minimax_h3_video was asked for %d canvas frame(s); its legal "
                "menu is %d..%d at 25 fps (the model's 17k+5 grid over its "
                "trained %d..%d band, converted from 24 fps). NO FALLBACK -- "
                "rendering a shorter clip and padding it back up is not a "
                "render of the ask, and on a chained lane the next segment "
                "would begin on a manufactured frame."
                % (target_frames, H3_CANVAS_RUNGS[0], H3_CANVAS_RUNGS[-1],
                   H3_MODEL_RUNGS[0], H3_MODEL_RUNGS[-1]))
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        graph = self._build_graph(request, image_name, plan, model_length,
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
                "minimax_h3_video asked its graph for %d model frame(s) and "
                "decoded %d. NO FALLBACK -- padding the difference is how a "
                "render that did not happen gets counted as one."
                % (model_length, len(frames)))
        # THE 24 -> 25 CONVERSION, and it is this lane's real deliverable.
        # Nearest-source-frame resampling of the uint8 batch, immediately before
        # the encoder. Relabelling instead would make every beat ~4% short: the
        # local encoder can only LABEL a rate (`-r` precedes `-i pipe:0`), it
        # can never resample, so a 192-frame render tagged 25 fps would play in
        # 7.68 s while the ledger and the audio both expect 8.000 s.
        frames = frames[list(canvas_index_map(model_length, canvas_rung))]
        _LOG.info(
            "[OTR video] minimax_h3_video 24->25 conversion: %d model frame(s) "
            "at %d fps (%.3f s) -> %d canvas frame(s) at %d fps (%.3f s)",
            model_length, H3_MODEL_FPS, model_length / float(H3_MODEL_FPS),
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
                "[OTR video] minimax_h3_video tail trim: delivered %d of %d "
                "canvas frame(s) (nearest legal rung at or above the %d-frame "
                "ask) @ %dx%d", len(frames), canvas_rung, target_frames,
                width, height)
        out_path = otr_engine_tmp_mp4("otr_minimax_h3_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path,
                                                  self.target_fps)
        # M7: PROVE the silent bt709/yuv420p contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info(
                "[OTR video] minimax_h3_video VRAM render-phase peak %s MB @ "
                "%dx%d model_len=%d canvas_len=%d",
                render_peak, width, height, model_length, n)
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
        installed nodes ship a decoder for each half. This lane's graph carries
        no audio VAE, so the file is silent by construction -- but "by
        construction" is precisely the kind of claim that survives a graph edit
        it should not survive, and G5 exists because a declaration checking a
        declaration proves nothing (lesson L4). So the container is re-probed
        HERE, at the seam where the field is written.
        """
        raw = raw or {}
        path = raw.get("out_path", "")
        if path:
            validate_silent_clip_contract(ffprobe_clip_fields(path),
                                          self.target_fps)
        clip = self._clip_from_raw(raw, request)
        clip["model_frame_count"] = raw.get("model_frame_count")
        return clip


__all__ = [
    "MiniMaxH3VideoEngine", "align_frame_count", "model_rungs",
    "canvas_frames_for_model", "model_frames_for_canvas", "canvas_index_map",
    "H3_MODEL_RUNGS", "H3_CANVAS_RUNGS", "H3_MODEL_FPS", "H3_CANVAS_FPS",
    "H3_RECIPE", "H3_RECIPE_RECEIPT",
]
