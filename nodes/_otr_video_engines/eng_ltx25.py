"""LTX 2.5 Distilled -- the SILENT video lane (Chunk A of the 2026-08-19 split).

ONE lane, one class, one graph: the beat's wide scene still is pinned to frame 0
and 97 frames are rendered from it at 832x480. The model's audio side is
computed and then DISCARDED at ``LTXVSeparateAVLatent`` -- ``LTXVAudioVAEDecode``
is never wired -- so the clip is always silent and only ``OTR_MasterAudioMux``
ever adds audio (invariant V-1).

WHY THIS IS ONE FLAT CLASS AND NOT A BASE WITH A SEAM. The operator's ruling,
in his words, is that this is a construction site: *"you gotta close it until
you can reopen the renovations"*, and *"you cant build the new wing unless you
tear down the old one and build the foundation and the old wing connection
branch."* The two sibling lanes this model will eventually carry -- ``mime``
(the clip's own foley IS the beat audio) and ``foley_plus`` (a foley bed under
the TTS) -- are Chunk B, and Chunk B is not deferred paperwork: it needs an
EXECUTION-ORDER change, because video renders four topological stages after the
master audio freezes (``OTR_EpisodeAssembler`` order 12, ``OTR_VideoRenderBatch``
order 16). A subclass hook designed today would be a doorway into a wing with no
foundation. So there is no hook. When the foundation exists, the sibling is a
deliberate copy of this file and the differences are visible in it.

THE TWO SIBLING LANES ARE RESERVED, NOT REGISTERED. ``ltx25_mime`` and
``ltx25_foley_plus`` are the settled internal names, publicly
``ltx25_high_mime`` and ``ltx25_high_foley_plus`` (GO_FORWARD, 2026-08-19).
They appear here so nobody takes them, and they are deliberately absent from the
registry: a menu row that cannot make an episode is worse than a missing one.
Per lesson L5 they will be THREE internal engines, never one id with a switch --
two public ids on one internal id collapses ``_INTERNAL_TO_PUBLIC`` and trips the
bijection assert AT IMPORT, which empties most of the ComfyUI menu.

THE PUBLIC ID IS ``ltx25_high_video``, AND ``high`` WAS SETTLED BY A RULING
RATHER THAN BY THE MEASUREMENT THAT WAS PLANNED. The convention is
``<model><version>_<low|high>_<capability>`` and G7.4 requires the token be
measured, never guessed -- guessing is what retired ``<vramtier>gb``, since
``wan_8gb`` really costs 12.5-13.2 GiB and cannot run on an 8 GB card. This
token was held UNSET for most of a day for exactly that reason, pending a clamp
test on a 4060. The operator then ruled the 4060 out entirely, which settles the
name by DELETING the question instead of answering it: the lane is 5080-only, so
``high`` is precisely what the token means. It also agrees with the only number
in evidence -- 14.48 GiB against a 14.5 GiB clamp is the most expensive local
lane in the roster, and ``low`` would have been false in the same direction the
retired token was false.

WHAT IS STILL UNMEASURED, SO NOTHING READS AS QUALIFIED: this lane has no cost
row and no envelope key, the evidence manifest records it as admission-unenforced
in words (G4.1), and it declares no ``compatible_boot_contracts``. The G8 solo
smoke is what closes those, and it is a SEPARATE receipt from the naming.

THE RECIPE IS NOT ON THE TABLE. Every number comes from :mod:`ltx25_recipe`,
which transcribes the lab's locked graph and is drift-gated against the lab's
actual file by ``tests/test_ltx25_recipe_matches_lab_golden.py``. Operator,
standing: *"no chasing vram recipes please... we are running on the Q3, that's
the safe one."* No Q5, no bigger canvas, no more frames, no CFG tuning. If a
value here turns out to be wrong that is a finding to REPORT, not a knob to turn.

Cold-import clean (V-12): module scope imports only stdlib plus the dep-free
shared helpers and the registry. torch and every LTX node class are resolved
LAZILY inside ``load`` / ``render_clip``. There are NO module-scope numeric env
reads at all (G6.3): the recipe is locked, so there is no number for an
environment to move, and a knob that reaches nothing is worse than no knob (L6).
The only env pins are weight FILENAMES, for a box that stores them under other
names. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging
import os

from .._otr_shared.still_plan_helpers import StillPlanRow
from .._otr_shared.role_compat import ROLES
from . import motion_common as _MC
from . import ltx25_recipe as R
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.eng_ltx25")

#: RESERVED, NOT REGISTERED -- see the module docstring. These are the settled
#: internal ids for the two Chunk B lanes. Naming them here is the cheapest way
#: to stop someone spending one of them on something else; registering them
#: would put two rows in the operator's dropdown that cannot render, because the
#: pre-freeze audio path they need does not exist yet.
LTX25_RESERVED_SIBLING_IDS = ("ltx25_mime", "ltx25_foley_plus")

# ---------------------------------------------------------------------------
# Weight resolution (G1). Every artifact resolves through ComfyUI's
# ``folder_paths`` so ``extra_model_paths.yaml`` is honoured -- never a bare
# os.path.exists on a hardcoded default, which is the defect G1.1 exists for and
# the one that shipped ``wan_i2v`` dead (lesson L1).
# ---------------------------------------------------------------------------

#: Byte floors. A truncated or wrong-name fetch is NAMED here rather than traced
#: out of a loader. Set well under the real artifacts on this box (10.7 GiB DiT,
#: 8.9 GiB encoder, 1.37 GiB video VAE, 348 MiB audio VAE) so a legitimate
#: re-quant does not trip them, but far above any HTML error page or LFS pointer.
_GiB = 1024 ** 3
_FLOOR_DIT = 6 * _GiB
_FLOOR_TEXT_ENCODER = 4 * _GiB
_FLOOR_VIDEO_VAE = int(0.5 * _GiB)
_FLOOR_AUDIO_VAE = int(0.1 * _GiB)


def _resolve(folder, name):
    """Resolve a model filename to a full path via ComfyUI ``folder_paths``.

    Honours ``extra_model_paths.yaml``, which matters concretely here: the LTX
    2.5 DiT lives in ``diffusion_models/`` on this box while the loader node
    asks for the ``unet`` folder key. ComfyUI's ``map_legacy`` aliases ``unet``
    to ``diffusion_models`` and this box's headless yaml maps that key at BOTH
    ``C:/ComfyUI-Models/diffusion_models`` and ``C:/ComfyUI-Models/unet``, so
    the lookup succeeds -- but only THROUGH folder_paths. A hand-built path
    would miss it.

    The join fallback exists only for the headless / CPU existence check where
    no folder_paths is registered; it is best-effort by design and never the
    production path.
    """
    if not name:
        return ""
    try:
        import folder_paths  # type: ignore
        p = folder_paths.get_full_path(folder, name)
        if p:
            return p
    except Exception:  # noqa: BLE001 - headless/CPU: fall through to the join
        pass
    here = os.path.abspath(__file__)
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here))))), "models", folder, name)


#: S1 per-model still plan (G7.4). This lane is I2V and the still is NOT
#: optional -- ``LTXVImgToVideoInplace`` at strength 1.0 IS the conditioning, so
#: a beat with no still has no graph. Hence ``required="always"`` on all three
#: scene rows, where the sibling ``ltx_video`` says ``when_ltx_i2v_enabled``
#: (that lane can fall back to a text-only path; this one cannot and must not).
#:
#: WIDE on every row: the lane renders 832x480 and declares ``render_aspect =
#: "wide"``, so a portrait still would be centre-cropped by the resize node and
#: lose the sides of the frame -- the same class of defect as the 2026-06-17
#: operator catch where a portrait still was cropped and lopped a head off.
_LTX25_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
)


@register
class Ltx25VideoEngine(_MC.MotionEngineBase):
    """LTX 2.5 Distilled, silent video. I2V on the beat's wide scene still.

    The graph is the lab's golden recipe
    (``vram-recipe-lab/recipes/ltx_2_5_golden_i2v_foley.json``) translated node
    for node, minus the four nodes OTR owns itself: ``FloatConstant`` becomes a
    literal, ``LTXVAudioVAEDecode`` is never wired (V-1), and ``CreateVideo`` /
    ``SaveVideo`` are replaced by this repo's own silent-mp4 encoder so the
    emitted file can be PROVED silent rather than declared so.
    """

    name = "ltx25_video"
    #: image_to_video, not audio_conditioned_video: nothing about this lane
    #: consumes audio. It is conditioned on a still and a text prompt, and the
    #: audio latent it computes is thrown away. ``audio_in`` on the id or the
    #: family here would claim the opposite of what the graph does.
    family = "image_to_video"
    still_plan = _LTX25_STILL_PLAN
    roles = ROLES
    #: SELECTABLE, NEVER A DEFAULT. Until the G8 solo smoke has run on this box
    #: nothing about this lane is qualified, so it must not become the automatic
    #: pick for any beat role. An operator choosing it is a decision; inheriting
    #: it would not be.
    default_roles = ()
    required_inputs = ("text_prompt", "init_image")
    #: NO FALLBACKS -- fail LOUD. A silent downgrade to another engine is how a
    #: lane's real failure gets recorded as somebody else's render.
    fallback_engine = None
    accepts_still = True
    render_aspect = "wide"
    requires_flag = None                 # registry IS the menu; no flag gate
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = R.LTX25_FPS

    #: UNVERIFIED, and False is the honest default rather than a verdict. This
    #: flag drives the release-gate warning and the release filename tag, never
    #: selection (the H3 precedent). No license attestation exists on this box
    #: for the LTX 2.5 Distilled weights the way the H3 attestation file does
    #: for H3, and the sibling LTX 2.3 lanes' ``True`` is about the LTX-2
    #: Community model, which is a different release. Claiming clean on
    #: inheritance is exactly the kind of assumption a release gate exists to
    #: catch, so it warns until the operator confirms at license review.
    #:
    #: THE ATTESTATION FILE IS NAMED WITHOUT ITS PATH PREFIX ON PURPOSE.
    #: ``tools/engine_matrix.py`` scrapes adapter sources for doc-directory
    #: citations and treats EVERY hit as that lane's FRAME-CAP evidence. Writing
    #: the real path here put "H3 license attestation" in this lane's cap-
    #: evidence column on the first generated matrix -- a licence document
    #: standing in as proof of a frame ceiling -- and then writing an elided
    #: form of the path in this very comment did it a second time, because the
    #: scraper's pattern accepts dots. So the rule is not "elide it", it is
    #: "do not write that prefix in an adapter at all unless the file behind it
    #: really is frame evidence". Both misfires were caught by READING the
    #: generated diff, which no test would have done for us.
    commercial_clean = False

    #: NO BOOT CONTRACT IS DECLARED, and the omission is the statement. An
    #: adapter that declares nothing runs under every contract; an adapter that
    #: declares a tuple is naming the boots it has been PROVEN on. This lane has
    #: been proven on none of them yet, so naming one would be a guess that
    #: refuses a boot it might be perfectly happy on. It gets its declaration
    #: from the G8 solo smoke, in the same change that fills the envelope key.
    #: Sage is handled where it is actually enforceable -- ``assert_usable``
    #: below -- not by a contract knob no launcher passes (L6).

    #: THE CANVAS, DECLARED (G2.1). 832x480, and it is not a preference: the lab
    #: rejected 768x432 because 432/32 = 13.5 corrupts the tensor and fails the
    #: PyTorch VAE decode, and rejected 1024x576 on OOM. Both axes are /32-legal
    #: (26 x 15) and this is the canvas the rest of the OTR video fleet already
    #: renders at. Declaring it is what stops ``build_request_from_shot`` falling
    #: through to the 1472x832 landscape default no matter what a profile says
    #: (lesson L2). Pinned by ``tests/test_ltx25_video_lane.py``.
    render_canvas = (R.LTX25_CANVAS_W, R.LTX25_CANVAS_H)

    #: ONE RUNG, AND IT IS 97 FRAMES. This is a discrete menu of exactly one
    #: length because that is what the locked recipe is: 97 at 25 fps = 3.88 s,
    #: the standard OTR shot. The temporal contract ``(97 - 1) % 8 == 0`` is what
    #: the model's temporal downsampling requires, and CLAUDE.md separately
    #: forbids raising the 97 trained-length cap -- so the single rung satisfies
    #: both constraints at once rather than by coincidence.
    #:
    #: ``allow_tail_trim`` is REQUIRED with a discrete menu (FrameContract
    #: refuses the combination otherwise) and it is also correct: a beat shorter
    #: than 3.88 s renders the rung and drops the surplus frames in order -- no
    #: mirror, no loop, no held frame. A beat LONGER than one rung is partitioned
    #: into chained 97-frame segments by the coverage planner, which is what
    #: ``discrete_frames`` makes ``frame_contract.can_split`` answer True to.
    #:
    #: CONTINUITY IS STRICT_FIRST_FRAME, and it is EARNED here rather than
    #: inherited from the siblings. ``LTXVImgToVideoInplace`` runs at strength
    #: **1.0**, a HARD pin of frame 0 to the supplied still, fixed rather than
    #: recipe-dependent -- which is exactly why ``ltx_audio_in`` declares only
    #: soft_reference (its strength varies 0.7/0.75/1.0 by recipe, and a
    #: contract that is only sometimes true is a jump cut).
    #:
    #: THE OBJECTION, AND WHY IT DOES NOT LAND. Frame 0 is not byte-identical
    #: to the still handed in -- the image makes a round trip through the video
    #: VAE (encoded at ``i2v``, decoded at ``decode``), so "starts EXACTLY on
    #: segment N's terminal frame" looks like an over-claim. The driver raised
    #: this against itself and could not settle it; the agy review lane settled
    #: it from the chaining code, 2026-08-19, and the answer is that the
    #: imprecise frame NEVER REACHES THE SCREEN:
    #:
    #:   * on a chained beat ``render_driver`` extracts segment N's terminal
    #:     frame to a PNG and hands it to segment N+1 as its init image;
    #:   * ``coverage_plan`` gives every segment after the first
    #:     ``drop_head=1`` (``drop_head=(drop if index else 0)``), so segment
    #:     N+1's frame 0 -- the VAE round trip of that still -- is DROPPED at
    #:     assembly.
    #:
    #: So the viewer sees segment N's real terminal frame followed by segment
    #: N+1's frame 1. Declaring soft_reference instead would push the planner
    #: to a JUMP join, which discards the extracted terminal frame and hard-cuts
    #: between unrelated renders -- strictly worse, and on this lane a beat over
    #: 3.88 s is the common case rather than the exception. What remains
    #: unproven is only whether the seam is optically clean on this model, which
    #: is a G8 eyeball, not a contract question.
    frame_contract = FrameContract(
        discrete_frames=(R.LTX25_FRAMES,),
        native_fps=R.LTX25_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )

    #: Terminal node: its IMAGE batch becomes the clip.
    _TERMINAL = "decode"

    # ---- weight tokens (env pins name a FILE, they cannot make one exist) ----
    def _dit_name(self):
        return os.environ.get("OTR_LTX25_DIT", R.LTX25_DIT_GGUF)

    def _text_encoder_name(self):
        return os.environ.get("OTR_LTX25_TEXT_ENCODER", R.LTX25_TEXT_ENCODER_GGUF)

    def _video_vae_name(self):
        return os.environ.get("OTR_LTX25_VIDEO_VAE", R.LTX25_VIDEO_VAE)

    def _audio_vae_name(self):
        return os.environ.get("OTR_LTX25_AUDIO_VAE", R.LTX25_AUDIO_VAE)

    def _weight_paths(self):
        """``(label, full_path, floor_bytes)`` for every required artifact.

        THE AUDIO VAE IS ON THIS LIST EVEN THOUGH THE LANE IS SILENT, and that
        surprises people every time. ``LTXVEmptyLatentAudio`` takes ``audio_vae``
        to MINT the audio latent and ``LTXVConcatAVLatent`` needs that latent to
        build the joint AV tensor the sampler consumes -- so a silent lane still
        loads the audio VAE and still pays for the audio side through all 8
        steps. It only skips the decode. "We discard the audio" never meant "we
        avoid paying for it", and omitting the weight here would fail the lane
        at graph time instead of at the gate.
        """
        return [
            ("LTX 2.5 DiT (GGUF)", _resolve("unet", self._dit_name()),
             _FLOOR_DIT),
            ("Gemma-4 12B text encoder (GGUF)",
             _resolve("text_encoders", self._text_encoder_name()),
             _FLOOR_TEXT_ENCODER),
            ("LTX 2.5 video VAE", _resolve("vae", self._video_vae_name()),
             _FLOOR_VIDEO_VAE),
            ("LTX 2.5 audio VAE", _resolve("vae", self._audio_vae_name()),
             _FLOOR_AUDIO_VAE),
        ]

    def _quant_label(self):
        """The quant token from the DiT basename (``Q3_K_M``) for the per-beat
        observability line and the clip's recipe receipt. Pure."""
        import re
        m = re.search(r"(Q\d+(?:_[A-Za-z0-9]+)*|fp8|fp16|bf16|int8)",
                      os.path.basename(str(self._dit_name())))
        return m.group(1) if m else ""

    # ---- the graph spec (classes resolve through wrapper_bridge) ----
    def _node_candidates(self):
        """Every ComfyUI class this graph needs, by logical node id.

        Two graph nodes share the ``VAELoader`` class on purpose -- see
        ``_build_graph`` for why the video VAE is split into an encode-side and
        a decode-side node.
        """
        return {
            "unet": ("UnetLoaderGGUF",),
            "te": ("CLIPLoaderGGUF",),
            "videovae": ("VAELoader",),
            "audiovae": ("VAELoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "cond": ("LTXVConditioning",),
            "loadimage": ("LoadImage",),
            "resize": ("ResizeImageMaskNode",),
            "preprocess": ("LTXVPreprocess",),
            "emptylatent": ("EmptyLTXVLatentVideo",),
            "emptyaudio": ("LTXVEmptyLatentAudio",),
            "i2v": ("LTXVImgToVideoInplace",),
            "concat": ("LTXVConcatAVLatent",),
            "modality": ("LTXVModalityGuidance",),
            "guider": ("LTXVDualCFGGuider",),
            "noise": ("RandomNoise",),
            "ksel": ("KSamplerSelect",),
            "sched": ("LTXVScheduler",),
            "sampler": ("SamplerCustomAdvanced",),
            "separate": ("LTXVSeparateAVLatent",),
            "decode": ("VAEDecodeTiled",),
        }

    # ---- usability: fail CLOSED, cheapest refusal first ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordered gate, and the order is the contract.

        1. **Sage** (G6.1). It costs nothing to check and everything to miss:
           this is the LTX family BUG-070 was written for, where int8-PV Sage
           process-ABORTS the render with no traceback. A refusal before any
           weight resolves beats a dead process after 10 GiB has loaded.
        2. **Node classes**, collecting EVERY miss before raising. Naming them
           one at a time turns a fresh install into a sequence of failed renders.
        3. **Weights**, by folder_paths resolution then byte floor, so a
           truncated fetch is named rather than traced out of a loader.
        4. **The canvas**, if the caller handed one -- pure arithmetic, so it is
           refused here rather than three stages into a graph.
        """
        _MC.assert_sage_not_patched(self.name, self.family)

        from . import wrapper_bridge as _wb
        mapping = _wb.node_class_mappings()
        absent = []
        for _logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 - collect every miss
                absent.append("/".join(candidates))
        if absent:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s -- LTX 2.5 "
                "needs a ComfyUI carrying comfy_extras/nodes_lt.py + "
                "nodes_lt_audio.py plus ComfyUI-GGUF for the two GGUF loaders; "
                "update both and restart"
                % (self.name, ", ".join(sorted(set(absent)))), kind="video")

        for label, path, floor in self._weight_paths():
            real = os.path.realpath(path) if path else ""
            if not real or not os.path.exists(real):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s not found at %r -- drop it in the matching models/ "
                    "folder or register that folder in extra_model_paths.yaml "
                    "(the OTR_LTX25_* variables only NAME a file, they cannot "
                    "make the loader find one)" % (self.name, label, path),
                    kind="video")
            if os.path.getsize(real) < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s at %r is only %d bytes, under the %d-byte floor -- "
                    "that is a truncated download or an LFS pointer, not the "
                    "artifact; re-fetch it"
                    % (self.name, label, real, os.path.getsize(real), floor),
                    kind="video")

        if request_template is not None:
            w, h = self._canvas_dims(request_template)
            if (w or h) and (w, h) != tuple(type(self).render_canvas):
                raise EngineUnusable(
                    self.name, self.family,
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s was offered a %dx%d canvas and declares %dx%d. The "
                    "recipe is locked at the declared canvas (the lab rejected "
                    "768x432 as VAE-illegal and 1024x576 on OOM), so a "
                    "different size is a REFUSAL, never a silent re-plan"
                    % (self.name, w, h, type(self).render_canvas[0],
                       type(self).render_canvas[1]), kind="video")
        return self.name

    # ---- multi-clip session identity ----
    def _weight_receipt(self, path):
        """A BOUNDED, stable receipt for one weight file: (basename, size,
        mtime_ns). Deliberately NOT a content hash -- the identity is re-read
        before every segment, and hashing a 10.7 GiB GGUF per segment would
        cost more than the render it guards. Size plus mtime catches a swapped
        or rebuilt weight, which is the whole question being asked.

        An unresolvable path returns a NAMED absence rather than raising:
        ``assert_usable`` already owns "this weight is missing" and says it far
        better, and raising here would turn a missing-weight run into a
        confusing session error instead of the usability refusal it is.
        """
        if not path:
            return ("<unresolved>", -1, -1)
        try:
            st = os.stat(path)
        except OSError:
            return (os.path.basename(path), -1, -1)
        return (os.path.basename(path), int(st.st_size), int(st.st_mtime_ns))

    def session_identity(self):
        """What this adapter's resident handles ARE -- engine plus weights.

        THIS LANE NEEDS ONE BECAUSE IT SPLITS. Its contract declares
        ``discrete_frames=(97,)``, which makes ``frame_contract.can_split``
        True, so a beat longer than 3.88 s is partitioned and rendered as
        chained segments. ``BeatSession`` REFUSES a multi-segment beat from an
        engine that cannot answer this (``SessionIdentityUnavailable``, no
        fallback), because nothing could then prove segment N renders with the
        model segment 1 loaded.

        THE COST OF LEARNING THIS LATE IS ON RECORD, which is why it is here
        rather than after the first long beat: the sibling ``ltx_video`` lane
        reached a live render gate without one and refused there, 730 seconds
        into a leg -- the most expensive possible place to find out. The full
        suite caught it here instead, on a lane that had never rendered.

        PRE-LOAD STABLE by contract: read once before the weights land, again
        after ``prepare()``, and before every segment, so it may only describe
        things that do not change across the load. Every per-segment value --
        prompt, seed, frame count, canvas, the still -- is deliberately absent.
        There is no recipe token to carry either, unlike the LTX 2.3 lanes:
        this lane has exactly one locked recipe, so the weights ARE the
        identity.

        Not cached, by design. The entire job is to notice a weight that MOVED,
        so the receipts are re-stat'ed on every ask -- a stat per weight, never
        a hash.
        """
        parts = [self.name]
        for label, path, _floor in self._weight_paths():
            parts.append("%s=%r" % (label, self._weight_receipt(path)))
        return tuple(str(part) for part in parts)

    # ---- residency ----
    def load(self):
        """Resolve the installed node CLASSES. No weights load here.

        The DiT, the encoder and both VAEs are loaded by their loader nodes
        inside the graph, which is precisely what lets ``free_after_use`` drop
        the 8.9 GiB text encoder before the sampler runs -- the single biggest
        reason this stack fits at all.
        """
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    # ---- the graph ----
    def _build_graph(self, plan, image_name, length, width, height):
        """The golden recipe as a declarative graph. Pure: no GPU, no weights.

        Node-for-node against ``ltx_2_5_golden_i2v_foley.json``, with four
        deliberate departures, each of which is an OTR contract rather than a
        recipe change:

        * ``FloatConstant`` (lab node 18) is a LITERAL here. It existed only to
          fan 25.0 out to three consumers; a node that carries a constant is a
          UI convenience, not part of the recipe.
        * ``LTXVAudioVAEDecode`` (lab node 34) is NEVER WIRED. That is V-1, and
          it is the entire difference between this lane and Chunk B.
        * ``CreateVideo`` + ``SaveVideo`` (lab 35, 75) are replaced by this
          repo's own encoder, so the emitted file can be PROVED silent by
          ffprobe instead of declared silent by a literal (G5.1, lesson L4).

        THERE IS NO STRUCTURAL ADDITION. Every remaining node maps 1:1 to the
        golden recipe, including ONE ``VAELoader`` for the video VAE feeding
        both the i2v encode and the final decode (lab node 2).

        A DRAFT OF THIS ADAPTER SPLIT THAT LOADER IN TWO AND IT WAS CUT, which
        is worth recording because the split looks obviously right and is not.
        The idea, inherited by resemblance from ``eng_ltx_av``, was that a
        separate encode-side node would let ``free_after_use`` drop the VAE the
        moment the still is planted, before the sampler's activation peak. It
        does not. ``wrapper_bridge._topo_order`` is Kahn's algorithm with ties
        broken on sorted node id, and a ``VAELoader`` has NO dependencies -- so
        the decode-side copy is scheduled in the FIRST batch regardless, and
        ``free_after_use`` cannot release it until its only consumer, the
        decode, has run at the very end. The split therefore moved nothing;
        it just called the loader twice.
        And the headroom it was supposed to buy does not exist to be bought:
        the lab's decomposition puts both VAEs at **0.0 GiB** at the peak
        (:data:`ltx25_recipe.LTX25_PEAK_DECOMPOSITION_GIB`). Caught by the agy
        review lane, 2026-08-19, from a grounded read of the scheduler -- the
        driver had flagged it as a suspicion and could not settle it alone.

        THE TWO PLACES THE LAB'S PROSE DISAGREES WITH ITS OWN FILE are both
        wired the FILE's way, because the file is what ran:

        * the first-frame anchor is ``LTXVImgToVideoInplace`` at strength 1.0,
          not the ``SetLatentNoiseMask`` the QA document describes;
        * the scheduler's ``latent`` comes from the ImgToVideoInplace OUTPUT,
          not from ``EmptyLTXVLatentVideo``. On the I2V path those are different
          tensors, and wiring it the documented way hands the scheduler a latent
          with no still baked in -- a wrong-but-running failure, the worst kind.

        AND ONE TRAP THAT IS NEITHER: ``ResizeImageMaskNode`` is a DynamicCombo
        node. The lab's API-format JSON flattens its sub-inputs into dotted keys
        (``resize_type.width``), which is a serialisation detail ComfyUI's own
        executor un-flattens before the call. This graph calls ``execute``
        DIRECTLY, and its real signature takes ``resize_type`` as a single
        nested dict -- so copying the dotted keys across would raise
        ``unexpected keyword argument 'resize_type.width'`` on the first render.
        """
        from . import wrapper_bridge as _wb
        W = _wb.Wire

        positive = plan.get("text_prompt") or "a vintage radio broadcast scene"
        seed = int(plan.get("seed", 0) or 0)
        fps = float(R.LTX25_FPS)

        return {
            # --- loaders ---
            "unet": {"class": "unet",
                     "inputs": {"unet_name": self._dit_name()}},
            "te": {"class": "te", "inputs": {
                "clip_name": self._text_encoder_name(), "type": "ltxv"}},
            "videovae": {"class": "videovae",
                         "inputs": {"vae_name": self._video_vae_name()}},
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": self._audio_vae_name()}},

            # --- conditioning ---
            # The negative TEXT is empty; that is the locked recipe value.
            #
            # THE NEGATIVE CONDITIONING IS *NOT* INERT, and this comment used to
            # claim it was (corrected 2026-08-19). The ordinary ComfyUI rule --
            # cfg 1.0 elides the uncond -- does NOT hold here, because the
            # locked sampler `euler_ancestral_cfg_pp` forces
            # `disable_cfg1_optimization=True` and consumes `uncond_denoised`
            # in its own step derivative. The unconditional branch is computed
            # every step and steers the output.
            #
            # SO DO NOT "OPTIMISE" THIS BY WIRING `neg` FROM `pos`. It looks
            # free, it deletes a whole 12B encode, and it would silently change
            # every render. That exact proposal was made during the OOM panel
            # and killed by reading which sampler is selected.
            "pos": {"class": "pos",
                    "inputs": {"clip": W("te", 0), "text": positive}},
            "neg": {"class": "neg",
                    "inputs": {"clip": W("te", 0),
                               "text": R.LTX25_NEGATIVE_PROMPT}},
            "cond": {"class": "cond", "inputs": {
                "positive": W("pos", 0), "negative": W("neg", 0),
                "frame_rate": fps}},

            # --- the still, conformed then planted ---
            "loadimage": {"class": "loadimage",
                          "inputs": {"image": image_name}},
            # THIS RESIZE HAS NEVER ACTUALLY RUN, and the next person to see a
            # cropping bug should know that first. The lab confirmed (2026-08-19)
            # that their input still was ALREADY 832x480, so in every test
            # behind this recipe the node was a no-op. OTR's stills are minted
            # per beat by the director and are the first images that will really
            # exercise it. Kept, and kept deliberately: it is the guarantee that
            # whatever the director hands us arrives at the VAE at exactly the
            # declared canvas. ``crop: center`` is the risk it carries -- a
            # still at a different aspect loses its edges rather than letterbox
            # -- which is also why every row of this lane's still_plan demands
            # WIDE.
            "resize": {"class": "resize", "inputs": {
                "input": W("loadimage", 0),
                "scale_method": "lanczos",
                "resize_type": {"resize_type": "scale dimensions",
                                "width": int(width), "height": int(height),
                                "crop": "center"}}},
            # img_compression 0 is a documented PASS-THROUGH -- ComfyUI's
            # ``preprocess`` returns the image untouched at 0 -- and the lab
            # confirmed the 0 is DELIBERATE, chosen against the node's default
            # of 35. So this is not a widget nobody set: it is a decision to
            # skip the trained compression prior, and the node is kept as the
            # named seam where that prior would come back if the decision is
            # ever revisited. Costs one no-op call.
            "preprocess": {"class": "preprocess", "inputs": {
                "image": W("resize", 0), "img_compression": 0}},

            # --- latents ---
            "emptylatent": {"class": "emptylatent", "inputs": {
                "width": int(width), "height": int(height),
                "length": int(length), "batch_size": 1}},
            "emptyaudio": {"class": "emptyaudio", "inputs": {
                "audio_vae": W("audiovae", 0),
                "frames_number": int(length), "frame_rate": fps,
                "batch_size": 1}},
            "i2v": {"class": "i2v", "inputs": {
                "vae": W("videovae", 0), "image": W("preprocess", 0),
                "latent": W("emptylatent", 0), "bypass": False,
                "strength": R.LTX25_I2V_ANCHOR_STRENGTH}},
            "concat": {"class": "concat", "inputs": {
                "video_latent": W("i2v", 0),
                "audio_latent": W("emptyaudio", 0)}},

            # --- sampling ---
            # LTXVModalityGuidance at scale 1.0 is a documented NO-OP -- the
            # node's own description says "Set to 1.0 to disable (no extra
            # pass)". It is wired because the golden graph wires it and because
            # its position in the model chain is where a future measured value
            # would go; at 1.0 it costs one wrap and no forward pass.
            "modality": {"class": "modality", "inputs": {
                "model": W("unet", 0),
                "modality_scale": R.LTX25_CFG_MODALITY,
                "start_percent": 0.0, "end_percent": 1.0}},
            # ALL THREE CFGs ARE 1.0 AND THAT IS A VRAM CONTRACT, NOT TASTE.
            # The lab measured any higher value pushing past 16 GiB -- an
            # instant OOM against the 14.5 GiB clamp. Leave them.
            #
            # The "cfg 1.0 means batch size 1" reasoning this comment used to
            # give is WRONG for this recipe (corrected 2026-08-19): the CFG++
            # sampler evaluates the uncond branch anyway. The measured 14.48 GiB
            # already includes whatever that costs; it was the EXPLANATION that
            # was wrong, not the number. See ltx25_recipe.LTX25_CFG_VIDEO.
            "guider": {"class": "guider", "inputs": {
                "model": W("modality", 0),
                "positive": W("cond", 0), "negative": W("cond", 1),
                "video_cfg": R.LTX25_CFG_VIDEO,
                "audio_cfg": R.LTX25_CFG_AUDIO}},
            "noise": {"class": "noise", "inputs": {"noise_seed": seed}},
            "ksel": {"class": "ksel",
                     "inputs": {"sampler_name": R.LTX25_SAMPLER}},
            # The scheduler's ``latent`` MUST be connected. Left dangling it
            # silently defaults to a 4096-token curve and ruins the motion-shift
            # maths -- again wrong-but-running. It takes the ANCHORED latent.
            "sched": {"class": "sched", "inputs": {
                "steps": R.LTX25_STEPS, "max_shift": 2.05, "base_shift": 0.95,
                "stretch": True, "terminal": 0.1, "latent": W("i2v", 0)}},
            "sampler": {"class": "sampler", "inputs": {
                "noise": W("noise", 0), "guider": W("guider", 0),
                "sampler": W("ksel", 0), "sigmas": W("sched", 0),
                "latent_image": W("concat", 0)}},

            # --- terminate on VIDEO ONLY (V-1) ---
            # ``separate`` slot 1 is the AUDIO latent and nothing consumes it.
            # That single unwired slot is the whole of this lane's silence, and
            # it is the seam Chunk B opens.
            #
            # NOTE THE DIRECTION, because it is the opposite of the natural
            # assumption: the golden recipe ALREADY WIRES ``LTXVAudioVAEDecode``
            # to this slot (lab node 34). Chunk A SUBTRACTS it. Chunk B is
            # therefore not "add a decoder" -- the decoder is trivial and the
            # lab has run it. Chunk B is getting the decoded audio into the
            # master BEFORE the master freezes, four topological stages earlier
            # than video renders. Nothing here is what makes that hard.
            "separate": {"class": "separate",
                         "inputs": {"av_latent": W("sampler", 0)}},
            # The decode knobs come from the RECIPE MODULE, not from literals
            # here and emphatically not from ``eng_ltx_av._decode_temporal_knobs``
            # -- that helper's env-driven 4096/8 whole-clip default is a
            # different recipe, and inheriting it by resemblance would spend
            # headroom this lane does not have.
            "decode": {"class": "decode", "inputs": {
                "samples": W("separate", 0), "vae": W("videovae", 0),
                "tile_size": R.LTX25_DECODE_TILE_SIZE,
                "overlap": R.LTX25_DECODE_OVERLAP,
                "temporal_size": R.LTX25_DECODE_TEMPORAL_SIZE,
                "temporal_overlap": R.LTX25_DECODE_TEMPORAL_OVERLAP}},
        }

    # ---- render ----
    def render_clip(self, request, prepared):
        """Render ONE 97-frame clip and encode it to a SILENT bt709 mp4.

        THE RESIDENCY DISCIPLINE IS HYGIENE, NOT HEADROOM, AND THE DIFFERENCE
        WAS A CORRECTION. The operator's standing instruction is *"when in doubt
        load, unload, reload, unload etc."*, and this method honours it: the
        canonical residue-freer runs BEFORE the graph, the surgical reclaim runs
        AFTER, and the graph itself runs with ``free_after_use``. An earlier
        draft of this docstring said that staging is what lets the stack fit
        under the ceiling. **It is not, and the lab corrected it**
        (:data:`ltx25_recipe.LTX25_PEAK_DECOMPOSITION_GIB`): the 14.48 GiB peak
        is 9.80 DiT weights + 3.20 activations + 1.48 allocator context, with
        the text encoder and both VAEs at ZERO, because ComfyUI has already
        spilled Gemma to system RAM by the time sampling starts. Freeing an
        encoder that was not resident buys nothing.

        What the staging IS for is residue from EARLIER IN THIS SAME PROCESS --
        the writer LLM and the TTS stages, which ``comfy.model_management``
        cannot see and which nothing else in the video path releases. That is
        real and worth doing. It is just not what makes the render fit.

        Either way it is a SEQUENCING discipline: it changes when weights are
        resident, never a graph parameter, so it does not collide with the
        recipe being locked.
        """
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4

        plan = self._build_render_request(request)
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "%s requires an init image on EVERY beat: the graph plants the "
                "still in-place at strength %.1f, so no still means no "
                "conditioning. NO FALLBACK -- a silent downgrade to text-only "
                "would render something the beat did not ask for"
                % (self.name, R.LTX25_I2V_ANCHOR_STRENGTH))

        width, height = tuple(type(self).render_canvas)
        # THE ASK IS RESOLVED BEFORE ANYTHING IS STAGED. This lane has exactly
        # one legal length; an ask above it is a refusal, and an ask below it
        # renders the rung and trims the tail.
        target_frames = int(plan["target_frame_count"] or 0)
        contract = type(self).frame_contract
        length = contract.smallest_legal_at_least(target_frames)
        if length is None:
            raise _wb.GraphExecutionError(
                "%s was asked for %d frame(s); its only legal length is %d "
                "(%.2f s at %d fps). A longer beat is PARTITIONED into chained "
                "segments by the coverage planner, never stretched here"
                % (self.name, target_frames, R.LTX25_FRAMES,
                   R.LTX25_FRAMES / float(R.LTX25_FPS), R.LTX25_FPS))

        classes = dict(getattr(self, "_classes", None)
                       or _wb.resolve_graph_classes(self._node_candidates()))
        image_name = _wb.stage_into_comfy_input(plan["init_image"])
        graph = self._build_graph(plan, image_name, length, width, height)

        _LOG.info(
            "[OTR video] %s PLAN dit=%s quant=%s canvas=%dx%d frames=%d "
            "steps=%d sampler=%s cfg=%.1f/%.1f/%.1f anchor=%.1f",
            self.name, os.path.basename(str(self._dit_name())),
            self._quant_label(), width, height, length, R.LTX25_STEPS,
            R.LTX25_SAMPLER, R.LTX25_CFG_VIDEO, R.LTX25_CFG_AUDIO,
            R.LTX25_CFG_MODALITY, R.LTX25_I2V_ANCHOR_STRENGTH)

        # LOAD PREFLIGHT. The canonical residue-freer, not a private
        # reimplementation: its own module docstring calls it "the single
        # canonical residue-freer", and it evicts the writer LLM and Bark --
        # out-of-band caches ``comfy.model_management`` cannot see and that
        # nothing else in the video path releases. Best-effort by contract; it
        # never raises, so a failure here degrades residency, never the render.
        try:
            from .._otr_vram_levers import free_otr_pipeline_residue
            free_otr_pipeline_residue(reason="%s load preflight" % self.name)
        except Exception as exc:  # noqa: BLE001 - residency is best-effort
            _LOG.warning("[%s] pre-load residue free failed (non-fatal): %s",
                         self.name, exc)

        # ``keep`` holds the two MODEL nodes (retained for V-4 teardown) and the
        # terminal. Everything else -- crucially the text encoder -- is dropped
        # by its last consumer.
        results = images = None
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(graph, classes, free_after_use=True,
                                    keep={"unet", "modality", self._TERMINAL})
            images = results[self._TERMINAL][0]
        finally:
            peak = probe.stop()
            if results is not None:
                self._retain_model_patchers(results, prepared)
            _wb.reclaim_idle_models(reason="%s post-decode" % self.name)
        if images is None:
            raise _wb.GraphExecutionError(
                "%s: run_graph produced no terminal image" % self.name)

        frames = _wb.images_to_uint8(images)
        # THE PIPELINE INVARIANT, before any trim. The graph was asked for
        # exactly ``length`` frames, so any other count means a node snapped
        # somewhere we did not plan for -- and finding that out after the trim
        # would be indistinguishable from a trim bug.
        if len(frames) != length:
            raise _wb.GraphExecutionError(
                "%s asked its graph for %d frame(s) and decoded %d. NO "
                "FALLBACK -- padding the difference is how a render that did "
                "not happen gets counted as one" % (self.name, length,
                                                    len(frames)))

        # THE TAIL TRIM, in real delivered frames: drop the surplus in order.
        # No mirror, no loop, no held frame. An ask at or above the rung is left
        # whole, and an ask of 0 (an unsized beat) is not a trim instruction.
        if 0 < target_frames < length:
            frames = frames[:target_frames]
            _LOG.info(
                "[OTR video] %s tail trim: delivered %d of %d frame(s) "
                "(ratio %.3f) @ %dx%d", self.name, len(frames), length,
                len(frames) / float(length), width, height)

        out_path = otr_engine_tmp_mp4("otr_%s_" % self.name)
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path,
                                                  self.target_fps)
        # PROVE the silent bt709/yuv420p contract on the file we just wrote
        # (G5.1). This lane's model genuinely produces audio, so "silent by
        # construction" is exactly the kind of claim that survives a graph edit
        # it should not survive.
        validate_silent_clip_contract(ffprobe_clip_fields(path),
                                      self.target_fps)
        if peak:
            _LOG.info("[%s] render-window VRAM peak: %d MB", self.name,
                      int(peak))

        return {
            "out_path": path, "frame_count": n, "vram_peak_mb": peak,
            "recipe": "ltx_2_5_golden_i2v_foley",
            "unet": os.path.basename(str(self._dit_name())),
            "quant": self._quant_label(), "use_lora": False,
            "canvas": "%dx%d" % (width, height),
            # THE HONESTY RECEIPTS, and ``native_frame_count`` is EQUAL to the
            # delivered count on this lane -- always, trimmed or not.
            #
            # THIS WAS WRONG IN THE FIRST DRAFT and the bug is worth recording
            # because no test caught it. It stamped the pre-trim RUNG, so a
            # 50-frame delivery advertised 97 native frames. ``acceptance.py``
            # is explicit that manufactured frames are APPENDED AT THE TAIL and
            # a segment's real frames are the prefix ``[0, native)``, so
            # native > delivered is not merely odd -- it is an over-claim of
            # exactly the shape a padding lane makes, and the grader compares
            # the two directly. This lane manufactures NOTHING: it renders the
            # rung and cuts surplus real frames off the end, so every frame
            # delivered is genuine and native == delivered is the only true
            # statement. ``tests/test_frame_receipt_conformance.py`` could not
            # see it because its ``_RAW`` stub never exercises the trim branch.
            "native_frame_count": n,
            "extension_mode": "none",
        }

    def canonicalize(self, raw, request, profile):
        """The silent CanonicalClip dict, with silence PROVED at the seam where
        the literal is written -- not inherited from the render call.

        A declaration checking a declaration proves nothing (lesson L4), and
        this lane is precisely the case the lesson was written for: the model
        emits audio and ComfyUI ships a decoder for it. Re-probing here means
        ``has_audio: False`` is a fact about the bytes on disk.
        """
        from . import wrapper_bridge as _wb  # noqa: F401 - error type parity
        raw = raw or {}
        path = raw.get("out_path", "")
        if path:
            validate_silent_clip_contract(ffprobe_clip_fields(path),
                                          self.target_fps)
        return self._clip_from_raw(raw, request)

    def _clip_from_raw(self, raw, request):
        """PURE: shape a render result into the silent CanonicalClip dict.

        SPLIT OUT OF ``canonicalize`` DELIBERATELY, and the registry-walk that
        forced it was right to. ``tests/test_frame_receipt_conformance.py``
        drives EVERY registered engine through this exact hook with a stub raw
        and no file on disk, so that a new lane cannot ship a clip row nothing
        can grade -- the fourth armed-consumer-without-a-producer defect in one
        week is what put that walk there. With the shaping inlined in
        ``canonicalize`` the walk could only reach it by going through the
        ffprobe, which cannot run on a stub path, so this lane's receipts were
        effectively untestable off-box.

        The division of labour is the point: this method is pure and provable
        on CPU, and ``canonicalize`` above it does the one thing that needs a
        real file -- proving the silence it is about to write down.
        """
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        path = raw.get("out_path", "")
        return {
            "clip_id": get("shot_id") or get("request_id") or "ltx25_clip",
            "type": "video", "path": path,
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            "vram_peak_mb": raw.get("vram_peak_mb"),
            "recipe": raw.get("recipe"), "unet": raw.get("unet"),
            "quant": raw.get("quant"), "use_lora": raw.get("use_lora"),
            "render_canvas": raw.get("canvas"),
            "native_frame_count": raw.get("native_frame_count"),
            "extension_mode": raw.get("extension_mode"),
        }

    def _retain_model_patchers(self, results, prepared):
        """V-4: keep the MODEL patchers the graph produced so teardown can
        ``detach(unpatch_all=True)`` them. Best-effort, never raises."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "modality"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an asset ref that may be a bare string
        OR a mapping carrying a ``path`` key."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _canvas_dims(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        return int(c_get("w", 0) or 0), int(c_get("h", 0) or 0)

    def _build_render_request(self, request):
        """Pure: the normalized inference request this graph consumes."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        assets = get("asset_refs") or {}
        init_image = (assets.get("init_image") or ""
                      if isinstance(assets, dict) else "")
        return {
            "init_image": self._ref_path(init_image),
            "text_prompt": get("text_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
        }


__all__ = ["Ltx25VideoEngine", "LTX25_RESERVED_SIBLING_IDS"]
