"""LTX 2.5 Distilled -- the silent video lane, and the foley lanes beside it.

ONE graph, several lanes. The beat's own wide scene still drives a 97-frame
first stage at 832x480, the selected HQ stage doubles that latent and re-anchors
the same still, and the only picture decode is 1664x960.

EVERY LANE HERE LOADS NATIVE SAFETENSORS THROUGH STOCK ComfyUI LOADERS
(``UNETLoader`` / ``CLIPLoader``). There is no third-party node dependency and no
gated weight: the DiT and the Gemma-4 12B text encoder come from
joeygambino/LTX-2.5-Quantized, which serves them anonymously.

* ``ltx25_video`` DISCARDS the model's audio side at ``LTXVSeparateAVLatent`` --
  it never resolves ``LTXVAudioVAEDecode`` at all -- so the clip is silent and
  only ``OTR_MasterAudioMux`` ever adds audio. It is the 16 GB silent lane, on
  the same mix4x8 DiT the 16 GB foley and mime lanes load.
* The foley lanes KEEP it: the audio latent is decoded to a WAV SIDECAR and mixed
  with the episode master at that same mux, at a fixed 0.50 / 0.50 (operator,
  2026-08-29; raised from 0.20 / 0.80). The per-segment mp4s stay silent and
  prove it; the BEAT clip additionally gains an AAC preview of the same audio at
  assembly (see the FoleyPlus class docstring), which the silent composite
  strips -- so the mux remains the only node that puts audio into the EPISODE.
* The mime lanes are the SAME mechanism at 1.00 / 0.00 (generate-and-discard):
  a silent performance carrying the video's own score.

A bed mixed UNDER the master at the mux does not need an execution-order
inversion -- the mux runs after video -- so there are exactly two seams,
``_on_graph_result`` and ``_after_video_graph``, both no-ops on the silent lane,
and they are the smallest pair that lets a sibling avoid copying either
``_build_graph`` or the 200-line ``render_clip``. Per lesson L5 these are
separate INTERNAL engines, never one id with a switch -- two public ids on one
internal id collapses ``_INTERNAL_TO_PUBLIC`` and trips the bijection assert AT
IMPORT, which empties most of the ComfyUI menu.

THE PUBLIC ID OF THE SILENT LANE IS ``ltx25_high_video``. The convention is
``<model><version>_<low|high>_<capability>``.

THE RECIPE IS NOT ON THE TABLE. Every number comes from :mod:`ltx25_recipe`,
which transcribes the lab's locked graph and is drift-gated against the lab's
actual file by ``tests/test_ltx25_recipe_matches_lab_golden.py``. No bigger
canvas, no more frames, no CFG tuning. If a value here turns out to be wrong
that is a finding to REPORT, not a knob to turn.

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
import re
import sys
import threading
import time

from .._otr_shared.still_plan_helpers import StillPlanRow
from .._otr_shared.role_compat import ROLES
from . import motion_common as _MC
from . import ltx25_recipe as R
from .foley_stems import (
    FOLEY_GAIN as LTX25_FOLEY_GAIN,
    FOLEY_RECEIPT_KEYS as LTX25_FOLEY_RECEIPT_KEYS,
    MASTER_GAIN_UNDER_FOLEY as LTX25_MASTER_GAIN_UNDER_FOLEY,
    FoleyStemError,
    samples_per_frame as foley_samples_per_frame,
    sha256_of_file,
    write_pcm16_wav,
)
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_LOG = logging.getLogger("OTR.eng_ltx25")

#: THE FOLEY STEM FORMAT LIVES IN ``foley_stems``, NOT HERE, and that is the
#: whole point of that module. Three stages touch a stem -- this engine WRITES
#: one per rendered segment, the coverage assembler CUTS and concatenates them
#: into one per beat, and ``OTR_MasterAudioMux`` READS and mixes the result --
#: so the format is a contract between three files rather than a detail of this
#: one. The constants come with it: ``FOLEY_GAIN`` / ``MASTER_GAIN_UNDER_FOLEY``
#: are the operator's fixed ruling (0.50/0.50 since 2026-08-29), and a second
#: copy of them here is exactly how two stages come to disagree about a mix. Cold-import clean --
#: ``foley_stems`` is stdlib-only at module scope (V-12).

#: EMPTY, AND DELIBERATELY KEPT. The tuple expresses a contract that still has
#: a test: an id named here is spoken for and must not be registered until it
#: can actually render an episode. The next LTX 2.5 sibling reserves its name
#: here first. An empty tuple says "nothing is pending", which is a different
#: and more useful statement than the symbol having been deleted.
LTX25_RESERVED_SIBLING_IDS = ()

#: The LTX 2.5 weights, by hardware tier. Sizes and the tier
#: notes are the PUBLISHER'S OWN, from joeygambino/LTX-2.5-Quantized's
#: MANIFEST.json, which also carries a sha256 per file:
#:
#:   comfy-w4a8      12.52 GB  "16 GB, no custom node"
#:   comfy-nvfp4     12.50 GB  "Blackwell only"
#:   comfy-fp8_e4m3fn 21.48 GB "24 GB+, widest GPU support"
#:   gemma4 w4a8     10.60 GB  text encoder, 4-bit
#:
#: They are quantisations of the Lightricks bf16 originals, served ungated
#: (verified HTTP 206 anonymously) -- the official repo is gated and needs a
#: token, which is why the mirror is named here.
#: THE 16 GB WEIGHT WAS CHOSEN BY A FOUR-WAY BAKE-OFF, not by file size.
#: One RTX 5080 Laptop, one still, one seed, the full two-stage lane with the
#: decode eviction in place -- the only variable being the DiT:
#:
#:   mix4x8-13.8GB   205.3 s   12.86 GB   quant_format "mixed:w4a8+int8"
#:   w4a8            210.3 s   11.66 GB   quant_format "asym_w4a8_int8"
#:   nvfp4           250.8 s   12.64 GB   BLACKWELL ONLY -- and the slowest
#:
#: mix4x8 IS w4a8 with 386 layers promoted to int8 -- read off the headers,
#: which carry `quant_mixed_hi_layers: 386` on the one and nothing on the
#: other. So it is not a speed-versus-precision trade: the same format family,
#: strictly more precision, and it happened to be fastest too. The 1.2 GB it
#: costs over w4a8 used to matter because the DiT had to share the card with
#: the decode; it does not now that `_make_room_for_decode` evicts first.
LTX25_NATIVE_DIT_16GB = "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors"
LTX25_NATIVE_DIT_BLACKWELL = "LTX25-distilled-DiT-comfy-nvfp4.safetensors"
#: 24 GB, ANY modern NVIDIA. NOT fp8_e4m3fn, which this lane shipped pointing
#: at and which DOES NOT EXIST.
#:
#: The publisher's own MANIFEST.json still advertises
#: ``LTX25-distilled-DiT-comfy-fp8_e4m3fn.safetensors`` at 21.48 GB, noted
#: "24 GB+, widest GPU support", with a sha256 -- and the repository does not
#: contain it. A stat returns exists=false and a find for ``*fp8*`` returns
#: nothing. The lineup was revised (fp8 dropped; int8 and two mix4x8 builds
#: added) and the manifest was never updated. This lane was built from that
#: manifest, so it was registered pointing at a file nobody can download.
#:
#: int8, AND IT WAS PICKED BY RUNNING IT, not by arithmetic.
#:
#: An earlier version of this comment reasoned that int8 "lands near 26.5 GB
#: and does NOT fit a 24 GB card ... int8 is a 32 GB-class file". That was an
#: estimate stated as a fact, and the operator rejected it on exactly those
#: grounds -- "says who, did it crash, did we test on a 24 GB machine". It had
#: not been run and we had not.
#:
#: MEASURED on a rented RTX 4090 (24,564 MiB, Ada, clean box, nothing else on
#: the card), one 97-frame clip through this lane's own graph:
#:
#:      int8         20.03 GB file   peak 23.5 GB   101.3 s   renders
#:      mix4x8-17GB  15.84 GB file   peak 23.1 GB   168.7 s   renders
#:
#: int8 fits with ~0.5 GB to spare and is 40% FASTER, which the static model
#: got backwards in both direction and magnitude -- ComfyUI sizes residency
#: against the card it finds, so weights-plus-activations arithmetic
#: systematically over-predicts. The foley decodes and muxes (3.88 s, 48 kHz
#: stereo AAC) and the operator judged the result good.
#:
#: It is also the PORTABLE choice, which decides the tier on its own: nvfp4 is
#: Blackwell-only, while INT8 tensor cores have shipped since Turing, so this
#: one file serves Ada, Ampere and Blackwell alike. And at 8 bits against
#: nvfp4's 4 it is the higher-precision weight -- the size IS the precision.
LTX25_NATIVE_DIT_WIDE = "LTX25-distilled-DiT-comfy-int8.safetensors"
LTX25_NATIVE_TEXT_ENCODER = "gemma4-12b-ltx25-comfy-w4a8.safetensors"

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
_FLOOR_UPSCALER = int(0.5 * _GiB)

#: Kill switch for the episode-scoped encoder residency below. It is an
#: opt-OUT flag, so the parse is the MIRROR of the ``voice_cast_mode`` opt-IN
#: parse and the reasoning is inverted with it: that one had to reject ``""``
#: and ``"false"`` as ENABLED once its default flipped, whereas here an unset
#: or empty value correctly means "keep the default", and only an explicit
#: disable token turns the cache off.
_ENCODER_CACHE_ENV = "OTR_LTX25_ENCODER_CACHE"
_CACHE_DISABLE_TOKENS = frozenset({"0", "false", "no", "off"})


def _encoder_cache_enabled():
    """ON unless explicitly disabled. Unset and empty both keep the default."""
    return (otr_env.get(_ENCODER_CACHE_ENV, "") or "").strip().lower() \
        not in _CACHE_DISABLE_TOKENS


def _copy_conditioning(out):
    """Hand out a PRIVATE outer list and metadata dicts; share the tensor.

    Verified during the 2026-08-20 arc: nothing on this graph mutates a
    conditioning in place -- ``LTXVConditioning`` goes through
    ``node_helpers.conditioning_set_values`` (``node_helpers.py:9-23``,
    ``n = [t[0], t[1].copy()]``), ``process_conds`` re-lists with
    ``conds[k][:]`` (``samplers.py:1040``), and both
    ``calculate_start_end_timesteps`` and
    ``resolve_areas_and_cond_masks_multidim`` are copy-on-write.

    So this is INSURANCE, not a fix, and it is taken anyway: the guarantee
    depends on upstream ComfyUI internals nobody here controls, the copy is a
    handful of dicts, and the failure it insures against is a silently wrong
    render on every beat after the first rather than a crash.
    """
    if not (isinstance(out, (tuple, list)) and out):
        return out
    cond = out[0]
    # ACCEPT A TUPLE HERE, NOT JUST A LIST. ComfyUI hands back a list today, but
    # a narrower check would SILENTLY SKIP THE COPY if any upstream wrapper ever
    # returned a tuple -- and a guard that quietly stops guarding is worse than
    # no guard, because the receipts still say it is on.
    if not isinstance(cond, (list, tuple)):
        return out
    cloned = [[e[0], dict(e[1])]
              if isinstance(e, (list, tuple)) and len(e) >= 2
              and isinstance(e[1], dict) else e
              for e in cond]
    return (type(cond)(cloned),) + tuple(out[1:])


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


def _assert_two_stage_execution(records, frame_count, *, two_stage=True):
    """Require executor-owned proof of what actually ran.

    `two_stage=False` is the FAST lane: no latent upscale, no refinement, and
    the decode lands at the model's native 832x480 rather than the doubled
    1664x960. The proof still exists -- it just proves a different graph, and
    it still fails closed if the decode is missing or the canvas is wrong.
    Loosening it to "any shape" would have made the fast lane unable to catch a
    silent geometry change, which is the one thing this check is for.
    """
    if two_stage:
        expected = (
            ("latent_upscale", "LTXVLatentUpsampler"),
            ("refine_sampler", "SamplerCustomAdvanced"),
            ("decode", "VAEDecodeTiled"),
        )
    else:
        expected = (("decode", "VAEDecodeTiled"),)
    if not isinstance(records, list) or len(records) != len(expected):
        # THE COUNT COMES FROM `expected`, not a literal 3. The fast lane proves
        # one node, so a hardcoded 3 here would have printed a wrong number in
        # the one message a reader consults when the proof fails.
        raise RuntimeError(
            "ltx25 %s execution proof expected %d node record(s), got %r"
            % ("two-stage" if two_stage else "single-stage", len(expected),
               len(records) if isinstance(records, list) else type(records).__name__))
    for ordinal, (record, wanted) in enumerate(zip(records, expected), 1):
        node_id, class_name = wanted
        if (record.get("node_id"), record.get("class_name"),
                record.get("ordinal")) != (node_id, class_name, ordinal):
            raise RuntimeError(
                "ltx25 two-stage execution proof mismatch at ordinal %d: %r"
                % (ordinal, record))

    shapes = records[-1].get("output_shapes")
    shape = shapes[0] if isinstance(shapes, list) and shapes else None
    want_h = R.LTX25_RENDER_CANVAS_H if two_stage else R.LTX25_CANVAS_H
    want_w = R.LTX25_RENDER_CANVAS_W if two_stage else R.LTX25_CANVAS_W
    legal = (isinstance(shape, list) and len(shape) == 4
             and shape[0] == int(frame_count)
             and shape[1:3] == [want_h, want_w]
             and shape[3] in (3, 4))
    if not legal:
        raise RuntimeError(
            "ltx25 %s decode returned %r; expected [%d,%d,%d,3|4]"
            % ("two-stage" if two_stage else "single-stage", shape,
               int(frame_count), want_h, want_w))
    return True


#: S1 per-model still plan (G7.4). This lane is I2V and the still is NOT
#: optional -- ``LTXVImgToVideoInplace`` at strength 1.0 IS the conditioning, so
#: a beat with no still has no graph. Hence ``required="always"`` on all three
#: scene rows: this lane has no text-only path and must not fall back to one.
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
                     ("cinematic three-quarter framing, the subject shown "
                      "whole with clear space around it inside frame, "
                      "balanced composition")),
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

    Stage one follows the lab golden; stage two follows the accepted
    ``ltx_2_5_two_stage.json`` chain. OTR replaces UI-only constants and save
    nodes, leaves the final audio output unwired (V-1), and uses its own
    silent-mp4 encoder so the emitted file is proved silent rather than merely
    declared so.
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
    engine_version = "2"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = R.LTX25_FPS

    #: UNVERIFIED, and False is the honest default rather than a verdict. This
    #: flag drives the release-gate warning and the release filename tag, never
    #: selection (the H3 precedent). No license attestation exists on this box
    #: for the LTX 2.5 Distilled weights the way the H3 attestation file does
    #: for H3. Claiming clean on
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
    #: recipe-dependent. A lane whose anchor strength varied by recipe could
    #: only declare soft_reference, because a contract that is only sometimes
    #: true is a jump cut.
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

    #: TRUE ON EVERY LANE BUT ONE. The x2 in-graph latent upscale and the
    #: refinement pass after it are the accepted HQ path and stay the default;
    #: a subclass sets this False to decode the stage-one latent at its native
    #: 832x480 instead. NOTHING OVERRIDES IT TODAY: the one lane that did,
    #: `ltx25_native_foley_lowres`, was unregistered 2026-09-23 when the
    #: operator settled that every LTX 2.5 8 GB lane keeps its upscaler --
    #: `otr_8gb_ltx` already gives that tier a faster option on a 2B model
    #: rather than on a degraded 12.86 GB one. The False branch stays proven
    #: in both directions by tests/test_ltx25_every_lane_builds_a_graph.py,
    #: and it is what `_output_canvas` and the conditional upscaler entry in
    #: `_weight_paths` are written against.
    #:
    #: IT BELONGS ON THIS CLASS BECAUSE THIS CLASS READS IT. `_build_graph` is
    #: defined here, so every LTX 2.5 lane reaches the attribute. When it was
    #: first added it sat on a subclass instead, and every lane that did not
    #: inherit that subclass raised AttributeError the moment it built a graph
    #: -- with the entire suite green, because no test built those graphs.
    #: tests/test_ltx25_every_lane_builds_a_graph.py is the guard now.
    _ingraph_upscale = True

    # ---- weight tokens (env pins name a FILE, they cannot make one exist) ----
    #: The DiT this lane loads. Each tier subclass names its own; the base is
    #: the 16 GB weight, which is what the silent ``ltx25_video`` lane renders.
    _native_dit = LTX25_NATIVE_DIT_16GB

    #: Where the text encoder runs, as the stock ``CLIPLoader`` device widget
    #: takes it: ``"cpu"`` or ``"default"`` (the accelerator).
    #:
    #: MEASURED, not chosen by taste. A GPU-side encode of the Gemma-4 12B
    #: encoder is what tips a 16 GB card, so the 16 GB lanes pin it to CPU. On
    #: the RTX PRO 4500 a CPU-placed encode held the process at 701% CPU with
    #: the GPU idle for minutes per beat, so a 24 GB+ lane paying the pin is
    #: spending minutes to protect headroom it already has.
    #:
    #: ``_encoder_cache_expects_cpu`` MUST track this. The cache's liveness
    #: check tests for CPU placement, so a lane that stops pinning and does
    #: not say so writes a cache entry it then rejects on every single read
    #: (the defect found live on 2026-09-21).
    _native_te_device = "cpu"

    #: Whether THIS lane's cache is expected to find its handle on the CPU.
    #: Read by ``_cached_clip_is_live``. A lane that sets ``_native_te_device``
    #: to ``"default"`` sets this False in the SAME change, or its own cache can
    #: never pass its own liveness check.
    _encoder_cache_expects_cpu = True

    def _dit_name(self):
        return otr_env.get("OTR_LTX25_NATIVE_DIT", self._native_dit)

    def _text_encoder_name(self):
        return otr_env.get("OTR_LTX25_NATIVE_TE", LTX25_NATIVE_TEXT_ENCODER)

    def _video_vae_name(self):
        return otr_env.get("OTR_LTX25_VIDEO_VAE", R.LTX25_VIDEO_VAE)

    def _audio_vae_name(self):
        return otr_env.get("OTR_LTX25_AUDIO_VAE", R.LTX25_AUDIO_VAE)

    def _upscaler_name(self):
        return otr_env.get("OTR_LTX25_UPSCALER", R.LTX25_UPSCALER_MODEL)

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
        paths = [
            ("LTX 2.5 DiT",
             _resolve("unet", self._dit_name()), _FLOOR_DIT),
            ("Gemma-4 12B text encoder",
             _resolve("text_encoders", self._text_encoder_name()),
             _FLOOR_TEXT_ENCODER),
            ("LTX 2.5 video VAE", _resolve("vae", self._video_vae_name()),
             _FLOOR_VIDEO_VAE),
            ("LTX 2.5 audio VAE", _resolve("vae", self._audio_vae_name()),
             _FLOOR_AUDIO_VAE),
        ]
        # THE UPSCALER IS ONLY REQUIRED BY A LANE THAT BUILDS ITS LOADER.
        #
        # This list feeds `assert_usable()`, which checks each file exists and
        # clears a size floor and raises EngineUnusable otherwise. The low-res
        # lane drops `upscale_loader` from its graph and its profile's
        # `required_models` accordingly -- so while this entry was
        # unconditional, a fresh install that fetched exactly the declared files
        # had no upscaler and the lane refused to start. Declared six weights,
        # demanded seven. Found by a refutation lane reading the inherited
        # method rather than the diff.
        #
        # The alternative -- put the file back in `required_models` -- would make
        # every install of this lane download 536 MB it never opens, which is
        # the opposite of why the lane exists.
        if self._ingraph_upscale:
            paths.append(
                ("LTX 2.5 latent spatial upscaler",
                 _resolve("latent_upscale_models", self._upscaler_name()),
                 _FLOOR_UPSCALER))
        return paths

    def _output_canvas(self):
        """``(width, height)`` this lane really decodes at.

        THREE PLACES REPORTED THE DOUBLED CANVAS UNCONDITIONALLY -- the PLAN log,
        the tail-trim log and the clip RECEIPT -- so a low-res lane announced
        1664x960 while writing 832x480. The receipt is the one that mattered: it
        is copied into the canonical clip, carried into the manifest by
        `render_driver`, and printed in the episode's CREDITS text by
        `otr_credits_roll`. A viewer would have read a resolution the file does
        not have. Found by a refutation lane grepping RENDER_CANVAS rather than
        reading the diff.
        """
        if self._ingraph_upscale:
            return R.LTX25_RENDER_CANVAS_W, R.LTX25_RENDER_CANVAS_H
        return R.LTX25_CANVAS_W, R.LTX25_CANVAS_H

    def _quant_label(self):
        """The quantisation, read off the DiT filename, for the per-beat
        observability line and the clip's recipe receipt. Pure."""
        base = os.path.basename(str(self._dit_name()))
        for tag in ("nvfp4", "fp8_e4m3fn", "w4a8", "w4a4", "int8", "mix4x8"):
            if tag in base:
                return tag
        return "native"

    # ---- the graph spec (classes resolve through wrapper_bridge) ----
    def _node_candidates(self):
        """Every ComfyUI class this graph needs, by logical node id.

        Several logical nodes intentionally share installed classes (the two
        I2V anchors, AV concat/separate pairs, and samplers). The graph keeps
        their roles distinct while the resolver verifies each installed class.
        """
        return {
            "unet": ("UNETLoader",),
            "te": ("CLIPLoader",),
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
            "upscale_loader": ("LatentUpscaleModelLoader",),
            "latent_upscale": ("LTXVLatentUpsampler",),
            "refine_i2v": ("LTXVImgToVideoInplace",),
            "refine_sigmas": ("ManualSigmas",),
            "refine_concat": ("LTXVConcatAVLatent",),
            "refine_sampler": ("SamplerCustomAdvanced",),
            "refine_separate": ("LTXVSeparateAVLatent",),
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
        for logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 - collect every miss
                absent.append("/".join(candidates))
        if absent:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s -- LTX 2.5 "
                "needs a ComfyUI carrying comfy_extras/nodes_lt.py + "
                "nodes_lt_audio.py. These are STOCK ComfyUI nodes, so update "
                "ComfyUI itself and restart"
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
        before every segment, and hashing a 13.8 GB DiT per segment would
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

    # ---- episode-scoped text-encoder residency -------------------------
    #
    # THE PROBLEM, COUNTED ON A LIVE LEG (2026-08-20): 15 shot renders, 13
    # reads of the Gemma-4 12B text encoder. A ratio of 1:1 --
    # every shot re-read the whole encoder off disk, ~63 s each, on top of a
    # 54.2 s CPU encode. That wall clock is what decides how many episodes
    # reach ``otr/obs/`` in a night.
    #
    # OWNERSHIP IS THE DESIGN, AND IT IS NOT THE OBVIOUS ONE. The cache is
    # STORED on the engine instance but OWNED by an EPISODE, and the reason is
    # that the engine instance is a process-lifetime SINGLETON: the registry
    # builds one adapter at import (``engine_registry_base.py:148-150``) and
    # ``get_engine`` hands back that same object forever. A cache that merely
    # lived on ``self`` would therefore never end.
    #
    # EVERY ENGINE-LEVEL HOOK WAS MEASURED AND ALL OF THEM RUN TOO OFTEN:
    # ``free_otr_pipeline_residue`` is this method's OWN per-shot preflight,
    # ``teardown`` and ``unload`` run once per BEAT via ``BeatSession.close``,
    # and the gpu_residency lease is acquired per beat. Evicting on any of
    # them would drop the cache immediately before the thing that needed it.
    # The real boundary is one layer up in the DRIVER -- ``run_episode`` --
    # which is why ``begin_encoder_scope`` / ``end_encoder_scope`` exist and
    # why they are called from there rather than from any engine hook.
    #
    # NO SCOPE OPEN == TODAY'S BEHAVIOUR, EXACTLY. ``_encoder_scope is None``
    # means every shot loads its own encoder, which is what a direct
    # ``render_single`` diagnostic or a unit test gets. Caching is a property
    # of running an EPISODE, not of being this engine.
    #: ``None`` = no episode owns the cache. A dict = an episode has claimed it
    #: and is responsible for releasing it. Class-level default so the
    #: registry's zero-arg construction stays cheap and no ``__init__``
    #: override is needed; every write assigns on the INSTANCE.
    _encoder_scope = None
    #: HOW MANY OWNERS HOLD THE SCOPE OPEN. This is a REFCOUNT, not a flag, and
    #: the reason is that the engine is a process-wide singleton while each
    #: ``run_episode`` keeps its OWN private ownership map.
    #:
    #: THE BUG THIS PREVENTS, caught by the r3 panel: the shipped
    #: `/otr/video_render_single` and `/otr/video_render_soak` routes start
    #: UNGUARDED DAEMON THREADS (``__init__.py:493-515``), and the soak route
    #: reaches ``run_episode``. With a bare flag, two overlapping runs both see
    #: "already open", and whichever finishes FIRST sets the scope to ``None``
    #: -- so the survivor's remaining beats run cold AND its private map still
    #: records the engine as opened, so it never reopens. Silent, and it
    #: restores one 8.86 GiB reload per beat for the rest of that episode.
    #:
    #: Sharing one cached encoder between overlapping episodes is safe on its
    #: own terms -- the key is the weight file, so both runs want the same
    #: CLIP. What was never safe was the LIFETIME, and that is what the count
    #: fixes. (Concurrent heavy rendering is separately governed by the
    #: cross-process gpu_residency lease and the sequential-execution rule;
    #: this refcount does not attempt to make it safe, only to stop the cache
    #: from being stranded or released out from under a live owner.)
    _encoder_scope_owners = 0
    #: Bumped on every OPEN and on every hard RELEASE, so a close owed by a
    #: previous owner can be recognised as stale instead of landing on the
    #: current scope.
    _encoder_generation = 0
    #: The lifecycle is read-modify-write on shared state and the shipped
    #: `/otr/video_render_*` routes run in UNGUARDED DAEMON THREADS
    #: (``__init__.py:493-515``), so `owners += 1` across two threads can lose
    #: an update. Class-level because the registry holds exactly one instance.
    _encoder_lock = threading.Lock()

    def begin_encoder_scope(self):
        """Claim the encoder cache for ONE episode. Never raises.

        Called by ``render_driver.run_episode``. Repeated calls from the SAME
        run are absorbed by the driver's own ownership map; repeated calls from
        DIFFERENT overlapping runs each take a reference, and the cache lives
        until the last one lets go.

        RETURNS AN OPAQUE TOKEN the caller must hand back to
        :meth:`end_encoder_scope`. ``None`` means no scope was taken.
        """
        if not _encoder_cache_enabled():
            return None
        with type(self)._encoder_lock:
            if self._encoder_scope is None:
                self._encoder_scope = {}
                self._encoder_generation = int(self._encoder_generation) + 1
                _LOG.info("[%s] encoder cache scope OPEN (episode-owned, "
                          "generation %d)", self.name, self._encoder_generation)
            self._encoder_scope_owners = int(self._encoder_scope_owners) + 1
            return self._encoder_generation

    def end_encoder_scope(self, token=None):
        """Drop ONE owner's claim; release the cache when the last one goes.

        NEVER raises -- it runs from a ``finally``. This is the only thing
        standing between "one 8.86 GiB load per episode" and "8.86 GiB resident
        until the server restarts".

        ``token`` IS WHAT MAKES A STALE CLOSE HARMLESS, and it exists because
        the r4 panel found a failure needing no thread race at all::

            begin(A)                  # generation 1, owners 1
            release_encoder_cache()   # kill switch: scope gone, owners 0
            begin(B)                  # a NEW scope
            end(A)                    # ...closed B's scope, not A's

        Nothing distinguished A's outstanding close from B's live ownership.
        Now a release BUMPS THE GENERATION, so A's token no longer matches and
        its close is a logged no-op. ``token=None`` keeps the old unconditional
        behaviour for any caller that does not carry one.
        """
        with type(self)._encoder_lock:
            if token is not None and token != self._encoder_generation:
                _LOG.info("[%s] ignoring a stale encoder-scope close "
                          "(token %s, current generation %s)",
                          self.name, token, self._encoder_generation)
                return
            self._encoder_scope_owners = max(
                0, int(self._encoder_scope_owners) - 1)
            if (self._encoder_scope_owners == 0
                    and self._encoder_scope is not None):
                self._encoder_scope = None
                _LOG.info("[%s] encoder cache scope CLOSED; the episode's text "
                          "encoder is released", self.name)

    def release_encoder_cache(self):
        """Drop the cache NOW regardless of owners. Never raises.

        Separate from :meth:`end_encoder_scope` on purpose: that one is a
        refcount decrement and must not be used to mean "off". This is the
        kill-switch path -- when the operator sets
        ``OTR_LTX25_ENCODER_CACHE=0`` mid-run, the 8.86 GiB goes, and a
        decrement would merely have taken one reference off a scope that
        several owners still hold open.
        """
        with type(self)._encoder_lock:
            had = self._encoder_scope is not None
            self._encoder_scope = None
            self._encoder_scope_owners = 0
            # BUMP THE GENERATION so every outstanding token is now stale and
            # the closes still owed by live owners cannot land on whatever
            # scope opens next. Without this the kill switch is itself the ABA
            # bug it was meant to be an escape hatch from.
            self._encoder_generation = int(self._encoder_generation) + 1
            if had:
                _LOG.info("[%s] encoder cache RELEASED by kill switch "
                          "(generation now %d)", self.name,
                          self._encoder_generation)

    def _encoder_cache_key(self):
        """Identity of the LOADED encoder, or ``None`` to force a MISS.

        ``None`` IS A REAL RETURN VALUE AND IT CLOSES A COLLISION. The obvious
        key is ``_weight_receipt``, but that returns ``("<unresolved>", -1, -1)``
        for an unresolvable path and ``(basename, -1, -1)`` on ``OSError`` --
        so two DIFFERENT broken states can compare EQUAL and produce a false
        HIT against a cache entry that has nothing to do with the weight now on
        disk. A broken stat is not an identity; it is the absence of one, and
        it is answered with a guaranteed miss.

        Carries ``st_dev``/``st_ino`` and the real path on top of size and
        mtime because this key now spans a whole episode rather than one
        segment: a swapped symlink target or a same-size same-mtime file at a
        different inode is exactly the case a longer-lived cache has to notice
        and the per-segment receipt never had to.
        """
        name = str(self._text_encoder_name())
        path = _resolve("text_encoders", name)
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (os.path.realpath(path), st.st_dev, st.st_ino, st.st_size,
                st.st_mtime_ns, name, "ltxv", ("cpu", "cpu"))

    @staticmethod
    def _cached_clip_is_live(out, expect_cpu=True):
        """Re-assert the placement invariant THIS LANE actually promises.

        Structural only -- no tensor work. A False here does NOT raise: it
        drops the cache and falls through to a full load, which asks the stock
        loader for the lane's placement again through its ``device`` widget.

        ``expect_cpu`` DEFAULTS TRUE so every existing call site, and every
        existing test that calls this UNBOUND on the class, keeps its exact
        original behaviour. The one caller that owns a real instance passes
        its own ``_encoder_cache_expects_cpu`` -- see that flag's docstring
        for why the two cannot drift apart without breaking the lane that
        declined the pin.

        A lane that declined the pin makes no placement promise at all: the
        model may legitimately sit on whichever device ComfyUI's own memory
        manager put it, and that device can change between beats without the
        handle being stale. So the only thing left to verify there is that
        the handle is still a REAL, materialized model -- not a specific
        device.
        """
        clip = out[0] if isinstance(out, (tuple, list)) and out else None
        patcher = getattr(clip, "patcher", None)
        if clip is None or patcher is None:
            return False
        if getattr(patcher, "model", None) is None:
            return False
        load_dev = str(getattr(patcher, "load_device", None) or "")
        off_dev = str(getattr(patcher, "offload_device", None) or "")
        if not expect_cpu:
            return bool(load_dev) and bool(off_dev)
        return "cpu" in load_dev and "cpu" in off_dev

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
        rather than after the first long beat: an earlier LTX lane reached a
        live render gate without one and refused there, 730 seconds
        into a leg -- the most expensive possible place to find out. The full
        suite caught it here instead, on a lane that had never rendered.

        PRE-LOAD STABLE by contract: read once before the weights land, again
        after ``prepare()``, and before every segment, so it may only describe
        things that do not change across the load. Every per-segment value --
        prompt, seed, frame count, canvas, the still -- is deliberately absent.
        There is no recipe token to carry either, unlike lanes with more than
        one recipe: this lane has exactly one locked recipe, so the weights
        ARE the identity.

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

        The DiT and both VAEs are loaded by their loader nodes inside the
        graph, so ``free_after_use`` can drop each one after its last consumer.

        THE TEXT ENCODER IS NO LONGER ALWAYS ONE OF THEM, and this docstring
        said it was. Since the episode-scoped cache (2026-08-20) the ``te``
        loader runs only on a cache MISS -- once per EPISODE rather than once
        per shot. On a HIT the handle arrives as an ``external_results`` entry
        and no loader node exists in the graph at all.

        **What did NOT change is the VRAM story**, which is the part worth not
        re-deriving: the encoder is pinned to CPU and the lab's own peak
        decomposition puts it at 0.0 GiB at the sampling peak, so keeping it
        resident costs system RAM (mmap-backed page cache), never headroom on
        the card. This lane fits because the DiT is 9.80 GiB of a 14.48 GiB
        peak, not because the encoder is evicted.
        """
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    # ---- the graph ----
    def _build_graph(self, plan, image_name, length, width, height):
        """The selected HQ recipe as a declarative graph. Pure: no GPU, no weights.

        Stage one is node-for-node against
        ``ltx_2_5_golden_i2v_foley.json``; the terminal chain is node-for-node
        against ``ltx_2_5_two_stage.json``. The deliberate OTR departures are:

        * ``FloatConstant`` (lab node 18) is a LITERAL here. It existed only to
          fan 25.0 out to three consumers; a node that carries a constant is a
          UI convenience, not part of the recipe.
        * ``LTXVAudioVAEDecode`` (lab node 34) is NEVER WIRED. That is V-1, and
          it is the entire difference between this lane and Chunk B.
        * ``CreateVideo`` + ``SaveVideo`` (lab 35, 75) are replaced by this
          repo's own encoder, so the emitted file can be PROVED silent by
          ffprobe instead of declared silent by a literal (G5.1, lesson L4).

        The HQ structural addition is exactly the accepted lab chain: separate
        the first AV sample, 2x latent upscale, plant the SAME role-specific
        still at full strength, reuse the same noise/guider/sampler with the
        three-step refine schedule, separate again, then perform the graph's
        ONLY video decode at 1664x960. ONE ``VAELoader`` feeds both anchors,
        the upscaler, and the final decode (lab node 2).

        A DRAFT OF THIS ADAPTER SPLIT THAT LOADER IN TWO AND IT WAS CUT, which
        is worth recording because the split looks obviously right and is not.
        The idea, inherited by resemblance from an earlier LTX audio lane, was that a
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

        # MOTION BAKED IN (2026-08-27). "a vintage radio broadcast scene" was
        # a scene NOUN -- no verb anywhere -- so the model's cheapest answer
        # was a pan across furniture. The default now names visible WORK, and
        # the actions are chosen to be foley-bearing on purpose (dials,
        # papers, switches make sound) so the same string serves the picture
        # on ltx25_video and both halves of the joint latent on foley/mime.
        positive = plan.get("text_prompt") or (
            "a vintage radio broadcast scene in motion, an operator working "
            "the console, hands turning dials and shuffling papers")
        seed = int(plan.get("seed", 0) or 0)
        fps = float(R.LTX25_FPS)

        graph = {
            # --- loaders ---
            "unet": {"class": "unet", "inputs": {
                "unet_name": self._dit_name(),
                # 'default' lets ComfyUI honour whatever the checkpoint
                # declares. The quantisation lives IN the file; forcing a
                # dtype here would fight it.
                "weight_dtype": "default"}},
            "te": {"class": "te", "inputs": {
                "clip_name": self._text_encoder_name(), "type": "ltxv",
                # Stock CLIPLoader takes a placement widget, so the pin is
                # an ordinary value. Per tier: small cards pin, big cards
                # do not.
                "device": self._native_te_device}},
            "videovae": {"class": "videovae",
                         "inputs": {"vae_name": self._video_vae_name()}},
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": self._audio_vae_name()}},
            "upscale_loader": {"class": "upscale_loader", "inputs": {
                "model_name": self._upscaler_name()}},

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

            # --- selected HQ stage two ---
            "separate": {"class": "separate",
                         "inputs": {"av_latent": W("sampler", 0)}},
            "latent_upscale": {"class": "latent_upscale", "inputs": {
                "samples": W("separate", 0),
                "upscale_model": W("upscale_loader", 0),
                "vae": W("videovae", 0)}},
            # LTXVLatentUpsampler deliberately drops noise_mask. The full-
            # strength second anchor recreates it while planting the same still
            # into the doubled latent, matching the lab graph that made the HQ
            # acceptance video.
            "refine_i2v": {"class": "refine_i2v", "inputs": {
                "vae": W("videovae", 0), "image": W("preprocess", 0),
                "latent": W("latent_upscale", 0), "bypass": False,
                "strength": R.LTX25_I2V_ANCHOR_STRENGTH}},
            # Comfy registers core's V3 ManualSigmas before external custom
            # nodes and protects that name from duplicate replacement. Its
            # direct Python parameter is therefore `sigmas`.
            "refine_sigmas": {"class": "refine_sigmas", "inputs": {
                "sigmas": R.LTX25_REFINE_SIGMAS}},
            "refine_concat": {"class": "refine_concat", "inputs": {
                "video_latent": W("refine_i2v", 0),
                "audio_latent": W("separate", 1)}},
            "refine_sampler": {"class": "refine_sampler", "inputs": {
                "noise": W("noise", 0), "guider": W("guider", 0),
                "sampler": W("ksel", 0), "sigmas": W("refine_sigmas", 0),
                "latent_image": W("refine_concat", 0)}},
            "refine_separate": {"class": "refine_separate", "inputs": {
                "av_latent": W("refine_sampler", 0)}},
            # Decode VIDEO ONLY. refine_separate slot 1 is the audio latent and
            # stays unwired, preserving V-1 while stage two sharpens the video.
            "decode": {"class": "decode", "inputs": {
                "samples": W("refine_separate" if self._ingraph_upscale
                             else "separate", 0),
                "vae": W("videovae", 0),
                "tile_size": R.LTX25_STAGE2_DECODE_TILE_SIZE,
                "overlap": R.LTX25_STAGE2_DECODE_OVERLAP,
                "temporal_size": R.LTX25_STAGE2_DECODE_TEMPORAL_SIZE,
                "temporal_overlap": R.LTX25_STAGE2_DECODE_TEMPORAL_OVERLAP}},
        }

        if not self._ingraph_upscale:
            # THE FAST LANE: drop stage two entirely rather than leave it
            # wired-but-unused. The decode above already reads `separate`, so
            # leaving these in place would have run the upscaler and the
            # refinement sampler and then discarded both -- the slowest
            # possible way to render at low resolution, which is the exact
            # opposite of why this lane exists.
            for stage_two in ("upscale_loader", "latent_upscale", "refine_i2v",
                              "refine_concat", "refine_sampler",
                              "refine_separate"):
                graph.pop(stage_two, None)

        return graph

    # ---- the two Chunk B seams (2026-08-26) ----
    #
    # THE MODULE DOCSTRING USED TO SAY THERE IS NO HOOK, and it was right to at
    # the time: "a subclass hook designed today would be a doorway into a wing
    # with no foundation". The foundation now exists (the foley bed mixes at
    # ``OTR_MasterAudioMux``, which runs AFTER video), so these two exist -- and
    # they are deliberately the SMALLEST pair that lets a sibling keep the
    # model's own audio without copying either ``_build_graph`` or the
    # 200-line ``render_clip``.
    #
    # BOTH ARE NO-OPS HERE. ``ltx25_video`` stays byte-identical in behaviour:
    # the first returns nothing and the second returns an empty dict, so the
    # silent lane's receipt literal is exactly what it always was.

    def _on_graph_result(self, node_id, out):
        """Called for EVERY executed node, while its output is still alive.

        Pure side effect; the return value is ignored. A sibling lane that
        needs an intermediate the graph is about to free copies it HERE --
        see ``_harvest`` for why anywhere later is a use-after-free."""

    def _after_video_graph(self, *, results, prepared, request, out_path,
                           frame_count) -> dict:
        """Extra receipt keys to fold into ``render_clip``'s return.

        Runs after ``reclaim_idle_models`` has dropped the DiT and after
        ``validate_silent_clip_contract`` has proved the mp4 -- which is the
        first moment ``out_path`` and ``frame_count`` both exist, so a sibling
        can prove its sidecar against the exact file it belongs to.

        ``{}`` on the silent lane, and nothing about that lane changes."""
        return {}

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
            "[OTR video] %s PLAN dit=%s quant=%s source=%dx%d output=%dx%d frames=%d "
            "steps=%d sampler=%s cfg=%.1f/%.1f/%.1f anchor=%.1f",
            self.name, os.path.basename(str(self._dit_name())),
            self._quant_label(), width, height, self._output_canvas()[0],
            self._output_canvas()[1], length, R.LTX25_STEPS,
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

        # RESIDENCY: reuse this episode's text encoder and its empty negative.
        # Both ride ``run_graph``'s existing ``external_results`` contract --
        # ids that are legal Wire sources, are never re-executed, and are added
        # to ``keep`` so ``free_after_use`` cannot evict them. That transport is
        # already proven by the 25 tests in
        # ``tests/test_video_wrapper_bridge_external_results.py``; what is new
        # here is only WHO owns the handles and for how long.
        #
        # THE KILL SWITCH ALSO EVICTS. Flipping OTR_LTX25_ENCODER_CACHE=0 in a
        # long-running server must not leave 8.86 GiB pinned by a scope opened
        # before the flip, so a disabled cache closes any open scope rather
        # than merely declining to read it.
        external = {}
        cache_on = _encoder_cache_enabled()
        if not cache_on:
            # HARD release, not a refcount decrement -- see
            # ``release_encoder_cache``. "Off" means the memory goes, not that
            # one of several owners lets go.
            self.release_encoder_cache()
        key = self._encoder_cache_key() if cache_on else None
        scope = self._encoder_scope
        # ``"clip" in scope`` RATHER THAN BARE TRUTHINESS. An open-but-empty
        # scope is the normal state on beat 0 and after an invalidation, and
        # ``if scope:`` reads that correctly only because the dict happens to be
        # empty. The moment anything ever stores a bookkeeping key at open time,
        # bare truthiness would walk an entry that has no encoder in it. Ask the
        # question actually being asked: is there a cached CLIP here?
        if scope and "clip" in scope:
            # A MISS MUST *RELEASE*, NOT MERELY DECLINE TO READ. Leaving a
            # keyed entry in place while its replacement loads holds TWO
            # 8.86 GiB encoders at once, and if this graph then fails the stale
            # one survives to the end of the episode. So an unusable entry is
            # dropped the moment it is known to be unusable -- the SCOPE stays
            # open (the episode still owns it), only its contents go.
            stale = None
            if key is None:
                stale = "the encoder weight is unresolvable or unstattable"
            elif scope.get("key") != key:
                stale = "the encoder weight changed under the episode"
            elif not self._cached_clip_is_live(
                    scope.get("clip"), expect_cpu=self._encoder_cache_expects_cpu):
                stale = "the cached encoder failed its CPU-placement check"
            if stale and scope.get("clip") is not None:
                _LOG.warning("[%s] dropping the cached text encoder -- %s",
                             self.name, stale)
            if stale:
                scope.clear()
            else:
                external["te"] = scope["clip"]
                graph.pop("te", None)
                if scope.get("neg") is not None:
                    external["neg"] = _copy_conditioning(scope["neg"])
                    graph.pop("neg", None)
        _LOG.info("[%s] encoder cache %s / negative %s (scope %s)", self.name,
                  "HIT" if "te" in external else "MISS",
                  "HIT" if "neg" in external else "MISS",
                  "open" if self._encoder_scope is not None else "none")

        # ``keep`` holds the two MODEL nodes (retained for V-4 teardown) and the
        # terminal.
        #
        # THE TEXT ENCODER'S FATE NOW DEPENDS ON THE CACHE, and this comment
        # used to say flatly that it "is dropped by its last consumer" -- true
        # before the episode cache and false after. On a MISS the ``te`` node is
        # in the graph and IS dropped from ``results`` once ``pos`` and ``neg``
        # have consumed it, exactly as before -- but ``_harvest`` has already
        # taken a reference, so the object survives to be published. On a HIT
        # the node is not in the graph at all; the handle arrives as an
        # external, and ``run_graph`` adds every external to ``keep``, so
        # ``free_after_use`` cannot touch it.
        results = images = None
        #: Harvested DURING the graph, published only if it finishes. ``te`` is
        #: dropped from ``results`` by ``free_after_use`` the moment ``pos`` and
        #: ``neg`` have consumed it, so catching it as it lands is the only way
        #: to hold it at all -- but see the publish block for why landing here
        #: is not the same as being kept.
        pending = {}

        # WHICH node feeds the decode, read off the graph rather than named.
        # The two-stage path wires ``refine_separate`` into it and a single-
        # stage one would wire ``separate``; asking the graph keeps the seam
        # correct for whichever it built, and silent (no eviction) if the shape
        # ever changes out from under it.
        _decode_feeder = None
        try:
            _samples = (graph.get(self._TERMINAL, {})
                        .get("inputs", {}).get("samples"))
            _decode_feeder = getattr(_samples, "src", None)
        except Exception:                  # noqa: BLE001 -- telemetry, never fatal
            _decode_feeder = None

        def _harvest(node_id, out):
            if node_id in ("te", "neg"):
                pending[node_id] = out
            # SEAM 1 (the foley bed, 2026-08-26). A NO-OP on this lane.
            #
            # IT FIRES HERE AND NOWHERE LATER, AND THAT IS THE WHOLE POINT.
            # ``run_graph`` calls ``on_result`` while the node's output is
            # still in ``results``, and ``free_after_use`` deletes
            # ``refine_separate`` the moment its last consumer has run -- it is
            # not in ``keep``. A subclass that stashed a reference now and
            # dereferenced it after ``reclaim_idle_models`` below would be
            # reading freed VRAM, which is a crash or silent garbage rather
            # than a saving. The foley lane copies to CPU inside this call.
            self._on_graph_result(node_id, out)
            # SEAM 2 (2026-09-22): MAKE ROOM FOR THE DECODE, AND ONLY IF THERE
            # IS NOT ANY.
            #
            # The terminal decode needs ~8.1 GB (R.LTX25_STAGE2_DECODE_NEEDS_MB,
            # measured). The sampler's DiT is 12-13 GB and has no further
            # consumer once the node feeding the decode has landed. On a 16 GB
            # card those two cannot coexist, so ComfyUI streams the decode and
            # it takes 400+ s instead of 34.7 s -- measured both ways on this
            # 5080, same graph, same weight, same seed.
            #
            # FIRES ONLY WHEN THE CARD IS ACTUALLY SHORT. A 32 GB box has the
            # room and evicting there would buy nothing while costing the next
            # beat a reload, which is the one real hazard of the bigger hammer.
            # The test is free VRAM against a measured need -- not a card-size
            # tier, because the thing that matters is whether the decode fits,
            # and that depends on what else this process is holding.
            if node_id == _decode_feeder:
                self._make_room_for_decode()
                # BRACKET THE DECODE, NOT THE GRAPH. An earlier cut of this
                # reset the counter before `run_graph` and read it after, so
                # the window covered the sampler's 12-13 GB DiT as well and the
                # figure it printed was the sampler's high-water, not the
                # decode's. Feeding THAT back into
                # `LTX25_STAGE2_DECODE_NEEDS_MB` would have made the decode's
                # need sampler-sized and evicted on every card forever. Caught
                # by a contrarian review before any leg acted on it.
                #
                # This branch fires on the node FEEDING the decode, which is
                # immediately before the decode runs, so the reset here plus
                # the read below measures the decode and the decode only.
                try:
                    import torch as _t_dec
                    if _t_dec.cuda.is_available():
                        _t_dec.cuda.reset_peak_memory_stats()
                except Exception:           # noqa: BLE001 -- telemetry only
                    pass
            if node_id == self._TERMINAL:
                try:
                    import torch as _t_dec2
                    if _t_dec2.cuda.is_available():
                        _LOG.info(
                            "[OTR video] %s: DECODE PEAK -- allocated %.0f MB, "
                            "reserved %.0f MB, measured across the decode node "
                            "alone. THIS is the figure comparable to "
                            "LTX25_STAGE2_DECODE_NEEDS_MB (%s MB), which was "
                            "measured on a 16 GB card and never on 8 GB.",
                            self.name,
                            _t_dec2.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                            _t_dec2.cuda.max_memory_reserved() / (1024.0 * 1024.0),
                            getattr(R, "LTX25_STAGE2_DECODE_NEEDS_MB", "?"))
                except Exception:           # noqa: BLE001 -- telemetry only
                    pass

        execution_records = []
        graph_started = time.perf_counter()
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(
                graph, classes, free_after_use=True,
                keep={"unet", "modality", self._TERMINAL},
                external_results=external, on_result=_harvest,
                audit_node_ids=(
                    {"latent_upscale", "refine_sampler", "decode"}
                    if self._ingraph_upscale else {"decode"}),
                execution_records=execution_records)
            images = results[self._TERMINAL][0]
        finally:
            render_elapsed_s = time.perf_counter() - graph_started
            peak = probe.stop()

            if results is not None:
                self._retain_model_patchers(results, prepared)
            _wb.reclaim_idle_models(reason="%s post-decode" % self.name)
        if images is None:
            raise _wb.GraphExecutionError(
                "%s: run_graph produced no terminal image" % self.name)
        try:
            _assert_two_stage_execution(
                execution_records, length, two_stage=self._ingraph_upscale)
        except RuntimeError as exc:
            raise _wb.GraphExecutionError(str(exc)) from exc
        # THE LINE SAYS WHICH GRAPH RAN, because this is the line a reader
        # greps to find out what a lane actually did. It used to hardcode
        # "TWO-STAGE PASS nodes=3" and the doubled canvas, which on a
        # single-stage lane would have printed a confident description of a
        # graph that never ran.
        _LOG.info(
            "[OTR video] %s %s nodes=%d decode=%dx%d render_elapsed_s=%.3f",
            self.name,
            "TWO-STAGE PASS" if self._ingraph_upscale else "SINGLE-STAGE PASS",
            3 if self._ingraph_upscale else 1,
            R.LTX25_RENDER_CANVAS_W if self._ingraph_upscale else R.LTX25_CANVAS_W,
            R.LTX25_RENDER_CANVAS_H if self._ingraph_upscale else R.LTX25_CANVAS_H,
            render_elapsed_s)

        # PUBLISH ON GRAPH SUCCESS, and only whole.
        #
        # THE BOUNDARY IS NAMED PRECISELY BECAUSE A REVIEW LANE CORRECTED AN
        # EARLIER, LOOSER CLAIM HERE. This point is "the graph produced a
        # terminal image" -- NOT "the render completed": the frame-count
        # invariant, the tail trim and the ffprobe silent-clip proof all still
        # lie below. Publishing here means the CONDITIONING and the ENCODER are
        # known good, which is all this cache holds; a later ffprobe failure
        # says nothing about either.
        #
        # NOT from inside ``on_result``, which is where it wants to go. That
        # hook fires the instant each node lands, so publishing there commits a
        # PARTIAL transaction -- state harvested from a graph that then died.
        # (An earlier version of this comment justified that with "holding
        # 8.86 GiB into the retry"; there IS no retry -- ``render_shot``
        # classifies and re-raises, ``render_driver.py:3435-3447``. The real
        # cost of publishing early is committing state from an unsuccessful
        # graph; the real cost of publishing late is one extra load after a
        # failure. The second is the cheaper mistake.)
        #
        # WHOLE, TOO: a scope that published ``te`` without ``neg`` would take
        # the encoder reload off the clock while still paying the negative
        # encode every shot -- a half-cache whose logs read like a working one.
        scope = self._encoder_scope
        if key is not None and scope is not None:
            clip_out = pending.get("te") or external.get("te")
            neg_out = pending.get("neg") or scope.get("neg")
            if clip_out is not None and neg_out is not None:
                scope.update({"key": key, "clip": clip_out, "neg": neg_out})

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
                len(frames) / float(length), self._output_canvas()[0],
                self._output_canvas()[1])

        out_path = otr_engine_tmp_mp4("otr_%s_" % self.name)
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path,
                                                  self.target_fps)
        # PROVE the silent bt709/yuv420p contract on the file we just wrote
        # (G5.1). This lane's model genuinely produces audio, so "silent by
        # construction" is exactly the kind of claim that survives a graph edit
        # it should not survive.
        validate_silent_clip_contract(ffprobe_clip_fields(path),
                                      self.target_fps)
        _LOG.info("[%s] render-window VRAM peak: %s MB", self.name,
                  int(peak) if peak is not None else "unknown")

        raw = {
            "out_path": path, "frame_count": n, "vram_peak_mb": peak,
            "render_elapsed_s": render_elapsed_s,
            "recipe": R.LTX25_TWO_STAGE_RECIPE_ID,
            "unet": os.path.basename(str(self._dit_name())),
            "quant": self._quant_label(), "use_lora": False,
            "canvas": "%dx%d" % self._output_canvas(),
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
        # SEAM 2. Empty on this lane, so the dict above IS the return here.
        raw.update(self._after_video_graph(
            results=results, prepared=prepared, request=request,
            out_path=path, frame_count=n))
        return raw

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
            "render_elapsed_s": raw.get("render_elapsed_s"),
            "native_frame_count": raw.get("native_frame_count"),
            "extension_mode": raw.get("extension_mode"),
        }

    #: MEASURED on the 4060: 8188 MiB total, 7399 MB free with the card
    #: otherwise empty -- so a CUDA context and the desktop hold ~789 MB that
    #: `mem_get_info` never offers us. There used to be a
    #: `_VRAM_CONTEXT_RESERVE_MB = 300` here; nothing ever read it. It existed
    #: to pad a refusal, and refusing on a predicted number was removed the
    #: same day, so it is deleted rather than left looking load-bearing. The
    #: measurement is the part worth keeping.

    def _settled_free_vram_mb(self, floor_mb, tries=12, pause_s=1.0):
        """Free VRAM once the release has actually landed.

        `reset_cast_buffers()` returns before the driver has handed the memory
        back. MEASURED on the 4060 at 1 s sampling: 6469 -> 4421 -> 2821 -> 821
        -> 789 MiB, so the figure was still falling three seconds after the
        call returned. Reading immediately saw 4940 MB when the settled value
        was ~7399 -- under-reporting a successful eviction by 2.5 GB, which is
        enough to refuse a card that had in fact freed plenty.

        Polls until the reading stops rising and STAYS there. Costs about
        four seconds on any card -- the old version claimed a big card "pays
        nothing", which was true and was exactly the bug: it exited on the
        first plateau, and on the 4060 the first plateau was 2.5 GB short.
        """
        import time
        # SAY WHAT IT SAW. Three cuts at the exit condition have all returned
        # the same 4940 MB on the 4060 -- four runs, three commits, identical
        # to the megabyte -- while that card's TRUE floor moved more than 2 GB
        # between legs (789 MiB used on one, 2821 MiB on the next). A reading
        # that is merely EARLY tracks the floor and varies with it. One that
        # does not move at all while the underlying quantity swings 2 GB is not
        # sampling the floor, and a fourth guess at the predicate will not find
        # out why.
        #
        # Nobody has yet seen the sequence this loop observes on that card: an
        # external nvidia-smi trace cannot see inside the twelve tries. So it
        # logs every reading with its timestamp, once per call, and the next
        # leg answers the question instead of extending it. The 4060 box asked
        # for exactly this, and it is right that four attempts have been
        # reasoned about and none measured.
        trace = []
        t0 = time.monotonic()

        def _done(value, why):
            _LOG.info("[OTR video] %s: settle %s -> %s | readings: %s",
                      self.name, why,
                      "None" if value is None else "%.0f MB" % value,
                      " ".join(trace) or "(none)")
            return value

        best = None
        flat = 0
        rose = False
        while True:
            now = _MC.free_vram_mb()
            trace.append("%.1fs=%s" % (time.monotonic() - t0,
                                       "None" if now is None else "%.0f" % now))
            if now is None:
                # A failed probe is not an answer: `floor_mb` is the UNSETTLED
                # pre-eviction reading this function exists to avoid, so one
                # bad sample must not republish it. Skip and keep waiting.
                tries = int(tries) - 1
                if tries <= 0:
                    return _done(best if best is not None else floor_mb,
                                 "gave up on unreadable card")
                time.sleep(float(pause_s))
                continue
            if best is None:
                best = now
                flat = 0
            elif now > best + 1.0:
                best = now
                flat = 0
                rose = True                 # the release is demonstrably live
            elif now >= best - 50.0:
                # At or near the high-water mark: a genuine plateau. RISE
                # FIRST, THEN TWO FLATS -- the opening reading is taken while
                # the release is still in flight, so a plateau never
                # interrupted by a rise proves only that nothing has happened
                # yet.
                best = max(best, now)
                flat += 1
                if rose and flat >= 2:
                    return _done(best, "settled (rise then two flat)")
                if flat >= 3 and not rose:
                    # Nothing is coming: three still reads at 1 s is longer
                    # than the release takes to begin, so this is an eviction
                    # that freed nothing rather than one still in flight.
                    return _done(best, "never moved")
            else:
                # A big DROP is not a plateau -- something else took memory, or
                # the release is still moving. Keep waiting.
                flat = 0
            tries = int(tries) - 1
            if tries <= 0:
                # The cap reports what is free NOW, not the high-water mark: a
                # peak seen once while a sustained collapse goes unmentioned is
                # the wrong number to hand a decode that is about to run.
                return _done(now if now is not None else best, "hit the cap")
            time.sleep(float(pause_s))

    def _total_vram_mb(self):
        """Card capacity, or None. `free_vram_mb` reads it and throws it away."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.mem_get_info()[1]) / (1024.0 * 1024.0)
        except Exception:                   # noqa: BLE001 -- telemetry only
            return None
        return None

    def _make_room_for_decode(self):
        """Evict before the terminal decode, IF the card is short.

        MEASURED ON A 16 GB RTX 5080, same graph, same weight, same seed, the
        only difference being this call:

            decode, transformer still resident   400+ s, killed, never finished
            decode, after this eviction           34.7 s
            whole two-stage render               196.3 s, foley muxed

        WHAT ACTUALLY HOLDS THE MEMORY, corrected 2026-09-23 after a Fable
        review refuted the first explanation written here. `unload_all_models`
        DOES own a staged model: `detach()` -> the dynamic `unpatch_model`
        override (`comfy/model_patcher.py:2132`) -> `partially_unload_ram(1e32)`
        for the pinned host memory and `partially_unload(None, 1e32)` ->
        `vbar.free_memory(1e32)` for the staged VRAM. The earlier claim that it
        "never owned the staged weights" was wrong.

        What it does NOT own is the aimdo CAST BUFFERS -- one reservation per
        offload stream in `STREAM_AIMDO_CAST_BUFFERS`, which grows and never
        shrinks. On an 8 GB card that is most of what stays behind, and only
        `reset_cast_buffers()` clears it. PROVEN on the 4060 at 1 s sampling:
        6469 -> 789 MiB, about 5.7 GB returned in three seconds, where the same
        card without this call sat at a flat 7910 MiB for sixteen minutes.

        ORDER MATTERS AND IS NOT OBVIOUS. `reset_cast_buffers` runs FIRST
        because its per-model half walks `current_loaded_models`
        (`model_management.py:1453`) and `unload_all_models` POPS that list
        (`:910`). Calling unload first leaves an empty list and that half
        silently does nothing. Found by a cursor review.

        WHY THE 5080 NEVER NEEDED THE SECOND CALL: there the transformer FITS,
        so it is resident and patcher-owned and the ordinary unload frees it.
        Eviction and streaming are mutually exclusive by construction -- on the
        card where you most need to evict, the thing you are reaching for is
        not held where you are looking."""
        free_mb = _MC.free_vram_mb()
        if free_mb is None:
            return                          # CPU box / unit tests: nothing to evict
        need = getattr(R, "LTX25_STAGE2_DECODE_NEEDS_MB", 8300)
        if free_mb >= need:
            _LOG.info(
                "[OTR video] %s: decode has room (%.0f MB free >= %d MB "
                "needed); leaving models resident", self.name, free_mb, need)
            return

        reset_ok = False
        try:
            import comfy.model_management as mm
            reset = getattr(mm, "reset_cast_buffers", None)
            if callable(reset):
                reset()                     # FIRST -- see the docstring on order
                reset_ok = True
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as exc:            # noqa: BLE001 -- a saving, never a failure
            _LOG.warning(
                "[OTR video] %s: could not evict before decode (%r); the "
                "decode will run under whatever is resident", self.name, exc)
            return

        # TELEMETRY MUST NOT KILL THE GRAPH. This call sits outside the
        # eviction's own try/except, and it sleeps and calls into torch -- a
        # raise here would escape `_harvest` and take the render with it, over
        # a number that only reaches a log line.
        #
        # `Exception`, NOT `BaseException`, and the distinction is load-bearing
        # rather than stylistic. ComfyUI declares
        # `InterruptProcessingException(BaseException)`
        # (`comfy/model_management.py`) specifically so that the `except
        # Exception` handlers scattered through node code cannot swallow a
        # user's Cancel. Catching BaseException here is the one thing that
        # would put this helper in that category. It happens not to intercept
        # Cancel today, because nothing in this call path reaches one of
        # ComfyUI's interrupt checkpoints -- but that is an accident of what
        # the helper calls, not a property to rely on, and BaseException also
        # eats KeyboardInterrupt on the synchronous diagnostic paths.
        #
        # AND AN EARLIER COMMENT HERE CLAIMED THIS GUARDS A DRIVER WEDGE. It
        # does not and cannot: a wedged `torch.cuda.synchronize()` never
        # returns and never raises, so no `except` clause of any breadth
        # rescues the thread. The try cap bounds the polling loop; nothing
        # bounds the synchronize, and that is an accepted risk, not a handled
        # one.
        try:
            after_mb = self._settled_free_vram_mb(free_mb)
        except Exception as exc:            # noqa: BLE001 -- telemetry only
            _LOG.warning(
                "[OTR video] %s: could not settle the free-VRAM reading (%r); "
                "reporting the unsettled figure", self.name, exc)
            after_mb = free_mb
        total_mb = self._total_vram_mb()     # already best-effort internally
        _LOG.info(
            "[OTR video] %s: EVICTED before decode -- %.0f MB free -> %.0f MB "
            "(needed %d, total %s, aimdo cast buffers %s)", self.name, free_mb,
            after_mb if after_mb is not None else -1.0, need,
            ("%.0f MB" % total_mb) if total_mb else "unknown",
            "reset" if reset_ok else "NOT AVAILABLE on this ComfyUI")

        if after_mb is None or after_mb >= need:
            return

        # NEVER REFUSE ON A NUMBER. ONLY AN OOM DECIDES.
        # (operator directive, 2026-09-23 -- hard.) In his words: "we should
        # never refuse on numbers, only OOM decided... I don't want any probes
        # flagging and dying for VRAM measurements, an OOM is the final
        # determinant."
        #
        # THIS FUNCTION EARNED THAT RULING. A predicted-shortfall refusal
        # shipped here and was wrong three times in one afternoon, every time
        # in the pessimistic direction:
        #   * `need` was 8300 MB against a card holding 8188 MiB in TOTAL, so
        #     it refused an empty 4060 -- unsatisfiable by arithmetic.
        #   * it read `after_mb` before the release had landed and saw 4940 MB
        #     when the settled figure one second later was ~7399 MB,
        #     under-reporting its own success by 2.5 GB.
        #   * so it refused a decode having just freed 5.7 GB, and we never
        #     learned whether that decode would have worked.
        #
        # THE ASYMMETRY IS THE WHOLE ARGUMENT. A real OOM names the allocation
        # that failed and the size it wanted: ground truth, and it says exactly
        # how far to shrink a tile. A numeric refusal produces a number the
        # guard invented about a render that never ran. And the expensive
        # failure this guard was built against -- the 968 s thrash -- was
        # residency, which the eviction above now fixes; it was never an OOM.
        #
        # So this is a WARNING and nothing more. The decode proceeds, and CUDA
        # gets to answer.
        # AN UNREACHABLE NEED IS NOT A SHORTFALL (2026-09-23, found by a
        # contrarian review of this very rewrite). `need` is 8300 MB, measured
        # on a free 16 GB card. A 4060 holds 8188 MiB in TOTAL, so on that card
        # `after_mb >= need` can never be true and this warning fired on EVERY
        # successful eviction -- including the one that returned 5.7 GB. The
        # rewrite above exists to stop a wrong number misdirecting a diagnosis,
        # and then graded the right number against an impossible bar, which
        # reads as failure just the same. Same defect as the refusal this
        # ruling removed, wearing a warning's clothes.
        if total_mb is not None and need > total_mb:
            _LOG.warning(
                "[OTR video] %s: this card cannot reach the measured need at "
                "any residency -- %.0f MB total against %d MB needed, and "
                "%.0f MB free after eviction. Nothing is wrong with the "
                "eviction; the decode geometry is too big for this card. The "
                "answer is a smaller decode, not more eviction. Proceeding: "
                "only a real OOM decides, and its error names the allocation "
                "to shrink to.",
                self.name, total_mb, need, after_mb)
            return
        _LOG.warning(
            "[OTR video] %s: decode starts SHORT -- %.0f MB free against %d MB "
            "measured need. Proceeding anyway: only a real OOM decides. If "
            "this OOMs, the error names the allocation that failed and that is "
            "the number to shrink the decode geometry to.",
            self.name, after_mb, need)

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


class Ltx25FoleyPlusEngine(Ltx25VideoEngine):
    """LTX 2.5 Distilled, KEEPING the audio the model already computed.

    Identical picture to ``ltx25_video`` -- same locked recipe, same two-stage
    graph, same 97-frame rung -- plus the one thing that lane throws away: the
    audio latent at ``refine_separate`` slot 1. It is decoded to a WAV sidecar
    and mixed WITH the episode master at ``OTR_MasterAudioMux``, at the fixed
    0.50 / 0.50 the operator ruled on 2026-08-29 (raised from the 2026-08-26
    0.20 / 0.80 after the bed proved inaudible by ear).

    THIS IS NOT THE SFX BED AND THE TWO MUST NEVER BE CONFLATED. The SFX bed
    was separately GENERATED effects from a dedicated model; it was ripped on
    2026-08-06 and is staying dead. This is the video model's OWN output --
    footsteps, room tone and a score computed for the exact picture it is
    rendering, which is the whole reason a joint AV model earns its keep.
    Operator: *"sfx bed is different than foley bed, i won't get the two
    confused."* Every field here is ``foley_``; ``sfx_`` is guarded by
    ``tests/test_rip_sfx_bed_guard.py``.

    THE DECODE IS A SECOND GRAPH, AND THAT IS A VRAM CONTRACT RATHER THAN A
    STYLE CHOICE. Wiring ``LTXVAudioVAEDecode`` into ``_build_graph`` would
    make the audio VAE a SECOND remaining consumer, so ``free_after_use`` could
    no longer drop it before sampling -- and the measured peak is 14.48 GiB
    against a 14.5 GiB clamp, which is 0.02 GiB of headroom. So the latent is
    copied to CPU as it lands, the video graph tears down completely, and only
    then does a two-node graph (``VAELoader`` + ``LTXVAudioVAEDecode``) run.
    The parent's ``_build_graph`` is UNTOUCHED, which is also what keeps
    ``tests/test_ltx25_recipe_matches_lab_golden.py`` true.

    THE STEM IS EXACTLY AS LONG AS THE MP4 BESIDE IT. The engine emits
    ``frame_count`` frames of audio -- the same count as the file it just
    wrote, after the parent's tail trim -- and nothing else. ``drop_head`` and
    ``keep_frames`` on a CHAINED beat belong to the coverage assembler, which
    applies them in sample space; doing it here as well would cut picture-
    locked audio twice.

    WHAT THIS LANE DOES *NOT* DO. The ENGINE does not touch the mp4: every
    per-segment mp4 stays silent, proves it with ffprobe, and ``has_audio``
    stays False on the segment row. The BEAT-level clip is a different story
    since 2026-08-29 (operator: a joint-AV clip he cannot hear is a failed
    render): after the coverage assembler cuts the beat stem, the render
    driver muxes that same audio INTO the beat mp4 as an AAC preview track
    (``foley_stems.mux_native_audio_into_beat_clip``) and flips the beat
    row's ``has_audio`` to True. The WAV sidecar stays the authoritative mix
    source, and ``OTRSilentComposite`` re-encodes every row with ``-an``, so
    the preview track can never reach the episode master -- V-1's real
    guarantee, "only ``OTR_MasterAudioMux`` puts audio into the EPISODE",
    holds exactly as before. Mime is the same mechanism at 1.00 / 0.00.

    NOT REGISTERED ITSELF. The per-tier subclasses below are the selectable
    lanes; this class exists so a foley fix lands on every tier at once. Its
    ``name`` is the family label its log lines and error messages carry until
    a tier overrides it.
    """

    name = "ltx25_foley_plus"
    engine_version = "1"

    #: SELECTABLE, NEVER A DEFAULT. Inherited from the parent and restated
    #: because it matters more here: this lane changes the EPISODE MASTER, so
    #: acquiring it by inheritance rather than by an operator choice would
    #: quietly re-mix an episode nobody asked to re-mix.
    default_roles = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: The CPU copy of ``refine_separate`` slot 1, harvested during the
        #: video graph and CONSUMED (set back to None) by the decode below, so
        #: a latent can never be spent twice or leak into the next segment.
        self._pending_audio_latent = None

    # ---- the extra classes this lane resolves ----
    def _node_candidates(self):
        """The parent's graph plus the two nodes the SECOND graph needs.

        Declared here rather than resolved ad hoc inside the decode so
        ``assert_usable`` gates on them at preflight, BY NAME, on a box whose
        ComfyUI predates ``nodes_lt_audio.py`` -- the parent already walks this
        dict and collects every miss. Harmless to the first graph: ``run_graph``
        iterates the GRAPH and only ever looks classes up by id.
        """
        cands = dict(super()._node_candidates())
        cands["audio_vae_loader"] = ("VAELoader",)
        cands["audio_decode"] = ("LTXVAudioVAEDecode",)
        return cands

    def _build_graph(self, plan, image_name, length, width, height):
        """The parent's graph, verbatim. The override exists ONLY to clear the
        harvest slot at the start of every render, so a latent left behind by a
        failed segment can never be decoded under the next segment's picture."""
        self._pending_audio_latent = None
        return super()._build_graph(plan, image_name, length, width, height)

    # ---- seam 1: copy the audio latent while it is still alive ----
    def _on_graph_result(self, node_id, out):
        """Copy ``refine_separate`` slot 1 to CPU, IMMEDIATELY.

        A ComfyUI LATENT is a DICT (``{"samples": tensor, ...}``, plus a
        ``noise_mask`` when the sampler carried one), not a tensor -- calling
        ``.cpu()`` on the dict would raise and abort the parent's graph. So the
        dict is rebuilt with every tensor inside it detached and copied.

        Copied rather than referenced because ``free_after_use`` deletes this
        node the moment its last consumer has run: a reference read after
        ``reclaim_idle_models`` is freed VRAM, which is a crash or silent
        garbage rather than a saving.
        """
        # WHICH NODE CARRIES THE AUDIO DEPENDS ON THE LANE. Stage two
        # re-samples the joint latent, so on an HQ lane the audio to keep is
        # `refine_separate`'s. The fast lane has no stage two at all, and its
        # audio is the stream `separate` already carries -- reading the
        # stage-two name there would match nothing, the harvest would never
        # fire, and the episode would come out with no foley bed and no error
        # to say why.
        want = "refine_separate" if self._ingraph_upscale else "separate"
        if node_id != want or not out or len(out) < 2:
            return
        self._pending_audio_latent = self._latent_to_cpu(out[1])

    @staticmethod
    def _latent_to_cpu(latent):
        """A CPU copy of one LATENT dict, DETACHED FROM THE GRAPH'S EXEC MODE.

        Nested tensors are left NESTED on purpose: ``LTXVAudioVAEDecode``
        already resolves them (``if audio_latent.is_nested: ... unbind()[-1]``,
        which takes the REFINED portion stage two concatenated), so unbinding
        here would duplicate handling the installed node does better.

        THE ``inference_mode(False)`` IS THE WHOLE POINT OF THE COPY, and it
        was added after a cursor QA lane caught the omission. This harvest runs
        from ``on_result``, which ``run_graph`` now calls INSIDE
        ``torch.inference_mode()`` -- and a clone taken inside that mode is
        itself an inference tensor, so the copy would inherit the lifetime of
        the graph it exists to outlive. Torch's own error text names the escape
        ("You can make a clone to get a normal tensor"), and the escape only
        works with the mode off. Measured both ways: clone inside -> inference
        tensor, clone under ``inference_mode(False)`` -> normal and mutable.

        It is not load-bearing TODAY -- the only consumer is the audio decode
        graph, which runs inside the mode anyway -- and that is exactly why it
        is worth stating rather than leaving to luck. A durable CPU copy that
        survives its graph should not carry that graph's execution flags, and
        the next edit that touches this latent should not have to know why it
        cannot be written to.
        """
        import torch
        if not isinstance(latent, dict):
            raise TypeError(
                "the LTX 2.5 foley harvest expected a LATENT dict from "
                "LTXVSeparateAVLatent slot 1 and got %r. NO GUESS -- decoding "
                "the wrong object is how a lane ships noise as foley"
                % (type(latent).__name__,))
        out = {}
        with torch.inference_mode(False):
            for key, val in latent.items():
                if hasattr(val, "detach") and hasattr(val, "cpu"):
                    out[key] = val.detach().cpu().clone()
                else:
                    out[key] = val
        return out

    # ---- seam 2: decode, conform to the mp4, write the durable stem ----
    def _after_video_graph(self, *, results, prepared, request, out_path,
                           frame_count):
        from . import wrapper_bridge as _wb

        latent = self._pending_audio_latent
        self._pending_audio_latent = None      # consume: never spendable twice
        if latent is None:
            raise _wb.GraphExecutionError(
                "%s finished its video graph without harvesting an audio "
                "latent. NO FALLBACK -- this lane exists to keep the model's "
                "own audio, and a silent stem is indistinguishable from a "
                "working one everywhere downstream" % self.name)

        classes = dict(getattr(self, "_classes", None)
                       or _wb.resolve_graph_classes(self._node_candidates()))
        # TWO NODES, AND THE LATENT ARRIVES AS AN EXTERNAL. ``external_results``
        # is the only injection path into ``run_graph``, and an id present in
        # BOTH the graph and the externals is a named error there rather than a
        # silent precedence rule -- which is why nothing here produces
        # ``audio_latent``.
        graph = {
            "audio_vae_loader": {"class": "audio_vae_loader", "inputs": {
                "vae_name": self._audio_vae_name()}},
            "audio_decode": {"class": "audio_decode", "inputs": {
                "samples": _wb.Wire("audio_latent", 0),
                "audio_vae": _wb.Wire("audio_vae_loader", 0)}},
        }
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        started = time.perf_counter()
        try:
            decoded = _wb.run_graph(
                graph, classes, terminal="audio_decode", free_after_use=True,
                external_results={"audio_latent": (latent,)})[0]
        finally:
            decode_peak = probe.stop()
            # The audio VAE is ~348 MiB and its decode runs with the DiT
            # already gone, so this reclaim is hygiene rather than headroom --
            # but it is the same discipline the video graph keeps, and a VAE
            # left resident across a chained beat is 348 MiB of nothing.
            _wb.reclaim_idle_models(reason="%s post-foley-decode" % self.name)
        decode_elapsed_s = time.perf_counter() - started

        stem_path, samples, channels, sample_rate = self._write_foley_stem(
            decoded, out_path, int(frame_count))
        duration_s = samples / float(sample_rate)
        # THE VRAM RECEIPT IS THE POINT OF THIS LINE, not the elapsed time. The
        # two-pass split exists to keep the audio VAE out of the sampling peak,
        # and the number that proves it is the peak measured HERE, after
        # teardown -- so it is logged on every beat rather than measured once.
        _LOG.info(
            "[OTR video] %s FOLEY decode: %d sample(s) x%dch @%d Hz "
            "(%.3f s over %d frame(s)) decode_peak_mb=%s elapsed_s=%.3f -> %s",
            self.name, samples, channels, sample_rate, duration_s,
            int(frame_count), int(decode_peak) if decode_peak is not None else "unknown",
            decode_elapsed_s, os.path.basename(stem_path))
        return {
            "foley_path": stem_path,
            "foley_sha256": sha256_of_file(stem_path),
            "foley_samples": int(samples),
            "foley_sample_rate": int(sample_rate),
            "foley_channels": int(channels),
            "foley_duration_s": float(duration_s),
        }

    def _write_foley_stem(self, decoded, out_path, frame_count):
        """Conform the decoded AUDIO to exactly ``frame_count`` frames and write
        it into the durable episode audio tree.

        THE LENGTH CONTRACT, and everything downstream rests on it: the stem
        carries ``frame_count * samples_per_frame`` samples -- the same picture
        the mp4 shows, to the sample. A surplus is CUT (the parent's tail trim
        already dropped those frames from the video); a shortfall under one
        frame is silence-padded and SAID SO; anything larger is a refusal,
        because a stem materially shorter than its picture means the decode
        disagreed with the sampler about how long the clip is.

        WRITTEN STRAIGHT TO THE DURABLE DIRECTORY, never to tmp.
        ``persist_episode_clips`` moves only ``clip['path']`` and ``mux()``
        sweeps ``_shared/tmp`` after muxing, so a stem staged in scratch is
        deleted before the mux that needs it ever opens it.
        """
        import numpy as _np

        from . import wrapper_bridge as _wb

        if not isinstance(decoded, dict) or "waveform" not in decoded:
            raise _wb.GraphExecutionError(
                "%s: LTXVAudioVAEDecode returned %r rather than an AUDIO dict"
                % (self.name, type(decoded).__name__))
        sample_rate = int(decoded.get("sample_rate") or 0)
        try:
            spf = foley_samples_per_frame(sample_rate, self.target_fps)
        except FoleyStemError as exc:
            raise _wb.GraphExecutionError("%s: %s" % (self.name, exc)) from exc

        wave = decoded["waveform"]
        arr = (wave.detach().cpu().float().numpy() if hasattr(wave, "detach")
               else _np.asarray(wave, dtype=_np.float32))
        while arr.ndim > 2:            # (B, C, n) -> (C, n)
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]

        want = int(frame_count) * spf
        have = int(arr.shape[-1])
        if want <= 0:
            raise _wb.GraphExecutionError(
                "%s was asked to write a foley stem for a %d-frame clip"
                % (self.name, int(frame_count)))
        if have > want:
            arr = arr[:, :want]
        elif have < want:
            short = want - have
            if short > spf:
                raise _wb.GraphExecutionError(
                    "%s decoded %d audio sample(s) for a %d-frame clip that "
                    "needs %d (%d short, over one whole frame of %d). NO "
                    "FALLBACK -- padding that much silence would hide a decode "
                    "that disagrees with the sampler about the clip's length"
                    % (self.name, have, int(frame_count), want, short, spf))
            _LOG.info(
                "[OTR video] %s foley stem was %d sample(s) under one frame; "
                "silence-padded to the picture length", self.name, short)
            arr = _np.concatenate(
                [arr, _np.zeros((arr.shape[0], short), dtype=arr.dtype)],
                axis=-1)

        dest = self._foley_dir() / (
            os.path.splitext(os.path.basename(out_path))[0] + "_foley.wav")
        samples, channels = write_pcm16_wav(dest, arr, sample_rate)
        return str(dest), samples, channels, sample_rate

    def _foley_dir(self):
        """``<episode>/audio/foley/``, resolved by the one owner of that answer.

        Delegated to ``foley_stems.durable_foley_dir`` rather than reimplemented
        so the ENGINE writes its per-segment stems into exactly the directory
        the coverage assembler later writes the beat stem into and the mux later
        reads both out of. Its refusal is re-raised as a graph error because
        that is the vocabulary this call site's caller speaks.
        """
        from . import foley_stems as _fs
        from . import wrapper_bridge as _wb
        try:
            return _fs.durable_foley_dir()
        except _fs.FoleyStemError as exc:
            raise _wb.GraphExecutionError("%s: %s" % (self.name, exc)) from exc

    # ---- the receipts survive the parent's closed dict ----
    def _clip_from_raw(self, raw, request):
        """The parent's silent clip row plus the foley receipts.

        The parent builds a CLOSED literal, so ``foley_*`` keys on ``raw`` die
        there unless they are copied out explicitly -- present-key-only, so a
        row that never had them does not acquire six nulls.

        ``has_audio`` STAYS FALSE and the mp4 stays silent. The foley is a
        SIDECAR; invariant V-1 is unchanged, and ``OTR_MasterAudioMux`` is
        still the only node that ever puts audio into a video.
        """
        clip = super()._clip_from_raw(raw, request)
        raw = raw or {}
        for key in LTX25_FOLEY_RECEIPT_KEYS:
            if key in raw:
                clip[key] = raw[key]
        return clip


# ---------------------------------------------------------------------------
# THE JOINT-AV POSITIVE FINISHER (2026-08-26)
# ---------------------------------------------------------------------------
#
# LTX 2.5 conditions the PICTURE and the AUDIO from ONE shared positive string:
# `_build_graph` hands `plan["text_prompt"]` to a single CLIPTextEncode and the
# guider samples both halves of the joint latent from it. There is no second
# audio-prompt channel to write to, so the only place to ask for sound is the
# end of the visual prompt.
#
# THE SHAPE IS THE LAB'S, NOT AN INVENTION. The VRAM lab's Golden Action Foley
# recipe (`vram-recipe-lab`, LTX_2_5_ON_16GB.md:183-195) proved this pattern on
# a live render: one string naming a VISIBLE event, the MATCHED sound that event
# makes, then an explicit refusal of speech -- "No speech, no voices, pure
# action." The lab rated foley and score the model's strong suit precisely
# because it scores the scene it is already drawing. This generalises that.
#
# WHY THE SOUND STAYS TIED TO "the visible action". The picture reads these same
# tokens. Ask for "instrumental score" on its own and the model is entitled to
# draw an orchestra; naming the sound as a property of the action already on
# screen is what stops the audio request becoming a second subject.
#
# `ltx25_video` is deliberately absent: it DISCARDS the audio latent, so an
# audio clause there would steer the picture for a track nobody keeps.

#: The lanes that KEEP the model's own audio. Exact internal ids -- never a
#: prefix match, because ``ltx25_video`` shares the prefix and must not finish.
_JOINT_AV_ENGINES = ("cloud_ltx25_foley_plus",)

#: THE INVARIANT TERMINATOR. Every finished joint-AV positive ends with this
#: exact clause, and the golden recipes are why it is worded this plainly.
#:
#: DO NOT "STRENGTHEN" IT. A previous attempt replaced it with a longer
#: prohibition and added seven more voice tokens to the conditioning, which is
#: the opposite of the goal -- text encoders handle negation badly, so naming
#: SPEECH more often makes speech MORE likely. The lab's good results use this
#: form, in this position, on this model. It is short on purpose.
_NO_VOICE_CLAUSE = "No speech, no voices."

#: How the named sounds are seated once they are chosen. "close and dry" asks
#: for the sound of a room the action is happening IN, rather than a scored or
#: reverberant mix the drama's own mixer would then have to fight.
#:
#: A TAG, NOT A CLAUSE (2026-08-28). This lane READS ITS PROMPT ALOUD, and a
#: transcribed stem quoted the old wording -- "close and dry in the room" --
#: back at us. Dropping the preposition costs nothing in conditioning and
#: leaves less that scans as a sentence to deliver.
_SOUND_FRAME = "close dry room tone"

#: At most three named sounds. The golden recipes name two or three; a longer
#: list starts competing with the visual half of the very same string, and on
#: this lane the picture and the audio are decoded from one latent.
_MAX_NAMED_SOUNDS = 3

#: ACTION CUE -> THE SOUNDS THAT ACTION ACTUALLY MAKES.
#:
#: THIS TABLE IS THE WHOLE FIX (operator ruling 2026-08-27). Production used to
#: append "matched environmental foley for the visible action, ambient room
#: tone" -- which names NO SOUND AT ALL and leaves the model to choose one. With
#: a human in frame it chose voice. The golden lab recipes never leave that
#: choice open: they name "heavy wooden thud", "papers rustling", "rain
#: drumming", "mechanical clanking". Naming the sound is the difference.
#:
#: Ordered, and the order is the priority -- the first three distinct matches
#: win, so the most sound-producing actions sit near the top. Matching reads the
#: composed positive, which already contains the beat's action text; that is
#: deliberate and keeps this a PURE function of the prompt with no new ledger
#: field, no writer pass, and no schema change (Option B: "no feature flag, no
#: new schema, no new architecture").
#:
#: CUES MATCH ON A LEADING WORD BOUNDARY, NOT AS BARE SUBSTRINGS, and that is
#: not a theoretical nicety -- a bare-substring pass demonstrably fired on
#: "investigate" (gate -> a slamming door), "grunt" (run -> footsteps) and
#: "sticking" (tick -> a clock). A quiet character beat asking for a slamming
#: door is exactly the kind of wrong cue that makes the whole bed untrustworthy.
#: The boundary is LEADING ONLY, so useful inflections still match: "paperwork"
#: and "papers" both hit `paper`, "steps" hits `step`.
#: INTENSITY WORDING IS DELIBERATE AND MEASURED-IN (operator ruling
#: 2026-08-29). The lab's golden recipes render their audio at ~-25 dB mean;
#: production stems were landing 10-40 dB under that, and the one textual
#: difference was that the goldens say HOW LOUD -- "a LOUD HEAVY wooden thud
#: ECHOING", "papers rustling VIOLENTLY", "rain drumming STEADILY". The
#: phrases below carry that intensity language on purpose. This is the
#: opposite of the damping-words rule and does not conflict with it: damping
#: words are banned from MOTION prompts because they still the picture;
#: intensity words on SOUND phrases are what raise the bed.
_SOUND_LEXICON = (
    # THE IMPACT ROW SITS FIRST, and first is the point: an impact is the
    # loudest, most beat-defining sound an action makes, and priority is
    # lexicon order. This row is also the 2026-08-29 fix for the desk-slam
    # defect: "slams his fist on the desk" used to name WOOD SCRAPING (the
    # old chair/table/desk row matched the noun, unconditionally) -- an
    # impact must never become furniture dragging.
    # "thump" is NOT a cue: leading-boundary it matches "thumping", which
    # prose uses for a HEARTBEAT far more often than for a blow -- the same
    # wrong-cue class as the desk row above. "pound" is kept as the judged
    # call the other way: "pounds the table/door" is the dominant use in
    # ACTION text, and the nonverbal director is instructed to write visible
    # body actions, not interior states.
    (("slam", "pound", "punch", "bang", "smack", "fist"),
     "a loud heavy wooden thud echoing"),
    (("door", "hatch", "gate"), "a door latch clacking and hinges creaking"),
    (("typewriter", "typing", "keyboard"), "typewriter keys striking"),
    (("switch", "lever", "console", "panel", "control"),
     "a sharp switch snap and close mechanical clicks"),
    (("dial", "knob", "tuning"), "close dial ticks and clicks"),
    # "document" is NOT a cue: it matches "documentary", and the
    # `archival_documentary` style pack's video cue is literally "archival
    # documentary", which put papers rustling under every beat of that pack.
    # "paper", "page", "file" and "letter" already carry the real cases.
    (("paper", "page", "letter", "file", "map", "note"),
     "papers rustling violently and pages snapping sharply"),
    (("spark", "electric", "wire", "current", "circuit"),
     "an electric buzz and sparks snapping"),
    (("engine", "motor", "machine", "gear", "piston", "turbine"),
     "heavy machinery clanking under a steady mechanical hum"),
    (("radio", "static", "transmit", "signal", "receiver", "dispatch"),
     "a strong burst of radio static"),
    (("glass", "bottle", "window", "mirror", "jar"), "glass clinking"),
    (("metal", "steel", "iron", "pipe", "chain", "rail"),
     "a hard metal clang and a bright latch snap"),
    (("rain", "storm", "downpour"), "rain drumming steadily"),
    (("wind", "gale", "draft"), "wind moaning"),
    (("fire", "flame", "burn", "match", "lantern"), "fire crackling"),
    (("water", "river", "sea", "wave", "pour"), "water sloshing"),
    (("key", "lock", "bolt"), "keys jangling and a lock turning"),
    (("phone", "telephone", "cradle"), "a receiver clattering onto its cradle"),
    (("bell", "chime"), "a single struck bell"),
    # "watch" is NOT a cue here. It was meant as the wristwatch, but "watches"
    # is far commoner as the VERB -- "she watches the horizon" was asking for
    # a ticking clock in an empty landscape. "clock" and "tick" carry it.
    (("clock", "tick"), "a clock ticking"),
    (("book", "ledger", "volume"),
     "a deep book thump and a stiff cover creak"),
    (("box", "crate", "lid", "case", "trunk"), "a wooden lid thudding"),
    # NO WEAPON CUE, AND THAT IS DELIBERATE (r3 finding, 2026-08-28). The
    # banana route -- "transform every weapon noun", `_otr_banana_route.apply`
    # -- runs AFTER this tail is composed, on a PINNED ordering. Proven live:
    # "the captain raises his revolver toward the hatch" named "a hammer
    # clicking back", and the banana route then rendered "raises his banana".
    # The picture showed a banana while the audio asked for a revolver hammer.
    # This was the ONLY lexicon entry whose subject that route rewrites, so
    # dropping it removes the whole collision class rather than patching a
    # symptom. Re-ordering the seam instead would touch a pinned QA ruling and
    # is recorded as an open design item, not smuggled in here.
    (("horse", "hoof", "hooves"), "hooves striking stone"),
    (("crowd", "street", "market", "platform"), "a crowd murmuring far off"),
    # SCRAPING NEEDS A DRAG VERB NOW (operator ruling 2026-08-29). The old
    # row fired on the FURNITURE NOUNS ("chair", "table", "desk", ...), so
    # any beat set at a desk asked for wood scraping whatever the action
    # was -- including a fist slam, which the impact row above now owns.
    # Scraping is a real sound only when something is actually dragged.
    # Leading-boundary audit: "drags/dragged/dragging", "scrapes/scraping",
    # "shoves/shoved", "slides/sliding" all match; "landslide" does not
    # (no word boundary before "slide"). Known accepted edge: "dragon"
    # matches "drag" -- motion text from the nonverbal director describes
    # physical actions, where that word does not occur.
    (("drag", "scrape", "shove", "slide"), "wood scraping across the floor"),
    (("coat", "cloth", "sleeve", "fabric", "curtain"), "cloth shifting"),
    (("step", "walk", "pace", "stride", "foot", "boot", "run"),
     "footsteps landing on the floor"),
)

#: Compiled once at import. Keyed by the cue string itself so the lexicon
#: above stays readable as data rather than as a wall of escaped patterns.
_CUE_PATTERNS = {cue: re.compile(r"\b" + re.escape(cue))
                 for cues, _sound in _SOUND_LEXICON for cue in cues}

#: WHEN NOTHING MATCHES, STILL NAME SOUNDS. Falling back to a category here
#: would reinstate exactly the defect this table exists to remove, so the
#: fallback is concrete: the quietest sound a present body makes.
#:
#: ONE CUE, NOT THREE (r3 finding, 2026-08-28). The fallback used to name cloth
#: AND footsteps AND objects striking wood. One latent decodes the picture and
#: the audio from this string, so three simultaneous events are three
#: instructions to the PICTURE as well -- on a beat whose action matched
#: nothing, which is precisely the beat least likely to contain them. A single
#: conservative cue cannot invent a walk or a dropped object.
_FALLBACK_SOUNDS = ("cloth shifting as the body settles",)


#: How long a shared span has to be before it counts as the identity text
#: leaking rather than an ordinary word both strings happen to use. Six words
#: is well past "a man in a coat" and well short of a real description.
#: A regex word boundary, held in a named constant because an inline
#: escape in this position was silently rewritten to a literal
#: backspace byte by a shell heredoc, which turned the cast-name
#: check into a pattern that could never match. One definition, one
#: place to verify.
_WORD_BOUNDARY = "\\b"

_IDENTITY_SPAN_WORDS = 6


def identity_leaks_in(positive, *, appearance="", names=()):
    """Names/description fragments that reached a JOINT-AV prompt. Pure.

    THE MODEL SPEAKS WHAT IT READS. On the mime and foley lanes the
    picture and the audio decode from ONE latent, so the positive prompt is
    also an audio script -- proven live 2026-08-28, where a mime beat rendered
    a woman SAYING "Queen of the Fairies" because her `character_description`
    opened with that title. Identity belongs in the conditioning STILL, whose
    scene_character row already mints the face unobstructed.

    Returns a list of human-readable findings, empty when clean. Reports; never
    raises and never rewrites the prompt -- the caller decides, and on this
    project a refused render is worse than a wrong sound.

    Matching is on a SHARED WORD SPAN, deliberately, not on capitalisation.
    A corpus audit of every joint-AV beat ever rendered showed the shape of
    the identity is irrelevant: "Queen of the Fairies" and "rustic weaver"
    were equally spoken, and a capitalisation test scored the second one
    clean. Length is the signal; case is not.
    """
    text = str(positive or "").lower()
    if not text:
        return []
    found = []
    for name in names or ():
        nm = str(name or "").strip().lower()
        # WHOLE WORDS ONLY, and this is not a nicety -- a bare substring pass
        # flagged the cast name "LEAR" inside the word "clearly", which every
        # prompt on this lane contains ("full face clearly visible"). A guard
        # that cries wolf on every beat of an episode is a guard nobody reads.
        # 3+ chars so an initial or a stray "a" cannot fire.
        if len(nm) >= 3 and re.search(
                _WORD_BOUNDARY + re.escape(nm) + _WORD_BOUNDARY, text):
            found.append("cast name %r is in the prompt" % str(name))
    look = str(appearance or "").strip().lower()
    if look:
        words = [w for w in look.replace(",", " ").split() if w]
        for i in range(0, max(0, len(words) - _IDENTITY_SPAN_WORDS + 1)):
            span = " ".join(words[i:i + _IDENTITY_SPAN_WORDS])
            if span in text:
                found.append("appearance text is in the prompt (%r...)" % span[:48])
                break
    return found


def named_sounds_for(positive):
    """The sounds this action would actually make, most telling first.

    Reads the composed positive because that is where the beat's action text
    already lives. Returns at most ``_MAX_NAMED_SOUNDS`` distinct phrases, and
    NEVER returns an empty list -- an unnamed sound request is the defect.
    Pure.
    """
    text = str(positive or "").lower()
    found = []
    for cues, sound in _SOUND_LEXICON:
        if sound in found:
            continue
        if any(_CUE_PATTERNS[cue].search(text) for cue in cues):
            found.append(sound)
            if len(found) >= _MAX_NAMED_SOUNDS:
                return found
    if not found:
        return list(_FALLBACK_SOUNDS)
    return found


def _join_sounds(sounds):
    """``a, b, c`` -- a TAG LIST, deliberately not prose.

    This used to render "a, b and c", which reads as a sentence fragment and
    is exactly the shape this lane has been caught speaking aloud. A bare
    comma list conditions the same and scans as a caption.
    """
    return ", ".join(sounds)


#: The LEGACY contiguous terminator -- what a finished joint-AV prompt ended
#: with under the pre-2026-08-29 append-at-the-tail shape, and what the
#: legacy branch of ``finish_joint_av_positive`` still appends for prompts
#: that never met the composer. Since the golden-shape reordering the frame
#: and the clause are NOT contiguous on a normally composed prompt (the
#: camera clause sits between them), so finished-ness is decided by
#: ``joint_av_prompt_is_finished`` -- frame PRESENT plus clause LAST -- and
#: never by matching this constant at the tail. Kept exported because the
#: two-constant pair is still the exact tail of a legacy-finished prompt.
JOINT_AV_TERMINATOR = "%s. %s" % (_SOUND_FRAME, _NO_VOICE_CLAUSE)


def build_joint_av_suffix(positive):
    """The LEGACY appended tail -- sounds named, frame, clause, all at the end.

    Shape::

        <named sounds>, close dry room tone. No speech, no voices.

    SUPERSEDED AS THE MAIN PATH on 2026-08-29: the open question of whether
    appending is as good as the golden recipes' interleaving was answered by
    measurement (production stems 10-40 dB under the goldens) and by operator
    ruling, and ``_ltx25_joint_av_core`` now seats the sounds inside the
    action at compose time. This suffix remains the tail for prompts that
    never met the composer -- overrides, old ledgers, fixtures -- so no input
    is ever left without its audio requirement. Pure.
    """
    return "%s, %s. %s" % (_join_sounds(named_sounds_for(positive)),
                           _SOUND_FRAME, _NO_VOICE_CLAUSE)


def joint_av_prompt_is_finished(text):
    """True when a joint-AV positive carries its full audio requirement.

    TWO CONDITIONS, BOTH REQUIRED: the invariant sound frame is PRESENT
    somewhere in the string (proving sounds were actually seated -- a bare
    no-voice tail proves nothing, which is the r3 false-receipt finding of
    2026-08-28), and the no-voice clause is the FINAL clause (its position is
    part of what works on this model). This replaces the old check that the
    frame-plus-clause pair sat contiguously at the tail: since the 2026-08-29
    golden-shape ruling the sounds and the frame sit BEFORE the camera clause,
    with only the no-voice clause at the very end, so the pair is no longer
    contiguous on a correctly finished prompt. Pure.
    """
    body = str(text or "")
    if _SOUND_FRAME not in body:
        return False
    return body.rstrip(" ,.;:").endswith(_NO_VOICE_CLAUSE.rstrip(" ,.;:"))


def sounds_named_in(text):
    """Which lexicon sound phrases a FINISHED prompt actually carries. Pure.

    The observability receipt used to re-run ``named_sounds_for`` over the
    pre-finish string, which was honest when sounds were appended at the
    finish seam. Now that the composers seat the sounds themselves, the
    receipt must read the final string -- and it must match PHRASES, not
    cues, because the phrases themselves contain cue words ("papers rustling
    violently" contains "papers") and a cue scan over a finished prompt
    re-matches its own output.
    """
    body = str(text or "")
    found = [sound for _cues, sound in _SOUND_LEXICON if sound in body]
    found.extend(s for s in _FALLBACK_SOUNDS if s in body and s not in found)
    return found


def finish_joint_av_positive(engine_id, positive):
    """Finish a joint-AV positive: guarantee sounds are seated and the
    no-voice clause is last.

    FINISHES, never replaces: the caller's visual core is preserved verbatim,
    so every engine's own prompt dialect, style cue and era tail survive
    untouched. Since the 2026-08-29 golden-shape ruling the NORMAL path
    arrives here with its sounds already seated by the composer (setting,
    expression, motion, sounds, sound frame, camera -- see
    ``_ltx25_joint_av_core``), and this seam appends exactly one thing: the
    no-voice clause, AFTER whatever the style-cue pass added, so the clause
    stays the final clause of the finished string.

    THE LEGACY TAIL SURVIVES for prompts that never went through the
    composer -- an operator override, an old ledger's ``text_prompt``, a
    fixture. A string with no sound frame in it still collects the full
    appended suffix exactly as before, so no input is ever left without its
    audio requirement.

    FOLEY AND MIME GET THE IDENTICAL STRING (operator, 2026-08-27: *"foley /
    mime same thing, they use the new foley prompting"* and *"the only
    difference between foley and mime is the mux layer"*).

    Takes NO dialogue argument, and that is the point. These lanes GENERATE
    audio but are not audio-IN lanes: nothing spoken may reach them, and the
    clause forbids voices outright.

    Returns ``positive`` unchanged for every engine outside
    ``_JOINT_AV_ENGINES``. Raises ``ValueError`` naming the engine when a
    Foley/Mime positive is blank -- the audio half would then be conditioned
    on nothing at all, which is a silent bad render rather than a loud one.
    Pure; stdlib only, so it imports cold on a CPU-only process.
    """
    eid = str(engine_id or "")
    if eid not in _JOINT_AV_ENGINES:
        return positive
    if not str(positive or "").strip():
        raise ValueError(
            "OTR_ltx25: %s was handed a blank positive prompt. LTX 2.5 "
            "conditions the picture AND the generated audio from this one "
            "string, so there would be nothing for the foley or the score to "
            "match." % eid)
    if joint_av_prompt_is_finished(positive):
        return positive
    core = positive.rstrip()
    trimmed = core.rstrip(" ,.;:")
    if _SOUND_FRAME in core:
        # The composer already seated the sounds; only the final clause is
        # owed. Appended with a sentence break so the clause reads as its own
        # sentence, exactly as the golden recipes end.
        return (trimmed + ". " + _NO_VOICE_CLAUSE) if trimmed \
            else _NO_VOICE_CLAUSE
    # LEGACY: no sound frame anywhere, so this string never met the composer.
    # The full derived tail is appended, same shape as before the 2026-08-29
    # reordering. The suffix is derived from the prompt, so once appended the
    # prompt carries sound words the lexicon would re-match -- which is why
    # idempotency keys on the frame-plus-clause predicate above, never on a
    # whole-suffix comparison.
    suffix = build_joint_av_suffix(core)
    return (trimmed + ", " + suffix) if trimmed else suffix


# =========================================================================== #
# PER-LANE MOTION PROMPTS -- EDIT HERE (Option B, operator ruling 2026-08-27)
# =========================================================================== #
#
# THREE LANES, THREE PROMPTS, NO SHARING. These classes are a subclass chain
# (mime <- foley_plus <- video), and that inheritance is deliberate for the
# MACHINERY -- same recipe, same graph, same 97-frame rung. It must never apply
# to the PROMPT: the operator's requirement is that one lane can later have slow
# motion, another backwards motion, and editing one may not touch another.
#
# HOW THAT IS ENFORCED: the render driver dispatches on
# ``type(engine).__dict__.get("compose_prompt")``. Only a formatter bound to the
# class ITSELF is visible; an inherited one is not. The explicit bindings at the
# bottom of this block put one entry in each class's own ``__dict__``, so the
# guarantee is mechanical rather than a matter of authoring discipline.
#
# THE BUDGET (operator's Option B matrix): one start -> development -> endpoint
# action arc, at most one minor reaction, one independent camera behaviour.
# LTX 2.5 is the local lane with the strongest evidence behind asking for a lot
# -- lane-7 A/B excursion 39-50 against a floor of 6 -- so identity and
# composition drift are the practical ceiling here, not the model's willingness
# to move.
#
# NO DAMPING WORDS, EVER. "subtle", "restrained", "barely", "stable" are what
# authored the silly pan (PBUG-20260827-04). Do not reintroduce them here.

#: Framing each silent LTX 2.5 lane carries FOR ITSELF. The driver's generic
#: ``startswith("ltx")`` suffix is skipped for a lane that composes its own
#: prompt, so this has to live here -- and it deliberately drops the old
#: "stable centered subject", which on a lane with no mouth to protect was an
#: instruction to hold still.
_LTX25_FRAMING = ("full face clearly visible, generous headroom, "
                  "the subject in real motion")

#: FOLEY SHARES THE SIBLINGS' FRAMING, and that is a correction (2026-08-27).
#: A brief patch gave this lane a hands-and-objects framing on the theory that a
#: face invites the model to synthesise a voice on the joint latent. It does --
#: but all three ltx25 lanes SHARE ONE STILL PLAN, whose `scene_character` row
#: mints "full head and shoulders with clear headroom inside frame, face
#: unobstructed". So a hands-only prompt argues with its own start frame, and
#: the operator has ruled that lip-sync and foley may share a still. The audio
#: is steered where the ruling puts it -- by NAMING THE SOUNDS, which is the
#: load-bearing half -- not by hiding the face from a picture that shows one.


def _ltx25_parts(inputs, *, include_appearance=True):
    """Subject, setting, expression, motion, then camera LAST.

    Shared ASSEMBLY, never a shared prompt: it only orders the parts the
    operator's matrix names. A lane that wants to diverge entirely stops
    calling it. Returns "" when the row carries no structured leaves.

    ``include_appearance=False`` DROPS the identity/appearance leaf, and it
    exists for exactly one reason (2026-08-28, proven on a published episode):
    on the JOINT-AV lanes the picture and the audio decode from ONE latent, so
    the prompt is also an audio script, and the model SPEAKS the proper nouns
    in it. `character_description` opens with a title -- "30s, Queen of the
    Fairies", "40s, rustic weaver" -- and a mime beat rendered a woman saying
    "Queen of the Fairies" aloud. Identity is already carried by the
    conditioning STILL, whose scene_character row mints the face unobstructed,
    so the text was redundant as well as harmful. The silent lane keeps its
    appearance: it discards the audio latent and has no mouth to protect.
    """
    keys = ("appearance", "setting", "expression", "motion")
    if not include_appearance:
        keys = keys[1:]
    parts = []
    for key in keys:
        value = str(inputs.get(key) or "").strip().strip(",")
        if value:
            parts.append(value)
    if not parts:
        return ""
    camera = str(inputs.get("camera") or "").strip().strip(",")
    if camera:
        parts.append(camera)          # camera AFTER the subject action
    return ", ".join(parts)


def _ltx25_joint_av_core(inputs):
    """The GOLDEN-SHAPED joint-AV core: sounds seated INSIDE the action, not
    appended after the camera.

    Order: setting, expression, motion, the named sounds, the sound frame,
    then camera LAST. This is the 2026-08-29 operator ruling ("prompt for
    foley like the golden recipe") made concrete: the lab's golden prompts
    interleave the sounds with the action that causes them -- "slamming his
    fist on the desk, a loud heavy wooden thud echoing ... rain drumming
    steadily" -- and they are the renders whose audio measures ~-25 dB mean
    while the old append-a-tag-list shape measured 10-40 dB under that. The
    deterministic equivalent of interleaving is seating the sound group
    directly after the motion clause that causes it, before the camera.

    THE SOUNDS ARE DERIVED FROM THE PRE-SOUND TEXT ONLY -- setting,
    expression, motion -- never from the camera clause and never from style
    cues (which are appended downstream, after compose). That kills two
    false-match classes structurally: a camera direction naming a lexicon
    word, and the `archival_documentary` style-cue class the lexicon comment
    records.

    NO appearance leaf, same as before: the joint latent SPEAKS proper nouns
    (proven live 2026-08-28, "Queen of the Fairies"), and identity is carried
    by the conditioning still. The no-voice clause is NOT added here --
    ``finish_joint_av_positive`` owns it, downstream of the style-cue pass, so
    it stays the final clause of the finished string.

    Returns "" when the row carries no structured leaves, exactly like
    ``_ltx25_parts``, so the caller falls back to the legacy path.
    """
    parts = []
    for key in ("setting", "expression", "motion"):
        value = str(inputs.get(key) or "").strip().strip(",")
        if value:
            parts.append(value)
    if not parts:
        return ""
    parts.extend(named_sounds_for(", ".join(parts)))
    parts.append(_SOUND_FRAME)
    camera = str(inputs.get("camera") or "").strip().strip(",")
    if camera:
        parts.append(camera)          # camera AFTER the sounds it records
    return ", ".join(parts)


def _ltx25_legacy_joint_av(inputs):
    """The legacy fallback for a lane that DECODES AUDIO FROM THE PROMPT.

    `_ltx25_legacy` returns the authored `text_prompt` verbatim, which is
    right for a silent lane and wrong for a joint-AV one: on those lanes the
    prompt is also an audio script, and a beat with no structured action falls
    back to a string that still carries the character's identity -- the exact
    text a published mime beat SPOKE ALOUD ("Queen of the Fairies").

    THE LAW ALLOWS THIS AND WOULD FORBID THE OBVIOUS ALTERNATIVE.
    `otr_shot_lock` forbids a Python vocabulary or token-overlap judge
    REPLACING an authored non-empty visual prompt. So this makes no judgement
    about words: it removes exactly one span, the `appearance` string our own
    composer injected, whose bytes we already hold. Everything the writer
    authored survives. If removing it would leave nothing, the original is
    returned UNCHANGED -- a beat with no picture direction at all is worse
    than a beat that names a face -- and the driver's `identity_leaks_in`
    guard reports the residue rather than this function silently inventing
    something. Pure.
    """
    raw = str(inputs.get("text_prompt") or "")
    look = str(inputs.get("appearance") or "").strip().strip(",")
    if not raw or not look:
        return raw
    trimmed = raw.replace(look, "").strip()
    # tidy only the punctuation the removal itself orphaned
    trimmed = trimmed.lstrip(" ,.;:").strip()
    while ", ," in trimmed:
        trimmed = trimmed.replace(", ,", ",")
    return trimmed if trimmed else raw


def _ltx25_legacy(inputs):
    """The one documented rule for a row with no structured leaves.

    Not a second path and not a feature switch -- the same formatter handling
    incomplete input. An older ledger keeps rendering exactly what it rendered
    before, which is replay correctness rather than a gate.
    """
    return str(inputs.get("text_prompt") or "")


def compose_ltx25_video(self, inputs):
    """``ltx25_video`` -- PUSH THE LIMIT. One decisive within-frame action.

    The operator's rule for the silent lanes is "push the limit to what it is
    capable of", and this is the lane with local proof behind that: every
    shipping-strength arm cleared the lab's motion floor several times over.
    """
    core = _ltx25_parts(inputs)
    if not core:
        return _ltx25_legacy(inputs)
    return "%s, %s" % (core.rstrip(" ,."), _LTX25_FRAMING)


def compose_ltx25_foley_plus(self, inputs):
    """The foley lanes -- the action must be VISIBLY SOUND-PRODUCING.

    **NOT an audio-in lane** (operator, explicitly: *"mime and foley ... do need
    more motion than audio in lanes -- they are not audio in lanes"*). It takes
    the full silent motion budget. What makes it different is that the picture
    has to EARN the bed: hands on dials, papers, switches, footfalls.

    NO IDENTITY WALL. This lane composes without the appearance leaf: the
    joint latent SPEAKS proper nouns out of the prompt, and identity is
    already fixed by the conditioning still. See ``_ltx25_joint_av_core``.

    THE SOUNDS ARE SEATED HERE, IN THE GOLDEN SHAPE (operator ruling
    2026-08-29): setting, expression, motion, then the named sounds and the
    sound frame, then camera -- sounds inside the action, the way the lab's
    golden recipes phrase the renders whose audio actually measures loud.
    ``finish_joint_av_positive`` still owns the final no-voice clause and
    appends it once, downstream of the style-cue pass, so the clause stays
    the last sentence of the finished string.
    """
    core = _ltx25_joint_av_core(inputs)
    if not core:
        return _ltx25_legacy_joint_av(inputs)
    return core.rstrip(" ,.")


def compose_ltx25_mime(self, inputs):
    """The mime lanes -- BYTE-IDENTICAL to foley. The mux is the difference.

    Operator ruling, restated 2026-08-28: *"foley and mime should have the
    same prompting, the only difference is the mux layer setting."* The lanes
    diverge at `FOLEY_LANE_GAINS` -- foley mixes its bed at 0.50 with the
    programme at 0.50; mime plays its generated audio at 1.00 with the
    programme muted to 0.00 -- and nowhere else.

    This stays its OWN function rather than an alias, because the lanes must
    remain independently re-wordable ("later one could have slow motion, some
    could have tulip motion"). Shared text, separate seams.

    NO IDENTITY WALL, for the same reason as its foley sibling -- this lane
    rendered a woman saying "Queen of the Fairies" out loud because her
    `character_description` began with that title. See ``_ltx25_joint_av_core``.

    **NOT an audio-in lane either.** The performance carries the beat with no
    speech at all, so the action reads larger than its siblings and lands on a
    held final pose. Like foley it seats its sounds in the golden shape via
    ``_ltx25_joint_av_core`` (2026-08-29), and the final no-voice clause
    belongs to ``finish_joint_av_positive`` alone.

    AND IT IS THE SAME SHAPE FOLEY GETS, verbatim (operator, 2026-08-27:
    *"the only difference between foley and mime is the mux layer"*). This lane
    used to receive a mood-led request for "instrumental scene score" -- a
    CATEGORY, which left the model to pick the sound and, with a person in
    frame, it picked voices. The PICTURE is still this lane's own: a mime beat
    plays larger and holds its endpoint, which is what the two formatters
    genuinely differ on.
    """
    core = _ltx25_joint_av_core(inputs)
    if not core:
        return _ltx25_legacy_joint_av(inputs)
    # IDENTICAL to foley by ruling. This lane keeps its OWN function so it can
    # diverge later without an architectural change.
    return core.rstrip(" ,.")


# THE EXPLICIT BINDINGS. One entry per class ``__dict__`` -- this is what makes
# the lanes independent in the dispatcher's eyes. Do NOT collapse these
# into a base-class method: that would restore exactly the sharing the operator
# ruled out, and the dispatcher would stop seeing the children entirely.
Ltx25VideoEngine.compose_prompt = compose_ltx25_video
Ltx25FoleyPlusEngine.compose_prompt = compose_ltx25_foley_plus


@register
class Ltx25NativeFoleyWideEngine(Ltx25FoleyPlusEngine):
    """24 GB AND UP, ANY modern NVIDIA. No Blackwell-only format.

    int8 is the weight, and it was chosen by running it rather than by sizing
    it. INT8 tensor cores have shipped since Turing, so this one file serves
    Ada, Ampere and Blackwell alike -- measured on a 4090 (101.3 s, 23.5 GB
    peak) and an RTX PRO 4500 (266.1 s, 26.8 GB peak), both a 97-frame clip
    with foley. It is the DEFAULT native choice; the Blackwell lane below is
    the optimisation, not the baseline.

    TWO EARLIER VERSIONS OF THIS DOCSTRING WERE WRONG, in opposite directions,
    and both were wrong the same way: they reasoned about the weight instead of
    loading it. The first named fp8_e4m3fn, a file the publisher's
    MANIFEST.json advertises with a size and a sha256 and the repository does
    not carry. The second ruled int8 out as a "32 GB-class file" that would
    land near 26.5 GB in flight -- wrong in direction AND magnitude, since int8
    both fits a 24 GB card and is forty percent faster than the alternative.
    ComfyUI sizes residency against the card it finds and offloads when it is
    tight, so weights-plus-activations arithmetic systematically over-predicts.
    The receipts live in apple/PROD_BUG_LOG.md; what belongs here is the file.
    """

    name = "ltx25_foley_24gb"
    engine_version = "1"
    default_roles = ()
    _native_dit = LTX25_NATIVE_DIT_WIDE
    #: 24 GB+ has the headroom; the CPU encode measured minutes per beat.
    _native_te_device = "default"
    #: Tracks the line above -- see the base class for why they are one fact.
    _encoder_cache_expects_cpu = False


@register
class Ltx25NativeFoleyBlackwellEngine(Ltx25FoleyPlusEngine):
    """24/32 GB BLACKWELL. NVFP4, and it is the small one.

    12.50 GB against the wide lane's 20.03 -- so Blackwell does not merely
    match the other tier, it frees ~7.5 GB that a larger canvas or a longer
    rung can spend. What that headroom should actually buy is a MEASUREMENT,
    not a guess, so the canvas stays inherited until a real leg reports a peak.
    (This line compared against 21.48 until 2026-09-22 -- the size of the fp8
    file that does not exist. The wide lane is int8 at 20.03 GB.)

    "Blackwell only" is the publisher's own wording. On anything older this
    lane is expected to refuse or fall back badly, which is why the wide lane
    above stays the default.

    PARKED 2026-09-23 (operator): *"park blackwell, we don't need it, we have
    int8 and mix4x8-13.8GB."* It stays REGISTERED because it is proven and
    therefore an honest dropdown choice -- 237.8 s on an RTX PRO 4500, 250.8 s
    on a 5080, both with foley. It simply has no profile and no further work
    planned, and that is a decision rather than an oversight.

    WHY IT LOST, measured the same night on one 16 GB Blackwell card, same
    still, same seed, same recipe:

        mix4x8-13.8GB   205.3 s      portable (Ampere+)
        w4a8            210.3 s      portable (Ampere+)
        nvfp4           250.8 s      BLACKWELL ONLY

    The Blackwell-native format was the slowest of the three ON BLACKWELL, so
    an architecture lock bought nothing. The attention angle closed too: LTX's
    DiT goes through ``optimized_attention`` (SDPA), comfy-kitchen's
    ``sol_attn`` is unreachable from that path, SageAttention process-aborts
    LTX-Video (BUG-070), and the VAE decoder already calls
    ``comfy_kitchen.na3d``. There was no Blackwell speedup left to claim."""

    name = "ltx25_foley_blackwell"
    engine_version = "1"
    default_roles = ()
    _native_dit = LTX25_NATIVE_DIT_BLACKWELL
    #: The smallest DiT of the three (12.50 GB), so this tier has the MOST
    #: headroom of all -- pinning here would be the least defensible.
    _native_te_device = "default"
    _encoder_cache_expects_cpu = False


@register
class Ltx25NativeFoley16gbEngine(Ltx25FoleyPlusEngine):
    """16 GB, for the 5080 -- so the native family covers the dev box too.

    mix4x8 won the 16 GB bake-off (see ``LTX25_NATIVE_DIT_16GB``). The silent
    ``ltx25_video`` lane and both 16 GB mime/audio-in lanes load the same file.
    """

    name = "ltx25_foley_16gb"
    engine_version = "1"
    default_roles = ()
    _native_dit = LTX25_NATIVE_DIT_16GB
    #: KEEPS THE PIN, inherited from the base. 16 GB is exactly the class the
    #: pin was written for, and this lane is the one that would tip without
    #: it. _encoder_cache_expects_cpu stays True with it.


class Ltx25NativeAudioInMixin:
    """AUDIO-IN: condition the picture on a real waveform instead of inventing one.

    THE SAME GRAPH AS THE FOLEY LANE IT IS MIXED INTO, with two changes, and
    both of them are forced by how the model works rather than chosen:

    1. THE ENCODED WAVEFORM IS FITTED TO THE PICTURE BEFORE IT IS SAMPLED.
       ``emptyaudio`` STAYS ``LTXVEmptyLatentAudio``, because it is the only
       node that computes the audio latent length the picture actually needs
       (``num_of_latents_from_frames(frames, fps)``). The beat's own waveform
       is encoded beside it by ``refencode`` and joined by ``refconcat``,
       whose ``fit_audio`` trims or zero-pads the encoded stream to exactly
       that length. Foley asks the model to invent a bed; this hands it the
       beat's own audio and asks it to move the picture to match.

    2. ``modality_scale`` RISES from 1.0 to 3.0. At 1.0 the guider is a
       documented NO-OP (``LTXVModalityGuidance``: "Set to 1.0 to disable (no
       extra pass)") and that is the right value for foley, where there is
       nothing to sync TO. It is the weaker value here: modality guidance
       STRENGTHENS the cross-modal coupling.

       IT DOES NOT CREATE IT, and an earlier draft of this docstring said
       it did. `a2v_cross_attn` and `v2a_cross_attn` both default to
       enabled (`comfy/ldm/lightricks/av_model.py`); scale 1.0 disables
       only the EXTRA guidance pass, not the attention. So 1.0 is a
       defensible shipping value for this lane and is the fallback if the
       16 GB tier turns out not to afford the extra pass. Corrected by a
       codex review.

    THE FIRST DESIGN SWAPPED THE CLASS BEHIND THE ``emptyaudio`` KEY and let
    ``concat`` read the encoder directly. That is what a reader expects to
    work, and it does not. ``LTXVEmptyLatentAudio`` is TOLD the frame count
    and returns exactly ``round(frames / fps * latents_per_second)`` latents;
    ``LTXVAudioVAEEncode`` returns whatever its own mel and downsample
    rounding produce for the waveform it was handed. MEASURED on a live pod
    leg (2026-09-23): a 97-frame beat whose reference WAV was ALREADY exactly
    3.88 s -- 171,108 samples at 44.1 kHz, which resamples to exactly the
    186,240 the clip needs at 48 kHz -- encoded to ONE latent fewer than the
    picture, decoded 3,360 samples short, and was refused by
    ``_write_foley_stem``. THE INPUT WAS NEVER WRONG; the round trip is lossy
    and only the empty-latent node knows the target. A waveform-level
    reconciliation was tried first and could not fire at all, because the
    waveform already had the exact duration asked of it.

    ``fit_audio`` LEAVES THE PADDED TAIL UNMASKED so the model generates it --
    its own documented behaviour for a clip shorter than the video, and
    better than handing the encoder silence to condition on.

    STAGE TWO NEEDS NO EQUIVALENT: ``refine_concat`` takes its audio from
    ``W("separate", 1)``, the stream stage one already sampled, so fixing the
    length here fixes it everywhere.

    THE MODEL REALLY DOES CONDITION ON AUDIO -- this is not an assumption.
    ``LTXVModalityGuidance`` works by running one extra pass with
    ``a2v_cross_attn`` and ``v2a_cross_attn`` set False and steering back
    toward the coupled prediction (`comfy_extras/nodes_lt.py`). Those switches
    exist because the transformer attends both ways between the streams; the
    reference default for the scale is 3.0, which is what this uses.

    THE COST IS ONE EXTRA FORWARD PASS PER STEP while guidance is active, on
    top of whatever the sampler already costs. That is the trade audio-in makes
    and it should be measured on this lane, not inherited from the foley
    timings.

    ``audio_ref`` IS ALREADY A FIRST-CLASS PER-BEAT ASSET -- the render driver
    supplies it (`render_driver.py`) and HuMo and the H3 audio-in lane already
    require it. Nothing new is plumbed; this lane simply declares that it needs
    it and fails LOUD without one, because a lane that silently fell back to an
    empty latent would be a foley lane wearing an audio-in name.
    """

    #: THE FAMILY IS THE WIRING, NOT A LABEL, and without it these lanes are
    #: registered code that cannot receive its own required input.
    #:
    #: `render_driver._uses_ambient_master_audio` keys on the family to decide
    #: which lanes get a bounded master slice when a beat carries no per-line
    #: timing. The ltx25 base declares `image_to_video` -- correct for it, and
    #: its own comment says why -- so an audio-in lane inheriting that would
    #: hard-require `audio_ref`, never be handed one on a lineless beat, and
    #: fail loud on exactly the beats the ambient slice exists to cover.
    #:
    #: `audio_conditioned_video` is the EXISTING audio-in family;
    #: `eng_minimax_h3` uses it the same way and says so. It also puts these
    #: lanes into `mouth_policy.AUDIO_IN_FAMILIES`, which is right -- a lane
    #: driven by real speech is a lip-sync lane -- and gives the credits roll
    #: its "audio-in video" wording for free.
    family = "audio_conditioned_video"

    #: THE PARENT'S FULL ROLE SET, and an earlier draft narrowed it here on a
    #: false premise. That draft declared cabinet roles only and claimed the
    #: narrowing was what kept these lanes out of `MouthPolicyError`. IT
    #: RESTRICTED NOTHING: `role_compat.engine_fits_role` is "PURELY
    #: capability -- every token in the engine's required_inputs must be
    #: available in the role", and it ignores this list outright. These lanes
    #: require init_image + audio_ref and so fit `character_video`, and the
    #: director could select them there regardless
    #: and then hit a plan-time refusal. Found by a codex review.
    #:
    #: `roles` is UI-sort / self-description metadata. The real gates are
    #: capability (above) and `_AUDIO_IN_CHARACTER_ENGINES` in render_driver,
    #: which these lanes now join -- see that list.
    roles = ("announcer_visual", "music_visual", "character_video")

    #: Lightricks' own reference default, and it DELIBERATELY DEPARTS from
    #: `ltx25_recipe.LTX25_CFG_MODALITY = 1.0`. That recipe note calls the
    #: three CFGs a VRAM contract and it is right about the two it was
    #: measured on -- but this knob differs in KIND, not degree: it adds an
    #: entire extra forward pass per step
    #: (`comfy_extras/nodes_lt.py` -- `LTXVModalityGuidance` calls
    #: `calc_cond_batch` inside its post-CFG hook). The foley lanes keep 1.0
    #: and are untouched; only audio-in reads this.
    #:
    #: THE EXTRA PASS CAN RAISE PEAK VRAM, not just time: the conditional
    #: prediction and the CFG result stay live while it runs, so it needs
    #: storage on top of them. A codex review refuted the "sequential means
    #: free" reading, and narrowing the guidance window reduces how MANY
    #: extra passes run without lowering the peak during an active one.
    #:
    #: NOT MEASURED ON 16 GB YET, and the 16 GB tier is the one clamped at
    #: 14.5 GiB. The first 16 GB leg reports peak VRAM. If it tips, 1.0 is a
    #: legitimate answer for that tier -- the cross-modal attention stays on
    #: either way -- and this constant is where that lands.
    _modality_scale = 3.0

    #: `comfy/sd.py` gives the LTX audio VAE `downscale_ratio = 4096`, and
    #: `vae_encode_crop_pixels` narrows any waveform that is not a multiple of
    #: it, taking HALF of the remainder `n % 4096` off the FRONT. (Written
    #: as prose on purpose: `tests/test_lane_preflight_matrix.py` scans MRO
    #: source for a floor-divide-by-two token as its heuristic for "this
    #: lane halves its canvas", and these lanes do not -- the operator
    #: expression below is about the audio crop, not the picture.)
    _VAE_AUDIO_CROP_MULTIPLE = 4096

    #: THE RATE THE ENCODER ACTUALLY SEES, which is not the file's.
    #: `VAE.__init__` sets `self.audio_sample_rate = 44100` (`comfy/sd.py`)
    #: and the LTX audio branch never overrides it -- it sets only
    #: `audio_sample_rate_output`, the DECODE side, which is 48 kHz. So
    #: `VAEEncodeAudio` compares the file against 44100 and resamples
    #: anything else BEFORE `vae.encode` runs its crop.
    _VAE_AUDIO_INPUT_RATE = 44100

    def _pad_reference_for_vae_crop(self, audio_path, length):
        """Pad the reference WAV so the encoder's centre-crop removes nothing.

        MEASURED (2026-09-23, this beat): the reference slice is 171,108
        samples. ``171108 % 4096 == 3172``, so ``vae_encode_crop_pixels``
        narrowed it to 167,936 and threw away **1,586 samples from the
        BEGINNING** -- 35.96 ms, most of a frame at 25 fps. The picture was
        then asked to follow audio that had lost its head and been shifted
        earlier than the beat it belongs to.

        Latent-space fitting cannot undo this: ``fit_audio`` pads or trims the
        TAIL, and the damage is at the front and already baked into the mel.
        The crop is also what made the encode come up one latent short in the
        first place -- 381 mel frames instead of 389.

        THE CROP IS A NO-OP ON AN EXACT MULTIPLE: ``x == dims[d]`` and
        ``narrow`` is never called. So the fix is to hand it a length it will
        not touch -- at least the clip's own duration, rounded up to the next
        multiple. The added tail is silence the model is free to generate into
        (`fit_audio` leaves the padded latent tail unmasked) and it is trimmed
        off the decoded output by ``_write_foley_stem`` anyway.

        This also covers the case that deleting the old waveform helper lost:
        a reference much SHORTER than the clip used to be cropped to zero
        samples and raise inside the encoder. Found by a codex review.

        EXACT WHEN THE FILE'S RATE IS THE VAE'S INPUT RATE, which it is on
        this path -- the driver writes 44.1 kHz slices and the LTX audio VAE
        takes 44.1 kHz, so ``VAEEncodeAudio`` does not resample. A file at
        another rate is resampled BEFORE the crop, so its padded length is no
        longer a guaranteed multiple; that is logged rather than silently
        assumed.

        Best-effort: an unreadable file is handed on unchanged so the graph
        fails downstream with its own named error instead of here.
        """
        import os
        try:
            from . import foley_stems as _fs
        except ImportError:                  # pragma: no cover -- flat imports
            return audio_path
        try:
            samples, rate = _fs.read_pcm16_wav(audio_path)
            orig_rate = int(rate)
        except Exception as exc:             # noqa: BLE001 -- never fatal here
            _LOG.warning(
                "[OTR video] %s could not read its audio_ref %r (%r); handing "
                "it to the encoder unchanged", self.name, audio_path, exc)
            return audio_path

        import numpy as np
        fps = int(self.target_fps) or 25
        mult = int(self._VAE_AUDIO_CROP_MULTIPLE)
        target_rate = int(self._VAE_AUDIO_INPUT_RATE)

        # RESAMPLE FIRST, or the alignment below is computed against a
        # length the encoder never sees. `VAEEncodeAudio` resamples to the
        # VAE's own rate BEFORE `vae.encode` crops, so a 4096-aligned file
        # at some other rate arrives unaligned and the crop eats the head
        # again -- reopening the very defect this helper exists to close.
        #
        # IT IS NOT A HYPOTHETICAL. These lanes serve `character_video`
        # beats, where the driver hands over the per-line VOICE wav
        # unchanged at whatever rate the TTS wrote it, and Bark writes
        # 24 kHz. Found by a Sonnet QA lane; the 44.1 kHz claim held only
        # for the master-slice path, which is pinned to 44100.
        #
        # Done with the SAME call ComfyUI would have made, just early and
        # at a length we control, so the waveform is not altered twice.
        if int(rate) != target_rate:
            try:
                import torch
                import torchaudio
                samples = torchaudio.functional.resample(
                    torch.from_numpy(np.ascontiguousarray(samples)),
                    int(rate), target_rate).numpy()
            except Exception as exc:       # noqa: BLE001
                # NEVER relabel a failed resample with the destination
                # rate. Say what is wrong and leave the rate honest; the
                # crop will misbehave and the log will say why.
                _LOG.warning(
                    "[OTR video] %s could not resample its audio_ref from "
                    "%d Hz to the encoder's %d Hz (%r); the VAE will "
                    "resample it AFTER this alignment and its crop may "
                    "take samples off the front of the beat",
                    self.name, int(rate), target_rate, exc)
            else:
                _LOG.info(
                    "[OTR video] %s resampled its audio_ref %d -> %d Hz to "
                    "match the encoder, so the crop alignment holds",
                    self.name, int(rate), target_rate)
                rate = target_rate

        need = int(round(int(length) / float(fps) * int(rate)))
        have = int(samples.shape[-1])
        # THE TARGET IS THE CLIP, NOT THE FILE. A reference longer than the
        # beat is not extra context: `fit_audio` narrows the encoded latent
        # with `narrow(dim, 0, length)` and discards the tail regardless, so
        # encoding all of it only spends time and VRAM on samples that are
        # thrown away -- which the 16 GB tier can least afford. Taking the
        # HEAD is the correct slice because the driver already cut this file
        # to the beat (`render_driver` owns `start_s + segment.offset_s`).
        want = ((need + mult - 1) // mult) * mult
        if want == have and int(rate) == int(orig_rate):
            return audio_path

        if have > want:
            samples = samples[..., :want]
        else:
            pad = np.zeros((samples.shape[0], want - have),
                           dtype=samples.dtype)
            samples = np.concatenate([samples, pad], axis=-1)

        import hashlib
        key = hashlib.sha256(
            os.path.abspath(str(audio_path)).encode("utf-8", "replace")
        ).hexdigest()[:12]
        out = os.path.join(
            os.path.dirname(str(audio_path)) or ".",
            "otr_%s_vaepad_%s_%d.wav" % (self.name, key, want))
        _fs.write_pcm16_wav(out, samples, rate)
        _LOG.info(
            "[OTR video] %s %s its audio_ref %d -> %d sample(s) at %d Hz so "
            "the VAE crop (multiple of %d) keeps the head; clip needs %d",
            self.name, "trimmed" if have > want else "padded",
            have, want, rate, mult, need)
        return out

    def _node_candidates(self):
        cand = dict(super()._node_candidates())
        cand["loadaudio"] = ("LoadAudio",)
        cand["refencode"] = ("LTXVAudioVAEEncode",)
        cand["refsolid"] = ("SolidMask",)
        cand["refmask"] = ("SetLatentNoiseMask",)
        cand["refconcat"] = ("LTXVConcatAVLatent",)
        return cand

    def _build_render_request(self, request):
        plan = super()._build_render_request(request)
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        # TOP-LEVEL, NOT UNDER asset_refs -- the schema splits them on purpose
        # (`schemas.py`: `asset_refs: dict[str, str]` beside
        # `audio_ref: Optional[AudioRef]`), and `build_request` writes
        # `"asset_refs": {"init_image": ...}` next to `"audio_ref": {...}`.
        # NOTHING in the tree ever writes `asset_refs["audio_ref"]`.
        #
        # The first draft copied the `init_image` shape and read the empty
        # slot, so every beat raised "requires an audio_ref and the beat
        # carried none" while the driver had supplied one all along. That the
        # render reached `_build_graph` at all is the proof: `_render_one`
        # calls `_assert_family_inputs_satisfiable` first, which checks the
        # TOP-LEVEL field -- a genuinely missing ref fails there as
        # FamilyInputGap and never gets this far.
        #
        # This is the read every other audio consumer uses: eng_humo,
        # eng_minimax_h3, eng_visualizer, and the cloud
        # adapters. Found by a cursor review after two failed live legs.
        plan["audio_path"] = self._ref_path(get("audio_ref"))
        return plan

    def _build_graph(self, plan, image_name, length, width, height):
        # LOCAL IMPORT, like every other method in this module. Module scope
        # here is stdlib-only by contract (V-12 cold-import), so `_wb` is bound
        # per-method and is NOT a module global -- the first draft of this
        # mixin used it as though it were and died on the first shot with
        # "name '_wb' is not defined", CRASH_BEFORE_LOAD, on a live pod leg.
        from . import wrapper_bridge as _wb
        g = super()._build_graph(plan, image_name, length, width, height)
        audio_path = str(plan.get("audio_path") or "")
        if not audio_path:
            raise _wb.GraphExecutionError(
                "%s requires an audio_ref and the beat carried none. NO "
                "FALLBACK -- this lane exists to follow a supplied waveform, "
                "and quietly generating one instead would ship a foley clip "
                "under an audio-in name" % (self.name,))
        audio_path = self._pad_reference_for_vae_crop(audio_path, length)
        audio_name = _wb.stage_into_comfy_input(audio_path)
        W = _wb.Wire
        g["loadaudio"] = {"class": "loadaudio",
                          "inputs": {"audio": audio_name}}
        # `emptyaudio` IS LEFT ALONE, and that is the fix -- it is the only
        # node that knows the audio latent length the picture needs. See the
        # class docstring for the leg that proved swapping it does not work.
        g["refencode"] = {"class": "refencode", "inputs": {
            "audio": W("loadaudio", 0),
            "audio_vae": W("audiovae", 0)}}
        # `concat` already emits an AV latent, which is exactly the branch
        # where LTXVConcatAVLatent swaps the audio stream and runs
        # `fit_audio` on it -- trimming or zero-padding the encoded waveform
        # to the empty latent length, padded tail left unmasked.
        # FREEZE THE SUPPLIED AUDIO, or the sampler simply discards it.
        # `LTXVAudioVAEEncode` returns no noise mask, and
        # `LTXVConcatAVLatent` substitutes `ones_like` for a missing audio
        # mask whenever the VIDEO side has one -- which the i2v anchor
        # always gives it. An all-ones mask means "generate this", and the
        # stage-one ladder starts at sigma 1.0, where
        # `sigma * noise + (1 - sigma) * latent_image` contributes exactly
        # NOTHING from the reference. The lane would have carried a
        # correctly sized reference waveform and then thrown it away.
        # This is the SolidMask(0) -> SetLatentNoiseMask pattern the LTX
        # audio-in graphs have always used: the audio latent rides FROZEN.
        #
        # It also completes the fit: `fit_audio` appends `ones_like(pad)`
        # to an EXISTING mask, so the supplied span stays frozen at 0 and
        # only the padded tail is left free for the model to generate.
        # With no mask to extend, that branch did nothing.
        g["refsolid"] = {"class": "refsolid", "inputs": {
            "value": 0.0, "width": int(width), "height": int(height)}}
        g["refmask"] = {"class": "refmask", "inputs": {
            "samples": W("refencode", 0), "mask": W("refsolid", 0)}}
        g["refconcat"] = {"class": "refconcat", "inputs": {
            "video_latent": W("concat", 0),
            "audio_latent": W("refmask", 0)}}
        g["sampler"]["inputs"]["latent_image"] = W("refconcat", 0)
        g["modality"]["inputs"]["modality_scale"] = float(self._modality_scale)
        return g


@register
class Ltx25NativeAudioIn16gbEngine(Ltx25NativeAudioInMixin,
                                   Ltx25NativeFoley16gbEngine):
    """16 GB audio-in. mix4x8, picture driven by the beat's own sound."""

    name = "ltx25_audio_in_16gb"
    engine_version = "1"
    default_roles = ()
    required_inputs = ("text_prompt", "init_image", "audio_ref")


@register
class Ltx25NativeAudioIn24gbEngine(Ltx25NativeAudioInMixin,
                                   Ltx25NativeFoleyWideEngine):
    """24 GB AND UP audio-in. int8, same coupling as the 16 GB lane."""

    name = "ltx25_audio_in_24gb"
    engine_version = "1"
    default_roles = ()
    required_inputs = ("text_prompt", "init_image", "audio_ref")


@register
class Ltx25NativeMime16gbEngine(Ltx25NativeFoley16gbEngine):
    """The 16 GB lane as a SILENT PERFORMANCE carrying its own score.

    This class body is almost empty on purpose: the picture, the harvest of
    the audio latent, the second-pass
    decode, the durable stem and the cut in the coverage assembler are all
    inherited unchanged. What differs is entirely at the mux, in a table --
    1.00 foley / 0.00 master instead of 0.50 / 0.50 -- and that row lives in
    ``foley_stems.FOLEY_LANE_GAINS``, not here.

    THE TTS AND MUSIC ARE STILL GENERATED AND THEN MIXED TO ZERO. That waste
    is the operator's own ruling (2026-08-26) and
    it is what lets this class be four lines instead of a new owner node and an
    execution-order inversion.

    THE ATTENUATION IS PER-WINDOW, which is why this lane is deliberately NOT
    in ``GLOBAL_MASTER_GAIN_LANES`` while its foley parent is. Engines are
    ROLE-WIDE: this id on ``character_video_model`` silences every character
    beat while the announcer and music roles still speak out of the SAME master
    WAV, so a global zero would silence the episode.
    """

    name = "ltx25_mime_16gb"
    engine_version = "1"

    #: SELECTABLE, NEVER A DEFAULT -- and here more emphatically than anywhere
    #: else in the roster. Inheriting it would mute every
    #: beat of a role nobody chose to mute.
    default_roles = ()


@register
class Ltx25NativeMime24gbEngine(Ltx25NativeFoleyWideEngine):
    """The 24 GB AND UP lane as a silent performance. int8.

    Same relationship to ``ltx25_foley_24gb`` that
    ``ltx25_mime_16gb`` has to the 16 GB lane, and the same one-row
    difference at the mux. Everything about the weight, the encoder placement
    and the recipe is inherited; see the sibling above for why the body is
    empty and why this id is per-window rather than global.
    """

    name = "ltx25_mime_24gb"
    engine_version = "1"
    default_roles = ()


# ---------------------------------------------------------------------------
# THE PROMPT FORMATTER IS BOUND PER CLASS, AND INHERITANCE DOES NOT COUNT.
#
# `render_driver` dispatches with
# ``type(engine).__dict__.get("compose_prompt")`` -- deliberately ``__dict__``
# and not ``hasattr``, so a subclass that merely INHERITS a formatter is
# treated as having none and falls through to the legacy composer. The
# assignments above therefore covered only the classes they named, while every
# TIER class silently used the legacy path.
#
# WHAT THAT COST, on beats those lanes render: no named sounds, and no
# ``No speech, no voices.`` terminator, because `finish_joint_av_positive`
# returns the positive UNCHANGED for any id outside `_JOINT_AV_ENGINES`. This
# module's own record of that prompt shape is a few hundred lines up: it
# "rendered a woman saying 'Queen of the Fairies' out loud".
#
# This is PBUG-20260923-05's mechanism one level up -- a subclass relationship
# the author reads as inheritance, against a dispatch that reads only what a
# class binds itself. Found by a Fable review, on lanes nobody was reviewing.
#
# Each tier takes the formatter its PARENT binds, which is the definition of
# parity: a tier differs by which weight loads, never by how it is prompted.
# THE AUDIO-IN LANES ARE DELIBERATELY ABSENT FROM BOTH LOOPS, and an earlier
# version of this block had them in the first one -- which was wrong in the
# direction that matters. `finish_joint_av_positive` says what this contract is
# for in its own words: "These lanes GENERATE audio but are not audio-IN lanes:
# nothing spoken may reach them, and the clause forbids voices outright."
# Binding the foley formatter to an audio-in lane, and listing it below,
# appended `No speech, no voices.` to a lane whose entire job is to FOLLOW
# supplied speech.
#
# The audio-in lanes bind NO formatter on purpose, so they keep the shared
# driver's composition -- the 240-char budget with the style cue front-loaded
# -- which a fresh formatter "would have to re-implement and initially got
# wrong". Found by a cursor QA pass on the commit that introduced it.
for _tier_cls in (Ltx25NativeFoleyWideEngine,
                  Ltx25NativeFoleyBlackwellEngine,
                  Ltx25NativeFoley16gbEngine):
    _tier_cls.compose_prompt = compose_ltx25_foley_plus
for _tier_cls in (Ltx25NativeMime16gbEngine, Ltx25NativeMime24gbEngine):
    _tier_cls.compose_prompt = compose_ltx25_mime
del _tier_cls

#: EVERY joint-AV lane, not the three that happened to exist when this tuple
#: was written. A lane absent from here gets its positive back UNFINISHED from
#: `finish_joint_av_positive` -- no named sounds, no terminator -- which is
#: silent, because an unfinished prompt still renders a clip.
#: EVERY lane that GENERATES its audio -- and no audio-IN lane, which is the
#: distinction `finish_joint_av_positive` draws in its own docstring. The
#: audio-in ids were briefly listed here and that appended a voice prohibition
#: to lanes conditioned on speech; `cloud_ltx25_audio_in` and the native
#: audio-in lanes are correctly absent.
_JOINT_AV_ENGINES = _JOINT_AV_ENGINES + (
    "ltx25_foley_16gb",
    "ltx25_foley_24gb",
    "ltx25_foley_blackwell",
    "ltx25_mime_16gb",
    "ltx25_mime_24gb",
)


__all__ = ["Ltx25VideoEngine", "Ltx25FoleyPlusEngine",
           "LTX25_RESERVED_SIBLING_IDS", "LTX25_FOLEY_RECEIPT_KEYS",
           "LTX25_FOLEY_GAIN", "LTX25_MASTER_GAIN_UNDER_FOLEY",
           "finish_joint_av_positive", "build_joint_av_suffix",
           "named_sounds_for", "sounds_named_in",
           "joint_av_prompt_is_finished", "JOINT_AV_TERMINATOR",
           "identity_leaks_in"]
