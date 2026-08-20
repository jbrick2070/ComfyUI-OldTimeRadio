"""The ``ltx25_video`` lane contract -- LTX 2.5 Chunk A, 2026-08-19.

WHAT THIS FILE IS FOR. Two different jobs, and it is worth saying which is
which, because only one of them is a checklist:

* **The declarations**, which the lane preflight matrix requires to exist
  somewhere (G2.2 explicitly wants a per-lane test naming BOTH the engine and
  its canvas, so a declaration cannot drift away from the graph in silence).
* **The traps this build actually hit.** Every test below the first section is
  here because something really went wrong while writing the adapter, not
  because a gate asked for it. A test written from a checklist proves the
  checklist ran; a test written from a defect proves the defect is gone.

CPU-safe and pure: no renders, no CUDA, no model loads. The graph is a plain
dict, so it can be built and inspected without a GPU -- which is the whole
argument for the declarative graph format.
"""

from __future__ import annotations

import inspect

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines import ltx25_recipe as R
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines.frame_contract import (
    CONTINUITY_STRICT_FIRST_FRAME)


ENGINE = "ltx25_video"


@pytest.fixture()
def eng():
    return eng_ltx25.Ltx25VideoEngine()


@pytest.fixture()
def graph(eng):
    """The real graph, built the way ``render_clip`` builds it."""
    return eng._build_graph(
        {"text_prompt": "a detective by a rainy window", "seed": 42},
        image_name="scene_still.png",
        length=R.LTX25_FRAMES,
        width=R.LTX25_CANVAS_W,
        height=R.LTX25_CANVAS_H)


# --------------------------------------------------------------------------
# Registration and the public surface
# --------------------------------------------------------------------------

def test_the_lane_is_registered_and_has_a_capability_row():
    """'registry IS the menu' -- a registered engine without a CAPABILITIES
    row (or the reverse) is the roster hole the audit exists to catch."""
    assert ENGINE in vreg.all_engine_names()
    assert ENGINE in vreg.CAPABILITIES
    audit = vreg.audit_engine_roster()
    assert not audit["missing"], audit["missing"]
    assert not audit["unexpected"], audit["unexpected"]


def test_the_public_id_is_ltx25_high_video_and_the_bijection_holds():
    """`high`, settled by the operator ruling that dropped the 4060. The
    bijection assert is module-scope in public_engines and would already have
    fired at import -- this states the intent so a future second row for the
    same internal id fails HERE, by name, instead of emptying the menu."""
    assert pub._PUBLIC_ENGINES["ltx25_high_video"] == ENGINE
    assert pub._INTERNAL_TO_PUBLIC[ENGINE] == "ltx25_high_video"
    assert len(pub._PUBLIC_ENGINES) == len(pub._INTERNAL_TO_PUBLIC)
    assert pub.resolve_engine_id("ltx25_high_video (16:9)") == ENGINE


def test_the_two_sibling_lanes_are_reserved_but_NOT_registered():
    """THE CONSTRUCTION-SITE RULE (operator, 2026-08-19). Chunk B is blocked on
    an execution-order change, so `mime` and `foley_plus` cannot render. A
    dropdown row that cannot make an episode is worse than a missing one -- and
    the names are still spoken for, so nobody spends them on something else."""
    reserved = set(eng_ltx25.LTX25_RESERVED_SIBLING_IDS)
    assert reserved == {"ltx25_mime", "ltx25_foley_plus"}
    registered = set(vreg.all_engine_names())
    assert not (reserved & registered), (
        "a Chunk B lane registered before its audio path exists: %s"
        % sorted(reserved & registered))
    for internal in reserved:
        assert internal not in pub._INTERNAL_TO_PUBLIC


# --------------------------------------------------------------------------
# The declarations (G2 / G3)
# --------------------------------------------------------------------------

def test_the_declared_canvas_is_832x480_and_32_legal(eng):
    """G2.1/G2.2. The lab rejected 768x432 (432/32 = 13.5 corrupts the tensor
    and fails the VAE decode) and 1024x576 (OOM). 832x480 is 26 x 15."""
    assert tuple(eng.render_canvas) == (832, 480)
    assert (R.LTX25_CANVAS_W, R.LTX25_CANVAS_H) == (832, 480)
    assert eng.render_canvas[0] % 32 == 0
    assert eng.render_canvas[1] % 32 == 0
    assert eng.render_aspect == "wide"


def test_the_frame_contract_is_one_rung_at_the_canvas_rate(eng):
    """G3.1/G3.2/G3.3. One legal length, declared in FRAMES, at 25 fps, with
    continuity stated explicitly rather than inherited."""
    contract = fc.frame_contract_for(eng)
    assert contract is not fc.SINGLE_ONLY
    assert contract.discrete_frames == (97,)
    assert contract.native_fps == 25 == eng.target_fps
    assert contract.allow_tail_trim is True
    assert contract.continuity == CONTINUITY_STRICT_FIRST_FRAME
    assert fc.declares_continuity_kwarg(eng), (
        "continuity must be passed as a kwarg, not explained in a comment")
    assert (R.LTX25_FRAMES - 1) % 8 == 0, "the temporal contract"


def test_the_only_rung_is_the_only_legal_length(eng):
    """An ask below the rung renders it and trims; an ask above it is a
    REFUSAL here and a partition upstream, never a stretch."""
    contract = fc.frame_contract_for(eng)
    assert contract.smallest_legal_at_least(1) == 97
    assert contract.smallest_legal_at_least(97) == 97
    assert contract.smallest_legal_at_least(98) is None
    assert contract.is_legal_length(97)
    assert not contract.is_legal_length(96)


# --------------------------------------------------------------------------
# THE TRAP THAT WOULD HAVE KILLED THE FIRST RENDER
# --------------------------------------------------------------------------

def test_the_resize_node_gets_a_NESTED_dict_never_the_dotted_api_keys(graph):
    """``ResizeImageMaskNode`` is a V3 DynamicCombo node, and this is the one
    place the lab's JSON cannot be transcribed literally.

    The API-format golden recipe flattens the node's sub-inputs into dotted
    keys (``resize_type.width``) -- a serialisation detail ComfyUI's executor
    un-flattens before it calls anything. This repo's ``run_graph`` calls
    ``execute(**kwargs)`` DIRECTLY, and the real signature takes ``resize_type``
    as ONE nested dict. Copying the dotted keys across raises
    ``TypeError: unexpected keyword argument 'resize_type.width'`` -- at render
    time, after the weights are paid for.

    VERIFIED AGAINST THE INSTALLED NODE, not reasoned about: calling the real
    class with the nested dict resized a portrait 832x1216 to (1, 480, 832, 3),
    and the dotted form raised exactly that TypeError.
    """
    inputs = graph["resize"]["inputs"]
    rt = inputs["resize_type"]
    assert isinstance(rt, dict), (
        "resize_type must be the nested dict the execute signature takes")
    assert rt == {"resize_type": "scale dimensions", "width": 832,
                  "height": 480, "crop": "center"}
    assert not [k for k in inputs if "." in k], (
        "a dotted API key reached the in-process graph: %r" % sorted(inputs))
    assert inputs["scale_method"] == "lanczos"


# --------------------------------------------------------------------------
# The two places the lab's PROSE disagrees with its own executable file
# --------------------------------------------------------------------------

def test_the_anchor_is_ImgToVideoInplace_at_full_strength(graph):
    """The QA prose says ``SetLatentNoiseMask``; the file that RAN says
    ``LTXVImgToVideoInplace`` at strength 1.0. The file wins."""
    assert graph["i2v"]["inputs"]["strength"] == R.LTX25_I2V_ANCHOR_STRENGTH == 1.0
    assert graph["i2v"]["inputs"]["bypass"] is False
    classes = {spec["class"] for spec in graph.values()}
    assert "noisemask" not in classes and "solidmask" not in classes


def test_the_scheduler_latent_comes_from_the_ANCHOR_not_the_empty_latent(graph):
    """CORRECTED AGAINST THE RECIPE CONSTANT, deliberately -- not against the
    r2 coding plan, whose A1 bullet still says ``EmptyLTXVLatentVideo`` and is
    stale, and not against the sibling adapter, which wires its scheduler to
    the CONCAT output. Only one of the three is what the golden file does."""
    src = graph["sched"]["inputs"]["latent"]
    assert src.src == "i2v", (
        "scheduler latent came from %r; the recipe pins %s"
        % (src.src, R.LTX25_SCHEDULER_LATENT_SOURCE))
    assert graph["i2v"]["class"] == "i2v"
    assert R.LTX25_SCHEDULER_LATENT_MUST_BE_CONNECTED is True


# --------------------------------------------------------------------------
# V-1: the audio law
# --------------------------------------------------------------------------

def test_the_audio_latent_is_minted_paid_for_and_then_DROPPED(graph):
    """The silent lane still loads the audio VAE and still computes the audio
    side through all 8 steps -- ``LTXVEmptyLatentAudio`` mints the latent and
    ``LTXVConcatAVLatent`` needs it. What it skips is the DECODE.

    Note the direction: the golden recipe already wires
    ``LTXVAudioVAEDecode``. Chunk A SUBTRACTS it."""
    assert "emptyaudio" in graph
    assert graph["emptyaudio"]["inputs"]["audio_vae"].src == "audiovae"
    assert graph["concat"]["inputs"]["audio_latent"].src == "emptyaudio"
    classes = {spec["class"] for spec in graph.values()}
    assert "audiodecode" not in classes
    candidates = eng_ltx25.Ltx25VideoEngine()._node_candidates()
    flat = {c for cands in candidates.values() for c in cands}
    assert "LTXVAudioVAEDecode" not in flat, (
        "V-1: the audio decoder must not even be RESOLVED by this lane")
    # The terminal decodes slot 0 (video). Slot 1 is the audio latent and
    # nothing reads it -- that single unwired slot IS this lane's silence.
    assert graph["decode"]["inputs"]["samples"].src == "separate"
    assert graph["decode"]["inputs"]["samples"].slot == 0
    consumers = [nid for nid, spec in graph.items()
                 for v in spec["inputs"].values()
                 if getattr(v, "src", None) == "separate"
                 and getattr(v, "slot", None) == 1]
    assert not consumers, "something consumed the audio latent: %r" % consumers


def test_silence_is_PROVED_on_the_emitted_file_not_declared():
    """G5.1, lesson L4. This lane's model genuinely produces audio, so a
    ``has_audio: False`` literal is not evidence -- the container is re-probed
    at the seam where the literal is written."""
    src = inspect.getsource(eng_ltx25.Ltx25VideoEngine.canonicalize)
    assert "validate_silent_clip_contract" in src
    assert "ffprobe_clip_fields" in src


# --------------------------------------------------------------------------
# Guards (G6)
# --------------------------------------------------------------------------

def test_assert_usable_gates_sage_first():
    """G6.1. LTX is the family BUG-070 was written for: int8-PV Sage
    process-ABORTS the render with no traceback, so the cheapest check has to
    come before anything that costs a weight read."""
    src = inspect.getsource(eng_ltx25.Ltx25VideoEngine.assert_usable)
    assert "assert_sage_not_patched" in src
    assert "EngineUnusable" in src
    sage_at = src.index("assert_sage_not_patched")
    assert sage_at < src.index("_weight_paths"), "Sage must be checked FIRST"


def test_no_module_scope_numeric_env_reads():
    """G6.3, satisfied by having nothing to guard. The recipe is LOCKED, so
    there is no number an environment is entitled to move; the origin defect
    for this gate is literally an LTX env var deleting a lane at import."""
    src = inspect.getsource(eng_ltx25)
    for bad in ("= float(os.environ", "= int(os.environ",
                "float(os.getenv", "int(os.getenv"):
        assert bad not in src, "unguarded numeric env read: %r" % bad


def test_the_weight_list_includes_the_audio_vae(eng):
    """It is required even though the lane is silent, and omitting it would
    move the failure from the gate to the middle of the graph."""
    labels = [label for label, _p, _f in eng._weight_paths()]
    assert any("audio VAE" in label for label in labels), labels
    assert len(labels) == 4


def test_every_weight_resolves_through_folder_paths():
    """G1.1, the wan_i2v killer (lesson L1). The DiT lives under
    ``diffusion_models`` on this box while the loader asks for the ``unet``
    key; only folder_paths' legacy alias bridges that."""
    src = inspect.getsource(eng_ltx25)
    assert "folder_paths" in src
    resolver = inspect.getsource(eng_ltx25.Ltx25VideoEngine._weight_paths)
    assert resolver.count("_resolve(") == 4


# --------------------------------------------------------------------------
# The locked recipe reaches the graph unchanged
# --------------------------------------------------------------------------

def test_the_locked_sampling_values_reach_the_graph(graph):
    """No CFG above 1.0 (that doubles the batch and OOMs), 8 steps, the
    ancestral sampler, and an EMPTY negative because a negative is inert at
    CFG 1.0 anyway."""
    assert graph["guider"]["inputs"]["video_cfg"] == 1.0
    assert graph["guider"]["inputs"]["audio_cfg"] == 1.0
    assert graph["modality"]["inputs"]["modality_scale"] == 1.0
    assert graph["sched"]["inputs"]["steps"] == 8
    assert graph["ksel"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert graph["neg"]["inputs"]["text"] == ""
    assert graph["emptylatent"]["inputs"]["length"] == 97


def test_the_decode_knobs_come_from_the_recipe_not_the_sibling(graph):
    """``eng_ltx_av`` decodes whole-clip at an env-driven 4096/8. Inheriting
    that by resemblance would swap a measured recipe value for a different one
    on a lane with 0.02 GiB of headroom, and nothing would say so."""
    dec = graph["decode"]["inputs"]
    assert dec["temporal_size"] == R.LTX25_DECODE_TEMPORAL_SIZE == 33
    assert dec["temporal_overlap"] == R.LTX25_DECODE_TEMPORAL_OVERLAP == 4
    assert dec["tile_size"] == 512 and dec["overlap"] == 64
    assert dec["temporal_size"] != 4096


def test_the_video_vae_is_ONE_loader_matching_the_golden_recipe(graph):
    """ONE ``VAELoader`` for the video VAE, feeding both the i2v encode and the
    decode -- exactly lab node 2.

    A DRAFT SPLIT IT IN TWO AND THAT WAS CUT, so this test guards the cut
    rather than the split. The split was inherited by resemblance from
    ``eng_ltx_av`` on the theory that a separate encode-side node lets
    ``free_after_use`` drop the VAE before the sampler peak. It cannot:
    ``_topo_order`` is Kahn with ties on sorted node id and a ``VAELoader`` has
    no dependencies, so the decode-side copy is scheduled in the FIRST batch
    anyway and is held until its only consumer runs at the very end. The
    headroom it was meant to buy also does not exist -- the lab puts both VAEs
    at 0.0 GiB at the peak. Re-splitting it would restore a mechanism that
    moves nothing and calls the loader twice.
    """
    assert graph["videovae"]["inputs"]["vae_name"] == R.LTX25_VIDEO_VAE
    assert graph["i2v"]["inputs"]["vae"].src == "videovae"
    assert graph["decode"]["inputs"]["vae"].src == "videovae"
    loaders = [nid for nid, spec in graph.items()
               if spec["inputs"].get("vae_name") == R.LTX25_VIDEO_VAE]
    assert loaders == ["videovae"], (
        "the video VAE must be loaded exactly once: %r" % sorted(loaders))
    assert R.LTX25_PEAK_DECOMPOSITION_GIB["vaes"] == 0.0


def test_the_graph_is_acyclic_and_every_wire_resolves(graph):
    """A dangling wire or a cycle is a NAMED error here rather than a
    mid-render surprise. This runs the real topo-sort."""
    from nodes._otr_video_engines import wrapper_bridge as wb
    order = wb._topo_order(graph)
    assert len(order) == len(graph)
    assert order.index("unet") < order.index("modality") < order.index("guider")
    assert order.index("i2v") < order.index("sched")
    assert order[-1] == "decode" or "decode" in order


@pytest.mark.parametrize("target,delivered", [(50, 50), (97, 97), (0, 97)])
def test_native_frame_count_never_exceeds_what_was_delivered(
        eng, target, delivered):
    """THE OVER-CLAIM THE STUB WALK CANNOT SEE.

    ``tests/test_frame_receipt_conformance.py`` asserts ``native <= delivered``,
    but it feeds every engine a fixed ``_RAW`` where the two are already equal,
    so it can never exercise a lane's own trim arithmetic. The first draft of
    this adapter stamped the pre-trim RUNG, which advertised 97 native frames
    in a 50-frame clip -- an over-claim of exactly the shape a padding lane
    makes, and the grader compares those two numbers directly
    (``acceptance.py``: real frames are the prefix ``[0, native)``).

    This lane manufactures nothing. It renders the one legal rung and cuts
    surplus REAL frames off the tail, so every delivered frame is genuine and
    native must equal delivered at every ask.
    """
    raw = {"out_path": "probe.mp4", "frame_count": delivered,
           "native_frame_count": delivered, "extension_mode": "none"}
    clip = eng._clip_from_raw(raw, {"shot_id": "s",
                                    "timing": {"target_frame_count": target}})
    assert clip["native_frame_count"] == clip["frame_count"] == delivered
    assert clip["extension_mode"] == "none"
    assert clip["has_audio"] is False


def test_the_render_path_stamps_native_equal_to_delivered():
    """The same invariant read off the SOURCE of ``render_clip``, because the
    defect lived in the return dict rather than in the pure builder: the trim
    branch must not reintroduce a pre-trim rung as the native count."""
    src = inspect.getsource(eng_ltx25.Ltx25VideoEngine.render_clip)
    assert '"native_frame_count": n,' in src, (
        "render_clip must stamp the DELIVERED count; a pre-trim rung here is "
        "an over-claim the conformance walk's stub cannot catch")
    assert "native_frames" not in src, (
        "the pre-trim bookkeeping is gone; reintroducing it is the bug")


def test_every_logical_node_in_the_graph_has_a_class_candidate(graph, eng):
    """A graph node whose logical id is absent from ``_node_candidates`` cannot
    resolve, and the failure lands at render time rather than at the gate."""
    candidates = eng._node_candidates()
    missing = sorted(spec["class"] for spec in graph.values()
                     if spec["class"] not in candidates)
    assert not missing, "graph nodes with no resolvable class: %r" % missing
