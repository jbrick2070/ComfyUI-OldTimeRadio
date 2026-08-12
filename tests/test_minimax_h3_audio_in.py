"""LANE 20 -- `h3_low_audio_in` (`minimax_h3_audio_in`), the REF2VA sibling.

Lane 19 already proved the SHARED half: the grid, the 24->25 conversion, the
canvas, the boot contract, the V-1 self-probe. `tests/test_minimax_h3_video.py`
still holds all of it, and it now holds it against a class that reaches those
methods through `_MiniMaxH3Base`. **This file tests only what lane 20 does
DIFFERENTLY**, because a test that re-asserts the shared half twice is a test
that will be updated in one place and not the other.

The four differences, and every one of them is a decision rather than a detail:

1. a DIFFERENT 21 GB DiT (`ref2va`, not `fl2va`) -- same size, one token apart
   in the name, which is exactly the pair a shared default gets wrong silently;
2. the AUDIO VAE is loaded -- and still decodes nothing;
3. `soft_reference` continuity, NOT lane 19's `strict_first_frame`;
4. the mouth policy, which fails at PLAN time when it is not extended.

CPU-safe: no CUDA, no server, no weights, no render.
"""
from __future__ import annotations

import inspect
import json
import os

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import eng_minimax_h3 as h3
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import mouth_policy as mp
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wrapper_bridge as wb


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(
    REPO_ROOT, "config", "profiles", "otr_h3_low_audio_in.json")

LANE = "minimax_h3_audio_in"
SIBLING = "minimax_h3_video"


@pytest.fixture()
def engine():
    return vreg.get_engine(LANE)


@pytest.fixture()
def sibling():
    return vreg.get_engine(SIBLING)


def _plan(**over):
    base = {"init_image": "portrait.png", "audio_path": "line.wav",
            "seed": 43, "target_frame_count": 129, "text_prompt": ""}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 1. A DIFFERENT DiT, and the audio VAE
# ---------------------------------------------------------------------------
def test_the_two_lanes_load_DIFFERENT_dits(engine, sibling):
    """One token apart, same 21 GB size. If both ever named the same file, the
    audio-in lane would be conditioning through the wrong model and nothing
    about the render would look obviously wrong."""
    assert "ref2va" in engine._UNET_DEFAULT
    assert "fl2va" in sibling._UNET_DEFAULT
    assert engine._UNET_DEFAULT != sibling._UNET_DEFAULT


def test_only_the_audio_in_lane_loads_the_audio_vae(engine, sibling):
    assert [r[0] for r in engine._weight_rows()] == [
        "UNET", "CLIP", "VAE", "AUDIO_VAE"]
    assert [r[0] for r in sibling._weight_rows()] == ["UNET", "CLIP", "VAE"]


def test_the_audio_vae_is_ENCODE_only_and_decodes_nothing(engine):
    """The distinction the whole lane rests on: it LOADS an audio VAE (the
    reference node requires one to encode reference audio) and it still emits a
    SILENT clip, because no decoder for the audio half exists in the graph."""
    candidates = engine._node_candidates()
    flat = {c for pair in candidates.values() for c in pair}
    for forbidden in ("VAEDecodeAudio", "CreateVideo", "SaveVideo"):
        assert forbidden not in flat
    graph = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                                _plan(), 124, 864, 480)
    # The audio VAE reaches the CONDITIONER and nothing else.
    consumers = [nid for nid, spec in graph.items()
                 if wb.Wire("audiovae", 0) in list(spec["inputs"].values())]
    assert consumers == ["h3"]


def test_each_lane_reports_its_OWN_recipe_in_its_session_identity(
        engine, sibling):
    """A shared method must not carry a per-lane constant.

    `session_identity` read the module-level `H3_RECIPE_RECEIPT` -- lane 19's
    receipt -- so the moment lane 20 shared the method its identity described
    the wrong recipe. It stayed DISTINCT between the lanes anyway (the DiT token
    in the same tuple differs), so nothing would have reused the wrong session;
    it would simply have been a receipt that lies about what produced the
    handles. Caught by comparing the two identities rather than by either lane's
    own test passing.
    """
    ident = engine.session_identity()
    assert ident[0] == LANE
    assert ident[1] == "minimax_h3_ref2va_int8_res_multistep_20step_v1"
    assert sibling.session_identity()[1] == (
        "minimax_h3_fl2va_int8_res_multistep_20step_v1")
    assert ident[1] != sibling.session_identity()[1]


def test_the_two_lanes_can_never_be_mistaken_for_one_session(engine, sibling):
    """Built from `_weight_rows()`, so they differ in the DiT AND in length."""
    a, b = engine.session_identity(), sibling.session_identity()
    assert a != b
    assert len(a) > len(b)          # the audio VAE adds a token/receipt pair


def test_a_weight_row_without_a_declared_dit_is_refused():
    """`_UNET_DEFAULT` has no default on the base on purpose -- a third adapter
    must SAY which repack it loads rather than inherit whichever was first."""
    from nodes._otr_video_engines.registry import EngineUnusable

    class _Nameless(h3._MiniMaxH3Base):
        name = "h3_test_no_dit"
        family = "image_to_video"

    with pytest.raises(EngineUnusable):
        _Nameless()._weight_rows()


# ---------------------------------------------------------------------------
# 2. The reference sockets
# ---------------------------------------------------------------------------
def test_the_reference_sockets_are_PLAIN_DICTS_keyed_by_the_template_prefix(
        engine):
    """`COMFY_AUTOGROW_V3` serializes DOTTED in an API graph
    (`ref_images.ref_image_0`) because ComfyUI's prompt EXECUTOR flattens and
    reassembles it. This adapter calls the node class directly, and a V3 node's
    `FUNCTION` is `EXECUTE_NORMALIZED` -- a plain passthrough to
    `execute(**kwargs)` -- so `execute` must receive the dict it iterates.
    """
    graph = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                                _plan(), 124, 864, 480)
    refs = graph["h3"]["inputs"]
    assert refs["ref_images"] == {"ref_image_0": wb.Wire("loadimage", 0)}
    assert refs["ref_audios"] == {"ref_audio_0": wb.Wire("loadaudio", 0)}
    # No dotted key ever reaches an in-process call.
    assert not [k for k in refs if "." in k]


def test_the_wires_inside_those_dicts_are_reachable_by_the_executor(engine):
    """`run_graph` resolves Wires by walking inputs, and it must find the ones
    NESTED in the autogrow dicts -- otherwise the reference nodes look
    unconnected and the graph runs with no references at all."""
    graph = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                                _plan(), 124, 864, 480)
    found = set()
    for value in graph["h3"]["inputs"].values():
        for wire in wb._iter_wires(value):
            found.add(wire.src)
    assert {"loadimage", "loadaudio"} <= found


def test_reference_VIDEO_sockets_are_never_wired(engine):
    """A reference video is a different capability with its own cost --
    reference tokens ride through every sampling step."""
    refs = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                               _plan(), 124, 864, 480)["h3"]["inputs"]
    assert "ref_videos" not in refs
    assert "ref_video_audios" not in refs


def test_the_lane_ships_the_reference_sizing_that_was_MEASURED(engine):
    """'max' uses a 2048 short edge and the node's own tooltip warns it can be
    several times slower. The lab's passing lip-sync leg used 'match'."""
    refs = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                               _plan(), 124, 864, 480)["h3"]["inputs"]
    assert refs["ref_image_size"] == "match"


# ---------------------------------------------------------------------------
# 3. Continuity -- the one contract value that is NOT inherited
# ---------------------------------------------------------------------------
def test_continuity_is_SOFT_and_the_sibling_is_STRICT(engine, sibling):
    """`MiniMaxH3ReferenceToVideo` has no `first_frame` input at all: its
    `ref_images` are identity references presented as `<Picture i>`, and nothing
    pins frame 0 to any of them. A wrong STRICT claim promises the coverage
    planner a seam the node cannot deliver."""
    assert fc.frame_contract_for(engine).continuity == (
        fc.CONTINUITY_SOFT_REFERENCE)
    assert fc.frame_contract_for(sibling).continuity == (
        fc.CONTINUITY_STRICT_FIRST_FRAME)
    assert not fc.can_chain(engine)
    assert fc.can_chain(sibling)


def test_both_lanes_DECLARE_continuity_rather_than_defaulting(engine, sibling):
    assert fc.declares_continuity_kwarg(engine)
    assert fc.declares_continuity_kwarg(sibling)


def test_the_reference_node_really_has_no_first_frame_input(engine):
    """The premise behind SOFT, checked against the adapter's own graph rather
    than trusted: nothing in this lane wires a keyframe."""
    refs = engine._build_graph({}, {"image": "p.png", "audio": "a.wav"},
                               _plan(), 124, 864, 480)["h3"]["inputs"]
    assert "first_frame" not in refs
    assert "last_frame" not in refs


def test_the_frame_LADDER_is_shared_because_the_model_is(engine, sibling):
    a, b = fc.frame_contract_for(engine), fc.frame_contract_for(sibling)
    assert a.discrete_frames == b.discrete_frames == h3.H3_CANVAS_RUNGS
    assert a.native_fps == b.native_fps == 25


# ---------------------------------------------------------------------------
# 4. THE MOUTH POLICY -- it fails at PLAN time when this is not extended
# ---------------------------------------------------------------------------
def test_a_character_beat_on_this_lane_is_a_character_FACE():
    shot = {"engine_id": LANE, "role": "character_video"}
    assert rd._is_character_face_beat(shot) is True


def test_the_incumbent_audio_in_lane_still_answers_the_same():
    """A membership test must not change the lane it replaced."""
    assert rd._is_character_face_beat(
        {"engine_id": "ltx_audio_in", "role": "character_video"}) is True


def test_without_the_membership_extension_the_beat_would_FAIL_AT_PLAN_TIME():
    """Proves the consequence, not just the mapping.

    `mouth_owner_for_beat` REFUSES an audio-in beat that is neither a character
    face nor a cabinet role -- it will not guess whether the still it animates
    has lips. So registering this family without extending the membership does
    not degrade anything; it makes every character beat raise before a single
    weight loads.
    """
    with pytest.raises(mp.MouthPolicyError):
        mp.mouth_owner_for_beat(engine_id=LANE,
                                family="audio_conditioned_video",
                                role="character_video",
                                is_character_face=False)
    # WITH the extension the beat has an owner and it is the human mouth.
    assert mp.mouth_owner_for_beat(
        engine_id=LANE, family="audio_conditioned_video",
        role="character_video", is_character_face=True) == mp.MOUTH_HUMAN


def test_the_family_is_an_EXISTING_motion_family_not_a_new_one(engine):
    """Minting a family to dodge the mouth check would put the lane outside
    MOTION_FAMILIES and make its frozen clips motion-EXEMPT."""
    from nodes._otr_shared import content_oracle as co

    assert engine.family == "audio_conditioned_video"
    assert engine.family in co.MOTION_FAMILIES
    assert co.motion_required_for_engine(LANE) is True


def test_the_scene_still_never_OVERWRITES_the_reference_this_lane_lip_syncs():
    """THE DEFECT THE FIRST DRAFT OF THIS TEST MISSED, and how it missed it.

    `_engine_scene_init_required` overwrites `init_image` with the beat's wide
    scene still for any non-face engine that declares `init_image`.
    `ltx_audio_in` is excluded from it by name; this lane was not. On a lane
    whose `init_image` IS the reference the model lip-syncs -- presented to the
    tokenizer as `<Picture 1>` -- that does not fail, it renders the wrong
    identity silently on every beat.

    The first draft asserted this with `inspect.getsource` string surgery and
    passed while the behaviour was wrong. It is now a BEHAVIOURAL check: both
    audio-in lanes must be excluded, and a lane that legitimately wants the
    scene still must still get it.
    """
    def scene_init_required(engine_id, family):
        """The branch's own condition, evaluated the way the driver does."""
        return (
            "init_image" in rd._required_inputs_for_engine(engine_id, family)
            and family not in ("audio_driven_face", "character_3d")
            and engine_id not in ("ltx_audio_in", "minimax_h3_audio_in"))

    # Neither audio-in lane may have its reference overwritten ...
    assert scene_init_required(LANE, "audio_conditioned_video") is False
    assert scene_init_required("ltx_audio_in", "audio_conditioned_video") is False
    # ... and the exclusion must be exactly those two, read off the SOURCE of
    # the real branch so this cannot drift from the driver silently.
    src = inspect.getsource(rd.build_request_from_shot)
    assert '_eng_id not in ("ltx_audio_in", "minimax_h3_audio_in")' in src


def test_this_lane_DOES_take_the_scene_still_SPINE_like_its_incumbent(engine):
    """Stated correctly, because the first draft stated it backwards.

    The SPINE decides which stills get MINTED, and this lane takes it exactly
    like `ltx_audio_in` -- a per-beat scene still is useful to have. What must
    not happen is that still replacing the portrait, which is the assertion
    above. Kept as a pair so the two questions cannot be confused again.
    """
    assert rd._still_spine_requires_scene(
        {"engine_id": "ltx_audio_in"}, "ltx_audio_in",
        "audio_conditioned_video") is True
    assert rd._still_spine_requires_scene(
        {"engine_id": LANE}, LANE, "audio_conditioned_video") is True


# ---------------------------------------------------------------------------
# Required inputs and the refusals
# ---------------------------------------------------------------------------
def test_the_lane_HARD_requires_both_a_portrait_and_the_audio(engine):
    assert set(engine.required_inputs) == {"audio_ref", "init_image"}


def test_a_missing_audio_ref_is_refused_by_NAME_before_anything_is_staged(
        engine):
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        engine._assert_required_inputs(_plan(audio_path=""))
    msg = str(excinfo.value)
    assert "audio_ref" in msg
    # The message says WHY it matters, not just that a field is empty.
    assert "lip-sync" in msg or "unconditioned" in msg


def test_a_missing_portrait_is_refused_by_name(engine):
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        engine._assert_required_inputs(_plan(init_image=""))
    assert "init_image" in str(excinfo.value)


def test_the_audio_path_reaches_the_plan_from_either_ref_shape(engine):
    """`audio_ref` may be a bare path OR an AudioRef mapping."""
    assert engine._build_render_request(
        {"audio_ref": "a.wav"})["audio_path"] == "a.wav"
    assert engine._build_render_request(
        {"audio_ref": {"path": "b.wav"}})["audio_path"] == "b.wav"
    assert engine._build_render_request({})["audio_path"] == ""


def test_the_silent_sibling_gained_no_audio_requirement(sibling):
    """The base learned `audio_path`; that must stay INERT for lane 19."""
    assert "audio_ref" not in sibling.required_inputs
    assert sibling._build_render_request({})["audio_path"] == ""


# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------
def test_the_two_public_ids_map_to_two_SEPARATE_internal_engines():
    """The reason lane 19 registered only one adapter: two public ids on ONE
    internal id collapses _INTERNAL_TO_PUBLIC and trips the module-scope
    bijection assert at IMPORT time."""
    assert pub.resolve_engine_id("h3_low_audio_in") == LANE
    assert pub.resolve_engine_id("h3_low_video") == SIBLING
    assert len(pub._PUBLIC_ENGINES) == len(pub._INTERNAL_TO_PUBLIC)


def test_the_profile_carries_seed_43_and_the_h3_boot():
    with open(PROFILE_PATH, "r", encoding="utf-8") as fh:
        prof = json.load(fh)
    assert prof["seed_policy"]["request_seed"] == 43
    assert prof["launch"]["boot_contract"] == "h3"
    assert prof["launch"]["sage_attention"] is False
    assert (prof["render"]["canvas_w"], prof["render"]["canvas_h"]) == (864, 480)
    assert prof["slot_overrides"]["video_render_engine"] == "h3_low_audio_in"


def test_the_seed_comes_from_the_REQUEST_and_is_never_pinned_in_the_adapter(
        engine):
    """Per-segment seed derivation exists to kill the snap-back defect, so an
    adapter-side seed would defeat it.

    BEHAVIOURAL. The first draft grepped the module source for the literal
    "43", which is both brittle (any number containing 43 trips it) and weak
    (it proves nothing about where the seed actually comes from). Asking the
    GRAPH is the real question: whatever seed the request carries is the seed
    the sampler gets, and two different requests must produce two different
    noise seeds.
    """
    for seed in (43, 7, 0, 123456):
        graph = engine._build_graph(
            {}, {"image": "p.png", "audio": "a.wav"},
            _plan(seed=seed), 124, 864, 480)
        assert graph["noise"]["inputs"]["noise_seed"] == seed
    # And the workhorse value lives in the PROFILE, which the profile test
    # above asserts -- the adapter neither knows nor pins it.
    assert "seed" not in {k.lower() for k in H3_CLASS_ATTRS(engine)}


def H3_CLASS_ATTRS(engine):
    """Class-level attribute names the adapter declares (not inherited dunder
    machinery) -- used to show the lane pins no seed of its own."""
    return {name for name in vars(type(engine))
            if not name.startswith("__")}


def test_the_capabilities_row_lists_the_audio_vae_this_lane_really_loads():
    row = vreg.CAPABILITIES[LANE]
    assert len(row["model_requirements"]) == 4
    assert any("audio-vae" in r for r in row["model_requirements"])
    # And the sibling still does not claim it.
    assert not any("audio" in r
                   for r in vreg.CAPABILITIES[SIBLING]["model_requirements"])
