"""Per-ENGINE coverage of the image phase's required-target enumeration.

Chunk C0 of the engine image-contract block (plan:
``kibitz-runs/2026-07-24-engine-image-contract/r2/final.md``).

Why this file exists: the 2026-07-23 qualification produced three still-spine
failures (`still_word` refusing the opening beat, `mesh_stage` with no fodder
for the opening object, `ltx_video` handed an empty init_image), and the
standing theory was that the image phase never enumerated what those engines
demanded. These pins prove that theory FALSE at HEAD -- the producer already
requires the opening-beat target for every still-consuming engine, and the
mesh fodder/plate PAIR for every mesh engine -- so the remaining suspect
surface is materialization and the render join, not enumeration.

They also lock the operator's 2026-07-25 "golden nuggets" directive at the one
seam a future engine is most likely to break: what each engine needs is asked
of the ENGINE, so adding an engine must not require editing a whitelist here.
Everything asserted below is CURRENT behaviour; a change to any of it is an
operator decision, not a refactor side effect.

Pure CPU: no ComfyUI runtime, no GPU, no image model, no LLM.
"""

from __future__ import annotations

import pytest

from nodes import otr_image_director as imgdir
from nodes import otr_image_gen_dispatcher as disp
from nodes import otr_meta_brief_image_prompt as mbp
from nodes._otr_shared import role_slots as _role_slots


OPENING_BEAT_ID = "b000_music_open"

_LINES = [
    {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer",
     "start_s": 8.4, "dur_s": 4.0},
    {"line_id": "b002", "speaker_role": "char_voice", "char_id": "c01",
     "start_s": 12.4, "dur_s": 6.0},
    {"line_id": "b005", "speaker_role": "music_close", "char_id": "music_close",
     "start_s": 18.4, "dur_s": 5.0},
]
_CAST = [{"char_id": "c01", "name": "BABA",
          "portrait_prompt": "a tall weathered spacer with a scar"}]
_META = {"episode_id": "ep_still_spine_coverage",
         "story_brief_terms": {"setting": ["a derelict orbital station"]}}


def _policies_for(engine_id):
    """The (video_policy, image_policy) pair for an episode whose every video
    role is ``engine_id`` -- the shape OTR_VideoDirector emits and
    OTR_ImageDirector forwards."""
    video_models = {
        slot: {"engine_id": engine_id, "custom": False}
        for slot in _role_slots.ROLE_TO_VIDEO_SLOT.values()
    }
    policy = {"policy_version": 2, "video_models": video_models}
    return policy, dict(policy)


def _required_target_ids(engine_id):
    video_policy, _image_policy = _policies_for(engine_id)
    payload, _warnings = mbp.derive_image_prompts(
        _CAST, _META, llm_fn=None, lines=_LINES,
        still_aspects={},
        mesh_fodder_roles=imgdir.mesh_fodder_roles_from_video_policy(
            video_policy),
        talking_roles={}, still_word_roles=None,
        video_models=video_policy["video_models"],
    )
    required = [str(row["object_id"])
                for row in (payload.get("required_scene_targets") or [])]
    minted = {str(obj["object_id"]) for obj in (payload.get("objects") or [])}
    # Every REQUIRED target must also be an object the producer authored -- a
    # required id with no object behind it is a manifest that can never be
    # satisfied.
    assert set(required) <= minted, sorted(set(required) - minted)
    return required


#: Engines whose live 2026-07-23 leg died on a missing still-spine input, plus
#: the two qualified scene-still consumers, as a regression cohort.
_SCENE_STILL_ENGINES = ("still_word", "ltx_video", "wan_ti2v", "humo")


@pytest.mark.parametrize("engine_id", _SCENE_STILL_ENGINES)
def test_scene_still_engines_require_the_opening_beat(engine_id):
    """`still_word` refused `shot_music_opening_001` live on 2026-07-23. The
    producer DOES require an opening-beat still for it (and for every other
    scene-still consumer), so enumeration is not the hole."""
    required = _required_target_ids(engine_id)
    assert f"still_{OPENING_BEAT_ID}" in required, required
    # ... and every ordinary beat keeps its own still (the 2026-06-18 rule that
    # ended the LTX-I2V missing-still degrade on dialogue beats).
    for beat_id in ("b001", "b002", "b005"):
        assert f"still_{beat_id}" in required, (beat_id, required)


def test_mesh_engine_requires_the_fodder_and_plate_pair_per_beat():
    """`mesh_stage` had no mesh fodder for the opening beat live. The producer
    requires BOTH members of the pair for every beat, opening included --
    isolated subject plus its own background plate, never one cinematic scene
    still (the 2026-06-21 'clay blob' lesson)."""
    required = _required_target_ids("mesh_stage")
    for beat_id in (OPENING_BEAT_ID, "b001", "b002", "b005"):
        assert f"meshfodder_{beat_id}" in required, (beat_id, required)
        assert f"plate_{beat_id}" in required, (beat_id, required)
    # A mesh lane must never fall back to the plain scene still.
    assert not [row for row in required if row.startswith("still_")], required


@pytest.mark.parametrize("engine_id", ("still_word", "still_pan", "still_flat",
                                       "still_motion"))
def test_still_family_consumes_a_still_without_declaring_init_image(engine_id):
    """The subtlety that makes a DERIVED requiredness rule wrong.

    The `still_*` family declares ``required_inputs=("text_prompt",)`` -- no
    ``init_image`` -- yet every one of them consumes a scene still and declares
    ``accepts_still=True``. `engine_consumes_still` reads the capability FIRST
    and only falls back to ``required_inputs``, which is the only reason these
    lanes get a still at all. Any future contract must therefore DECLARE
    requiredness, never derive it from ``required_inputs`` or family.
    """
    _video_policy, image_policy = _policies_for(engine_id)
    capabilities = disp.still_consumer_capabilities(image_policy)
    assert capabilities == {"announcer_visual": True, "music_visual": True,
                            "character_video": True}, engine_id

    from nodes._otr_video_engines import registry as vreg

    engine = vreg.get_engine(engine_id)
    assert "init_image" not in tuple(getattr(engine, "required_inputs", ()))
    assert getattr(engine, "accepts_still", None) is True
    assert disp.engine_consumes_still(engine) is True


def test_humo_portrait_and_wide_siblings_keep_their_declared_aspects():
    """Golden nugget (2026-06-17): HuMo renders PORTRAIT while its 16:9
    siblings render WIDE, and the still must match or the wide render
    decapitates the subject. Pinned per engine, from the live registry."""
    from nodes._otr_video_engines import registry as vreg

    expected = {
        "humo": "portrait",
        "humo_1.7B": "portrait",
        "humo_1.7B_169": "wide",
        "humo_14B_169": "wide",
        "wan_ti2v": "wide",
        "mesh_stage": "wide",
    }
    for engine_id, aspect in expected.items():
        assert getattr(
            vreg.get_engine(engine_id), "render_aspect", "") == aspect, engine_id


def test_talking_capability_is_read_through_the_hook_not_truthiness():
    """Golden nugget (S4b 2026-07-02 + proof8): which lane needs LIPS decides
    whether stills are minted face-forward.

    `ltx_audio_in` exposes ``wants_talking_prompt`` as a bound METHOD, so a
    bare ``getattr(...)`` truthiness test reports True for ANY engine that
    defines it, inverting lips/no-lips. The director calls the hook instead.
    This pins the call semantics so a refactor cannot regress to truthiness.
    """
    from nodes._otr_video_engines import registry as vreg
    from nodes.otr_video_director import OTRVideoDirector

    hook = getattr(vreg.get_engine("ltx_audio_in"), "wants_talking_prompt", None)
    assert callable(hook), "the talking capability is a hook, not a flag"

    resolved = OTRVideoDirector._role_talking({
        slot: {"engine_id": "viz_mxc_cpu", "custom": False}
        for slot in _role_slots.ROLE_TO_VIDEO_SLOT.values()
    })
    # A visualizer defines no hook at all -> False, never a truthy method object.
    assert resolved == {"announcer_visual": False, "music_visual": False,
                        "character_video": False}


def test_no_video_engine_is_silently_exempt_from_the_image_dropdown():
    """Operator invariant, 2026-08-21: *"All video dropdowns should obey the
    image gen dropdowns unless of course viz -- there is no image gen for
    visualization video models."*

    Obedience itself is not the fragile part: `engine_consumes_still` reads one
    capability, `resolve_engine_for_role` reads the operator's per-role pick,
    and `MotionEngineBase.accepts_still = True` means a new motion lane inherits
    obedience for free. The fragile part is SILENCE. `engine_consumes_still`
    falls back to `required_inputs` when the flag is absent, so an engine that
    subclasses no motion base AND forgets the flag AND does not list
    `init_image` resolves to False -- it mints no still, the operator's chosen
    image model is never invoked for that role, and nothing anywhere says so.
    The episode renders; it just quietly stops obeying the dropdown.

    So the rule pinned here is not "which engines consume a still" -- that is a
    per-engine decision and the parametrized pins above already own it. It is
    that refusing the dropdown must always be a DECLARED choice: every engine
    that mints no still says `accepts_still = False` out loud. Derived from the
    live registry, never a whitelist, so a new engine is covered the day it
    registers (this file's founding rule) and cannot be exempted by omission.
    """
    from nodes._otr_video_engines import registry as vreg

    obeys, exempt, silent = [], [], []
    for name in sorted(vreg.all_engine_names()):
        engine = vreg.get_engine(name)
        if disp.engine_consumes_still(engine):
            obeys.append(name)
        elif getattr(engine, "accepts_still", None) is False:
            exempt.append(name)
        else:
            silent.append(name)

    assert silent == [], (
        "these video engines mint no still yet never declared "
        "accepts_still=False, so they refuse the operator's image-gen dropdown "
        f"silently -- declare the flag either way: {silent}")
    # Both buckets stay populated, or the sweep above would pass vacuously.
    assert obeys, "no engine consumes the selected still -- the sweep is broken"
    assert exempt, "no engine is exempt -- the viz lanes should be"
