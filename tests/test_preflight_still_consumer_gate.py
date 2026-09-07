"""The image preflight must not demand weights a role's video lane never uses.

PBUG-20260907-03. The four ``viz_*`` visualizers are procedural and
audio-reactive: they declare ``accepts_still = False`` and an explicit
``still_plan = ()``, the still dispatcher honours that and mints nothing, and
the VIDEO dropdown already says so in words -- "(audio-reactive, no scene
image)". Only ``_otr_visual_assets.plan_prompt`` disagreed, adding every image
slot to the download set unconditionally.

The cost was measured, not theorised: a 2026-09-07 4060 episode published as
``..._vmcp__none__...`` -- image field ``none``, because not one still was
minted -- after the preflight had fetched 20.6 GB of ``z_image_turbo`` to reach
it. Both AMD profiles pair ``viz_mxc_cpu`` with ``z_image_turbo``.

WHAT IS DELIBERATELY NOT TESTED HERE, because it must never become true:
hiding, filtering or reordering a dropdown row. Every image engine stays
listed and selectable and the pick is still resolved (so an empty slot still
refuses); only the DOWNLOAD is skipped.
"""
import pytest

from nodes import _otr_visual_assets as va


# --------------------------------------------------------------------------
# The predicate itself, against the REAL registry -- not a fixture that could
# agree with a wrong answer.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("engine_id", [
    "viz_mxc_cpu", "viz_green", "viz_mxc_mandala", "viz_camera",
])
def test_visualizers_are_proven_to_mint_no_still(engine_id):
    assert va._proven_no_still(engine_id, None) is True


@pytest.mark.parametrize("engine_id", ["ltx_8gb", "still_flat", "still_pan"])
def test_still_consuming_lanes_still_require_their_image_weights(engine_id):
    assert va._proven_no_still(engine_id, None) is False


# --------------------------------------------------------------------------
# FAIL-SAFE DIRECTION. Absence of proof is never proof of absence: anything
# unknown or broken must REQUIRE the weights, because skipping a download the
# render then needs is a broken episode.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("engine_id", ["", None, "not_a_real_engine", "z_image_turbo"])
def test_unknown_or_empty_engine_requires_the_weights(engine_id):
    assert va._proven_no_still(engine_id, None) is False


def test_a_raising_probe_requires_the_weights():
    def explode(_):
        raise RuntimeError("registry unavailable")
    assert va._proven_no_still("viz_mxc_cpu", explode) is False


def test_a_probe_returning_a_non_false_value_requires_the_weights():
    # `is False`, never truthiness: None means "no proof", not "no still".
    assert va._proven_no_still("viz_mxc_cpu", lambda _: None) is False
    assert va._proven_no_still("viz_mxc_cpu", lambda _: 0) is False


def test_image_slot_pairing_is_derived_not_guessed():
    from nodes._otr_shared.role_slots import ROLE_TO_VIDEO_SLOT
    paired = {va._image_slot_for(v) for v in ROLE_TO_VIDEO_SLOT.values()}
    assert paired == set(va._IMAGE_SLOTS)


# --------------------------------------------------------------------------
# End to end through plan_prompt on a minimal gated prompt.
# --------------------------------------------------------------------------
def _prompt(video_by_slot, image_engine="z_image_turbo"):
    inputs = {"gate_in": ["1", 0]}
    inputs.update(video_by_slot)
    for slot in va._IMAGE_SLOTS:
        inputs[slot] = image_engine
    return {"2": {"class_type": "OTR_VideoDirector", "inputs": inputs}}


def _plan(prompt):
    """Inject identity resolvers so the test drives the SLOT LOGIC, while the
    still-consumer predicate stays the real registry-backed one."""
    def freeze(video_models):
        from nodes._otr_shared.role_slots import ROLE_TO_VIDEO_SLOT
        return {r: video_models[s] for r, s in ROLE_TO_VIDEO_SLOT.items()
                if video_models.get(s)}
    return va.plan_prompt(prompt, "1", resolve_video=lambda e: e, freeze_video=freeze)


ALL_VIZ = {s: "viz_mxc_cpu" for s in va._VIDEO_SLOTS}


def test_all_viz_lanes_plan_no_image_weights_at_all():
    plan = _plan(_prompt(ALL_VIZ))
    assert "z_image_turbo" not in plan["engines"]
    assert plan["engines"] == {"viz_mxc_cpu"}
    notes = " ".join(plan["skipped"])
    for slot in va._IMAGE_SLOTS:
        assert slot in notes, "every skipped role must be logged, never silent"
    assert "mints no still" in notes


def test_one_still_consuming_lane_still_pulls_the_image_engine():
    mixed = dict(ALL_VIZ)
    mixed["character_video_model"] = "ltx_8gb"
    plan = _plan(_prompt(mixed))
    assert "z_image_turbo" in plan["engines"], (
        "the character lane consumes a still, so its image weights are required")


def test_no_viz_anywhere_is_byte_for_byte_the_old_behaviour():
    plan = _plan(_prompt({s: "ltx_8gb" for s in va._VIDEO_SLOTS}))
    assert plan["engines"] == {"ltx_8gb", "z_image_turbo"}
    assert plan["skipped"] == []


def test_an_empty_image_slot_still_refuses_even_on_a_no_still_lane():
    """The pick is resolved BEFORE the skip, so an incomplete graph fails just
    as loudly as before. Only the download changes, never the validation."""
    prompt = _prompt(ALL_VIZ)
    prompt["2"]["inputs"]["music_image_model"] = ""
    with pytest.raises(va.VisualAssetError) as err:
        _plan(prompt)
    assert "music_image_model" in str(err.value)


# --------------------------------------------------------------------------
# THE ISOLATION CONTRACT, and why it is pinned here.
#
# The first cut of this fix hard-imported `_otr_shared.role_slots` inside
# `plan_prompt`. Run as part of the full suite that PASSED, because an earlier
# test had already put `_otr_shared` into `sys.modules`; run alone,
# `tests/test_visual_assets_stdlib.py` failed 19 tests, because it deliberately
# loads this module with NO parent package to hold it to the cold-import-clean
# promise in its own docstring.
#
# AN ORDER-DEPENDENT GREEN IS WORSE THAN A RED, so the mapping is now INJECTED
# (the pattern `resolve_video` / `freeze_video` already use) and the fallback
# resolver never raises. These tests pin both halves.
# --------------------------------------------------------------------------
def test_the_role_slot_map_resolver_never_raises():
    assert isinstance(va._default_role_video_slots(), dict)


def test_an_unavailable_role_map_skips_NOTHING_and_requires_every_engine():
    """`{}` is the fail-safe value, in the same direction as _proven_no_still:
    no role pairs to a slot, so no skip is possible and every selected image
    engine is required -- exactly the pre-fix behaviour."""
    plan = va.plan_prompt(_prompt(ALL_VIZ), "1",
                          resolve_video=lambda e: e,
                          freeze_video=lambda vm: {"announcer_visual": "viz_mxc_cpu"},
                          role_video_slots={})
    assert "z_image_turbo" in plan["engines"]
    assert not [n for n in plan["skipped"] if "mints no still" in n]


def test_the_injected_map_is_what_drives_the_skip():
    """Injecting only ONE role proves the pairing is read from the map rather
    than assumed from slot ordering: the other two roles keep their weights."""
    plan = va.plan_prompt(_prompt(ALL_VIZ), "1",
                          resolve_video=lambda e: e,
                          freeze_video=lambda vm: {"announcer_visual": "viz_mxc_cpu"},
                          role_video_slots={"announcer_visual": "announcer_video_model"})
    notes = " ".join(plan["skipped"])
    assert "announcer_image_model" in notes
    assert "music_image_model" not in notes
    assert "character_image_model" not in notes
    # the engine is still required, because two roles still consume a still
    assert "z_image_turbo" in plan["engines"]
