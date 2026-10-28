"""Scene regions: the beat's own objects, each with its own region.

Pins the two rules that matter. The objects come ONLY from the episode's own key-object pool and a noun
the beat negates is never a region (operator ruling 2026-09-03: beat reference ranks, it never sources);
and the masks are deterministic, bounded and laid out by count.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import ghost_scene_regions as R
from nodes._otr_video_engines import ghost_signal_author as GSA

META = {
    "key_objects": ["a brass lantern", "a handwritten ledger", "a coal truck"],
    "story_brief_terms": {"setting": ["the harbour"]},
}

CHARACTER = dict(role="character_video", mode="object")


def _objects(beat_text, **kw):
    args = dict(CHARACTER)
    args.update(kw)
    return R.resolve_scene_objects(META, beat_text=beat_text, **args)


def test_a_bookend_beat_gets_no_regions():
    out = R.resolve_scene_objects(META, role="announcer_visual", mode="object",
                                  beat_text="the lantern is lit")
    assert out == {"objects": (), "source": "none"}


def test_an_episode_with_no_key_objects_gets_no_regions():
    out = R.resolve_scene_objects({"key_objects": []}, beat_text="the lantern", **CHARACTER)
    assert out["objects"] == ()


def test_the_objects_the_dialogue_names_become_regions_in_pool_spelling():
    out = _objects("Bring the lantern over here, I need the ledger open.")
    assert set(out["objects"]) == {"a brass lantern", "a handwritten ledger"}
    assert len(out["objects"]) == 2


def test_the_kernel_object_leads_so_the_region_reinforces_the_global_prompt():
    out = _objects("Bring the lantern over here, I need the ledger open.")
    kernel, _ = GSA.resolve_crux_kernel(META, beat_text="Bring the lantern over here, I need the ledger open.",
                                           **CHARACTER)
    assert kernel.startswith(out["objects"][0])


def test_a_noun_the_beat_only_negates_is_never_a_region():
    line = "This isn't just a list of truck routes, it is the heartbeat of the harbour."
    out = _objects(line)
    assert "a coal truck" not in out["objects"]


def test_no_region_is_ever_a_noun_outside_the_pool():
    out = _objects("Hand me the telescope and the violin, quickly.")
    pool = set(GSA.ghost_subject_candidates(META))
    assert set(out["objects"]) <= pool


def test_at_most_three_regions():
    meta = {"key_objects": ["a lantern", "a ledger", "a coat", "a key"]}
    out = R.resolve_scene_objects(meta, role="character_video", mode="object",
                                  beat_text="the lantern, the ledger, the coat and the key")
    assert len(out["objects"]) <= R.SCENE_REGION_MAX_OBJECTS


def test_the_same_inputs_give_the_same_answer():
    a = _objects("Bring the lantern, I need the ledger.")
    b = _objects("Bring the lantern, I need the ledger.")
    assert a == b


@pytest.mark.parametrize("line", [
    "There is no lantern in this room.",
    "I don't see a lantern anywhere.",
    "We have no ledger and never had a lantern.",
    "Take anything else instead of the lantern.",
])
def test_a_bare_negation_the_kernels_own_cues_miss_never_makes_a_region(line):
    out = _objects(line)
    assert "a brass lantern" not in out["objects"]


def test_the_negation_does_not_reach_across_a_sentence():
    out = _objects("There is no ledger here. Bring the lantern.")
    assert out["objects"][0] == "a brass lantern"


def test_a_word_inside_another_word_is_not_a_mention():
    meta = {"key_objects": ["a brass ring"]}
    out = R.resolve_scene_objects(meta, role="character_video", mode="object",
                                  beat_text="During the storm she waited by the clamp.")
    assert out["objects"] == ()


def test_a_plural_in_the_dialogue_still_counts():
    out = _objects("Light the lanterns, all of them.")
    assert "a brass lantern" in out["objects"]


def test_the_positional_kernel_pick_gets_no_region():
    """The odometer tier is chosen by where the beat sits, not by what it says."""
    out = _objects("Nothing is happening here at all.")
    assert out == {"objects": (), "source": "none"}


def test_the_authors_ranked_pick_for_the_beat_does_get_a_region():
    out = _objects("Nothing is happening here at all.", authored_subject="a handwritten ledger")
    assert out["objects"] == ("a handwritten ledger",)
    assert out["source"] == "authored_subject"


def test_a_pick_the_beat_negates_is_refused_even_when_the_author_ranked_it():
    out = _objects("There is no ledger on this desk.", authored_subject="a handwritten ledger")
    assert "a handwritten ledger" not in out["objects"]


def test_two_pool_entries_for_the_same_thing_are_one_region():
    meta = {"key_objects": ["a ledger", "a handwritten ledger"]}
    out = R.resolve_scene_objects(meta, role="character_video", mode="object",
                                  beat_text="Bring the ledger here.")
    assert len(out["objects"]) == 1


@pytest.mark.parametrize("count,layout", [(1, "center"), (2, "halves"), (3, "thirds")])
def test_the_layout_follows_the_count(count, layout):
    assert R.region_layout_name(count) == layout
    masks = R.region_masks(count, 512, 288)
    assert len(masks) == count
    for mask in masks:
        assert mask.shape == (288, 512)
        assert mask.dtype.name == "float32"
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
        assert float(mask.max()) > 0.9          # every region reaches full strength somewhere


@pytest.mark.parametrize("bad", [0, 4, -1, "x", None])
def test_a_count_outside_one_to_three_is_refused_by_name(bad):
    with pytest.raises(ValueError):
        R.region_layout_name(bad)


def test_halves_are_left_and_right_and_overlap_only_at_the_seam():
    left, right = R.region_masks(2, 512, 288)
    assert left[:, :64].mean() > 0.95 and right[:, :64].mean() < 0.05
    assert right[:, -64:].mean() > 0.95 and left[:, -64:].mean() < 0.05
    assert float(abs(left + right - 1.0).max()) < 1e-5     # a partition of unity


def test_thirds_each_own_their_own_third():
    masks = R.region_masks(3, 510, 288)
    for index, mask in enumerate(masks):
        center = int((index + 0.5) * 510 / 3)
        assert mask[:, center].mean() > 0.95
        for other, other_mask in enumerate(masks):
            if other != index:
                assert other_mask[:, center].mean() < 0.05


def test_the_center_mask_is_strongest_in_the_middle_and_gone_at_the_corners():
    (mask,) = R.region_masks(1, 512, 288)
    assert mask[160, 256] > 0.95
    assert mask[0, 0] < 0.01 and mask[-1, -1] < 0.01


def test_masks_and_digest_are_deterministic():
    one = R.region_masks(3, 512, 288)
    two = R.region_masks(3, 512, 288)
    assert R.mask_digest(one) == R.mask_digest(two)
    assert R.mask_digest(one) != R.mask_digest(R.region_masks(2, 512, 288))


def test_a_tiny_frame_is_refused():
    with pytest.raises(ValueError):
        R.region_masks(2, 4, 4)
