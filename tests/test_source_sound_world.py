"""Chunk 2a: the sound world comes from the source, not from a dice roll.

The defect: on adaptation lanes the drawn style's `sound_world` reached the
prompt grammar, the ledger stamp and the episode canon -- so a Wells
time-travel chapter was given "a fire in the grate, a mantel clock, a teacup".
"""
from __future__ import annotations

import pytest

from nodes import _otr_source_world as world
from nodes import _otr_style_catalog as cat


# ---------------------------------------------------------------------------
# derivation: loyal to the source, deterministic, honest when it finds nothing
# ---------------------------------------------------------------------------

def test_a_sea_story_sounds_like_the_sea():
    body = "The ship stood off the harbour, and the tide ran under her deck."
    out = world.derive_source_sound_world(body)
    assert "hull" in out or "water" in out
    assert "teacup" not in out


def test_a_parlour_story_sounds_like_a_house():
    body = "He sat in the parlour by the fire, and the clock struck the hour."
    out = world.derive_source_sound_world(body)
    assert "grate" in out or "clock" in out


def test_elements_are_ordered_by_first_appearance_in_the_work():
    # A story that opens at sea and ends indoors leads with the sea.
    body = "The sea was grey that morning. Later, a fire in the parlour."
    keys = world.detect_environment_elements(body)
    assert keys.index("sea") < keys.index("fire")


def test_derivation_is_deterministic():
    body = "Thunder over the moor, and a horse on the road below."
    assert (world.derive_source_sound_world(body)
            == world.derive_source_sound_world(body))


def test_a_source_naming_nothing_recognisable_gets_the_neutral_default():
    body = "He considered the proposition, and found it wanting."
    assert (world.derive_source_sound_world(body)
            == world.NEUTRAL_PERIOD_SOUND_WORLD)


def test_the_derivation_never_returns_empty():
    # An empty sound world would let a caller silently fall back to the drawn
    # catalog palette -- the defect this closes.
    for body in ("", "   ", "xyzzy", "1234"):
        assert world.derive_source_sound_world(body).strip()


def test_element_count_is_capped():
    body = (
        "The ship in the storm, wind and fire, a clock, a horse, a train, "
        "machinery, a crowd, a court, a sword, the forest at night, the "
        "parlour, the church, the snow."
    )
    out = world.derive_source_sound_world(body)
    assert out.count(";") < 6


def test_receipt_explains_the_derivation_without_the_body():
    body = "The ship stood off the harbour by night."
    receipt = world.sound_world_receipt(body, source_ref="u:1")
    assert receipt["sound_world_version"] == world.SOUND_WORLD_VERSION
    assert "sea" in receipt["elements_detected"]
    assert receipt["used_neutral_default"] is False
    assert "harbour" not in str(receipt)


def test_receipt_flags_the_neutral_default():
    receipt = world.sound_world_receipt("He considered the proposition.")
    assert receipt["used_neutral_default"] is True
    assert receipt["elements_used"] == []


def test_non_string_body_is_refused_by_name():
    with pytest.raises(world.SourceWorldError):
        world.detect_environment_elements(None)


# ---------------------------------------------------------------------------
# figurative language must not lead the world
# ---------------------------------------------------------------------------

def test_one_figurative_mention_in_a_long_work_does_not_qualify():
    # "a storm of protest" opening a drawing-room novel would otherwise LEAD
    # the sound world, because elements are ordered by first appearance.
    body = ("A storm of protest greeted him. " + "He sat in the parlour. " * 700)
    keys = world.detect_environment_elements(body)
    assert "storm" not in keys
    assert "house" in keys


def test_a_genuine_setting_still_leads_a_long_work():
    body = ("The sea was grey. " * 60) + ("He sat in the parlour. " * 600)
    keys = world.detect_environment_elements(body)
    assert keys[0] == "sea"


def test_a_short_excerpt_still_counts_a_single_mention():
    # In a few hundred words "he sat by the fire" IS the setting.
    body = "He sat by the fire and waited for the hour to pass."
    assert "fire" in world.detect_environment_elements(body)


def test_the_receipt_reports_the_threshold_it_used():
    short = world.sound_world_receipt("He sat by the fire.")
    long_body = "He sat by the fire. " * 700
    assert short["minimum_hits_required"] == 1
    assert world.sound_world_receipt(long_body)["minimum_hits_required"] == 2


# ---------------------------------------------------------------------------
# vocabulary collisions found on real corpus text
# ---------------------------------------------------------------------------

def test_a_domestic_hall_is_a_house_not_a_castle():
    # "hall" was in BOTH patterns and court sorted first at equal position,
    # so an entrance hall drew "a great room, cloth and iron".
    keys = world.detect_environment_elements("The entrance hall was dark. "
                                             "The hall clock struck.")
    assert "house" in keys
    assert "court" not in keys


def test_a_real_castle_still_reads_as_a_court():
    keys = world.detect_environment_elements(
        "The castle throne room. The castle gate.")
    assert "court" in keys


@pytest.mark.parametrize("figurative", [
    "a train of gloomy spectres followed him, a train of sorrows",
    "Silver gave a little whistle, then a whistle again",
    "the engine of his ambition, the engine of his will",
])
def test_polysemous_railway_words_do_not_summon_a_locomotive(figurative):
    # Dumas' 1815 prison and 18th-century piracy both got a railway platform.
    assert "rail" not in world.detect_environment_elements(figurative)


def test_the_malvolio_garden_scene_derives_a_garden():
    # From a published leg: Twelfth Night 2.5 shipped with "thunder over a
    # heath, a guttering torch, a raven, rain on a castle wall" over a comic
    # garden scene. The derivation then found NOTHING and fell back to the
    # neutral bed, because the vocabulary was novel-shaped and had no garden.
    body = ("In the garden, a letter waits where Malvolio must find it. "
            "MARIA: Get you all three into the box tree. Malvolio's coming "
            "down this walk.")
    keys = world.detect_environment_elements(body)
    assert "garden" in keys
    derived = world.derive_source_sound_world(body)
    assert derived != world.NEUTRAL_PERIOD_SOUND_WORLD
    assert "thunder" not in derived


def test_an_actual_railway_still_reads_as_rail():
    keys = world.detect_environment_elements(
        "The railway platform was empty. The locomotive waited at the platform.")
    assert "rail" in keys


# ---------------------------------------------------------------------------
# the contract: ONE value reaches the prompt, the stamp and the canon
# ---------------------------------------------------------------------------

def _a_real_slug() -> str:
    return cat.adaptation_slugs()[0]


def test_grammar_carries_the_source_sound_world_not_the_drawn_one():
    slug = _a_real_slug()
    drawn = cat.get_style(slug)["sound_world"]
    mine = "water against a hull, rigging, gulls"
    grammar = cat.render_style_grammar(slug, sound_world=mine)
    assert mine in grammar
    assert drawn not in grammar


def test_grammar_without_an_override_is_unchanged():
    slug = _a_real_slug()
    assert cat.render_style_grammar(slug) == cat.render_style_grammar(
        slug, sound_world="")


def test_contract_feeds_stamp_and_grammar_from_the_same_value():
    mine = "an engine breathing at a platform, a whistle"
    contract = cat.build_story_contract(
        "seed-1", "a brief", "", {}, source_sound_world=mine)
    # The stamp the writer copies into meta...
    assert contract.sound_world == mine
    # ...and the prompt block the model actually reads.
    assert f"Sound world: {mine}." in contract.grammar


def test_contract_without_a_source_world_keeps_the_drawn_palette():
    plain = cat.build_story_contract("seed-1", "a brief", "", {})
    drawn = cat.get_style(plain.slug)["sound_world"]
    assert plain.sound_world == drawn
    assert f"Sound world: {drawn}." in plain.grammar


def test_style_selection_is_unaffected_by_the_override():
    # The override replaces the WORLD, never the style pick -- a changed slug
    # would silently move story engine and ending mode too.
    a = cat.build_story_contract("seed-7", "a brief", "", {})
    b = cat.build_story_contract(
        "seed-7", "a brief", "", {}, source_sound_world="something else")
    assert a.slug == b.slug
    assert a.story_engine == b.story_engine
    assert a.ending_mode == b.ending_mode
    assert a.ending_tag == b.ending_tag


def test_an_empty_override_is_ignored_rather_than_blanking_the_world():
    contract = cat.build_story_contract(
        "seed-2", "a brief", "", {}, source_sound_world="")
    assert contract.sound_world == cat.get_style(contract.slug)["sound_world"]


def test_the_wells_regression_a_time_machine_is_not_a_tea_party():
    # The shipped defect, as a test: the drawn adaptation palette offers a
    # domestic parlour; the source is a laboratory demonstration.
    body = (
        "The Time Traveller produced the apparatus, and set the lever. "
        "The machinery in the laboratory turned upon itself."
    )
    derived = world.derive_source_sound_world(body)
    contract = cat.build_story_contract(
        "wells-seed", "a brief", "", {}, source_sound_world=derived)
    assert "teacup" not in contract.grammar
    assert "teacup" not in contract.sound_world
    assert ("bench" in contract.sound_world
            or "mechanism" in contract.sound_world)
