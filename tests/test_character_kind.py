"""A character's KIND -- human, animal, machine or creature -- as data on the
cast row, so an animal is drawn as an animal.

The bug (live, 2026-10-19 and 10-20): every portrait and scene prompt was
anchored "adult woman, " / "adult man, " / "person, ", and every cast row has a
binary gender because the voice needs one -- so "an Egyptian-type adult cat"
became a cat-woman, and with reference stills every beat followed her. Two
attempts to GUESS non-human from the appearance text both dropped real
people's anchors ("Sailor with a parrot") and were reverted (main 83375f0f).
The fix carries what the WRITER says: My Story's treatment writes ``kind``.
"human", not "person" (operator, 2026-10-20): a talking cat can be a person
in a story; it is never a human.

The contract every test below holds: a row without ``kind`` -- every lane but
My Story, and every row written before -- renders byte-identically.
"""
from __future__ import annotations

import json
import random

import pytest

from nodes import _otr_my_story as MS
from nodes import otr_meta_brief_image_prompt as MB
from nodes import otr_shot_lock as SL
from nodes._otr_shared import cast_text as CT
from nodes._otr_story_brief_helpers import compose_still_prompt

CAT = {"char_id": "c04", "name": "Cat", "gender": "female", "kind": "animal",
       "appearance": "An Egyptian-type adult cat, slender, dark gray fur, large ears"}
JANE = {"char_id": "c02", "name": "Jane", "gender": "female",
        "appearance": "a tall weathered spacer with a scar"}
META = {"episode_id": "ep_kind", "episode_title": "The Toll",
        "story_brief_terms": {"setting": ["a dockside station"]}}


@pytest.mark.parametrize("value, kind", [
    (None, "human"), ("", "human"), ("HUMAN", "human"), ("person", "human"),
    ("Animal", "animal"), (" machine ", "machine"), ("creature", "creature"),
    ("cat", "human"), ("robot", "human"),
])
def test_kind_normalizes_and_defaults_to_human(value, kind):
    assert CT.normalize_kind(value) == kind


def test_the_treatment_carries_kind_into_the_cast_row():
    member = MS.CastMember(name="Cat", appearance=CAT["appearance"],
                           gender="female", kind="Animal")
    assert member.kind == "animal"
    assert MS.CastMember(name="Jane", gender="female").kind == "human"
    treatment = MS.StoryTreatment.model_construct(cast=[member])
    rows = MS._assign_voices(treatment, random.Random(0))
    assert [r.get("kind") for r in rows if r.get("name") == "Cat"] == ["animal"]


def test_the_ledger_keeps_a_non_human_kind_and_nothing_for_a_human():
    from nodes.production_ledger import Ledger
    led = Ledger.__new__(Ledger)
    led.data = {}
    led.set_cast([dict(CAT), dict(JANE), dict(JANE, char_id="c05", kind="person")])
    by_id = {r["char_id"]: r for r in led.data["cast"]}
    assert by_id["c04"]["kind"] == "animal"
    assert "kind" not in by_id["c02"] and "kind" not in by_id["c05"]


def test_an_animal_gets_no_human_anchor_and_a_human_keeps_it():
    assert MB._ensure_gender_anchor(CAT["appearance"], CAT) == CAT["appearance"]
    assert MB._ensure_gender_anchor("a figure in the rain", JANE).startswith("adult woman, ")
    # "person" written by hand reads as human: anchored exactly as before.
    assert MB._ensure_gender_anchor("a figure", dict(JANE, kind="person")) == \
        MB._ensure_gender_anchor("a figure", JANE)


def test_a_human_request_is_byte_identical_and_an_animal_request_says_animal():
    human = MB._build_char_prompt_request(JANE, META, "a dockside station", "portrait")
    assert "CHARACTER THEMSELVES -- a person with a clearly visible face, " in human
    assert "the character is a person in the STORY's world, not a performer" in human
    cat = MB._build_char_prompt_request(CAT, META, "a dockside station", "portrait")
    assert "a person" not in cat
    assert "the animal its appearance describes, its face clearly visible" in cat
    assert "the character is an animal in the STORY's world" in cat
    scene_h = MB._build_char_scene_request(JANE, META, "a dock", {"text": "Hi."})
    assert "CHARACTER THEMSELVES -- a person with a clearly visible face -- as the subject" in scene_h
    assert "the character is a person in the STORY's world." in scene_h
    scene_c = MB._build_char_scene_request(CAT, META, "a dock", {"text": "Meow."})
    assert "a person" not in scene_c


def test_the_deterministic_fallback_draws_the_kind_not_a_costumed_person():
    out = compose_still_prompt(META, kind="portrait",
                               char_entry={"char_id": "c09", "kind": "machine"})
    assert out.startswith("a machine, clearly visible")
    human = compose_still_prompt(META, kind="portrait", char_entry={"char_id": "c09"})
    assert human.startswith("a period-dressed character, face clearly visible")


def test_video_prompts_follow_the_kind():
    ledger = {"cast": [CAT, JANE]}
    assert not SL._appearance_for_char(ledger, "c04").startswith("adult")
    assert SL._appearance_for_char(ledger, "c02").startswith("adult woman, ")


def test_the_correction_context_names_a_non_human_kind_only():
    assert MB._kind_entry(CAT) == {"kind": "animal"}
    assert MB._kind_entry(JANE) == {} and MB._kind_entry(dict(JANE, kind="person")) == {}


def test_the_treatment_prompt_asks_for_kind():
    from pathlib import Path
    seam = json.loads(Path(MS.__file__).with_name("story_packs").joinpath(
        "my_story", "my_story.json").read_text(encoding="utf-8"))[
        "prompt_stages"]["my_story_treatment_system"]
    assert '"kind": "human"' in seam
    assert 'kind is what the character IS: "human", "animal", "machine" or "creature"' in seam
    assert "A human's appearance names their gender plainly -- a woman or a man" in seam
