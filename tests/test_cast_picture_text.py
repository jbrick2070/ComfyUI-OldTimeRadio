"""A still reads only the drawable part of a cast description.

PBUG-20261018-03: FLUX 3 Klein lettered "Presence. ..." and "Voice: Warm ...
each word a carefully placed stone ..." into two stills of RASHIDA VOLKOV on an
8 GB leg (554ec6f6, storybook_engraving). The still path's own appearance
lookup in otr_meta_brief_image_prompt.py passed the description through whole.
"""
from __future__ import annotations

from nodes import otr_meta_brief_image_prompt as ip
from nodes._otr_shared.cast_picture_text import picture_description

#: Rashida's description from that leg's ledger, verbatim.
RASHIDA = (
    "40s, lead. Face: Oval, slight downturned mouth, deep-set eyes with "
    "perpetual tension, narrow nose, firm jawline, tightly pulled-back dark "
    "hair showing stress lines. Distinctive detail: A faint, almost invisible "
    "tremor in her lower lip when she holds a lie. Presence: Carries the "
    "heavy, suffocating weight of moral compromise and exhaustion. Voice: "
    "Warm, low register, constantly measured, each word a carefully placed "
    "stone on a precarious path.")


def test_a_still_keeps_the_face_and_drops_the_prose():
    out = picture_description(RASHIDA, still=True)
    assert out.startswith("40s, lead. Oval, slight downturned mouth")
    assert "tremor in her lower lip" in out
    for gone in ("Voice", "Presence", "Face:", "Distinctive detail:",
                 "carefully placed stone", "moral compromise"):
        assert gone not in out, gone


def test_a_video_prompt_keeps_presence_and_drops_only_voice():
    out = picture_description(RASHIDA, still=False)
    assert "Presence: Carries the heavy" in out
    assert "Face: Oval" in out
    assert "Voice" not in out and "carefully placed stone" not in out


def test_edges_never_raise_and_leave_plain_text_alone():
    assert picture_description("", still=True) == ""
    assert picture_description(None, still=True) == ""
    assert picture_description(42, still=False) == ""
    assert picture_description("A man at 10:30. Face: grim.", still=True) == \
        "A man at 10:30. grim."
    assert picture_description("Face: sharp. Voice: low", still=True) == "sharp."


def test_the_still_lookup_and_its_fallback_both_clean_the_row():
    row = {"char_id": "c02", "character_description": RASHIDA}
    for text in (ip._appearance_for_char([row], "c02"),
                 ip._row_picture_text(row)):
        assert "Voice" not in text and "Presence" not in text
        assert "Face:" not in text and "deep-set eyes" in text
    # portrait_prompt wins the key chain and is cleaned the same way.
    row2 = {"char_id": "c02", "portrait_prompt": RASHIDA,
            "character_description": "ignored"}
    assert "Voice" not in ip._appearance_for_char([row2], "c02")


def test_only_casting_labels_are_stripped_wherever_they_sit():
    """Ordinary prose with a colon is not a label (Sonnet QA on ba27903e)."""
    assert picture_description("She whispers: run. Face: oval.", still=True) \
        == "She whispers: run. oval."
    assert picture_description("Note the scar: it runs deep.", still=True) \
        == "Note the scar: it runs deep."
    assert picture_description("Hair: grey, eyes: green. Build: tall.",
                               still=True) == "grey, green. tall."


def test_a_section_runs_to_the_next_header_not_to_a_period():
    """Two sentence-based cleaners leaked Voice words through abbreviations
    (Sonnet QA on 6d8687b2: "e.g." and "a.m."). Sections end at a header."""
    for voice in ("Voice: low, e.g. gravelly.",
                  "Voice: starts at 9 a.m. sharp, gravelly.",
                  "Voice: warm, from St. Louis, gravelly."):
        for still in (True, False):
            out = picture_description("Face: oval. " + voice, still=still)
            assert "gravelly" not in out and "Voice" not in out, (voice, still)
            assert "oval" in out
    # Voice is last in the format, so an unlabelled tail goes with it.
    assert picture_description(
        "Face: oval. Voice: warm. A scar on the cheek.", still=True) == "oval."
    # A missing period still ends a section at the next header.
    assert picture_description(
        "Presence: calm Face: oval, grey eyes.", still=True) == "oval, grey eyes."
    # A sub-label inside a dropped section cannot bring its words back.
    assert picture_description(
        "Presence: He waits. Look: nothing. Face: oval.", still=True) == "oval."
    # A sub-label used as prose inside a kept section stays prose.
    assert picture_description(
        "Face: a hard look: haunted eyes.", still=True) ==         "a hard look: haunted eyes."
    assert picture_description("Face:oval Voice:low", still=True) == "oval"


def test_the_char_scene_composer_reads_the_cleaned_description():
    """compose_still_prompt built the scene_character subject from the raw row
    -- the deterministic char-scene fallback's whole prompt (Sonnet QA)."""
    from nodes import _otr_story_brief_helpers as helpers
    from nodes._otr_visual_styles import get_visual_style
    row = {"char_id": "c02", "character_description": RASHIDA}
    for kind in ("scene_character", "portrait"):
        p = helpers.compose_still_prompt(
            {}, kind=kind, role="character_visual", beat_id="b002",
            char_entry=row, style=get_visual_style({}))
        assert "deep-set eyes" in p, kind
        for gone in ("Voice", "Presence", "Face:", "carefully placed stone"):
            assert gone not in p, (kind, gone)
