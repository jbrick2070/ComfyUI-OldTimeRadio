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
