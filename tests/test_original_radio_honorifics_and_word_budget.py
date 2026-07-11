"""original_radio cleanup chunk (2026-07-11) -- the two P1 fixes from
docs/2026-07-11-original-radio-26roll-review.md.

P1-2: deterministic honorific stripping on LLM-generated cast names
      ('DR. ELI CROSS' -> 'ELI CROSS'), applied at parse time on BOTH
      creative-front cast models, so the name is normalized the moment it is
      accepted -- before the concept/select gates and before the key_terms
      grounding corpus is built.

P1-1: the per-beat length rider is now SCALED to the beat's own word
      allocation (concrete technique steering), and the writer stamps the
      planned + actual word budget on meta.word_budget. Word counts stay
      ADVISORY -- these tests pin that NOTHING rejects/rerolls/trims on count,
      and that small-target beats keep the ORIGINAL one-breath rider
      byte-for-byte.

ASCII-only source, no em-dash.
"""
from __future__ import annotations

import pytest

from nodes import _otr_line_composer as LC
from nodes._otr_original_radio import (
    CastSketchEntry,
    ConceptPitches,
    SelectCastEntry,
    SelectedConcept,
    strip_leading_honorifics,
)


# --------------------------------------------------------------------------- #
# P1-2 -- honorific stripping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("DR. ELI CROSS", "ELI CROSS"),
    ("Dr. Eli Cross", "Eli Cross"),
    ("PROF. MARTHA VANE", "MARTHA VANE"),
    ("Mrs. Martha Vane", "Martha Vane"),
    ("CAPT. ELI CROSS", "ELI CROSS"),
    ("Capt Eli Cross", "Eli Cross"),          # no trailing period
    ("REVEREND ELI CROSS", "ELI CROSS"),
    ("LT. COL. ELI CROSS", "ELI CROSS"),      # stacked titles
    ("MISS MARTHA VANE", "MARTHA VANE"),
    # already clean -> untouched
    ("ELI CROSS", "ELI CROSS"),
    ("MARTHA VANE", "MARTHA VANE"),
])
def test_strip_leading_honorifics(raw, expected):
    assert strip_leading_honorifics(raw) == expected


def test_strip_only_leads_never_eats_the_surname():
    """Unlike fable2 (one-word surname billing), original_radio keeps the WHOLE
    remaining two-word personal name -- the strip must not collapse to the last
    token."""
    assert strip_leading_honorifics("DR. ELI CROSS") == "ELI CROSS"
    assert strip_leading_honorifics("DR. ELI CROSS") != "CROSS"


def test_title_only_name_is_left_alone_for_the_empty_name_gate():
    """A name that is nothing BUT a title must not be normalized into an empty
    string -- the normalizer never manufactures an empty cast name; the
    existing empty-name gates own that rejection."""
    assert strip_leading_honorifics("DR.") == "DR."
    assert strip_leading_honorifics("") == ""


def test_non_string_passthrough():
    assert strip_leading_honorifics(None) is None
    assert strip_leading_honorifics(42) == 42


def test_cast_sketch_entry_normalizes_at_parse_time():
    e = CastSketchEntry(name="DR. ELI CROSS", role="lead")
    assert e.name == "ELI CROSS"


def test_select_cast_entry_normalizes_at_parse_time():
    e = SelectCastEntry(name="DR. ELI CROSS", want="w", pressure="p",
                        relationship="r")
    assert e.name == "ELI CROSS"


def test_dr_eli_cross_cast_name_and_label_consistency():
    """The regression the review calls for: 'DR. ELI CROSS' emitted anywhere on
    the creative front lands as ONE consistent cast label everywhere -- the
    concept pitch's cast_sketch AND the selected cast agree, so cast-membership
    matching downstream never sees the titled variant."""
    pitches = ConceptPitches.model_validate({
        "pitches": [{
            "logline": "A lab loses its own clock.",
            "hook": "The countdown answers back.",
            "radio_device": "a ticking relay",
            "cast_sketch": [
                {"name": "DR. ELI CROSS", "role": "lead"},
                {"name": "MRS. MARTHA VANE", "role": "foil"},
            ],
        }],
    })
    sketch_names = [c.name for c in pitches.pitches[0].cast_sketch]
    assert sketch_names == ["ELI CROSS", "MARTHA VANE"]

    sel = SelectedConcept.model_validate({
        "selected_index": 0,
        "pitch": {"logline": "A lab loses its own clock.", "hook": "h",
                  "radio_device": "d"},
        "cast": [
            {"name": "DR. ELI CROSS", "want": "w", "pressure": "p",
             "relationship": "r"},
            {"name": "MRS. MARTHA VANE", "want": "w", "pressure": "p",
             "relationship": "r"},
        ],
        "selection_rationale": "it plays.",
    })
    cast_names = [c.name for c in sel.cast]
    assert cast_names == ["ELI CROSS", "MARTHA VANE"]
    # label consistency: the selected cast label matches the sketch label
    assert cast_names == sketch_names


# --------------------------------------------------------------------------- #
# P1-1 -- target-scaled length steering (advisory, never a gate)
# --------------------------------------------------------------------------- #
def test_small_target_rider_is_byte_identical_to_the_original():
    """Beats at/below the one-breath threshold must render the ORIGINAL rider
    byte-for-byte -- every short-target lane stays unchanged."""
    original = ("Keep it spoken-length -- one breath, concrete, no "
                "nested clauses.")
    for tw in (0, 1, 12, 15, 24):
        assert LC._length_steering(tw) == original
    # non-numeric / missing target falls back to the original rider too
    assert LC._length_steering(None) == original
    assert LC._length_steering("") == original


def test_large_target_gets_concrete_develop_the_beat_steering():
    """Above the threshold the rider stops telling a 50-word beat to be one
    breath, and steers concretely toward the beat's own allocation."""
    out = LC._length_steering(50)
    assert "one breath" not in out.lower().replace(
        "more than one breath", "")  # only as the contrast clause
    assert "50 spoken words" in out
    assert "two or three spoken sentences" in out
    assert "complicate" in out
    # steering, not padding
    assert "padding" in out


def test_length_steering_threshold_boundary():
    original = ("Keep it spoken-length -- one breath, concrete, no "
                "nested clauses.")
    assert LC._length_steering(LC._ONE_BREATH_MAX_WORDS) == original
    assert LC._length_steering(LC._ONE_BREATH_MAX_WORDS + 1) != original


def test_length_steering_is_advisory_only_no_gate():
    """The steering is PROMPT TEXT only. It must not introduce any reject /
    reroll / trim surface: the composer's word bands are unchanged, and the
    rider is a plain string."""
    assert isinstance(LC._length_steering(50), str)
    # the WARN-only band constants are untouched
    from nodes.OTR_LedgerScriptWriter import (
        WORD_BUDGET_RATIO_HI, WORD_BUDGET_RATIO_LO,
    )
    assert WORD_BUDGET_RATIO_LO == 0.7
    assert WORD_BUDGET_RATIO_HI == 1.3


def _req(target_words: int, **kw) -> "LC.LineRequest":
    return LC.LineRequest(
        speaker="ELI CROSS", intent="reveal", mood="tense",
        target_words=target_words, canon_header="", last_lines=[], **kw
    )


def test_build_user_prompt_threads_the_scaled_rider():
    """The rider actually reaches the composed prompt, scaled by the beat's
    target_words (both the conflict_object branch and the default branch)."""
    small = LC._build_user_prompt(_req(15))
    assert "one breath, concrete, no nested clauses." in small
    assert "Word count target: 15." in small

    large = LC._build_user_prompt(_req(50))
    assert "50 spoken words" in large
    assert "two or three spoken sentences" in large
    assert "Word count target: 50." in large

    # conflict_object branch carries the same scaled rider
    large_conflict = LC._build_user_prompt(_req(50, conflict_object="the relay"))
    assert "the relay" in large_conflict
    assert "50 spoken words" in large_conflict
