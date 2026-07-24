"""Original-radio authored cast-name preservation coverage."""
from __future__ import annotations

from nodes._otr_original_radio import (
    CastSketchEntry,
    ConceptPitches,
    SelectCastEntry,
    SelectedConcept,
)


def test_cast_models_preserve_titles_case_and_multiword_names():
    authored = "Dr. Eli Cross-Smythe"
    assert CastSketchEntry(name=authored, role="lead").name == authored
    assert SelectCastEntry(
        name=authored,
        want="protect the archive",
        pressure="the deadline",
        relationship="colleague",
    ).name == authored


def test_concept_and_selection_preserve_the_same_authored_names():
    names = ["DR. ELI CROSS", "Mrs. Martha Vane"]
    pitches = ConceptPitches.model_validate({
        "pitches": [{
            "logline": "A lab loses its own clock.",
            "hook": "The countdown answers back.",
            "radio_device": "a ticking relay",
            "cast_sketch": [
                {"name": names[0], "role": "lead"},
                {"name": names[1], "role": "foil"},
            ],
        }],
    })
    selected = SelectedConcept.model_validate({
        "selected_index": 0,
        "pitch": {
            "logline": "A lab loses its own clock.",
            "hook": "The countdown answers back.",
            "radio_device": "a ticking relay",
        },
        "cast": [
            {
                "name": name,
                "want": "act",
                "pressure": "time",
                "relationship": "colleague",
            }
            for name in names
        ],
        "selection_rationale": "it plays",
    })

    assert [row.name for row in pitches.pitches[0].cast_sketch] == names
    assert [row.name for row in selected.cast] == names
