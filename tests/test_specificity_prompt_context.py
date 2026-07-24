"""Positive coverage for optional source-term prompt context."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_specificity import (  # noqa: E402
    MAX_ANCHORS,
    derive_specificity_anchors,
    inject_anchors_into_header,
    sanitize_for_prompt,
)


def test_optional_source_terms_preserve_authored_vocabulary():
    terms = [
        "WALLOPS ISLAND",
        "NASA",
        "transgender people",
        "smoking-room ledger",
        "three robotic arms",
    ]
    assert derive_specificity_anchors(terms) == terms


def test_optional_source_terms_are_stably_deduped_and_prompt_bounded():
    terms = ["NASA", "nasa", "one", "two", "three", "four", "five", "six"]
    assert derive_specificity_anchors(terms) == [
        "NASA", "one", "two", "three", "four",
    ]
    assert len(derive_specificity_anchors(terms)) == MAX_ANCHORS


@pytest.mark.parametrize("value", [None, [], ["", "   "]])
def test_empty_optional_source_terms_remain_empty(value):
    assert derive_specificity_anchors(value) == []


def test_source_terms_are_added_as_prompt_context_only():
    header = "EPISODE CONTEXT"
    out = inject_anchors_into_header(
        header, ["transgender people", "a velvet chair", "smoking room"],
    )
    assert out == (
        "EPISODE CONTEXT\n\n"
        "Optional source terms (use only when they fit naturally; do not force "
        "them into every line):\n"
        "- transgender people\n"
        "- a velvet chair\n"
        "- smoking room"
    )
    assert inject_anchors_into_header(header, []) == header


def test_prompt_transport_normalizes_controls_without_word_filtering():
    assert sanitize_for_prompt("an experimental\nquantum\tcomputer") == (
        "an experimental quantum computer"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
