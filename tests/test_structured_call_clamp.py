"""Structured-call resilience: clamp an over-long capped field instead of
aborting the whole episode (2026-06-18).

A weak/verbose model overflowed the outline's `time_of_day` (max 40 chars) with
a whole sentence -> the entire render aborted. _parse_and_validate now clamps a
`string_too_long` field to its declared max_length and re-validates -- but ONLY
on an output that would otherwise fail, and ONLY for the over-long string fields;
every other validation error still propagates, and a good output is untouched.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from nodes import _otr_structured_call as sc


class _Shape(BaseModel):
    time_of_day: str = Field(max_length=40)
    beats: int


def test_overlong_string_is_clamped_not_aborted():
    raw = ('{"time_of_day": "Midnight to 8 AM - the deadly hours when heat '
           'illnesses spike across the city", "beats": 3}')
    inst = sc._parse_and_validate(raw, _Shape)
    assert len(inst.time_of_day) <= 40
    assert inst.time_of_day.startswith("Midnight to 8 AM")
    assert inst.beats == 3


def test_clamp_trims_at_word_boundary():
    # 43 chars of words -> clamp to <=40 at a word boundary (no mid-word chop).
    raw = '{"time_of_day": "alpha bravo charlie delta echo foxtrot", "beats": 1}'
    inst = sc._parse_and_validate(raw, _Shape)
    assert len(inst.time_of_day) <= 40
    assert not inst.time_of_day.endswith(" ")
    # the clamp cut at a space, so the surviving text is whole words
    assert all(w in "alpha bravo charlie delta echo foxtrot" for w in inst.time_of_day.split())


def test_within_limit_value_is_untouched():
    raw = '{"time_of_day": "Night", "beats": 2}'
    inst = sc._parse_and_validate(raw, _Shape)
    assert inst.time_of_day == "Night"  # byte-identical, no coercion


def test_other_validation_errors_still_raise():
    # `beats` is the wrong type and there is no over-long string to clamp ->
    # the original ValidationError must propagate (never masked).
    raw = '{"time_of_day": "Dawn", "beats": "not-an-int"}'
    with pytest.raises(ValidationError):
        sc._parse_and_validate(raw, _Shape)


def test_overlong_plus_other_error_surfaces_the_other_error():
    # Clamp the string, but the bad `beats` must still surface after re-validate.
    raw = ('{"time_of_day": "' + "x" * 80 + '", "beats": "nope"}')
    with pytest.raises(ValidationError):
        sc._parse_and_validate(raw, _Shape)


class _NestedItem(BaseModel):
    hook: str = Field(max_length=20)


class _NestedShape(BaseModel):
    pitches: list[_NestedItem]


def test_nested_list_field_is_clamped_not_aborted():
    # scifi_fable2 S1b live smoke (2026-07-10): pitches.0.hook overflowed
    # and the old top-level-only clamp skipped it -> whole episode died.
    raw = ('{"pitches": [{"hook": "this hook is far far far too long to '
           'survive the cap"}]}')
    inst = sc._parse_and_validate(raw, _NestedShape)
    assert len(inst.pitches[0].hook) <= 20


def test_nested_clamp_reports_the_full_path():
    bad = {"pitches": [{"hook": "x" * 60}]}
    try:
        _NestedShape.model_validate(bad)
    except ValidationError as ve:
        repaired, clamped = sc._clamp_overlong_strings(bad, ve)
        assert clamped == ["pitches.0.hook"]
        assert len(repaired["pitches"][0]["hook"]) <= 20
        # the ORIGINAL dict is never mutated (deep copy)
        assert len(bad["pitches"][0]["hook"]) == 60
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")


def test_helper_returns_none_when_nothing_clampable():
    try:
        _Shape.model_validate({"time_of_day": "ok", "beats": "bad"})
    except ValidationError as ve:
        assert sc._clamp_overlong_strings({"time_of_day": "ok", "beats": "bad"}, ve) is None
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")


def test_helper_clamps_and_reports():
    long_v = "y" * 100
    try:
        _Shape.model_validate({"time_of_day": long_v, "beats": 1})
    except ValidationError as ve:
        repaired, clamped = sc._clamp_overlong_strings({"time_of_day": long_v, "beats": 1}, ve)
        assert clamped == ["time_of_day"]
        assert len(repaired["time_of_day"]) <= 40
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")
