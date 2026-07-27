"""LANE 2 -- a measurement clip's receipt names what the cell DEPARTED from.

THE DEFECT. Since B6 every prequalification clip stamped one generic
``+prequalification``, so a winning sweep artifact could not prove WHICH knob
values produced it. The four cells of the 2026-07-27 ltx sweep are
indistinguishable in ``meta.render_engines`` -- the record that outlives the
run -- and the winner was selected from a table kept outside the ledger.

This file covers the shared FORMAT (``recipe_departures``). The per-adapter
wiring is covered next to each adapter's own freeze tests.

CPU-only: pure functions over plain data, no engine, no GPU, no environment.
"""

from __future__ import annotations

import re

import pytest

from nodes._otr_video_engines import recipe_departures as rd


# --------------------------------------------------------------------------- #
# the diff -- RESOLVED values, never env presence
# --------------------------------------------------------------------------- #
def test_only_values_that_actually_DIFFER_are_named():
    frozen = {"steps": 8, "cfg": 1.0, "sampler": "euler"}
    resolved = {"steps": 8, "cfg": 2.5, "sampler": "euler"}
    assert rd.departures(frozen, resolved) == {"cfg": 2.5}


def test_setting_a_knob_to_the_value_it_already_HAS_is_not_a_departure():
    """An operator who exports OTR_LTX_8GB_STEPS=8 when 8 is frozen has changed
    nothing. A receipt claiming a departure there would describe a cell that
    does not exist -- which is worse than the generic mark it replaces."""
    frozen = {"steps": 8, "tiled_vae": True}
    assert rd.departures(frozen, {"steps": 8, "tiled_vae": True}) == {}


def test_departures_are_SORTED_so_one_cell_always_stamps_one_string():
    frozen = {"a": 1, "b": 2, "c": 3}
    out = rd.departures(frozen, {"c": 9, "a": 7, "b": 8})
    assert list(out) == ["a", "b", "c"]


def test_a_key_the_frozen_recipe_never_had_is_REFUSED_BY_NAME():
    """It means the recipe and its resolver have drifted apart, and reporting
    it as a departure would describe a knob no version of the recipe had.

    MATCHED ON THE MESSAGE, not merely on KeyError. The mutation round caught
    this test passing with the named guard DELETED: the comprehension below it
    raises a bare KeyError on the same input anyway, so `pytest.raises(KeyError)`
    proved nothing about the guard it was written for. An incidental exception
    from one line down is not the refusal this test is named after."""
    with pytest.raises(KeyError, match="drifted"):
        rd.departures({"steps": 8}, {"steps": 8, "invented": 1})
    # ...and it names WHICH key, so the operator is not left bisecting.
    with pytest.raises(KeyError, match="invented"):
        rd.departures({"steps": 8}, {"steps": 8, "invented": 1})


def test_a_key_the_resolver_did_not_report_is_simply_not_considered():
    # The caller passes the knobs that ACTUALLY BOUND its render -- tile
    # geometry a tiled=off cell never reached must not be reported as a
    # departure describing a clip it had no effect on.
    frozen = {"steps": 8, "vae_tile": 512}
    assert rd.departures(frozen, {"steps": 12}) == {"steps": 12}


# --------------------------------------------------------------------------- #
# the rendered value -- short, safe, deterministic
# --------------------------------------------------------------------------- #
def test_booleans_read_on_and_off():
    # The vocabulary the consent-act knobs already use, and the shape the bug
    # report asked for by name: ..._v2+prequalification[tiled_vae=off].
    assert rd.render_value(True) == "on"
    assert rd.render_value(False) == "off"


def test_numbers_render_inline_and_round_trip_readably():
    assert rd.render_value(12) == "12"
    assert rd.render_value(2.25) == "2.25"
    assert rd.render_value(1.0) == "1.0"


def test_a_long_value_becomes_a_DIGEST_not_a_truncation():
    """A negative prompt is a real departure and it is prose. Truncating would
    look like a value and could collide; a digest is honest about being one.

    PINNED AS HEX. An earlier draft asserted only the ``#`` prefix, the length
    and distinctness -- all of which ``"#" + text[:8]`` satisfies, so a
    truncation wearing a costume passed the test named for refusing one. The
    hex shape is the property that tells the two apart."""
    long_a = "a paragraph of negative prompt " * 4
    long_b = "a different paragraph entirely " * 4
    ra, rb = rd.render_value(long_a), rd.render_value(long_b)
    assert re.fullmatch(r"#[0-9a-f]{8}", ra)
    assert ra != rb                                   # two cells stay distinct
    assert ra == rd.render_value(long_a)              # and one cell is stable


def test_two_long_values_sharing_a_PREFIX_still_differ():
    """The failure a truncation actually causes, stated directly. Two negative
    prompts that begin identically -- which real ones do, they all start with
    "low quality, worst quality, ..." -- would collapse to ONE receipt under a
    prefix-based rendering, re-creating the indistinguishable cells LANE 2
    exists to remove, on the very field that motivated the digest."""
    shared = "low quality, worst quality, blurry, distorted, watermark, "
    a, b = rd.render_value(shared + "and a tail"), rd.render_value(shared + "and another")
    assert re.fullmatch(r"#[0-9a-f]{8}", a)
    assert a != b


@pytest.mark.parametrize("length,digested", [(24, False), (25, True)])
def test_the_inline_boundary_is_pinned(length, digested):
    # A `<=` -> `<` slip would only misfire at exactly this boundary.
    value = "x" * length
    assert value.isascii() and rd._INLINE_SAFE.match(value)
    assert rd.render_value(value).startswith("#") is digested


def test_a_value_with_unsafe_punctuation_is_digested_not_escaped():
    # A comma or a bracket would make the suffix unreadable at a glance and
    # unparseable if anyone ever parses it. Escaping would invent a second
    # dialect nobody documented.
    for unsafe in ("has,comma", "has]bracket", "has=equals", "has space"):
        assert rd.render_value(unsafe).startswith("#")


def test_a_short_safe_string_stays_READABLE():
    assert rd.render_value("euler") == "euler"
    assert rd.render_value("cpu") == "cpu"


# --------------------------------------------------------------------------- #
# the suffix
# --------------------------------------------------------------------------- #
def test_a_cell_that_changed_NOTHING_still_marks_itself():
    """B6's contract is that a measurement run marks its own artifacts. Losing
    the mark when the departure list happens to be empty would put a sweep clip
    back into production's namespace on exactly the cell that looks most like
    production."""
    assert rd.format_suffix({}) == rd.PREQUALIFICATION_SUFFIX


def test_the_suffix_names_the_departures_in_the_documented_shape():
    assert rd.format_suffix({"tiled_vae": False}) \
        == "+prequalification[tiled_vae=off]"


def test_several_departures_are_comma_separated_in_key_order():
    out = rd.receipt_suffix(
        {"steps": 8, "cfg": 1.0, "t5_device": "cpu"},
        {"steps": 12, "cfg": 1.0, "t5_device": "default"})
    assert out == "+prequalification[steps=12,t5_device=default]"


def test_the_marked_suffix_still_STARTS_with_the_bare_mark():
    # Everything downstream that greps for a sweep artifact keys on this.
    assert rd.format_suffix({"steps": 12}).startswith(rd.PREQUALIFICATION_SUFFIX)


def test_receipt_suffix_is_departures_then_format_in_one_call():
    frozen = {"steps": 8, "tiled_vae": True}
    resolved = {"steps": 8, "tiled_vae": False}
    assert rd.receipt_suffix(frozen, resolved) \
        == rd.format_suffix(rd.departures(frozen, resolved))
