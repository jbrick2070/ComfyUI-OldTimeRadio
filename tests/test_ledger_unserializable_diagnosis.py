"""PBUG-20260812-04 -- a live pydantic model reached the ledger, and the error
named the type but not the place.

FOUND BY A LIVE RUN: the `public_domain` leg of the cross-bank writer gate
(2026-08-12) died 123.66 s in, in `OTR_LedgerScriptWriter`:

    [Ledger] save failed: Object of type VisualStyleCardModel is not JSON
    serializable
    RuntimeError: failed to save ledger after visual_style pack embedding

THE DEFECT ITSELF is fixed at its source -- `run_story_brief_reflection` returns
`visual_card` as a live `VisualStyleCardModel` (`_otr_story_brief.py:643`), and
`OTR_LedgerScriptWriter` merged that whole delta into `meta` with
`meta.update()`. The serialized copy written moments later as
`meta["visual_style_card"]` did not remove the raw model sitting beside it, and
`meta` IS the ledger. The model is now popped before the merge.

THESE TESTS COVER THE OTHER HALF: the error message. `json.dumps` names only the
offending TYPE, and the ledger carries 600+ keys, so the message starts a hunt
instead of ending one. That cost real diagnosis time TWICE in one day --
PBUG-20260812-02 (a bound method reaching the writer's prompt) and this. Both
were one `dict.update` from recurring, so the CLASS is worth naming.

`_where_unserializable` runs only on the failure path and must never raise: a
diagnostic that throws inside an error handler replaces a reported failure with
a confusing one.
"""
from __future__ import annotations

import json

import pytest

from nodes import _otr_ledger


class NotJson:
    """Stands in for `VisualStyleCardModel` -- any object json cannot encode."""


def type_error():
    """A REAL `json.dumps` TypeError, not a hand-written one, so the message
    text this function keys on is the interpreter's own."""
    try:
        json.dumps({"x": NotJson()})
    except TypeError as exc:
        return exc
    raise AssertionError("json.dumps accepted an unencodable object")


def where(ledger):
    return _otr_ledger._where_unserializable(ledger, type_error())


# ---------------------------------------------------------------------------
# It names the PLACE
# ---------------------------------------------------------------------------
def test_it_names_the_dotted_path_and_the_type():
    """The live shape: the offender sat at `meta.visual_card`."""
    got = where({"meta": {"visual_style": "dynamic", "visual_card": NotJson()}})
    assert "meta.visual_card" in got
    assert "NotJson" in got


def test_it_finds_an_offender_nested_in_a_list():
    got = where({"cast": [{"name": "Ada"}, {"name": "Bo", "voice": NotJson()}]})
    assert "cast[1].voice" in got


def test_it_finds_an_offender_the_HEALTHY_keys_are_hiding():
    """A realistic ledger is mostly fine -- the point is finding the one key
    that is not, without reading 600 of them."""
    ledger = {"meta": {"a": 1, "b": "two", "c": [1, 2, 3], "d": {"e": True}},
              "lines": [{"speaker": "Ada", "text": "hi"}] * 50,
              "images": {"cache_index": {"deadbeef": "x.png"}},
              "beats": [{"id": i} for i in range(40)]}
    assert where(ledger) == "", "a clean ledger must report nothing"
    ledger["meta"]["visual_card"] = NotJson()
    assert "meta.visual_card" in where(ledger)


def test_an_unencodable_KEY_is_reported_too():
    """`json.dumps` rejects a non-primitive key as well, and that reads as the
    same opaque message."""
    got = where({"meta": {NotJson(): "value"}})
    assert "meta" in got and "NotJson" in got


# ---------------------------------------------------------------------------
# It must never lie, and never raise
# ---------------------------------------------------------------------------
def test_a_clean_ledger_reports_nothing():
    assert where({"meta": {"ok": True}, "lines": []}) == ""


@pytest.mark.parametrize("exc", [
    OSError("disk full"),
    PermissionError("locked"),
    ValueError("something else"),
    TypeError("a different TypeError entirely"),
])
def test_a_NON_serialization_failure_gets_no_path(exc):
    """Disk, permissions and rename failures must not be dressed up as a
    serialization problem -- that would send the reader to the wrong file."""
    assert _otr_ledger._where_unserializable({"meta": {"x": NotJson()}}, exc) == ""


@pytest.mark.parametrize("ledger", [
    None, "a string", 42, [], {},
    {"self": None},
])
def test_it_survives_any_ledger_shape(ledger):
    assert isinstance(_otr_ledger._where_unserializable(ledger, type_error()), str)


def test_a_RECURSIVE_ledger_still_gets_its_diagnosis():
    """A self-referencing container must not cost the answer.

    Without cycle detection the walker recurses until `RecursionError`, which
    the outer handler catches -- so nothing hangs, but the path is lost exactly
    when the structure is at its most confusing. The cycle is visited once and
    the real offender beyond it is still found.
    """
    ledger = {"meta": {}}
    ledger["meta"]["self"] = ledger["meta"]     # the cycle comes FIRST
    ledger["meta"]["bad"] = NotJson()
    assert "meta.bad" in _otr_ledger._where_unserializable(ledger, type_error())


def test_a_SHARED_subtree_is_not_mistaken_for_a_cycle_before_it_is_searched():
    """Visiting by id must not skip a subtree that legitimately appears twice
    BEFORE the offender is reached in it."""
    shared = {"ok": 1}
    ledger = {"a": shared, "b": shared, "c": {"bad": NotJson()}}
    assert "c.bad" in _otr_ledger._where_unserializable(ledger, type_error())


def test_it_never_raises_even_on_a_hostile_object():
    """An object whose `__class__`/repr misbehaves must not take down the
    handler."""
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    got = _otr_ledger._where_unserializable({"meta": {"x": Hostile()}}, type_error())
    assert isinstance(got, str)


# ---------------------------------------------------------------------------
# It is wired into the real save path
# ---------------------------------------------------------------------------
def test_save_ledger_safe_returns_False_and_LOGS_the_path(tmp_path, caplog):
    """End to end: the real save path must fail closed AND say where."""
    target = tmp_path / "ep_ledger.json"
    ledger = {"meta": {"visual_card": NotJson()}, "episode_id": "ep"}
    with caplog.at_level("WARNING"):
        ok = _otr_ledger.save_ledger_safe(target, ledger)
    assert ok is False, "a ledger that cannot serialize must not report success"
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "not JSON serializable" in logged
    assert "meta.visual_card" in logged, (
        "the warning names the type but not the place -- the hunt this was "
        "written to end")
    assert not target.exists(), "a failed save must leave no partial ledger"
