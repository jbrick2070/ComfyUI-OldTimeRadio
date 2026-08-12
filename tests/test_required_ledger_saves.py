"""A required ledger save must REFUSE when it did not land.

`Ledger.save()` returns the path on success and **None on failure, and never
raises** (`production_ledger.py`). So a bare `led.save()` is a write that can
silently not happen -- and the ledger is the source of truth every downstream
node reads FROM DISK. A boundary that "saved" without saving hands the next node
stale or absent state, and the render goes wrong somewhere else entirely with
nothing pointing back at the real cause.

THIS IS THE THIRD FORM OF ONE DEFECT IN A SINGLE SESSION:

* PBUG-20260812-02 -- a value that could not serialize (a bound method),
* PBUG-20260812-04 -- a live pydantic model reaching the ledger,
* the fable2 deal receipt, which was added to make dead legs reproducible and
  then called `save()` unchecked, i.e. a receipt that could silently not happen.

Each time the write failed quietly and the consequence surfaced far away. The
review lane that found the remaining unchecked sites enumerated them by line;
these tests cover the shared helper and the boundaries it now guards.

REQUIRED vs DIAGNOSTIC is a real distinction and both are legitimate. A required
boundary is one whose output the NEXT stage reads -- it must fail closed. A
diagnostic checkpoint may warn and continue, but it has to say so out loud
rather than swallowing the result.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_scifi_fable2 as fable2


class _Ledger:
    """Minimal stand-in with the REAL contract: a path on success, None on
    failure, never raising."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def save(self):
        self.calls += 1
        return self._result


def test_a_successful_save_returns_quietly():
    led = _Ledger("C:/x/ep_ledger.json")
    fable2.require_ledger_save(led, "the thing")
    assert led.calls == 1


def test_a_FAILED_save_raises_instead_of_continuing():
    """The whole point. Previously this returned None and the pipeline carried
    on writing to a ledger that was not on disk."""
    led = _Ledger(None)
    with pytest.raises(fable2.Fable2Error) as caught:
        fable2.require_ledger_save(led, "the TTS delivery-text stamp")
    assert "did not persist" in str(caught.value)


def test_the_refusal_NAMES_what_was_being_saved():
    """A bare 'save failed' sends the reader hunting. The boundary name is the
    difference between one look and an hour."""
    led = _Ledger(None)
    with pytest.raises(fable2.Fable2Error) as caught:
        fable2.require_ledger_save(led, "the fable2 pass receipts")
    assert "the fable2 pass receipts" in str(caught.value)


def test_the_refusal_points_at_the_warning_that_carries_the_CAUSE():
    """`save_ledger_safe` already logs a WARNING naming the type and the dotted
    location of an unserializable value. The refusal should send the reader
    there rather than restating it."""
    led = _Ledger(None)
    with pytest.raises(fable2.Fable2Error) as caught:
        fable2.require_ledger_save(led, "x")
    message = str(caught.value)
    assert "OTR_Ledger" in message and "warning" in message.lower()


def test_it_explains_WHY_it_refuses_rather_than_just_that_it_did():
    """The next reader needs to know that continuing is worse than stopping."""
    led = _Ledger(None)
    with pytest.raises(fable2.Fable2Error) as caught:
        fable2.require_ledger_save(led, "x")
    assert "downstream" in str(caught.value)
    assert "disk" in str(caught.value)


# ---------------------------------------------------------------------------
# The boundaries it now guards
# ---------------------------------------------------------------------------
def test_the_TTS_DELIVERY_STAMP_boundary_is_checked():
    """The voice nodes read this stamp. If it did not persist they speak stale
    text -- and the stamp exists precisely so they do not."""
    src = inspect.getsource(fable2)
    idx = src.find("stamp_text_for_tts_delivery(led)")
    assert idx > 0, "test is stale: the delivery stamp moved"
    following = src[idx:idx + 400]
    assert "require_ledger_save" in following
    assert "led.save()" not in following.split("require_ledger_save")[0]


def test_the_PASS_RECEIPTS_boundary_is_checked():
    src = inspect.getsource(fable2)
    idx = src.find('f2["pass_receipts"] = receipts')
    assert idx > 0, "test is stale: the receipts stamp moved"
    following = src[idx:idx + 400]
    assert "require_ledger_save" in following


def test_the_DEAL_RECEIPT_stays_a_warning_not_a_refusal():
    """Deliberately NOT required. The episode is still renderable without the
    deal receipt, and killing a render over a diagnostic would be a worse trade
    than losing the diagnostic -- but it must WARN, never swallow."""
    src = inspect.getsource(fable2)
    idx = src.find("[scifi_fable2] deal:")
    assert idx > 0
    following = src[idx:idx + 900]
    assert "led.save() is None" in following
    assert "log.warning" in following
    assert "require_ledger_save" not in following


def test_the_helper_is_documented_as_REQUIRED_only():
    """A helper this blunt must say where it belongs, or it gets used on
    diagnostic checkpoints and turns a lost log line into a dead episode."""
    doc = inspect.getdoc(fable2.require_ledger_save) or ""
    assert "REQUIRED" in doc
    assert "Diagnostic" in doc or "diagnostic" in doc
