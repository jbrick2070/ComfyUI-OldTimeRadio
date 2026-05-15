"""S32 B6 -- writer meta extends with per-helper / per-phase slot tracking.

Two new meta fields added by the writer:
* `meta["slot_calls_by_helper"]`: dict mapping helper-name ->
  per-slot call counts. Example:
      {"pick_style": {"creative": 1, "technical": 1},
       "lock_cast":  {"creative": 1, "technical": 0}}
* `meta["slot_transitions_by_phase"]`: ordered list of
  {phase, from_slot, to_slot, from_id, to_id} records, one per
  slot transition fired during the run.

Implementation lives on `_SlotScheduler` (the writer's existing
per-slot accountant). New context-manager method
`helper_context(name)` is called by the writer around each
helper invocation; calls made within the block get attributed
to the named helper.

Tests:
* `test_meta_slot_calls_by_helper_shape` -- exercise the
  scheduler directly with paired generators and `helper_context`
  blocks; assert the dict shape is correct.
* `test_meta_slot_transitions_by_phase_differing_slots` --
  differing-slots config: transition records carry the right
  phase labels and slot directions.
* `test_default_config_zero_transitions` -- same-model config:
  transitions == 0 and the by-phase list is empty.

The runtime byte-comparison for audio at B6 is captured in the
existing `tests/test_audio_byte_identical.py` baselines (default
+ differing-slots).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_scheduler(creative_id: str, technical_id: str):
    """Build a _SlotScheduler with a stub request_slot path so tests
    don't actually hit transformers. Replaces the loader's
    `request_slot` with a fake that returns an empty dict."""
    from nodes.OTR_LedgerScriptWriter import _SlotScheduler

    sched = _SlotScheduler(
        creative_id=creative_id,
        technical_id=technical_id,
        top_p=0.92,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    return sched


def _stub_request_slot(monkeypatch):
    """Replace `_otr_model_loader.request_slot` so
    `_SlotScheduler._account_and_get_entry` returns a stub
    cache_entry without loading any real model."""
    from nodes import _otr_model_loader as _OTRML

    monkeypatch.setattr(
        _OTRML, "request_slot",
        lambda slot, model_id: {"model": None, "tokenizer": None},
    )


def test_meta_slot_calls_by_helper_shape(monkeypatch):
    """Exercise the scheduler with `helper_context` blocks; assert
    `slot_calls_by_helper` carries the right per-helper dicts.
    """
    _stub_request_slot(monkeypatch)
    sched = _make_scheduler("creative/model", "technical/model")

    with sched.helper_context("pick_style"):
        sched._account_and_get_entry("creative")
        sched._account_and_get_entry("technical")

    with sched.helper_context("lock_cast"):
        sched._account_and_get_entry("creative")
        sched._account_and_get_entry("creative")

    with sched.helper_context("build_news_briefs"):
        sched._account_and_get_entry("technical")

    expected = {
        "pick_style":        {"creative": 1, "technical": 1},
        "lock_cast":         {"creative": 2, "technical": 0},
        "build_news_briefs": {"creative": 0, "technical": 1},
    }
    assert sched.slot_calls_by_helper == expected, (
        f"slot_calls_by_helper shape drifted. Expected:\n  {expected}\n"
        f"Got:\n  {sched.slot_calls_by_helper}"
    )


def test_meta_slot_transitions_by_phase_differing_slots(monkeypatch):
    """Differing-slots config: each cross-slot call produces a
    transition record with the right phase label + slot direction.
    """
    _stub_request_slot(monkeypatch)
    sched = _make_scheduler("creative/model", "technical/model")

    # Simulate: pick_style (creative + technical), lock_cast (creative).
    with sched.helper_context("pick_style"):
        sched._account_and_get_entry("creative")
        sched._account_and_get_entry("technical")  # transition 1: C->T

    with sched.helper_context("lock_cast"):
        sched._account_and_get_entry("creative")  # transition 2: T->C

    # Two cross-slot transitions captured.
    assert len(sched.slot_transitions_by_phase) == 2, (
        f"Expected 2 transitions; got "
        f"{len(sched.slot_transitions_by_phase)}: "
        f"{sched.slot_transitions_by_phase}"
    )
    t1 = sched.slot_transitions_by_phase[0]
    t2 = sched.slot_transitions_by_phase[1]

    # Transition 1: pick_style C -> T
    assert t1["phase"] == "pick_style"
    assert t1["from_slot"] == "creative"
    assert t1["to_slot"] == "technical"
    assert t1["from_id"] == "creative/model"
    assert t1["to_id"] == "technical/model"

    # Transition 2: lock_cast T -> C
    assert t2["phase"] == "lock_cast"
    assert t2["from_slot"] == "technical"
    assert t2["to_slot"] == "creative"


def test_default_config_zero_transitions(monkeypatch):
    """Default config (creative == technical): no transitions; the
    by-phase list stays empty. Per-helper call counts still populate.
    """
    _stub_request_slot(monkeypatch)
    sched = _make_scheduler("same/model", "same/model")

    with sched.helper_context("pick_style"):
        sched._account_and_get_entry("creative")
        sched._account_and_get_entry("technical")

    with sched.helper_context("lock_cast"):
        sched._account_and_get_entry("creative")

    assert sched.transitions == 0, (
        f"Default-config should produce 0 transitions; got "
        f"{sched.transitions}"
    )
    assert sched.slot_transitions_by_phase == [], (
        f"Default-config slot_transitions_by_phase must be empty; "
        f"got {sched.slot_transitions_by_phase}"
    )
    # Per-helper accounting still populates even when slot ids match.
    assert sched.slot_calls_by_helper["pick_style"] == {
        "creative": 1, "technical": 1,
    }
    assert sched.slot_calls_by_helper["lock_cast"] == {
        "creative": 1, "technical": 0,
    }
