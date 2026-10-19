"""tests/test_cast_voice_slots.py -- VC chunk 3.

meta.cast_voice_slots stamp (writer side) + CastLock's bank caster matching on
the stamped timbre/age_band (not just gender) + the EnsembleSlot.age_band field.

Hermetic: no GPU, no network. lock_cast runs through a canned generate_fn;
the _auto_registry read is verified by capturing the kwargs passed to the
deterministic caster.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for p in (_REPO_ROOT, _NODES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nodes import _otr_casting as _OTRC  # noqa: E402
from nodes import _otr_voice_bank as VB  # noqa: E402
from nodes.cast_lock import CastLock  # noqa: E402


def _canned_fn(n):
    """generate_fn that returns valid DescriptionResponse JSON n times."""
    seq = iter(range(n))

    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        i = next(seq)
        return json.dumps({
            "appearance": f"A weathered colonist number {i} who keeps the relay alive.",
            "speech_signature": "clipped and terse",
        })
    return fn


# --------------------------------------------------------------------------- #
# EnsembleSlot.age_band
# --------------------------------------------------------------------------- #
def test_ensemble_slot_has_age_band_default_adult():
    slot = _OTRC.EnsembleSlot(
        char_id="c02", name="ALICE", gender="female", timbre="warm", role="lead")
    assert slot.age_band == "adult"


# --------------------------------------------------------------------------- #
# lock_cast stamps meta.cast_voice_slots
# --------------------------------------------------------------------------- #
def test_lock_cast_stamps_cast_voice_slots():
    cast, meta = _OTRC.lock_cast(
        creative_fn=_canned_fn(8),
        num_characters=3,
        news_seed="A relay station loses contact with the colony.",
        style="closed_room_suspense",
        rng=random.Random(42),
        cast_seed=42,
        force_lemmy=False,
    )
    slots = meta.get("cast_voice_slots")
    assert isinstance(slots, dict) and slots

    # Every cast row has a slot keyed by char_id with the required shape.
    for row in cast:
        cid = row["char_id"]
        assert cid in slots, f"missing voice slot for {cid}"
        s = slots[cid]
        assert set(s) == {
            "gender", "timbre", "role", "age_band",
            "speech_signature",
        }
        assert isinstance(s["timbre"], list)  # list -> bank set-intersection
        assert s["gender"] == row.get("gender", "")

    # Open (non-announcer) character slots carry a real timbre + adult age_band.
    open_rows = [r for r in cast if r["name"] != "ANNOUNCER"]
    sample = slots[open_rows[0]["char_id"]]
    assert sample["timbre"] and sample["timbre"][0] in _OTRC._TIMBRE_VOCAB
    assert sample["age_band"] == "adult"
    assert sample["role"] in _OTRC._ROLE_VOCAB


def test_cast_voice_slots_carry_no_description_digest():
    """The digest was written and never read; it left with the split
    (2026-10-18). A slot holds voice-fit facts only."""
    cast, meta = _OTRC.lock_cast(
        creative_fn=_canned_fn(8), num_characters=2,
        news_seed="x", style="open", rng=random.Random(7), cast_seed=7,
        force_lemmy=False,
    )
    for row in cast:
        assert "description_digest" not in meta["cast_voice_slots"][row["char_id"]]


# --------------------------------------------------------------------------- #
# CastLock._auto_registry reads the slot's timbre/age_band
# --------------------------------------------------------------------------- #
def test_auto_registry_feeds_slot_timbre_and_age_to_caster(monkeypatch):
    captured = {}

    def fake_assign(**kw):
        captured.update(kw)
        return SimpleNamespace(
            voice_ref_id="vz_test", engine=kw["engine"], commercial_clean=True)

    # Patch on the package module cast_lock imports from (same object).
    monkeypatch.setattr(VB, "assign_voice_for_slot", fake_assign)

    led = {
        "meta": {
            "episode_seed": 99,
            "cast_voice_slots": {
                "c02": {"gender": "male", "timbre": ["gravelly"],
                        "age_band": "elder", "role": "lead"},
            },
        },
        "cast": [{"char_id": "c02", "name": "BOB", "gender": "male"}],
        "lines": [],
    }
    report: list = []
    CastLock()._auto_registry(
        led, led["cast"], "default", False, report)

    assert captured, "_auto_registry did not resolve a char engine / call the caster"
    assert captured.get("timbre") == ("gravelly",)
    assert captured.get("age_band") == "elder"
    # The stamped ref landed on the entry (I-4 stamp).
    assert led["cast"][0].get("voice_ref_id") == "vz_test"


def test_auto_registry_legacy_ledger_without_slots_still_casts(monkeypatch):
    """No cast_voice_slots (legacy ledger) -> entry-level fields used, no crash."""
    captured = {}

    def fake_assign(**kw):
        captured.update(kw)
        return SimpleNamespace(
            voice_ref_id="vz_legacy", engine=kw["engine"], commercial_clean=True)

    monkeypatch.setattr(VB, "assign_voice_for_slot", fake_assign)
    led = {
        "meta": {"episode_seed": 1},  # no cast_voice_slots
        "cast": [{"char_id": "c02", "name": "BOB", "gender": "male"}],
        "lines": [],
    }
    CastLock()._auto_registry(led, led["cast"], "default", False, [])
    assert captured.get("timbre") == ()      # empty entry-level fallback
    assert captured.get("age_band") == ""
