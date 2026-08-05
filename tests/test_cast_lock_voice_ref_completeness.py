"""Every non-announcer cast row names the voice that actually renders.

The hole this closes: `assign_voice_for_slot` raises for a gender the bank cannot
serve -- `other` is 20% of every roll and the bank carries zero rows for it --
and CastLock used to report the row "NOT cast" and move on, leaving no
`voice_ref_id`. The render path then drew a gender-agnostic reference of its own,
so the ledger did not name the voice that spoke. Downstream consumers read
FIELDS, not intentions.

Both readers of that draw now call one shared helper, which is the point: a
second copy in the caster would let the ledger name one voice while the render
opened another.
"""
from __future__ import annotations

import json

import pytest

from nodes._otr_voice_bank import (
    VoiceCastingError, assign_voice_for_slot, gender_agnostic_fallback_ref,
    load_voice_bank,
)


def _ledger(cast, meta=None):
    return json.dumps(
        {"meta": meta or {"episode_seed": 42}, "cast": cast, "lines": []})


_UNSERVABLE_CAST = [
    {"char_id": "c1", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c2", "name": "RIVER", "gender": "other",
     "voice_preset": "v2/en_speaker_9"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
     "voice_preset": "v2/en_speaker_6"},
]


# ------------------------------------------------------- the premise is real
def test_the_bank_cannot_serve_gender_other():
    entries, _ = load_voice_bank()
    assert not [e for e in entries if e.gender == "other"], (
        "if the bank ever gains an 'other' row this whole path changes")
    with pytest.raises(VoiceCastingError):
        assign_voice_for_slot(
            role="char_voice", engine="indextts2", char_id="c2",
            gender="other", timbre=(), age_band="adult", episode_seed=42)


# ---------------------------------------------------------- the shared draw
def test_the_fallback_helper_is_deterministic_and_role_preferring():
    entries, _ = load_voice_bank()
    a = gender_agnostic_fallback_ref(
        entries, engine="indextts2", char_id="c2", episode_seed=42)
    b = gender_agnostic_fallback_ref(
        entries, engine="indextts2", char_id="c2", episode_seed=42)
    assert a is not None and a.voice_ref_id == b.voice_ref_id
    assert a.engine == "indextts2"
    assert "char_voice" in a.roles

    other = gender_agnostic_fallback_ref(
        entries, engine="indextts2", char_id="c9", episode_seed=42)
    assert other is not None
    assert gender_agnostic_fallback_ref(
        entries, engine="no_such_engine", char_id="c2", episode_seed=42) is None


def test_castlock_stamps_the_same_reference_the_render_path_would_open():
    """The whole reason the helper is shared. If these two ever disagree, the
    ledger is lying about which voice spoke.
    """
    from nodes.cast_lock import CastLock
    from nodes import _otr_voice_node_common as vnc

    out = CastLock().lock(
        script_json=_ledger(_UNSERVABLE_CAST),
        cast_voice_policy="auto_registry")
    row = {e["char_id"]: e for e in json.loads(out[0])["cast"]}["c2"]

    entries, _ = load_voice_bank()
    expected = gender_agnostic_fallback_ref(
        entries, engine=row["voice_engine"], char_id="c2", episode_seed=42)
    assert row["voice_ref_id"] == expected.voice_ref_id

    # And the render-time resolver opens that same id's file.
    path = vnc._resolve_clone_ref_path(row["voice_engine"], row, 42)
    assert path, "the stamped reference must resolve to a real file on disk"


# ------------------------------------------------------- ledger completeness
def test_every_non_announcer_row_carries_a_voice_ref_id():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_UNSERVABLE_CAST),
        cast_voice_policy="auto_registry")
    cast = {e["char_id"]: e for e in json.loads(out[0])["cast"]}
    for cid in ("c1", "c2"):
        assert cast[cid].get("voice_ref_id"), cid


def test_voice_cast_fallback_is_defined_on_every_row_it_considers():
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_UNSERVABLE_CAST),
        cast_voice_policy="auto_registry")
    cast = {e["char_id"]: e for e in json.loads(out[0])["cast"]}
    assert cast["c1"]["voice_cast_fallback"] == ""
    assert cast["c2"]["voice_cast_fallback"] == "gender_unservable"


def test_the_unservable_row_is_not_refused_or_gender_restricted():
    """This is a LEDGER fix, not a content gate. The row still renders, and the
    roll that produced 'other' is untouched.
    """
    from nodes.cast_lock import CastLock

    out = CastLock().lock(
        script_json=_ledger(_UNSERVABLE_CAST),
        cast_voice_policy="auto_registry")
    cast = {e["char_id"]: e for e in json.loads(out[0])["cast"]}
    assert cast["c2"]["gender"] == "other", "the gender must NOT be rewritten"
    assert cast["c2"]["voice_ref_id"]


# ------------------------------------------- the direct path stays reachable
def test_the_render_fallback_still_works_for_a_row_castlock_never_stamped():
    """preserve_ledger rows and pre-2026-08-05 ledgers carry no voice_ref_id, so
    the resolver's own gender-agnostic branch must stay live.

    Constructed with voice_ref_id ABSENT on purpose: once CastLock stamps, the
    vrid lookup wins first and this branch is skipped, so a test built from a
    freshly-cast row would be exercising dead code and proving nothing.
    """
    from nodes import _otr_voice_node_common as vnc

    unstamped = {"char_id": "c2", "name": "RIVER", "gender": "other"}
    assert "voice_ref_id" not in unstamped
    path = vnc._resolve_clone_ref_path("indextts2", unstamped, 42)
    assert path, "a cloning engine must never be left without a reference"

    entries, _ = load_voice_bank()
    expected = gender_agnostic_fallback_ref(
        entries, engine="indextts2", char_id="c2", episode_seed=42)
    assert expected.voice_ref_id in path or path.endswith(
        expected.ref_path.replace("/", "\\").split("\\")[-1])
