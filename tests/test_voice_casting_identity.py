# -*- coding: utf-8 -*-
"""What must still be true after the recurring-character voice rewrite.

WHY THIS FILE EXISTS. A bespoke qualified/provisional/unrouted voice-ROUTE
subsystem is being replaced by a plain table of recurring-character catalogue
assignments. Most of what that subsystem tests dies with it -- the receipt
validators, the degradation reason codes, the audition manifests. But a set of
properties underneath it are not about routes at all, and a cutover that quietly
dropped them would look green the whole way: the clear-then-stamp atomicity, the
dispatch identity a cast row resolves to, the in-graph cache identity, and the
reservation that keeps one character's own recording out of everybody else's
draw.

Those are captured HERE, against the CURRENT code, BEFORE anything is deleted.
That ordering is the point. Characterization written after a cutover describes
the new behaviour and proves nothing about what was lost.

THE ONE THING THAT CHANGES WHEN THE CUTOVER LANDS is `_lock_recurring_row`
below. Every test in this file reaches CastLock through that single seam, so
retargeting the file is one function rewrite rather than eighteen. Nothing else
here names a tier, a route, a receipt or a policy.

A NOTE ON ABSENCE. Where a test asserts a field is not present it uses
`not in`, never a falsy check. A field written as "" or {} is a different
outcome from a field never written, and collapsing the two is how a probe
destroys the finding it was built to make.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from nodes import _otr_voice_route as ROUTE
from nodes._otr_voice_bank import load_voice_bank
from nodes.cast_lock import CastLock

# Real bank rows, so every bank lookup below has something true to find.
CLONE_REF = "cb_peter_yearsley"          # chatterbox, a mirrored local wav
KOKORO_REF = "bm_george"                 # kokoro, a `.pt` voice tensor
PROVIDER_REF = "el_daniel"               # elevenlabs, a provider voice id
PROVIDER_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"

RECURRING = "LEMMY"                      # the only recurring character today

CAST = [
    {"char_id": "c01", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c02", "name": RECURRING, "gender": "male",
     "voice_preset": "v2/en_speaker_8"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
     "voice_preset": "v2/en_speaker_6"},
]


def _ledger(cast=None, meta=None):
    return json.dumps({
        "meta": meta or {"episode_seed": 42},
        "cast": json.loads(json.dumps(cast if cast is not None else CAST)),
        "lines": [],
    })


def _rows(ledger_json):
    return {e["char_id"]: e for e in json.loads(ledger_json)["cast"]}


# ---------------------------------------------------------------------------
# THE SEAM. Everything tier-shaped in this file is here and nowhere else.
#
# Today the only way to get CastLock to stamp a recurring character's assigned
# voice is through the provisional-route policy, so that is what this builds.
# When the catalogue table lands, rewrite THIS function to install the table and
# delete `_assignment_record`; every test below should then pass unchanged. If
# one does not, the cutover changed a behaviour this file says must survive, and
# that is the signal it exists to give.
# ---------------------------------------------------------------------------

def _assignment_record(engine, voice_ref_id, identity_kind):
    """A record that makes CastLock assign `voice_ref_id` on `engine`."""
    state = ("configured_unrendered" if identity_kind == "provider_voice"
             else "rendered_pending_listen")
    receipt = {
        "engine": engine,
        "identity_kind": identity_kind,
        "identity_id": voice_ref_id,
        "state": state,
    }
    if state == "rendered_pending_listen":
        receipt.update({
            "audition_manifest_path": "otr/episodes/lemmy_cross_engine/MANIFEST.json",
            "audition_manifest_sha256": "a" * 64,
            "neutral_clip_path": "otr/episodes/lemmy_cross_engine/%s_neutral.wav" % engine,
            "neutral_clip_sha256": "b" * 64,
            "emotional_clip_path": "otr/episodes/lemmy_cross_engine/%s_emotional.wav" % engine,
            "emotional_clip_sha256": "c" * 64,
            "rendered_utc": "2026-08-16T21:00:00Z",
        })
    if identity_kind == "provider_voice":
        receipt.setdefault("provider", engine)
        receipt.setdefault("provider_voice_id", PROVIDER_VOICE_ID)
    return {
        "route_id": "lemmy-%s-provisional-v1" % engine,
        "route_contract_version": 1,
        "engine": engine,
        "voice_ref_id": voice_ref_id,
        "provisional_receipt": receipt,
    }


@pytest.fixture()
def assign(monkeypatch):
    """Install an assignment for the recurring character on one engine."""
    def install(engine, voice_ref_id, identity_kind):
        monkeypatch.setattr(
            "nodes.cast_lock._lemmy_voice_policy",
            lambda: {
                "policy_version": "recurring-character-characterization",
                "character_key": "lemmy",
                "approved_native_routes": {},
                "provisional_native_routes": {
                    engine: _assignment_record(engine, voice_ref_id, identity_kind),
                },
            },
        )
    return install


def _lock_recurring_row(assign, engine, voice_ref_id, identity_kind,
                        *, cast=None, policy="auto_registry",
                        voice_bank="default_clean"):
    """Lock a cast and hand back the recurring character's stamped row."""
    assign(engine, voice_ref_id, identity_kind)
    out = CastLock().lock(
        script_json=_ledger(cast=cast), voice_bank=voice_bank,
        char_voice_engine=engine, cast_voice_policy=policy)[0]
    return _rows(out)


@pytest.fixture(scope="module")
def bank():
    return load_voice_bank()[0]


# ---------------------------------------------------------------------------
# 1. CLEAR-THEN-STAMP IS ATOMIC.
#
# A row re-cast onto a new engine must not keep one field from the old one. A
# leftover provider id or reference path is the defect shape here: it renders,
# it renders with the WRONG voice, and nothing reports it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["auto_registry", "preserve_ledger"])
@pytest.mark.parametrize("stale_field", [
    "provider_voice_id", "voice_ref_path", "ref_path",
])
def test_a_stale_engine_identity_is_cleared_on_a_re_stamp(assign, policy,
                                                          stale_field):
    """Whatever the previous engine wrote is gone after a successful re-stamp."""
    cast = json.loads(json.dumps(CAST))
    cast[1][stale_field] = "STALE::from-a-previous-engine"

    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav",
                               cast=cast, policy=policy)
    row = rows["c02"]
    assert row.get(stale_field) != "STALE::from-a-previous-engine", (
        "%s survived the re-stamp in %s mode; the row now carries one field "
        "from the engine it used to be cast on" % (stale_field, policy))


@pytest.mark.parametrize("policy", ["auto_registry", "preserve_ledger"])
def test_a_stale_route_field_is_cleared_on_a_re_stamp(assign, policy):
    """A leftover `voice_route` from a prior lock does not survive.

    Kept after the cutover for a reason that outlives the route concept: the
    voice node RAISES on a non-empty `voice_route` whose status is not
    qualified, so a stale one is a dead render on every line, not a cosmetic
    field.
    """
    cast = json.loads(json.dumps(CAST))
    cast[1]["voice_route"] = {"status": "qualified", "route_id": "stale-v0"}

    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav",
                               cast=cast, policy=policy)
    assert not rows["c02"].get("voice_route"), (
        "a stale voice_route survived a %s re-stamp" % policy)


def test_a_row_the_caster_never_reaches_is_left_exactly_as_it_arrived(assign):
    """Byte-for-byte, every field. Not "mostly unchanged"."""
    cast = json.loads(json.dumps(CAST))
    cast[1]["voice_engine"] = "bark"
    cast[1]["voice_preset"] = "v2/en_speaker_8"
    before = json.loads(json.dumps(cast[0]))       # MONTY, never claimed

    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav",
                               cast=cast, policy="preserve_ledger")
    after = rows["c01"]
    for key, value in before.items():
        assert after.get(key) == value, (
            "field %r on an untouched row changed from %r to %r"
            % (key, value, after.get(key)))


def test_only_the_recurring_row_takes_the_assignment(assign):
    """The other rows keep their own draw and gain none of its identity."""
    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav")
    assert rows["c02"].get("voice_ref_id") == CLONE_REF
    for other in ("c01", "a1"):
        assert rows[other].get("voice_ref_id") != CLONE_REF, (
            "%s received the recurring character's assigned voice" % other)


# ---------------------------------------------------------------------------
# 2. THE WRITER'S PRESET IS CLEARED WHEN A NON-BARK ENGINE TAKES THE ROW.
#
# The preset is a Bark concept. Leaving it set while another engine renders is
# how a row ends up describing two different voices.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,ref,kind", [
    ("chatterbox", CLONE_REF, "local_wav"),
    ("kokoro", KOKORO_REF, "bank_voice_id"),
])
def test_a_non_bark_assignment_clears_the_writer_preset(assign, engine, ref,
                                                        kind):
    rows = _lock_recurring_row(assign, engine, ref, kind)
    row = rows["c02"]
    assert row.get("voice_preset") == "", (
        "%s kept the Bark preset %r" % (engine, row.get("voice_preset")))
    assert row.get("voice_ref_id"), "the row got no reference at all"


# ---------------------------------------------------------------------------
# 3. BOTH CASTING MODES REACH THE ASSIGNMENT, AND preserve_ledger STILL LEAVES
#    AN UNCLAIMED ROW COMPLETELY ALONE.
#
# `preserve_ledger` is a real casting mode, not an old-save compatibility path;
# the Python `lock()` default is preserve_ledger while the widget default is
# auto_registry, so both are shipping surfaces.
# ---------------------------------------------------------------------------

def test_preserve_ledger_reaches_the_assignment(assign):
    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav",
                               policy="preserve_ledger")
    assert rows["c02"].get("voice_ref_id") == CLONE_REF
    assert not rows["c02"].get("voice_route")


def test_preserve_ledger_leaves_an_unassigned_row_with_nothing_added(assign):
    """No engine mapping for this row means it gains nothing -- not a blank."""
    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav",
                               policy="preserve_ledger")
    monty = rows["c01"]
    assert "voice_ref_id" not in monty, (
        "preserve_ledger added voice_ref_id=%r to a row it should not have "
        "touched; absent and blank are different outcomes"
        % monty.get("voice_ref_id"))


# ---------------------------------------------------------------------------
# 4. WHAT THE ADAPTER IS ACTUALLY HANDED.
#
# The three identity kinds reach the renderer through three different fields.
# This is the half that a stamp-only test cannot see: a row can carry the right
# id and still hand the adapter nothing.
# ---------------------------------------------------------------------------

def _dispatch_identity(engine, cast_row, episode_seed=42):
    """Mirror `_render_per_line`: what the adapter would really receive."""
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_voice_node_common import (
        _engine_requires_voice_ref, _resolve_clone_ref_path,
    )

    adapter = get_engine(engine)
    ref_field = getattr(adapter, "voice_ref_field", "voice_ref_path")
    if ref_field == "voice_ref_path":
        voice_ref = cast_row.get("voice_ref_path") or cast_row.get("ref_path")
    else:
        voice_ref = cast_row.get(ref_field)
    if _engine_requires_voice_ref(adapter) and not voice_ref:
        voice_ref = _resolve_clone_ref_path(engine, cast_row, episode_seed)
    return ref_field, voice_ref


def test_a_local_wav_assignment_reaches_the_exact_bank_file(assign, bank):
    rows = _lock_recurring_row(assign, "chatterbox", CLONE_REF, "local_wav")
    ref_field, voice_ref = _dispatch_identity("chatterbox", rows["c02"])
    assert ref_field == "voice_ref_path"
    assert voice_ref and os.path.isfile(voice_ref), voice_ref
    entry = next(e for e in bank if e.voice_ref_id == CLONE_REF)
    assert os.path.normpath(voice_ref).endswith(
        os.path.normpath(entry.ref_path).split(os.sep)[-1]), (
        "dispatch resolved %r, which is not the bank file %r -- a male "
        "reference the gender fallback would also have been happy with is not "
        "the same thing as the assigned one" % (voice_ref, entry.ref_path))


def test_a_bank_voice_id_assignment_reaches_voice_ref_id(assign):
    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF, "bank_voice_id")
    ref_field, voice_ref = _dispatch_identity("kokoro", rows["c02"])
    assert ref_field == "voice_ref_id"
    assert voice_ref == KOKORO_REF


def test_a_provider_assignment_reaches_the_provider_voice_id(assign):
    rows = _lock_recurring_row(assign, "cloud_elevenlabs", PROVIDER_REF,
                               "provider_voice")
    row = rows["c02"]
    assert row.get("provider_voice_id") == PROVIDER_VOICE_ID or \
        row.get("voice_ref_id") == PROVIDER_REF, (
        "the provider identity reached neither field: %r" % (row,))


@pytest.mark.parametrize("engine,ref,kind", [
    ("chatterbox", CLONE_REF, "local_wav"),
    ("kokoro", KOKORO_REF, "bank_voice_id"),
    ("cloud_elevenlabs", PROVIDER_REF, "provider_voice"),
])
def test_no_assignment_writes_a_route_and_none_of_them_raises(assign, engine,
                                                              ref, kind):
    """The per-line dispatch call runs for every row of every episode."""
    rows = _lock_recurring_row(assign, engine, ref, kind)
    row = rows["c02"]
    assert "voice_route" not in row or not row["voice_route"]
    resolved = ROUTE.resolve_and_verify_reference(row, engine)
    assert resolved is ROUTE.LEGACY_REFERENCE, resolved
    assert resolved.is_policy_route is False


# ---------------------------------------------------------------------------
# 5. IN-GRAPH CACHE IDENTITY.
#
# ComfyUI reruns a node when IS_CHANGED moves. An ordinary episode must keep
# returning the literal "static" -- not "something stable" -- or every shipping
# graph loses its caching.
# ---------------------------------------------------------------------------

def _voice_node():
    from nodes.batch_character_voices import BatchCharacterVoices
    return BatchCharacterVoices


def _ledger_with_assignment(voice_ref_id, engine):
    cast = json.loads(json.dumps(CAST))
    cast[1].update({
        "voice_ref_id": voice_ref_id,
        "voice_engine": engine,
        ROUTE.CAST_ROW_TIER_FIELD: ROUTE.ROUTE_TIER_PROVISIONAL,
        ROUTE.CAST_ROW_ROUTE_ID_FIELD: "lemmy-%s-provisional-v1" % engine,
        ROUTE.CAST_ROW_REASON_FIELD: "",
    })
    return _ledger(cast=cast)


def test_an_ordinary_ledger_is_the_literal_string_static():
    assert _voice_node().IS_CHANGED(
        script_json=_ledger(), engine="chatterbox") == "static"


def test_an_assigned_row_is_fingerprinted_rather_than_static():
    out = _voice_node().IS_CHANGED(
        script_json=_ledger_with_assignment(CLONE_REF, "chatterbox"),
        engine="chatterbox")
    assert out != "static"
    assert isinstance(out, str) and len(out) == 64, out


def test_a_pt_voice_tensor_is_fingerprinted_too():
    """Not only WAVs -- kokoro's identity is a `.pt`, and it still moves."""
    out = _voice_node().IS_CHANGED(
        script_json=_ledger_with_assignment(KOKORO_REF, "kokoro"),
        engine="kokoro")
    assert out != "static"
    assert isinstance(out, str) and len(out) == 64, out


def test_two_different_identities_fingerprint_differently():
    a = _voice_node().IS_CHANGED(
        script_json=_ledger_with_assignment(CLONE_REF, "chatterbox"),
        engine="chatterbox")
    b = _voice_node().IS_CHANGED(
        script_json=_ledger_with_assignment("cb_bill_boerst", "chatterbox"),
        engine="chatterbox")
    assert a != b


def test_an_unreadable_local_identity_fails_OPEN(monkeypatch):
    """A missing file reruns the node; it does not wedge the graph on a stale
    cache entry. NaN is how ComfyUI spells "always rerun"."""
    monkeypatch.setattr(ROUTE, "sha256_of_file", lambda *_a, **_k: None)
    out = _voice_node().IS_CHANGED(
        script_json=_ledger_with_assignment(CLONE_REF, "chatterbox"),
        engine="chatterbox")
    assert isinstance(out, float) and math.isnan(out), out


def test_a_cloud_identity_never_touches_the_filesystem(monkeypatch):
    """A provider voice has no local bytes to hash, and reaching for them would
    be a file read on every cache probe of every cloud render."""
    from nodes import _otr_voice_node_common as VNC

    def _forbidden(*_a, **_k):
        raise AssertionError("a cloud identity hashed a local file")

    monkeypatch.setattr(ROUTE, "sha256_of_file", _forbidden)
    out = VNC._provisional_identity_fingerprint("cloud_elevenlabs", PROVIDER_REF)
    assert out == "provider:%s" % PROVIDER_VOICE_ID, out


# ---------------------------------------------------------------------------
# 6. THE RESERVATION.
#
# One character's own recording is his and nobody else's. The voices he BORROWS
# from the shared catalogue are not -- reserving those would pull the preferred
# announcer out of the pool to protect a cameo, trading one defect for another.
# ---------------------------------------------------------------------------

def test_the_owned_recording_is_never_drawn_for_another_character(bank):
    """Swept across seeds and engines, not asserted once and hoped."""
    from nodes._otr_voice_bank import assign_voice_for_slot, reserved_voice_ref_ids

    reserved = reserved_voice_ref_ids()
    assert reserved, "nothing is reserved -- this test would prove nothing"

    for engine in ("indextts2", "chatterbox", "dia"):
        for seed in range(40):
            ref = assign_voice_for_slot(
                role="char_voice", engine=engine, char_id="c03", gender="male",
                timbre=(), age_band="adult", episode_seed=seed,
                casting_policy_version="v1", allow_voice_reuse=True,
                used_voice_ref_ids=set(), bank=bank)
            if ref is None:
                continue
            rid = ref.voice_ref_id
            assert rid not in reserved, (
                "engine=%s seed=%d drew the reserved %r for an ordinary "
                "character" % (engine, seed, rid))


def test_only_the_owned_recording_is_reserved_not_the_borrowed_catalogue():
    """`bm_george` is a shared kokoro catalogue voice tagged preferred_announcer;
    the cloud ids are shared too. Reserving those would starve the announcer."""
    from nodes._otr_voice_bank import reserved_voice_ref_ids

    reserved = reserved_voice_ref_ids()
    assert "idx_lemmy_algenib_cockney_v1" in reserved, (
        "the owned clone is not reserved: %r" % (sorted(reserved),))
    for borrowed in ("bm_george", "el_daniel", "gt_algenib"):
        assert borrowed not in reserved, (
            "%r is a SHARED catalogue voice and must stay castable" % borrowed)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
