# -*- coding: utf-8 -*-
"""What must still be true after the recurring-character voice rewrite.

WHY THIS FILE EXISTS. A bespoke qualified/provisional/unrouted voice-route
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

def _registry(engine, voice_ref_id):
    """A one-character registry mapping `engine` to `voice_ref_id`."""
    return {RECURRING: {engine: voice_ref_id}}


@pytest.fixture()
def assign(monkeypatch):
    """Install a recurring-character assignment for one engine."""
    def install(engine, voice_ref_id):
        monkeypatch.setattr(
            "config.cast_pools.RECURRING_CHARACTER_VOICES",
            _registry(engine, voice_ref_id))
    return install


def _lock_recurring_row(assign, engine, voice_ref_id, *, cast=None,
                        policy="auto_registry", voice_bank="default_clean"):
    """Lock a cast and hand back every stamped row by char_id."""
    assign(engine, voice_ref_id)
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

    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF,
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

    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF,
                               cast=cast, policy=policy)
    assert "voice_route" not in rows["c02"], (
        "a stale voice_route survived a %s re-stamp as %r; the clear must "
        "DELETE the key, not blank it -- a consumer that reads truthiness and "
        "one that reads presence would then disagree"
        % (policy, rows["c02"].get("voice_route")))


def test_a_row_the_caster_never_reaches_is_left_exactly_as_it_arrived(assign):
    """Byte-for-byte, every field. Not "mostly unchanged"."""
    cast = json.loads(json.dumps(CAST))
    cast[1]["voice_engine"] = "bark"
    cast[1]["voice_preset"] = "v2/en_speaker_8"
    before = json.loads(json.dumps(cast[0]))       # MONTY, never claimed

    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF,
                               cast=cast, policy="preserve_ledger")
    after = rows["c01"]
    for key, value in before.items():
        assert after.get(key) == value, (
            "field %r on an untouched row changed from %r to %r"
            % (key, value, after.get(key)))


def test_only_the_recurring_row_takes_the_assignment(assign):
    """The other rows keep their own draw and gain none of its identity."""
    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF)
    assert rows["c02"].get("voice_ref_id") == KOKORO_REF
    for other in ("c01", "a1"):
        assert rows[other].get("voice_ref_id") != KOKORO_REF, (
            "%s received the recurring character's assigned voice" % other)


# ---------------------------------------------------------------------------
# 2. THE WRITER'S PRESET IS CLEARED WHEN A NON-BARK ENGINE TAKES THE ROW.
#
# The preset is a Bark concept. Leaving it set while another engine renders is
# how a row ends up describing two different voices.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,ref", [
    ("kokoro", KOKORO_REF),
    ("cloud_elevenlabs", PROVIDER_REF),
])
def test_a_non_bark_assignment_clears_the_writer_preset(assign, engine, ref):
    rows = _lock_recurring_row(assign, engine, ref)
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
    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF,
                               policy="preserve_ledger")
    assert rows["c02"].get("voice_ref_id") == KOKORO_REF
    assert "voice_route" not in rows["c02"]


def test_preserve_ledger_leaves_an_unassigned_row_with_nothing_added(assign):
    """No engine mapping for this row means it gains nothing -- not a blank."""
    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF,
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


def test_a_clone_engine_takes_the_ordinary_draw_and_reaches_a_real_file(assign):
    """The closest surviving relative of a test whose subject went away.

    This used to assert that a `local_wav` IDENTITY KIND dispatched to the exact
    bank file it named. Identity kinds were a property of the route system and
    went with it. The table deliberately maps NO clone engine, because a clone
    row is reserved and a recurring character does not need an assignment to
    avoid being cast as somebody else.

    What still has to hold, and is checked nowhere else, is that a clone engine
    resolves to a REAL FILE ON DISK through the ordinary draw -- an engine that
    requires a reference and is handed a path that is not there renders nothing.
    """
    rows = _lock_recurring_row(assign, "chatterbox", KOKORO_REF)
    ref_field, voice_ref = _dispatch_identity("chatterbox", rows["c02"])
    assert ref_field == "voice_ref_path"
    assert voice_ref and os.path.isfile(voice_ref), (
        "chatterbox dispatch resolved %r, which is not a file on disk" % voice_ref)


def test_a_bank_voice_id_assignment_reaches_voice_ref_id(assign):
    rows = _lock_recurring_row(assign, "kokoro", KOKORO_REF)
    ref_field, voice_ref = _dispatch_identity("kokoro", rows["c02"])
    assert ref_field == "voice_ref_id"
    assert voice_ref == KOKORO_REF


def test_a_provider_assignment_reaches_the_provider_voice_id(assign):
    rows = _lock_recurring_row(assign, "cloud_elevenlabs", PROVIDER_REF)
    row = rows["c02"]
    assert row.get("provider_voice_id") == PROVIDER_VOICE_ID or \
        row.get("voice_ref_id") == PROVIDER_REF, (
        "the provider identity reached neither field: %r" % (row,))


@pytest.mark.parametrize("policy", ["auto_registry", "preserve_ledger"])
@pytest.mark.parametrize("engine,ref", [
    ("kokoro", KOKORO_REF),
    ("cloud_elevenlabs", PROVIDER_REF),
])
def test_a_locked_row_sheds_every_retired_route_field(assign, engine, ref,
                                                      policy):
    """Seeded, then cleared. The seeding is the entire point.

    THE FIELD NAMES ARE THE REAL ONES, and getting them wrong is how the first
    version of this test managed to be permanently true: it asserted
    `voice_route_reason`, which `git log --all -S` shows never existed in this
    repo, and `voice_route_id`, which is a LEDGER-LINE field and was never a
    cast-row key. Both assertions passed by construction. The three fields a
    cast row actually carried are below, and they match
    `cast_lock._STALE_IDENTITY_FIELDS` -- if that tuple is edited and this is
    not, one of them stops being cleared and nothing else notices.

    A row locked before 2026-09-24 still carries these, so this is a real
    migration path and not a hypothetical one.
    """
    retired = {
        "lemmy_route_tier": "qualified",
        "lemmy_route_id": "lemmy-indextts2-algenib-cockney-v1",
        "lemmy_route_reason_code": "",
        "voice_route": {"status": "qualified", "route_id": "stale-v0"},
    }
    cast = json.loads(json.dumps(CAST))
    cast[1].update(retired)
    # The seeded row must really carry them, or this proves nothing.
    assert all(k in cast[1] for k in retired), cast[1]

    rows = _lock_recurring_row(assign, engine, ref, cast=cast, policy=policy)
    row = rows["c02"]
    survivors = [k for k in retired if k in row]
    assert not survivors, (
        "a re-locked row kept %r -- it now carries a tier claim about a "
        "subsystem that no longer exists, and the clear must DELETE the key "
        "rather than blank it, so presence and truthiness cannot disagree"
        % (survivors,))


def test_the_retired_field_list_here_matches_the_one_the_code_clears():
    """The two lists are spelled separately on purpose; pin them together.

    `_STALE_IDENTITY_FIELDS` holds the retired names as literals because the
    module that defined them is deleted. That is the right call and it is also
    exactly how a test drifts from the code it guards -- so the drift is
    caught here instead of being discovered on a stale ledger.
    """
    from nodes.cast_lock import _STALE_IDENTITY_FIELDS

    # THE EXACT SET, not a subset. Asserting "these four are present" lets the
    # tuple grow an eighth name that no test has ever seen cleared, which is
    # how a list-versus-list pin quietly stops being one.
    assert set(_STALE_IDENTITY_FIELDS) == {
        # engine-specific identity a re-cast row must not keep
        "voice_ref_path",
        "ref_path",
        "provider_voice_id",
        # the retired route vocabulary, spelled as literals in both places
        # because the module that defined them is deleted
        "voice_route",
        "lemmy_route_tier",
        "lemmy_route_id",
        "lemmy_route_reason_code",
    }, (
        "the cleared-field set moved. Add the new name to the seeded row in "
        "test_a_locked_row_sheds_every_retired_route_field as well, or it is "
        "cleared by code that no test ever watches clear it: %r"
        % (sorted(_STALE_IDENTITY_FIELDS),))


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
    from nodes import _otr_voice_node_common as VNC
    monkeypatch.setattr(VNC, "sha256_of_file", lambda *_a, **_k: None)
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

    monkeypatch.setattr(VNC, "sha256_of_file", _forbidden)
    out = VNC._bank_identity_fingerprint("cloud_elevenlabs", PROVIDER_REF)
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


# ---------------------------------------------------------------------------
# 7. A RESERVATION REACHES ITS OWNER.
#
# Section 6 proves the clone is withheld from everybody else. That is only half
# of a reservation, and for a while it was the only half that was true: between
# the casting cutover and 2026-09-24 Lemmy's chatterbox and dia clones were
# withheld from every other character AND never delivered to him, so he was cast
# on an ordinary librivox voice while his own recording sat reserved for nobody.
# Measured against the pre-cutover commit, not inferred.
# ---------------------------------------------------------------------------

_CLONE_ENGINES = ("indextts2", "chatterbox", "dia")


@pytest.mark.parametrize("policy", ["auto_registry", "preserve_ledger"])
@pytest.mark.parametrize("engine", _CLONE_ENGINES)
def test_the_reserved_recording_is_delivered_to_the_character_it_names(
        engine, policy):
    """His own voice, on every engine that has a recording of it, in BOTH modes.

    No `assign` fixture here on purpose: the whole point is that this works
    from the BANK's `reserved_for`, with nothing in the catalogue table naming
    these ids. `recurring_character_voice("LEMMY", "indextts2")` returns "" and
    must keep doing so -- the table holds shared rows only.

    THE POLICY PARAMETER IS NOT DECORATION. The first version of this test ran
    auto_registry only, and preserve_ledger -- which is `lock()`'s own default
    -- still had the whole bug: its pre-filter asked the catalogue table before
    the reserved scan could run, so on these three engines the row was dropped
    and Lemmy came back with no voice at all. A fix proven in one mode of a
    two-mode switch is proven in half the product.
    """
    from nodes._otr_voice_bank import load_voice_bank

    bank = load_voice_bank()[0]
    owned = [e for e in bank
             if e.engine == engine
             and str(getattr(e, "reserved_for", "") or "").strip().casefold()
             == RECURRING.casefold()]
    assert len(owned) == 1, (
        "expected exactly one %s row reserved for %s, found %d"
        % (engine, RECURRING, len(owned)))

    out = CastLock().lock(
        script_json=_ledger(), voice_bank="default_clean",
        char_voice_engine=engine, cast_voice_policy=policy)[0]
    row = _rows(out)["c02"]
    assert row.get("voice_ref_id") == owned[0].voice_ref_id, (
        "%s was cast on %r while his own reserved %s recording %r went to "
        "nobody -- a reservation that never reaches its owner withholds a "
        "voice from everyone"
        % (RECURRING, row.get("voice_ref_id"), engine, owned[0].voice_ref_id))


@pytest.mark.parametrize("policy", ["auto_registry", "preserve_ledger"])
def test_the_reservation_outranks_a_catalogue_assignment(assign, policy):
    """When both exist, his own recording wins.

    Seeded by pointing the TABLE at a shared row on an engine where he also
    owns a reserved one. The shared row is a stand-in; the recording is him.
    """
    from nodes._otr_voice_bank import load_voice_bank

    bank = load_voice_bank()[0]
    owned = next(e for e in bank
                 if e.engine == "chatterbox"
                 and str(getattr(e, "reserved_for", "") or "").strip().casefold()
                 == RECURRING.casefold())
    stand_in = next(e for e in bank
                    if e.engine == "chatterbox"
                    and e.voice_ref_id != owned.voice_ref_id
                    and not str(getattr(e, "reserved_for", "") or "").strip())

    rows = _lock_recurring_row(assign, "chatterbox", stand_in.voice_ref_id,
                               policy=policy)
    assert rows["c02"].get("voice_ref_id") == owned.voice_ref_id, (
        "the catalogue stand-in %r beat his own recording %r"
        % (stand_in.voice_ref_id, owned.voice_ref_id))


def test_an_ambiguous_reservation_refuses_rather_than_guessing(monkeypatch):
    """Two rows reserved for one character on one engine is a broken bank.

    Picking either would make the cast depend on file order, which is the same
    reasoning the exactly-one-match rule uses on the catalogue path. The row
    falls through to the ordinary draw and the report says why.
    """
    from nodes import cast_lock as CL
    from nodes._otr_voice_bank import load_voice_bank

    bank = list(load_voice_bank()[0])
    owned = next(e for e in bank
                 if e.engine == "chatterbox"
                 and str(getattr(e, "reserved_for", "") or "").strip().casefold()
                 == RECURRING.casefold())
    import dataclasses

    twin = dataclasses.replace(owned,
                               voice_ref_id=owned.voice_ref_id + "_twin")
    assert twin.voice_ref_id != owned.voice_ref_id, (
        "the twin is not distinct, so this would be testing one row twice")

    ref, miss = CL._recurring_character_bank_ref(
        {"char_id": "c02", "name": RECURRING}, "chatterbox",
        bank + [twin], "en")
    assert ref is None, "an ambiguous reservation was resolved to %r" % (ref,)
    assert "reserved" in miss, miss


@pytest.mark.parametrize("engine", _CLONE_ENGINES)
def test_a_reserved_recording_that_is_not_on_this_machine_is_not_delivered(
        engine, monkeypatch):
    """The clone is his voice only where his bytes actually are.

    ALL THREE reserved rows name ONE private clip,
    `models/TTS/refs/indextts2/lemmy_algenib_cockney_v1.wav`, and NOTHING puts
    it on a fresh machine: `scripts/otr_dl_indextts2_refs.py` has no entry for
    it, and `scripts/otr_provision.py` skips reserved rows on purpose --
    "refusing to distribute a private recording" -- so the provisioner reports
    green on a box where the file is absent.

    Without this guard casting hands the adapter a path that is not there and
    the voice path fails loud by design (the 2026-07-03 no-fallback rip). 51
    profiles put a clone engine on the character slot, including the rented-pod
    starter, so a cameo would turn a working episode into a dead render on
    every machine except the one that recorded the clip. It looked correct here
    for exactly that reason -- the clip is on this box and essentially nowhere
    else, which is the same shape as the dormant-import defect.

    The fallback must also be LOUD: an operator who hears a stranger needs to
    be told the reference is missing, because fetching it is the whole fix.
    """
    import os

    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path, "exists",
        lambda p: False if "lemmy_algenib_cockney" in str(p)
        else real_exists(p))

    out, _n, *rest = CastLock().lock(
        script_json=_ledger(), voice_bank="default_clean",
        char_voice_engine=engine, cast_voice_policy="auto_registry")
    row = _rows(out)["c02"]
    got = str(row.get("voice_ref_id") or "")

    assert "lemmy_algenib_cockney" not in got, (
        "a reserved recording that is NOT on this machine was delivered as %r; "
        "the adapter would be handed a path that does not exist and the line "
        "would fail to render" % (got,))
    assert got, (
        "the character was left with no voice at all; a missing reserved "
        "reference must fall through to the ordinary draw, not to nothing")

    said = chr(10).join(x for x in rest if isinstance(x, str))
    assert "not on this machine" in said, (
        "the fallback was SILENT. The operator hears a stranger and is told "
        "nothing about why, when the fix is to fetch the clip. Report: %s"
        % (said[:600],))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
