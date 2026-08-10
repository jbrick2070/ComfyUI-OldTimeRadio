"""Plan 5.3 (second half) -- reference resolution, receipts, and IS_CHANGED.

THE THING BEING PREVENTED. A receipt that names a route is worthless if the
render used a different file, and a fingerprint that says "static" is a lie the
moment a pinned reference can be swapped underneath it. So the tests that matter
most here are the ones asserting that a PROVED route renders its OWN bytes, and
that changing those bytes changes IS_CHANGED.

THE OTHER THING BEING PREVENTED, equally: none of this may touch a render that
has no route. Every legacy assertion below is a byte-identity assertion.

Headless. No engine, no model, no GPU, no network.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from nodes import _otr_voice_route as ROUTE

SUPPORTED_VERSION = ROUTE.SUPPORTED_ROUTE_CONTRACT_VERSIONS[0]


def _wav(tmp_path, body=b"RIFF....WAVEfmt the-pinned-reference-bytes"):
    p = tmp_path / "lemmy_algenib_cockney_v1.wav"
    p.write_bytes(body)
    return p, hashlib.sha256(body).hexdigest()


def _route(tmp_path, **over):
    path, digest = _wav(tmp_path)
    route = {
        "route_id": "lemmy-indextts2-algenib-v1",
        "route_contract_version": SUPPORTED_VERSION,
        "status": "qualified",
        "engine": "indextts2",
        "voice_ref_id": "vz_peter_yearsley",
        "reference_kind": "local_wav",
        "ref_path": str(path),
        "source_ref_sha256": digest,
        "qualification_record_id": "qr-lemmy-0001",
        "runtime": {
            "model_id": "IndexTTS-2",
            "engine_impl_version": "1.0.0",
            "weight_revision": "abc123",
        },
    }
    route.update(over)
    return route


def _row(route=None, **over):
    row = {"char_id": "c02", "name": "LEMMY", "voice_ref_id": "vz_peter_yearsley"}
    if route is not None:
        row["voice_route"] = route
    row.update(over)
    return row


def _bank_lookup(voice_ref_id):
    from nodes._otr_voice_bank import load_voice_bank
    return next((e for e in load_voice_bank()[0]
                 if e.voice_ref_id == voice_ref_id and e.engine == "indextts2"),
                None)


# ---------------------------------------------------------------------------
# The legacy row: nothing changes, and that is the assertion.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", [
    {"char_id": "c01", "voice_ref_id": "vz_bill_boerst"},
    {"char_id": "c01", "voice_route": {}},
    {"char_id": "c01", "voice_route": None},
    {},
    "not a dict",
])
def test_a_row_without_a_route_resolves_to_the_empty_identity(row):
    got = ROUTE.resolve_and_verify_reference(row, "indextts2")
    assert got.is_policy_route is False
    assert got.request_fields() == ROUTE.LEGACY_REFERENCE_IDENTITY


def test_the_legacy_identity_is_all_empty_so_cache_keys_do_not_move():
    """The schema grew in 5.3's first half. This is the guard that says it did
    not move a single existing line's key."""
    assert ROUTE.LEGACY_REFERENCE_IDENTITY == {
        "route_id": "",
        "route_contract_version": 0,
        "qualification_record_id": "",
        "weight_revision": "",
        "source_ref_sha256": "",
    }


def test_a_legacy_row_never_needs_a_bank_or_an_engine():
    """A row with no route must resolve even when nothing else is available --
    otherwise adding this call would break every existing render."""
    def explode(_):                              # pragma: no cover
        raise AssertionError("legacy row consulted the bank")

    assert ROUTE.resolve_and_verify_reference(
        _row(), None, bank_lookup=explode).is_policy_route is False


# ---------------------------------------------------------------------------
# The proved route.
# ---------------------------------------------------------------------------
def test_a_proved_route_supplies_its_identity_and_its_hash(tmp_path):
    route = _route(tmp_path)
    got = ROUTE.resolve_and_verify_reference(
        _row(route), "indextts2", bank_lookup=_bank_lookup)
    assert got.is_policy_route is True
    assert got.route_id == "lemmy-indextts2-algenib-v1"
    assert got.qualification_record_id == "qr-lemmy-0001"
    assert got.weight_revision == "abc123"
    assert got.source_ref_sha256 == route["source_ref_sha256"]
    assert got.ref_path == route["ref_path"]
    assert got.request_fields()["route_id"] == "lemmy-indextts2-algenib-v1"


def test_the_bytes_are_rehashed_at_point_of_use(tmp_path):
    """A receipt proved at cast time says nothing about the file five minutes
    later, so the check runs again here."""
    route = _route(tmp_path)
    with open(route["ref_path"], "wb") as fh:
        fh.write(b"a completely different recording")
    with pytest.raises(ROUTE.VoiceRouteError, match="BYTES DO NOT MATCH"):
        ROUTE.resolve_and_verify_reference(_row(route), "indextts2")


@pytest.mark.parametrize("over,fragment", [
    ({"engine": "chatterbox"}, "ENGINE DISAGREEMENT"),
    ({"status": "candidate"}, "status"),
    ({"route_contract_version": 99}, "does not understand"),
    ({"voice_ref_id": ""}, "names no voice_ref_id"),
    ({"reference_kind": "telepathy"}, "not a defined kind"),
    ({"source_ref_sha256": "nope"}, "not 64 hex"),
    ({"ref_path": ""}, "ref_path is missing"),
])
def test_an_unproved_route_raises_rather_than_rendering(tmp_path, over, fragment):
    with pytest.raises(ROUTE.VoiceRouteError, match=fragment):
        ROUTE.resolve_and_verify_reference(
            _row(_route(tmp_path, **over)), "indextts2")


def test_a_route_cannot_be_proved_against_an_unknown_engine(tmp_path):
    with pytest.raises(ROUTE.VoiceRouteError, match="active voice engine is unknown"):
        ROUTE.resolve_and_verify_reference(_row(_route(tmp_path)), "")


def test_a_reference_missing_from_disk_raises(tmp_path):
    route = _route(tmp_path)
    import os
    os.remove(route["ref_path"])
    with pytest.raises(ROUTE.VoiceRouteError, match="does not exist"):
        ROUTE.resolve_and_verify_reference(_row(route), "indextts2")


def test_a_ref_id_absent_from_the_bank_raises(tmp_path):
    route = _route(tmp_path, voice_ref_id="no_such_reference")
    with pytest.raises(ROUTE.VoiceRouteError, match="not in the active voice bank"):
        ROUTE.resolve_and_verify_reference(
            _row(route), "indextts2", bank_lookup=_bank_lookup)


# ---------------------------------------------------------------------------
# provider_voice: no local file, ever.
# ---------------------------------------------------------------------------
def test_a_provider_route_is_never_treated_as_a_local_file():
    """gt_algenib's ref_sha256 is the literal string 'cloud'. Demanding a file
    for a cloud voice is the category error this branch refuses to make."""
    route = {
        "route_id": "lemmy-google-algenib-v1",
        "route_contract_version": SUPPORTED_VERSION,
        "status": "qualified",
        "engine": "google_tts",
        "voice_ref_id": "gt_algenib",
        "reference_kind": "provider_voice",
        "ref_path": "",
        "source_ref_sha256": "cloud",
        "qualification_record_id": "qr-g-1",
        "runtime": {"weight_revision": "v7"},
    }
    got = ROUTE.resolve_and_verify_reference(_row(route), "google_tts")
    assert got.is_policy_route is True
    assert got.reference_kind == "provider_voice"
    # No local bytes are claimed -- the nonsense sha is dropped, not hashed.
    assert got.source_ref_sha256 == ""
    assert got.ref_path == ""


# ---------------------------------------------------------------------------
# The request builder carries route identity into the cache key.
# ---------------------------------------------------------------------------
def _build(**kw):
    from nodes._otr_resolved_request import build_resolved_request
    base = dict(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="p1", engine_impl_version="1",
        char_id="c02", line_id="L1", prepared_text="hello",
    )
    base.update(kw)
    return build_resolved_request(**base)


def test_route_identity_reaches_the_request_and_moves_the_cache_key(tmp_path):
    plain = _build()
    routed = _build(**ROUTE.resolve_and_verify_reference(
        _row(_route(tmp_path)), "indextts2").request_fields())
    assert routed.route_id == "lemmy-indextts2-algenib-v1"
    assert routed.weight_revision == "abc123"
    assert routed.cache_key != plain.cache_key, (
        "a qualified route must key its own audio")


def test_a_legacy_request_keeps_the_defaults_that_preserve_its_key():
    req = _build(**ROUTE.LEGACY_REFERENCE.request_fields())
    assert (req.route_id, req.route_contract_version) == ("", 0)
    assert (req.qualification_record_id, req.weight_revision) == ("", "")
    assert req.cache_key == _build().cache_key


def test_two_routes_differing_only_in_weight_revision_do_not_share_audio():
    a = _build(route_id="r1", weight_revision="w1")
    b = _build(route_id="r1", weight_revision="w2")
    assert a.cache_key != b.cache_key


# ---------------------------------------------------------------------------
# IS_CHANGED (plan 5.3): a fingerprint, not a constant.
# ---------------------------------------------------------------------------
def _ledger_json(cast):
    return json.dumps({"meta": {"episode_seed": 1}, "cast": cast, "lines": []})


def _is_changed(**kw):
    from nodes.batch_character_voices import BatchCharacterVoices
    return BatchCharacterVoices.IS_CHANGED(engine="indextts2", **kw)


def test_a_ledger_with_no_routes_is_still_the_literal_string_static():
    """Byte-identical in-graph caching for every render shipping today. Not
    'something stable' -- the same string as before."""
    assert _is_changed() == "static"
    assert _is_changed(ledger_json=_ledger_json([_row()])) == "static"


def test_an_unparseable_ledger_stays_static_rather_than_disabling_caching():
    assert _is_changed(ledger_json="{not json at all") == "static"


def test_a_routed_ledger_fingerprints_instead(tmp_path):
    led = _ledger_json([_row(_route(tmp_path))])
    got = _is_changed(ledger_json=led)
    assert got != "static"
    assert len(got) == 64
    assert got == _is_changed(ledger_json=led), "fingerprint must be stable"


def test_swapping_the_reference_bytes_changes_the_fingerprint(tmp_path):
    """The whole reason 'static' had to go. Same ledger, same route, different
    audio on disk -- ComfyUI must not serve the previous render."""
    route = _route(tmp_path)
    led = _ledger_json([_row(route)])
    before = _is_changed(ledger_json=led)
    with open(route["ref_path"], "wb") as fh:
        fh.write(b"different reference audio entirely")
    assert _is_changed(ledger_json=led) != before


def test_bumping_the_contract_version_changes_the_fingerprint(tmp_path):
    a = _is_changed(ledger_json=_ledger_json([_row(_route(tmp_path))]))
    b = _is_changed(ledger_json=_ledger_json(
        [_row(_route(tmp_path, route_contract_version=2))]))
    assert a != b


def test_an_unreadable_pinned_reference_fails_OPEN(tmp_path):
    route = _route(tmp_path)
    led = _ledger_json([_row(route)])
    import os
    os.remove(route["ref_path"])
    got = _is_changed(ledger_json=led)
    assert got != got, "an unreadable expected local file must return NaN"


def test_a_provider_route_fingerprints_without_touching_the_filesystem(monkeypatch):
    """Never a network call, and never a file read for a cloud voice."""
    def explode(_path):                          # pragma: no cover
        raise AssertionError("provider_voice route hashed a local file")

    monkeypatch.setattr(ROUTE, "sha256_of_file", explode)
    led = _ledger_json([_row({
        "route_id": "lemmy-google-algenib-v1",
        "route_contract_version": SUPPORTED_VERSION,
        "status": "qualified",
        "engine": "indextts2",
        "voice_ref_id": "gt_algenib",
        "reference_kind": "provider_voice",
        "source_ref_sha256": "cloud",
        "runtime": {"weight_revision": "v7"},
    })])
    got = _is_changed(ledger_json=led)
    assert got != "static" and len(got) == 64


# ---------------------------------------------------------------------------
# Per-line receipts.
# ---------------------------------------------------------------------------
def test_the_stamp_helper_records_route_and_sample_rate():
    from nodes._otr_ledger import stamp_per_line_audio_meta

    led = {"lines": [{"line_id": "L1", "text": "hi"}]}
    assert stamp_per_line_audio_meta(
        led, "L1", tts_engine="indextts2",
        voice_route_id="lemmy-indextts2-algenib-v1", sample_rate=24000)
    line = led["lines"][0]
    assert line["voice_route_id"] == "lemmy-indextts2-algenib-v1"
    assert line["sample_rate"] == 24000
    assert line["tts_engine"] == "indextts2"


def test_the_new_receipt_fields_are_skipped_when_empty():
    """Skip-when-empty, like every field beside them: a later role must not be
    able to blank a receipt an earlier one wrote."""
    from nodes._otr_ledger import stamp_per_line_audio_meta

    led = {"lines": [{"line_id": "L1", "text": "hi",
                      "voice_route_id": "kept", "sample_rate": 24000}]}
    stamp_per_line_audio_meta(led, "L1", tts_engine="bark")
    assert led["lines"][0]["voice_route_id"] == "kept"
    assert led["lines"][0]["sample_rate"] == 24000


def test_the_flush_reports_WHICH_lines_failed_not_just_how_many(tmp_path):
    """The count alone cannot answer 'did the qualified route's own receipt
    land?'. Without the id set, an unrelated line's failed stamp would throw
    away good, fully evidenced route audio."""
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(
        {"lines": [{"line_id": "L1", "text": "hi"}]}), encoding="utf-8")
    meta = {"paths": {"ledger_path": str(ledger_path)}}

    failed = set()
    degraded = _persist_ledger_stamps(
        meta,
        [("L1", {"tts_engine": "indextts2"}),
         ("L_NOT_IN_LEDGER", {"tts_engine": "indextts2"})],
        __import__("logging").getLogger("test"),
        failed_line_ids=failed,
    )
    assert degraded == 1
    assert failed == {"L_NOT_IN_LEDGER"}, (
        "the good line must not be blamed for the bad one")


def test_a_missing_ledger_path_blames_every_stamp(tmp_path):
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    failed = set()
    degraded = _persist_ledger_stamps(
        {"paths": {}}, [("L1", {"tts_engine": "bark"})],
        __import__("logging").getLogger("test"), failed_line_ids=failed)
    assert degraded == 1
    assert failed == {"L1"}


def test_the_flush_still_works_without_the_new_argument(tmp_path):
    """Three positional args is the existing call shape, including a spy in
    tests/test_audio_cache_wiring.py."""
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    assert _persist_ledger_stamps(
        {"paths": {}}, [("L1", {"tts_engine": "bark"})],
        __import__("logging").getLogger("test")) == 1


def test_voice_route_id_is_in_the_null_shape_audit_vocabulary():
    """A new optional string field the post-freeze audit does not know about is
    a field that can silently be None on disk."""
    from nodes._otr_ledger_consumers import _OPTIONAL_STRING_FIELDS

    assert "voice_route_id" in _OPTIONAL_STRING_FIELDS
