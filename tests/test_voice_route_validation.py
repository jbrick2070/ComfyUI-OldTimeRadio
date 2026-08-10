"""A qualified voice route must PROVE itself, not merely be filled in.

The predecessor check (`is_qualified_route`) verified that receipt fields were
present and non-blank. That killed the bare-string "approval" it was written
for, but "the fields are filled in" is not "the claim is true": a receipt can
name a file that does not exist, a hash that does not match the bytes on disk,
an engine that disagrees with the one actually rendering, or a rights approval
that expired -- and every field stays non-blank.

Each test below removes exactly ONE truth from an otherwise-perfect route and
asserts the validator refuses it. The `test_a_complete_route_validates` control
is what stops the whole file from being fail-always, which would prove nothing
and would block a real audition from ever registering.
"""
from __future__ import annotations

import copy
import hashlib
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_voice_route import (  # noqa: E402
    REFERENCE_KINDS,
    RIGHTS_STATUSES,
    ROUTE_STATUSES,
    TECHNICAL_VERDICTS,
    parse_utc,
    sha256_of_file,
    validate_qualified_voice_route,
)

NOW = datetime(2026, 8, 10, 21, 0, 0, tzinfo=timezone.utc)
HEX64 = "a" * 64


@pytest.fixture()
def local_ref(tmp_path):
    """A real file on disk, with its real hash -- so byte checks are exercised
    against reality rather than against a mock that always agrees."""
    p = tmp_path / "lemmy_algenib_cockney_v1.wav"
    p.write_bytes(b"RIFF....WAVEfmt   not really audio, but real bytes")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def route(local_ref):
    path, sha = local_ref
    return {
        "route_id": "lemmy-indextts2-algenib-cockney-v1",
        "route_contract_version": 1,
        "qualification_record": {
            "status": "qualified",
            "character_key": "LEMMY",
            "engine": "indextts2",
            "voice_ref_id": "idx_lemmy_algenib_cockney_v1",
            "source_provider_ref_id": "gt_algenib",
            "reference": {
                "kind": "local_wav",
                "absolute_path": path,
                "source_ref_sha256": sha,
                "bank_ref_sha256": sha,
            },
            "runtime": {
                "engine_impl_version": "1",
                "model_id": "indextts2",
                "weight_revision": "rev-a",
            },
            "technical_verdict": "pass",
            "audition_manifest": {"path": "docs/evidence/x.json", "sha256": HEX64},
            "rights": {
                "status": "approved",
                "source": "google gemini api",
                "source_tier": "unpaid",
                "terms_snapshot_ref": "docs/2026-08-10-G0-RIGHTS-DECISION-CARD-lemmy.md",
                "terms_snapshot_date": "2026-08-10",
                "scope": "clone reference for local engines",
                "decided_at": "2026-08-10T20:37:17Z",
                "expires_at": None,
                "revoked_at": None,
            },
        },
    }


def _bank(engine="indextts2"):
    return lambda vid: {"voice_ref_id": vid, "engine": engine}


def _check(route, **kw):
    kw.setdefault("bank_lookup", _bank())
    kw.setdefault("active_engine", "indextts2")
    return validate_qualified_voice_route(route, NOW, **kw)


# ---------------------------------------------------------------------------
# THE CONTROL -- without this the suite could be fail-always and prove nothing
# ---------------------------------------------------------------------------

def test_a_complete_route_validates(route):
    v = _check(route)
    assert v.ok, v.summary
    assert v.reasons == ()
    assert bool(v) is True


# ---------------------------------------------------------------------------
# the three statuses, each of which must be EXACTLY right
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [s for s in ROUTE_STATUSES if s != "qualified"])
def test_every_non_qualified_status_is_refused(route, status):
    route["qualification_record"]["status"] = status
    assert not _check(route).ok


@pytest.mark.parametrize("verdict", [v for v in TECHNICAL_VERDICTS if v != "pass"])
def test_every_non_pass_verdict_is_refused(route, verdict):
    route["qualification_record"]["technical_verdict"] = verdict
    assert not _check(route).ok


@pytest.mark.parametrize("rs", [r for r in RIGHTS_STATUSES if r != "approved"])
def test_every_non_approved_rights_status_is_refused(route, rs):
    route["qualification_record"]["rights"]["status"] = rs
    assert not _check(route).ok


@pytest.mark.parametrize("bogus", ["", None, "QUALIFIED", "ok", "approved-ish", 1])
def test_an_undefined_status_value_is_refused_rather_than_guessed(route, bogus):
    """Fail-closed on vocabulary. A status nobody defined is a status nobody
    can reason about -- including a case variant of a legal one."""
    route["qualification_record"]["status"] = bogus
    assert not _check(route).ok


# ---------------------------------------------------------------------------
# rights lifecycle -- an approval is not permanent
# ---------------------------------------------------------------------------

def test_revoked_rights_refuse_even_when_status_still_says_approved(route):
    """Revocation must win over a stale status field."""
    route["qualification_record"]["rights"]["revoked_at"] = "2026-08-09T00:00:00Z"
    v = _check(route)
    assert not v.ok
    assert any("REVOKED" in r for r in v.reasons), v.summary


def test_expired_rights_are_refused(route):
    route["qualification_record"]["rights"]["expires_at"] = (
        (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z"))
    v = _check(route)
    assert not v.ok
    assert any("EXPIRED" in r for r in v.reasons), v.summary


def test_rights_expiring_in_the_future_are_fine(route):
    route["qualification_record"]["rights"]["expires_at"] = (
        (NOW + timedelta(days=365)).isoformat().replace("+00:00", "Z"))
    assert _check(route).ok


def test_an_unparseable_expiry_is_refused_not_ignored(route):
    """A timestamp nobody can parse must not read as 'no expiry'."""
    route["qualification_record"]["rights"]["expires_at"] = "next tuesday"
    assert not _check(route).ok


@pytest.mark.parametrize("field", ["source", "terms_snapshot_ref",
                                   "terms_snapshot_date", "scope", "decided_at"])
def test_rights_provenance_fields_are_each_required(route, field):
    """An approval nobody can audit is not an approval."""
    route["qualification_record"]["rights"][field] = ""
    assert not _check(route).ok


def test_a_missing_rights_block_is_refused(route):
    del route["qualification_record"]["rights"]
    v = _check(route)
    assert not v.ok
    assert any("never inferred from silence" in r for r in v.reasons), v.summary


# ---------------------------------------------------------------------------
# THE ENGINE TRIPLE -- the check that stops a policy approving one engine
# while a different one actually renders
# ---------------------------------------------------------------------------

def test_route_engine_must_match_the_active_scalar_engine(route):
    v = _check(route, active_engine="chatterbox")
    assert not v.ok
    assert any("ENGINE DISAGREEMENT" in r for r in v.reasons), v.summary


def test_route_engine_must_match_the_bank_entry_engine(route):
    v = _check(route, bank_lookup=_bank("dia"))
    assert not v.ok
    assert any("ENGINE DISAGREEMENT" in r for r in v.reasons), v.summary


def test_a_voice_ref_absent_from_the_bank_is_refused(route):
    v = _check(route, bank_lookup=lambda vid: None)
    assert not v.ok
    assert any("not present in the voice bank" in r for r in v.reasons), v.summary


def test_a_raising_bank_lookup_is_refused_not_crashed(route):
    def boom(vid):
        raise RuntimeError("bank on fire")
    v = _check(route, bank_lookup=boom)
    assert not v.ok


# ---------------------------------------------------------------------------
# the reference bytes -- the check a filled-in receipt cannot fake
# ---------------------------------------------------------------------------

def test_a_missing_reference_file_is_refused(route, tmp_path):
    route["qualification_record"]["reference"]["absolute_path"] = str(
        tmp_path / "not_here.wav")
    v = _check(route)
    assert not v.ok
    assert any("does not exist" in r for r in v.reasons), v.summary


def test_bytes_that_do_not_match_the_receipt_are_refused(route):
    """THE POINT OF THE WHOLE MODULE. Every field is present and well-formed;
    only the actual bytes disagree."""
    route["qualification_record"]["reference"]["source_ref_sha256"] = "b" * 64
    v = _check(route)
    assert not v.ok
    assert any("DO NOT MATCH" in r for r in v.reasons), v.summary


@pytest.mark.parametrize("bad", ["", "cloud", "xyz", "A" * 63, "A" * 65, None])
def test_a_hash_that_is_not_64_hex_is_refused(route, bad):
    """`ref_sha256: "cloud"` is a real value in the bank today -- it is what a
    provider route carries, and it must never satisfy a local_wav route."""
    route["qualification_record"]["reference"]["source_ref_sha256"] = bad
    assert not _check(route).ok


def test_a_provider_voice_route_is_not_asked_for_local_bytes(route):
    """The confusion that made gt_algenib look usable as a local IndexTTS2
    reference: a cloud route has no file to hash and must not be judged as if
    it did."""
    ref = route["qualification_record"]["reference"]
    ref.clear()
    ref.update({"kind": "provider_voice", "provider": "google_tts",
                "provider_voice_id": "Algenib"})
    assert _check(route).ok


def test_a_provider_route_still_needs_its_provider_fields(route):
    ref = route["qualification_record"]["reference"]
    ref.clear()
    ref.update({"kind": "provider_voice", "provider": "", "provider_voice_id": ""})
    assert not _check(route).ok


@pytest.mark.parametrize("kind", ["", None, "wav", "local", "LOCAL_WAV"])
def test_an_undefined_reference_kind_is_refused(route, kind):
    route["qualification_record"]["reference"]["kind"] = kind
    assert not _check(route).ok


# ---------------------------------------------------------------------------
# envelope, runtime identity, evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", [0, 2, 99, None, "1"])
def test_an_unsupported_contract_version_is_refused(route, version):
    """A validator that accepts a schema it was not written for is guessing."""
    route["route_contract_version"] = version
    assert not _check(route).ok


@pytest.mark.parametrize("field", ["engine_impl_version", "model_id",
                                   "weight_revision"])
def test_runtime_identity_fields_are_each_required(route, field):
    route["qualification_record"]["runtime"][field] = ""
    assert not _check(route).ok


def test_a_missing_audition_manifest_is_refused(route):
    del route["qualification_record"]["audition_manifest"]
    v = _check(route)
    assert not v.ok
    assert any("no evidence packet" in r for r in v.reasons), v.summary


def test_a_manifest_hash_that_is_not_64_hex_is_refused(route):
    route["qualification_record"]["audition_manifest"]["sha256"] = "pending"
    assert not _check(route).ok


@pytest.mark.parametrize("bogus", [None, "", [], 0, "a string"])
def test_non_dict_records_are_refused_rather_than_crashing(bogus):
    assert not validate_qualified_voice_route(bogus, NOW).ok


def test_reasons_are_reported_so_a_rejection_is_actionable(route):
    """A bare False on a route someone spent a listening session earning is a
    support ticket. Every failure names the field."""
    route["qualification_record"]["status"] = "candidate"
    route["qualification_record"]["runtime"]["model_id"] = ""
    v = _check(route)
    assert not v.ok
    assert len(v.reasons) >= 2
    assert v.summary and "route OK" not in v.summary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_parse_utc_accepts_the_Z_spelling_and_rejects_naive(route):
    assert parse_utc("2026-08-10T20:37:17Z") == datetime(
        2026, 8, 10, 20, 37, 17, tzinfo=timezone.utc)
    assert parse_utc("2026-08-10T20:37:17") is None      # naive: no zone guess
    assert parse_utc("nonsense") is None
    assert parse_utc(None) is None


def test_sha256_of_file_returns_none_instead_of_raising(tmp_path, local_ref):
    path, sha = local_ref
    assert sha256_of_file(path) == sha
    assert sha256_of_file(str(tmp_path / "absent.bin")) is None


def test_the_legacy_helper_cannot_authorize_a_route():
    """`is_qualified_route` stays as a compatibility check, but it accepts a
    receipt whose file does not exist -- which is exactly why authorization
    moved here."""
    from config.cast_pools import QUALIFICATION_RECEIPT_REQUIRED_FIELDS, is_qualified_route
    legacy = {"engine": "indextts2",
              "qualification_receipt": {f: "x" for f in
                                        QUALIFICATION_RECEIPT_REQUIRED_FIELDS}}
    assert is_qualified_route(legacy)                      # legacy says yes...
    assert not validate_qualified_voice_route(legacy, NOW).ok   # ...this does not
