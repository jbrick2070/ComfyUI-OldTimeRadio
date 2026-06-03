"""Wave 1 / 1f -- FileAudioCache implementation + slim migration (G0).

Headless. Exercises key derivation, sidecar round-trip, the schema-drift
re-render rule, the release manifest scan, and the read-only ledger migration
checks (which must never rewrite a legacy raw-delegation script).
"""
from __future__ import annotations

import copy
import os
import pathlib

import pytest
import torch

from nodes._otr_audio_cache import (
    AudioCacheRecord,
    CacheMigrationError,
    FileAudioCache,
    assert_registry_ledger_has_voice_ref_id,
    cache_key_for,
    detect_ledger_schema_version,
    needs_rerender,
    record_from_request,
)
from nodes._otr_resolved_request import build_resolved_request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _req(**kw):
    base = dict(
        role="char_voice", engine_name="chatterbox",
        engine_profile_id="char_chatterbox_v1", engine_impl_version="1",
        char_id="c1", line_id="l1", prepared_text="hello world",
        voice_ref_id="cc_male_warm", commercial_clean=True,
        sample_rate=24000, channels=1,
    )
    base.update(kw)
    return build_resolved_request(**base)


def _audio(n=16):
    return {"waveform": torch.zeros(1, 1, n), "sample_rate": 24000}


# ----------------------------------------------------------------------------
# Cache key completeness + record round-trip
# ----------------------------------------------------------------------------
def test_cache_key_reflects_identity_fields():
    base_key = _req().cache_key
    assert cache_key_for(_req()) == base_key  # single keying rule (I-6)
    mutations = [
        ("char_id", "c2"), ("line_id", "l2"), ("engine_name", "indextts2"),
        ("engine_profile_id", "char_indextts2_v1"), ("voice_ref_id", "other"),
        ("prepared_text", "different text"), ("role", "announcer_voice"),
        ("sample_rate", 22050), ("occurrence", 3),
    ]
    for field, val in mutations:
        assert cache_key_for(_req(**{field: val})) != base_key, field


def test_record_roundtrips_through_dict():
    rec = record_from_request(
        _req(), audio_path="a.npy", audio_sha256="deadbeef",
        allowed_for_release=True,
    )
    assert AudioCacheRecord.from_dict(rec.to_dict()) == rec
    assert rec.cache_key == _req().cache_key


# ----------------------------------------------------------------------------
# Read / write round-trip
# ----------------------------------------------------------------------------
def test_put_get_roundtrip(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    assert cache.has(req) is False
    rec = cache.put(req, _audio(), allowed_for_release=True)
    assert cache.has(req) is True
    got = cache.get(req)
    assert got is not None
    assert got.cache_key == req.cache_key
    assert got.audio_sha256 == rec.audio_sha256
    assert os.path.exists(rec.audio_path)
    assert got.allowed_for_release is True


def test_get_miss_returns_none(tmp_path):
    assert FileAudioCache(str(tmp_path)).get(_req()) is None


def test_construct_does_no_io(tmp_path):
    target = tmp_path / "not_created_yet"
    FileAudioCache(str(target))  # constructing must not create the dir (C-5)
    assert not target.exists()


# ----------------------------------------------------------------------------
# Slim migration: schema drift -> re-render
# ----------------------------------------------------------------------------
def test_version_drift_forces_rerender(tmp_path):
    writer = FileAudioCache(str(tmp_path))
    req = _req()
    writer.put(req, _audio(8))
    rec = writer.get(req)
    assert rec is not None
    assert needs_rerender(rec) is False
    assert needs_rerender(rec, target_request_schema_version="999") is True
    # A reader targeting a different schema sees a miss (re-render).
    reader = FileAudioCache(str(tmp_path), request_schema_version="999")
    assert reader.get(req) is None


# ----------------------------------------------------------------------------
# Release manifest scan
# ----------------------------------------------------------------------------
def test_iter_and_releasable_records(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    a, b = _req(char_id="a"), _req(char_id="b")
    cache.put(a, _audio(4), allowed_for_release=True)
    cache.put(b, _audio(4), allowed_for_release=False)
    assert len(list(cache.iter_records())) == 2
    releasable = cache.releasable_records()
    assert [r.cache_key for r in releasable] == [a.cache_key]


def test_iter_records_on_missing_dir_is_empty(tmp_path):
    cache = FileAudioCache(str(tmp_path / "absent"))
    assert list(cache.iter_records()) == []


# ----------------------------------------------------------------------------
# Ledger migration checks (read-only)
# ----------------------------------------------------------------------------
def test_old_registry_ledger_missing_voice_ref_id_fails():
    ledger = {
        "meta": {"cast_lock_revision": 1},
        "cast": [{"char_id": "c1", "voice_preset": "v2/en_speaker_3"}],
    }
    with pytest.raises(CacheMigrationError):
        assert_registry_ledger_has_voice_ref_id(ledger)


def test_cast_locked_ledger_with_voice_ref_id_passes():
    ledger = {
        "meta": {"cast_lock_revision": 2},
        "cast": [{"char_id": "c1", "voice_ref_id": "cc_male_warm"}],
    }
    assert_registry_ledger_has_voice_ref_id(ledger)  # no raise


def test_legacy_ledger_without_cast_lock_is_left_alone():
    ledger = {"meta": {}, "cast": [{"char_id": "c1", "voice_preset": "v2/en_speaker_3"}]}
    assert_registry_ledger_has_voice_ref_id(ledger)  # legacy path: no enforcement


def test_detect_ledger_schema_version():
    assert detect_ledger_schema_version({"meta": {"ledger_schema_version": "2"}}) == "2"
    assert detect_ledger_schema_version({"meta": {"schema_version": "3"}}) == "3"
    assert detect_ledger_schema_version({"meta": {}}) == "1"
    assert detect_ledger_schema_version({}) == "1"


def test_migration_does_not_rewrite_legacy_raw_script():
    ledger = {
        "meta": {"cast_lock_revision": 1},
        "cast": [{"char_id": "c1", "voice_ref_id": "x"}],
        "lines": [{"line_id": "l1", "text": "the raw script must not change"}],
    }
    snapshot = copy.deepcopy(ledger)
    detect_ledger_schema_version(ledger)
    assert_registry_ledger_has_voice_ref_id(ledger)
    assert ledger == snapshot  # read-only: no mutation of the delegated script


def test_source_is_ascii_no_em_dash():
    src = (REPO_ROOT / "nodes" / "_otr_audio_cache.py").read_text(encoding="utf-8")
    assert "—" not in src
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
