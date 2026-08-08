"""Tests for the audio-cache PROTOCOL + canonical record (piece 5)."""
import dataclasses

import pytest

from nodes._otr_audio_cache import (
    CACHE_SCHEMA_VERSION,
    AudioCache,
    AudioCacheRecord,
    cache_key_for,
    record_from_request,
)
from nodes._otr_resolved_request import build_resolved_request


def _req(**over):
    base = dict(
        role="char_voice",
        engine_name="bark",
        engine_profile_id="char_bark_v1",
        engine_impl_version="1",
        char_id="C1",
        line_id="L1",
        prepared_text="hello there",
        commercial_clean=True,
    )
    base.update(over)
    return build_resolved_request(**base)


def test_schema_version_present():
    # Bumped to "2" in the cloud-audio-cache chunk (2026-08-08) when
    # AudioCacheRecord gained actual_sample_rate + provider_model_id.
    assert CACHE_SCHEMA_VERSION == "2"


def test_record_is_frozen():
    rec = AudioCacheRecord(cache_key="abc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.cache_key = "xyz"  # type: ignore[misc]


def test_record_roundtrip_dict():
    rec = AudioCacheRecord(
        cache_key="k", engine_name="bark", sample_rate=24000,
        commercial_clean=True, allowed_for_release=True,
    )
    d = rec.to_dict()
    assert d["cache_key"] == "k" and d["sample_rate"] == 24000
    assert AudioCacheRecord.from_dict(d) == rec


def test_from_dict_ignores_unknown_keys():
    rec = AudioCacheRecord.from_dict(
        {"cache_key": "k", "totally_new_future_field": 99}
    )
    assert rec.cache_key == "k"


def test_from_dict_requires_cache_key():
    with pytest.raises(ValueError):
        AudioCacheRecord.from_dict({"engine_name": "bark"})


def test_cache_key_is_request_identity_key():
    r = _req()
    assert cache_key_for(r) == r.cache_key


def test_record_from_request_maps_identity_fields():
    r = _req(voice_ref_id="vr_42")
    rec = record_from_request(
        r, audio_path="/x/a.wav", audio_sha256="deadbeef",
        allowed_for_release=True,
    )
    assert rec.cache_key == r.cache_key
    assert rec.engine_name == "bark"
    assert rec.role == "char_voice"
    assert rec.voice_ref_id == "vr_42"
    assert rec.commercial_clean is True
    assert rec.allowed_for_release is True
    assert rec.audio_path == "/x/a.wav"
    assert rec.request_schema_version == r.request_schema_version


def test_complete_stub_satisfies_protocol():
    class _Cache:
        def key_for(self, request):
            return cache_key_for(request)

        def has(self, request):
            return False

        def get(self, request):
            return None

        def put(self, request, audio, *, allowed_for_release=False,
                actual_sample_rate=None, provider_model_id=""):
            return record_from_request(
                request, allowed_for_release=allowed_for_release,
                actual_sample_rate=actual_sample_rate,
                provider_model_id=provider_model_id,
            )

        def load(self, request):
            return None

        def iter_records(self):
            return []

    assert isinstance(_Cache(), AudioCache)


def test_incomplete_stub_does_not_satisfy_protocol():
    class _Partial:
        def has(self, request):
            return False

    assert not isinstance(_Partial(), AudioCache)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
