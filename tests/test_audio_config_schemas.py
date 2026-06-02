"""Tests for the Wave-0 voice-bank + cache-sidecar JSON schemas (piece 7).

Dependency-free: a tiny JSON-Schema-subset validator checks the contract
(required keys + declared types) so the suite needs no jsonschema package.
"""
import dataclasses
import json
import pathlib

import pytest

from nodes._otr_audio_cache import AudioCacheRecord, record_from_request
from nodes._otr_resolved_request import build_resolved_request

CONFIG = pathlib.Path(__file__).resolve().parent.parent / "config"
BANK_SCHEMA = CONFIG / "voice_bank_entry_schema.json"
SIDECAR_SCHEMA = CONFIG / "audio_cache_sidecar_schema.json"


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _type_ok(value, decl):
    types = decl if isinstance(decl, list) else [decl]
    for t in types:
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "null" and value is None:
            return True
    return False


def _validate(instance, schema):
    """Minimal validator: required presence + property type (+ array item type)."""
    for req in schema.get("required", []):
        if req not in instance:
            raise AssertionError(f"missing required key: {req}")
    props = schema.get("properties", {})
    for key, value in instance.items():
        spec = props.get(key)
        if not spec or "type" not in spec:
            continue
        if not _type_ok(value, spec["type"]):
            raise AssertionError(f"{key}={value!r} not of type {spec['type']}")
        if spec["type"] == "array" and isinstance(value, list):
            item_spec = spec.get("items", {})
            if "type" in item_spec:
                for it in value:
                    if not _type_ok(it, item_spec["type"]):
                        raise AssertionError(f"{key} item {it!r} bad type")
    return True


def test_schemas_are_valid_json_and_well_formed():
    for p in (BANK_SCHEMA, SIDECAR_SCHEMA):
        schema = _load(p)
        assert schema["$schema"].startswith("https://json-schema.org/")
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict) and schema["properties"]
        assert set(schema["required"]) <= set(schema["properties"]), p.name


def test_bank_schema_has_e1_fields():
    schema = _load(BANK_SCHEMA)
    e1 = {
        "voice_ref_id", "engine", "gender", "timbre", "roles",
        "age_band", "ref_path", "ref_sha256", "commercial_clean",
    }
    assert e1 <= set(schema["properties"])
    assert e1 <= set(schema["required"])


def test_bank_sample_validates_and_bad_sample_fails():
    schema = _load(BANK_SCHEMA)
    good = {
        "voice_ref_id": "vr_narrator_a",
        "engine": "chatterbox",
        "gender": "male",
        "timbre": ["warm", "low"],
        "roles": ["char_voice", "announcer_voice"],
        "age_band": "adult",
        "ref_path": "voices/narrator_a.wav",
        "ref_sha256": "0" * 64,
        "commercial_clean": True,
    }
    assert _validate(good, schema)

    missing_commercial = dict(good)
    missing_commercial.pop("commercial_clean")
    with pytest.raises(AssertionError):
        _validate(missing_commercial, schema)

    wrong_type = dict(good, timbre="warm")  # should be an array
    with pytest.raises(AssertionError):
        _validate(wrong_type, schema)


def test_sidecar_schema_mirrors_audio_cache_record():
    schema = _load(SIDECAR_SCHEMA)
    record_fields = {f.name for f in dataclasses.fields(AudioCacheRecord)}
    assert set(schema["properties"]) == record_fields, (
        "cache sidecar schema drifted from AudioCacheRecord"
    )
    assert schema["required"] == ["cache_key"]


def test_real_record_to_dict_validates_against_sidecar_schema():
    schema = _load(SIDECAR_SCHEMA)
    req = build_resolved_request(
        role="char_voice", engine_name="bark", engine_profile_id="char_bark_v1",
        engine_impl_version="1", char_id="C1", line_id="L1",
        prepared_text="line one", commercial_clean=True,
    )
    rec = record_from_request(
        req, audio_path="/x/a.wav", audio_sha256="ab" * 32,
        allowed_for_release=True,
    )
    assert _validate(rec.to_dict(), schema)
    # A null voice_ref_id (the common char-voice case) must validate too.
    rec2 = AudioCacheRecord(cache_key="k", voice_ref_id=None, commercial_clean=None)
    assert _validate(rec2.to_dict(), schema)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
