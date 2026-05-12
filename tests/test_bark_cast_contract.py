"""tests/test_bark_cast_contract.py

P2 Gate 3 — BatchBarkGenerator must hard-raise on empty / non-v2/*
``cast.voice_preset``. No Director fallback, no
``_voice_preset_for_character`` helper, no production_plan_json
socket. Last line of defense after Gate 1 (writer) and Gate 2
(FreezeCascade Phase 0/10).
"""
from __future__ import annotations

import json

import pytest


def _ledger_with_preset(preset_value: str | None) -> str:
    """Build a minimal L3 ledger JSON with a single character line whose
    cast row carries the given voice_preset."""
    led = {
        "schema_version": "l3-2026-05-14",
        "cast": [
            {
                "char_id": "c01",
                "name": "LEMMY",
                "voice_preset": preset_value,
                "traits": "gruff",
                "tts_model": "bark",
            },
        ],
        "lines": [
            {
                "line_id": "b001",
                "char_id": "c01",
                "speaker_role": "character",
                "text": "Hello.",
                "traits": "gruff",
            },
        ],
        "meta": {"gen_params_initial": {"style": "closed_room_suspense"}},
    }
    return json.dumps(led)


def test_bark_has_no_production_plan_json_socket():
    """Voice-path cleanbreak: production_plan_json socket is gone from Bark."""
    from nodes.batch_bark_generator import BatchBarkGenerator
    schema = BatchBarkGenerator.INPUT_TYPES()
    required = set(schema.get("required", {}))
    optional = set(schema.get("optional", {}))
    assert "production_plan_json" not in required, (
        "production_plan_json must not be a required input"
    )
    assert "production_plan_json" not in optional, (
        "production_plan_json must not be an optional input"
    )


def test_bark_voice_preset_for_character_helper_is_gone():
    """The Director-derived helper must be deleted, not gated."""
    import nodes.batch_bark_generator as mod
    assert not hasattr(mod, "_voice_preset_for_character"), (
        "_voice_preset_for_character helper must be deleted (Standing Directive)"
    )
    # And any of its supporting state.
    for stale in ("_CHARACTER_VOICE_CACHE", "_FEMALE_PRESETS",
                  "_MALE_PRESETS", "_BARK_VOICE_PRESETS"):
        assert not hasattr(mod, stale), (
            f"{stale} must be deleted along with _voice_preset_for_character"
        )


def test_bark_raises_on_empty_voice_preset():
    """Empty preset is a writer contract violation."""
    from nodes.batch_bark_generator import BatchBarkGenerator
    node = BatchBarkGenerator()
    with pytest.raises(ValueError, match="voice_preset"):
        node.generate_batch(_ledger_with_preset(""))


def test_bark_raises_on_none_voice_preset():
    from nodes.batch_bark_generator import BatchBarkGenerator
    node = BatchBarkGenerator()
    with pytest.raises(ValueError, match="voice_preset"):
        node.generate_batch(_ledger_with_preset(None))


def test_bark_raises_on_non_v2_voice_preset():
    """Kokoro id on a Bark row is a contract violation."""
    from nodes.batch_bark_generator import BatchBarkGenerator
    node = BatchBarkGenerator()
    with pytest.raises(ValueError, match=r"voice_preset|v2/"):
        node.generate_batch(_ledger_with_preset("bm_george"))


def test_bark_signature_has_no_production_plan_json_kwarg():
    """Drop the legacy kwarg in lockstep with the socket deletion."""
    import inspect
    from nodes.batch_bark_generator import BatchBarkGenerator
    sig = inspect.signature(BatchBarkGenerator.generate_batch)
    assert "production_plan_json" not in sig.parameters, (
        "production_plan_json kwarg must be removed from generate_batch"
    )
