"""Wave 1 / 1a -- OTR_BatchCharacterVoices generic character-voice node.

Headless (CUDA masked by tests/conftest.py). The byte-identical batch path is
exercised by stubbing the legacy delegate with a node whose class NAME matches
the legacy ``BatchBarkGenerator`` so the frozen-manifest widget lookup resolves,
then asserting verbatim delegation (exact string identity + frozen widget tuple)
and passthrough sha equality. The per-line + fail-closed paths are exercised
without any engine library installed.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "batch_character_voices.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


# ----------------------------------------------------------------------------
# Per-line bark stub. The audio clean-break (1a) flipped bark to a per_line
# registry engine, so OTR_BatchCharacterVoices no longer delegates to a batch
# node -- it calls adapter.generate_voice per dialogue line. We stub that one
# method so the dispatch contract (voice_ref routing, packing, gate, teardown)
# runs without loading Bark (CUDA masked by conftest).
# ----------------------------------------------------------------------------
def _stub_bark_generate_voice(monkeypatch, recorder):
    """Point the singleton bark adapter's per-line generate_voice at a stub."""
    from nodes._otr_audio_engines import get_engine

    bark = get_engine("bark")

    def _gen(text, voice_ref, delivery_vector, seed):
        recorder.append({"text": text, "voice_ref": voice_ref, "seed": seed})
        return {"waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
                "sample_rate": 24000}

    monkeypatch.setattr(bark, "generate_voice", _gen)
    return bark


def _one_char_line_ledger(preset="v2/en_speaker_3"):
    """Minimal L3 ledger: one character line whose cast row carries the preset."""
    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [{"char_id": "c01", "name": "LEMMY", "voice_preset": preset}],
        "lines": [{"line_id": "l1", "char_id": "c01",
                   "speaker_role": "character", "text": "Hello there."}],
        "meta": {"episode_seed": "seed-1"},
    })


def _serialized_slots(input_types: dict) -> list:
    """Widget slots that occupy widgets_values (forceInput sockets excluded)."""
    names = []
    for bucket in ("required", "optional"):
        for name, spec in (input_types.get(bucket) or {}).items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            is_widget = isinstance(t, (list, tuple)) or (
                isinstance(t, str) and t.upper() in _WIDGET_TYPES
            )
            opts = (
                spec[1]
                if isinstance(spec, (list, tuple)) and len(spec) > 1
                and isinstance(spec[1], dict)
                else {}
            )
            if not is_widget or opts.get("forceInput"):
                continue
            names.append(name)
    return names


# ----------------------------------------------------------------------------
# INPUT_TYPES / widget-vector contract
# ----------------------------------------------------------------------------
def test_input_types_widget_vector_exact():
    from nodes.batch_character_voices import BatchCharacterVoices as B

    it = B.INPUT_TYPES()
    for key in ("script_json", "ledger_json", "gate_in"):
        spec = (it.get("required", {}).get(key) or it.get("optional", {}).get(key))
        assert spec is not None, f"{key} missing from INPUT_TYPES"
        assert spec[1].get("forceInput") is True, f"{key} must be forceInput"
    all_keys = set(it.get("required", {})) | set(it.get("optional", {}))
    assert "seed" not in all_keys, "no input may be named 'seed' (D)"
    assert "gate_in" in it.get("optional", {})
    # Only engine serializes as a widget (forceInputs are sockets). stereo_policy
    # surface removed 2026-07-04 (widget-audit Batch 1); single option "mono_safe"
    # -- the generate() kwarg still defaults to "mono_safe".
    assert _serialized_slots(it) == ["engine"]
    assert "done" in B.RETURN_NAMES


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.batch_character_voices import BatchCharacterVoices as B

    it = B.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    # google_tts (direct BYO API, dropdown-opt-in) APPENDED after elevenlabs;
    # index 0 stays the byte-identical indextts2.
    assert engines == [
        "indextts2", "chatterbox", "dia", "bark", "kokoro", "elevenlabs",
        "google_tts",
    ]
    assert it["required"]["engine"][1]["default"] == "indextts2"
    # Order is stable across opt-in flags.
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    monkeypatch.setenv("OTR_ENABLE_INDEXTTS2", "1")
    assert list(B.INPUT_TYPES()["required"]["engine"][0]) == engines


def test_input_types_safe_with_bad_configs(monkeypatch):
    import nodes._otr_engine_profiles as ep
    from nodes.batch_character_voices import BatchCharacterVoices as B

    def _boom(role):
        raise RuntimeError("profiles unavailable")

    monkeypatch.setattr(ep, "legacy_first_engines", _boom)
    it = B.INPUT_TYPES()  # must not raise (C-5)
    engines = list(it["required"]["engine"][0])
    # UPDATED 2026-08-16. This assertion used to pin a five-engine list, which
    # is what the tuple had drifted to: elevenlabs and google_tts were appended
    # to the profiles table and this hardcoded stand-in was never updated. The
    # test was pinning the drift, so it had become the bug's bodyguard -- a
    # degraded boot offered a dropdown that could not even represent a saved
    # graph using either engine, and the test said that was correct.
    #
    # The fallback's job is to STAND IN for the profiles list when the profiles
    # import raises, so equality with that list is the property worth pinning;
    # tests/test_tts_voice_preflight_matrix.py holds the two equal directly.
    assert engines == ["indextts2", "chatterbox", "dia", "bark", "kokoro",
                       "elevenlabs", "google_tts"]  # hardcoded fallback
    assert engines[0] == "indextts2", "index 0 is the byte-identical default"
    assert engines, "engine combo must never be empty (C-5)"


# ----------------------------------------------------------------------------
# Per-line bark dispatch (audio clean-break 1a: bark is per_line, not batch)
# ----------------------------------------------------------------------------
def test_bark_per_line_routes_voice_preset_into_ref_slot(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.batch_character_voices import BatchCharacterVoices

    calls = []
    _stub_bark_generate_voice(monkeypatch, calls)
    out = BatchCharacterVoices().generate(
        script_json=_one_char_line_ledger("v2/en_speaker_3"), engine="bark",
    )
    assert len(calls) == 1
    # appendix A: bark declares voice_ref_field="voice_preset", so the dispatch
    # routes cast.voice_preset (NOT a clip path) into the positional ref slot.
    assert calls[0]["voice_ref"] == "v2/en_speaker_3"
    assert_audio_batch_contract(out[0], where="test")
    assert out[2].startswith("char_voice:done")


def test_bark_per_line_packs_audio_contract(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.batch_character_voices import BatchCharacterVoices

    _stub_bark_generate_voice(monkeypatch, [])
    out = BatchCharacterVoices().generate(
        script_json=_one_char_line_ledger(), engine="bark",
    )
    audio = assert_audio_batch_contract(out[0], where="test")
    assert int(audio["waveform"].shape[0]) == 1  # one packed clip
    assert int(audio["sample_rate"]) == 24000
    assert isinstance(out, tuple) and len(out) == 3
    assert isinstance(out[1], str)


# ----------------------------------------------------------------------------
# Fail-closed dispatch taxonomy (C-6)
# ----------------------------------------------------------------------------
def test_dispatch_fails_closed_with_taxonomy(monkeypatch):
    from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
    from nodes.batch_character_voices import BatchCharacterVoices

    node = BatchCharacterVoices()
    script = '{"lines": [], "cast": [], "meta": {}}'

    with pytest.raises(EngineUnusable) as unknown:
        node.generate(script_json=script, engine="no_such_engine")
    assert unknown.value.reason is EngineUsabilityReason.MALFORMED_CONFIG

    with pytest.raises(EngineUnusable) as wrong_role:
        node.generate(script_json=script, engine="stable_audio_3")  # music-only, not a char voice
    assert wrong_role.value.reason is EngineUsabilityReason.INCOMPATIBLE_PROFILE

    # C6: no GATED_BY_FLAG dispatch case any more -- a registered engine is
    # selectable (the registry IS the menu). The fail-closed taxonomy here is
    # MALFORMED_CONFIG (unknown engine) + INCOMPATIBLE_PROFILE (wrong role) above.


# ----------------------------------------------------------------------------
# Gate + empty-batch behavior (E.5 / C-4)
# ----------------------------------------------------------------------------
def test_zero_line_role_still_emits_gate_and_empty_batch(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")  # per-line engine selectable
    from nodes.batch_character_voices import BatchCharacterVoices

    script = '{"lines": [], "cast": [], "meta": {}}'
    audio, log_str, done = BatchCharacterVoices().generate(
        script_json=script, engine="chatterbox",
    )
    assert tuple(audio["waveform"].shape) == (1, 1, 0)  # canonical empty batch
    assert isinstance(log_str, str)
    assert done.startswith("char_voice:done"), "done must always be emitted"


def test_gate_in_consumed_without_crashing(monkeypatch):
    from nodes.batch_character_voices import BatchCharacterVoices

    _stub_bark_generate_voice(monkeypatch, [])
    out = BatchCharacterVoices().generate(
        script_json=_one_char_line_ledger(), engine="bark",
        gate_in="upstream:done:engine=x:clips=3",
    )
    # gate_in creates the ordering edge only; done is this node's own sentinel.
    assert out[2].startswith("char_voice:done")


# ----------------------------------------------------------------------------
# Registration + lazy-import contract (C-5 / piece 6)
# ----------------------------------------------------------------------------
def test_node_class_matches_class_registry():
    from nodes._otr_class_registry import NEW_NODE_SPECS, expected_category
    from nodes.batch_character_voices import BatchCharacterVoices

    spec = NEW_NODE_SPECS["OTR_BatchCharacterVoices"]
    assert spec.class_name == "BatchCharacterVoices"
    assert BatchCharacterVoices.CATEGORY == expected_category("OTR_BatchCharacterVoices")
    assert BatchCharacterVoices.FUNCTION == "generate"


def test_node_wired_into_init_by_table_not_literal_key():
    text = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "new_node_modules_table" in text, "init must merge the audio table"
    literal_keys = set(re.findall(r'"(OTR_\w+)"\s*:', text))
    assert "OTR_BatchCharacterVoices" not in literal_keys, (
        "new audio node must be merged via the table, not a literal key"
    )


def test_importing_node_triggers_no_engine_lib_imports():
    import importlib
    import sys

    importlib.import_module("nodes.batch_character_voices")
    for lib in ("chatterbox", "indextts", "stable_audio_tools"):
        assert lib not in sys.modules, f"engine lib {lib} imported at node import"


def test_source_is_ascii_no_em_dash():
    src = NODE_SRC.read_text(encoding="utf-8")
    assert "—" not in src, "em-dash forbidden in OTR python source (CLAUDE.md)"
    src.encode("ascii")  # ASCII-only source (cp1252 subprocess decode safety)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
