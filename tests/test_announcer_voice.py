"""Wave 1 / 1b -- OTR_AnnouncerVoice generic announcer-voice node.

Headless (CUDA masked by tests/conftest.py). After the audio clean-break 1b,
kokoro is a per_line engine, so the dispatch is exercised by stubbing the
per-line ``generate_voice`` (and the C-7 voice-file check) and asserting the
announcer per-line contract: episode-seeded voice pick, packing, gate, teardown.
Fail-closed and zero-line paths run with no engine library installed.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "announcer_voice.py"
COMMON_SRC = REPO_ROOT / "nodes" / "_otr_voice_node_common.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


def _stub_kokoro_generate_voice(monkeypatch, recorder):
    """Stub the per-line kokoro generate_voice and make begin_episode's C-7
    voice-file check pass, so the dispatch contract runs without kokoro / GPU."""
    from nodes._otr_audio_engines import eng_kokoro, get_engine

    # __file__ exists, so begin_episode's os.path.exists preflight passes.
    monkeypatch.setattr(eng_kokoro, "_kokoro_voice_path", lambda v: __file__)
    kokoro = get_engine("kokoro")

    def _gen(text, voice_ref, delivery_vector, seed):
        recorder.append({"text": text, "voice_ref": voice_ref, "seed": seed})
        return {"waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
                "sample_rate": 24000}

    monkeypatch.setattr(kokoro, "generate_voice", _gen)
    return kokoro


def _one_announcer_line_ledger(episode_seed="seed-1"):
    """Minimal L3 ledger: one announcer line + an announcer cast row."""
    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [{"char_id": "ann", "name": "ANNOUNCER",
                  "speaker_role": "announcer"}],
        "lines": [{"line_id": "a1", "char_id": "ann",
                   "speaker_role": "announcer", "text": "Tonight, on the air."}],
        "meta": {"episode_seed": episode_seed},
    })


def _serialized_slots(input_types: dict) -> list:
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


def test_input_types_widget_vector_exact():
    from nodes.announcer_voice import AnnouncerVoice as A

    it = A.INPUT_TYPES()
    for key in ("script_json", "ledger_json", "gate_in"):
        spec = (it.get("required", {}).get(key) or it.get("optional", {}).get(key))
        assert spec is not None, f"{key} missing from INPUT_TYPES"
        assert spec[1].get("forceInput") is True, f"{key} must be forceInput"
    all_keys = set(it.get("required", {})) | set(it.get("optional", {}))
    assert "seed" not in all_keys
    assert "gate_in" in it.get("optional", {})
    # stereo_policy surface removed 2026-07-04 (widget-audit Batch 1); single
    # option "mono_safe" -- the generate() kwarg still defaults to "mono_safe".
    assert _serialized_slots(it) == ["engine"]
    assert "done" in A.RETURN_NAMES


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.announcer_voice import AnnouncerVoice as A

    it = A.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    # google_tts (direct BYO API, dropdown-opt-in) APPENDED after elevenlabs;
    # index 0 stays the byte-identical kokoro.
    assert engines == ["kokoro", "chatterbox", "elevenlabs", "google_tts"]
    assert it["required"]["engine"][1]["default"] == "kokoro"
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    assert list(A.INPUT_TYPES()["required"]["engine"][0]) == engines


def test_input_types_safe_with_bad_configs(monkeypatch):
    import nodes._otr_engine_profiles as ep
    from nodes.announcer_voice import AnnouncerVoice as A

    def _boom(role):
        raise RuntimeError("profiles unavailable")

    monkeypatch.setattr(ep, "legacy_first_engines", _boom)
    engines = list(A.INPUT_TYPES()["required"]["engine"][0])
    assert engines == ["kokoro", "chatterbox"]
    assert engines


def test_kokoro_per_line_picks_episode_voice_and_packs(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.announcer_voice import AnnouncerVoice

    calls = []
    _stub_kokoro_generate_voice(monkeypatch, calls)
    out = AnnouncerVoice().generate(
        script_json=_one_announcer_line_ledger("seed-1"), engine="kokoro",
    )
    assert len(calls) == 1
    # begin_episode picks one pool voice (seeded); the cast has no voice_ref_id,
    # so the ref slot is None and eng_kokoro uses the episode pick internally.
    assert calls[0]["voice_ref"] is None
    assert_audio_batch_contract(out[0], where="test")
    assert out[2].startswith("announcer:done")


def test_kokoro_per_line_returns_audio_contract(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.announcer_voice import AnnouncerVoice

    _stub_kokoro_generate_voice(monkeypatch, [])
    out = AnnouncerVoice().generate(
        script_json=_one_announcer_line_ledger(), engine="kokoro",
    )
    assert isinstance(out, tuple) and len(out) == 3
    audio = assert_audio_batch_contract(out[0], where="test")
    assert int(audio["waveform"].shape[0]) == 1
    assert isinstance(out[1], str)
    assert out[2].startswith("announcer:done")


def test_kokoro_announcer_voice_pick_is_episode_seeded():
    """Deterministic per-episode pick from the curated pool -- preserves what
    listeners heard from the legacy node (same pool, same seed derivation)."""
    from nodes._otr_audio_engines.eng_kokoro import (
        ANNOUNCER_VOICE_POOL, _pick_announcer_voice,
    )

    a = _pick_announcer_voice("seed-1")
    b = _pick_announcer_voice("seed-1")
    assert a == b and a in ANNOUNCER_VOICE_POOL


def test_kokoro_begin_episode_c7_named_error_when_voice_absent(monkeypatch):
    """C-7: a missing voice file is a NAMED error at preflight, never a download."""
    from nodes._otr_audio_engines import eng_kokoro
    from nodes._otr_audio_engines.registry import (
        EngineUnusable, EngineUsabilityReason,
    )

    monkeypatch.setattr(eng_kokoro, "_kokoro_voice_path",
                        lambda v: "/otr/does/not/exist.pt")
    eng = eng_kokoro.KokoroEngine()
    with pytest.raises(EngineUnusable) as exc:
        eng.begin_episode({"episode_seed": "seed-1"})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL


def test_dispatch_fails_closed_with_taxonomy(monkeypatch):
    from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
    from nodes.announcer_voice import AnnouncerVoice

    node = AnnouncerVoice()
    script = '{"lines": [], "cast": [], "meta": {}}'

    with pytest.raises(EngineUnusable) as unknown:
        node.generate(script_json=script, engine="no_such_engine")
    assert unknown.value.reason is EngineUsabilityReason.MALFORMED_CONFIG

    with pytest.raises(EngineUnusable) as wrong_role:
        node.generate(script_json=script, engine="bark")  # char-voice only
    assert wrong_role.value.reason is EngineUsabilityReason.INCOMPATIBLE_PROFILE

    # C6: no GATED_BY_FLAG dispatch case any more -- a registered engine is
    # selectable (the registry IS the menu). The fail-closed taxonomy here is
    # MALFORMED_CONFIG (unknown engine) + INCOMPATIBLE_PROFILE (wrong role) above.


def test_zero_line_role_still_emits_gate_and_empty_batch(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    from nodes.announcer_voice import AnnouncerVoice

    script = '{"lines": [], "cast": [], "meta": {}}'
    audio, log_str, done = AnnouncerVoice().generate(
        script_json=script, engine="chatterbox",
    )
    assert tuple(audio["waveform"].shape) == (1, 1, 0)
    assert done.startswith("announcer:done")


def test_announcer_lines_only_routing(monkeypatch):
    """The per-line path must iterate announcer lines, not character lines."""
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    from nodes.announcer_voice import AnnouncerVoice

    # Only character lines present -> announcer node sees zero in-role lines.
    script = (
        '{"lines": [{"line_id": "c1", "speaker_role": "character", '
        '"text": "hi", "char_id": "x"}], "cast": [], "meta": {}}'
    )
    audio, _log, done = AnnouncerVoice().generate(
        script_json=script, engine="chatterbox",
    )
    assert tuple(audio["waveform"].shape) == (1, 1, 0)  # no announcer work
    assert done.startswith("announcer:done")


def test_node_class_matches_class_registry():
    from nodes._otr_class_registry import NEW_NODE_SPECS, expected_category
    from nodes.announcer_voice import AnnouncerVoice

    spec = NEW_NODE_SPECS["OTR_AnnouncerVoice"]
    assert spec.class_name == "AnnouncerVoice"
    assert AnnouncerVoice.CATEGORY == expected_category("OTR_AnnouncerVoice")
    assert AnnouncerVoice.FUNCTION == "generate"


def test_node_wired_into_init_by_table_not_literal_key():
    text = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "new_node_modules_table" in text
    literal_keys = set(re.findall(r'"(OTR_\w+)"\s*:', text))
    assert "OTR_AnnouncerVoice" not in literal_keys


def test_importing_node_triggers_no_engine_lib_imports():
    import importlib
    import sys

    importlib.import_module("nodes.announcer_voice")
    for lib in ("chatterbox", "indextts", "kokoro", "stable_audio_tools"):
        assert lib not in sys.modules, f"engine lib {lib} imported at node import"


def test_shares_voice_node_base():
    from nodes._otr_voice_node_common import OTRVoiceNodeBase
    from nodes.announcer_voice import AnnouncerVoice
    from nodes.batch_character_voices import BatchCharacterVoices

    assert issubclass(AnnouncerVoice, OTRVoiceNodeBase)
    assert issubclass(BatchCharacterVoices, OTRVoiceNodeBase)


def test_source_is_ascii_no_em_dash():
    for src_path in (NODE_SRC, COMMON_SRC):
        src = src_path.read_text(encoding="utf-8")
        assert "—" not in src, f"em-dash forbidden in {src_path.name}"
        src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
