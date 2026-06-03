"""Wave 1 / 1b -- OTR_AnnouncerVoice generic announcer-voice node.

Headless (CUDA masked by tests/conftest.py). The byte-identical batch path is
exercised by stubbing the legacy delegate with a node whose class NAME matches
the legacy ``KokoroAnnouncer`` so the frozen-manifest widget lookup resolves,
then asserting verbatim delegation + passthrough sha equality. Fail-closed and
zero-line paths run with no engine library installed.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "announcer_voice.py"
COMMON_SRC = REPO_ROOT / "nodes" / "_otr_voice_node_common.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


class KokoroAnnouncer:  # noqa: N801 -- intentional name-mirror for the manifest lookup
    FUNCTION = "render"

    def __init__(self):
        self.calls = []

    def render(self, script_json, episode_seed="", voice_override="random", speed=0.95):
        self.calls.append((script_json, episode_seed, voice_override, speed))
        h = hashlib.sha256(
            f"{script_json}|{episode_seed}|{voice_override}|{speed}".encode("utf-8")
        ).digest()
        wf = torch.tensor([b / 255.0 for b in h[:8]], dtype=torch.float32).reshape(1, 1, 8)
        return ({"waveform": wf, "sample_rate": 24000}, "stub render log", "bm_george")


def _install_stub_kokoro(monkeypatch):
    from nodes._otr_audio_engines import get_engine

    kokoro = get_engine("kokoro")
    stub = KokoroAnnouncer()
    monkeypatch.setattr(kokoro, "make_batch_node", lambda: stub)
    return stub


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
    assert _serialized_slots(it) == ["engine", "stereo_policy"]
    assert "done" in A.RETURN_NAMES


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.announcer_voice import AnnouncerVoice as A

    it = A.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    assert engines == ["kokoro", "chatterbox"]
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


def test_no_mutation_exact_string_and_frozen_widgets(monkeypatch):
    from nodes.announcer_voice import AnnouncerVoice

    stub = _install_stub_kokoro(monkeypatch)
    script = '{"lines": [{"line_id": "a1", "speaker_role": "announcer"}], "meta": {}}'
    AnnouncerVoice().generate(script_json=script, engine="kokoro")
    assert len(stub.calls) == 1
    passed_json, episode_seed, voice_override, speed = stub.calls[0]
    assert passed_json is script, "script_json must pass through verbatim (I-3)"
    # Frozen manifest widget tuple for OTR_KokoroAnnouncer (sans script_json).
    assert (episode_seed, voice_override, speed) == ("", "random", 0.95)


def test_golden_legacy_direct_vs_wrapper_sha(monkeypatch):
    from nodes._otr_audio_utils import audio_sha16
    from nodes.announcer_voice import AnnouncerVoice

    stub = _install_stub_kokoro(monkeypatch)
    script = '{"lines": [], "cast": [], "meta": {}}'
    direct_audio, _log, _voice = stub.render(script, "", "random", 0.95)
    out = AnnouncerVoice().generate(script_json=script, engine="kokoro")
    assert audio_sha16(out[0]) == audio_sha16(direct_audio)


def test_execute_stub_returns_audio_contract(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.announcer_voice import AnnouncerVoice

    _install_stub_kokoro(monkeypatch)
    out = AnnouncerVoice().generate(script_json='{"x": 1}', engine="kokoro")
    assert isinstance(out, tuple) and len(out) == 3
    assert_audio_batch_contract(out[0], where="test")
    assert out[2].startswith("announcer:done")


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

    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    with pytest.raises(EngineUnusable) as gated:
        node.generate(script_json=script, engine="chatterbox")
    assert gated.value.reason is EngineUsabilityReason.GATED_BY_FLAG


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
