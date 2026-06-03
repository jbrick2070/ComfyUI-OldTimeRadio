"""Wave 1 / 1a -- OTR_BatchCharacterVoices generic character-voice node.

Headless (CUDA masked by tests/conftest.py). The byte-identical batch path is
exercised by stubbing the legacy delegate with a node whose class NAME matches
the legacy ``BatchBarkGenerator`` so the frozen-manifest widget lookup resolves,
then asserting verbatim delegation (exact string identity + frozen widget tuple)
and passthrough sha equality. The per-line + fail-closed paths are exercised
without any engine library installed.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "batch_character_voices.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


# ----------------------------------------------------------------------------
# Stub legacy delegate. Class NAME mirrors the real legacy node so the
# config/legacy_invocation_manifest.json lookup in _frozen_batch_widgets
# resolves and returns the frozen [temperature, bypass_freeze_halt] tuple.
# ----------------------------------------------------------------------------
class BatchBarkGenerator:  # noqa: N801 -- intentional name-mirror for the manifest lookup
    FUNCTION = "generate_batch"

    def __init__(self):
        self.calls = []

    def generate_batch(self, script_json, temperature=0.7, bypass_freeze_halt=False):
        self.calls.append((script_json, temperature, bypass_freeze_halt))
        # Deterministic waveform derived from the EXACT args, so any mutation of
        # script_json or a different widget tuple would change the sha.
        h = hashlib.sha256(
            f"{script_json}|{temperature}|{bypass_freeze_halt}".encode("utf-8")
        ).digest()
        wf = torch.tensor([b / 255.0 for b in h[:8]], dtype=torch.float32).reshape(1, 1, 8)
        return ({"waveform": wf, "sample_rate": 24000}, "stub batch log")


def _install_stub_bark(monkeypatch):
    """Point the singleton bark adapter's make_batch_node at the stub above."""
    from nodes._otr_audio_engines import get_engine

    bark = get_engine("bark")
    stub = BatchBarkGenerator()
    monkeypatch.setattr(bark, "make_batch_node", lambda: stub)
    return stub


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
    # Only engine + stereo_policy serialize as widgets (forceInputs are sockets).
    assert _serialized_slots(it) == ["engine", "stereo_policy"]
    assert "done" in B.RETURN_NAMES


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.batch_character_voices import BatchCharacterVoices as B

    it = B.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    assert engines == ["bark", "chatterbox", "indextts2"]
    assert it["required"]["engine"][1]["default"] == "bark"
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
    assert engines == ["bark", "chatterbox", "indextts2"]  # hardcoded fallback
    assert engines, "engine combo must never be empty (C-5)"


# ----------------------------------------------------------------------------
# Byte-identical batch delegation (I-3)
# ----------------------------------------------------------------------------
def test_no_mutation_exact_string_and_frozen_widgets(monkeypatch):
    from nodes.batch_character_voices import BatchCharacterVoices

    stub = _install_stub_bark(monkeypatch)
    script = '{"lines": [{"line_id": "l1"}], "cast": [], "meta": {}}'
    BatchCharacterVoices().generate(script_json=script, engine="bark")
    assert len(stub.calls) == 1
    passed_json, temperature, bypass = stub.calls[0]
    assert passed_json is script, "script_json must pass through verbatim (I-3)"
    assert (temperature, bypass) == (0.7, False), "frozen manifest widget tuple"


def test_golden_legacy_direct_vs_wrapper_sha(monkeypatch):
    from nodes._otr_audio_utils import audio_sha16
    from nodes.batch_character_voices import BatchCharacterVoices

    stub = _install_stub_bark(monkeypatch)
    script = '{"lines": [], "cast": [], "meta": {}}'
    direct_audio, _ = stub.generate_batch(script, 0.7, False)
    out = BatchCharacterVoices().generate(script_json=script, engine="bark")
    assert audio_sha16(out[0]) == audio_sha16(direct_audio)


def test_execute_stub_returns_audio_contract(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.batch_character_voices import BatchCharacterVoices

    _install_stub_bark(monkeypatch)
    out = BatchCharacterVoices().generate(script_json='{"x": 1}', engine="bark")
    assert isinstance(out, tuple) and len(out) == 3
    assert_audio_batch_contract(out[0], where="test")
    assert isinstance(out[1], str)
    assert out[2].startswith("char_voice:done")


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
        node.generate(script_json=script, engine="kokoro")  # announcer-only
    assert wrong_role.value.reason is EngineUsabilityReason.INCOMPATIBLE_PROFILE

    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    with pytest.raises(EngineUnusable) as gated:
        node.generate(script_json=script, engine="chatterbox")
    assert gated.value.reason is EngineUsabilityReason.GATED_BY_FLAG


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

    _install_stub_bark(monkeypatch)
    out = BatchCharacterVoices().generate(
        script_json='{"x": 1}', engine="bark",
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
