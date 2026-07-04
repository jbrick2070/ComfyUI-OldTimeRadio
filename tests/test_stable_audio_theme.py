"""Wave 1 / 1c -- OTR_StableAudioTheme generic theme-music node.

Headless (CUDA masked by tests/conftest.py). The byte-identical batch path is
exercised by stubbing the legacy delegate with a node whose class NAME matches
the legacy ``MusicGenTheme`` (three AUDIO outputs), asserting verbatim
delegation + per-cue passthrough sha. The fail-closed taxonomy includes the
HF-token gate for the opt-in Stable Audio engine.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NODE_SRC = REPO_ROOT / "nodes" / "stable_audio_theme.py"

_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})


def _stub_musicgen_generate_clip(monkeypatch, recorder):
    """Stub the per-cue clip generate_clip so the theme dispatch runs without
    transformers / GPU. Audio clean-break 1c: musicgen is a clip engine."""
    from nodes._otr_audio_engines import get_engine

    musicgen = get_engine("musicgen")

    def _gen(prompt, duration_s, seed):
        recorder.append({"prompt": prompt, "duration_s": duration_s, "seed": seed})
        return {"waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
                "sample_rate": 32000}

    monkeypatch.setattr(musicgen, "generate_clip", _gen)
    return musicgen


def _ledger(meta=None):
    import json

    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [], "lines": [], "meta": meta or {},
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
    from nodes.stable_audio_theme import StableAudioTheme as T

    it = T.INPUT_TYPES()
    for key in ("script_json", "ledger_json", "gate_in"):
        spec = (it.get("required", {}).get(key) or it.get("optional", {}).get(key))
        assert spec is not None, f"{key} missing"
        assert spec[1].get("forceInput") is True, f"{key} must be forceInput"
    all_keys = set(it.get("required", {})) | set(it.get("optional", {}))
    assert "seed" not in all_keys
    assert "gate_in" in it.get("optional", {})
    # stereo_policy surface removed 2026-07-04 (widget-audit Batch 1); single
    # option "mono_safe" -- the generate() kwarg still defaults to "mono_safe".
    assert _serialized_slots(it) == ["engine"]
    for name in ("opening_theme_audio", "closing_theme_audio",
                 "interstitial_theme_audio", "done"):
        assert name in T.RETURN_NAMES


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.stable_audio_theme import StableAudioTheme as T

    it = T.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    # PROMOTED 2026-06-03: SA3 is index 0 = the shipped music default.
    # sonilo (cloud, dropdown-opt-in) APPENDED 2026-07-03 -- selectable but never
    # the default; index 0 stays the byte-identical stable_audio_3.
    assert engines == ["stable_audio_3", "musicgen", "stable_audio_music", "sonilo"]
    assert it["required"]["engine"][1]["default"] == "stable_audio_3"
    monkeypatch.setenv("OTR_ENABLE_STABLE_AUDIO", "1")
    assert list(T.INPUT_TYPES()["required"]["engine"][0]) == engines


def test_input_types_safe_with_bad_configs(monkeypatch):
    import nodes._otr_engine_profiles as ep
    from nodes.stable_audio_theme import StableAudioTheme as T

    def _boom(role):
        raise RuntimeError("profiles unavailable")

    monkeypatch.setattr(ep, "legacy_first_engines", _boom)
    engines = list(T.INPUT_TYPES()["required"]["engine"][0])
    assert engines == ["musicgen", "stable_audio_music"]
    assert engines


def test_musicgen_clip_renders_three_cues(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []
    _stub_musicgen_generate_clip(monkeypatch, calls)
    out = StableAudioTheme().generate(script_json=_ledger(), engine="musicgen")
    assert isinstance(out, tuple) and len(out) == 5
    assert len(calls) == 3  # opening / closing / interstitial via _render_clips
    for i in range(3):
        assert_audio_batch_contract(out[i], where=f"test.cue{i}")
    assert isinstance(out[3], str)
    assert out[4].startswith("music:done")


def test_musicgen_clip_prompt_from_meta_brief(monkeypatch):
    """The cue prompt is composed from the Meta brief (clean-break 1c)."""
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []
    _stub_musicgen_generate_clip(monkeypatch, calls)
    meta = {
        "story_brief_status": "ok",
        "story_brief_terms": {"setting": ["dim room"], "atmosphere": ["tense"]},
        "music_mood_terms": ["sombre", "uneasy"],
    }
    StableAudioTheme().generate(script_json=_ledger(meta), engine="musicgen")
    opening = calls[0]["prompt"]
    assert "sombre" in opening
    assert opening.endswith("instrumental only, no dialogue, no vocals")


def test_dispatch_fails_closed_with_taxonomy(monkeypatch):
    from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
    from nodes.stable_audio_theme import StableAudioTheme

    node = StableAudioTheme()
    script = '{"meta": {}}'

    with pytest.raises(EngineUnusable) as unknown:
        node.generate(script_json=script, engine="no_such_engine")
    assert unknown.value.reason is EngineUsabilityReason.MALFORMED_CONFIG

    with pytest.raises(EngineUnusable) as wrong_role:
        node.generate(script_json=script, engine="bark")  # char-voice only
    assert wrong_role.value.reason is EngineUsabilityReason.INCOMPATIBLE_PROFILE

    # C6: no GATED_BY_FLAG dispatch case any more -- a registered engine is
    # selectable (the registry IS the menu). The fail-closed taxonomy here is
    # MALFORMED_CONFIG (unknown engine) + INCOMPATIBLE_PROFILE (wrong role) above.


def test_stable_audio_missing_hf_token_fails_closed(monkeypatch):
    """Flag on but no HF_TOKEN -> MISSING_HF_TOKEN on the generate path (C-5)."""
    from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
    from nodes.stable_audio_theme import StableAudioTheme

    monkeypatch.setenv("OTR_ENABLE_STABLE_AUDIO", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(EngineUnusable) as missing:
        StableAudioTheme().generate(
            script_json='{"meta": {}}', engine="stable_audio_music",
        )
    assert missing.value.reason is EngineUsabilityReason.MISSING_HF_TOKEN


def test_node_class_matches_class_registry():
    from nodes._otr_class_registry import NEW_NODE_SPECS, expected_category
    from nodes.stable_audio_theme import StableAudioTheme

    spec = NEW_NODE_SPECS["OTR_StableAudioTheme"]
    assert spec.class_name == "StableAudioTheme"
    assert StableAudioTheme.CATEGORY == expected_category("OTR_StableAudioTheme")
    assert StableAudioTheme.FUNCTION == "generate"


def test_node_wired_into_init_by_table_not_literal_key():
    text = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "new_node_modules_table" in text
    literal_keys = set(re.findall(r'"(OTR_\w+)"\s*:', text))
    assert "OTR_StableAudioTheme" not in literal_keys


def test_importing_node_triggers_no_engine_lib_imports():
    import importlib
    import sys

    importlib.import_module("nodes.stable_audio_theme")
    for lib in ("stable_audio_tools", "audiocraft", "chatterbox"):
        assert lib not in sys.modules, f"engine lib {lib} imported at node import"


def test_source_is_ascii_no_em_dash():
    src = NODE_SRC.read_text(encoding="utf-8")
    assert "—" not in src, "em-dash forbidden in OTR python source (CLAUDE.md)"
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
