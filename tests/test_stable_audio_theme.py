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


def _ledger(meta=None, lines=None):
    import json

    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [], "lines": lines or [], "meta": meta or {},
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
    # 720-bakeoff C3: the 3-AUDIO opening/closing/interstitial surface is
    # retired -- ONE cue batch + manifest + render_log + done.
    assert T.RETURN_NAMES == (
        "cue_audio_clips", "cue_manifest_json", "render_log", "done")
    assert T.RETURN_TYPES == ("AUDIO", "STRING", "STRING", "STRING")


def test_engine_dropdown_legacy_first_and_stable(monkeypatch):
    from nodes.stable_audio_theme import StableAudioTheme as T

    it = T.INPUT_TYPES()
    engines = list(it["required"]["engine"][0])
    # PROMOTED 2026-06-03: SA3 is index 0 = the shipped music default.
    # sonilo (cloud, dropdown-opt-in) and google_lyria (direct-api, BYO key) are
    # selectable but never defaults; index 0 stays byte-identical stable_audio_3.
    assert engines == [
        "stable_audio_3",
        "musicgen",
        "stable_audio_music",
        "sonilo",
        "google_lyria",
    ]
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


def test_musicgen_clip_renders_two_cues(monkeypatch):
    """Legacy lane (no ledger.music[]): SYNTHESIZE the 3 fixed cues into ONE
    padded batch + a manifest with one row per cue (720-bakeoff C3)."""
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes import _otr_cue_manifest as CM
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []
    _stub_musicgen_generate_clip(monkeypatch, calls)
    out = StableAudioTheme().generate(script_json=_ledger(), engine="musicgen")
    assert isinstance(out, tuple) and len(out) == 4
    assert len(calls) == 2  # opening / closing via _render_clips
    cue_audio, manifest_json, render_log, done = out
    assert_audio_batch_contract(cue_audio, where="test.cue_batch")
    assert int(cue_audio["waveform"].shape[0]) == 2   # 2 batch rows
    manifest = CM.parse_manifest(manifest_json, batch_size=2)
    assert manifest is not None
    placements = sorted(r["placement"] for r in manifest["cues"])
    assert placements == ["closing", "opening"]
    cue_ids = sorted(r["cue_id"] for r in manifest["cues"])
    assert cue_ids == ["closing", "opening"]
    # every row maps to a real batch index + positive sample_count
    for r in manifest["cues"]:
        assert 0 <= r["batch_index"] < 3
        assert r["sample_count"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", r["cue_spec_sha256"])
        assert r["anchor_line_id"] is None
    assert isinstance(render_log, str)
    assert done.startswith("music:done")
    assert "cues=2" in done


def test_legacy_synth_binds_ordered_music_sentinels(monkeypatch):
    """A legacy bank's synthesized manifest is enough to materialize a timed
    ledger: every available sentinel becomes the cue's durable anchor."""
    from nodes import _otr_cue_manifest as CM
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []
    _stub_musicgen_generate_clip(monkeypatch, calls)
    lines = [
        {"line_id": "b000", "speaker_role": "music_open"},
        {"line_id": "b005", "speaker_role": "music_inter"},
        {"line_id": "b900", "speaker_role": "music_close"},
    ]
    out = StableAudioTheme().generate(
        script_json=_ledger(lines=lines), engine="musicgen",
    )
    manifest = CM.parse_manifest(out[1], batch_size=2)
    by_cue = {r["cue_id"]: r for r in manifest["cues"]}

    assert by_cue["opening"]["anchor_line_id"] == "b000"
    assert by_cue["closing"]["anchor_line_id"] == "b900"
    assert all(row["cue_spec_sha256"] for row in by_cue.values())


@pytest.mark.parametrize(
    ("placement", "cue_id", "expected"),
    [
        ("open", "music_open", "opening"),
        ("music_open", "legacy", "opening"),
        ("close", "music_close", "closing"),
        ("music_close", "legacy", "closing"),
        ("inter", "music_inter", "interstitial"),
        ("music_inter", "legacy", "interstitial"),
        (None, "inter_07", "interstitial"),
    ],
)
def test_canonical_placement_accepts_historical_aliases(
    placement, cue_id, expected,
):
    from nodes.stable_audio_theme import StableAudioTheme

    assert StableAudioTheme._canonical_placement(placement, cue_id) == expected


def test_scifi_news_pro_music_rows_render_by_cue_id(monkeypatch):
    """scifi_news_pro lane: one clip per authored ledger.music[] row, keyed by cue_id,
    carrying its placement + anchor_line_id into the manifest (720-bakeoff C3).
    inter_NN maps to placement 'interstitial' (MF-B: never a KeyError)."""
    import json

    from nodes import _otr_cue_manifest as CM
    from nodes.stable_audio_theme import StableAudioTheme

    calls = []
    _stub_musicgen_generate_clip(monkeypatch, calls)
    ledger = json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [], "lines": [], "meta": {},
        "music": [
            {"cue_id": "opening", "placement": "opening",
             "anchor_line_id": "shot_000_music",
             "generation_prompt": "slow open", "target_duration_s": 12.0},
            {"cue_id": "inter_01", "placement": "interstitial",
             "anchor_line_id": "shot_001_music",
             "generation_prompt": "bridge", "target_duration_s": 4.0},
            {"cue_id": "closing", "placement": "closing",
             "anchor_line_id": "shot_002_music",
             "generation_prompt": "resolve", "target_duration_s": 8.0},
        ],
    })
    out = StableAudioTheme().generate(script_json=ledger, engine="musicgen")
    assert len(calls) == 3
    # authored prompts pass through verbatim (composer NOT invoked)
    assert {c["prompt"] for c in calls} == {"slow open", "bridge", "resolve"}
    manifest = CM.parse_manifest(out[1], batch_size=3)
    by_cue = {r["cue_id"]: r for r in manifest["cues"]}
    assert set(by_cue) == {"opening", "inter_01", "closing"}
    assert by_cue["inter_01"]["placement"] == "interstitial"
    assert by_cue["inter_01"]["anchor_line_id"] == "shot_001_music"
    # every row carries its authored anchor into the manifest, which is how
    # reconcile_ledger_music hands it back to ledger.music[] consumers
    anchors = {r["anchor_line_id"]: r["cue_id"] for r in manifest["cues"]}
    assert anchors["shot_001_music"] == "inter_01"


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
