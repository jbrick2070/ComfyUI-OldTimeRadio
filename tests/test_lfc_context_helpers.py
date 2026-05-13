"""tests/test_lfc_context_helpers.py

LFC sprint Commit 7 -- regression coverage for the Macro-Bible +
Micro-Recap context helpers used by Phase 4 / 5 / 6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_context as _LFC_CTX  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger_data(lines=None, *, with_music_inter=False,
                  outline_beats=None, news_terms=None, scene_synopses=None):
    """Build a representative L3-shaped ledger_data dict."""
    cast = [
        {"char_id": "c01", "name": "ALICE",
         "traits": "calm engineer",
         "voice_preset": "v2/en_speaker_6",
         "catchphrases": ["check the calibration"]},
        {"char_id": "c02", "name": "BOB",
         "traits": "skeptical analyst",
         "voice_preset": "v2/en_speaker_2"},
        {"char_id": "announcer", "name": "ANNOUNCER",
         "traits": "warm broadcast voice",
         "voice_preset": "v2/en_speaker_9"},
    ]

    if lines is None:
        lines = [
            {"line_id": "l001", "beat_id": "b001",
             "speaker_role": "music_open", "char_id": "",
             "text": "Opening theme"},
            {"line_id": "l002", "beat_id": "b002",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Welcome back."},
            {"line_id": "l003", "beat_id": "b003",
             "speaker_role": "character", "char_id": "c01",
             "text": "Bob, the readings are off."},
            {"line_id": "l004", "beat_id": "b004",
             "speaker_role": "character", "char_id": "c02",
             "text": "Then someone tampered with it."},
        ]
        if with_music_inter:
            lines.append({
                "line_id": "l005", "beat_id": "b005",
                "speaker_role": "music_inter", "char_id": "",
                "text": "Interstitial",
            })
            lines.append({
                "line_id": "l006", "beat_id": "b006",
                "speaker_role": "character", "char_id": "c01",
                "text": "Look at this trace.",
            })
            lines.append({
                "line_id": "l007", "beat_id": "b007",
                "speaker_role": "character", "char_id": "c02",
                "text": "That can't be right.",
            })

    meta = {
        "episode_title": "The Calibration Drift",
        "style": "mission_control_procedural",
    }
    if outline_beats is not None:
        meta["outline"] = {"beats": outline_beats}
    if news_terms is not None:
        meta["news"] = {"key_terms": news_terms}
    if scene_synopses is not None:
        meta["scene_synopses"] = scene_synopses

    return {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_ctx_test",
        "cast": cast,
        "scenes": [], "shots": [],
        "beats": [], "lines": lines,
        "music": [], "clips": [],
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Token estimation + truncation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_short_string(self):
        assert _LFC_CTX.estimate_tokens("hello") == 2

    def test_empty_string_is_1(self):
        # estimate_tokens guarantees >= 1.
        assert _LFC_CTX.estimate_tokens("") >= 1

    def test_long_string(self):
        # 400 chars -> ~100 tokens at 1/4.
        s = "x" * 400
        assert _LFC_CTX.estimate_tokens(s) == 100

    def test_truncate_within_cap_returns_unchanged(self):
        s = "short"
        truncated = _LFC_CTX._truncate_to_token_cap(s, cap=100)
        assert truncated == s

    def test_truncate_marker_appended_when_over_cap(self):
        s = "line one. " * 200  # 2000 chars => ~500 tokens
        truncated = _LFC_CTX._truncate_to_token_cap(s, cap=50)
        assert "..." in truncated
        assert _LFC_CTX.estimate_tokens(truncated) <= 50 + 1


# ---------------------------------------------------------------------------
# Macro-Bible
# ---------------------------------------------------------------------------


class TestMacroBible:
    def test_includes_episode_logline(self):
        out = _LFC_CTX.build_macro_bible(_ledger_data())
        assert "EPISODE:" in out
        assert "The Calibration Drift" in out
        assert "mission_control_procedural" in out

    def test_includes_cast(self):
        out = _LFC_CTX.build_macro_bible(_ledger_data())
        assert "CAST:" in out
        assert "ALICE" in out
        assert "BOB" in out
        assert "ANNOUNCER" in out

    def test_includes_catchphrases_when_present(self):
        out = _LFC_CTX.build_macro_bible(_ledger_data())
        assert "check the calibration" in out

    def test_includes_news_key_terms(self):
        out = _LFC_CTX.build_macro_bible(
            _ledger_data(news_terms=["calibration", "drift"]),
        )
        assert "ALLOWED ENTITIES" in out
        assert "calibration" in out
        assert "drift" in out

    def test_includes_arc_skeleton(self):
        beats = [
            {"beat_id": "b001", "intent": "open scene"},
            {"beat_id": "b002", "intent": "reveal mystery"},
            {"beat_id": "b003", "intent": "resolve tension"},
        ]
        out = _LFC_CTX.build_macro_bible(
            _ledger_data(outline_beats=beats),
        )
        assert "ARC SKELETON:" in out
        assert "b001: open scene" in out
        assert "b002: reveal mystery" in out
        assert "b003: resolve tension" in out

    def test_token_cap_enforced(self):
        # Tighten the cap below the size of the natural bible so we
        # confirm truncation fires.
        out = _LFC_CTX.build_macro_bible(
            _ledger_data(), token_cap=50,
        )
        assert _LFC_CTX.estimate_tokens(out) <= 51

    def test_accepts_ledger_obj(self):
        from types import SimpleNamespace
        led = SimpleNamespace(data=_ledger_data())
        out = _LFC_CTX.build_macro_bible(led)
        assert "EPISODE:" in out

    def test_empty_cast_does_not_crash(self):
        data = _ledger_data()
        data["cast"] = []
        out = _LFC_CTX.build_macro_bible(data)
        assert "CAST:" not in out  # section omitted when empty
        assert "EPISODE:" in out


# ---------------------------------------------------------------------------
# Scene partitioning
# ---------------------------------------------------------------------------


class TestScenesFromLines:
    def test_no_music_inter_single_scene(self):
        scenes = _LFC_CTX.scenes_from_lines(_ledger_data())
        assert len(scenes) == 1
        assert scenes[0]["scene_idx"] == 0
        assert len(scenes[0]["lines"]) == 4

    def test_with_music_inter_two_scenes(self):
        scenes = _LFC_CTX.scenes_from_lines(
            _ledger_data(with_music_inter=True),
        )
        assert len(scenes) == 2
        assert scenes[0]["scene_idx"] == 0
        assert scenes[1]["scene_idx"] == 1
        # music_inter ends scene 0 (exclusive); scene 1 starts at
        # the marker (inclusive).
        assert scenes[0]["end_idx"] == 4
        assert scenes[1]["start_idx"] == 4

    def test_empty_lines_returns_empty_list(self):
        scenes = _LFC_CTX.scenes_from_lines({"lines": []})
        assert scenes == []


# ---------------------------------------------------------------------------
# Micro-Recap -- scene
# ---------------------------------------------------------------------------


class TestMicroRecapScene:
    def test_scene_recap_includes_lines(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="scene", target=0,
        )
        assert "SCENE 0:" in out
        assert "l003" in out
        assert "Bob, the readings are off." in out

    def test_scene_recap_includes_active_speakers(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="scene", target=0,
        )
        assert "active speakers" in out
        # ALICE + BOB + ANNOUNCER all appear in scene 0.
        assert "ALICE" in out or "calm engineer" in out
        assert "BOB" in out or "skeptical analyst" in out

    def test_scene_recap_pulls_prior_context_for_later_scenes(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(with_music_inter=True),
            scope="scene", target=1,
        )
        assert "SCENE 1:" in out
        assert "prior context" in out
        # Last 3 voiced lines of scene 0 should appear.
        assert "Bob, the readings are off." in out or (
            "Then someone tampered" in out
        )

    def test_scene_out_of_range_returns_message(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="scene", target=99,
        )
        assert "out of range" in out

    def test_invalid_target_returns_message(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="scene", target="not a scene",
        )
        assert "invalid scene target" in out

    def test_scene_recap_accepts_scene_dict(self):
        data = _ledger_data()
        scenes = _LFC_CTX.scenes_from_lines(data)
        out = _LFC_CTX.build_micro_recap(
            data, scope="scene", target=scenes[0],
        )
        assert "SCENE 0:" in out


# ---------------------------------------------------------------------------
# Micro-Recap -- speaker
# ---------------------------------------------------------------------------


class TestMicroRecapSpeaker:
    def test_speaker_recap_includes_voice_card(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="speaker", target="c01",
        )
        assert "SPEAKER:" in out
        assert "ALICE" in out

    def test_speaker_recap_includes_speakers_lines_only(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="speaker", target="c01",
        )
        # c01 says only one line in fixture.
        assert "Bob, the readings are off." in out
        # c02's line must NOT appear.
        assert "Then someone tampered" not in out

    def test_speaker_recap_invalid_target_returns_message(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="speaker", target="",
        )
        assert "invalid speaker target" in out


# ---------------------------------------------------------------------------
# Micro-Recap -- episode
# ---------------------------------------------------------------------------


class TestMicroRecapEpisode:
    def test_episode_uses_cached_synopses_when_present(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(scene_synopses=[
                "Scene 0 opens with a calibration mystery.",
                "Scene 1 resolves the tension with a discovery.",
            ]),
            scope="episode", target=None,
        )
        assert "EPISODE OVERVIEW:" in out
        assert "calibration mystery" in out
        assert "discovery" in out

    def test_episode_falls_back_to_outline_intents(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(outline_beats=[
                {"beat_id": "b001", "intent": "raise the stakes"},
            ]),
            scope="episode", target=None,
        )
        assert "raise the stakes" in out


# ---------------------------------------------------------------------------
# Token caps
# ---------------------------------------------------------------------------


class TestMicroRecapTokenCap:
    def test_scene_recap_respects_cap(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="scene", target=0, token_cap=20,
        )
        assert _LFC_CTX.estimate_tokens(out) <= 21
        assert "..." in out

    def test_unknown_scope_returns_message(self):
        out = _LFC_CTX.build_micro_recap(
            _ledger_data(), scope="invalid", target=None,
        )
        assert "unknown scope" in out
