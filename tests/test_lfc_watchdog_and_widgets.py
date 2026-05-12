"""tests/test_lfc_watchdog_and_widgets.py

LFC sprint Commit 12 -- regression coverage for the VRAM watchdog,
the speaker-stats voice-drift detector, the cascade widget surface,
and the workflow JSON widgets_values positional contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_watchdog as _WD  # noqa: E402


# ---------------------------------------------------------------------------
# Speaker stats
# ---------------------------------------------------------------------------


def _line(line_id, char_id, role, text, *, skip=False):
    return {
        "line_id": line_id, "char_id": char_id,
        "speaker_role": role, "text": text, "skip": skip,
    }


class TestSpeakerStats:
    def test_single_speaker_stats(self):
        lines = [
            _line("l001", "c01", "character", "Hello there general."),
            _line("l002", "c01", "character", "How are you today friend."),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert "c01" in stats
        s = stats["c01"]
        assert s.line_count == 2
        assert s.total_words == 8  # 3 + 5
        assert s.mean_line_length == 4.0

    def test_multiple_speakers_stats(self):
        lines = [
            _line("l001", "c01", "character", "alpha beta"),
            _line("l002", "c02", "character", "gamma delta epsilon"),
            _line("l003", "c01", "character", "zeta eta"),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert stats["c01"].line_count == 2
        assert stats["c02"].line_count == 1
        assert stats["c01"].mean_line_length == 2.0
        assert stats["c02"].mean_line_length == 3.0

    def test_skipped_lines_excluded(self):
        lines = [
            _line("l001", "c01", "character", "real line one two three"),
            _line("l002", "c01", "character", "skipped", skip=True),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert stats["c01"].total_words == 5
        assert stats["c01"].line_count == 1

    def test_non_voiced_lines_excluded(self):
        lines = [
            _line("l001", "sfx", "sfx", "door slam crash"),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert stats == {}

    def test_vocab_diversity_TTR(self):
        # 4 unique tokens out of 4 total = 1.0 TTR.
        lines = [
            _line("l001", "c01", "character", "alpha beta gamma delta"),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert stats["c01"].vocab_diversity == 1.0

    def test_vocab_diversity_with_repetition(self):
        # 3 unique out of 6 total = 0.5 TTR.
        lines = [
            _line("l001", "c01", "character",
                  "alpha beta gamma alpha beta gamma"),
        ]
        stats = _WD.compute_speaker_stats(lines)
        assert stats["c01"].vocab_diversity == 0.5


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestFlagLineDrift:
    def _stats(self, *, mean_len=10.0, ttr=0.7):
        return _WD.SpeakerStats(
            char_id="c01", line_count=5, total_words=50,
            mean_line_length=mean_len, vocab_diversity=ttr,
        )

    def test_aligned_line_no_drift(self):
        drifted, reason = _WD.flag_line_drift(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            self._stats(mean_len=10.0, ttr=0.7),
        )
        assert drifted is False
        assert reason == ""

    def test_too_long_drift(self):
        # Mean = 10; this line = 25 words = 150% deviation.
        too_long = " ".join(f"w{i}" for i in range(25))
        drifted, reason = _WD.flag_line_drift(
            too_long, self._stats(mean_len=10.0, ttr=0.7),
        )
        assert drifted is True
        assert "word_count" in reason
        assert "150%" in reason

    def test_too_short_drift(self):
        # Mean = 10; this line = 2 words = 80% deviation.
        drifted, reason = _WD.flag_line_drift(
            "alpha beta", self._stats(mean_len=10.0, ttr=0.7),
        )
        assert drifted is True
        assert "word_count" in reason

    def test_vocab_collapse_drift(self):
        # Mean TTR 0.7; line has 0 diversity (all same token).
        drifted, reason = _WD.flag_line_drift(
            "go go go go go go go go go go",
            self._stats(mean_len=10.0, ttr=0.7),
        )
        assert drifted is True
        assert "line_diversity" in reason

    def test_zero_mean_returns_no_drift(self):
        drifted, reason = _WD.flag_line_drift(
            "anything",
            _WD.SpeakerStats(
                char_id="c01", line_count=0, total_words=0,
                mean_line_length=0.0, vocab_diversity=0.0,
            ),
        )
        assert drifted is False
        assert reason == ""

    def test_empty_line_returns_no_drift(self):
        drifted, _ = _WD.flag_line_drift(
            "", self._stats(mean_len=10.0, ttr=0.7),
        )
        assert drifted is False


# ---------------------------------------------------------------------------
# VRAM watchdog
# ---------------------------------------------------------------------------


class TestVRAMWatchdog:
    def test_under_ceiling_returns_false(self):
        over, current = _WD.vram_over_ceiling(
            ceiling_gb=14.0,
            measure_fn=lambda: 8.5,
        )
        assert over is False
        assert current == 8.5

    def test_over_ceiling_returns_true(self):
        over, current = _WD.vram_over_ceiling(
            ceiling_gb=14.0,
            measure_fn=lambda: 14.5,
        )
        assert over is True
        assert current == 14.5

    def test_exactly_at_ceiling_returns_false(self):
        over, _ = _WD.vram_over_ceiling(
            ceiling_gb=14.0,
            measure_fn=lambda: 14.0,
        )
        assert over is False  # strict >

    def test_torch_vram_returns_zero_when_unavailable(self):
        # Smoke test the real measurement fn -- in the test env
        # there's no live LLM allocation, so it returns 0.0 (or a
        # small value if pytorch is loaded for unrelated tests).
        gb = _WD._torch_vram_allocated_gb()
        assert isinstance(gb, float)
        assert gb >= 0.0


# ---------------------------------------------------------------------------
# Cascade widget surface
# ---------------------------------------------------------------------------


class TestCascadeWidgetSurface:
    def test_all_widgets_present(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        opt = schema["optional"]
        # Each new toggle present and typed BOOLEAN.
        for k in ("enable_phase_3_polish",
                  "polish_announcer_beats",
                  "enable_phase_4_scene_coherence",
                  "enable_phase_4_5_smart_suggestion",
                  "enable_phase_5_voice_drift",
                  "enable_phase_7_audio_readiness",
                  "enable_phase_8_video_readiness"):
            assert k in opt, f"missing widget {k}"
            assert opt[k][0] == "BOOLEAN"
        # VRAM ceiling float.
        assert "vram_ceiling_gb" in opt
        assert opt["vram_ceiling_gb"][0] == "FLOAT"

    def test_phase_3_polish_default_off(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert schema["optional"]["enable_phase_3_polish"][1][
            "default"
        ] is False

    def test_phase_4_5_default_off(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert schema["optional"][
            "enable_phase_4_5_smart_suggestion"
        ][1]["default"] is False

    def test_phase_7_default_on(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert schema["optional"][
            "enable_phase_7_audio_readiness"
        ][1]["default"] is True

    def test_phase_8_default_on(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert schema["optional"][
            "enable_phase_8_video_readiness"
        ][1]["default"] is True

    def test_vram_ceiling_default_14gb(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert schema["optional"]["vram_ceiling_gb"][1][
            "default"
        ] == 14.0


# ---------------------------------------------------------------------------
# Workflow JSON widgets_values positional contract
# ---------------------------------------------------------------------------


class TestWorkflowJsonWidgetsValues:
    def _cascade_node(self):
        wf_path = (
            Path(__file__).resolve().parents[1]
            / "workflows" / "otr_scifi_16gb_full.json"
        )
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        for node in wf["nodes"]:
            if node.get("type") == "OTR_LedgerFreezeCascade":
                return node
        pytest.fail("OTR_LedgerFreezeCascade node not found in workflow")

    def test_workflow_node_has_nine_widget_values(self):
        node = self._cascade_node()
        # model_id + 7 BOOLEAN + 1 FLOAT = 9 positional widgets.
        # Phase 4 toggle landed in commit 8. Phase 6 toggle lands
        # in commit 10.
        wv = node.get("widgets_values") or []
        assert len(wv) == 9

    def test_model_id_at_slot_0(self):
        node = self._cascade_node()
        assert node["widgets_values"][0] == (
            "mistralai/Mistral-Nemo-Instruct-2407"
        )

    def test_default_booleans_at_slots_1_thru_7(self):
        node = self._cascade_node()
        wv = node["widgets_values"]
        # enable_phase_3_polish, polish_announcer_beats,
        # enable_phase_4_scene_coherence,
        # enable_phase_4_5_smart_suggestion,
        # enable_phase_5_voice_drift all default OFF (False).
        for i in (1, 2, 3, 4, 5):
            assert wv[i] is False, f"slot {i} expected False, got {wv[i]!r}"
        # enable_phase_7 / 8 default ON (True).
        assert wv[6] is True
        assert wv[7] is True

    def test_vram_ceiling_at_slot_8(self):
        node = self._cascade_node()
        assert node["widgets_values"][8] == 14.0

    def test_output_socket_4_is_freeze_verdict(self):
        node = self._cascade_node()
        outputs = node.get("outputs") or []
        assert len(outputs) >= 5
        assert outputs[4]["name"] == "freeze_verdict"
