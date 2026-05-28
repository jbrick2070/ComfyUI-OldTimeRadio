"""tests/test_otr_story_room_commit.py -- Wave 3 bridge.

Pin contract: OTR_StoryRoomCommit takes a Story Room extraction
payload and overwrites ledger.lines[*].text per beat_id when
commit=True. Defaults dormant; never raises (PD1).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nodes.OTR_StoryRoomCommit import OTR_StoryRoomCommit


def _make_node():
    return OTR_StoryRoomCommit()


def _ok_extraction_payload(
    rows=None, premise: str = "test premise",
) -> str:
    """Return a serialized StoryRoomExtraction JSON with status='ok'."""
    if rows is None:
        rows = [
            {"beat_id": "b001", "speaker": "ANNOUNCER", "text": "Open."},
            {"beat_id": "b002", "speaker": "REN BLACK", "text": "I won't sign."},
            {"beat_id": "b003", "speaker": "DR MAEVE", "text": "Then we wait."},
            {"beat_id": "b004", "speaker": "ANNOUNCER", "text": "Good night."},
        ]
    return json.dumps({
        "status": "ok",
        "premise": premise,
        "cast": [],
        "beats": [],
        "dialogue": rows,
        "audio_cues": [],
        "running_facts": [],
        "arc": None,
    })


def _legacy_ledger() -> dict:
    return {
        "episode_id": "ep_test",
        "schema_version": "l3-2026-05-14",
        "lines": [
            {"line_id": "l001", "beat_id": "b001",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Legacy open", "word_count": 2, "char_count": 11,
             "skip": False},
            {"line_id": "l002", "beat_id": "b002",
             "speaker_role": "character", "char_id": "c01",
             "text": "Legacy line 2", "word_count": 3, "char_count": 13,
             "skip": False},
            {"line_id": "l003", "beat_id": "b003",
             "speaker_role": "character", "char_id": "c02",
             "text": "Legacy line 3", "word_count": 3, "char_count": 13,
             "skip": False},
            {"line_id": "l004", "beat_id": "b004",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Legacy close", "word_count": 2, "char_count": 12,
             "skip": False},
        ],
        "meta": {"episode_title": "Test"},
    }


# ---------------------------------------------------------------------------
# Dormant pass-through
# ---------------------------------------------------------------------------


class TestDormantPassThrough:
    def test_commit_false_returns_input_unchanged(self):
        node = _make_node()
        out = node.run(
            script_json="legacy script",
            story_room_extraction=_ok_extraction_payload(),
            commit=False,
        )
        assert out == ("legacy script",)

    def test_empty_extraction_passes_through(self):
        node = _make_node()
        out = node.run(
            script_json="legacy",
            story_room_extraction="",
            commit=True,
        )
        assert out == ("legacy",)

    def test_dormant_status_passes_through(self):
        node = _make_node()
        payload = json.dumps({"status": "dormant", "dialogue": []})
        out = node.run(
            script_json="legacy",
            story_room_extraction=payload,
            commit=True,
        )
        assert out == ("legacy",)

    def test_failed_status_passes_through(self):
        node = _make_node()
        payload = json.dumps({"status": "failed", "reason": "test"})
        out = node.run(
            script_json="legacy",
            story_room_extraction=payload,
            commit=True,
        )
        assert out == ("legacy",)

    def test_malformed_json_passes_through(self):
        node = _make_node()
        out = node.run(
            script_json="legacy",
            story_room_extraction="not valid json {",
            commit=True,
        )
        assert out == ("legacy",)

    def test_extraction_with_no_dialogue_passes_through(self):
        node = _make_node()
        payload = json.dumps({"status": "ok", "dialogue": []})
        out = node.run(
            script_json="legacy",
            story_room_extraction=payload,
            commit=True,
        )
        assert out == ("legacy",)


# ---------------------------------------------------------------------------
# Active commit path
# ---------------------------------------------------------------------------


class TestActiveCommit:
    def _patch_ledger_io(self, ledger_state: dict, tmp_path):
        """Patch _otr_ledger.in_flight_ledger_path / load / save."""
        led_path = tmp_path / "ep_test_ledger.json"
        led_path.write_text(json.dumps(ledger_state), encoding="utf-8")

        def fake_load(path):
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                return None

        def fake_save(path, ledger):
            try:
                Path(path).write_text(
                    json.dumps(ledger), encoding="utf-8",
                )
                return True
            except Exception:
                return False

        return led_path, fake_load, fake_save

    def test_commit_overwrites_line_text(self, tmp_path):
        led_path, fake_load, fake_save = self._patch_ledger_io(
            _legacy_ledger(), tmp_path,
        )
        node = _make_node()
        with patch("nodes._otr_ledger.in_flight_ledger_path", return_value=led_path), \
             patch("nodes._otr_ledger.load_ledger_safe", side_effect=fake_load), \
             patch("nodes._otr_ledger.save_ledger_safe", side_effect=fake_save):
            out = node.run(
                script_json="passthrough",
                story_room_extraction=_ok_extraction_payload(),
                commit=True,
            )

        # script_json passes through unchanged.
        assert out == ("passthrough",)

        # Ledger on disk now has the Story Room lines.
        ledger = json.loads(led_path.read_text(encoding="utf-8"))
        line_by_beat = {l["beat_id"]: l for l in ledger["lines"]}
        assert line_by_beat["b002"]["text"] == "I won't sign."
        assert line_by_beat["b003"]["text"] == "Then we wait."
        assert line_by_beat["b001"]["text"] == "Open."
        assert line_by_beat["b004"]["text"] == "Good night."

        # Audit stamp landed.
        commit_meta = ledger["meta"]["story_room_commit"]
        assert commit_meta["committed"] is True
        assert commit_meta["rows_attempted"] == 4
        assert commit_meta["rows_committed"] == 4
        assert commit_meta["rows_skipped"] == 0

        # Totals recomputed.
        expected_total_words = sum(
            l["word_count"] for l in ledger["lines"]
        )
        assert ledger["total_word_count"] == expected_total_words

    def test_commit_skips_unknown_beat_ids(self, tmp_path):
        led_path, fake_load, fake_save = self._patch_ledger_io(
            _legacy_ledger(), tmp_path,
        )
        rows = [
            {"beat_id": "b002", "speaker": "REN", "text": "Refusal."},
            {"beat_id": "b999", "speaker": "GHOST",
             "text": "this beat doesn't exist"},
        ]
        node = _make_node()
        with patch("nodes._otr_ledger.in_flight_ledger_path", return_value=led_path), \
             patch("nodes._otr_ledger.load_ledger_safe", side_effect=fake_load), \
             patch("nodes._otr_ledger.save_ledger_safe", side_effect=fake_save):
            node.run(
                script_json="x",
                story_room_extraction=_ok_extraction_payload(rows=rows),
                commit=True,
            )
        ledger = json.loads(led_path.read_text(encoding="utf-8"))
        commit_meta = ledger["meta"]["story_room_commit"]
        assert commit_meta["rows_attempted"] == 2
        assert commit_meta["rows_committed"] == 1
        assert commit_meta["rows_skipped"] == 1

    def test_commit_skips_empty_text_rows(self, tmp_path):
        led_path, fake_load, fake_save = self._patch_ledger_io(
            _legacy_ledger(), tmp_path,
        )
        rows = [
            {"beat_id": "b002", "speaker": "REN", "text": ""},
        ]
        node = _make_node()
        with patch("nodes._otr_ledger.in_flight_ledger_path", return_value=led_path), \
             patch("nodes._otr_ledger.load_ledger_safe", side_effect=fake_load), \
             patch("nodes._otr_ledger.save_ledger_safe", side_effect=fake_save):
            node.run(
                script_json="x",
                story_room_extraction=_ok_extraction_payload(rows=rows),
                commit=True,
            )
        ledger = json.loads(led_path.read_text(encoding="utf-8"))
        # b002 text NOT overwritten.
        b002 = next(l for l in ledger["lines"] if l["beat_id"] == "b002")
        assert b002["text"] == "Legacy line 2"


# ---------------------------------------------------------------------------
# PD1 defensive paths
# ---------------------------------------------------------------------------


class TestDefensivePaths:
    def test_missing_in_flight_ledger_path_passes_through(self):
        node = _make_node()
        with patch("nodes._otr_ledger.in_flight_ledger_path", return_value=None):
            out = node.run(
                script_json="legacy",
                story_room_extraction=_ok_extraction_payload(),
                commit=True,
            )
        assert out == ("legacy",)

    def test_load_failure_passes_through(self):
        node = _make_node()
        with patch("nodes._otr_ledger.in_flight_ledger_path",
                   return_value=Path("/nonexistent/path")), \
             patch("nodes._otr_ledger.load_ledger_safe", return_value=None):
            out = node.run(
                script_json="legacy",
                story_room_extraction=_ok_extraction_payload(),
                commit=True,
            )
        assert out == ("legacy",)

    def test_node_class_surface(self):
        """Sanity pins on the Comfy node contract."""
        assert OTR_StoryRoomCommit.RETURN_TYPES == ("STRING",)
        assert OTR_StoryRoomCommit.RETURN_NAMES == ("script_json",)
        assert OTR_StoryRoomCommit.FUNCTION == "run"
        types = OTR_StoryRoomCommit.INPUT_TYPES()
        assert "script_json" in types["required"]
        assert "story_room_extraction" in types["required"]
        assert "commit" in types["required"]
        # commit defaults False (dormant).
        assert types["required"]["commit"][1]["default"] is False
