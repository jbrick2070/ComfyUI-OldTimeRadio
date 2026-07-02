"""tests/test_phase2b_progressive_ledger.py — progressive ledger writes.

Covers (per script-writing-architecture synthesis §3 Phase 2B):

  * init_lines_from_outline   — skeleton stamped at outline-validate
                                time; voiced rows start with text==""
  * update_line_text          — in-place text update on one beat row
                                with char_count + word_count recomputed
  * synthetic crash recovery  — partial ledger after K of N composes;
                                surviving rows kept, unfinished rows
                                still have text==""
  * atomic-save semantics     — every save uses write-tmp + rename
                                (existing Ledger.save behavior; we
                                pin that it still holds with the new
                                progressive pattern)
  * set_lines remains usable  — pre-Phase-2B callers (build_test_ledger
                                _from_director, scene builders) keep
                                working

Pure-Python. No GPU. No LLM.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as PL  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeBeat:
    beat_id: str
    speaker: str = ""
    speaker_role: str = "character"
    intent: str = "speak"
    target_words: int = 25
    mood: str = "tense"
    arc_phase: Optional[str] = None


@dataclass
class FakeOutline:
    beats: list


@pytest.fixture
def tmp_ledger_dir(tmp_path):
    """Isolated per-test output dir so saves never collide."""
    d = tmp_path / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def fresh_ledger(tmp_ledger_dir):
    led = PL.Ledger(
        episode_id="pending_test",
        out_dir=str(tmp_ledger_dir),
    )
    led.set_cast([
        {"char_id": "c01", "name": "ALICE"},
        {"char_id": "c02", "name": "BOB"},
        {"char_id": "c03", "name": "ANNOUNCER"},
    ])
    return led


@pytest.fixture
def fake_outline_3acts():
    return FakeOutline(beats=[
        FakeBeat("b001", "NARRATOR", "music_open", "cold open", 5, "bold"),
        FakeBeat("b002", "ANNOUNCER", "announcer", "intro", 25, "steady",
                 arc_phase="setup"),
        FakeBeat("b003", "ALICE", "character", "hears signal", 25, "curious",
                 arc_phase="setup"),
        FakeBeat("b004", "BOB", "character", "warns", 25, "worried",
                 arc_phase="setup"),
        FakeBeat("b005", "NARRATOR", "music_inter", "transition", 5, "bold"),
        FakeBeat("b006", "ALICE", "character", "decides", 25, "determined",
                 arc_phase="complication"),
        FakeBeat("b007", "ANNOUNCER", "announcer", "close", 25, "steady",
                 arc_phase="resolution"),
        FakeBeat("b008", "NARRATOR", "music_close", "fade", 5, "resolute"),
    ])


# ---------------------------------------------------------------------------
# init_lines_from_outline
# ---------------------------------------------------------------------------


class TestInitLinesFromOutline:

    def test_stamps_one_row_per_beat(self, fresh_ledger, fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        assert len(fresh_ledger.data["lines"]) == 8

    def test_voiced_rows_start_empty(self, fresh_ledger, fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        rows = {r["beat_id"]: r for r in fresh_ledger.data["lines"]}
        # Voiced beats start with text == ""
        assert rows["b002"]["text"] == ""
        assert rows["b003"]["text"] == ""
        assert rows["b004"]["text"] == ""
        # S1 (2026-06-22): non-voiced rows with NO sfx_cue start text=""
        # too -- the generic beat intent ("cold open"/"transition"/"fade")
        # MUST NOT bleed into the line text / script transcript. The intent
        # is preserved separately in beat_intent.
        assert rows["b001"]["text"] == ""
        assert rows["b005"]["text"] == ""
        assert rows["b008"]["text"] == ""
        assert rows["b001"]["beat_intent"] == "cold open"
        assert rows["b005"]["beat_intent"] == "transition"

    def test_non_voiced_rows_never_carry_text(self, fresh_ledger):
        # rip-sfx-broll (2026-07-01): the sfx_cue carrier is GONE -- every
        # non-voiced (music) row is a pure render contract with text=="",
        # unconditionally.
        outline = FakeOutline(beats=[
            FakeBeat("b001", "NARRATOR", "music_open", "cold open", 0,
                     "tense"),
            FakeBeat("b002", "NARRATOR", "music_inter", "transition", 5,
                     "bold"),
        ])
        fresh_ledger.init_lines_from_outline(outline)
        rows = {r["beat_id"]: r for r in fresh_ledger.data["lines"]}
        assert rows["b001"]["text"] == ""
        assert rows["b002"]["text"] == ""
        # the narrative intent is preserved separately for prompt consumers
        assert rows["b001"]["beat_intent"] == "cold open"

    def test_speaker_role_and_arc_phase_stamped(self, fresh_ledger,
                                                 fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        rows = {r["beat_id"]: r for r in fresh_ledger.data["lines"]}
        assert rows["b003"]["speaker_role"] == "character"
        assert rows["b003"]["arc_phase"] == "setup"
        assert rows["b006"]["arc_phase"] == "complication"
        assert rows["b001"]["speaker_role"] == "music_open"
        # music beats have no arc_phase in this fixture.
        assert rows["b001"]["arc_phase"] is None

    def test_char_id_lookup_from_name_map(self, fresh_ledger,
                                          fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        rows = {r["beat_id"]: r for r in fresh_ledger.data["lines"]}
        assert rows["b003"]["char_id"] == "c01"  # ALICE
        assert rows["b004"]["char_id"] == "c02"  # BOB
        assert rows["b002"]["char_id"] == "announcer"
        assert rows["b001"]["char_id"] == "music_open"

    def test_compose_flags_init_empty(self, fresh_ledger, fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        for row in fresh_ledger.data["lines"]:
            assert row.get("compose_flags") == []


# ---------------------------------------------------------------------------
# update_line_text
# ---------------------------------------------------------------------------


class TestUpdateLineText:

    def test_updates_existing_row_in_place(self, fresh_ledger,
                                            fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        ok = fresh_ledger.update_line_text("b003", "I found the signal.")
        assert ok is True
        row = next(r for r in fresh_ledger.data["lines"]
                   if r["beat_id"] == "b003")
        assert row["text"] == "I found the signal."

    def test_recomputes_char_and_word_counts(self, fresh_ledger,
                                              fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        fresh_ledger.update_line_text("b003", "Hello there world.")
        row = next(r for r in fresh_ledger.data["lines"]
                   if r["beat_id"] == "b003")
        assert row["char_count"] == len("Hello there world.")
        assert row["word_count"] == 3

    def test_returns_false_when_beat_id_not_found(self, fresh_ledger,
                                                   fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        ok = fresh_ledger.update_line_text("b999", "no such beat")
        assert ok is False

    def test_does_not_create_new_rows(self, fresh_ledger,
                                       fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        before = len(fresh_ledger.data["lines"])
        fresh_ledger.update_line_text("b999", "ignored")
        after = len(fresh_ledger.data["lines"])
        assert before == after


# ---------------------------------------------------------------------------
# Synthetic crash recovery
# ---------------------------------------------------------------------------


class TestProgressiveCrashRecovery:

    def test_partial_compose_leaves_coherent_disk_state(self, fresh_ledger,
                                                         fake_outline_3acts):
        """Simulate a crash after composing 2 of 3 voiced beats. The
        ledger on disk should have all skeleton rows + the 2 composed
        texts; the 3rd row's text is still empty (signals 'pending')."""
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        fresh_ledger.save()  # post-init save

        # Compose 2 of the 3 character beats (b003, b004), leave b006
        # at the empty placeholder.
        fresh_ledger.update_line_text("b003", "Composed line one.")
        fresh_ledger.save()
        fresh_ledger.update_line_text("b004", "Composed line two.")
        fresh_ledger.save()
        # ... crash here. b006 never gets updated.

        # Re-load the ledger from disk.
        path = fresh_ledger.path
        import json
        with open(path, "r", encoding="utf-8") as f:
            disk = json.load(f)

        rows = {r["beat_id"]: r for r in disk["lines"]}
        # All 8 skeleton rows survived.
        assert len(rows) == 8
        # Composed beats have text.
        assert rows["b003"]["text"] == "Composed line one."
        assert rows["b004"]["text"] == "Composed line two."
        # Uncomposed character beat still has empty placeholder.
        assert rows["b006"]["text"] == ""
        # S1: non-voiced music beats (no sfx_cue) survive with text=""
        # and the intent preserved in beat_intent.
        assert rows["b001"]["text"] == ""
        assert rows["b001"]["beat_intent"] == "cold open"


# ---------------------------------------------------------------------------
# set_lines back-compat
# ---------------------------------------------------------------------------


class TestSetLinesBackCompat:

    def test_set_lines_still_works(self, fresh_ledger):
        """Pre-Phase-2B callers that build a full line list and stamp
        it once via set_lines must keep working."""
        fresh_ledger.set_lines([
            {"line_id": "l001", "beat_id": "b001", "char_id": "c01",
             "text": "Hello", "boundary": "shot_start"},
            {"line_id": "l002", "beat_id": "b002", "char_id": "c02",
             "text": "World", "boundary": "continue"},
        ])
        assert len(fresh_ledger.data["lines"]) == 2
        assert fresh_ledger.data["lines"][0]["text"] == "Hello"
        assert fresh_ledger.data["lines"][1]["text"] == "World"


# ---------------------------------------------------------------------------
# Save semantics (atomic write)
# ---------------------------------------------------------------------------


class TestSaveAtomicity:

    def test_save_produces_real_json_file(self, fresh_ledger,
                                           fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        path = fresh_ledger.save()
        assert path is not None
        assert os.path.exists(path)
        # The temp file pattern (.tmp suffix) must not be left behind.
        d = os.path.dirname(path)
        leftovers = [
            f for f in os.listdir(d)
            if f.endswith(".tmp") and "ledger" in f.lower()
        ]
        assert not leftovers, f"leftover temp files: {leftovers}"

    def test_per_line_saves_dont_grow_filesystem_garbage(
            self, fresh_ledger, fake_outline_3acts):
        """Stress: 8 saves in a row (mimicking the writer's
        per-iteration save) must not leave any stale temp files."""
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        for bid in ("b002", "b003", "b004", "b006", "b007"):
            fresh_ledger.update_line_text(bid, f"text for {bid}")
            fresh_ledger.save()
        d = os.path.dirname(fresh_ledger.path)
        leftovers = [
            f for f in os.listdir(d)
            if f.endswith(".tmp") and "ledger" in f.lower()
        ]
        assert not leftovers

    def test_save_assigns_merged_payload_back_to_self_data(
            self, fresh_ledger, fake_outline_3acts):
        """Wiring-review #5 (2026-05-11): after save(), in-memory
        ledger == on-disk JSON byte-for-byte. Without this, any
        script_json built downstream from `led.data` can diverge
        from disk.
        """
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        path = fresh_ledger.save()
        assert path is not None
        import json
        with open(path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        # In-memory == on-disk after save. Specifically: any field the
        # save path stamps (meta.paths, schema_version) is reflected
        # in self.data, not just in the file.
        assert fresh_ledger.data.get("schema_version") == on_disk.get("schema_version")
        assert fresh_ledger.data.get("meta", {}).get("paths") == on_disk.get("meta", {}).get("paths")
        assert fresh_ledger.data["lines"] == on_disk["lines"]


class TestUpdateLineTextReturnCheck:
    """Wiring-review #4 (2026-05-11): writer MUST check
    update_line_text return value. False is the signal that the
    ledger skeleton lacks the beat — never silently accept it."""

    def test_update_line_text_returns_false_on_missing_beat(
            self, fresh_ledger, fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        ok = fresh_ledger.update_line_text("b_no_such_beat", "x")
        assert ok is False

    def test_update_line_text_returns_true_on_existing_beat(
            self, fresh_ledger, fake_outline_3acts):
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        ok = fresh_ledger.update_line_text("b003", "I see it.")
        assert ok is True


class TestPeekLedgerAndHasCurrent:
    """Wiring-review #3 (2026-05-11): non-creating accessors for
    downstream nodes that fail loud when called without a writer
    upstream."""

    def test_peek_ledger_returns_none_when_no_writer(self):
        # Reset module state.
        PL._CURRENT = None
        assert PL.peek_ledger() is None
        assert PL.has_current_ledger() is False

    def test_has_current_ledger_false_on_empty_ledger(self, tmp_path):
        # Build a ledger with NO lines stamped yet.
        d = tmp_path / "audio"
        d.mkdir(parents=True, exist_ok=True)
        PL._CURRENT = PL.Ledger("pending_test", str(d))
        assert PL.has_current_ledger() is False

    def test_has_current_ledger_true_after_init_lines(
            self, fresh_ledger, fake_outline_3acts):
        # fresh_ledger is the module's _CURRENT after stamping.
        PL._CURRENT = fresh_ledger
        fresh_ledger.init_lines_from_outline(
            fake_outline_3acts,
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        assert PL.has_current_ledger() is True
        assert PL.peek_ledger() is fresh_ledger
