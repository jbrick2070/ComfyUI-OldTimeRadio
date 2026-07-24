"""docs/sprint_drafts/build2/test_craft_floor.py -- Build 2 draft tests.

Covers the deterministic Tier-A craft floor:
  * clean pass on a known-good slot-formatted draft (zero false-reds),
  * one focused case per failure code in FAILURE_CODES,
  * determinism: same input twice -> identical verdict.

PURE: no torch, no live model, no ComfyUI import. The module under test
is staged outside the `nodes/` package, so it is loaded by file path
here. When the integration session moves it to nodes/_otr_craft_floor.py,
swap the loader for `from nodes import _otr_craft_floor as cf`.

Run staged (from repo root):
    python -m pytest docs/sprint_drafts/build2/test_craft_floor.py -v

UTF-8 no BOM. SFW.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Promoted (Build 2, 2026-05-28): the module now lives in the nodes/
# package, so import it the normal way. The staged file-path
# SourceFileLoader block was removed on promotion per WIRING_SPEC.md
# section 1 step 4.
# ---------------------------------------------------------------------------

from nodes import _otr_craft_floor as cf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _manifest():
    """A known-good 4-slot manifest (announcer bookends + two-hander)."""
    return [
        {"slot_id": "d001", "speaker": "ANNOUNCER"},
        {"slot_id": "d002", "speaker": "REN BLACK"},
        {"slot_id": "d003", "speaker": "DR. MAEVE COLE"},
        {"slot_id": "d004", "speaker": "ANNOUNCER"},
    ]


def _clean_raw():
    """A clean slot-formatted draft matching the manifest."""
    return (
        "d001|ANNOUNCER: From the late desk, a story that would not "
        "stay filed.\n"
        "d002|REN BLACK: I read the ledger twice. The numbers do not "
        "lie, Maeve.\n"
        "d003|DR. MAEVE COLE: Then we both go down with what we sign "
        "tonight.\n"
        "d004|ANNOUNCER: And the lamp in the records room burned till "
        "dawn."
    )


# ---------------------------------------------------------------------------
# Parser helper contract
# ---------------------------------------------------------------------------


def test_parse_slot_lines_shape():
    rows = cf.parse_slot_lines(_clean_raw())
    assert len(rows) == 4
    assert rows[0] == (
        "d001",
        "ANNOUNCER",
        "From the late desk, a story that would not stay filed.",
    )
    assert rows[2][0] == "d003"
    assert rows[2][1] == "DR. MAEVE COLE"


def test_parse_preserves_internal_pause_as_one_block():
    raw = "d002|REN BLACK: I will not sign that paper. ... Not tonight."
    rows = cf.parse_slot_lines(raw)
    assert len(rows) == 1
    assert rows[0][2] == "I will not sign that paper. ... Not tonight."


def test_normalize_round_trips():
    line = cf.normalize_slot_line("d007", "  REN BLACK ", " hold the line ")
    assert line == "d007|REN BLACK: hold the line"
    rows = cf.parse_slot_lines(line)
    assert rows == [("d007", "REN BLACK", "hold the line")]


# ---------------------------------------------------------------------------
# Clean pass -- zero false-reds (Build 2 gate)
# ---------------------------------------------------------------------------


def test_clean_pass_zero_failures():
    result = cf.evaluate_tier_a(_clean_raw(), _manifest())
    assert result.passed is True
    assert result.failures == []
    assert result.failure_codes == []
    assert len(result.parsed_slots) == 4


def test_clean_pass_accepts_tuple_manifest():
    manifest = [
        ("d001", "ANNOUNCER"),
        ("d002", "REN BLACK"),
        ("d003", "DR. MAEVE COLE"),
        ("d004", "ANNOUNCER"),
    ]
    result = cf.evaluate_tier_a(_clean_raw(), manifest)
    assert result.passed is True


def test_clean_pass_accepts_ledger_native_key():
    # Ledger lines carry `dialogue_slot_id`, not `slot_id`.
    manifest = [
        {"dialogue_slot_id": "d001", "speaker": "ANNOUNCER"},
        {"dialogue_slot_id": "d002", "speaker": "REN BLACK"},
        {"dialogue_slot_id": "d003", "speaker": "DR. MAEVE COLE"},
        {"dialogue_slot_id": "d004", "speaker": "ANNOUNCER"},
    ]
    result = cf.evaluate_tier_a(_clean_raw(), manifest)
    assert result.passed is True


# ---------------------------------------------------------------------------
# One case per failure code
# ---------------------------------------------------------------------------


def test_slot_count_mismatch():
    # Drop the last block; manifest still expects 4.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|REN BLACK: I read the ledger twice and it still holds.\n"
        "d003|DR. MAEVE COLE: Then we both go down with it tonight."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.SLOT_COUNT_MISMATCH in result.failure_codes


def test_slot_order_mismatch():
    # Swap d002 and d003 positions; same ids, wrong order.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d003|DR. MAEVE COLE: Then we both go down with it tonight here.\n"
        "d002|REN BLACK: I read the ledger twice and it still holds firm.\n"
        "d004|ANNOUNCER: And the lamp in the records room burned till dawn."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.SLOT_ORDER_MISMATCH in result.failure_codes
    # The order failure should point at the first divergent position.
    order_fail = next(
        f for f in result.failures if f.code == cf.SLOT_ORDER_MISMATCH
    )
    assert order_fail.index == 1


def test_speaker_mismatch():
    # d002 should be REN BLACK; here it is the wrong speaker.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|DR. MAEVE COLE: I read the ledger twice and it still holds.\n"
        "d003|DR. MAEVE COLE: Then we both go down with it tonight here.\n"
        "d004|ANNOUNCER: And the lamp in the records room burned till dawn."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.SPEAKER_MISMATCH in result.failure_codes
    spk_fail = next(
        f for f in result.failures if f.code == cf.SPEAKER_MISMATCH
    )
    assert spk_fail.slot_id == "d002"


def test_empty_line():
    # d003 carries only whitespace text.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|REN BLACK: I read the ledger twice and it still holds firm.\n"
        "d003|DR. MAEVE COLE:   \n"
        "d004|ANNOUNCER: And the lamp in the records room burned till dawn."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.EMPTY_LINE in result.failure_codes
def test_short_nonempty_line_is_accepted():
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|REN BLACK: No.\n"
        "d003|DR. MAEVE COLE: Then we both go down with it tonight here.\n"
        "d004|ANNOUNCER: And the lamp in the records room burned till dawn."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is True
    assert result.failure_codes == []


def test_parse_error():
    # d003 block opens with the slot prefix (so it IS treated as its own
    # block) but is malformed: a pipe with no speaker before the colon.
    # The full d###|SPEAKER: text pattern fails -> PARSE_ERROR.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|REN BLACK: I read the ledger twice and it still holds firm.\n"
        "d003|: missing the speaker before the colon entirely here now.\n"
        "d004|ANNOUNCER: And the lamp in the records room burned till dawn."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.PARSE_ERROR in result.failure_codes


def test_duplicate_slot():
    # d002 appears twice; d004 is dropped to keep this focused on the dup.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|REN BLACK: I read the ledger twice and it still holds firm.\n"
        "d002|REN BLACK: I read it a third time and it still holds firm.\n"
        "d003|DR. MAEVE COLE: Then we both go down with it tonight here."
    )
    result = cf.evaluate_tier_a(raw, _manifest())
    assert result.passed is False
    assert cf.DUPLICATE_SLOT in result.failure_codes
    dup_fail = next(
        f for f in result.failures if f.code == cf.DUPLICATE_SLOT
    )
    assert dup_fail.slot_id == "d002"


def test_failure_code_vocabulary_is_complete():
    # Every code referenced by tests must be in the published vocabulary.
    for code in (
        cf.SLOT_COUNT_MISMATCH,
        cf.SLOT_ORDER_MISMATCH,
        cf.SPEAKER_MISMATCH,
        cf.EMPTY_LINE,
        cf.PARSE_ERROR,
        cf.DUPLICATE_SLOT,
    ):
        assert code in cf.FAILURE_CODES


# ---------------------------------------------------------------------------
# Determinism: same input -> identical verdict
# ---------------------------------------------------------------------------


def test_determinism_clean():
    r1 = cf.evaluate_tier_a(_clean_raw(), _manifest())
    r2 = cf.evaluate_tier_a(_clean_raw(), _manifest())
    assert r1.to_dict() == r2.to_dict()


def test_determinism_failing():
    # A deliberately multi-failure draft: wrong speaker, missing manifest
    # row, and a parse-error block. The short nonempty line is valid.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|DR. MAEVE COLE: I read the ledger twice and it holds firm.\n"
        "d003|DR. MAEVE COLE: No.\n"
        "garbled line with no slot prefix at all in this block here."
    )
    manifest = _manifest()
    r1 = cf.evaluate_tier_a(raw, manifest)
    r2 = cf.evaluate_tier_a(raw, manifest)
    assert r1.passed is False
    assert r1.to_dict() == r2.to_dict()
    # Codes must come back in the same deterministic order each run.
    assert r1.failure_codes == r2.failure_codes


def test_determinism_failure_order_is_stable():
    # Repeat with a fresh manifest object to confirm the order is a
    # property of the inputs, not of object identity / dict iteration.
    raw = (
        "d001|ANNOUNCER: From the late desk, a story that would not stay.\n"
        "d002|DR. MAEVE COLE: I read the ledger twice and it holds firm.\n"
        "d003|DR. MAEVE COLE: No.\n"
        "garbled line with no slot prefix at all in this block here."
    )
    r1 = cf.evaluate_tier_a(raw, _manifest())
    r2 = cf.evaluate_tier_a(raw, _manifest())
    assert r1.failure_codes == r2.failure_codes


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
