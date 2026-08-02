"""Every cast member gets a VOICE, enforced where proofs are minted (S0, pure).

PBUG-20260802-02: gemma cast "The Relay" and wrote its five lines as pure stage
direction. Every existing check passed -- raw text looked voiced, proofs were
minted from it, THEN the writer-tail cleanup stripped the stage directions,
found nothing sayable, emptied the rows, and the freeze gate refused a
proof-coverage mismatch five stages after the real answer.

The decisive test here replays the REAL production ledger of that failure and
requires the gate to fail it BY NAME, before a proof exists.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes._otr_cast_voice_coverage import (
    CastVoiceCoverageError,
    require_voice_coverage,
)
from nodes._otr_content_authorship import (
    ContentAuthorshipError,
    stamp_receipt,
)

_LIVE_FAILURE = pathlib.Path(
    r"C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes"
    r"/pending_20260802_024714/audio/pending_20260802_024714_ledger.json")


def _ledger(cast, lines):
    return {"meta": {"source_bank": "testbank"}, "cast": cast, "lines": lines}


def _cast(*rows):
    out = [{"char_id": "c01", "name": "ANNOUNCER", "role": "announcer"}]
    out += [dict(r) for r in rows]
    return out


def _line(cid, text, *, skip=False):
    return {"line_id": "l_%s_%d" % (cid, id(text) % 997), "char_id": cid,
            "text": text, "skip": skip, "speaker_role": "character"}


ANN = {"char_id": "announcer", "text": "Tonight, a tale.", "skip": False,
       "line_id": "l_ann", "speaker_role": "announcer"}


def test_a_fully_voiced_cast_passes():
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"},
              {"char_id": "c03", "name": "The Relay"}),
        [ANN, _line("c02", "Read it back to me."),
         _line("c03", "Coordinates confirmed. Course is wrong.")])
    require_voice_coverage(led, owner_bank="testbank")  # no raise


def test_STAGE_DIRECTION_ONLY_is_not_a_voice():
    """The live defect, minimally: raw text present, nothing sayable.

    `clean_spoken_text('(static crackles)') == ''` -- measured, not assumed.
    The old checks accepted this row; the cleanup later emptied it. The gate
    must refuse it UP FRONT, naming the character.
    """
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"},
              {"char_id": "c03", "name": "The Relay"}),
        [ANN, _line("c02", "Read it back to me."),
         _line("c03", "(static crackles)"),
         _line("c03", "[low hum]")])
    with pytest.raises(CastVoiceCoverageError) as exc:
        require_voice_coverage(led, owner_bank="testbank")
    err = exc.value
    assert [m["char_id"] for m in err.missing] == ["c03"]
    assert "The Relay" in str(err)
    assert "reroll the cast" in str(err)


def test_the_REAL_failing_production_ledger_is_refused_BY_NAME():
    """The live artifact that cost the ltx_video leg, replayed through the gate.

    Its five c03 rows carry `tts_skip_reason='empty_spoken_text_at_ledger_cleanup'`
    -- the cleanup already emptied them -- so the gate must fail it and must
    name c03/The Relay, not a shot id.
    """
    if not _LIVE_FAILURE.exists():
        pytest.skip("live failure ledger not on this box")
    led = json.loads(_LIVE_FAILURE.read_text(encoding="utf-8"))
    with pytest.raises(CastVoiceCoverageError) as exc:
        require_voice_coverage(
            led, owner_bank=str(led["meta"].get("source_bank") or ""))
    assert any(m["char_id"] == "c03" for m in exc.value.missing)
    assert "The Relay" in str(exc.value)


def test_a_SKIPPED_line_does_not_count_as_a_voice():
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"}),
        [ANN, _line("c02", "Perfectly sayable.", skip=True)])
    with pytest.raises(CastVoiceCoverageError):
        require_voice_coverage(led, owner_bank="testbank")


def test_the_ANNOUNCER_is_matched_by_its_sentinel_not_its_cast_id():
    """Announcer lines carry char_id='announcer', never the roster id (c01).
    Matching raw ids would flag every episode's announcer as silent."""
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"}),
        [ANN, _line("c02", "One good line.")])
    require_voice_coverage(led, owner_bank="testbank")  # no raise

    silent_announcer = _ledger(
        _cast({"char_id": "c02", "name": "Elias"}),
        [_line("c02", "One good line.")])
    with pytest.raises(CastVoiceCoverageError) as exc:
        require_voice_coverage(silent_announcer, owner_bank="testbank")
    assert [m["char_id"] for m in exc.value.missing] == ["c01"]


def test_stamp_receipt_now_REFUSES_before_minting_a_proof():
    """The wiring assertion: the gate lives at the shared mint site, so a
    ledger with a mime in the cast never gets an authorship receipt at all --
    no proof exists to become 'a proof of nothing'."""
    led = _ledger(
        _cast({"char_id": "c03", "name": "The Relay"}),
        [ANN, _line("c03", "(static crackles)")])
    with pytest.raises(CastVoiceCoverageError):
        stamp_receipt(led, owner_bank="testbank",
                      accepted_artifacts={"script": "irrelevant"})
    assert "content_authorship" not in (led.get("meta") or {}), (
        "the receipt must NOT be stamped when coverage fails")


def test_a_voiced_ledger_still_mints_its_receipt_exactly_as_before():
    """The gate must not change the happy path: same rows in, receipt stamped."""
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"}),
        [ANN, _line("c02", "Course corrected. Send it.")])
    receipt = stamp_receipt(led, owner_bank="testbank",
                            accepted_artifacts={"script": "artifact-bytes"})
    assert led["meta"]["content_authorship"] is receipt
    assert {r["line_id"] for r in receipt["line_proofs"]} == \
        {r["line_id"] for r in led["lines"]}


def test_typed_error_carries_the_facts_a_retry_prompt_needs():
    """codex MUST-FIX 6: ladders must not parse exception text."""
    led = _ledger(
        _cast({"char_id": "c02", "name": "Elias"},
              {"char_id": "c03", "name": "The Relay"}),
        [ANN, _line("c02", "A line."), _line("c03", "(hum)")])
    with pytest.raises(CastVoiceCoverageError) as exc:
        require_voice_coverage(led, owner_bank="somebank")
    err = exc.value
    assert err.owner_bank == "somebank"
    assert err.missing == [{"char_id": "c03", "name": "The Relay"}]
    assert err.cast_total == 3 and err.voiced_total == 1
