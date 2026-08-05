"""Independent source banks wave 6 -- the shared tail's ledger cleanup pass.

The contract under test is the SPLIT: content is repaired, structure is named.
A required writer-owned field with no owner and no value is the only thing this
pass may fail on; unsafe language, thin prose and stale counts are repaired.
"""
from __future__ import annotations

import pytest

from nodes import _otr_ledger_cleanup as lc


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _complete_ledger() -> dict:
    """A minimal ledger that already satisfies every writer-owned field."""
    return {
        "cast": [
            {"char_id": "c01", "name": "Nan Reyes"},
            {"char_id": "announcer", "name": "ANNOUNCER"},
        ],
        "lines": [
            {
                "line_id": "L001",
                "char_id": "announcer",
                "speaker_role": "announcer",
                "text": "Tonight, from the lighthouse.",
            },
            {
                "line_id": "L002",
                "char_id": "c01",
                "speaker_role": "character",
                "text": "The lamp has not turned since Tuesday.",
            },
        ],
        "meta": {"episode_title": "The Still Lamp", "episode_seed": 7},
    }


class _RecordingSlot:
    """Stands in for the technical LLM slot; records that it was called."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls = 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        return self.response


# ---------------------------------------------------------------------------
# a complete ledger is left alone
# ---------------------------------------------------------------------------


def test_complete_ledger_is_a_no_op_and_costs_no_llm_call():
    ledger = _complete_ledger()
    slot = _RecordingSlot()
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=slot, bank_id="lighthouse")
    assert receipt["status"] == "complete"
    assert receipt["missing"] == []
    assert receipt["prose_fills"] == []
    assert receipt["safety"]["status"] == "retired"
    assert slot.calls == 0
    assert ledger["meta"]["ledger_cleanup"]["schema_version"] == (
        lc.LEDGER_CLEANUP_VERSION
    )


def test_cleanup_is_idempotent():
    ledger = _complete_ledger()
    first = lc.run_ledger_cleanup(ledger, slot_fn=None)
    second = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert first["status"] == second["status"] == "complete"
    assert second["deterministic_fills"] == []


# ---------------------------------------------------------------------------
# deterministic completion
# ---------------------------------------------------------------------------


def test_missing_line_id_is_minted_uniquely():
    ledger = _complete_ledger()
    ledger["lines"][1]["line_id"] = ""
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    minted = ledger["lines"][1]["line_id"]
    assert minted
    assert minted != ledger["lines"][0]["line_id"]


def test_duplicate_line_id_is_deduplicated_not_dropped():
    ledger = _complete_ledger()
    ledger["lines"][1]["line_id"] = "L001"
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    ids = [row["line_id"] for row in ledger["lines"]]
    assert len(ids) == len(set(ids)) == 2
    assert ledger["lines"][1]["text"].startswith("The lamp")


def test_blank_speaker_role_resolves_from_a_real_cast_char_id():
    ledger = _complete_ledger()
    ledger["lines"][1]["speaker_role"] = ""
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["lines"][1]["speaker_role"] == "character"


def test_blank_speaker_role_with_no_cast_row_is_named_not_guessed():
    ledger = _complete_ledger()
    ledger["lines"][1]["speaker_role"] = ""
    ledger["lines"][1]["char_id"] = "ghost"
    with pytest.raises(lc.LedgerIncompleteError) as excinfo:
        lc.run_ledger_cleanup(ledger, slot_fn=None, bank_id="lighthouse")
    assert any("speaker_role" in row for row in excinfo.value.missing)
    assert "lighthouse" in str(excinfo.value)


def test_an_announcer_row_with_no_cast_entry_is_not_a_gap():
    """The standing counter-example to "every voiced char_id needs a cast row".

    The ANNOUNCER speaks on nearly every episode, carries char_id="announcer",
    and lives in the Kokoro voice namespace instead of the cast's -- so it
    legitimately has no cast[] entry. The freeze gate requires only a
    non-empty char_id on a voiced row, and this pass must not be stricter
    than the authority it completes for.
    """
    ledger = _complete_ledger()
    ledger["cast"] = [{"char_id": "c01", "name": "Nan Reyes"}]
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"
    assert ledger["lines"][0]["char_id"] == "announcer"


def test_a_voiced_row_with_no_char_id_at_all_is_still_a_gap():
    ledger = _complete_ledger()
    ledger["lines"][1]["char_id"] = ""
    with pytest.raises(lc.LedgerIncompleteError) as excinfo:
        lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert any("char_id" in row for row in excinfo.value.missing)


def test_voiced_row_with_no_spoken_surface_becomes_an_explicit_skip():
    ledger = _complete_ledger()
    ledger["lines"][1]["text"] = "   "
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    row = ledger["lines"][1]
    assert row["skip"] is True
    assert row["text"] == ""
    assert row["tts_skip_reason"] == lc._EMPTY_TEXT_SKIP_REASON


def test_skip_row_with_text_keeps_the_text_and_drops_the_skip():
    """A row somebody wrote words into is not a skip. Keep the story."""
    ledger = _complete_ledger()
    ledger["lines"][1]["skip"] = True
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    row = ledger["lines"][1]
    assert row["skip"] is False
    assert row["text"] == "The lamp has not turned since Tuesday."


def test_skip_row_without_a_reason_gets_one():
    ledger = _complete_ledger()
    ledger["lines"].append({
        "line_id": "L003",
        "char_id": "c01",
        "speaker_role": "character",
        "text": "",
        "skip": True,
    })
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["lines"][2]["tts_skip_reason"]


def test_null_tts_skip_reason_becomes_an_empty_string():
    ledger = _complete_ledger()
    ledger["lines"][0]["tts_skip_reason"] = None
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["lines"][0]["tts_skip_reason"] == ""


def test_stale_text_metrics_are_restamped():
    ledger = _complete_ledger()
    ledger["lines"][1]["word_count"] = 999
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["lines"][1]["word_count"] != 999


def test_cast_row_without_a_char_id_is_minted_and_stays_referenceable():
    ledger = _complete_ledger()
    ledger["cast"].append({"char_id": "", "name": "Second Keeper"})
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["cast"][2]["char_id"]
    ids = [row["char_id"] for row in ledger["cast"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# THE LAW -- content is repaired, never a story-fail
# ---------------------------------------------------------------------------


def test_spoken_language_is_left_exactly_as_written(monkeypatch):
    """The safety repair pass is RETIRED -- the author's words survive.

    This test used to assert the opposite: that "damn" was rewritten out of a
    delivered row. Operator directive 2026-08-05 removed content guardrails
    from the generation path, so the cleanup must now leave the line alone. It
    is inverted rather than deleted so a re-armed repair pass fails loudly.
    """
    called = {"n": 0}

    def _must_not_run(ledger_data, slot_fn):
        called["n"] += 1
        return {"status": "patched", "patch_count": 1, "hits": []}

    monkeypatch.setattr(
        "nodes._otr_content_safety.apply_safety_cleanup", _must_not_run)

    ledger = _complete_ledger()
    original = "The damn lamp has not turned since Tuesday."
    ledger["lines"][1]["text"] = original
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=_RecordingSlot())

    assert receipt["status"] == "complete"
    assert ledger["lines"][1]["text"] == original, "the cleanup edited the author"
    assert called["n"] == 0, "the retired safety repair pass ran"
    assert receipt["safety"]["status"] == "retired"


def test_the_safety_receipt_key_survives_with_a_defined_value():
    """A ripped pass may not leave an unowned ledger field.

    meta.ledger_cleanup.safety is read by the freeze cascade's receipt
    plumbing, so the key keeps its established shape on EVERY path rather than
    disappearing with the pass that used to write it.
    """
    ledger = _complete_ledger()
    ledger["lines"][1]["text"] = "The damn lamp has not turned since Tuesday."
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"
    safety = receipt["safety"]
    assert safety["status"] == "retired"
    assert safety["patch_count"] == 0
    assert safety["residual_hits"] == 0
    assert ledger["meta"]["ledger_cleanup"]["safety"] == safety


def test_length_and_style_never_appear_in_the_gap_report():
    """A short, plain, unremarkable script is a complete ledger."""
    ledger = _complete_ledger()
    ledger["lines"][0]["text"] = "Hi."
    ledger["lines"][1]["text"] = "Ok."
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"


# ---------------------------------------------------------------------------
# prose completion
# ---------------------------------------------------------------------------


def test_missing_title_falls_back_to_the_source_when_the_llm_declines():
    ledger = _complete_ledger()
    ledger["meta"]["episode_title"] = ""
    ledger["meta"]["news"] = {"headline": "Lighthouse lamp stops turning"}
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"
    assert ledger["meta"]["episode_title"] == "Lighthouse lamp stops turning"
    assert receipt["prose_fills"][0]["action"] == "derived_from_source"


def test_missing_title_with_no_source_receipt_is_a_named_gap():
    ledger = _complete_ledger()
    ledger["meta"]["episode_title"] = ""
    with pytest.raises(lc.LedgerIncompleteError) as excinfo:
        lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert any("episode_title" in row for row in excinfo.value.missing)


def test_llm_title_fill_is_used_when_the_slot_answers(monkeypatch):
    ledger = _complete_ledger()
    ledger["meta"]["episode_title"] = ""
    monkeypatch.setattr(
        lc, "_llm_episode_title", lambda _ledger, _slot: "The Still Lamp")
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=_RecordingSlot())
    assert ledger["meta"]["episode_title"] == "The Still Lamp"
    assert receipt["prose_fills"][0]["action"] == "llm_filled"


# ---------------------------------------------------------------------------
# the seed receipt is owned elsewhere -- and minting one here breaks the tail
# ---------------------------------------------------------------------------


def test_the_pass_never_mints_or_touches_a_seed_receipt():
    """One owner per field, and neither seed owner is this pass.

    A legacy lane's seeded cast picker stamps cast_contract.cast_seed
    upstream; a content-owned lane's episode_seed is stamped by the writer
    tail right after this call. Minting one here also breaks reproducibility:
    a fresh seed is not derivable from the inputs, so the tail could no
    longer be byte-identical across two runs of the same inputs.
    """
    ledger = _complete_ledger()
    del ledger["meta"]["episode_seed"]
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"
    assert "episode_seed" not in ledger["meta"]
    assert not any("seed" in row for row in receipt["missing"])


def test_an_existing_seed_receipt_is_left_exactly_as_it_was():
    ledger = _complete_ledger()
    ledger["meta"]["cast_contract"] = {"cast_seed": 11}
    lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert ledger["meta"]["episode_seed"] == 7
    assert ledger["meta"]["cast_contract"]["cast_seed"] == 11


# ---------------------------------------------------------------------------
# scope -- fields owned by a DOWNSTREAM producer are not gaps here
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["style", "render_engines", "image_engines", "music_engine", "episode_id"],
)
def test_downstream_owned_meta_fields_are_not_reported_as_gaps(field):
    ledger = _complete_ledger()
    assert field not in ledger["meta"]
    gaps = lc.required_field_gaps(ledger)
    assert not any(field in row for row in gaps), gaps


def test_absent_timeline_and_clips_are_not_gaps_at_the_writer_boundary():
    ledger = _complete_ledger()
    assert "clips" not in ledger
    receipt = lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert receipt["status"] == "complete"


# ---------------------------------------------------------------------------
# the allowed-role authority is shared, never copied
# ---------------------------------------------------------------------------


def test_allowed_roles_come_from_the_freeze_module():
    from nodes._otr_ledger_freeze import ALLOWED_SPEAKER_ROLES

    assert lc._ALLOWED_SPEAKER_ROLES() is ALLOWED_SPEAKER_ROLES


def test_a_role_outside_the_shared_enum_is_a_named_gap():
    ledger = _complete_ledger()
    ledger["lines"][1]["speaker_role"] = "narrator"
    with pytest.raises(lc.LedgerIncompleteError) as excinfo:
        lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert any("narrator" in row for row in excinfo.value.missing)


def test_every_gap_is_reported_at_once_not_one_per_run():
    ledger = _complete_ledger()
    ledger["meta"]["episode_title"] = ""
    ledger["lines"][1]["speaker_role"] = "narrator"
    ledger["cast"][0]["name"] = ""
    with pytest.raises(lc.LedgerIncompleteError) as excinfo:
        lc.run_ledger_cleanup(ledger, slot_fn=None)
    assert len(excinfo.value.missing) >= 3
