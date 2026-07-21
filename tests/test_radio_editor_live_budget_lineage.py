"""Live-backed regressions for requested budgets and split-row lineage.

PBUG-20260721-05/06: the legacy radio editor used a fixed 350-word target for
an explicitly requested 180-word episode, trusted an LLM word projection, and
minted synthetic beat ids for split line rows. Pure Python; no LLM/GPU.
"""
from __future__ import annotations

from types import SimpleNamespace

from nodes._otr_ledger_consistency import assert_ledger_consistency
from nodes._otr_radio_editor import (
    BeatEdit,
    RadioEditPlan,
    apply_plan,
    make_post_validator,
    needs_length_normalization,
    normalize_length,
    voiced_beats,
)
from nodes.production_ledger import Ledger


def _words(count: int, token: str = "word") -> str:
    return " ".join([token] * count)


def _line(
    line_id: str,
    words: int,
    *,
    role: str = "character",
    beat_id: str | None = None,
    token: str = "word",
) -> dict:
    text = _words(words, token)
    return {
        "line_id": line_id,
        "beat_id": beat_id or line_id,
        "char_id": "announcer" if role == "announcer" else "c01",
        "speaker_role": role,
        "text": text,
        "char_count": len(text),
        "word_count": words,
        "dialogue_slot_id": f"d{line_id}",
        "compose_flags": [],
    }


def _ledger(
    character_counts: list[int],
    announcer_counts: list[int] | None = None,
    *,
    target: int = 180,
    band: list[float] | None = None,
) -> dict:
    lines = [
        _line(f"b{i:03d}", count)
        for i, count in enumerate(character_counts, start=1)
    ]
    start = len(lines) + 1
    lines.extend(
        _line(f"b{i:03d}", count, role="announcer")
        for i, count in enumerate(announcer_counts or [], start=start)
    )
    beats = [
        {"beat_id": row["beat_id"], "line_ids": [row["line_id"]]}
        for row in lines
    ]
    return {
        "cast": [{"char_id": "c01", "name": "ALICE"}],
        "beats": beats,
        "lines": lines,
        "meta": {
            "word_budget": {
                "target_words": target,
                "band": band or [0.7, 1.3],
            },
            "visual_plan": {"characters": {}, "scenes": []},
        },
    }


def test_requested_budget_counts_character_body_not_announcer_overhead():
    # Reproduces the stopped live shape: 148 character words plus 67
    # announcer words is a good 180-word body, not a 215-word overrun.
    ledger = _ledger([30, 30, 30, 29, 29], [34, 33])

    assert needs_length_normalization(ledger) is False

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("an in-band requested body must not call an LLM")

    plan, report = normalize_length(
        ledger,
        editor_model="test-creative",
        slot_fn=must_not_call,
        recompose_fn=lambda _i, text, _hint: text,
    )
    assert plan is None
    assert report["status"] == "SKIPPED_IN_SPEC"


def test_announcer_still_owns_the_spoken_breath_cap():
    ledger = _ledger([30, 30, 30, 30, 30, 30], [36])
    assert needs_length_normalization(ledger) is True


def test_validator_accepts_good_actual_despite_false_model_projection():
    ledger = _ledger([30, 30, 30, 30, 30, 30])
    plan = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="KEEP")],
        projected_word_total=999,
    )

    error = make_post_validator(voiced_beats(ledger), ledger)(plan)

    assert error is None
    assert plan.projected_word_total == 999  # preserve the forensic LLM claim


def test_validator_rejects_bad_actual_despite_good_model_projection():
    ledger = _ledger([30] * 10)
    plan = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="KEEP")],
        projected_word_total=180,
    )

    error = make_post_validator(voiced_beats(ledger), ledger)(plan)

    assert error is not None
    assert "simulated character-word total 300" in error
    assert "projected_word_total=180" in error


def test_micro_repair_can_fix_a_row_during_advisory_episode_drift():
    ledger = _ledger([30, 30])  # intentionally below the advisory band
    clean_plan = RadioEditPlan(
        edits=[BeatEdit(
            beat_index=0,
            action="SHORTEN_LINE",
            new_line=_words(10),
        )],
        projected_word_total=40,
    )

    local_validator = make_post_validator(
        voiced_beats(ledger), ledger, enforce_word_band=False
    )
    global_validator = make_post_validator(
        voiced_beats(ledger), ledger, enforce_word_band=True
    )
    assert local_validator(clean_plan) is None
    assert global_validator(clean_plan) is not None

    over_cap_plan = RadioEditPlan(
        edits=[BeatEdit(
            beat_index=0,
            action="SHORTEN_LINE",
            new_line=_words(36),
        )],
        projected_word_total=66,
    )
    assert "Guard2" in str(local_validator(over_cap_plan))


def test_malformed_present_budget_skips_without_mutating_or_calling_llm():
    ledger = _ledger([30, 30])
    ledger["meta"]["word_budget"]["band"] = "not-a-ratio-pair"
    before = [dict(row) for row in ledger["lines"]]

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("a malformed budget must not call an LLM")

    plan, report = normalize_length(
        ledger,
        editor_model="test-creative",
        slot_fn=must_not_call,
        recompose_fn=lambda _i, text, _hint: text,
    )
    assert plan is None
    assert report["status"] == "SKIPPED_INVALID_BUDGET"
    assert ledger["lines"] == before


def test_split_child_keeps_parent_beat_and_syncs_retained_beat_membership():
    ledger = _ledger([10, 10])
    outline = {
        "beats": [
            {"beat_id": "b001"},
            {"beat_id": "b002"},
        ]
    }
    plan = RadioEditPlan(
        edits=[
            BeatEdit(
                beat_index=0,
                action="SPLIT_LINE",
                new_line="First sentence stays here. Second sentence follows.",
            ),
            BeatEdit(beat_index=1, action="CUT_LINE"),
        ],
        projected_word_total=8,
    )

    report = apply_plan(ledger, plan)

    assert [(row["line_id"], row["beat_id"]) for row in ledger["lines"]] == [
        ("b001", "b001"),
        ("b001_s1", "b001"),
    ]
    by_beat = {row["beat_id"]: row for row in ledger["beats"]}
    assert by_beat["b001"]["line_ids"] == ["b001", "b001_s1"]
    assert by_beat["b002"]["line_ids"] == []
    assert report["added_line_ids"] == ["b001_s1"]
    assert report["removed_line_ids"] == ["b002"]
    assert report["added_beat_ids"] is report["added_line_ids"]
    assert assert_ledger_consistency(outline=outline, ledger=ledger) == []


def test_repeated_split_pass_never_reuses_an_existing_child_line_id():
    ledger = _ledger([10])
    ledger["lines"].append(_line("b001_s1", 5, beat_id="b001"))
    ledger["beats"][0]["line_ids"] = ["b001", "b001_s1"]
    plan = RadioEditPlan(
        edits=[BeatEdit(
            beat_index=0,
            action="SPLIT_LINE",
            new_line="First sentence stays here. Another sentence is new.",
        )],
        projected_word_total=8,
    )

    apply_plan(ledger, plan)

    line_ids = [row["line_id"] for row in ledger["lines"]]
    assert line_ids == ["b001", "b001_s2", "b001_s1"]
    assert len(line_ids) == len(set(line_ids))
    assert {row["beat_id"] for row in ledger["lines"]} == {"b001"}
    assert ledger["beats"][0]["line_ids"] == line_ids


def test_outline_initialization_materializes_the_durable_beat_collection(tmp_path):
    outline = SimpleNamespace(beats=[
        SimpleNamespace(
            beat_id="b001",
            speaker="ALICE",
            speaker_role="character",
            intent="answers the signal",
            mood="steady",
            target_words=20,
            arc_phase="setup",
            dialogue_slot_id="d001",
        ),
        SimpleNamespace(
            beat_id="b002",
            speaker="ANNOUNCER",
            speaker_role="announcer",
            intent="closes",
            mood="warm",
            target_words=12,
            arc_phase="resolution",
            dialogue_slot_id="d002",
        ),
    ])
    ledger = Ledger("pending_test", str(tmp_path))

    ledger.init_lines_from_outline(outline, {"ALICE": "c01"})

    assert [row["beat_id"] for row in ledger.data["beats"]] == ["b001", "b002"]
    assert [row["line_ids"] for row in ledger.data["beats"]] == [
        ["b001"],
        ["b002"],
    ]
    assert ledger.data["total_beats"] == 2
