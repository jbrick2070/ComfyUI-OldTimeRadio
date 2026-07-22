"""Live-backed regressions for requested budgets and split-row lineage.

PBUG-20260721-05/06: the legacy radio editor used a fixed 350-word target for
an explicitly requested 180-word episode, trusted an LLM word projection, and
minted synthetic beat ids for split line rows. Pure Python; no LLM/GPU.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from nodes._otr_ledger_consistency import assert_ledger_consistency
from nodes._otr_radio_editor import (
    BeatEdit,
    FinalWordDeliveryError,
    RadioEditPlan,
    apply_plan,
    fit_final_word_delivery,
    make_post_validator,
    needs_length_normalization,
    normalize_length,
    voiced_beats,
)
from nodes._otr_text_metrics import set_line_text_metrics
from nodes._otr_word_delivery import (
    character_text_sha256,
    delivery_word_bounds,
    stamp_contract,
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
    lower, upper = delivery_word_bounds(target)
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
                "band": band or [lower / target, upper / target],
            },
            "source_bank": "original",
            "visual_plan": {"characters": {}, "scenes": []},
        },
    }


def test_requested_budget_counts_character_body_not_announcer_overhead():
    # The common final law is 163..200 for target 180. Announcer overhead is
    # still excluded, so a 170-word character body plus 67 announcer words is
    # in-spec rather than a 237-word overrun.
    ledger = _ledger([34, 34, 34, 34, 34], [34, 33])

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


# ---------------------------------------------------------------------------
# PBUG-20260721-18: final inline word delivery
# ---------------------------------------------------------------------------


_FIT_BASE = (
    "I know the signal is real, and I will stay until we understand what it "
    "asks of us tonight."
)
_FIT_EXPANDED = (
    "I know the signal is real, and I will stay until we understand what it "
    "asks of us tonight. We must know it now."
)


def _fit_ledger() -> dict:
    ledger = _ledger([19, 19, 30, 30, 30, 30], target=180)
    set_line_text_metrics(ledger["lines"][0], _FIT_BASE)
    set_line_text_metrics(ledger["lines"][1], _FIT_BASE)
    assert sum(
        row["word_count"] for row in ledger["lines"]
        if row["speaker_role"] == "character"
    ) == 158
    stamp_contract(
        ledger["meta"],
        target_words=180,
        planned_voiced_words=180,
        owner="inline:original",
    )
    return ledger


def _fit_call(ledger: dict, creative_fn, technical_fn):
    return fit_final_word_delivery(
        ledger,
        creative_fn=creative_fn,
        technical_fn=technical_fn,
        creative_model="test/creative",
        technical_model="test/technical",
        source_bank_id="original",
        canon_header="SETTING: a signal room\nCAST: ALICE",
    )


def test_common_integer_delivery_law_covers_180_and_320_words():
    assert delivery_word_bounds(180) == (163, 200)
    assert delivery_word_bounds(320) == (289, 356)


def test_inline_fit_survives_failed_pair_then_repairs_one_row_only():
    ledger = _fit_ledger()
    before = copy.deepcopy(ledger["lines"])
    creative_outputs = iter([
        "Buy now during this commercial break for a new product.",
        _FIT_EXPANDED,
    ])

    def creative(*_args, **_kwargs):
        return next(creative_outputs)

    def technical(*_args, **_kwargs):
        return ""

    receipt = _fit_call(ledger, creative, technical)

    assert receipt["status"] == "pass"
    assert receipt["final_character_words"] == 163
    assert receipt["cycles"][0]["adopted"] is False
    assert "fake commercial" in receipt["cycles"][0]["attempts"][0]["error"]
    assert receipt["cycles"][1]["adopted"] is True
    assert receipt["cycles"][1]["accepted_slot"] == "creative"
    changed = [
        i for i, (old, new) in enumerate(zip(before, ledger["lines"]))
        if old != new
    ]
    assert changed == [1]
    for key in (
        "line_id", "beat_id", "char_id", "speaker_role", "dialogue_slot_id",
    ):
        assert ledger["lines"][1][key] == before[1][key]
    assert ledger["meta"]["word_budget"]["actual_text_sha256"] == (
        character_text_sha256(ledger)
    )


def test_inline_fit_rejects_no_progress_until_typed_exhaustion():
    ledger = _fit_ledger()
    before = copy.deepcopy(ledger["lines"])

    def empty(*_args, **_kwargs):
        return ""

    with pytest.raises(FinalWordDeliveryError, match="exhausted"):
        _fit_call(ledger, empty, empty)

    assert ledger["lines"] == before
    receipt = ledger["meta"]["inline_word_fit"]
    assert receipt["status"] == "repair_exhausted"
    assert receipt["actual_drift"] is True
    assert receipt["final_character_words"] == 158
    assert len(receipt["cycles"]) == receipt["repair_budget"]
