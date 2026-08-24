"""PBUG-20260802-02, third manifestation (shakespeare/MARIA, 2026-08-24).

A composition pass can allocate a cast slot and not serve it dialogue under a
tight beat budget. The freeze cascade's universal backstop
(``_otr_ledger_freeze.cast_coverage_gaps``) always refuses this correctly --
the negative control below proves the guard still fires -- but nothing
upstream ever tried to fix it. ``_otr_cast_coverage_repair`` does, inside the
writer's own tail, before the freeze cascade node ever runs.

THE STANDING PHILOSOPHY BEING TESTED (``_otr_cast_voice_coverage.py``): gates
only ever refuse; repair is the producer's job. So the tests here are less
"does the gate fire" (already covered) and more "does the repair leave a row
EXACTLY as it was on failure" -- the correctness property that makes this
change additive-only rather than a new way to ship a broken episode.
"""
from __future__ import annotations

import pytest

from nodes import _otr_cast_coverage_repair as repair
from nodes import _otr_freeze_cascade as freeze_cascade
from nodes._otr_ledger_freeze import cast_coverage_gaps
from nodes._otr_line_composer import LineCompositionFailedError
from nodes._otr_shakespeare_sources import cast_presence_from_text

MARIA_SCENE_TEXT = """[Enter Sir Toby, Sir Andrew, and Fabian.]

TOBY  Come thy ways, Signior Fabian.

[Enter Maria.]

TOBY  Here comes the little villain.

MARIA  Get you all three into the boxtree. Malvolio's
coming down this walk. Observe him.
[She exits.]

[Enter Malvolio.]

MALVOLIO  'Tis but fortune, all is fortune.
"""


def _base_ledger():
    """A minimal ledger shaped like the real freeze-gate contract: three cast
    rows, two with dialogue, one (MARIA) silent -- tonight's exact shape.
    Callers append their own MARIA lines[]/beats[] row (or none, for Mode 2)
    after calling this."""
    cast = [
        {"char_id": "c01", "name": "MALVOLIO", "traits": "vain steward",
         "voice_preset": "v2/en_speaker_1"},
        {"char_id": "c02", "name": "TOBY", "traits": "boisterous uncle",
         "voice_preset": "v2/en_speaker_2"},
        {"char_id": "c03", "name": "MARIA", "traits": "clever gentlewoman",
         "voice_preset": "v2/en_speaker_3"},
    ]
    lines = [
        {"line_id": "b001", "beat_id": "b001", "char_id": "c01",
         "speaker": "MALVOLIO", "text": "'Tis but fortune, all is fortune.",
         "skip": False, "arc_phase": "setup", "speaker_role": "character",
         "beat_intent": "Muse aloud.", "compose_flags": []},
        {"line_id": "b002", "beat_id": "b002", "char_id": "c02",
         "speaker": "TOBY", "text": "Here comes the little villain.",
         "skip": False, "arc_phase": "setup", "speaker_role": "character",
         "beat_intent": "Announce Malvolio.", "compose_flags": []},
    ]
    beats = [{"beat_id": r["beat_id"], "char_id": r["char_id"],
              "speaker": r["speaker"], "line_ids": [r["beat_id"]]}
             for r in lines]
    return {"cast": cast, "lines": lines, "beats": beats,
            "meta": {"source_bank": "shakespeare"}}


class _Led:
    """The minimal ``led`` shape ``repair_zero_coverage_cast`` needs: a
    ``.data`` attribute. Matches what ``led.data`` means everywhere else in
    this codebase without pulling in the full Ledger class."""

    def __init__(self, data):
        self.data = data


def _creative_fn(text_or_exc):
    def generate(messages, *, temperature, max_new_tokens, stop=None):
        if isinstance(text_or_exc, Exception):
            raise text_or_exc
        return text_or_exc
    return generate


# --------------------------------------------------------------------------- #
# cast_coverage_gaps -- the extraction is a behavior-preserving refactor
# --------------------------------------------------------------------------- #

def test_the_negative_control_still_fires():
    """A silent MARIA is a real gap. If this ever stops firing, the repair
    below is patching a defect that no longer exists to patch."""
    ledger = _base_ledger()
    assert ("c03", "MARIA") in cast_coverage_gaps(ledger)


def test_no_gaps_when_every_cast_member_speaks():
    ledger = _base_ledger()
    ledger["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "Get you all three into the boxtree.",
        "skip": False,
    })
    assert cast_coverage_gaps(ledger) == []


def test_a_skipped_line_still_counts_as_a_gap():
    ledger = _base_ledger()
    ledger["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True,
    })
    assert ("c03", "MARIA") in cast_coverage_gaps(ledger)


def test_the_announcer_is_never_a_gap():
    ledger = _base_ledger()
    ledger["cast"].append({"char_id": "c04", "name": "ANNOUNCER",
                            "voice_preset": "bm_george"})
    # no line at all for c04/ANNOUNCER
    assert not any(cid == "c04" for cid, _ in cast_coverage_gaps(ledger))


# --------------------------------------------------------------------------- #
# cast_presence_from_text -- the shakespeare fidelity graft's source data
# --------------------------------------------------------------------------- #

def test_presence_finds_marias_real_first_speech():
    presence = cast_presence_from_text(MARIA_SCENE_TEXT)
    assert presence["MARIA"]["line_count"] == 1
    assert presence["MARIA"]["first_speech"].startswith(
        "Get you all three into the boxtree.")


def test_presence_against_the_real_shipped_source_file():
    """Pinned against the actual file the operator's live leg drew from --
    a fixture text is not proof the real corpus parses the same way."""
    import io
    path = (
        "config/source_banks/shakespeare/sources/"
        "twelfth_night__act2_scene5.txt"
    )
    text = io.open(path, encoding="utf-8").read()
    presence = cast_presence_from_text(text)
    assert presence["MARIA"]["line_count"] == 3
    assert presence["MARIA"]["first_speech"].startswith(
        "Get you all three into the boxtree.")
    assert presence["TOBY"]["line_count"] == 28


def test_presence_is_empty_on_text_with_no_speaker_lines():
    assert cast_presence_from_text("just narration, no speakers here.") == {}


# --------------------------------------------------------------------------- #
# repair_zero_coverage_cast -- MODE 1 (an existing slot produced nothing)
# --------------------------------------------------------------------------- #

def test_mode1_fills_an_existing_silent_slot():
    data = _base_ledger()
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True,
        "tts_skip_reason": "empty_spoken_text_at_ledger_cleanup",
        "arc_phase": "setup", "beat_intent": "React to the trap.",
        "compose_flags": [],
    })
    led = _Led(data)

    receipt = repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("Get you all three into the boxtree."),
        canon_header="TITLE: Malvolio's Letter", style_descriptor="",
        source_bank_id="shakespeare", meta=data["meta"],
    )

    assert receipt["repaired"] == ["c03"]
    assert receipt["failed"] == []
    assert cast_coverage_gaps(data) == []
    row = next(r for r in data["lines"] if r["char_id"] == "c03")
    assert row["text"] == "Get you all three into the boxtree."
    assert row["skip"] is False
    assert "cast_coverage_repair" in row["compose_flags"]
    # Mode 1 never mints a row -- exactly one c03 line before and after.
    assert sum(1 for r in data["lines"] if r["char_id"] == "c03") == 1


# --------------------------------------------------------------------------- #
# repair_zero_coverage_cast -- MODE 2 (Stage 2 never allocated a beat at all)
# --------------------------------------------------------------------------- #

def test_mode2_mints_exactly_one_new_row_for_an_unallocated_cast_member():
    data = _base_ledger()  # MARIA has NO lines/beats row at all
    led = _Led(data)

    receipt = repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("Observe him, for the love of mockery."),
        canon_header="TITLE: Malvolio's Letter", style_descriptor="",
        source_bank_id="shakespeare", meta=data["meta"],
    )

    assert receipt["repaired"] == ["c03"]
    assert cast_coverage_gaps(data) == []
    maria_lines = [r for r in data["lines"] if r["char_id"] == "c03"]
    assert len(maria_lines) == 1
    assert maria_lines[0]["text"] == "Observe him, for the love of mockery."
    assert maria_lines[0]["skip"] is False
    maria_beats = [b for b in data["beats"] if b["char_id"] == "c03"]
    assert len(maria_beats) == 1


def test_mode2_beat_id_never_collides():
    data = _base_ledger()
    led = _Led(data)
    repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("A line."), canon_header="",
        style_descriptor="", source_bank_id="shakespeare", meta=data["meta"],
    )
    ids = [r["beat_id"] for r in data["lines"]]
    assert len(ids) == len(set(ids)), f"beat_id collision: {ids}"


def test_mode2_row_is_never_added_to_a_pydantic_outline():
    """The repair is ledger-only. Nothing here should even import an Outline
    class -- if it did, this test would need to construct one to prove
    non-membership, which is itself the signal something drifted."""
    import nodes._otr_cast_coverage_repair as mod
    assert "Outline" not in dir(mod)


# --------------------------------------------------------------------------- #
# Failure leaves the row byte-identical to today's refuse-and-halt
# --------------------------------------------------------------------------- #

def test_composition_exhaustion_leaves_the_row_untouched():
    data = _base_ledger()
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True,
        "tts_skip_reason": "empty_spoken_text_at_ledger_cleanup",
        "compose_flags": [],
    })
    before = [dict(r) for r in data["lines"]]
    led = _Led(data)

    receipt = repair.repair_zero_coverage_cast(
        led,
        creative_fn=_creative_fn(
            LineCompositionFailedError([("", "empty")], None)),
        canon_header="", style_descriptor="", source_bank_id="shakespeare",
        meta=data["meta"],
    )

    assert receipt["repaired"] == []
    assert receipt["failed"] == ["c03"]
    assert data["lines"] == before
    # The gate must still fire -- a failed repair is not silently accepted.
    assert ("c03", "MARIA") in cast_coverage_gaps(data)


def test_an_empty_composed_line_counts_as_failure_not_success():
    data = _base_ledger()
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True, "compose_flags": [],
    })
    led = _Led(data)
    receipt = repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("   "), canon_header="",
        style_descriptor="", source_bank_id="shakespeare", meta=data["meta"],
    )
    assert receipt["failed"] == ["c03"]
    assert cast_coverage_gaps(data) != []


def test_no_gaps_is_a_pure_no_op():
    data = _base_ledger()
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "A real line.", "skip": False,
    })
    before_lines = [dict(r) for r in data["lines"]]
    before_cast = [dict(r) for r in data["cast"]]
    led = _Led(data)

    receipt = repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("should never be called"),
        canon_header="", style_descriptor="", source_bank_id="shakespeare",
        meta=data["meta"],
    )

    assert receipt == {
        "schema_version": repair.SCHEMA_VERSION,
        "attempted": [], "repaired": [], "failed": [],
    }
    assert data["lines"] == before_lines
    assert data["cast"] == before_cast


# --------------------------------------------------------------------------- #
# cast[] and cast_seed are never touched -- CastLock's replay contract holds
# --------------------------------------------------------------------------- #

def test_repair_never_touches_cast_or_cast_contract():
    data = _base_ledger()
    data["meta"]["cast_contract"] = {
        "cast_seed": 424242, "num_characters_request": 3,
    }
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True, "compose_flags": [],
    })
    before_cast = [dict(r) for r in data["cast"]]
    before_contract = dict(data["meta"]["cast_contract"])
    led = _Led(data)

    repair.repair_zero_coverage_cast(
        led, creative_fn=_creative_fn("A real line."), canon_header="",
        style_descriptor="", source_bank_id="shakespeare", meta=data["meta"],
    )

    assert data["cast"] == before_cast
    assert data["meta"]["cast_contract"] == before_contract


# --------------------------------------------------------------------------- #
# The shakespeare fidelity graft -- verbatim source, not free invention
# --------------------------------------------------------------------------- #

def test_the_fidelity_graft_grounds_the_repair_in_marias_real_words():
    data = _base_ledger()
    data["meta"]["source_meta"] = {
        "scene_label": "Twelfth Night, Act 2, Scene 5",
        "cast_hints_presence": cast_presence_from_text(MARIA_SCENE_TEXT),
    }
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True, "compose_flags": [],
    })
    led = _Led(data)
    captured = {}

    def generate(messages, *, temperature, max_new_tokens, stop=None):
        captured["prompt"] = "\n".join(m.get("content", "") for m in messages)
        return "Get you all three into the boxtree."

    repair.repair_zero_coverage_cast(
        led, creative_fn=generate, canon_header="", style_descriptor="",
        source_bank_id="shakespeare", meta=data["meta"],
    )

    assert "boxtree" in captured["prompt"]
    assert "SOURCE" in captured["prompt"]


def test_no_fidelity_data_composes_without_a_source_block():
    """A bank/scene with no cast_hints_presence must compose exactly like any
    other repair line -- absence of grounding data is not an error."""
    data = _base_ledger()
    data["lines"].append({
        "line_id": "b003", "beat_id": "b003", "char_id": "c03",
        "speaker": "MARIA", "text": "", "skip": True, "compose_flags": [],
    })
    led = _Led(data)
    captured = {}

    def generate(messages, *, temperature, max_new_tokens, stop=None):
        captured["prompt"] = "\n".join(m.get("content", "") for m in messages)
        return "A freely composed line."

    receipt = repair.repair_zero_coverage_cast(
        led, creative_fn=generate, canon_header="", style_descriptor="",
        source_bank_id="shakespeare", meta=data["meta"],
    )

    assert receipt["repaired"] == ["c03"]
    assert "SOURCE" not in captured["prompt"]


# --------------------------------------------------------------------------- #
# The gate this repair runs under -- config-driven, never a hardcoded list
# --------------------------------------------------------------------------- #

def test_shakespeare_is_gated_in_scifi_news_pro_is_gated_out():
    """The writer-tail call site skips this repair entirely for
    content_owned_readonly lanes (their own earlier gate already covers
    them, and a deliberately-silent entity like a Relay must never be forced
    to speak). This is resolve_freeze_policy's real, live classification --
    not a hardcoded bank list, which is exactly what would drift."""
    shakespeare_policy = freeze_cascade.resolve_freeze_policy(
        {"source_bank": "shakespeare"})
    scifi_news_pro_policy = freeze_cascade.resolve_freeze_policy(
        {"source_bank": "scifi_news_pro"})
    assert shakespeare_policy.run_inline_safety_cleanup is True
    assert scifi_news_pro_policy.run_inline_safety_cleanup is False
