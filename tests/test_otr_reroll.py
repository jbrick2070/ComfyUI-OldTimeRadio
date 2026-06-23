"""tests/test_otr_reroll.py -- Sprint 5C targeted reroll loop.

Covers `nodes/_otr_reroll.py` and the composer-side `reroll_hint` surface:

  1. `build_reroll_line_request` reconstructs a LineRequest from meta +
     the ledger line row, including a scene-local last_lines window.
  2. `_build_user_prompt` renders the REVISE block when `reroll_hint` is
     set and drops it when empty.
  3. the `reroll_hint` keyword param on `compose_line` overlays the
     critic hint onto the (frozen) request so the REVISE block reaches
     the prompt.
  4. `run_targeted_reroll` is a no-op when the critic named no targets.
  5. `run_targeted_reroll` resolves: rerolls the flagged line, the
     critic re-run comes back clean -> `reroll_resolved`, line updated.
  6. `run_targeted_reroll` exhausts the cap: the critic keeps flagging
     -> `needs_full_rerun`, the pre-reroll lines are restored.
  7. `run_targeted_reroll` NEVER raises on a raising generate_fn.
  8. a reroll target whose line_id is absent from the ledger is recorded
     in reroll_history and skipped without crashing.

Pure-Python. Fake `generate_fn` for every LLM call -- NO GPU, NO HF, NO
real model. Mirrors the fake-generate-fn pattern in
`tests/test_otr_story_critic.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    compose_line,
)
from nodes._otr_reroll import (  # noqa: E402
    MAX_REROLL_CYCLES,
    RerollDisposition,
    build_reroll_line_request,
    run_targeted_reroll,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_GOOD_DIALOGUE = "I never touched that safe and you know it."


def _report_with_target(line_id="l001", hint="give the line a sharper edge"):
    """A StoryCriticReport JSON string flagging one reroll target."""
    return json.dumps({
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [{"line_id": line_id, "reason": "states a fact"}],
        "arc_verdict": "uneven",
        "reroll_targets": [{"line_id": line_id, "hint": hint}],
        "render_priority": ["l001", "l002"],
    })


def _clean_report():
    """A StoryCriticReport JSON string with no reroll targets."""
    return json.dumps({
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "strong",
        "reroll_targets": [],
        "render_priority": ["l001", "l002"],
    })


def _mk_reroll_generate_fn(critic_replies, dialogue=_GOOD_DIALOGUE):
    """Fake generate_fn serving BOTH the composer and the critic.

    Routes by the system prompt: a critic call (its system prompt names
    a 'story-quality critic') gets the next canned JSON report; any other
    call is a composer call and gets canned dialogue. Absorbs the
    composer's `stop=` kwarg via **kwargs.
    """
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens, **kwargs):
        sys_text = (messages[0].get("content") if messages else "") or ""
        if "critic" in sys_text.lower():
            i = state["i"]
            state["i"] += 1
            if i < len(critic_replies):
                return critic_replies[i]
            return critic_replies[-1] if critic_replies else "{}"
        return dialogue

    return fn


def _raising_generate_fn():
    """Fake generate_fn that always raises -- an LLM loader crash."""

    def fn(messages, *, temperature, max_new_tokens, **kwargs):
        raise RuntimeError("simulated model loader failure")

    return fn


class _FakeLedger:
    """Minimal `production_ledger.Ledger`-like stub: `.data` plus the
    `update_line_text` / `_recompute_totals` surface `run_targeted_reroll`
    uses.
    """

    def __init__(self, data: dict):
        self.data = data
        self.recompute_calls = 0

    def update_line_text(self, beat_id, text, compose_flags_append=None) -> bool:
        # Mirror Ledger.update_line_text (C0 2026-06-22): accept + apply
        # compose_flags_append with the same safe-list append semantics so the
        # reroll path under test exercises the real signature.
        safe = text or ""
        for row in self.data.get("lines", []):
            if (row.get("beat_id") == beat_id
                    or row.get("line_id") == beat_id):
                row["text"] = safe
                row["word_count"] = len(safe.split())
                row["char_count"] = len(safe)
                if compose_flags_append:
                    existing = row.get("compose_flags")
                    if not isinstance(existing, list):
                        existing = []
                    for _f in compose_flags_append:
                        existing.append(_f)
                    row["compose_flags"] = existing
                return True
        return False

    def _recompute_totals(self):
        self.recompute_calls += 1


def _ledger_data(story_critic_report=None):
    """A small post-critic ledger dict -- the `Ledger.data` shape.

    Two character lines plus one announcer beat. `meta` carries the
    Sprint 5C writer-stamped composer context.
    """
    meta = {
        "target_words": 500,
        "style": "noir mystery",
        "canon_header": "EPISODE_TITLE: TBD\nSETTING: a shuttered bank.",
        "outline_spine": "1. the discovery\n2. the accusation",
        "theme": "trust frays under pressure",
        "style_descriptor": "noir_mystery_clipped",
        "allowed_roster": ["EDNA", "MARLOWE", "ANNOUNCER"],
    }
    if story_critic_report is not None:
        meta["story_critic_report"] = story_critic_report
    return {
        "meta": meta,
        "cast": [
            {"char_id": "c01", "name": "EDNA", "speaker_role": "character",
             "gender": "female",
             "character_description": "weary bank clerk in her 50s"},
            {"char_id": "c02", "name": "MARLOWE", "speaker_role": "character",
             "gender": "male", "character_description": "tired detective"},
            {"char_id": "announcer", "name": "ANNOUNCER",
             "speaker_role": "announcer"},
        ],
        "lines": [
            {
                "line_id": "l001", "beat_id": "l001", "char_id": "c01",
                "speaker_role": "character", "arc_phase": "setup",
                "traits": "wary", "beat_intent": "establish the theft",
                "target_words": 14, "word_count": 9,
                "text": "The safe was open when I walked in this morning.",
            },
            {
                "line_id": "l002", "beat_id": "l002", "char_id": "c02",
                "speaker_role": "character", "arc_phase": "rising",
                "traits": "probing", "beat_intent": "press the witness",
                "target_words": 12, "word_count": 8,
                "text": "And you are certain nobody else had a key?",
            },
            {
                "line_id": "l003", "beat_id": "l003", "char_id": "announcer",
                "speaker_role": "announcer", "arc_phase": "setup",
                "text": "We return after this word from our sponsor.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# 1. build_reroll_line_request
# ---------------------------------------------------------------------------


def test_build_reroll_line_request_pulls_meta_and_row_context():
    data = _ledger_data()
    row = data["lines"][1]  # l002 -- MARLOWE, has l001 before it
    req = build_reroll_line_request(data, row, data["cast"])

    assert req.speaker == "MARLOWE"
    assert req.intent == "press the witness"
    assert req.mood == "probing"
    assert req.target_words == 12
    assert req.arc_phase == "rising"
    assert req.canon_header == data["meta"]["canon_header"]
    assert req.outline_spine == data["meta"]["outline_spine"]
    assert req.theme == data["meta"]["theme"]
    assert req.style_descriptor == data["meta"]["style_descriptor"]
    assert req.allowed_roster == frozenset({"EDNA", "MARLOWE", "ANNOUNCER"})
    assert req.speaker_role == "character"
    # scene-local last_lines: l001 precedes l002 with no boundary.
    assert req.last_lines == [
        ("EDNA", "The safe was open when I walked in this morning."),
    ]
    assert req.prev_speaker == "EDNA"
    assert "MARLOWE" in req.character_voice_card


def test_build_reroll_line_request_threads_story_quality_flag():
    """L2 (R3): build_reroll_line_request threads meta.story_quality_v2_enabled
    so a reroll composes under the SAME authoring contract the first pass did.
    Absent key => False (byte-identical pre-R3 reroll)."""
    data = _ledger_data()
    row = data["lines"][1]
    # absent -> False
    assert build_reroll_line_request(
        data, row, data["cast"]
    ).story_quality_v2_enabled is False
    # explicit True -> True
    data["meta"]["story_quality_v2_enabled"] = True
    assert build_reroll_line_request(
        data, row, data["cast"]
    ).story_quality_v2_enabled is True


def test_build_reroll_line_request_target_words_fallback():
    """A row missing target_words falls back to word_count, then a const."""
    data = _ledger_data()
    row = dict(data["lines"][0])
    row["target_words"] = None
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.target_words == 9  # falls back to word_count

    row["word_count"] = 0
    req2 = build_reroll_line_request(data, row, data["cast"])
    assert req2.target_words == 30  # _FALLBACK_TARGET_WORDS


def test_build_reroll_line_request_reconstructs_dramatic_frame():
    """STEP 6 (2026-06-22): a reroll reconstructs the per-line dramatic frame
    the writer stamped on meta.line_dramatic_frame -- objective / obstacle /
    turn / subtext / tension / dramatic_question / next_turn. Before this,
    build_reroll_line_request rebuilt only arc_phase, so a rerolled line was
    composed blind to the frame (and tension) it was meant to repair against."""
    data = _ledger_data()
    data["meta"]["line_dramatic_frame"] = {
        "l002": {
            "objective": "press the witness to crack",
            "obstacle": "she will lose her job if she talks",
            "turn": "guarded -> confessing",
            "subtext": "he already knows she lied",
            "tension": 4,
            "dramatic_question": "Who opened the safe?",
            "next_turn": "the confession lands",
        }
    }
    row = data["lines"][1]  # l002
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.beat_objective == "press the witness to crack"
    assert req.beat_obstacle == "she will lose her job if she talks"
    assert req.beat_turn == "guarded -> confessing"
    assert req.beat_subtext == "he already knows she lied"
    assert req.beat_tension == 4
    assert req.dramatic_question == "Who opened the safe?"
    assert req.next_turn == "the confession lands"


def test_build_reroll_line_request_no_frame_is_empty_defaults():
    """No meta.line_dramatic_frame -> the reroll frame fields default empty /
    tension 0 (PD1: never raise; legacy ledgers reroll unchanged)."""
    data = _ledger_data()
    row = data["lines"][1]
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.beat_objective == ""
    assert req.beat_turn == ""
    assert req.beat_tension == 0


def test_build_reroll_line_request_rejects_out_of_range_tension():
    """A malformed tension on the frame coerces to 0 (unset), never renders."""
    data = _ledger_data()
    data["meta"]["line_dramatic_frame"] = {"l002": {"tension": 9}}
    req = build_reroll_line_request(data, data["lines"][1], data["cast"])
    assert req.beat_tension == 0


# ---------------------------------------------------------------------------
# 1b. continuity_slice rebuild on reroll (v2 follow-up, 2026-05-25)
# ---------------------------------------------------------------------------


def _continuity_dump(*, location="A shuttered bank, after hours.",
                     active_props=None, facts=None):
    """Build a meta['continuity'] dump in the ContinuityState shape.

    Returns the JSON-friendly dict the writer stamps via
    ``continuity_state.model_dump()`` so tests can drop it onto
    ``meta['continuity']`` without importing pydantic.
    """
    return {
        "location": location,
        "active_props": list(active_props or []),
        "facts": list(facts or []),
    }


def _fact(text, *, known_by=None, hidden_from=None, established_beat=0):
    return {
        "fact": text,
        "known_by": list(known_by or []),
        "hidden_from": list(hidden_from or []),
        "established_beat": int(established_beat),
    }


def test_build_reroll_line_request_continuity_slice_surfaces_known_fact():
    """A fact MARLOWE knows, established at beat 0, surfaces at index 1."""
    data = _ledger_data()
    data["meta"]["continuity"] = _continuity_dump(
        active_props=["a forced lockbox"],
        facts=[
            _fact("Edna lied about the lockbox being sealed.",
                  known_by=["MARLOWE"], established_beat=0),
        ],
    )
    row = data["lines"][1]  # MARLOWE at index 1
    req = build_reroll_line_request(data, row, data["cast"])

    assert req.continuity_slice, "expected a non-empty continuity slice"
    assert "CONTINUITY CONSTRAINTS" in req.continuity_slice
    assert "MARLOWE KNOWS" in req.continuity_slice
    assert "Edna lied about the lockbox" in req.continuity_slice
    assert "A shuttered bank" in req.continuity_slice
    assert "a forced lockbox" in req.continuity_slice


def test_build_reroll_line_request_continuity_slice_filters_unestablished_fact():
    """A fact with established_beat > line_index is filtered out for known_by."""
    data = _ledger_data()
    data["meta"]["continuity"] = _continuity_dump(
        location="",  # drop the location header so we test facts in isolation
        facts=[
            _fact("Marlowe finds the second key in the desk.",
                  known_by=["MARLOWE"], established_beat=2),
        ],
    )
    row = data["lines"][1]  # MARLOWE at index 1 -- BEFORE the fact is true
    req = build_reroll_line_request(data, row, data["cast"])

    # The future-fact has nowhere else to surface and the location is
    # empty, so the whole slice should collapse to "".
    assert req.continuity_slice == ""


def test_build_reroll_line_request_continuity_slice_surfaces_hidden_fact_always():
    """A hidden_from fact ignores established_beat -- a secret stays a secret."""
    data = _ledger_data()
    data["meta"]["continuity"] = _continuity_dump(
        location="",
        facts=[
            _fact("Edna is the saboteur the police are looking for.",
                  hidden_from=["MARLOWE"], established_beat=99),
        ],
    )
    row = data["lines"][1]  # MARLOWE at index 1, fact's established_beat=99
    req = build_reroll_line_request(data, row, data["cast"])

    assert "MARLOWE must NOT reference" in req.continuity_slice
    assert "Edna is the saboteur" in req.continuity_slice


def test_build_reroll_line_request_continuity_slice_empty_when_meta_missing():
    """No meta['continuity'] -> empty slice (the writer never stamped it)."""
    data = _ledger_data()
    assert "continuity" not in data["meta"]
    row = data["lines"][1]
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.continuity_slice == ""


def test_build_reroll_line_request_continuity_slice_empty_when_neutral():
    """A neutral state -- no location, no props, no facts -- renders empty."""
    data = _ledger_data()
    data["meta"]["continuity"] = _continuity_dump(
        location="", active_props=[], facts=[],
    )
    row = data["lines"][1]
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.continuity_slice == ""


def test_build_reroll_line_request_continuity_slice_never_raises_on_garbage():
    """Malformed meta['continuity'] degrades to empty -- Prime Directive 1."""
    data = _ledger_data()
    # Wrong shape for ContinuityState.model_validate: facts is a string.
    data["meta"]["continuity"] = {
        "location": "fine",
        "active_props": [],
        "facts": "not a list of facts",
    }
    row = data["lines"][1]
    req = build_reroll_line_request(data, row, data["cast"])
    assert req.continuity_slice == ""

    # And a totally wrong type for meta['continuity'] itself.
    data["meta"]["continuity"] = "not even a dict"
    req2 = build_reroll_line_request(data, row, data["cast"])
    assert req2.continuity_slice == ""


# ---------------------------------------------------------------------------
# 2. REVISE block rendering in _build_user_prompt
# ---------------------------------------------------------------------------


def _minimal_request(**over):
    base = dict(
        speaker="EDNA", intent="press the point", mood="tense",
        target_words=12, canon_header="EPISODE_TITLE: TBD", last_lines=[],
    )
    base.update(over)
    return LineRequest(**base)


def test_revise_block_renders_when_reroll_hint_set():
    req = _minimal_request(reroll_hint="open on her fear, not the inventory")
    prompt = _build_user_prompt(req)
    assert "REVISE:" in prompt
    assert "open on her fear, not the inventory" in prompt


def test_revise_block_absent_when_reroll_hint_empty():
    req = _minimal_request()
    prompt = _build_user_prompt(req)
    assert "REVISE:" not in prompt


# ---------------------------------------------------------------------------
# 3. reroll_hint keyword param on compose_line
# ---------------------------------------------------------------------------


def test_compose_line_reroll_hint_param_reaches_prompt():
    captured = []

    def fn(messages, *, temperature, max_new_tokens, **kwargs):
        captured.append(messages)
        return _GOOD_DIALOGUE

    req = _minimal_request()  # no reroll_hint on the request itself
    compose_line(
        creative_fn=fn, req=req,
        reroll_hint="land the betrayal harder",
    )
    assert captured, "generate_fn was never called"
    user_msg = captured[0][1]["content"]
    assert "REVISE:" in user_msg
    assert "land the betrayal harder" in user_msg


# ---------------------------------------------------------------------------
# 4. no reroll targets -> no-op
# ---------------------------------------------------------------------------


def test_no_reroll_targets_is_a_noop():
    data = _ledger_data(story_critic_report=json.loads(_clean_report()))
    led = _FakeLedger(data)
    original = data["lines"][0]["text"]

    disp = run_targeted_reroll(_raising_generate_fn(), led)

    assert isinstance(disp, RerollDisposition)
    assert disp.verdict == "no_reroll"
    assert disp.cycles_run == 0
    assert disp.lines_rerolled == 0
    assert data["lines"][0]["text"] == original  # untouched
    assert data["meta"]["cycle_count"] == 0
    assert data["meta"]["reroll_verdict"] == "no_reroll"


def test_missing_critic_report_is_a_noop():
    """No meta.story_critic_report at all -> treated as an empty report."""
    led = _FakeLedger(_ledger_data())  # no report stamped
    disp = run_targeted_reroll(_raising_generate_fn(), led)
    assert disp.verdict == "no_reroll"
    assert disp.cycles_run == 0


# ---------------------------------------------------------------------------
# 5. reroll resolves
# ---------------------------------------------------------------------------


def test_reroll_resolved_updates_line_and_stamps_meta():
    data = _ledger_data(story_critic_report=json.loads(_report_with_target()))
    led = _FakeLedger(data)
    original = data["lines"][0]["text"]
    # The critic re-run after cycle 1 comes back clean.
    fn = _mk_reroll_generate_fn([_clean_report()])

    disp = run_targeted_reroll(fn, led)

    assert disp.verdict == "reroll_resolved"
    assert disp.cycles_run == 1
    assert disp.lines_rerolled == 1
    # l001 was re-composed -- its text changed.
    assert data["lines"][0]["text"] != original
    assert data["lines"][0]["text"].strip() != ""
    # meta audit trail.
    assert data["meta"]["cycle_count"] == 1
    assert data["meta"]["reroll_verdict"] == "reroll_resolved"
    history = data["meta"]["reroll_history"]
    assert len(history) == 1
    assert history[0]["status"] == "rerolled"
    assert history[0]["line_id"] == "l001"
    assert history[0]["old_text"] == original
    # the final (clean) critic re-run is restamped.
    assert data["meta"]["story_critic_report"]["reroll_targets"] == []
    assert data["meta"]["story_critic_report"]["arc_verdict"] == "strong"


# ---------------------------------------------------------------------------
# 6. reroll exhausts the cap -> needs_full_rerun, KEEPS repairs (STEP 4
#    repair-then-ship -- no pre-reroll restore)
# ---------------------------------------------------------------------------


def test_reroll_exhaustion_keeps_repairs_and_flags_full_rerun():
    data = _ledger_data(story_critic_report=json.loads(_report_with_target()))
    led = _FakeLedger(data)
    original = data["lines"][0]["text"]
    # The critic keeps flagging l001 on every re-run (scoped re-runs + the
    # final whole-episode pass all return the last reply when exhausted).
    fn = _mk_reroll_generate_fn([_report_with_target(), _report_with_target()])

    disp = run_targeted_reroll(fn, led)

    assert disp.verdict == "needs_full_rerun"
    assert disp.cycles_run == MAX_REROLL_CYCLES
    # STEP 4 repair-then-ship: the re-composed line is KEPT, not restored.
    assert data["lines"][0]["text"] != original
    assert data["lines"][0]["text"].strip() != ""
    assert data["meta"]["cycle_count"] == MAX_REROLL_CYCLES
    assert data["meta"]["reroll_verdict"] == "needs_full_rerun"
    # hit the cap, did not diverge.
    assert data["meta"]["reroll_diverged"] is False
    assert len(data["meta"]["reroll_history"]) >= 1


# ---------------------------------------------------------------------------
# 6b. STEP 4 convergence: a reroll that surfaces a NEIGHBOR issue folds the
#     neighbor into the next cycle and converges (does not false-halt).
# ---------------------------------------------------------------------------


def test_reroll_neighbor_joins_next_scope_and_converges():
    data = _ledger_data(story_critic_report=json.loads(_report_with_target()))
    led = _FakeLedger(data)
    l001_original = data["lines"][0]["text"]
    l002_original = data["lines"][1]["text"]
    # Cycle 1 scoped re-run: l001 cleared but neighbor l002 newly flagged.
    # Cycle 2 scoped re-run: clean. Final whole-episode pass: clean.
    fn = _mk_reroll_generate_fn([
        _report_with_target(line_id="l002", hint="give l002 a real turn"),
        _clean_report(),
        _clean_report(),
    ])

    disp = run_targeted_reroll(fn, led)

    assert disp.verdict == "reroll_resolved"
    assert disp.cycles_run == 2
    assert disp.lines_rerolled == 2          # l001 then the neighbor l002
    assert data["lines"][0]["text"] != l001_original
    assert data["lines"][1]["text"] != l002_original
    rerolled_ids = {
        h["line_id"] for h in data["meta"]["reroll_history"]
        if h.get("status") == "rerolled"
    }
    assert rerolled_ids == {"l001", "l002"}
    assert data["meta"]["reroll_diverged"] is False


# ---------------------------------------------------------------------------
# 6c. STEP 4 divergence: when a cycle INCREASES the outstanding flag count,
#     the loop halts to repair-then-ship (keeps repairs, flags full rerun).
# ---------------------------------------------------------------------------


def _report_two_targets(a="l001", b="l002"):
    return json.dumps({
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "uneven",
        "reroll_targets": [
            {"line_id": a, "hint": "sharpen"},
            {"line_id": b, "hint": "sharpen"},
        ],
        "render_priority": ["l001", "l002"],
    })


def test_reroll_diverges_on_outstanding_count_increase():
    data = _ledger_data(story_critic_report=json.loads(_report_with_target()))
    led = _FakeLedger(data)
    l001_original = data["lines"][0]["text"]
    # Cycle 1 scoped re-run flags BOTH l001 (still) and l002 (new) -> count
    # rises 1 -> 2 -> divergence halt. Final whole-episode pass still flags.
    fn = _mk_reroll_generate_fn([_report_two_targets(), _report_two_targets()])

    disp = run_targeted_reroll(fn, led)

    assert disp.verdict == "needs_full_rerun"
    assert disp.cycles_run == 1               # halted after one cycle
    assert data["meta"]["reroll_diverged"] is True
    # repair-then-ship: l001's re-composed text is KEPT.
    assert data["lines"][0]["text"] != l001_original


# ---------------------------------------------------------------------------
# STEP 5 (2026-06-22): failed_dimension prefixes the reroll hint (no re-craft).
# ---------------------------------------------------------------------------


def _report_with_dim(line_id="l001", hint="sharpen", dim="knowledge"):
    return json.dumps({
        "continuity_issues": [], "voice_drift": [], "flat_lines": [],
        "arc_verdict": "uneven",
        "reroll_targets": [{"line_id": line_id, "hint": hint,
                            "failed_dimension": dim}],
        "render_priority": ["l001", "l002"],
    })


def test_failed_dimension_prefixes_reroll_hint():
    data = _ledger_data(story_critic_report=json.loads(_report_with_dim()))
    led = _FakeLedger(data)
    # Cycle 1 re-run clean -> converges; the history records the prefixed hint.
    fn = _mk_reroll_generate_fn([_clean_report()])

    run_targeted_reroll(fn, led)

    rerolled = [h for h in data["meta"]["reroll_history"]
                if h.get("status") == "rerolled"]
    assert rerolled and rerolled[0]["hint"].startswith("[knowledge] ")


def test_unspecified_dimension_adds_no_prefix():
    data = _ledger_data(
        story_critic_report=json.loads(_report_with_dim(dim="unspecified")))
    led = _FakeLedger(data)
    fn = _mk_reroll_generate_fn([_clean_report()])

    run_targeted_reroll(fn, led)

    rerolled = [h for h in data["meta"]["reroll_history"]
                if h.get("status") == "rerolled"]
    assert rerolled and not rerolled[0]["hint"].startswith("[")


# ---------------------------------------------------------------------------
# 7. never raises on a raising generate_fn
# ---------------------------------------------------------------------------


def test_run_targeted_reroll_never_raises_on_raising_fn():
    data = _ledger_data(story_critic_report=json.loads(_report_with_target()))
    led = _FakeLedger(data)
    original = data["lines"][0]["text"]

    # Must not raise -- a failed re-composition keeps the original line,
    # and the critic re-run degrades to clean() so the loop ends.
    disp = run_targeted_reroll(_raising_generate_fn(), led)

    assert isinstance(disp, RerollDisposition)
    assert disp.verdict in ("reroll_resolved", "reroll_error", "no_reroll")
    assert disp.lines_rerolled == 0
    assert data["lines"][0]["text"] == original  # line kept intact
    # the failed compose attempt is recorded.
    history = data["meta"].get("reroll_history", [])
    assert any(h.get("status") == "compose_failed" for h in history)


# ---------------------------------------------------------------------------
# 8. a reroll target absent from the ledger is skipped, not fatal
# ---------------------------------------------------------------------------


def test_unknown_reroll_target_line_id_is_skipped():
    report = json.loads(_report_with_target(line_id="l001"))
    # Add a second target pointing at a line_id that is NOT in the ledger.
    report["reroll_targets"].append(
        {"line_id": "l999", "hint": "this line does not exist"}
    )
    data = _ledger_data(story_critic_report=report)
    led = _FakeLedger(data)
    fn = _mk_reroll_generate_fn([_clean_report()])

    disp = run_targeted_reroll(fn, led)

    # the run completes; l001 rerolled, l999 recorded as not found.
    assert disp.verdict == "reroll_resolved"
    history = data["meta"]["reroll_history"]
    assert any(
        h["line_id"] == "l999" and h["status"] == "line_not_found"
        for h in history
    )
    assert any(
        h["line_id"] == "l001" and h["status"] == "rerolled"
        for h in history
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    sys.exit(pytest.main([__file__, "-v"]))
