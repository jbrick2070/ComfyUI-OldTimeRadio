"""tests/test_otr_story_room.py -- Sprint 10B Wave 2 Agent F.

Unit tests for the Story Room loop. Pure tests: no torch, no GPU, no
network. Writer + Editor LLM calls are stubbed with `make_stub_*`
fixtures so the loop body is exercised against deterministic
transcripts.

Pins (per design Section 3.3 + Section 4 Wave 2):
- StoryRoomTurn / StoryRoomTranscript shape.
- Loop terminates clean when Editor passes cycle 0.
- Loop terminates dirty when Editor never passes (cap fires).
- max_total_turns enforced (Writer that produces infinite drafts
  terminates inside the budget).
- terminated_clean=False when EditorCallFailedError fires but a Writer
  draft is present (soft-pass keeps the episode shippable... but the
  pass_decision is True because the soft-pass verdict says so; see the
  Section 4 Wave 2 risk row note pinned in the loop body).
- StoryRoomCallFailedError on no-Writer-draft path.
- to_dict() round-trip + cascade source pins.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_director_brief as director_mod_module
from nodes import _otr_editor_pass as editor_mod_module
from nodes import _otr_story_room as story_room_module


@pytest.fixture(scope="module")
def story_room_mod():
    return story_room_module


@pytest.fixture(scope="module")
def director_mod():
    return director_mod_module


@pytest.fixture(scope="module")
def editor_mod():
    return editor_mod_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brief(director_mod):
    DirectorBrief = director_mod.DirectorBrief
    return DirectorBrief(
        news_premise=(
            "A whaling-era recording of a humpback aria has been "
            "matched to a living pod off Maui."
        ),
        dramatic_question=(
            "Will Suri publish the recording knowing it will draw "
            "tourists who could harass the pod?"
        ),
        opposed_desires=(
            "SURI wants the pod's song to be heard. JONAS wants the "
            "pod left alone."
        ),
        arc_shape=(
            "Suri argues for publication; Jonas pushes back; Suri "
            "chooses a partial release with sound-only, no coordinates."
        ),
        audience_feel="quiet weight",
        forbidden_drift=["alarm-bell theatrics"],
        raw_brief=(
            "Two researchers, one recording, one decision. Keep the "
            "pod off-stage; the choice is the action."
        ),
    )


def make_writer_stub(drafts, *, raise_on=None):
    """Build a Writer generate_fn that yields each draft in order.

    drafts: list[str] -- the i-th call returns drafts[i] (mod len).
    raise_on: optional set[int] of 1-indexed call indices that raise
              RuntimeError instead of returning a draft.
    """
    calls = {"n": 0, "last_messages": None, "last_temperature": None}
    raise_on = raise_on or set()

    def writer_fn(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        calls["last_messages"] = messages
        calls["last_temperature"] = temperature
        if calls["n"] in raise_on:
            raise RuntimeError(f"writer-stub raise at call {calls['n']}")
        return drafts[(calls["n"] - 1) % len(drafts)]

    writer_fn.calls = calls
    return writer_fn


def make_editor_stub(verdicts):
    """Build an Editor generate_fn that returns each verdict-JSON.

    verdicts: list[dict] -- each dict matches the EditorVerdictSchema
    shape. The i-th call returns json.dumps(verdicts[i]).
    """
    calls = {"n": 0}

    def editor_fn(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        return json.dumps(verdicts[(calls["n"] - 1) % len(verdicts)])

    editor_fn.calls = calls
    return editor_fn


# ---------------------------------------------------------------------------
# StoryRoomTurn / StoryRoomTranscript shape pins (Section 3.3)
# ---------------------------------------------------------------------------


def test_storyroom_turn_fields(story_room_mod):
    """StoryRoomTurn must expose exactly role / cycle / content."""
    turn = story_room_mod.StoryRoomTurn(role="writer", cycle=0, content="x")
    assert turn.role == "writer"
    assert turn.cycle == 0
    assert turn.content == "x"


def test_storyroom_transcript_defaults(story_room_mod, director_mod):
    """StoryRoomTranscript ships with the Section 3.3 defaults."""
    t = story_room_mod.StoryRoomTranscript(director_brief=_brief(director_mod))
    assert t.turns == []
    assert t.editor_verdicts == []
    assert t.final_draft == ""
    assert t.terminated_clean is False
    assert t.total_turns == 0
    assert t.elapsed_seconds == 0.0


def test_storyroom_transcript_to_dict_round_trip(
    story_room_mod, director_mod, editor_mod,
):
    """to_dict emits JSON-friendly nested dicts the consumer can parse."""
    t = story_room_mod.StoryRoomTranscript(director_brief=_brief(director_mod))
    t.turns.append(story_room_mod.StoryRoomTurn(
        role="writer", cycle=0, content="first draft",
    ))
    t.editor_verdicts.append(editor_mod.EditorVerdict(
        pass_decision=True,
        failing_axes=[],
        per_axis_notes={},
        overall_note="ship",
        cycle=0,
    ))
    t.final_draft = "first draft"
    t.terminated_clean = True
    t.total_turns = 3
    t.elapsed_seconds = 1.25
    payload = t.to_dict()
    # JSON-serializable end-to-end.
    s = json.dumps(payload)
    again = json.loads(s)
    assert again["final_draft"] == "first draft"
    assert again["terminated_clean"] is True
    assert again["editor_verdicts"][0]["pass_decision"] is True
    assert again["director_brief"]["news_premise"].startswith("A whaling")


# ---------------------------------------------------------------------------
# Loop body -- happy path
# ---------------------------------------------------------------------------


def _ship_verdict(cycle: int = 0) -> dict:
    return {
        "pass_decision": True,
        "failing_axes": [],
        "per_axis_notes": {},
        "overall_note": "draft holds the brief; ship.",
        "cycle": cycle,
    }


def _fail_verdict(cycle: int = 0) -> dict:
    return {
        "pass_decision": False,
        "failing_axes": ["premise_clarity"],
        "per_axis_notes": {
            "premise_clarity": (
                "The recording / pod premise is missing from the "
                "dialogue; the listener cannot identify the news."
            ),
        },
        "overall_note": "name the recording in dialogue and revise.",
        "cycle": cycle,
    }


def test_loop_terminates_clean_on_cycle0_ship(
    story_room_mod, director_mod,
):
    """Editor passes on the first cycle -> terminated_clean=True."""
    writer = make_writer_stub(["SURI: I'd publish. JONAS: It's a quiet pod."])
    editor = make_editor_stub([_ship_verdict(cycle=0)])
    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="humpback aria recording matched to a living pod",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        editor_generate_fn=editor,
    )
    assert t.terminated_clean is True
    assert t.final_draft.startswith("SURI:")
    assert len(t.editor_verdicts) == 1
    assert t.editor_verdicts[0].pass_decision is True
    # Director + Writer + Editor = 3 turns.
    assert t.total_turns == 3
    # Writer was only called once (the first non-empty draft is
    # sent to the Editor; the rest of the cycle-0 budget is not
    # consumed speculatively).
    assert writer.calls["n"] == 1


def test_loop_revises_then_ships(story_room_mod, director_mod):
    """Editor fails cycle 0, passes cycle 1 -> terminated_clean=True."""
    writer = make_writer_stub([
        "SURI: I'd publish.",                          # cycle 0 draft
        "SURI: I have the recording. JONAS: Not yet.", # revision
    ])
    editor = make_editor_stub([
        _fail_verdict(cycle=0),
        _ship_verdict(cycle=1),
    ])
    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="humpback aria recording matched to a living pod",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        editor_generate_fn=editor,
    )
    assert t.terminated_clean is True
    assert "I have the recording" in t.final_draft
    assert len(t.editor_verdicts) == 2
    assert t.editor_verdicts[0].pass_decision is False
    assert t.editor_verdicts[1].pass_decision is True
    assert writer.calls["n"] == 2


# ---------------------------------------------------------------------------
# Loop body -- failure / boundary modes
# ---------------------------------------------------------------------------


def test_loop_terminates_dirty_when_editor_never_passes(
    story_room_mod, director_mod,
):
    """Editor fails every cycle -> terminated_clean=False; last draft preserved."""
    writer = make_writer_stub([
        "SURI: draft 1",
        "SURI: draft 2",
        "SURI: draft 3",
    ])
    editor = make_editor_stub([
        _fail_verdict(cycle=0),
        _fail_verdict(cycle=1),
        _fail_verdict(cycle=2),
    ])
    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="seed",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        editor_generate_fn=editor,
        max_editor_cycles=3,
    )
    assert t.terminated_clean is False
    # Last good draft preserved as final_draft.
    assert t.final_draft == "SURI: draft 3"
    assert len(t.editor_verdicts) == 3
    # Director(1) + cycle-0 writer(1) + editor cycle 0 fail(1) +
    # revision writer(1) + editor cycle 1 fail(1) + revision writer(1) +
    # editor cycle 2 fail(1) = 7. The final editor cycle skips its
    # revision turn because cycle_idx + 1 >= max_editor_cycles.
    assert t.total_turns == 7


def test_max_total_turns_enforced(story_room_mod, director_mod):
    """A Writer that produces infinite drafts terminates at the cap.

    The loop must NOT run past max_total_turns regardless of the
    Editor's verdicts. We set the cap below the natural cycle budget
    so the cap fires first.
    """
    writer = make_writer_stub(["SURI: infinite draft"])
    editor = make_editor_stub([_fail_verdict(cycle=i) for i in range(10)])
    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="seed",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        editor_generate_fn=editor,
        max_writer_turns=2,
        max_editor_cycles=10,
        max_total_turns=5,
    )
    assert t.total_turns <= 5
    assert t.terminated_clean is False
    # Cap fired before all 10 editor cycles ran.
    assert len(t.editor_verdicts) < 10


def test_loop_raises_when_writer_never_produces_draft(
    story_room_mod, director_mod,
):
    """All Writer turns raise -> StoryRoomCallFailedError."""
    writer = make_writer_stub(
        ["unused"], raise_on={1, 2, 3, 4},
    )
    editor = make_editor_stub([_ship_verdict()])
    with pytest.raises(story_room_mod.StoryRoomCallFailedError) as exc_info:
        story_room_mod.run_story_room(
            director_brief=_brief(director_mod),
            news_seed="seed",
            cast=["SURI", "JONAS"],
            writer_generate_fn=writer,
            editor_generate_fn=editor,
            max_writer_turns=3,
        )
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, RuntimeError)
    # Editor was never called -- there was nothing for it to read.
    assert editor.calls["n"] == 0


def test_loop_handles_empty_writer_draft_then_real_one(
    story_room_mod, director_mod,
):
    """Writer's first draft is empty; second is real; loop continues."""
    writer = make_writer_stub([
        "",                                # empty draft -> ignored
        "SURI: real draft", "SURI: real",
    ])
    editor = make_editor_stub([_ship_verdict(cycle=0)])
    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="seed",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        editor_generate_fn=editor,
        max_writer_turns=3,
    )
    assert t.terminated_clean is True
    assert t.final_draft == "SURI: real draft"


def test_loop_soft_passes_on_editor_call_failure(
    story_room_mod, director_mod, editor_mod, monkeypatch,
):
    """When the Editor raises EditorCallFailedError the loop soft-passes."""
    writer = make_writer_stub(["SURI: a draft"])

    # Stub run_editor inside the story_room module to raise once.
    def _bad_editor(*args, **kwargs):
        raise editor_mod.EditorCallFailedError(
            attempts=2,
            last_raw_output="not json",
            last_error=ValueError("parse failed"),
        )
    monkeypatch.setattr(story_room_mod, "run_editor", _bad_editor)

    t = story_room_mod.run_story_room(
        director_brief=_brief(director_mod),
        news_seed="seed",
        cast=["SURI", "JONAS"],
        writer_generate_fn=writer,
        # Editor stub never gets called -- run_editor is patched.
        editor_generate_fn=make_editor_stub([_ship_verdict()]),
    )
    # Soft-pass: pass_decision=True per the design Section 4 Wave 2
    # risk row, terminated_clean rides True in that case (the loop
    # took the Editor's pass-decision at face value).
    assert t.editor_verdicts[0].pass_decision is True
    assert t.editor_verdicts[0].failing_axes == []
    assert "EditorCallFailedError" in t.editor_verdicts[0].overall_note
    assert t.final_draft == "SURI: a draft"
    assert t.terminated_clean is True


# ---------------------------------------------------------------------------
# Constants pin (so a future tweak shows up in regression)
# ---------------------------------------------------------------------------


def test_loop_budget_defaults(story_room_mod):
    """Section 1 caps -- a refactor that loosens them must update tests."""
    assert story_room_mod.DEFAULT_MAX_WRITER_TURNS == 4
    assert story_room_mod.DEFAULT_MAX_EDITOR_CYCLES == 3
    assert story_room_mod.DEFAULT_MAX_TOTAL_TURNS == 18


def test_writer_sampling_defaults(story_room_mod):
    """Writer sampling defaults; revision T < base T (convergence)."""
    assert story_room_mod.WRITER_BASE_TEMPERATURE > \
        story_room_mod.WRITER_REVISION_TEMPERATURE
    assert 0.0 < story_room_mod.WRITER_REVISION_TEMPERATURE < 2.0
    assert 0.0 < story_room_mod.WRITER_BASE_TEMPERATURE < 2.0
    assert story_room_mod.WRITER_MAX_NEW_TOKENS >= 1024


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_run_story_room_rejects_bad_budgets(story_room_mod, director_mod):
    """Out-of-range budgets raise ValueError, never silently degrade."""
    with pytest.raises(ValueError):
        story_room_mod.run_story_room(
            director_brief=_brief(director_mod),
            news_seed="seed",
            cast=[],
            writer_generate_fn=make_writer_stub(["x"]),
            editor_generate_fn=make_editor_stub([_ship_verdict()]),
            max_writer_turns=0,
        )
    with pytest.raises(ValueError):
        story_room_mod.run_story_room(
            director_brief=_brief(director_mod),
            news_seed="seed",
            cast=[],
            writer_generate_fn=make_writer_stub(["x"]),
            editor_generate_fn=make_editor_stub([_ship_verdict()]),
            max_editor_cycles=0,
        )
    with pytest.raises(ValueError):
        story_room_mod.run_story_room(
            director_brief=_brief(director_mod),
            news_seed="seed",
            cast=[],
            writer_generate_fn=make_writer_stub(["x"]),
            editor_generate_fn=make_editor_stub([_ship_verdict()]),
            max_total_turns=1,
        )


# ---------------------------------------------------------------------------
# Prompt builder spot checks
# ---------------------------------------------------------------------------


def test_writer_user_prompt_first_pass(story_room_mod, director_mod):
    """First-pass user prompt names the brief, cast, and news seed."""
    body = story_room_mod.build_writer_user_prompt(
        brief=_brief(director_mod),
        cast=["SURI", "JONAS"],
        news_seed="recording matched to a living pod",
        cycle=0,
    )
    assert "DIRECTOR'S BRIEF" in body
    assert "SURI" in body and "JONAS" in body
    assert "recording matched to a living pod" in body
    # No editor-notes block on first pass.
    assert "EDITOR NOTES" not in body
    assert "Draft the episode now" in body


def test_writer_user_prompt_revision_includes_notes(
    story_room_mod, director_mod, editor_mod,
):
    """Revision-pass prompt includes the last draft + editor notes."""
    body = story_room_mod.build_writer_user_prompt(
        brief=_brief(director_mod),
        cast=["SURI", "JONAS"],
        news_seed="seed",
        last_draft="SURI: stub draft",
        last_editor_verdict=editor_mod.EditorVerdict(
            pass_decision=False,
            failing_axes=["premise_clarity"],
            per_axis_notes={"premise_clarity": "name the recording"},
            overall_note="revise",
            cycle=0,
        ),
        cycle=1,
    )
    assert "stub draft" in body
    assert "EDITOR NOTES" in body
    assert "premise_clarity" in body
    assert "Revise the draft" in body


# ---------------------------------------------------------------------------
# Module source pins so a refactor that drops the LLM-slot tags fails
# ---------------------------------------------------------------------------


def test_story_room_source_has_slot_tags(story_room_mod):
    """The two LLM call sites must carry the rule-6 slot tags."""
    src = Path(story_room_mod.__file__).read_text(encoding="utf-8")
    # Writer call site is creative; editor call site is technical.
    assert "# LLM slot: creative" in src
    assert "# LLM slot: technical" in src
    # And the loop must call run_editor.
    assert "run_editor(" in src


def test_story_room_imports_wave1_dataclasses(story_room_mod):
    """Section 3.3 contract: importable DirectorBrief + EditorVerdict."""
    src = Path(story_room_mod.__file__).read_text(encoding="utf-8")
    assert "from ._otr_director_brief import DirectorBrief" in src
    assert (
        "from ._otr_editor_pass import EditorCallFailedError, "
        "EditorVerdict, run_editor"
    ) in src
