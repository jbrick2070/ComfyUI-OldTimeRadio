"""P5 must hand its ONE typed-repair shot every defect at once.

Live failure that motivated this (45-word campaign, leg `ltx_8gb`):

    attempt 1/3 base call
      -> "P5 compact draft line IDs do not exactly cover the accepted graph
          (missing=[], unknown=['l011', 'l012', 'l013'])"
    attempt 2/3 typed repair
      -> "l001: spoken text is production markup"
    ERROR exhausted the retry ladder after 2 attempt(s)

The repair obeyed the only complaint it was given -- it dropped the three
invented IDs -- and then died on a defect that was present in attempt 1 and
never mentioned. `structured_call` deliberately does NOT retry a repair that
was schema-valid but content-invalid, so there is no third shot to spend. The
validator therefore has to report everything it can see, the first time.

Two guarantees are pinned here:

  1. `_validate_p5_structure` reports EVERY offending spoken line, not the
     first one it trips over.
  2. When `compile_script_text_draft` refuses the draft outright, the raw
     rows are still scanned, and those findings ride along with the compile
     refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodes import _otr_scifi_codex as lane
from nodes import _otr_ledger_cleanup as ledger_cleanup
from nodes import _otr_structured_call as structured_call


def _cast_stub() -> SimpleNamespace:
    return SimpleNamespace(
        cast=[
            SimpleNamespace(name="ANNOUNCER"),
            SimpleNamespace(name="Ada Sterling"),
        ]
    )


def _score_stub(roles: dict[str, str]) -> SimpleNamespace:
    """A score whose beats own `roles` -- {line_id: speaker_role}."""
    return SimpleNamespace(
        scenes=[
            SimpleNamespace(
                beats=[
                    SimpleNamespace(line_ids=[line_id], speaker_role=role)
                    for line_id, role in roles.items()
                ]
            )
        ]
    )


def _script_stub(rows: list[tuple[str, str, str]]) -> SimpleNamespace:
    """rows = [(line_id, speaker_role, text)]."""
    return SimpleNamespace(
        lines=[
            SimpleNamespace(
                line_id=line_id, speaker_role=role, text=text, skip=False,
            )
            for line_id, role, text in rows
        ]
    )


def _a6_cast() -> lane.CastPlanV4:
    return lane.CastPlanV4(cast=[
        {
            "char_id": "announcer",
            "name": "ANNOUNCER",
            "character_description": "A calm witness.",
            "gender": "neutral",
            "role_in_conflict": "Frames the signal.",
            "voice_slot": "announcer",
        },
        {
            "char_id": "c01",
            "name": "Ada Sterling",
            "character_description": "A careful radio astronomer.",
            "gender": "female",
            "role_in_conflict": "Must answer the signal.",
            "voice_slot": "c01",
        },
    ])


def _a6_score() -> lane.RadioScoreV4:
    return lane.RadioScoreV4(
        title="Signal at Meridian",
        premise="A signal asks for an answer.",
        setting="Meridian observatory.",
        advisory_word_plan=lane.AdvisoryWordPlanV4(
            advisory_total_center=6,
            per_beat=[{
                "beat_id": "b001",
                "advisory_word_center": 6,
            }],
        ),
        scenes=[{
            "scene_id": "scene_001",
            "env": "Observatory",
            "description": "Receivers glow.",
            "shots": [{
                "shot_id": "shot_001",
                "scene_id": "scene_001",
                "description": "Ada at the receiver.",
                "visual_prompt": "A radio astronomer at a blue receiver.",
            }],
            "beats": [{
                "beat_id": "b001",
                "scene_id": "scene_001",
                "shot_id": "shot_001",
                "speaker": "Ada Sterling",
                "char_id": "c01",
                "speaker_role": "character",
                "line_ids": ["l001"],
                "order": 1,
                "intent": "Answer the signal.",
                "arc_phase": "arrival",
                "fact_ids": [],
                "advisory_voiced_word_center": 6,
            }],
        }],
        music_cues=[{
            "cue_id": "music_open",
            "placement": "open",
            "description": "A low radio pulse.",
            "generation_prompt": "Low radio pulse.",
            "anchor_line_id": "l001",
            "anchor_beat_id": "b001",
        }],
    )


# --------------------------------------------------------------------------
# _spoken_text_finding -- the shared per-line rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("A quiet signal crosses the dark observatory.", None),
        ("   ", "l001: spoken text is empty"),
        ("two\nlines", "l001: spoken text contains control markup"),
        ("(SFX: papers rustling)", "l001: spoken text is production markup"),
        ("[MUSIC STING]", "l001: spoken text is production markup"),
        (
            "Ada Sterling: I am drowning in data.",
            "l001: spoken text starts with a role label",
        ),
        ("SFX: a door closes", "l001: spoken text starts with a role label"),
        ("```json", "l001: spoken text starts with a role label"),
        (
            "[breath catches] (a long pause)",
            "l001: spoken text cleans to an empty spoken surface",
        ),
        ("(softly) The signal is still there.", None),
        # The label rule keys on the LOCKED cast label exactly. An
        # abbreviated or parenthesised prefix -- "ADA (V.O.):" against a cast
        # row named "Ada Sterling" -- is NOT caught, and this pins that
        # honestly rather than pretending otherwise. It is not what killed the
        # live run: there the offender was l001, a pure production-markup
        # line, which the fullmatch rule below does catch.
        ("ADA (V.O.): This is Dr. Ada Sterling.", None),
    ],
)
def test_spoken_text_finding_classifies_one_line(text, expected):
    pattern = lane._spoken_label_pattern(_cast_stub())
    assert lane._spoken_text_finding("l001", text, pattern) == expected


def test_label_pattern_covers_the_locked_cast_and_the_fixed_labels():
    pattern = lane._spoken_label_pattern(_cast_stub())
    for label in ("Ada Sterling", "ANNOUNCER", "NARRATOR", "SFX", "MUSIC"):
        assert pattern.match(f"{label}: something") is not None
    assert pattern.match("Someone Else: something") is None


# --------------------------------------------------------------------------
# _validate_p5_structure -- report EVERY offending line
# --------------------------------------------------------------------------

def _patch_graph_checks(monkeypatch):
    """Silence the graph/roster gates so the spoken loop is isolated."""
    monkeypatch.setattr(lane, "_validate_script_graph", lambda *a, **k: None)
    monkeypatch.setattr(
        lane, "_validate_script_roster_contract", lambda *a, **k: None
    )


def test_validate_p5_structure_reports_every_offending_line(monkeypatch):
    _patch_graph_checks(monkeypatch)
    script = _script_stub([
        ("l001", "character", "(SFX: fingers typing on a keyboard)"),
        ("l002", "character", "A clean spoken line that breaks nothing."),
        ("l003", "announcer", "Ada Sterling: this one wears a role label."),
        ("l004", "character", "   "),
        ("l005", "character", "A gun waits beside the receiver."),
        ("l006", "announcer", "The damn relay keeps calling."),
    ])
    error = lane._validate_p5_structure(script, _cast_stub(), _score_stub({}))
    assert error is not None
    assert "l001: spoken text is production markup" in error
    assert "l003: spoken text starts with a role label" in error
    assert "l004: spoken text is empty" in error
    # Content findings RETIRED 2026-08-05 (operator directive). The structural
    # defects above still report; the words do not. Inverted rather than
    # deleted so a re-armed content check fails here.
    assert "weapon" not in error
    assert "profanity" not in error
    assert "l005" not in error
    assert "l006" not in error
    assert "l002" not in error


def test_validate_p5_structure_rerolls_a_line_that_hit_the_ceiling(monkeypatch):
    """A spoken line forced shut by its ceiling must reroll, never be spoken.

    The P5 half of the 2026-08-13 runaway fix. `ScriptTextDraftLineV4.text`
    carries a structural ceiling so a looping decode cannot spend the whole
    context window inside one string -- but lm-format-enforcer closes that
    string by returning only the quote at the ceiling, and pydantic accepts it
    because len == max_length is valid. Nothing downstream measures the line.

    So without this finding the cap would be WORSE than the unbounded field it
    replaced: a line cut off mid-word would be frozen into the ledger and
    spoken by TTS. The check lives here, in the validator, because a finding
    returned from this function becomes a rerollable PostValidationError --
    a raise from the compiler would not.

    Not hypothetical: the 2026-08-13 leg ran away in P5 as well as P3, 8,128
    tokens over 12 minutes. (PBUG-20260729-02 is the same family on this pass,
    but its root cause was an unenforced ARRAY ceiling rather than a long
    string -- related pathology, different surface.)
    """
    _patch_graph_checks(monkeypatch)
    # 40 chars SHORT of the ceiling: lmfe can force the quote up to one
    # max-token-length (76 chars on this project's widest local tokenizer)
    # early, so this is what a real forced closure looks like and it must be
    # caught by the guard band rather than by an exact-hit test.
    forced = lane._SCRIPT_TEXT_DRAFT_MAX_LINE_CHARS - 40
    script = _script_stub([
        ("l001", "character", "A clean spoken line that breaks nothing."),
        ("l002", "character", "x" * forced),
    ])

    error = lane._validate_p5_structure(script, _cast_stub(), _score_stub({}))

    assert error is not None
    assert "l002" in error
    assert "degeneracy threshold" in error
    assert "rerolled" in error
    # The clean line is not implicated.
    assert "l001" not in error


def test_validate_p5_ceiling_check_covers_skipped_and_music_lines(monkeypatch):
    """The ceiling check runs BEFORE the spoken-role filter, on purpose.

    The other findings in this validator only apply to spoken character and
    announcer rows, because they judge spoken markup. A truncated string is not
    a markup opinion -- it is a broken artifact wherever it lands, and a music
    or skipped row carrying one still means the decode ran away.
    """
    _patch_graph_checks(monkeypatch)
    ceiling = lane._SCRIPT_TEXT_DRAFT_MAX_LINE_CHARS
    script = _script_stub([
        ("l001", "music_open", "x" * ceiling),
    ])

    error = lane._validate_p5_structure(script, _cast_stub(), _score_stub({}))

    assert error is not None
    assert "l001" in error
    assert "degeneracy threshold" in error


def test_validate_p5_structure_single_defect_message_is_unchanged(monkeypatch):
    """One bad line still yields the bare, historical message -- no joining
    artifacts -- so existing pins on the exact string keep holding."""
    _patch_graph_checks(monkeypatch)
    script = _script_stub([
        ("l001", "character", "(SFX: fingers typing on a keyboard)"),
        ("l002", "character", "A clean spoken line that breaks nothing."),
    ])
    error = lane._validate_p5_structure(script, _cast_stub(), _score_stub({}))
    assert error == "l001: spoken text is production markup"


def test_validate_p5_structure_accepts_a_clean_script(monkeypatch):
    _patch_graph_checks(monkeypatch)
    script = _script_stub([
        ("l001", "character", "A quiet signal crosses the dark observatory."),
        ("l002", "announcer", "The night shift holds its breath."),
    ])
    assert lane._validate_p5_structure(
        script, _cast_stub(), _score_stub({})
    ) is None


def test_validate_p5_structure_ignores_skipped_and_unspoken_rows(monkeypatch):
    _patch_graph_checks(monkeypatch)
    script = SimpleNamespace(lines=[
        SimpleNamespace(
            line_id="l001", speaker_role="character",
            text="(SFX: this row is skipped)", skip=True,
        ),
        SimpleNamespace(
            line_id="l002", speaker_role="sfx",
            text="(SFX: this row is not spoken)", skip=False,
        ),
    ])
    assert lane._validate_p5_structure(
        script, _cast_stub(), _score_stub({})
    ) is None


# --------------------------------------------------------------------------
# _p5_raw_spoken_findings -- scan the UNCOMPILED draft
# --------------------------------------------------------------------------

def test_raw_scan_finds_markup_the_compile_refusal_would_have_hidden():
    draft = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "(SFX: fingers typing on a keyboard)"},
        {"line_id": "l002", "text": "A clean spoken line that breaks nothing."},
        {"line_id": "l011", "text": "(SFX: an invented row the score never asked for)"},
    ])
    score = _score_stub({"l001": "character", "l002": "character"})
    findings = lane._p5_raw_spoken_findings(draft, score, _cast_stub())
    assert findings == ["l001: spoken text is production markup"]


def test_raw_scan_refuses_to_judge_a_line_id_the_score_does_not_own():
    """An invented ID has no speaker_role. Judging its text would be inventing
    a contract the score never wrote."""
    draft = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l011", "text": "(SFX: an invented row)"},
        {"line_id": "l012", "text": "ADA: another invented row."},
    ])
    score = _score_stub({"l001": "character"})
    assert lane._p5_raw_spoken_findings(draft, score, _cast_stub()) == []


def test_raw_scan_skips_rows_the_score_marks_unspoken():
    draft = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "(SFX: a door closes)"},
    ])
    score = _score_stub({"l001": "sfx"})
    assert lane._p5_raw_spoken_findings(draft, score, _cast_stub()) == []


def test_raw_scan_reports_every_offending_row():
    draft = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "(SFX: fingers typing)"},
        {"line_id": "l002", "text": "Ada Sterling: I am drowning in data."},
        {"line_id": "l003", "text": "A gun waits by the clean relay."},
    ])
    score = _score_stub({
        "l001": "character", "l002": "character", "l003": "announcer",
    })
    findings = lane._p5_raw_spoken_findings(draft, score, _cast_stub())
    # "spoken safety: l003: weapon='gun'" no longer appears -- content findings
    # were retired 2026-08-05. l003 is otherwise a clean line, so it drops out.
    assert findings == [
        "l001: spoken text is production markup",
        "l002: spoken text starts with a role label",
    ]


def test_raw_scan_is_clean_when_every_owned_row_is_clean():
    draft = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "A quiet signal crosses the observatory."},
    ])
    score = _score_stub({"l001": "character"})
    assert lane._p5_raw_spoken_findings(draft, score, _cast_stub()) == []


# --------------------------------------------------------------------------
# The ladder's post_validator -- BOTH complaints reach the one repair shot
# --------------------------------------------------------------------------

class _LadderProbe(Exception):
    """Aborts the P5 ladder once the post_validator has been exercised."""


def _capture_validator(monkeypatch, candidate):
    """Run the P5 ladder far enough to call its post_validator on
    `candidate`, and return what the validator said."""
    said: dict[str, object] = {}

    def fake_invoke(**kwargs):
        said["error"] = kwargs["post_validator"](candidate)
        raise _LadderProbe()

    monkeypatch.setattr(lane, "invoke_codex_structured", fake_invoke)
    with pytest.raises(_LadderProbe):
        lane._call_script_text_draft(
            slot_fn=lambda *a, **k: "",
            pack=SimpleNamespace(),
            artifact_inputs={},
            score=_score_stub({"l001": "character", "l002": "character"}),
            cast=_cast_stub(),
            max_new_tokens=None,
            call_journal={},
        )
    return said["error"]


def test_compile_refusal_carries_the_markup_findings_with_it(monkeypatch):
    """THE live bug. The draft both misses the graph and speaks markup; the
    repair must be told both, because it only gets one turn."""
    def refuse(_draft, _score):
        raise lane.CodexGraphError(
            "P5 compact draft line IDs do not exactly cover the accepted "
            "graph (missing=[], unknown=['l011'])"
        )

    monkeypatch.setattr(lane, "compile_script_text_draft", refuse)
    candidate = lane.ScriptTextDraftV4(lines=[
        {
            "line_id": "l001",
            "text": "(SFX: gun beside the keyboard)",
        },
        {"line_id": "l002", "text": "A clean spoken line."},
        {"line_id": "l011", "text": "An invented row."},
    ])
    error = _capture_validator(monkeypatch, candidate)
    assert error is not None
    assert "do not exactly cover the accepted graph" in error
    assert "l001: spoken text is production markup" in error
    # The word inside the markup is no longer a second finding (content
    # findings retired 2026-08-05); the MARKUP finding is what must ride along.
    assert "weapon" not in error


def test_compile_refusal_alone_stays_bare_when_the_rows_are_clean(monkeypatch):
    def refuse(_draft, _score):
        raise lane.CodexGraphError("P5 compact draft has duplicate line IDs")

    monkeypatch.setattr(lane, "compile_script_text_draft", refuse)
    candidate = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "A quiet signal crosses the observatory."},
    ])
    error = _capture_validator(monkeypatch, candidate)
    assert error == "P5 compact draft has duplicate line IDs"


def test_cleanup_empty_surface_runs_real_p5_reauthor_rung():
    bad_text = "[breath catches] (a long pause)"
    good_text = "The signal is still there."
    responses = [
        lane.ScriptTextDraftV4(
            lines=[{"line_id": "l001", "text": bad_text}],
        ).model_dump_json(),
        lane.ScriptTextDraftV4(
            lines=[{"line_id": "l001", "text": good_text}],
        ).model_dump_json(),
    ]
    calls = []

    def slot(messages, **kwargs):
        calls.append((kwargs["temperature"], messages))
        return responses[len(calls) - 1]

    journal = {}
    script = lane._call_script_text_draft(
        slot_fn=slot,
        pack=SimpleNamespace(prompt_stages={
            "codex_play_system": "Write the spoken line.",
            "codex_coda_contract_system": "Return only the compact draft.",
        }),
        artifact_inputs={},
        score=_a6_score(),
        cast=_a6_cast(),
        max_new_tokens=None,
        call_journal=journal,
    )

    assert [call[0] for call in calls] == pytest.approx([
        .72, structured_call._REPAIR_TEMPERATURE,
    ])
    assert structured_call._REPAIR_TEMPERATURE != pytest.approx(.32)
    assert script.lines[0].text == good_text
    assert script.lines[0].skip is False
    repair_text = "\n".join(
        str(message.get("content") or "") for message in calls[1][1]
    )
    assert "CRITICAL:" in repair_text
    assert "l001: spoken text cleans to an empty spoken surface" in repair_text
    attempts = journal["calls"][0]["attempts"]
    assert [attempt["status"] for attempt in attempts] == [
        "rejected", "accepted",
    ]
    assert attempts[0]["error_type"] == "PostValidationError"


def test_defective_p5_candidate_is_abandoned_for_fresh_divergent_fiction(
        monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    rejected = lane.ScriptTextDraftV4(
        lines=[{
            "line_id": "l001",
            # Trigger moved from a content term to production markup
            # 2026-08-05: content is no longer a defect, but "abandon the
            # rejected candidate and write fresh divergent fiction" is still
            # the behavior under test.
            "text": "(SFX: REJECTED_PROSE static over the receiver)",
        }],
    ).model_dump_json()
    accepted_text = (
        "Far from the receiver, a gardener teaches moonflowers to sing."
    )
    accepted = lane.ScriptTextDraftV4(
        lines=[{"line_id": "l001", "text": accepted_text}],
    ).model_dump_json()
    responses = [rejected, rejected, accepted]
    prompts = []

    def slot(messages, **_kwargs):
        prompts.append(messages)
        return responses[len(prompts) - 1]

    journal = {}
    script = lane._call_script_text_draft(
        slot_fn=slot,
        pack=SimpleNamespace(prompt_stages={
            "codex_play_system": "Write the spoken line.",
            "codex_coda_contract_system": "Return only the compact draft.",
        }),
        artifact_inputs={},
        score=_a6_score(),
        cast=_a6_cast(),
        max_new_tokens=None,
        call_journal=journal,
    )

    assert script.lines[0].text == accepted_text
    assert len(prompts) == 3
    assert [entry["status"] for entry in journal["calls"]] == [
        "failed", "accepted",
    ]
    assert "writer_retry" in prompts[2][1]["content"]
    assert "REJECTED_PROSE" not in prompts[2][1]["content"]
    assert "REJECTED_PROSE" not in str(journal)


def test_canonicalization_defect_retires_candidate_before_acceptance(
        monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    rejected_text = "OTHER: Ada Sterling: The signal is clear."
    accepted_text = "The signal is clear, and the receiver is stable."
    rejected = lane.ScriptTextDraftV4(
        lines=[{"line_id": "l001", "text": rejected_text}],
    ).model_dump_json()
    accepted = lane.ScriptTextDraftV4(
        lines=[{"line_id": "l001", "text": accepted_text}],
    ).model_dump_json()
    responses = [rejected, rejected, accepted]
    prompts = []

    def slot(messages, **_kwargs):
        prompts.append(messages)
        return responses[len(prompts) - 1]

    journal = {}
    script = lane._call_script_text_draft(
        slot_fn=slot,
        pack=SimpleNamespace(prompt_stages={
            "codex_play_system": "Write the spoken line.",
            "codex_coda_contract_system": "Return only the compact draft.",
        }),
        artifact_inputs={},
        score=_a6_score(),
        cast=_a6_cast(),
        max_new_tokens=None,
        call_journal=journal,
    )

    assert script.lines[0].text == accepted_text
    assert len(prompts) == 3
    assert [entry["status"] for entry in journal["calls"]] == [
        "failed", "accepted",
    ]
    assert "starts with a role label" in str(prompts[1])


def test_without_empty_surface_finding_cleanup_would_skip_the_line(monkeypatch):
    bad_text = "[breath catches] (a long pause)"
    monkeypatch.setattr(lane, "clean_spoken_text", lambda text: text)
    calls = []

    def slot(_messages, **_kwargs):
        calls.append(True)
        if len(calls) > 1:
            pytest.fail("counterfactual P5 unexpectedly tried to re-author")
        return lane.ScriptTextDraftV4(
            lines=[{"line_id": "l001", "text": bad_text}],
        ).model_dump_json()

    accepted = lane._call_script_text_draft(
        slot_fn=slot,
        pack=SimpleNamespace(prompt_stages={
            "codex_play_system": "Write the spoken line.",
            "codex_coda_contract_system": "Return only the compact draft.",
        }),
        artifact_inputs={},
        score=_a6_score(),
        cast=_a6_cast(),
        max_new_tokens=None,
        call_journal={},
    )
    assert len(calls) == 1
    assert accepted.lines[0].text == bad_text

    row = accepted.lines[0].model_dump(mode="json")
    row.update({"char_count": 999, "word_count": 999})
    actions = ledger_cleanup._complete_deterministic({
        "cast": [],
        "lines": [row],
    })
    assert row["skip"] is True
    assert row["text"] == ""
    assert row["char_count"] == 0
    assert row["word_count"] == 0
    assert row["tts_skip_reason"] == ledger_cleanup._EMPTY_TEXT_SKIP_REASON
    assert {
        "field": "line_id=l001.skip",
        "action": "marked_explicit_skip_no_spoken_surface",
    } in actions


def test_validator_still_refuses_a_non_draft_result(monkeypatch):
    error = _capture_validator(monkeypatch, SimpleNamespace())
    assert error == "P5 compact result is not a ScriptTextDraftV4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
