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
    ])
    error = lane._validate_p5_structure(script, _cast_stub(), _score_stub({}))
    assert error is not None
    assert "l001: spoken text is production markup" in error
    assert "l003: spoken text starts with a role label" in error
    assert "l004: spoken text is empty" in error
    assert "l002" not in error


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
        {"line_id": "l003", "text": "A clean spoken line."},
    ])
    score = _score_stub({
        "l001": "character", "l002": "character", "l003": "announcer",
    })
    findings = lane._p5_raw_spoken_findings(draft, score, _cast_stub())
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
        {"line_id": "l001", "text": "(SFX: fingers typing on a keyboard)"},
        {"line_id": "l002", "text": "A clean spoken line."},
        {"line_id": "l011", "text": "An invented row."},
    ])
    error = _capture_validator(monkeypatch, candidate)
    assert error is not None
    assert "do not exactly cover the accepted graph" in error
    assert "l001: spoken text is production markup" in error


def test_compile_refusal_alone_stays_bare_when_the_rows_are_clean(monkeypatch):
    def refuse(_draft, _score):
        raise lane.CodexGraphError("P5 compact draft has duplicate line IDs")

    monkeypatch.setattr(lane, "compile_script_text_draft", refuse)
    candidate = lane.ScriptTextDraftV4(lines=[
        {"line_id": "l001", "text": "A quiet signal crosses the observatory."},
    ])
    error = _capture_validator(monkeypatch, candidate)
    assert error == "P5 compact draft has duplicate line IDs"


def test_validator_still_refuses_a_non_draft_result(monkeypatch):
    error = _capture_validator(monkeypatch, SimpleNamespace())
    assert error == "P5 compact result is not a ScriptTextDraftV4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
