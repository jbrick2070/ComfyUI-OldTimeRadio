"""tests/test_otr_story_critic.py -- Sprint 5B story-quality critic.

Covers the whole-script story critic in `nodes/_otr_story_critic.py`:

  1. a well-formed JSON response parses into a `StoryCriticReport` with
     all 6 rubric sections populated.
  2. a garbage (unparseable) response returns `StoryCriticReport.clean()`
     and never raises.
  3. a raising `generate_fn` is swallowed -> clean() report, no raise.
  4. an invalid `arc_verdict` value is rejected by the Literal -- the
     retry ladder exhausts and the call ends on a clean() report.
  5. the `post_validator` rejects a report citing an unknown `line_id`;
     the ladder exhausts and the call ends on a clean() report.
  6. `StoryCriticReport.clean()` is the all-empty `arc_verdict="unverified"`
     FAIL-CLOSED fallback (story-ledger DRIFT 2026-06-25 -- it no longer
     claims "strong"); `run_story_critic` stamps `status_out` provenance.

Pure-Python. Fake `generate_fn` for every LLM call -- NO GPU, NO HF, NO
real model. Mirrors the fake-generate-fn pattern in
`tests/test_phase3_ledger_reviewer.py` and `tests/test_structured_call.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_story_critic import (  # noqa: E402
    ContinuityIssue,
    FlatLine,
    RerollTarget,
    StoryCriticReport,
    VoiceDriftNote,
    _scope_character_lines,
    run_story_critic,
)


# ---------------------------------------------------------------------------
# STEP 4 (2026-06-22 story+cast fix): scope helper -- the critic re-scores
# only the targeted lines + their continuity neighbors.
# ---------------------------------------------------------------------------


def _char_line(line_id):
    return {"line_id": line_id, "speaker_role": "character", "text": "x"}


def test_scope_character_lines_none_returns_all():
    lines = [_char_line(f"l{i:03d}") for i in range(1, 6)]
    assert _scope_character_lines(lines, None) == lines


def test_scope_character_lines_includes_immediate_neighbors():
    lines = [_char_line(f"l{i:03d}") for i in range(1, 6)]  # l001..l005
    scoped = _scope_character_lines(lines, {"l003"})
    ids = [ln["line_id"] for ln in scoped]
    assert ids == ["l002", "l003", "l004"]   # target + prev/next neighbor


def test_scope_character_lines_edges_and_missing_ids():
    lines = [_char_line(f"l{i:03d}") for i in range(1, 6)]
    # l001 at the start -> only itself + l002; a missing id is ignored.
    scoped = _scope_character_lines(lines, {"l001", "l999"})
    assert [ln["line_id"] for ln in scoped] == ["l001", "l002"]


def test_scope_character_lines_empty_scope_returns_none_lines():
    lines = [_char_line("l001")]
    assert _scope_character_lines(lines, set()) == []


# ---------------------------------------------------------------------------
# STEP 5 (2026-06-22): failed_dimension on RerollTarget -- optional, enum,
# back-compat with stored reports + clean().
# ---------------------------------------------------------------------------


def test_reroll_target_failed_dimension_defaults_unspecified():
    t = RerollTarget(line_id="l001", hint="sharpen it")
    assert t.failed_dimension == "unspecified"


def test_reroll_target_failed_dimension_accepts_enum():
    for dim in ("knowledge", "pressure", "relationship", "decision",
                "obstacle", "tension", "unspecified"):
        assert RerollTarget(line_id="l", hint="h",
                            failed_dimension=dim).failed_dimension == dim


def test_reroll_target_failed_dimension_rejects_unknown():
    import pytest as _pytest
    from pydantic import ValidationError
    with _pytest.raises(ValidationError):
        RerollTarget(line_id="l", hint="h", failed_dimension="vibes")


def test_old_report_without_failed_dimension_still_validates():
    """Back-compat: a stored meta.story_critic_report whose reroll_targets
    predate failed_dimension parses, defaulting to 'unspecified' (no new
    REQUIRED field)."""
    old = {
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "uneven",
        "reroll_targets": [{"line_id": "l001", "hint": "sharpen"}],
        "render_priority": [],
    }
    rep = StoryCriticReport(**old)
    assert rep.reroll_targets[0].failed_dimension == "unspecified"
    # clean() is still valid with no new required fields.
    assert StoryCriticReport.clean().reroll_targets == []


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _mk_generate_fn(*replies):
    """Fake generate_fn returning the given strings in order.

    Signature matches the project slot-fn convention:
    `fn(messages, *, temperature, max_new_tokens) -> str`. Once the
    canned replies are exhausted the last reply is repeated, so a
    multi-attempt ladder keeps seeing the same (failing) response.
    """
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        if i < len(replies):
            return replies[i]
        return replies[-1] if replies else "{}"

    return fn


def _raising_generate_fn():
    """Fake generate_fn that raises -- stands in for an LLM loader crash."""

    def fn(messages, *, temperature, max_new_tokens):
        raise RuntimeError("simulated model loader failure")

    return fn


def _fake_ledger():
    """A small candidate ledger dict -- the `Ledger.data` shape.

    Two character lines plus one non-character beat, enough to exercise
    the critic's character-line scope and the post_validator's line_id
    cross-check.
    """
    return {
        "meta": {"target_words": 500, "style": "noir mystery"},
        "cast": [
            {"char_id": "c01", "name": "EDNA", "speaker_role": "character"},
            {"char_id": "c02", "name": "MARLOWE", "speaker_role": "character"},
        ],
        "lines": [
            {
                "line_id": "l001", "beat_id": "l001", "char_id": "c01",
                "speaker_role": "character", "arc_phase": "setup",
                "traits": "wary", "beat_intent": "establish the theft",
                "target_words": 14, "word_count": 8,
                "text": "The safe was open when I walked in this morning.",
            },
            {
                "line_id": "l002", "beat_id": "l002", "char_id": "c02",
                "speaker_role": "character", "arc_phase": "rising",
                "traits": "probing", "beat_intent": "press the witness",
                "target_words": 12, "word_count": 7,
                "text": "And you are certain nobody else had a key?",
            },
            {
                "line_id": "l003", "beat_id": "l003", "char_id": "announcer",
                "speaker_role": "announcer", "arc_phase": "setup",
                "text": "We return after this word from our sponsor.",
            },
        ],
    }


def _cast_rows():
    return _fake_ledger()["cast"]


def _full_report_json():
    """A well-formed response that populates all 6 rubric sections."""
    return json.dumps({
        "continuity_issues": [
            {"line_id": "l002",
             "issue": "Marlowe knows about the key before it is mentioned."}
        ],
        "voice_drift": [
            {"char_id": "c01",
             "note": "Edna shifts from wary to chatty without a beat."}
        ],
        "flat_lines": [
            {"line_id": "l001",
             "reason": "states the discovery as a fact, plays no fear."}
        ],
        "arc_verdict": "uneven",
        "reroll_targets": [
            {"line_id": "l001",
             "hint": "open on Edna's dread, not an inventory of the safe."}
        ],
        "render_priority": ["l002", "l001"],
    })


# ---------------------------------------------------------------------------
# 1. Well-formed response -> all 6 sections populated.
# ---------------------------------------------------------------------------


def test_well_formed_response_populates_all_six_sections():
    fn = _mk_generate_fn(_full_report_json())
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())

    assert isinstance(report, StoryCriticReport)

    # Section 1 -- continuity.
    assert len(report.continuity_issues) == 1
    assert isinstance(report.continuity_issues[0], ContinuityIssue)
    assert report.continuity_issues[0].line_id == "l002"
    assert report.continuity_issues[0].issue

    # Section 2 -- voice drift.
    assert len(report.voice_drift) == 1
    assert isinstance(report.voice_drift[0], VoiceDriftNote)
    assert report.voice_drift[0].char_id == "c01"

    # Section 3 -- flat lines.
    assert len(report.flat_lines) == 1
    assert isinstance(report.flat_lines[0], FlatLine)
    assert report.flat_lines[0].line_id == "l001"

    # Section 4 -- arc verdict.
    assert report.arc_verdict == "uneven"

    # Section 5 -- reroll targets, with a concrete actionable hint.
    assert len(report.reroll_targets) == 1
    assert isinstance(report.reroll_targets[0], RerollTarget)
    assert report.reroll_targets[0].line_id == "l001"
    assert report.reroll_targets[0].hint

    # Section 6 -- render priority, dramatic peaks first.
    assert report.render_priority == ["l002", "l001"]


def test_well_formed_empty_report_validates():
    """An all-empty 'strong' report from the model is a valid result --
    a clean script legitimately produces one."""
    fn = _mk_generate_fn(json.dumps({
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "strong",
        "reroll_targets": [],
        "render_priority": [],
    }))
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())
    assert report.arc_verdict == "strong"
    assert report.reroll_targets == []
    assert report.render_priority == []


# ---------------------------------------------------------------------------
# 2. Garbage response -> clean() fallback, never raises.
# ---------------------------------------------------------------------------


def test_garbage_response_returns_clean_report():
    # Same unparseable junk on every attempt -> ladder exhausts.
    fn = _mk_generate_fn("this is not JSON at all {{{")
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())

    assert isinstance(report, StoryCriticReport)
    # Identical to the FAIL-CLOSED fallback: all sections empty,
    # arc_verdict="unverified" (story-ledger DRIFT 2026-06-25 -- a crashed
    # critic no longer claims "strong").
    assert report.arc_verdict == "unverified"
    assert report.continuity_issues == []
    assert report.voice_drift == []
    assert report.flat_lines == []
    assert report.reroll_targets == []
    assert report.render_priority == []


def test_garbage_response_does_not_raise():
    fn = _mk_generate_fn("<<< broken >>>")
    # The whole point: no exception escapes into the pipeline.
    run_story_critic(fn, _fake_ledger(), _cast_rows())


# ---------------------------------------------------------------------------
# 3. Raising generate_fn -> swallowed, clean() report.
# ---------------------------------------------------------------------------


def test_raising_generate_fn_is_swallowed():
    fn = _raising_generate_fn()
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())

    assert isinstance(report, StoryCriticReport)
    assert report.arc_verdict == "unverified"
    assert report.reroll_targets == []
    assert report.render_priority == []


# ---------------------------------------------------------------------------
# 4. Invalid arc_verdict Literal -> ladder exhausts -> clean() report.
# ---------------------------------------------------------------------------


def test_invalid_arc_verdict_rejected_ends_clean():
    bad = json.dumps({
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "catastrophic",   # not in the Literal set
        "reroll_targets": [],
        "render_priority": [],
    })
    # Same schema-invalid response on every attempt -> the ladder cannot
    # recover and the never-raises contract returns clean().
    fn = _mk_generate_fn(bad)
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())

    assert isinstance(report, StoryCriticReport)
    assert report.arc_verdict == "unverified"


def test_invalid_arc_verdict_then_valid_recovers():
    """If a later ladder rung returns a valid response, the critic
    recovers -- the invalid Literal alone does not doom the call."""
    bad = json.dumps({
        "continuity_issues": [], "voice_drift": [], "flat_lines": [],
        "arc_verdict": "catastrophic",
        "reroll_targets": [], "render_priority": [],
    })
    fn = _mk_generate_fn(bad, _full_report_json())
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())
    assert report.arc_verdict == "uneven"
    assert len(report.reroll_targets) == 1


# ---------------------------------------------------------------------------
# 5. post_validator rejects an unknown line_id -> ladder exhausts -> clean.
# ---------------------------------------------------------------------------


def test_post_validator_rejects_unknown_line_id():
    # `l999` is not in the fake ledger -- a hallucinated id. The
    # post_validator rejects, the ladder exhausts on the repeated bad
    # response, and the never-raises contract returns clean().
    hallucinated = json.dumps({
        "continuity_issues": [
            {"line_id": "l999", "issue": "references a line that does not exist"}
        ],
        "voice_drift": [],
        "flat_lines": [],
        "arc_verdict": "strong",
        "reroll_targets": [],
        "render_priority": [],
    })
    fn = _mk_generate_fn(hallucinated)
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())

    assert isinstance(report, StoryCriticReport)
    assert report.arc_verdict == "unverified"
    assert report.continuity_issues == []


def test_post_validator_rejects_unknown_id_in_render_priority():
    bad = json.dumps({
        "continuity_issues": [], "voice_drift": [], "flat_lines": [],
        "arc_verdict": "strong", "reroll_targets": [],
        "render_priority": ["l001", "l404"],   # l404 hallucinated
    })
    fn = _mk_generate_fn(bad)
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())
    # Rejected -> ladder exhausts -> clean fallback.
    assert report.render_priority == []


def test_post_validator_accepts_known_line_ids():
    """A report whose every line_id is real clears the post_validator."""
    fn = _mk_generate_fn(_full_report_json())
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())
    # _full_report_json only cites l001 / l002, both present.
    assert len(report.reroll_targets) == 1
    assert report.reroll_targets[0].line_id == "l001"


def test_post_validator_lenient_on_empty_ledger():
    """An empty ledger has no ids to cross-check; the guard must accept
    rather than reject every report."""
    empty_ledger = {"meta": {}, "cast": [], "lines": []}
    fn = _mk_generate_fn(_full_report_json())
    report = run_story_critic(fn, empty_ledger, [])
    # No line_id cross-check possible -> the report validates as-is.
    assert report.arc_verdict == "uneven"


# ---------------------------------------------------------------------------
# 6. StoryCriticReport.clean() -- the safe fallback.
# ---------------------------------------------------------------------------


def test_clean_classmethod_is_all_empty_unverified():
    # story-ledger DRIFT (2026-06-25): clean() is the FAIL-CLOSED fallback --
    # all sections empty but arc_verdict="unverified", NOT "strong".
    report = StoryCriticReport.clean()
    assert isinstance(report, StoryCriticReport)
    assert report.arc_verdict == "unverified"
    assert report.continuity_issues == []
    assert report.voice_drift == []
    assert report.flat_lines == []
    assert report.reroll_targets == []
    assert report.render_priority == []


# ---------------------------------------------------------------------------
# 7. story-ledger DRIFT (2026-06-25): kill the fail-OPEN. ArcVerdict gains the
#    "unverified" FAIL-CLOSED sentinel; every failure path returns it (NOT
#    "strong"). The freeze cascade derives meta.story_critic_status from it.
# ---------------------------------------------------------------------------


def test_arcverdict_literal_includes_unverified():
    from nodes._otr_story_critic import ArcVerdict

    assert "unverified" in ArcVerdict.__args__
    # the real verdicts are still present (no accidental replacement)
    for v in ("strong", "uneven", "flat", "mid_collapse"):
        assert v in ArcVerdict.__args__


def test_success_returns_real_verdict_not_unverified():
    fn = _mk_generate_fn(_full_report_json())
    report = run_story_critic(fn, _fake_ledger(), _cast_rows())
    # a real evaluation NEVER yields the unverified sentinel
    assert report.arc_verdict == "uneven"


def test_every_failure_path_is_unverified_not_strong():
    # exhausted ladder (garbage), raising slot fn -- BOTH fail-closed.
    for fn in (_mk_generate_fn("not json at all {{{"), _raising_generate_fn()):
        report = run_story_critic(fn, _fake_ledger(), _cast_rows())
        assert report.arc_verdict == "unverified", (
            "a crashed critic must fail CLOSED (unverified), never claim strong"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
