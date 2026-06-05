"""tests/test_bug_local_282_to_285_quad_fix.py

Regression pins for the BUG-LOCAL-282..285 quad-fix landed 2026-05-27
after live console evidence on episode pending_20260527_152446
("Pine Barrens Inferno", NASA firefighter premise). The fixes are
independent shapes -- bundled because the same live run surfaced all
four.

BUG-LOCAL-282 -- Stage1Beat schema rejected length_target_words=0
                 on music_inter beats (speaker='MUSIC'); Stage 1
                 shadow pass exhausted retry budget every run.
                 Fix: ge=0 on the field + model_validator
                 re-asserts ge=5 for voiced beats only.

BUG-LOCAL-283 -- NewsBriefs._MAX_KEY_TERMS=6 too tight; the LLM
                 consistently wants 7 distinct terms (NASA, FireSense,
                 Wildfire, Thermal Sensor, Fire Mission, Fire Bulldozer,
                 Firefighter) and burned the structural-retry call
                 to land back inside 6. Bumped to 7.

BUG-LOCAL-284 -- ReviewerEdit.payload required-str rejected
                 doctor-emitted annotate/skip rows without a payload.
                 Doctor's edits pass exhausted attempts. Fix:
                 payload defaults to ""; annotate consumer treats
                 empty as a no-op.

BUG-LOCAL-285 -- run_story_brief_reflection ladder exhausted with
                 "no decodable top-level JSON object found: line 1
                 column 1 (char 0)" -- empty model output, not a
                 truncated JSON. The v2 schema's nine fields need
                 ~200-300 tokens of JSON; the 160-token cap was
                 starving Mistral-Nemo into emitting nothing.
                 Bumped to 512.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodes._otr_stage1_plan import Stage1Beat
from nodes import news_interpreter as _ni
from nodes._otr_ledger_reviewer import ReviewerEdit
from nodes import _otr_story_brief as _osb


# ---------------------------------------------------------------------------
# BUG-LOCAL-282
# ---------------------------------------------------------------------------


class TestBugLocal282MusicBeatAllowsZeroWords:
    """Stage1Beat with speaker='MUSIC' may carry length_target_words=0."""

    def _good_music_beat(self, **overrides):
        defaults = dict(
            beat_id="b004",
            speaker="MUSIC",
            intent="A short music interlude between scenes.",
            length_target_words=0,
            emotional_register="suspended",
        )
        defaults.update(overrides)
        return Stage1Beat(**defaults)

    def test_music_beat_zero_words_accepted(self):
        beat = self._good_music_beat(length_target_words=0)
        assert beat.length_target_words == 0
        assert beat.speaker == "MUSIC"

    def test_music_beat_nonzero_words_accepted(self):
        # MUSIC beats with positive target are still allowed (no
        # upper-bound change).
        beat = self._good_music_beat(length_target_words=5)
        assert beat.length_target_words == 5

    def test_character_beat_zero_words_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Stage1Beat(
                beat_id="b001",
                speaker="REN BLACK",
                intent="Open with a refusal.",
                length_target_words=0,
                emotional_register="guarded",
            )
        msg = str(exc_info.value)
        assert "length_target_words" in msg
        assert "voiced beats" in msg or ">= 5" in msg

    def test_announcer_beat_zero_words_rejected(self):
        with pytest.raises(ValidationError):
            Stage1Beat(
                beat_id="b001",
                speaker="ANNOUNCER",
                intent="Frame the open in one line.",
                length_target_words=0,
                emotional_register="level",
            )

    def test_character_beat_min_floor_5_still_enforced(self):
        with pytest.raises(ValidationError):
            Stage1Beat(
                beat_id="b001",
                speaker="REN BLACK",
                intent="Open with a refusal.",
                length_target_words=4,  # 4 < 5
                emotional_register="guarded",
            )
        # And the floor itself is accepted.
        beat = Stage1Beat(
            beat_id="b001",
            speaker="REN BLACK",
            intent="Open with a refusal.",
            length_target_words=5,
            emotional_register="guarded",
        )
        assert beat.length_target_words == 5

    def test_upper_bound_still_200(self):
        # ge change does not relax the upper cap.
        with pytest.raises(ValidationError):
            Stage1Beat(
                beat_id="b001",
                speaker="REN BLACK",
                intent="Open with a refusal.",
                length_target_words=201,
                emotional_register="guarded",
            )


# ---------------------------------------------------------------------------
# BUG-LOCAL-283
# ---------------------------------------------------------------------------


class TestBugLocal283NewsBriefsCapBumpedTo7:
    """NewsBriefs.key_terms now accepts up to 7 entries."""

    def test_max_key_terms_constant_is_7(self):
        assert _ni._MAX_KEY_TERMS == 7

    def test_seven_key_terms_validates(self):
        seven = [
            "NASA", "FireSense", "Wildfire", "Thermal Sensor",
            "Fire Mission", "Fire Bulldozer", "Firefighter",
        ]
        # Schema constructs cleanly.
        nb = _ni.NewsBriefs(
            scene_atmosphere="A pine barrens at dusk with smoke on the ridge.",
            casting_brief="Two researchers on opposite sides of a publication call.",
            script_brief=(
                "A NASA team has fielded a heat sensor; the deploy on the "
                "ridge is now a question of who decides when the bulldozer "
                "moves and who lives with the call."
            ),
            news_close_brief="The deploy lands; the call holds.",
            key_terms=seven,
        )
        assert len(nb.key_terms) == 7

    def test_eight_key_terms_coerced_to_7(self):
        # BUG-LOCAL-264 (2026-06-03): 8 key_terms are now COERCED to the first 7,
        # NOT rejected. A weak model overruns the cap; the old rejection lost the
        # whole NewsBriefs object -> generic announcer intro/outro. (Supersedes the
        # prior test_eight_key_terms_rejected.)
        eight = [
            "NASA", "FireSense", "Wildfire", "Thermal Sensor",
            "Fire Mission", "Fire Bulldozer", "Firefighter", "Pine Barrens",
        ]
        nb = _ni.NewsBriefs(
            scene_atmosphere="x" * 50,
            casting_brief="y" * 50,
            script_brief="z" * 80,
            news_close_brief="w" * 50,
            key_terms=eight,
        )
        assert nb.key_terms == eight[:7]
        assert len(nb.key_terms) == 7

    def test_min_key_terms_constant_unchanged(self):
        # Lower-bound CONSTANT unchanged. The schema's min_length is
        # 1 (not _MIN_KEY_TERMS); the constant drives downstream
        # post-validation, not the pydantic gate.
        assert _ni._MIN_KEY_TERMS == 2

    def test_zero_key_terms_rejected_by_schema(self):
        # The pydantic gate's min_length=1 still rejects an empty
        # list.
        with pytest.raises(ValidationError):
            _ni.NewsBriefs(
                scene_atmosphere="x" * 50,
                casting_brief="y" * 50,
                script_brief="z" * 80,
                news_close_brief="w" * 50,
                key_terms=[],
            )


# ---------------------------------------------------------------------------
# BUG-LOCAL-284
# ---------------------------------------------------------------------------


class TestBugLocal284ReviewerEditPayloadOptional:
    """ReviewerEdit.payload now defaults to '' so annotate/skip rows
    without a payload no longer fail the schema."""

    def test_annotate_without_payload_validates(self):
        edit = ReviewerEdit(line_id="b002", action="annotate")
        assert edit.payload == ""
        assert edit.action == "annotate"

    def test_skip_without_payload_validates(self):
        edit = ReviewerEdit(line_id="b002", action="skip")
        assert edit.payload == ""

    def test_rewrite_without_payload_validates_but_emits_empty_text(self):
        # rewrite + empty payload is unusual but the schema-side fix
        # accepts it; the apply step's `edit.payload or ""` already
        # coerces and just produces an empty rewrite.
        edit = ReviewerEdit(line_id="b002", action="rewrite")
        assert edit.payload == ""

    def test_payload_explicitly_supplied_round_trips(self):
        edit = ReviewerEdit(
            line_id="b002", action="rewrite",
            payload="REN BLACK: I won't sign that paper.",
            rationale="strengthens the refusal",
        )
        assert edit.payload == "REN BLACK: I won't sign that paper."
        assert edit.rationale == "strengthens the refusal"

    def test_annotate_empty_payload_does_not_stamp_reviewer_note(self):
        # Source pin: the apply step's annotate branch now treats
        # empty payload as a no-op rather than NoneType stamping.
        from pathlib import Path
        src = Path(
            __import__("nodes._otr_ledger_reviewer", fromlist=["_otr_ledger_reviewer"])
            .__file__
        ).read_text(encoding="utf-8")
        # Pin: the annotate branch uses `(edit.payload or "").strip()`
        # gate before stamping reviewer_note.
        assert '(edit.payload or "").strip()' in src
        assert 'if note:' in src


# ---------------------------------------------------------------------------
# BUG-LOCAL-285
# ---------------------------------------------------------------------------


class TestBugLocal286DoctorEditsEmptyOutputShipsClean:
    """run_script_doctor_edits with an exhausted ladder whose last
    error is the "empty output" JSON-decode shape now returns CLEAN
    (no edits needed) instead of needs_full_rerun. Other failure
    shapes still route to needs_full_rerun."""

    def test_empty_output_routes_to_clean_verdict(self):
        from unittest.mock import patch
        from nodes import _otr_ledger_reviewer as _lr
        from nodes._otr_ledger_reviewer import (
            run_script_doctor_edits,
            ScriptDoctorReport,
            ScriptDoctorDiagnosis,
            LineDiagnosis,
        )
        import json as _j

        # Build an exhausted-ladder StructuredCallFailedError whose
        # last_error is the empty-output JSON-decode message the live
        # log surfaces.
        StructuredCallFailedError = _lr.StructuredCallFailedError
        empty_err = _j.JSONDecodeError(
            "Expecting value", "", 0,
        )
        exc = StructuredCallFailedError(
            helper_name="run_script_doctor_edits",
            attempts=3,
            last_error=empty_err,
        )

        def _bad_structured_call(*args, **kwargs):
            raise exc

        with patch.object(
            _lr, "structured_call", side_effect=_bad_structured_call,
        ):
            ledger = {
                "lines": [
                    {"line_id": "b001", "speaker_role": "announcer", "text": "x"},
                ],
                "cast": [],
            }
            diagnosis = ScriptDoctorDiagnosis(diagnoses=[
                LineDiagnosis(line_id="b001", failure="flat_exposition"),
            ])
            report = run_script_doctor_edits(
                lambda *a, **k: "",   # generate_fn
                ledger,                # candidate_ledger
                [],                    # cast_rows
                diagnosis,             # diagnosis
                10,                    # edit_cap
            )
        # BUG-286: empty output -> clean (no rerun blocking).
        assert report.overall_verdict == "clean"
        assert report.edits == []

    def test_real_schema_error_still_routes_to_needs_full_rerun(self):
        """A real schema validation failure should NOT trigger the
        empty-output bypass. The "line 1 column 1 (char 0)" match
        anchor only fires for the empty-output decode shape."""
        from unittest.mock import patch
        from nodes import _otr_ledger_reviewer as _lr
        from nodes._otr_ledger_reviewer import (
            run_script_doctor_edits,
            ScriptDoctorDiagnosis,
            LineDiagnosis,
        )
        from pydantic import ValidationError

        # Cook up a ValidationError. We don't need a real schema --
        # just any ValidationError instance works for the type
        # check inside the exception handler.
        try:
            _lr.ScriptDoctorReport.model_validate(
                {"edits": [{"line_id": "b001", "action": "rewrite",
                            "payload": 12345}],  # int where str expected
                 "overall_verdict": "clean"}
            )
        except ValidationError as ve:
            schema_err = ve

        exc = _lr.StructuredCallFailedError(
            helper_name="run_script_doctor_edits",
            attempts=3,
            last_error=schema_err,
        )

        def _bad(*args, **kwargs):
            raise exc

        with patch.object(_lr, "structured_call", side_effect=_bad):
            report = run_script_doctor_edits(
                lambda *a, **k: "",
                {"lines": [], "cast": []},
                [],
                ScriptDoctorDiagnosis(diagnoses=[]),
                10,
            )
        # Real schema error still trips the full-rerun gate.
        assert report.overall_verdict == "needs_full_rerun"


class TestBugLocal285ReflectionTokenBudgetBumped:
    """run_story_brief_reflection got max_new_tokens >= 512 so the v2
    schema's nine fields fit without starving the model into empty
    output."""

    def test_max_new_tokens_at_least_512(self):
        assert _osb._REFLECTION_MAX_NEW_TOKENS >= 512

    def test_prompt_version_still_v2(self):
        # The bump is paired with the v2 schema; if we ever roll v2
        # back to v1 (5 fewer fields) the cap can drop too.
        assert _osb._PROMPT_VERSION == "v2"

    def test_structural_retry_temperature_strictly_below_base(self):
        # PD invariant -- structural retries lower entropy, never
        # raise it. The cap change does not touch this.
        assert _osb._REFLECTION_STRUCTURAL_RETRY_TEMPERATURE < \
            _osb._REFLECTION_TEMPERATURE
