"""P3 must not accept a score that leaves a cast member with no beat.

THE PRODUCTION FAILURE (measured 2026-08-05, batch v2): the `scifi_news` lane
went 0-for-4 while shakespeare went 8/8, public_domain 4/4 and original 1/1.
One of those failures burned 45.1 minutes of GPU before dying. Every one died
the same way -- `_otr_cast_voice_coverage` refusing at stamp_receipt because a
cast row owned no SAYABLE line -- and the live logs show it is specifically an
uncovered ANNOUNCER surviving P3 into P5.

WHY THE FIX IS HERE AND NOT AT P5. Beat/shot/scene ownership is compiled from
the accepted score; P5 authors only line_id and text INTO that graph. A cast
member who owns no beat after P3 can never be given a line by any number of P5
redrafts, so a P5-level coverage check could only burn the ladder and then fail
anyway -- converting a 5-minute failure into a 45-minute one.

WHY IT IS RECOVERABLE AND NOT FATAL. A prior review made this advisory to stop
cast_coverage becoming an accidental fatal successor to the removed beat-count
gate. That concern is respected: this raises a DRAFT error, which
`validate_draft` turns into a PostValidationError, which the retry ladder and
then the fresh-candidate loop treat as recoverable. The score is retired and
redrafted; the episode is not failed. `_codex_target_beat_count` returns
max(cast_count, 3, ...), so the topology always reserves a beat per cast member
-- undercoverage is a drafting accident, not an impossibility.
"""

from __future__ import annotations

import pytest

from nodes import _otr_scifi_codex as lane


def _cast() -> lane.CastPlanV4:
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


def _advisory(beats: int) -> lane.AdvisoryWordPlanV4:
    return lane.AdvisoryWordPlanV4(
        advisory_total_center=6 * beats,
        per_beat=[
            {"beat_id": f"b{index + 1:03d}", "advisory_word_center": 6}
            for index in range(beats)
        ],
    )


def _facts() -> lane.FactIndexV4:
    span = lane.SourceSpanV4(
        field="summary", start=0, end=22, quote="A signal reached Meridian",
    )
    return lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id="F01",
            claim="A signal reached the Meridian array.",
            source_spans=[span],
        )],
        entities=[lane.EntityV4(
            entity_id="E01", name="Meridian", source_spans=[span],
        )],
        numbers=[],
        tone="measured",
        payload_sha256="0" * 64,
    )


def _draft(char_ids: "list[str]") -> lane.RadioScoreDraftV4:
    """One scene, one shot, one beat per requested char_id."""
    return lane.RadioScoreDraftV4(
        title="Signal at Meridian",
        premise="A signal asks for an answer.",
        setting="Meridian observatory.",
        scenes=[
            lane.RadioScoreDraftSceneV4(
                env="Observatory",
                description="Receivers glow against a cold window.",
                shots=[
                    lane.RadioScoreDraftShotV4(
                        description="Ada at the receiver.",
                        visual_prompt="A radio astronomer at a blue receiver.",
                    )
                ],
                beats=[
                    lane.RadioScoreDraftBeatV4(
                        shot_index=0,
                        char_id=char_id,
                        line_count=1,
                        intent="Answer the signal.",
                        arc_phase="arrival",
                        fact_ids=[],
                    )
                    for char_id in char_ids
                ],
            )
        ],
        music_cues=[
            lane.RadioScoreDraftMusicCueV4(
                cue_id="music_open",
                description="A patient electronic chord.",
                generation_prompt="Instrumental radio texture.",
                anchor_beat_index=0,
                anchor_line_index=0,
            )
        ],
    )


def test_cast_coverage_is_a_recoverable_draft_error_code():
    """It must be a DRAFT code, or the retry ladder cannot carry it."""
    assert "cast_coverage" in lane._DRAFT_ERROR_CODES


def test_a_score_that_covers_every_cast_member_compiles():
    score = lane.compile_radio_score_draft(
        _draft(["announcer", "c01"]), _advisory(2), _cast(), _facts(),
    )
    covered = {
        beat.char_id for scene in score.scenes for beat in scene.beats
    }
    assert covered == {"announcer", "c01"}


def test_an_uncovered_announcer_retires_the_draft_and_names_it():
    """The exact live failure: the announcer is planned but owns no beat."""
    with pytest.raises(lane.RadioScoreDraftCompileError) as excinfo:
        lane.compile_radio_score_draft(
            _draft(["c01"]), _advisory(1), _cast(), _facts(),
        )
    error = excinfo.value
    assert error.code == "cast_coverage"
    # The missing id must be NAMED -- the redraft prompt is built from this
    # message, so an unnamed complaint cannot be acted on.
    assert "announcer" in error.detail
    assert "1/2" in error.detail


def test_an_uncovered_character_is_caught_too():
    with pytest.raises(lane.RadioScoreDraftCompileError) as excinfo:
        lane.compile_radio_score_draft(
            _draft(["announcer"]), _advisory(1), _cast(), _facts(),
        )
    assert excinfo.value.code == "cast_coverage"
    assert "c01" in excinfo.value.detail


def test_the_error_is_classified_recoverable_by_the_candidate_loop():
    """Proves the loop will REDRAFT rather than fail the episode.

    `validate_draft` converts a RadioScoreDraftCompileError into a returned
    error string, which the structured-call machinery raises as
    PostValidationError -- and that IS in the recoverable set. Without this the
    fix would swap a late failure for an early one instead of a retry.
    """
    assert lane._candidate_error_is_recoverable(
        lane.PostValidationError("draft.cast_coverage at scenes[].beats[]")
    )


def test_the_score_prompt_states_the_invariant():
    """The model cannot satisfy a rule it is never told."""
    import json
    from pathlib import Path

    pack = json.loads(
        (Path(__file__).resolve().parents[1]
         / "nodes" / "story_packs" / "scifi_news" / "scifi_news.json")
        .read_text(encoding="utf-8")
    )
    prompt = pack["prompt_stages"]["codex_radio_score_system"]
    assert "must own at least one beat" in prompt
    assert "announcer included" in prompt
