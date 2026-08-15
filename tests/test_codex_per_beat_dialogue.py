"""tests/test_codex_per_beat_dialogue.py -- one beat per job, then review.

PBUG-20260814-03. The `scifi_news` dialogue job used to write the WHOLE play
in one call -- up to twenty-four rows in a single reply -- and the published
2026-08-13 ledger shows what came back: narrated third-person prose with the
dialogue quoted inside it. `l002` is 100% narration and no dialogue at all;
TTS read every stage direction on air. The operator's diagnosis, written
before that artifact was read: "A model asked for one beat writes that beat.
A model asked for a whole act writes a summary of one."

What is pinned here is the SHAPE, because the shape is the fix:

  * one dialogue job per accepted beat, carrying only that beat's line ids;
  * a WINDOW, not the whole play -- this beat, the spine it sits in, and what
    the listener has already heard, so the writer can answer the line before
    this one instead of averaging an act;
  * one review job per scene, after its last beat, which may rewrite rows
    against the spine but may not add, drop or renumber them;
  * the next scene is written against what the review LEFT;
  * per-job decode budgets. Right-size the job; never raise the guard.

Pure-Python. No GPU. No LLM.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_codex as lane  # noqa: E402


# ---------------------------------------------------------------------------
# A two-scene score: scene_001 has two beats, scene_002 has one.
# ---------------------------------------------------------------------------

def _cast() -> lane.CastPlanV4:
    return lane.CastPlanV4(cast=[
        {
            "char_id": "announcer", "name": "ANNOUNCER",
            "character_description": "A calm witness.",
            "gender": "neutral", "role_in_conflict": "Frames the signal.",
            "voice_slot": "announcer",
        },
        {
            "char_id": "c01", "name": "Ada Sterling",
            "character_description": "A careful radio astronomer.",
            "gender": "female", "role_in_conflict": "Must answer the signal.",
            "voice_slot": "c01",
        },
    ])


def _beat(beat_id, scene_id, shot_id, speaker, char_id, role, line_ids,
          order, intent, arc_phase) -> dict[str, Any]:
    return {
        "beat_id": beat_id, "scene_id": scene_id, "shot_id": shot_id,
        "speaker": speaker, "char_id": char_id, "speaker_role": role,
        "line_ids": list(line_ids), "order": order, "intent": intent,
        "arc_phase": arc_phase, "fact_ids": ["F01"],
    }


def _score() -> lane.RadioScoreV4:
    return lane.RadioScoreV4(
        title="Signal at Meridian",
        premise="A signal asks for an answer.",
        setting="Meridian observatory.",
        advisory_word_plan=lane.AdvisoryWordPlanV4(per_beat=[
            {"beat_id": "b000"}, {"beat_id": "b001"}, {"beat_id": "b002"},
        ]),
        scenes=[
            {
                "scene_id": "scene_001", "env": "Observatory",
                "description": "Receivers glow beneath the dome.",
                "shots": [{
                    "shot_id": "shot_001", "scene_id": "scene_001",
                    "description": "Ada at the receiver.",
                    "visual_prompt": "An astronomer at a blue receiver.",
                }],
                "beats": [
                    _beat("b000", "scene_001", "shot_001", "ANNOUNCER",
                          "announcer", "announcer", ["l001"], 1,
                          "Open the night.", "arrival"),
                    _beat("b001", "scene_001", "shot_001", "Ada Sterling",
                          "c01", "character", ["l002", "l003"], 2,
                          "Name what she heard.", "pressure"),
                ],
            },
            {
                "scene_id": "scene_002", "env": "Control room",
                "description": "The console hums alone.",
                "shots": [{
                    "shot_id": "shot_002", "scene_id": "scene_002",
                    "description": "The console at dawn.",
                    "visual_prompt": "A quiet console under grey light.",
                }],
                "beats": [
                    _beat("b002", "scene_002", "shot_002", "Ada Sterling",
                          "c01", "character", ["l004"], 3,
                          "Choose.", "reversal"),
                ],
            },
        ],
        music_cues=[{
            "cue_id": "music_open", "placement": "open",
            "description": "A low radio pulse.",
            "generation_prompt": "Low sustained radio pulse.",
            "anchor_line_id": "l001", "anchor_beat_id": "b000",
        }],
    )


#: The smallest context window any shipped writer offers -- the `context_window`
#: default in `_otr_model_catalog`, which `gemma-4-12b-it` (the saved
#: runtime-qualified default) carries. A generation request at or above this can
#: never be served, however short the prompt is, so it is a hard ceiling on any
#: `max_new_tokens` the lane asks for rather than a tuning target.
_SMALLEST_SHIPPED_CONTEXT_WINDOW = 8192


class _Schedule:
    """Records every job and answers each with `text_for(line_id)`."""

    def __init__(self, *, review=None, text_for=None):
        self.jobs: list[dict[str, Any]] = []
        self._review = review or (lambda line_id, text: text)
        self._text_for = text_for or (
            lambda line_id: f"Spoken row {line_id} in the moment."
        )

    def __call__(self, **kwargs):
        self.jobs.append(kwargs)
        inputs = kwargs["artifact_inputs"]
        if kwargs["pass_id"] == "P5B":
            rows = [
                {"line_id": lid, "text": self._text_for(lid)}
                for lid in inputs["this_beat"]["line_ids"]
            ]
            result: Any = lane.BeatTextDraftV4(lines=rows)
        elif kwargs["pass_id"] == "P5R":
            text_by_id = {
                row["line_id"]: row["text"] for row in inputs["scene_rows"]
            }
            result = lane.SceneReviewDraftV4(lines=[
                {
                    "line_id": lid,
                    "text": self._review(lid, text_by_id[lid]),
                }
                for lid in inputs["scene_line_ids"]
            ])
        else:  # pragma: no cover - the assertion is the contract
            raise AssertionError(f"unexpected pass {kwargs['pass_id']}")
        error = kwargs["post_validator"](result)
        assert error is None, error
        return result

    def of(self, pass_id: str) -> list[dict[str, Any]]:
        return [job for job in self.jobs if job["pass_id"] == pass_id]


def _run(monkeypatch, schedule: _Schedule, journal=None):
    monkeypatch.setattr(lane, "invoke_codex_structured", schedule)
    return lane._call_script_text_draft(
        slot_fn=lambda *_a, **_k: "",
        pack=SimpleNamespace(prompt_stages={
            "codex_play_system": "Write the spoken line.",
        }),
        artifact_inputs={
            "story_context": {"title": "Signal at Meridian"},
            "fact_index": {"facts": [], "tone": "measured"},
            # The whole-play graph the old single call was handed. It must NOT
            # ride along into a beat job.
            "accepted_line_graph": [{"line_id": "l001"}],
        },
        score=_score(),
        cast=_cast(),
        call_journal={} if journal is None else journal,
    )


# ---------------------------------------------------------------------------
# 1. One job per beat, one review per scene
# ---------------------------------------------------------------------------

def test_one_dialogue_job_per_beat_and_one_review_per_scene(monkeypatch):
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    assert [job["pass_id"] for job in schedule.jobs] == [
        "P5B", "P5B", "P5R",   # scene_001: two beats, then its review
        "P5B", "P5R",          # scene_002: one beat, then its review
    ]


def test_a_dialogue_job_is_handed_only_its_own_line_ids(monkeypatch):
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    beat_jobs = schedule.of("P5B")
    assert [job["artifact_inputs"]["this_beat"]["line_ids"] for job in
            beat_jobs] == [["l001"], ["l002", "l003"], ["l004"]]
    assert [job["artifact_inputs"]["this_beat"]["beat_id"] for job in
            beat_jobs] == ["b000", "b001", "b002"]


def test_the_window_is_the_beat_not_the_play(monkeypatch):
    """The whole point. A beat job never sees the full line graph.

    Handing the model every row is what produced a summary of an act instead
    of an act; the window is what makes the smaller job possible.
    """
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    first = schedule.of("P5B")[0]["artifact_inputs"]
    assert "accepted_line_graph" not in first
    assert "accepted_line_ids" not in first
    assert set(first) == {
        "story_context", "fact_index", "this_scene", "this_beat",
        "rows_so_far",
    }
    assert first["this_scene"]["scene_id"] == "scene_001"
    assert [row["beat_id"] for row in first["this_scene"]["beats"]] == [
        "b000", "b001",
    ]
    assert first["this_beat"]["intent"] == "Open the night."
    assert first["this_beat"]["arc_phase"] == "arrival"
    assert first["this_beat"]["speaker"] == "ANNOUNCER"


def test_a_beat_sees_what_the_listener_has_already_heard(monkeypatch):
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    windows = [
        job["artifact_inputs"]["rows_so_far"] for job in schedule.of("P5B")
    ]
    assert windows[0] == []
    assert [row["line_id"] for row in windows[1]] == ["l001"]
    assert windows[1][0]["speaker"] == "ANNOUNCER"
    assert windows[1][0]["text"] == "Spoken row l001 in the moment."
    # The third beat opens a new scene and still carries the whole episode so
    # far -- a scene boundary is not an amnesia boundary.
    assert [row["line_id"] for row in windows[2]] == ["l001", "l002", "l003"]


# ---------------------------------------------------------------------------
# 2. The review
# ---------------------------------------------------------------------------

def test_the_review_reads_its_scenes_rows_and_its_spine(monkeypatch):
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    first_review = schedule.of("P5R")[0]["artifact_inputs"]
    assert first_review["scene_line_ids"] == ["l001", "l002", "l003"]
    assert [row["line_id"] for row in first_review["scene_rows"]] == [
        "l001", "l002", "l003",
    ]
    assert first_review["this_scene"]["scene_id"] == "scene_001"
    assert [row["intent"] for row in first_review["this_scene"]["beats"]] == [
        "Open the night.", "Name what she heard.",
    ]


def test_a_rewritten_row_is_what_ships(monkeypatch):
    """Code detects; a MODEL rewrites. This is the rewrite landing."""
    def review(line_id, text):
        return "The receiver answered first." if line_id == "l002" else text

    schedule = _Schedule(review=review)
    script = _run(monkeypatch, schedule)

    by_id = {line.line_id: line.text for line in script.lines}
    assert by_id["l002"] == "The receiver answered first."
    assert by_id["l001"] == "Spoken row l001 in the moment."


def test_the_next_scene_is_written_against_what_the_review_left(monkeypatch):
    def review(line_id, text):
        return "The receiver answered first." if line_id == "l002" else text

    schedule = _Schedule(review=review)
    _run(monkeypatch, schedule)

    # The scene_002 beat is the third dialogue job; its window must carry the
    # REVIEWED text for l002, not the draft the review replaced.
    window = schedule.of("P5B")[2]["artifact_inputs"]["rows_so_far"]
    rewritten = next(row for row in window if row["line_id"] == "l002")
    assert rewritten["text"] == "The receiver answered first."


def test_a_review_that_drops_a_row_is_refused(monkeypatch):
    """Review may rewrite. It may not add, drop or renumber."""
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    validator = schedule.of("P5R")[0]["post_validator"]
    short = lane.SceneReviewDraftV4(lines=[
        {"line_id": "l001", "text": "Only the first row came back."},
    ])
    error = validator(short)
    assert error is not None
    assert "do not exactly cover the closed set" in error
    assert "missing=['l002', 'l003']" in error

    invented = lane.SceneReviewDraftV4(lines=[
        {"line_id": "l001", "text": "The first row."},
        {"line_id": "l002", "text": "The second row."},
        {"line_id": "l003", "text": "The third row."},
        {"line_id": "l009", "text": "A row the scene never had."},
    ])
    error = validator(invented)
    assert error is not None
    assert "unknown=['l009']" in error


# ---------------------------------------------------------------------------
# 3. Budgets and the receipt
# ---------------------------------------------------------------------------

def test_every_job_carries_its_own_right_sized_decode_budget(monkeypatch):
    """Right-size the job, never raise the guard.

    The whole-play pass reserved a full provider window because its size
    could not be pre-judged. A beat cannot legally exceed two rows, so it
    does not need one -- and a budget matched to the job is what stops a
    degenerate call spending an episode's allowance on one sentence.
    """
    schedule = _Schedule()
    _run(monkeypatch, schedule)

    for job in schedule.of("P5B"):
        assert job["max_new_tokens"] == lane._BEAT_TEXT_MAX_OUTPUT_TOKENS
        assert job["prompt_must_fit"] is True
    for job in schedule.of("P5R"):
        # SIZED TO THE SCENE IN HAND, not to the schema ceiling. This assertion
        # used to read `== lane._SCENE_REVIEW_MAX_OUTPUT_TOKENS`, and that
        # constant was 8320 against an 8192-token context window -- so the test
        # passed by agreeing with a request that could never be served, and a
        # live scifi_news leg died on it at 10:18 (PBUG-20260815-10). Pin the
        # PROPERTY that matters instead of the number that was wrong.
        assert job["max_new_tokens"] == lane.scene_review_output_tokens(
            len(job["artifact_inputs"]["scene_line_ids"]))
        assert job["prompt_must_fit"] is True
        assert job["max_new_tokens"] < _SMALLEST_SHIPPED_CONTEXT_WINDOW, (
            "a request larger than the whole context can never be served, "
            "however short the prompt is"
        )
    # The beat budget is a fraction of the scene's, which is a fraction of a
    # whole script -- the property that makes it a right-sized job rather
    # than a relabelled ceiling.
    assert lane._BEAT_TEXT_MAX_OUTPUT_TOKENS < \
        lane.scene_review_output_tokens(lane._RADIO_SCORE_MAX_BEATS_PER_SCENE)


#: The P5R prompt measured on a real leg (PBUG-20260815-10's own log line:
#: "prompt requires 1203 input tokens"). Used as the prompt allowance a request
#: must leave room for -- an assertion about output alone would pass on a budget
#: that still cannot run.
_MEASURED_P5R_PROMPT_TOKENS = 1203


def test_NO_schema_legal_scene_can_outgrow_the_context_window():
    """The hole the first fix for PBUG-20260815-10 left open.

    Replacing the impossible 8320 CONSTANT with a request sized to the scene
    cured the constant and left the death reachable, because SCENE SIZE IS
    MODEL-CONTROLLED: the schema accepts up to `_RADIO_SCORE_MAX_BEATS_PER_SCENE`
    (8) beats x 2 lines = 16 rows, and the P3 prompt explicitly invites "at most
    8 beats per scene". Sixteen rows is 8320 tokens -- the exact number that
    killed a live leg -- and it bites from 7 beats up, since 7296 + a
    ~1203-token prompt already exceeds 8192.

    So the guarantee has to hold for EVERY size the schema permits, not for the
    size the topology intends. Walk all of them.
    """
    max_rows = (
        lane._RADIO_SCORE_MAX_BEATS_PER_SCENE
        * lane._RADIO_SCORE_MAX_LINES_PER_BEAT
    )
    for rows in range(1, max_rows + 1):
        per_call = min(rows, lane._SCENE_REVIEW_MAX_ROWS_PER_CALL)
        request = lane.scene_review_output_tokens(per_call)
        assert request + _MEASURED_P5R_PROMPT_TOKENS < \
            _SMALLEST_SHIPPED_CONTEXT_WINDOW, (
                f"a {rows}-row scene requests {request} tokens, which cannot "
                f"be served with a {_MEASURED_P5R_PROMPT_TOKENS}-token prompt"
            )


def test_an_OVERSIZED_scene_is_reviewed_in_CHUNKS_and_loses_no_row(monkeypatch):
    """The worst scene the schema allows still gets reviewed, in pieces.

    Chunking costs the writing nothing: every call still carries the whole
    scene's rows and spine, so the model reads the entire scene back exactly as
    the design intends -- only the rows it must RETURN are split.
    """
    ids = [f"l{i:03d}" for i in range(
        lane._RADIO_SCORE_MAX_BEATS_PER_SCENE
        * lane._RADIO_SCORE_MAX_LINES_PER_BEAT)]
    jobs: list[dict[str, Any]] = []

    def fake_invoke(**kwargs):
        jobs.append(kwargs)
        wanted = kwargs["artifact_inputs"]["scene_line_ids"]
        return lane.SceneReviewDraftV4(lines=[
            {"line_id": lid, "text": f"Reviewed row {lid} in the moment."}
            for lid in wanted
        ])

    monkeypatch.setattr(lane, "invoke_codex_structured", fake_invoke)
    out = lane._call_scene_review(
        slot_fn=lambda **k: "",
        pack=None,
        artifact_inputs={
            "scene_line_ids": list(ids),
            "scene_rows": [{"line_id": i, "text": "x"} for i in ids],
        },
        scene_line_ids=ids,
        label_pattern=re.compile(r"."),
        call_journal={},
    )

    assert [str(row.line_id) for row in out] == ids, "a row was lost"
    assert len(jobs) == 2, "16 rows must split into two 8-row calls"
    for job in jobs:
        assert job["max_new_tokens"] + _MEASURED_P5R_PROMPT_TOKENS < \
            _SMALLEST_SHIPPED_CONTEXT_WINDOW
        # The whole scene stays visible in every call -- only the return set
        # narrows. Losing that would make chunking a quality regression.
        assert len(job["artifact_inputs"]["scene_rows"]) == len(ids)


def test_a_REAL_scene_still_takes_exactly_one_review_call(monkeypatch):
    """Chunking must be invisible in production. A 4-beat scene is 8 rows."""
    ids = [f"l{i:03d}" for i in range(lane._SCENE_REVIEW_MAX_ROWS_PER_CALL)]
    jobs: list[dict[str, Any]] = []

    def fake_invoke(**kwargs):
        jobs.append(kwargs)
        return lane.SceneReviewDraftV4(lines=[
            {"line_id": lid, "text": f"Reviewed row {lid} in the moment."}
            for lid in kwargs["artifact_inputs"]["scene_line_ids"]
        ])

    monkeypatch.setattr(lane, "invoke_codex_structured", fake_invoke)
    lane._call_scene_review(
        slot_fn=lambda **k: "", pack=None,
        artifact_inputs={"scene_line_ids": list(ids), "scene_rows": []},
        scene_line_ids=ids, label_pattern=re.compile(r"."), call_journal={},
    )
    assert len(jobs) == 1


def test_the_beat_array_ceiling_is_the_beats_own(monkeypatch):
    """An unenforced array ceiling is the root cause of PBUG-20260729-02.

    A job that can legally emit two rows cannot run away into twenty-four.
    """
    with pytest.raises(Exception):
        lane.BeatTextDraftV4(lines=[
            {"line_id": f"l{index:03d}", "text": "row"}
            for index in range(1, lane._RADIO_SCORE_MAX_LINES_PER_BEAT + 2)
        ])


def test_the_beat_window_is_capped_on_both_axes():
    """Uncapped, the window is a cliff that falls on the LAST beat.

    Twelve beats of two lines, each legally up to the degeneracy threshold, is
    a ~285,000-character prompt. Real episodes are nowhere near it, and
    prompt_must_fit would refuse rather than truncate -- but the refusal would
    land on the longest episode's final beat, which is the worst place to
    discover a limit.
    """
    many = [{"line_id": f"l{i:03d}", "speaker": "Ada", "text": "word"}
            for i in range(40)]
    kept = lane._recent_window(many)
    assert len(kept) == lane._BEAT_WINDOW_MAX_ROWS
    # The TAIL is what a beat needs: it answers the line before it.
    assert kept[-1]["line_id"] == "l039"

    huge = [
        {"line_id": "l001", "speaker": "Ada", "text": "x" * 20_000},
        {"line_id": "l002", "speaker": "Leo", "text": "the last word"},
    ]
    kept = lane._recent_window(huge)
    assert [row["line_id"] for row in kept] == ["l002"]

    # A single oversized row still survives when it is all there is, so the
    # beat is never handed an empty window by the cap alone.
    assert lane._recent_window(huge[:1])[0]["line_id"] == "l001"


def test_a_short_episode_window_is_untouched(monkeypatch):
    schedule = _Schedule()
    _run(monkeypatch, schedule)
    windows = [j["artifact_inputs"]["rows_so_far"]
               for j in schedule.of("P5B")]
    assert [len(w) for w in windows] == [0, 1, 3]


def test_the_schedule_is_receipted(monkeypatch):
    journal: dict[str, Any] = {}
    schedule = _Schedule()
    _run(monkeypatch, schedule, journal=journal)

    receipt = journal["script_schedule"]
    assert receipt["shape"] == "per_beat_dialogue_then_scene_review"
    assert receipt["beat_dialogue_jobs"] == 3
    assert receipt["scene_review_jobs"] == 2
    assert receipt["accepted_line_count"] == 4
    assert receipt["accepted_transport"]["schema"] == "ScriptTextDraftV4"
    assert len(receipt["accepted_transport"]["sha256"]) == 64


def test_the_assembled_script_is_the_scores_own_order(monkeypatch):
    schedule = _Schedule()
    script = _run(monkeypatch, schedule)

    assert [line.line_id for line in script.lines] == [
        "l001", "l002", "l003", "l004",
    ]
    assert [line.beat_id for line in script.lines] == [
        "b000", "b001", "b001", "b002",
    ]
    assert [line.speaker_role for line in script.lines] == [
        "announcer", "character", "character", "character",
    ]
