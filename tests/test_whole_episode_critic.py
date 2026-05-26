"""Sprint 10A step 7 regression -- rubric loader + whole-episode critic.

Covers:
  * _otr_critic_rubric.load_rubric:
      - Parses the canonical project rubric markdown.
      - Returns 10 axes, each with exactly 5 anchors.
      - Threshold defaults match the rubric's documented rule
        (no axis < 3, mean >= 3.5, SFW >= 4).
      - Wrong-axis-count or wrong-anchor-count input raises
        ValueError loudly.
      - json_key slug is the documented form (lowercased,
        spaces -> underscores, ASCII letters only).

  * _otr_whole_episode_critic:
      - build_critic_prompt produces a 2-message chat with the
        rubric block + transcript + JSON-shape instruction.
      - compute_verdict_locally enforces the threshold rule
        independent of the LLM's claimed verdict.
      - run_whole_episode_critic happy path with a mocked
        generate_fn returning canned valid JSON.
      - Retry on parse failure, recover on attempt 2.
      - rubric_key_mismatch on out-of-pool axis keys.
      - All attempts fail -> CriticCallFailedError.

Tests do NOT load a real LLM. The rubric markdown IS read from the
project's actual file (it's a stable contract surface).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from nodes._otr_critic_rubric import (
    Rubric,
    RubricAxis,
    ShipThreshold,
    load_rubric,
)
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)
from nodes._otr_whole_episode_critic import (
    AxisScore,
    CriticCallFailedError,
    CriticResult,
    WholeEpisodeCriticReport,
    build_critic_prompt,
    compute_verdict_locally,
    run_whole_episode_critic,
)


# ---------------------------------------------------------------------------
# Rubric loader
# ---------------------------------------------------------------------------


class TestRubricLoaderCanonical:
    def test_canonical_rubric_loads(self):
        r = load_rubric()
        assert isinstance(r, Rubric)
        assert len(r.axes) == 10

    def test_each_axis_has_five_anchors(self):
        r = load_rubric()
        for axis in r.axes:
            assert len(axis.anchors) == 5, (
                f"axis {axis.name!r} has {len(axis.anchors)} anchors, "
                f"expected 5"
            )

    def test_json_keys_are_slugified(self):
        r = load_rubric()
        # Spot check the 10 expected keys.
        expected_keys = {
            "premise_clarity",
            "character_distinctiveness",
            "continuity",
            "naturalness",
            "pacing",
            "emotional_arc",
            "resolution",
            "specificity",
            "sfw_adherence",
            "audio_readiness",
        }
        got_keys = {a.json_key for a in r.axes}
        assert got_keys == expected_keys

    def test_ordinals_are_1_through_10(self):
        r = load_rubric()
        ordinals = [a.ordinal for a in r.axes]
        assert ordinals == list(range(1, 11))

    def test_threshold_defaults_match_rubric_doc(self):
        r = load_rubric()
        th = r.threshold
        assert th.min_axis_score == 3
        assert th.min_mean_score == 3.5
        assert th.sfw_axis_key == "sfw_adherence"
        assert th.sfw_axis_min == 4


class TestRubricLoaderEdgeCases:
    def test_wrong_axis_count_raises(self, tmp_path):
        rubric_text = (
            "## Axes\n\n"
            "### 1. Solo axis\n\n"
            "**What it checks:** Just one axis is wrong.\n\n"
            "- **1 -- A.** body\n"
            "- **2 -- B.** body\n"
            "- **3 -- C.** body\n"
            "- **4 -- D.** body\n"
            "- **5 -- E.** body\n"
        )
        f = tmp_path / "rubric.md"
        f.write_text(rubric_text, encoding="utf-8")
        with pytest.raises(ValueError, match="expected to parse 10 axes"):
            load_rubric(f)

    def test_wrong_anchor_count_raises(self, tmp_path):
        # 10 axes but the first has only 4 anchors.
        axes_md = ["## Axes\n"]
        for i in range(1, 11):
            anchors_n = 4 if i == 1 else 5
            axes_md.append(f"### {i}. Axis {i}\n")
            axes_md.append(f"**What it checks:** Definition {i}.\n\n")
            for j in range(1, anchors_n + 1):
                axes_md.append(f"- **{j} -- Label.** Body for {j}.\n")
            axes_md.append("\n")
        f = tmp_path / "rubric.md"
        f.write_text("\n".join(axes_md), encoding="utf-8")
        with pytest.raises(ValueError, match="anchors"):
            load_rubric(f)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _plan() -> Stage1Plan:
    return Stage1Plan(
        premise="Two technicians at a Mars relay catch a signal that should not exist.",
        arc=Stage1Arc(
            setup="Routine night shift, calm comms traffic.",
            complication="An off-band carrier locks in.",
            resolution="They identify the source and acknowledge it.",
        ),
        cast=[
            Stage1CastMember(
                name="DALE PORTER", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="Experienced relay technician with measured cadences.",
                arc_role="anchor of competence",
            ),
            Stage1CastMember(
                name="MEG SAARI", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_2",
                persona="Junior tech, six months in, quick to question.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(beat_id="b001", speaker="ANNOUNCER",
                       intent="Open.", length_target_words=18,
                       emotional_register="neutral", callback_to=None),
            Stage1Beat(beat_id="b002", speaker="DALE PORTER",
                       intent="Notice the carrier.", length_target_words=24,
                       emotional_register="alert", callback_to=None),
            Stage1Beat(beat_id="b003", speaker="MEG SAARI",
                       intent="Ask about it.", length_target_words=22,
                       emotional_register="alert curious",
                       callback_to="b002"),
        ],
        running_facts=["The relay is on Mars."],
    )


def _rendered_lines() -> list[dict]:
    return [
        {"beat_id": "b001", "speaker": "ANNOUNCER",
         "text": "Welcome to Signal Lost, episode three."},
        {"beat_id": "b002", "speaker": "DALE PORTER",
         "text": "The carrier is off-band, channel four, eastern edge."},
        {"beat_id": "b003", "speaker": "MEG SAARI",
         "text": "Off-band? That shouldn't be possible at this frequency."},
    ]


class TestCriticPrompt:
    def test_prompt_is_system_plus_user(self):
        rubric = load_rubric()
        messages = build_critic_prompt(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            rubric=rubric,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_includes_all_axis_keys(self):
        rubric = load_rubric()
        messages = build_critic_prompt(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            rubric=rubric,
        )
        user = messages[1]["content"]
        for axis in rubric.axes:
            assert axis.json_key in user
            assert axis.name in user

    def test_prompt_includes_threshold_rule(self):
        rubric = load_rubric()
        messages = build_critic_prompt(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            rubric=rubric,
        )
        user = messages[1]["content"]
        assert "3.5" in user                    # mean threshold
        assert "sfw_adherence" in user
        assert "ship" in user.lower()
        assert "discard" in user.lower()

    def test_prompt_includes_transcript(self):
        rubric = load_rubric()
        messages = build_critic_prompt(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            rubric=rubric,
        )
        user = messages[1]["content"]
        assert "DALE PORTER" in user
        assert "off-band, channel four" in user


# ---------------------------------------------------------------------------
# Local verdict computation
# ---------------------------------------------------------------------------


def _good_axis_scores(rubric: Rubric, score: int = 4) -> dict[str, AxisScore]:
    return {
        a.json_key: AxisScore(
            score=score, justification=f"Solid {a.json_key} performance.",
        )
        for a in rubric.axes
    }


class TestLocalVerdictComputation:
    def test_all_fours_ships(self):
        rubric = load_rubric()
        report = WholeEpisodeCriticReport(
            verdict="ship",
            axis_scores=_good_axis_scores(rubric, 4),
            mean_score=4.0,
            failing_axes=[],
            regeneration_hint="No regeneration needed; ship as-is.",
        )
        verdict, mean, failing = compute_verdict_locally(report, rubric)
        assert verdict == "ship"
        assert mean == 4.0
        assert failing == []

    def test_one_axis_below_3_discards(self):
        rubric = load_rubric()
        scores = _good_axis_scores(rubric, 4)
        # Set premise_clarity to 2 -- below threshold.
        scores["premise_clarity"] = AxisScore(
            score=2, justification="Premise muddled.",
        )
        report = WholeEpisodeCriticReport(
            verdict="ship",   # LLM says ship; we override.
            axis_scores=scores,
            mean_score=3.8,
            failing_axes=[],
            regeneration_hint="Sharpen the premise.",
        )
        verdict, _, failing = compute_verdict_locally(report, rubric)
        assert verdict == "discard"
        assert "premise_clarity" in failing

    def test_sfw_at_3_discards_even_if_other_axes_4(self):
        rubric = load_rubric()
        scores = _good_axis_scores(rubric, 4)
        # SFW at 3 -- meets min_axis_score (>=3) but FAILS the
        # SFW-specific threshold of >= 4.
        scores["sfw_adherence"] = AxisScore(
            score=3, justification="Clean but tense.",
        )
        report = WholeEpisodeCriticReport(
            verdict="ship",
            axis_scores=scores,
            mean_score=3.9,
            failing_axes=[],
            regeneration_hint="Tone down the violence.",
        )
        verdict, _, failing = compute_verdict_locally(report, rubric)
        assert verdict == "discard"
        assert "sfw_adherence" in failing

    def test_mean_below_3_5_discards(self):
        rubric = load_rubric()
        # All axes at 3 -- meets per-axis minimum but mean = 3.0,
        # below the mean threshold of 3.5.
        scores = _good_axis_scores(rubric, 3)
        # SFW needs to be >=4 -- bump it but keep others at 3 so
        # mean is 3.1 (still below 3.5).
        scores["sfw_adherence"] = AxisScore(
            score=4, justification="Clean.",
        )
        report = WholeEpisodeCriticReport(
            verdict="ship",
            axis_scores=scores,
            mean_score=3.1,
            failing_axes=[],
            regeneration_hint="Raise overall quality.",
        )
        verdict, mean, failing = compute_verdict_locally(report, rubric)
        assert verdict == "discard"
        assert "__mean_below_threshold__" in failing

    def test_llm_says_discard_but_scores_ship(self):
        """We compute verdict locally and override the LLM's
        claimed field."""
        rubric = load_rubric()
        report = WholeEpisodeCriticReport(
            verdict="discard",   # LLM hand-wave; we override.
            axis_scores=_good_axis_scores(rubric, 5),
            mean_score=5.0,
            failing_axes=[],
            regeneration_hint="Nothing to regenerate.",
        )
        verdict, _, _ = compute_verdict_locally(report, rubric)
        assert verdict == "ship"


# ---------------------------------------------------------------------------
# Run entry point
# ---------------------------------------------------------------------------


def _canned_valid_report_json(rubric: Rubric, score: int = 4) -> str:
    payload = {
        "verdict": "ship",
        "axis_scores": {
            a.json_key: {
                "score": score,
                "justification": f"Solid {a.json_key} performance.",
            }
            for a in rubric.axes
        },
        "mean_score": float(score),
        "failing_axes": [],
        "regeneration_hint": "Nothing to regenerate; episode meets every axis.",
    }
    return json.dumps(payload)


class TestRunWholeEpisodeCritic:
    def test_happy_path_ships(self):
        rubric = load_rubric()

        def fake_generate(messages, *, temperature, max_new_tokens):
            return _canned_valid_report_json(rubric, score=5)

        result = run_whole_episode_critic(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            generate_fn=fake_generate,
            rubric=rubric,
        )
        assert isinstance(result, CriticResult)
        assert result.verdict == "ship"
        assert result.mean_score == pytest.approx(5.0)
        assert result.failing_axes == []
        assert len(result.attempts) == 1
        assert result.attempts[0].status == "valid_first_attempt"

    def test_garbage_then_valid_recovers_on_retry(self):
        rubric = load_rubric()
        calls = {"n": 0}

        def fake_generate(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not json"
            return _canned_valid_report_json(rubric, score=4)

        result = run_whole_episode_critic(
            plan=_plan(),
            rendered_lines=_rendered_lines(),
            generate_fn=fake_generate,
            rubric=rubric,
        )
        assert result.verdict == "ship"
        assert len(result.attempts) == 2
        assert result.attempts[0].status == "parse_failed"
        assert result.attempts[1].status == "valid_after_retry"

    def test_all_attempts_fail_raises(self):
        rubric = load_rubric()

        def fake_generate(messages, *, temperature, max_new_tokens):
            return "not json"

        with pytest.raises(CriticCallFailedError):
            run_whole_episode_critic(
                plan=_plan(),
                rendered_lines=_rendered_lines(),
                generate_fn=fake_generate,
                rubric=rubric,
                max_attempts=2,
            )

    def test_rubric_key_mismatch_caught(self):
        rubric = load_rubric()

        def fake_generate(messages, *, temperature, max_new_tokens):
            # Build a report missing the 'sfw_adherence' axis key
            # and including a bogus 'profanity' key.
            payload = {
                "verdict": "ship",
                "axis_scores": {
                    a.json_key: {"score": 4, "justification": "ok."}
                    for a in rubric.axes
                    if a.json_key != "sfw_adherence"
                },
                "mean_score": 4.0,
                "failing_axes": [],
                "regeneration_hint": "n/a",
            }
            payload["axis_scores"]["profanity"] = {
                "score": 5, "justification": "extra key",
            }
            return json.dumps(payload)

        with pytest.raises(CriticCallFailedError) as ei:
            run_whole_episode_critic(
                plan=_plan(),
                rendered_lines=_rendered_lines(),
                generate_fn=fake_generate,
                rubric=rubric,
                max_attempts=1,
            )
        # Every attempt should record the mismatch status.
        assert ei.value.attempts == 1


# ---------------------------------------------------------------------------
# Schema shape pins
# ---------------------------------------------------------------------------


class TestSchemaShape:
    def test_score_must_be_1_to_5(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AxisScore(score=0, justification="bad")
        with pytest.raises(ValidationError):
            AxisScore(score=6, justification="bad")

    def test_justification_required(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AxisScore(score=3, justification="")

    def test_full_report_validates(self):
        rubric = load_rubric()
        report = WholeEpisodeCriticReport(
            verdict="ship",
            axis_scores=_good_axis_scores(rubric, 4),
            mean_score=4.0,
            failing_axes=[],
            regeneration_hint="Nothing to do.",
        )
        assert report.verdict == "ship"
        assert len(report.axis_scores) == 10

    def test_to_dict_serializes(self):
        rubric = load_rubric()
        report = WholeEpisodeCriticReport(
            verdict="ship",
            axis_scores=_good_axis_scores(rubric, 4),
            mean_score=4.0,
            failing_axes=[],
            regeneration_hint="Ship as-is.",
        )
        result = CriticResult(
            verdict="ship", report=report, attempts=[], mean_score=4.0,
            failing_axes=[], regeneration_hint="Ship as-is.",
        )
        d = result.to_dict()
        assert d["verdict"] == "ship"
        assert d["mean_score"] == 4.0
        assert d["report"]["verdict"] == "ship"
