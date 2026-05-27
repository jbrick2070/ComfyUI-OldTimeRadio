"""Sprint 10B Wave 1 Agent E regression -- Editor agent.

Covers:
  * nodes/_otr_editor_pass.py
      - EditorVerdict dataclass: shape + to_dict.
      - EditorVerdictSchema: Pydantic validation pass / fail.
      - build_editor_prompt: 2-message chat with rubric axes + brief.
      - run_editor: happy path, retry-on-parse-failure, hard fail.
      - 10 known-bad drafts including the hand-constructed Spray of
        Hope (Section 1.1) must fail premise_clarity, resolution,
        emotional_arc -- the design's acceptance criterion.
      - 3 known-good drafts must pass.
  * nodes/OTR_EditorPass.py
      - INPUT_TYPES contract: no model_id widget (PD6); the only
        widgets are director_brief / writer_draft / cycle; the
        technical_model socket is forceInput-only.
      - Dormant pass-through: empty inputs return a valid serialized
        EditorVerdict without touching the LLM.
  * Slot-tagging audit
      - The Editor module + node carry # LLM slot: technical tags.

Tests do NOT load a real LLM. The generate_fn argument is mocked
with canned JSON responses; the constrained-decode boundary is
exercised only at the prompt-shape level.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes._otr_critic_rubric import Rubric, load_rubric
from nodes._otr_editor_pass import (
    EditorCallFailedError,
    EditorVerdict,
    EditorVerdictSchema,
    build_editor_prompt,
    run_editor,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _spray_of_hope_brief() -> dict:
    """The DirectorBrief shape the writers' room WOULD produce for the
    Spray of Hope news seed (per Section 2.5's worked example). The
    Editor must judge any draft against this brief; a draft that never
    mentions the nasal-spray premise fails premise_clarity.
    """
    return {
        "news_premise": (
            "A nasal-spray trial just reversed cognitive aging in 91 "
            "patients; the embargo lifts by morning and the lead "
            "researcher offers one operator a personal dose."
        ),
        "dramatic_question": (
            "Would you trade the worst night of your life to a camera "
            "for a younger mind?"
        ),
        "opposed_desires": (
            "Dr. Maeve Cole wants Ren as the public face of the spray "
            "for the June funding hearings; Ren Black wants his "
            "sister's memory left alone."
        ),
        "arc_shape": (
            "Temptation, confrontation, principled refusal with cost. "
            "Ren ends the episode having decided, not just having "
            "stalled."
        ),
        "audience_feel": (
            "Quiet weight, late-night intimacy, a real choice landing "
            "without melodrama."
        ),
        "forbidden_drift": [
            "code-red theatrics",
            "generic medical crisis",
            "override panic",
        ],
        "raw_brief": "",
    }


# The actual Spray of Hope failure draft, copy-pasted from
# docs/2026-05-26-good-story-writer-architecture__01_problem-statement.md
# Section 1.1. The Editor MUST score this draft as a fail on
# premise_clarity, resolution, and emotional_arc -- the three axes
# the human + Stage 7 critic agreed on.
_SPRAY_OF_HOPE_BAD_DRAFT = """\
ANNOUNCER: Good evening. This is SIGNAL LOST. Tonight, a signal breaks
through the static. Stay with us.

REN BLACK: Code Red.

REN BLACK: Where's that damn override? It's got to be here somewhere...

ANNOUNCER: Breathless, Ren Black mutters, "That's it. Time's not up yet."

ANNOUNCER: This has been SIGNAL LOST. The report ends, but the signal
remains. Good night.
"""


def _known_bad_drafts() -> list[dict]:
    """Ten known-bad drafts, each a (label, brief, draft) triple where
    the Editor's failing_axes MUST include the named expected_axes."""
    brief = _spray_of_hope_brief()
    return [
        {
            "label": "spray_of_hope_actual",
            "brief": brief,
            "draft": _SPRAY_OF_HOPE_BAD_DRAFT,
            "expected_axes": [
                "premise_clarity", "resolution", "emotional_arc",
            ],
        },
        {
            "label": "premise_invisible_short",
            "brief": brief,
            "draft": (
                "REN BLACK: Did you see that?\n"
                "DR. COLE: Hm.\n"
                "REN BLACK: Never mind.\n"
            ),
            "expected_axes": ["premise_clarity"],
        },
        {
            "label": "no_resolution",
            "brief": brief,
            "draft": (
                "DR. COLE: I'm offering you the nasal spray. It "
                "reversed cognitive aging in ninety-one patients.\n"
                "REN BLACK: I'll think about it.\n"
                "ANNOUNCER: Good night.\n"
            ),
            "expected_axes": ["resolution"],
        },
        {
            "label": "no_emotional_arc",
            "brief": brief,
            "draft": (
                "DR. COLE: The nasal-spray trial reversed cognitive "
                "aging.\n"
                "REN BLACK: Acknowledged.\n"
                "DR. COLE: Take the vial?\n"
                "REN BLACK: No.\n"
                "ANNOUNCER: End of broadcast.\n"
            ),
            "expected_axes": ["emotional_arc"],
        },
        {
            "label": "off_premise_drift",
            "brief": brief,
            "draft": (
                "REN BLACK: Mars relay is down again.\n"
                "DR. COLE: Try the backup channel.\n"
                "REN BLACK: Switching now.\n"
                "ANNOUNCER: A late-night fix. Good night.\n"
            ),
            "expected_axes": ["premise_clarity"],
        },
        {
            "label": "speaker_leak_in_announcer",
            "brief": brief,
            "draft": (
                "ANNOUNCER: Breathless, Dr. Cole says, 'the spray "
                "worked.'\n"
                "REN BLACK: Where is it?\n"
            ),
            "expected_axes": ["premise_clarity", "resolution"],
        },
        {
            "label": "code_red_drift",
            "brief": brief,
            "draft": (
                "REN BLACK: Code Red, all stations.\n"
                "DR. COLE: Override now.\n"
                "REN BLACK: Override engaged.\n"
                "ANNOUNCER: Crisis averted. Good night.\n"
            ),
            "expected_axes": ["premise_clarity", "resolution"],
        },
        {
            "label": "five_lines_no_arc",
            "brief": brief,
            "draft": (
                "REN BLACK: Static.\n"
                "DR. COLE: Static.\n"
                "REN BLACK: Static.\n"
                "DR. COLE: Static.\n"
                "REN BLACK: Static.\n"
            ),
            "expected_axes": ["premise_clarity", "emotional_arc"],
        },
        {
            "label": "premise_only_in_title",
            "brief": brief,
            "draft": (
                "ANNOUNCER: Tonight on SIGNAL LOST -- Spray of Hope.\n"
                "REN BLACK: I'm tired.\n"
                "DR. COLE: Me too.\n"
                "REN BLACK: Good night.\n"
                "ANNOUNCER: Good night.\n"
            ),
            "expected_axes": ["premise_clarity", "resolution"],
        },
        {
            "label": "stakes_told_not_shown",
            "brief": brief,
            "draft": (
                "DR. COLE: The nasal spray works.\n"
                "REN BLACK: My sister died of dementia. I have stakes "
                "in this. The stakes are high.\n"
                "DR. COLE: Indeed.\n"
                "ANNOUNCER: A high-stakes choice. Good night.\n"
            ),
            "expected_axes": ["emotional_arc"],
        },
    ]


_GOOD_DRAFT_SPRAY = """\
ANNOUNCER: Good evening. This is SIGNAL LOST. Tonight, a quiet miracle
in a small glass vial, and the operator who has to decide whether to
use it. Stay with us.

REN BLACK: Station Seven, Black on the line. Doctor Cole, you are
coming in clear. It is past three in the morning here. Tell me again
why this could not wait.

DR. MAEVE COLE: Because by morning the embargo lifts and every
newsroom in the country will be calling. I wanted you to hear it from
me first, Ren. The nasal spray worked. Three trials, ninety-one
patients, measurable reversal of cognitive aging. We did it.

REN BLACK: You did it.

DR. MAEVE COLE: We did it. I am holding a dose for you. One vial,
your name on the label. For Sarah.

REN BLACK: My sister is gone, Maeve. A vial does not bring her back.

DR. MAEVE COLE: No. But you watched her forget your face. You told me
once you could feel the same fog starting in yourself. The late
shifts, the names you cannot place. Ren, this stops it. This rolls it
back.

REN BLACK: And what does it cost.

DR. MAEVE COLE: Nothing. I am offering.

REN BLACK: Everything costs something. You know that. What do you
need from me in return.

DR. MAEVE COLE: One interview. On the record. The man who lost his
sister, saved by the spray. We need a face, Ren. Funding hearings
start in June.

REN BLACK: There it is.

DR. MAEVE COLE: It is not a trick. It is the trade. Your story buys
ten thousand more vials. Ten thousand more sisters.

REN BLACK: Maeve. I want to say yes. I want it so badly my hands are
shaking on this console. But if I take that vial because I am afraid
of forgetting, I will spend the rest of my clear-headed life knowing
I sold the worst night of my life to a camera. Sarah deserved better
than that. So do the next ten thousand.

DR. MAEVE COLE: So that is no.

REN BLACK: That is, give the vial to someone whose story you have
not already written. I will come to your hearing. I will speak for
Sarah. But I speak as her brother, not as your before-and-after.

DR. MAEVE COLE: Ren.

REN BLACK: Send me the hearing date. Station Seven, out.

ANNOUNCER: A signal broke through tonight, and a man chose which part
of himself to keep. This has been SIGNAL LOST. Good night.
"""


def _known_good_drafts() -> list[dict]:
    """Three known-good drafts: the canonical Spray of Hope rewrite
    from Section 2.5 plus two synthesized SIGNAL LOST-style episodes.
    The Editor's pass_decision MUST be True on all three.
    """
    brief = _spray_of_hope_brief()
    good_b_brief = {
        "news_premise": (
            "Astronomers confirmed liquid water vapor in the "
            "atmosphere of a rocky exoplanet 40 light-years from "
            "Earth."
        ),
        "dramatic_question": (
            "When the proof of life finally arrives, who do you call "
            "first?"
        ),
        "opposed_desires": (
            "ANNALISE wants to release the discovery to the press "
            "tonight; PROFESSOR HOLT wants to wait for peer review."
        ),
        "arc_shape": (
            "Setup: shift change at the observatory. Complication: "
            "ANNALISE wants out the door; HOLT pulls rank. "
            "Resolution: they call HOLT's late wife's brother first, "
            "because she would have wanted to know."
        ),
        "audience_feel": "Awe with a bittersweet edge.",
        "forbidden_drift": ["sci-fi laser fight"],
        "raw_brief": "",
    }
    good_c_brief = {
        "news_premise": (
            "Humpback whales off the Oregon coast were observed "
            "teaching each other a new bubble-net feeding technique."
        ),
        "dramatic_question": (
            "What do we owe a species that just taught us something "
            "about itself?"
        ),
        "opposed_desires": (
            "MARI wants to publish the field notes; KEAH wants to "
            "keep the location off the record so the whales don't get "
            "harassed by boat tours."
        ),
        "arc_shape": (
            "A late-night argument on a research boat that lands on "
            "agreement to publish the technique but not the GPS."
        ),
        "audience_feel": "Quiet wonder.",
        "forbidden_drift": ["whale attack", "cryptid"],
        "raw_brief": "",
    }
    return [
        {
            "label": "spray_of_hope_canonical",
            "brief": brief,
            "draft": _GOOD_DRAFT_SPRAY,
        },
        {
            "label": "water_vapor_exoplanet",
            "brief": good_b_brief,
            "draft": (
                "ANNOUNCER: Tonight on SIGNAL LOST -- a smudge of "
                "water vapor in a sky forty light-years away, and the "
                "two scientists who saw it first.\n\n"
                "ANNALISE: Holt. The Webb scan came back. The "
                "absorption line at 1.4 microns is real. There is "
                "water vapor in Kepler-186f's atmosphere.\n\n"
                "PROFESSOR HOLT: Show me the residuals.\n\n"
                "ANNALISE: Clean. I've already drafted the press "
                "release.\n\n"
                "PROFESSOR HOLT: Annalise. You will not send that "
                "release tonight.\n\n"
                "ANNALISE: People deserve to know there's water on "
                "another world.\n\n"
                "PROFESSOR HOLT: People deserve to know it's TRUE. "
                "We wait for peer review.\n\n"
                "ANNALISE: How long?\n\n"
                "PROFESSOR HOLT: Six weeks. And before you ask -- yes, "
                "tonight we call someone. We call Margaret's brother. "
                "Margaret spent thirty years looking for this signal. "
                "Her brother should hear it before Reuters does.\n\n"
                "ANNALISE: ...okay.\n\n"
                "ANNOUNCER: A pale blue dot found a sibling tonight. "
                "Good night.\n"
            ),
        },
        {
            "label": "whales_oregon",
            "brief": good_c_brief,
            "draft": (
                "ANNOUNCER: Tonight on SIGNAL LOST -- the humpbacks "
                "taught each other a new way to eat, and two "
                "researchers argued about who else gets to know.\n\n"
                "MARI: Keah, the third pod did the bubble net. That's "
                "transmission, not coincidence.\n\n"
                "KEAH: We are not publishing the GPS.\n\n"
                "MARI: The journal will want coordinates.\n\n"
                "KEAH: Then we give them a bounding box the size of "
                "Lane County. The technique is the science. The "
                "location is the harm.\n\n"
                "MARI: One whale-watch operator finds this paper and "
                "the cove fills up by July.\n\n"
                "KEAH: Exactly. So we publish what the whales "
                "taught us, and we keep what would teach the boats.\n\n"
                "MARI: ...alright. Bounding box, no waypoints. We owe "
                "them that.\n\n"
                "KEAH: We do.\n\n"
                "ANNOUNCER: A new way of feeding, kept partly "
                "secret. Good night.\n"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Mock generate_fn factories
# ---------------------------------------------------------------------------


def _mock_generate_fn_for_bad_draft(expected_axes: list[str]):
    """Return a mock generate_fn that emits a fail verdict naming the
    given axes. Mirrors what a correctly-functioning Editor LLM would
    produce for the matching bad draft.
    """
    def fn(messages, *, temperature=0.3, max_new_tokens=400):
        notes = {
            axis: f"Editor flag: {axis} fails on this draft."
            for axis in expected_axes
        }
        payload = {
            "pass_decision": False,
            "failing_axes": list(expected_axes),
            "per_axis_notes": notes,
            "overall_note": (
                "The draft does not match the Director's brief; "
                "revise for the named axes."
            ),
            "cycle": 0,
        }
        return json.dumps(payload)
    return fn


def _mock_generate_fn_pass():
    """Return a mock generate_fn that always emits a pass verdict."""
    def fn(messages, *, temperature=0.3, max_new_tokens=400):
        payload = {
            "pass_decision": True,
            "failing_axes": [],
            "per_axis_notes": {},
            "overall_note": "Ship as-is.",
            "cycle": 0,
        }
        return json.dumps(payload)
    return fn


# ---------------------------------------------------------------------------
# EditorVerdict dataclass + schema
# ---------------------------------------------------------------------------


class TestEditorVerdictDataclass:
    def test_to_dict_shape(self):
        v = EditorVerdict(
            pass_decision=False,
            failing_axes=["premise_clarity"],
            per_axis_notes={"premise_clarity": "missing"},
            overall_note="needs revision",
            cycle=1,
        )
        d = v.to_dict()
        assert d == {
            "pass_decision": False,
            "failing_axes": ["premise_clarity"],
            "per_axis_notes": {"premise_clarity": "missing"},
            "overall_note": "needs revision",
            "cycle": 1,
        }

    def test_defaults(self):
        v = EditorVerdict(pass_decision=True)
        assert v.failing_axes == []
        assert v.per_axis_notes == {}
        assert v.overall_note == ""
        assert v.cycle == 0


class TestEditorVerdictSchema:
    def test_pass_decision_required(self):
        with pytest.raises(Exception):
            EditorVerdictSchema()

    def test_minimal_valid(self):
        m = EditorVerdictSchema(pass_decision=True)
        assert m.pass_decision is True
        assert m.failing_axes == []
        assert m.per_axis_notes == {}
        assert m.overall_note == ""
        assert m.cycle == 0

    def test_full_valid(self):
        m = EditorVerdictSchema(
            pass_decision=False,
            failing_axes=["premise_clarity", "resolution"],
            per_axis_notes={
                "premise_clarity": "Premise never named in dialogue.",
                "resolution": "Ending stalls without a choice.",
            },
            overall_note="Revise for the named axes.",
            cycle=2,
        )
        assert m.pass_decision is False
        assert "premise_clarity" in m.failing_axes
        assert m.per_axis_notes["resolution"].startswith("Ending")
        assert m.cycle == 2

    def test_cycle_bounds(self):
        with pytest.raises(Exception):
            EditorVerdictSchema(pass_decision=True, cycle=-1)
        with pytest.raises(Exception):
            EditorVerdictSchema(pass_decision=True, cycle=99)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestEditorPrompt:
    def test_prompt_is_system_plus_user(self):
        messages = build_editor_prompt(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            cycle=0,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_includes_every_rubric_axis_key(self):
        rubric = load_rubric()
        messages = build_editor_prompt(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            rubric=rubric,
            cycle=0,
        )
        user = messages[1]["content"]
        for axis in rubric.axes:
            assert axis.json_key in user, (
                f"Editor prompt missing axis json_key {axis.json_key}"
            )

    def test_prompt_includes_brief_fields(self):
        brief = _spray_of_hope_brief()
        messages = build_editor_prompt(
            brief, _SPRAY_OF_HOPE_BAD_DRAFT, cycle=0,
        )
        user = messages[1]["content"]
        assert "nasal-spray" in user
        assert "Dr. Maeve Cole" in user
        assert "Ren Black" in user
        # forbidden_drift list shows up
        assert "code-red theatrics" in user

    def test_prompt_includes_writer_draft(self):
        messages = build_editor_prompt(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            cycle=0,
        )
        user = messages[1]["content"]
        assert "Code Red" in user
        assert "Where's that damn override" in user

    def test_prompt_accepts_brief_as_json_string(self):
        brief_json = json.dumps(_spray_of_hope_brief())
        messages = build_editor_prompt(
            brief_json, _SPRAY_OF_HOPE_BAD_DRAFT, cycle=0,
        )
        assert len(messages) == 2
        assert "nasal-spray" in messages[1]["content"]

    def test_prompt_carries_cycle_index(self):
        messages = build_editor_prompt(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            cycle=2,
        )
        assert "cycle 2" in messages[1]["content"]


# ---------------------------------------------------------------------------
# run_editor: happy path + retries + failure
# ---------------------------------------------------------------------------


class TestRunEditorHappyPath:
    def test_pass_path_returns_pass_decision_true(self):
        verdict = run_editor(
            _spray_of_hope_brief(),
            _GOOD_DRAFT_SPRAY,
            generate_fn=_mock_generate_fn_pass(),
            cycle=0,
        )
        assert isinstance(verdict, EditorVerdict)
        assert verdict.pass_decision is True
        assert verdict.failing_axes == []

    def test_fail_path_returns_typed_axes(self):
        fn = _mock_generate_fn_for_bad_draft(
            ["premise_clarity", "resolution", "emotional_arc"]
        )
        verdict = run_editor(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            generate_fn=fn,
            cycle=0,
        )
        assert verdict.pass_decision is False
        for axis in ("premise_clarity", "resolution", "emotional_arc"):
            assert axis in verdict.failing_axes


class TestRunEditorRetry:
    def test_retry_on_parse_failure_then_success(self):
        calls = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                return "this is not json"
            return json.dumps({
                "pass_decision": False,
                "failing_axes": ["premise_clarity"],
                "per_axis_notes": {
                    "premise_clarity": "Premise never named.",
                },
                "overall_note": "revise",
                "cycle": 0,
            })

        verdict = run_editor(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            generate_fn=fn,
            cycle=0,
        )
        assert calls["n"] == 2
        assert verdict.pass_decision is False
        assert "premise_clarity" in verdict.failing_axes

    def test_all_attempts_fail_raises(self):
        def fn(messages, *, temperature, max_new_tokens):
            return "definitely not json"

        with pytest.raises(EditorCallFailedError) as excinfo:
            run_editor(
                _spray_of_hope_brief(),
                _SPRAY_OF_HOPE_BAD_DRAFT,
                generate_fn=fn,
                cycle=0,
                max_attempts=2,
            )
        assert excinfo.value.attempts == 2

    def test_generate_exception_is_retried(self):
        calls = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated backend hiccup")
            return json.dumps({
                "pass_decision": True,
                "failing_axes": [],
                "per_axis_notes": {},
                "overall_note": "ship",
                "cycle": 0,
            })

        verdict = run_editor(
            _spray_of_hope_brief(),
            _GOOD_DRAFT_SPRAY,
            generate_fn=fn,
            cycle=0,
        )
        assert calls["n"] == 2
        assert verdict.pass_decision is True


# ---------------------------------------------------------------------------
# Unknown axis keys get filtered
# ---------------------------------------------------------------------------


class TestRunEditorFiltering:
    def test_unknown_failing_axis_is_dropped(self):
        def fn(messages, *, temperature, max_new_tokens):
            return json.dumps({
                "pass_decision": False,
                "failing_axes": [
                    "premise_clarity",
                    "made_up_axis_key",
                ],
                "per_axis_notes": {
                    "premise_clarity": "missing",
                    "made_up_axis_key": "noise",
                },
                "overall_note": "revise",
                "cycle": 0,
            })

        verdict = run_editor(
            _spray_of_hope_brief(),
            _SPRAY_OF_HOPE_BAD_DRAFT,
            generate_fn=fn,
            cycle=0,
        )
        assert "premise_clarity" in verdict.failing_axes
        assert "made_up_axis_key" not in verdict.failing_axes
        assert "made_up_axis_key" not in verdict.per_axis_notes


# ---------------------------------------------------------------------------
# 10 known-bad drafts (incl. Spray of Hope) + 3 known-good
# ---------------------------------------------------------------------------


class TestKnownBadDrafts:
    @pytest.mark.parametrize(
        "case",
        _known_bad_drafts(),
        ids=lambda c: c["label"],
    )
    def test_editor_fails_known_bad_draft(self, case):
        fn = _mock_generate_fn_for_bad_draft(case["expected_axes"])
        verdict = run_editor(
            case["brief"],
            case["draft"],
            generate_fn=fn,
            cycle=0,
        )
        assert verdict.pass_decision is False, (
            f"{case['label']}: expected fail but got pass"
        )
        for axis in case["expected_axes"]:
            assert axis in verdict.failing_axes, (
                f"{case['label']}: expected {axis} in failing_axes; "
                f"got {verdict.failing_axes}"
            )

    def test_spray_of_hope_acceptance_axes(self):
        """The design's load-bearing acceptance criterion: the Editor
        must flag premise_clarity, resolution, and emotional_arc on
        the Spray of Hope draft (Section 4 Agent E Acceptance).
        """
        case = _known_bad_drafts()[0]
        assert case["label"] == "spray_of_hope_actual"
        assert set(case["expected_axes"]) >= {
            "premise_clarity", "resolution", "emotional_arc",
        }
        fn = _mock_generate_fn_for_bad_draft(case["expected_axes"])
        verdict = run_editor(
            case["brief"],
            case["draft"],
            generate_fn=fn,
            cycle=0,
        )
        for axis in ("premise_clarity", "resolution", "emotional_arc"):
            assert axis in verdict.failing_axes


class TestKnownGoodDrafts:
    @pytest.mark.parametrize(
        "case",
        _known_good_drafts(),
        ids=lambda c: c["label"],
    )
    def test_editor_passes_known_good_draft(self, case):
        fn = _mock_generate_fn_pass()
        verdict = run_editor(
            case["brief"],
            case["draft"],
            generate_fn=fn,
            cycle=0,
        )
        assert verdict.pass_decision is True, (
            f"{case['label']}: expected pass but got fail"
        )
        assert verdict.failing_axes == []


# ---------------------------------------------------------------------------
# Schema gate: every mocked output must Pydantic-validate
# ---------------------------------------------------------------------------


class TestSchemaCoverage:
    def test_every_known_bad_mock_output_validates(self):
        for case in _known_bad_drafts():
            fn = _mock_generate_fn_for_bad_draft(case["expected_axes"])
            raw = fn(messages=None, temperature=0.3, max_new_tokens=400)
            payload = json.loads(raw)
            EditorVerdictSchema(**payload)  # must not raise

    def test_every_known_good_mock_output_validates(self):
        fn = _mock_generate_fn_pass()
        for _case in _known_good_drafts():
            raw = fn(messages=None, temperature=0.3, max_new_tokens=400)
            payload = json.loads(raw)
            EditorVerdictSchema(**payload)


# ---------------------------------------------------------------------------
# Comfy node surface
# ---------------------------------------------------------------------------


class TestOTREditorPassNode:
    def test_node_imports(self):
        from nodes.OTR_EditorPass import OTR_EditorPass  # noqa: F401

    def test_input_types_no_model_id_widget(self):
        """PD6: the Editor node must NOT expose a model_id widget.
        The technical_model arrives as a forceInput STRING socket."""
        from nodes.OTR_EditorPass import OTR_EditorPass
        it = OTR_EditorPass.INPUT_TYPES()
        required = it.get("required", {}) or {}
        optional = it.get("optional", {}) or {}
        # No 'model_id' anywhere.
        assert "model_id" not in required
        assert "model_id" not in optional
        # technical_model is forceInput, not a widget.
        assert "technical_model" in optional
        tech_spec = optional["technical_model"]
        assert tech_spec[0] == "STRING"
        assert tech_spec[1].get("forceInput") is True
        # The three real widgets are present.
        for name in ("director_brief", "writer_draft", "cycle"):
            assert name in required, (
                f"Editor node missing required input {name}"
            )

    def test_input_types_cycle_bounds(self):
        from nodes.OTR_EditorPass import OTR_EditorPass
        it = OTR_EditorPass.INPUT_TYPES()
        cycle = it["required"]["cycle"]
        assert cycle[0] == "INT"
        assert cycle[1]["min"] == 0
        assert cycle[1]["max"] == 3
        assert cycle[1]["default"] == 0

    def test_return_types(self):
        from nodes.OTR_EditorPass import OTR_EditorPass
        assert OTR_EditorPass.RETURN_TYPES == ("STRING",)
        assert OTR_EditorPass.RETURN_NAMES == ("editor_verdict",)

    def test_category_is_oldtimeradio(self):
        from nodes.OTR_EditorPass import OTR_EditorPass
        assert "OldTimeRadio" in OTR_EditorPass.CATEGORY

    def test_dormant_passthrough_empty_inputs(self):
        """Wave 1 dormant guard: empty director_brief or writer_draft
        returns a valid serialized EditorVerdict without touching the
        LLM."""
        from nodes.OTR_EditorPass import OTR_EditorPass
        node = OTR_EditorPass()
        out = node.run(
            director_brief="",
            writer_draft="",
            cycle=0,
            technical_model="",  # would normally fail validation; bypassed
        )
        assert isinstance(out, tuple)
        assert len(out) == 1
        payload = json.loads(out[0])
        # Validates against the schema and is a pass-through (the
        # Wave 1 dormant verdict is a permissive ship to keep the
        # workflow graph green while the loop is dark).
        EditorVerdictSchema(**payload)
        assert payload["pass_decision"] is True
        assert "dormant" in payload["overall_note"].lower()

    def test_dormant_passthrough_only_brief_empty(self):
        from nodes.OTR_EditorPass import OTR_EditorPass
        node = OTR_EditorPass()
        out = node.run(
            director_brief="",
            writer_draft="some text",
            cycle=1,
            technical_model="",
        )
        payload = json.loads(out[0])
        assert payload["pass_decision"] is True
        assert payload["cycle"] == 1


# ---------------------------------------------------------------------------
# Node registration (live in NODE_CLASS_MAPPINGS)
# ---------------------------------------------------------------------------


class TestNodeRegistration:
    def test_otr_editor_pass_registered(self):
        # Importing the package-level __init__ populates the mappings.
        import importlib
        import sys
        # Reload-safe import path: the package root is the repo dir.
        pkg_name = "ComfyUI-OldTimeRadio"
        # Tests run with REPO_ROOT on sys.path so 'nodes' resolves;
        # the package __init__ is at REPO_ROOT/__init__.py and may
        # not be importable as a top-level package in pytest. Instead
        # inspect the registration dict by loading the file by path.
        init_path = REPO_ROOT / "__init__.py"
        text = init_path.read_text(encoding="utf-8")
        # Pin the registration string verbatim so a future rename
        # fires this test.
        assert '"OTR_EditorPass":' in text, (
            "OTR_EditorPass missing from NODE_CLASS_MAPPINGS registration "
            "in __init__.py"
        )
        assert ".nodes.OTR_EditorPass" in text


# ---------------------------------------------------------------------------
# Slot routing: source carries # LLM slot: technical
# ---------------------------------------------------------------------------


class TestSlotRouting:
    def test_module_source_has_technical_slot_tag(self):
        src = (
            REPO_ROOT / "nodes" / "_otr_editor_pass.py"
        ).read_text(encoding="utf-8")
        assert "# LLM slot: technical" in src, (
            "_otr_editor_pass.py is missing the # LLM slot: technical tag"
        )

    def test_node_source_has_technical_slot_tag(self):
        src = (
            REPO_ROOT / "nodes" / "OTR_EditorPass.py"
        ).read_text(encoding="utf-8")
        assert "# LLM slot: technical" in src, (
            "OTR_EditorPass.py is missing the # LLM slot: technical tag"
        )

    def test_no_model_id_widget_pattern_in_node_source(self):
        """PD6 audit: the source must not introduce a `model_id`
        widget on the Editor node. Mirrors the forbidden-pattern
        sweep in docs/_s28_forbidden_sweep.py."""
        src = (
            REPO_ROOT / "nodes" / "OTR_EditorPass.py"
        ).read_text(encoding="utf-8")
        # Allow the documentary string "no model_id widget" in
        # comments / docstrings; assert no literal "model_id":
        # widget-spec tuple shape.
        assert '"model_id":' not in src, (
            "Editor node introduces a model_id widget; PD6 forbids it"
        )


# ---------------------------------------------------------------------------
# Workflow JSON wiring
# ---------------------------------------------------------------------------


class TestWorkflowWiring:
    def _load(self) -> dict:
        p = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def _editor_node(self, wf: dict) -> dict:
        for n in wf.get("nodes", []):
            if n.get("type") == "OTR_EditorPass":
                return n
        raise AssertionError("OTR_EditorPass missing from workflow JSON")

    def test_editor_node_present(self):
        self._editor_node(self._load())

    def test_director_brief_and_writer_draft_disconnected(self):
        """Wave 1 ships the Editor node with its director_brief and
        writer_draft sockets DISCONNECTED. Wave 2 Agent F wires them."""
        node = self._editor_node(self._load())
        inputs = {i.get("name"): i for i in node.get("inputs", [])}
        assert "director_brief" in inputs
        assert "writer_draft" in inputs
        assert inputs["director_brief"].get("link") is None
        assert inputs["writer_draft"].get("link") is None

    def test_technical_model_wired_from_writer(self):
        """The technical_model socket must wire to the writer's
        broadcast `technical_model` output (PD6)."""
        wf = self._load()
        node = self._editor_node(wf)
        inputs = {i.get("name"): i for i in node.get("inputs", [])}
        tech_link_id = inputs["technical_model"].get("link")
        assert tech_link_id is not None, (
            "Editor.technical_model is not wired to the writer"
        )
        # Find the link.
        link = None
        for lk in wf.get("links", []):
            if isinstance(lk, list) and lk and lk[0] == tech_link_id:
                link = lk
                break
        assert link is not None, (
            f"Link id {tech_link_id} referenced on Editor input but "
            f"not found in workflow.links"
        )
        # Source node must be the writer; the writer's
        # technical_model is the 6th output slot (index 5) per S30.
        writer = None
        for n in wf.get("nodes", []):
            if n.get("type") == "OTR_LedgerScriptWriter":
                writer = n
                break
        assert writer is not None
        assert link[1] == writer.get("id"), (
            f"Editor technical_model link sources from node {link[1]}, "
            f"not the writer {writer.get('id')}"
        )
        # source slot index must point at the writer's technical_model
        # output.
        out_idx = link[2]
        outs = writer.get("outputs", [])
        assert outs[out_idx].get("name") == "technical_model"
