"""tests/test_otr_director_brief.py -- Sprint 10B Wave 1 Agent D.

Covers the Director module + the OTR_DirectorBrief node per design
Section 4 Wave 1 Agent D's "Tests" block:

  * Unit: 10 different news seeds; every output passes DirectorBrief
    Pydantic validation.
  * Content audit: every brief contains identifiable references to the
    news premise and to rubric axis themes (keyword-overlap heuristic).
  * Integration: the Comfy node loads (import-check),
    INPUT_TYPES() returns the expected shape, RETURN_TYPES is
    ('STRING',).
  * Slot routing: assert the source file carries the
    '# LLM slot: technical' tag at the call site.

The module is DORMANT in Wave 1; these tests exercise the module's
pure surfaces with a mock generate_fn so no real LLM is loaded.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest
from pydantic import ValidationError

from nodes._otr_director_brief import (
    DirectorBrief,
    DirectorBriefSchema,
    DirectorCallFailedError,
    build_director_prompt,
    run_director,
)


# ---------------------------------------------------------------------------
# Test news-seed fixtures
# ---------------------------------------------------------------------------

# Ten diverse news seeds spanning science / medicine / biology /
# disaster / politics-neutral genres. The seeds are invented for the
# test suite (the production demo seeds live in
# docs/2026-05-26-10b-demo/seeds.txt and are operator-frozen on
# run-day per design Section 6.2).
_TEN_NEWS_SEEDS: list[dict[str, str]] = [
    {
        "slug": "exoplanet_water",
        "seed": (
            "Astronomers report the first confirmed detection of water "
            "vapor in the atmosphere of a rocky exoplanet 40 light-years "
            "away."
        ),
        "premise_keywords": ["water", "exoplanet", "atmosphere"],
    },
    {
        "slug": "pancreatic_blood_test",
        "seed": (
            "A new blood test detects pancreatic cancer up to three "
            "years before symptoms appear in a 4,000-patient trial."
        ),
        "premise_keywords": ["blood", "pancreatic", "cancer"],
    },
    {
        "slug": "humpback_bubble_net",
        "seed": (
            "Researchers find that humpback whales off the Oregon coast "
            "are teaching each other a new bubble-net feeding technique "
            "never observed before."
        ),
        "premise_keywords": ["humpback", "bubble-net", "feeding"],
    },
    {
        "slug": "nasal_brain_aging",
        "seed": (
            "Scientists say they have reversed brain aging with a "
            "simple nasal spray in a 91-patient trial."
        ),
        "premise_keywords": ["nasal", "brain", "aging"],
    },
    {
        "slug": "artemis_rocket",
        "seed": (
            "NASA's Artemis program completes a full-duration static "
            "fire of the SLS rocket second stage, clearing the path to "
            "a crewed lunar test flight."
        ),
        "premise_keywords": ["artemis", "rocket", "lunar"],
    },
    {
        "slug": "antarctic_ice_shelf",
        "seed": (
            "An ice shelf the size of Manhattan calved from the West "
            "Antarctic peninsula overnight; researchers say satellite "
            "imagery caught the break-up in real time."
        ),
        "premise_keywords": ["ice", "antarctic", "shelf"],
    },
    {
        "slug": "fungal_carbon_capture",
        "seed": (
            "A common soil fungus has been shown to pull carbon dioxide "
            "out of the air at rates that, scaled up, could offset a "
            "small nation's annual emissions."
        ),
        "premise_keywords": ["fungus", "carbon", "soil"],
    },
    {
        "slug": "deep_sea_cable",
        "seed": (
            "A team of engineers laid the longest single-segment "
            "deep-sea data cable on record, linking research stations "
            "on opposite sides of the Pacific."
        ),
        "premise_keywords": ["cable", "deep-sea", "pacific"],
    },
    {
        "slug": "honeybee_recovery",
        "seed": (
            "A multi-year apiary census reports the first sustained "
            "recovery in North American honeybee populations since the "
            "colony-collapse era began."
        ),
        "premise_keywords": ["honeybee", "colony", "apiary"],
    },
    {
        "slug": "geothermal_breakthrough",
        "seed": (
            "Engineers in Iceland demonstrated drilling into a layer of "
            "supercritical geothermal water, producing ten times the "
            "power per well of conventional geothermal."
        ),
        "premise_keywords": ["geothermal", "iceland", "drilling"],
    },
]


_TEN_CAST_PAIRS: list[list[str]] = [
    ["DR. CAS HOLLAND", "REN OKAFOR"],
    ["NURSE MAYA LIU", "TOM WALSH"],
    ["DR. MIRA SOL", "CAPTAIN AUSTEN"],
    ["REN BLACK", "DR. MAEVE COLE"],
    ["CMDR. ANNA REYES", "DR. PAUL KIM"],
    ["FIELD LEAD JO MENDEZ", "DR. ELI ROSEN"],
    ["DR. SAM PETERS", "MAGGIE FOLEY"],
    ["CHIEF ENGINEER LINA", "DR. CARLOS DELGADO"],
    ["BEEKEEPER GRACE MOREL", "DR. WEI ZHANG"],
    ["FOREMAN BJORN", "DR. SIGRID HALLDORS"],
]


# ---------------------------------------------------------------------------
# Mock generate_fn -- returns a schema-shaped JSON string per call
# ---------------------------------------------------------------------------


def _make_mock_generate_fn(
    seed_slug: str,
    cast: list[str],
    premise_keywords: list[str],
) -> "callable":
    """Build a mock generate_fn that returns a schema-shaped brief.

    The mock is deterministic per (slug, cast): it bakes the cast names
    into opposed_desires and the premise keywords into news_premise so
    the content-audit tests can assert keyword overlap. Real production
    constrained-decode would do the same naturally.
    """
    a = cast[0] if cast else "FIRST"
    b = cast[1] if len(cast) > 1 else "SECOND"
    kw1 = premise_keywords[0]
    kw2 = premise_keywords[1] if len(premise_keywords) > 1 else premise_keywords[0]

    payload = {
        "news_premise": (
            f"The episode is about the {kw1} {kw2} story. "
            f"A new development reshapes what {a} and {b} thought possible."
        ),
        "dramatic_question": (
            f"Will {a} accept what the {kw1} discovery costs?"
        ),
        "opposed_desires": (
            f"{a} wants to protect what was. {b} wants to chase what "
            f"the {kw1} result makes possible."
        ),
        "arc_shape": (
            f"Open in disbelief. Turn on a direct demand from {b}. "
            f"Close on {a} making a choice with cost."
        ),
        "audience_feel": (
            "A quiet weight; the audience leaves thinking, not cheering."
        ),
        "forbidden_drift": [
            "code-red theatrics",
            "generic medical crisis",
        ],
        "raw_brief": (
            f"This is the {seed_slug} episode. The {kw1} {kw2} premise "
            f"must be audible inside the first thirty seconds. {a} and "
            f"{b} carry the two-hander; the arc lands on a principled "
            f"refusal or acceptance with a real cost. Avoid the genre "
            f"drift toward action; keep the weight quiet and human."
        ),
    }

    def _mock(messages, *, temperature, max_new_tokens):
        return json.dumps(payload)

    return _mock


# ---------------------------------------------------------------------------
# Unit -- 10 news seeds, every output passes Pydantic validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,cast",
    list(zip(_TEN_NEWS_SEEDS, _TEN_CAST_PAIRS)),
    ids=[f["slug"] for f in _TEN_NEWS_SEEDS],
)
def test_run_director_returns_valid_brief_for_each_seed(fixture, cast):
    """Every one of the 10 fixtures yields a DirectorBrief and a valid
    DirectorBriefSchema instance under the mock generate_fn."""
    mock_fn = _make_mock_generate_fn(
        fixture["slug"], cast, fixture["premise_keywords"],
    )

    brief = run_director(
        news_seed=fixture["seed"],
        cast_names=cast,
        generate_fn=mock_fn,
    )

    # Returned object is a DirectorBrief with all six prose fields
    # populated.
    assert isinstance(brief, DirectorBrief)
    for field_name in (
        "news_premise",
        "dramatic_question",
        "opposed_desires",
        "arc_shape",
        "audience_feel",
        "raw_brief",
    ):
        value = getattr(brief, field_name)
        assert isinstance(value, str) and value.strip(), (
            f"Field {field_name} empty for fixture {fixture['slug']}"
        )

    # forbidden_drift is a list of short phrases.
    assert isinstance(brief.forbidden_drift, list)
    assert all(isinstance(p, str) and p for p in brief.forbidden_drift)

    # The same payload re-validates as a DirectorBriefSchema (the
    # constrained-decode shape).
    schema = DirectorBriefSchema.model_validate(brief.to_dict())
    assert schema.news_premise == brief.news_premise


# ---------------------------------------------------------------------------
# Content audit -- premise keywords AND rubric axes themes appear
# ---------------------------------------------------------------------------


def _has_keyword(text: str, keyword: str) -> bool:
    """Case-insensitive substring match on a single keyword."""
    return keyword.lower() in (text or "").lower()


@pytest.mark.parametrize(
    "fixture,cast",
    list(zip(_TEN_NEWS_SEEDS, _TEN_CAST_PAIRS)),
    ids=[f["slug"] for f in _TEN_NEWS_SEEDS],
)
def test_brief_contains_news_premise_keywords(fixture, cast):
    """Every brief names at least one premise keyword in news_premise.

    The Director's load-bearing job is to NAME the news premise
    concretely. This test is the deterministic keyword-overlap heuristic
    the design Section 4 Wave 1 Agent D "Tests" block asks for.
    """
    mock_fn = _make_mock_generate_fn(
        fixture["slug"], cast, fixture["premise_keywords"],
    )
    brief = run_director(
        news_seed=fixture["seed"],
        cast_names=cast,
        generate_fn=mock_fn,
    )
    keywords = fixture["premise_keywords"]
    hits = [k for k in keywords if _has_keyword(brief.news_premise, k)]
    assert hits, (
        f"news_premise for {fixture['slug']!r} did not contain any of "
        f"{keywords}; got: {brief.news_premise!r}"
    )


@pytest.mark.parametrize(
    "fixture,cast",
    list(zip(_TEN_NEWS_SEEDS, _TEN_CAST_PAIRS)),
    ids=[f["slug"] for f in _TEN_NEWS_SEEDS],
)
def test_brief_names_both_cast_in_opposed_desires(fixture, cast):
    """Every brief names BOTH cast members in opposed_desires.

    Design rule: "Opposed desires must be specific, not abstract. Name
    BOTH characters from the cast list and say exactly what each
    wants." The mock honours this; the test pins the contract.
    """
    mock_fn = _make_mock_generate_fn(
        fixture["slug"], cast, fixture["premise_keywords"],
    )
    brief = run_director(
        news_seed=fixture["seed"],
        cast_names=cast,
        generate_fn=mock_fn,
    )
    for character_name in cast:
        assert _has_keyword(brief.opposed_desires, character_name), (
            f"opposed_desires did not name {character_name!r} for "
            f"fixture {fixture['slug']!r}: {brief.opposed_desires!r}"
        )


def test_prompt_includes_rubric_axes_when_supplied():
    """When a Rubric is supplied, the user prompt mentions its axis
    json_keys -- the Director uses the axis list as design hint so the
    brief sets up an episode that can pass the critic."""
    # Stub rubric with three axes; the real Rubric has 10 but the
    # prompt builder duck-types axis.json_key / axis.name.
    class _StubAxis:
        def __init__(self, json_key):
            self.json_key = json_key
            self.name = json_key

    class _StubRubric:
        axes = [
            _StubAxis("premise_clarity"),
            _StubAxis("emotional_arc"),
            _StubAxis("resolution"),
        ]
        threshold = None

    messages = build_director_prompt(
        "A test news seed.", ["ALPHA", "BETA"], rubric=_StubRubric(),
    )
    full_text = "\n".join(m["content"] for m in messages)
    for axis_key in ("premise_clarity", "emotional_arc", "resolution"):
        assert axis_key in full_text, (
            f"Director prompt did not mention rubric axis "
            f"{axis_key!r}; full prompt: {full_text!r}"
        )


def test_prompt_omits_rubric_block_when_rubric_none():
    """No rubric -> no rubric hint block in the user prompt."""
    messages = build_director_prompt(
        "A test news seed.", ["ALPHA", "BETA"], rubric=None,
    )
    full_text = "\n".join(m["content"] for m in messages)
    assert "Critic rubric axes" not in full_text


# ---------------------------------------------------------------------------
# Failure mode -- exhausted retry budget raises DirectorCallFailedError
# ---------------------------------------------------------------------------


def test_run_director_raises_on_exhausted_retry_budget():
    """A generate_fn that always returns invalid JSON exhausts the
    ladder and raises DirectorCallFailedError (loud failure, not silent
    sentinel -- the consuming node decides how to surface it)."""
    def _bad_fn(messages, *, temperature, max_new_tokens):
        return "not json at all"

    with pytest.raises(DirectorCallFailedError) as excinfo:
        run_director(
            news_seed="A seed.",
            cast_names=["A", "B"],
            generate_fn=_bad_fn,
            max_attempts=2,
        )
    assert excinfo.value.attempts == 2
    assert excinfo.value.last_error is not None


def test_run_director_raises_on_schema_violation():
    """Schema-violating output (missing required field) also exhausts
    the ladder -- the Director never silently emits a partial brief."""
    bad_payload = json.dumps({
        "news_premise": "Too short.",  # below min_length=20
        # missing every other field
    })

    def _bad_fn(messages, *, temperature, max_new_tokens):
        return bad_payload

    with pytest.raises(DirectorCallFailedError):
        run_director(
            news_seed="A seed.",
            cast_names=["A", "B"],
            generate_fn=_bad_fn,
            max_attempts=2,
        )


def test_run_director_recovers_on_second_attempt():
    """A first-attempt failure followed by a valid second attempt
    returns the second-attempt brief cleanly."""
    good_payload = json.dumps({
        "news_premise": (
            "A solid one-sentence premise about the lab test results."
        ),
        "dramatic_question": (
            "Can the protagonist accept the test results in time?"
        ),
        "opposed_desires": (
            "ALPHA wants to publish. BETA wants to wait for more data."
        ),
        "arc_shape": (
            "Open in disagreement, escalate to confrontation, close on a "
            "tentative agreement."
        ),
        "audience_feel": "An anxious but hopeful clarity at the close.",
        "forbidden_drift": ["explosion theatrics"],
        "raw_brief": (
            "A two-hander between ALPHA and BETA over a test result. "
            "Keep the weight in the disagreement, not in the science."
        ),
    })

    call_count = {"n": 0}

    def _flaky_fn(messages, *, temperature, max_new_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "garbage"
        return good_payload

    brief = run_director(
        news_seed="A seed.",
        cast_names=["ALPHA", "BETA"],
        generate_fn=_flaky_fn,
        max_attempts=2,
    )
    assert call_count["n"] == 2
    assert "ALPHA" in brief.opposed_desires
    assert "BETA" in brief.opposed_desires


# ---------------------------------------------------------------------------
# DirectorBrief.to_dict round-trip
# ---------------------------------------------------------------------------


def test_director_brief_to_dict_round_trip():
    """to_dict / DirectorBrief(**payload) round-trips losslessly so
    downstream agents can rebuild the dataclass from the JSON wire
    payload."""
    original = DirectorBrief(
        news_premise="Premise sentence about a test news event.",
        dramatic_question="Will it work in time?",
        opposed_desires="A wants X; B wants Y.",
        arc_shape="Open, turn, close.",
        audience_feel="Quiet weight.",
        forbidden_drift=["one", "two"],
        raw_brief="Full prose paragraph for downstream consumers.",
    )
    payload = original.to_dict()
    rebuilt = DirectorBrief(**payload)
    assert rebuilt == original


# ---------------------------------------------------------------------------
# Pydantic schema -- length bounds enforced
# ---------------------------------------------------------------------------


def test_director_brief_schema_rejects_empty_prose_field():
    """An empty news_premise is below min_length and rejects."""
    with pytest.raises(ValidationError):
        DirectorBriefSchema(
            news_premise="",
            dramatic_question="A real question that meets length.",
            opposed_desires="A meaningful opposed desires sentence.",
            arc_shape="A meaningful arc shape sentence here.",
            audience_feel="A meaningful audience feel sentence.",
            forbidden_drift=[],
            raw_brief="",
        )


def test_director_brief_schema_rejects_too_many_drift_entries():
    """More than 5 forbidden_drift entries fails the schema cap."""
    six_drift = ["a", "b", "c", "d", "e", "f"]
    with pytest.raises(ValidationError):
        DirectorBriefSchema(
            news_premise="A real premise sentence that meets length.",
            dramatic_question="A real question that meets length.",
            opposed_desires="A meaningful opposed desires sentence.",
            arc_shape="A meaningful arc shape sentence here.",
            audience_feel="A meaningful audience feel sentence.",
            forbidden_drift=six_drift,
            raw_brief="",
        )


# ---------------------------------------------------------------------------
# Node integration -- import + INPUT_TYPES + RETURN_TYPES
# ---------------------------------------------------------------------------


def test_node_class_imports_clean():
    """The Comfy node module imports without ComfyUI present and
    exposes the OTR_DirectorBrief class."""
    from nodes.OTR_DirectorBrief import OTR_DirectorBrief
    assert hasattr(OTR_DirectorBrief, "run")
    assert hasattr(OTR_DirectorBrief, "INPUT_TYPES")
    assert hasattr(OTR_DirectorBrief, "RETURN_TYPES")


def test_node_return_types_is_single_string():
    """RETURN_TYPES is ('STRING',) -- one output socket carrying the
    serialized DirectorBrief JSON."""
    from nodes.OTR_DirectorBrief import OTR_DirectorBrief
    assert OTR_DirectorBrief.RETURN_TYPES == ("STRING",)
    assert OTR_DirectorBrief.RETURN_NAMES == ("director_brief",)


def test_node_input_types_has_no_model_id_widget():
    """PD6: the node MUST NOT expose a `model_id` widget. The
    technical_model id arrives via a STRING input socket only."""
    from nodes.OTR_DirectorBrief import OTR_DirectorBrief
    spec = OTR_DirectorBrief.INPUT_TYPES()
    required = spec.get("required", {})
    optional = spec.get("optional", {})

    # No model_id widget anywhere.
    assert "model_id" not in required
    assert "model_id" not in optional

    # news_seed + cast_names are the required free-text widgets.
    assert "news_seed" in required
    assert "cast_names" in required

    # technical_model is force-input (no widget); it lives under
    # 'optional' so the node loads in the Comfy editor without an
    # immediate wire, and require_model fails loud at run-time if the
    # socket arrives unwired.
    assert "technical_model" in optional
    tm_spec = optional["technical_model"]
    # Spec shape: (type, opts_dict)
    assert tm_spec[0] == "STRING"
    assert tm_spec[1].get("forceInput") is True


def test_node_category_and_function():
    """CATEGORY + FUNCTION match the project convention so the node
    lands in the OldTimeRadio menu under v2."""
    from nodes.OTR_DirectorBrief import OTR_DirectorBrief
    assert OTR_DirectorBrief.CATEGORY == "OldTimeRadio/v2"
    assert OTR_DirectorBrief.FUNCTION == "run"


def test_node_input_types_marks_news_seed_multiline():
    """news_seed widget is multiline so a long news body fits."""
    from nodes.OTR_DirectorBrief import OTR_DirectorBrief
    spec = OTR_DirectorBrief.INPUT_TYPES()
    news_seed_spec = spec["required"]["news_seed"]
    assert news_seed_spec[0] == "STRING"
    assert news_seed_spec[1].get("multiline") is True


# ---------------------------------------------------------------------------
# Slot routing -- the source file carries the # LLM slot: technical tag
# ---------------------------------------------------------------------------


_DIRECTOR_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "_otr_director_brief.py"
)
_DIRECTOR_NODE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "OTR_DirectorBrief.py"
)


def test_director_module_tags_call_site_as_technical():
    """The single generate_fn call in _otr_director_brief.py carries
    a `# LLM slot: technical` tag within +/- 8 lines (the slot sweep
    boundary) so test_llm_slot_sweep.py keeps passing."""
    text = _DIRECTOR_MODULE_PATH.read_text(encoding="utf-8")
    tag = re.search(r"#\s*LLM\s+slot:\s*technical", text)
    assert tag is not None, (
        "_otr_director_brief.py missing '# LLM slot: technical' tag"
    )
    # And the source carries no 'creative' slot tag -- the Director is
    # technical-only.
    assert not re.search(r"#\s*LLM\s+slot:\s*creative", text), (
        "_otr_director_brief.py should not tag any call site 'creative'"
    )


def test_director_node_tags_request_slot_as_technical():
    """OTR_DirectorBrief.py acquires the technical slot via
    request_slot('technical', ...); the call site must carry the
    '# LLM slot: technical' tag for the slot-sweep CI gate."""
    text = _DIRECTOR_NODE_PATH.read_text(encoding="utf-8")
    assert "request_slot" in text, (
        "OTR_DirectorBrief.py must call _otr_model_loader.request_slot"
    )
    assert re.search(r"#\s*LLM\s+slot:\s*technical", text), (
        "OTR_DirectorBrief.py missing '# LLM slot: technical' tag"
    )
