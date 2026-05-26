"""Sprint 10A step 3-A regression -- Stage1Plan schema + semantic validators.

Covers:
  * The base happy-path plan parses cleanly.
  * Every documented field bound (min/max length, regex, Literal,
    enum, ge/le) rejects the out-of-bounds value at the pydantic
    layer.
  * Every cross-field semantic constraint in validate_plan_semantics
    raises ValueError when violated:
      - duplicate cast names
      - duplicate voice_ids
      - duplicate beat_ids
      - non-monotonic beat_ids
      - beat speaker not in cast and not a reserved speaker
      - callback_to to a non-existent beat
      - callback_to to a forward / self beat
  * `parse_and_validate_plan` end-to-end on a known-good plan returns
    a Stage1Plan instance.
  * The JSON schema export round-trips and contains the documented
    constant names (so commit 3-B's lm-format-enforcer integration
    has a stable shape to bind to).

These tests pin the schema as a contract surface. Downstream consumers
(step 4 cast audit, step 5 validators, step 6 Stage 2 prompt builder,
step 7 critic) build against the field names + types asserted here.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from nodes._otr_stage1_plan import (
    BEATS_MAX,
    BEATS_MIN,
    BEAT_ID_PATTERN,
    CAST_MAX,
    CAST_MIN,
    RUNNING_FACTS_MAX,
    RESERVED_SPEAKERS,
    VOICE_ID_PATTERN,
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
    get_plan_json_schema,
    parse_and_validate_plan,
    validate_plan_semantics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _good_plan_dict() -> dict:
    """A minimal valid plan. Tests mutate copies of this to probe failure modes."""
    return {
        "premise": "Two technicians at a Mars relay catch a signal that should not exist.",
        "arc": {
            "setup": "Routine night shift, calm comms traffic.",
            "complication": "An off-band carrier locks in and refuses to drop.",
            "resolution": "They identify the source and decide whether to acknowledge it.",
        },
        "cast": [
            {
                "name": "DALE PORTER",
                "gender": "male",
                "pronouns": "he/him",
                "voice_id": "v2/en_speaker_5",
                "persona": "Experienced relay technician. Speaks in measured cadences. Trusts procedure over hunches but knows when to bend it.",
                "arc_role": "anchor of competence",
            },
            {
                "name": "MEG SAARI",
                "gender": "female",
                "pronouns": "she/her",
                "voice_id": "v2/en_speaker_2",
                "persona": "Junior tech, six months on the post. Quick to question, slow to panic. Asks the obvious questions out loud.",
                "arc_role": "pressure source and witness",
            },
        ],
        "beats": [
            {
                "beat_id": "b001",
                "speaker": "ANNOUNCER",
                "intent": "Open the episode with a date stamp and the location.",
                "length_target_words": 18,
                "emotional_register": "neutral procedural",
                "callback_to": None,
            },
            {
                "beat_id": "b002",
                "speaker": "DALE PORTER",
                "intent": "Establish the routine night shift and Dale's role.",
                "length_target_words": 24,
                "emotional_register": "calm grounded",
                "callback_to": None,
            },
            {
                "beat_id": "b003",
                "speaker": "MEG SAARI",
                "intent": "Notice the off-band carrier and ask Dale about it.",
                "length_target_words": 22,
                "emotional_register": "alert curious",
                "callback_to": "b002",
            },
            {
                "beat_id": "b004",
                "speaker": "ANNOUNCER",
                "intent": "Close the episode with a date stamp and a hook.",
                "length_target_words": 16,
                "emotional_register": "measured close",
                "callback_to": None,
            },
        ],
        "running_facts": [
            "The relay is on Mars.",
            "Dale is senior to Meg.",
            "The off-band carrier should be impossible at this frequency.",
        ],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_good_plan_parses(self):
        plan = Stage1Plan(**_good_plan_dict())
        assert plan.premise.startswith("Two technicians")
        assert len(plan.cast) == 2
        assert len(plan.beats) == 4
        assert plan.running_facts[0] == "The relay is on Mars."

    def test_parse_and_validate_returns_instance(self):
        plan = parse_and_validate_plan(_good_plan_dict())
        assert isinstance(plan, Stage1Plan)

    def test_callback_none_normalized_to_none(self):
        d = _good_plan_dict()
        d["beats"][0]["callback_to"] = ""
        plan = Stage1Plan(**d)
        assert plan.beats[0].callback_to is None


# ---------------------------------------------------------------------------
# Field-level constraints (pydantic schema layer)
# ---------------------------------------------------------------------------


class TestFieldBounds:
    def test_premise_too_short_rejected(self):
        d = _good_plan_dict()
        d["premise"] = "short."
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_premise_too_long_rejected(self):
        d = _good_plan_dict()
        d["premise"] = "x" * 301
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_gender_outside_literal_rejected(self):
        d = _good_plan_dict()
        d["cast"][0]["gender"] = "cat"
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_pronouns_outside_literal_rejected(self):
        d = _good_plan_dict()
        d["cast"][0]["pronouns"] = "xe/xem"
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_voice_id_wrong_pattern_rejected(self):
        d = _good_plan_dict()
        d["cast"][0]["voice_id"] = "v2/en_speaker_42"
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_voice_id_bm_prefix_rejected(self):
        d = _good_plan_dict()
        # bm_* is Kokoro announcer territory (BUG-LOCAL-276 root cause) -- must reject.
        d["cast"][0]["voice_id"] = "bm_fable"
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_beat_id_wrong_pattern_rejected(self):
        d = _good_plan_dict()
        d["beats"][0]["beat_id"] = "b1"
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_length_target_below_min_rejected(self):
        d = _good_plan_dict()
        d["beats"][0]["length_target_words"] = 4
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_length_target_above_max_rejected(self):
        d = _good_plan_dict()
        d["beats"][0]["length_target_words"] = 201
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_cast_empty_rejected(self):
        d = _good_plan_dict()
        d["cast"] = []
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_cast_too_many_rejected(self):
        d = _good_plan_dict()
        base = d["cast"][0]
        d["cast"] = []
        for i in range(CAST_MAX + 1):
            row = copy.deepcopy(base)
            row["name"] = f"NAME {i}"
            row["voice_id"] = f"v2/en_speaker_{i % 10}"
            d["cast"].append(row)
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_beats_too_few_rejected(self):
        d = _good_plan_dict()
        d["beats"] = d["beats"][:2]  # 2 < BEATS_MIN
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_callback_to_bad_pattern_rejected(self):
        d = _good_plan_dict()
        d["beats"][2]["callback_to"] = "beat_2"
        with pytest.raises((ValidationError, ValueError)):
            Stage1Plan(**d)

    def test_running_facts_max_enforced(self):
        d = _good_plan_dict()
        d["running_facts"] = ["fact"] * (RUNNING_FACTS_MAX + 1)
        with pytest.raises(ValidationError):
            Stage1Plan(**d)

    def test_running_fact_empty_string_rejected(self):
        d = _good_plan_dict()
        d["running_facts"] = [""]
        with pytest.raises(ValidationError):
            Stage1Plan(**d)


# ---------------------------------------------------------------------------
# Semantic cross-field validators
# ---------------------------------------------------------------------------


class TestSemanticValidators:
    def test_duplicate_cast_names_rejected(self):
        d = _good_plan_dict()
        d["cast"][1]["name"] = d["cast"][0]["name"]
        d["cast"][1]["voice_id"] = "v2/en_speaker_3"  # keep voice unique
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="cast names must be unique"):
            validate_plan_semantics(plan)

    def test_duplicate_voice_ids_rejected(self):
        d = _good_plan_dict()
        d["cast"][1]["voice_id"] = d["cast"][0]["voice_id"]
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="voice_ids must be unique"):
            validate_plan_semantics(plan)

    def test_duplicate_beat_ids_rejected(self):
        d = _good_plan_dict()
        d["beats"][2]["beat_id"] = d["beats"][1]["beat_id"]
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="beat_ids must be unique"):
            validate_plan_semantics(plan)

    def test_non_monotonic_beat_ids_rejected(self):
        d = _good_plan_dict()
        # swap last two beat_ids to produce a descending pair
        d["beats"][2]["beat_id"], d["beats"][3]["beat_id"] = (
            d["beats"][3]["beat_id"],
            d["beats"][2]["beat_id"],
        )
        # restore callback consistency so the monotonic check fails first
        d["beats"][2]["callback_to"] = None
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="monotonically increasing"):
            validate_plan_semantics(plan)

    def test_beat_speaker_not_in_cast_or_reserved_rejected(self):
        d = _good_plan_dict()
        d["beats"][1]["speaker"] = "GHOST"  # not in cast, not reserved
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="neither a cast member"):
            validate_plan_semantics(plan)

    def test_reserved_speaker_accepted(self):
        # The good plan already has ANNOUNCER on b001 + b004; validate once more
        # explicitly with MUSIC inserted in place of a cast speaker.
        d = _good_plan_dict()
        d["beats"][2]["speaker"] = "MUSIC"
        d["beats"][2]["callback_to"] = None  # MUSIC beats don't carry callbacks naturally
        plan = Stage1Plan(**d)
        validate_plan_semantics(plan)  # should not raise

    def test_callback_to_nonexistent_beat_rejected(self):
        d = _good_plan_dict()
        d["beats"][2]["callback_to"] = "b999"
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="not in the plan"):
            validate_plan_semantics(plan)

    def test_callback_to_self_rejected(self):
        d = _good_plan_dict()
        d["beats"][2]["callback_to"] = d["beats"][2]["beat_id"]
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="earlier beat"):
            validate_plan_semantics(plan)

    def test_callback_to_forward_rejected(self):
        d = _good_plan_dict()
        d["beats"][0]["callback_to"] = d["beats"][2]["beat_id"]
        plan = Stage1Plan(**d)
        with pytest.raises(ValueError, match="earlier beat"):
            validate_plan_semantics(plan)


# ---------------------------------------------------------------------------
# Constants and reserved set
# ---------------------------------------------------------------------------


class TestConstants:
    def test_reserved_speakers_includes_announcer_and_music(self):
        assert "ANNOUNCER" in RESERVED_SPEAKERS
        assert "MUSIC" in RESERVED_SPEAKERS

    def test_voice_pattern_matches_v2_speakers(self):
        import re

        rx = re.compile(VOICE_ID_PATTERN)
        for n in range(10):
            assert rx.match(f"v2/en_speaker_{n}"), f"v2/en_speaker_{n} should match"
        # Negatives
        assert not rx.match("v2/en_speaker_10")
        assert not rx.match("v2/de_speaker_3")
        assert not rx.match("bm_fable")

    def test_beat_id_pattern(self):
        import re

        rx = re.compile(BEAT_ID_PATTERN)
        assert rx.match("b001")
        assert rx.match("b999")
        assert not rx.match("b1")
        assert not rx.match("b1000")
        assert not rx.match("a001")

    def test_bounds_constants_match_schema_export(self):
        schema = get_plan_json_schema()
        cast_props = schema["properties"]["cast"]
        assert cast_props["minItems"] == CAST_MIN
        assert cast_props["maxItems"] == CAST_MAX
        beats_props = schema["properties"]["beats"]
        assert beats_props["minItems"] == BEATS_MIN
        assert beats_props["maxItems"] == BEATS_MAX
        running_props = schema["properties"]["running_facts"]
        assert running_props["maxItems"] == RUNNING_FACTS_MAX


# ---------------------------------------------------------------------------
# JSON schema export (for the lm-format-enforcer integration in commit 3-B)
# ---------------------------------------------------------------------------


class TestSchemaExport:
    def test_schema_has_top_level_required_fields(self):
        schema = get_plan_json_schema()
        required = set(schema.get("required", []))
        for field in ("premise", "arc", "cast", "beats"):
            assert field in required, f"top-level required missing {field!r}"
        # running_facts has a default and is therefore not in the required set;
        # the field still appears in properties.
        assert "running_facts" in schema["properties"]

    def test_schema_cast_member_required_fields(self):
        schema = get_plan_json_schema()
        cm = schema["$defs"]["Stage1CastMember"]
        required = set(cm.get("required", []))
        for field in ("name", "gender", "pronouns", "voice_id", "persona", "arc_role"):
            assert field in required, f"cast member required missing {field!r}"

    def test_schema_beat_required_fields(self):
        schema = get_plan_json_schema()
        b = schema["$defs"]["Stage1Beat"]
        required = set(b.get("required", []))
        for field in (
            "beat_id",
            "speaker",
            "intent",
            "length_target_words",
            "emotional_register",
        ):
            assert field in required, f"beat required missing {field!r}"
        # callback_to is Optional[str] with default None -- not required.
        assert "callback_to" not in required

    def test_arc_three_fields(self):
        schema = get_plan_json_schema()
        arc = schema["$defs"]["Stage1Arc"]
        required = set(arc.get("required", []))
        assert required == {"setup", "complication", "resolution"}


# ---------------------------------------------------------------------------
# Inner models (one-off positive checks)
# ---------------------------------------------------------------------------


class TestInnerModels:
    def test_arc_alone_parses(self):
        arc = Stage1Arc(
            setup="The relay station is online.",
            complication="An off-band signal locks in.",
            resolution="Dale and Meg log the signal and call it in.",
        )
        assert arc.setup.startswith("The relay")

    def test_cast_member_alone_parses(self):
        cm = Stage1CastMember(
            name="DALE PORTER",
            gender="male",
            pronouns="he/him",
            voice_id="v2/en_speaker_5",
            persona="Experienced relay technician. Speaks in measured cadences.",
            arc_role="anchor of competence",
        )
        assert cm.voice_id == "v2/en_speaker_5"

    def test_beat_alone_parses(self):
        b = Stage1Beat(
            beat_id="b007",
            speaker="DALE PORTER",
            intent="Hand the binoculars to Meg.",
            length_target_words=20,
            emotional_register="quiet decision",
            callback_to=None,
        )
        assert b.beat_id == "b007"
