"""Positive schema and preservation contracts for the Sci-Fi News lane."""

from __future__ import annotations

import typing

from pydantic import BaseModel

from nodes import _otr_scifi_codex as lane


_JSON_IMPOSSIBLE = (tuple, set, frozenset)


def _strict_models():
    for name in dir(lane):
        obj = getattr(lane, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and (obj.model_config or {}).get("strict") is True
        ):
            yield name, obj


def _offending_types(annotation) -> list[type]:
    found: list[type] = []
    origin = typing.get_origin(annotation)
    if origin in _JSON_IMPOSSIBLE:
        found.append(origin)
    if annotation in _JSON_IMPOSSIBLE:
        found.append(annotation)
    for arg in typing.get_args(annotation):
        found.extend(_offending_types(arg))
    return found


def test_strict_lane_models_are_json_representable():
    violations = []
    for model_name, model in _strict_models():
        for field_name, field in model.model_fields.items():
            bad = _offending_types(field.annotation)
            if bad:
                violations.append(
                    f"{model_name}.{field_name}: {field.annotation!r}"
                )
    assert violations == []


def test_p1_authored_prose_is_preserved_without_length_classification():
    question = "Would the listening station answer? " + ("echo " * 300).strip()
    consequence = "The crew may choose silence. " + ("signal " * 300).strip()
    ending = "Let the final choice remain theirs. " + ("dawn " * 300).strip()
    artifact = lane.DramaticQuestionV4(
        question=question,
        consequence=consequence,
        ending_direction=ending,
    )
    assert artifact.question == question
    assert artifact.consequence == consequence
    assert artifact.ending_direction == ending


def test_p2_cast_names_and_descriptions_are_preserved_exactly():
    authored_name = "Doctor Vesper Quill of the Long Observatory"
    authored_description = "An exact authored description. " + ("detail " * 300).strip()
    role = "Keeps the source conflict alive. " + ("pressure " * 300).strip()
    cast = lane.CastPlanV4(
        cast=[
            lane.CastPlanRowV4(
                char_id="announcer",
                name="ANNOUNCER",
                character_description="The fixed transport announcer.",
                gender="unspecified",
                role_in_conflict="Frames the broadcast.",
                voice_slot="announcer",
            ),
            lane.CastPlanRowV4(
                char_id="c01",
                name=authored_name,
                character_description=authored_description,
                gender="an authored identity phrase",
                role_in_conflict=role,
                voice_slot="c01",
            ),
        ]
    )
    assert lane._validate_cast_plan(cast) is None
    row = cast.cast[1]
    assert row.name == authored_name
    assert row.character_description == authored_description
    assert row.role_in_conflict == role


def test_p3_score_draft_preserves_arbitrarily_long_authored_fields():
    title = "The Patient Signal " + ("returns " * 300).strip()
    premise = "A crew hears an impossible reply. " + ("because " * 300).strip()
    setting = "The listening room beyond the moon. " + ("window " * 300).strip()
    env = "A quiet receiver hall. " + ("relay " * 300).strip()
    scene_description = "The decision unfolds here. " + ("moment " * 300).strip()
    shot_description = "The operator watches the dial. " + ("image " * 300).strip()
    visual_prompt = "Velvet chair, cigarette smoke, neon ledger. " + ("texture " * 300).strip()
    intent = "Quill refuses the easy answer. " + ("choice " * 300).strip()
    arc_phase = "An authored phase name " + ("turn " * 300).strip()
    cue_description = "A patient electronic chord. " + ("tone " * 300).strip()
    generation_prompt = "Instrumental radio texture. " + ("sound " * 300).strip()

    draft = lane.RadioScoreDraftV4(
        title=title,
        premise=premise,
        setting=setting,
        scenes=[
            lane.RadioScoreDraftSceneV4(
                env=env,
                description=scene_description,
                shots=[
                    lane.RadioScoreDraftShotV4(
                        description=shot_description,
                        visual_prompt=visual_prompt,
                    )
                ],
                beats=[
                    lane.RadioScoreDraftBeatV4(
                        shot_index=0,
                        char_id="c01",
                        line_count=1,
                        intent=intent,
                        arc_phase=arc_phase,
                        fact_ids=[],
                    )
                ],
            )
        ],
        music_cues=[
            lane.RadioScoreDraftMusicCueV4(
                cue_id="music_open",
                description=cue_description,
                generation_prompt=generation_prompt,
                anchor_beat_index=0,
                anchor_line_index=0,
            )
        ],
    )

    assert draft.title == title
    assert draft.premise == premise
    assert draft.setting == setting
    assert draft.scenes[0].env == env
    assert draft.scenes[0].description == scene_description
    assert draft.scenes[0].shots[0].description == shot_description
    assert draft.scenes[0].shots[0].visual_prompt == visual_prompt
    assert draft.scenes[0].beats[0].intent == intent
    assert draft.scenes[0].beats[0].arc_phase == arc_phase
    assert draft.music_cues[0].description == cue_description
    assert draft.music_cues[0].generation_prompt == generation_prompt


def test_p0_provenance_artifact_keeps_finite_resource_bounds():
    schema = lane.FactIndexV4.model_json_schema()
    properties = schema["properties"]
    assert properties["facts"]["maxItems"] == 6
    assert properties["entities"]["maxItems"] == 4
    assert properties["numbers"]["maxItems"] == 4
    assert lane.SourceSpanV4.model_json_schema()["properties"]["quote"]["maxLength"] == 240
    assert lane.p0_output_token_budget() == 2800
