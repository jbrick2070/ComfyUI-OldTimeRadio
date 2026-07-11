"""Cross-lane structural guards for the sci-fi source banks.

These are CLASS-level regressions, not one-off fixes: each guard here pins a
defect that killed a live canonical roll on ONE lane and could equally kill the
others. Codex, Gemini, and Sonnet each hand a strict Pydantic model a blob of
model-emitted JSON, so a contract that JSON cannot express is not a style
problem -- it is an artifact that can never validate, no matter what the model
writes.

Live case (2026-07-11, Gemini P1, first roll to reach it):
    pitches: Input should be a valid tuple
      [type=tuple_type, input_value=[{'premise': ...}], input_type=list]
`PitchSlateV4.pitches` was annotated `tuple[PitchV4, PitchV4, PitchV4]`. JSON has
no tuple -- a model can only ever emit an array -- and the lane's `_Strict` config
(`strict=True`) refuses to coerce a list into a tuple. The pass was unsatisfiable
by construction. The fix expresses the same "exactly three" contract as a
length-pinned list.
"""
from __future__ import annotations

import importlib
import pathlib
import typing

import pytest
from pydantic import BaseModel

LANE_MODULES = (
    "nodes._otr_scifi_codex",
    "nodes._otr_scifi_gemini",
    "nodes._otr_scifi_sonnet",
)

# Types JSON cannot represent. A JSON document has exactly one sequence type
# (array) and one mapping type (object); it has no tuple, set, or frozenset.
JSON_IMPOSSIBLE = (tuple, set, frozenset)


def _strict_models(module):
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and (obj.model_config or {}).get("strict") is True
        ):
            yield name, obj


def _offending_types(annotation) -> list:
    """Every JSON-impossible origin anywhere inside a (possibly nested) annotation."""
    found = []
    origin = typing.get_origin(annotation)
    if origin in JSON_IMPOSSIBLE:
        found.append(origin)
    if annotation in JSON_IMPOSSIBLE:
        found.append(annotation)
    for arg in typing.get_args(annotation):
        found.extend(_offending_types(arg))
    return found


@pytest.mark.parametrize("module_name", LANE_MODULES)
def test_no_strict_lane_field_is_unrepresentable_in_json(module_name):
    """A strict lane model may not require a type JSON cannot produce.

    tuple / set / frozenset are unsatisfiable under `strict=True` when the value
    arrives from `json.loads` -- the pass can never validate. Use a length-pinned
    `list` (min_length == max_length) to express a fixed-arity contract instead.
    """
    module = importlib.import_module(module_name)
    violations = []
    for model_name, model in _strict_models(module):
        for field_name, field in model.model_fields.items():
            bad = _offending_types(field.annotation)
            if bad:
                violations.append(
                    f"{module_name}.{model_name}.{field_name}: "
                    f"{field.annotation!r} requires {bad[0].__name__}, "
                    f"which JSON cannot express"
                )
    assert not violations, (
        "strict lane model fields must be expressible in JSON -- a model can only "
        "ever emit an array, and strict mode will not coerce it:\n  "
        + "\n  ".join(violations)
    )


def test_gemini_pitch_slate_still_pins_exactly_three_pitches():
    """The JSON-native fix must not loosen the three-pitch contract."""
    from nodes import _otr_scifi_gemini as lane

    field = lane.PitchSlateV4.model_fields["pitches"]
    constraints = {
        type(meta).__name__: getattr(meta, "min_length", None) or getattr(meta, "max_length", None)
        for meta in field.metadata
    }
    assert constraints, "pitches must stay length-pinned"

    def pitch(n):
        return {"premise": f"p{n}", "setting": f"s{n}", "tonal_palette": f"t{n}"}

    # Exactly three validates -- from a LIST, the only thing JSON can deliver.
    slate = lane.PitchSlateV4.model_validate({"pitches": [pitch(0), pitch(1), pitch(2)]})
    assert len(slate.pitches) == 3
    # Two or four still fail closed.
    for count in (2, 4):
        with pytest.raises(Exception):
            lane.PitchSlateV4.model_validate(
                {"pitches": [pitch(i) for i in range(count)]}
            )


@pytest.mark.parametrize(
    "module_name,invoke_name",
    (
        ("nodes._otr_scifi_gemini", "invoke_gemini_structured"),
        ("nodes._otr_scifi_sonnet", "invoke_sonnet_structured"),
    ),
)
def test_source_grounded_p0_refuses_to_be_left_truncated(module_name, invoke_name):
    """P0 carries the source payload: it must fail loud, not lose its prefix.

    Parity with the Codex lane, which already pins this. A provenance prompt that
    is silently left-truncated drops the system/schema prefix and yields a
    confidently wrong artifact instead of an honest failure.
    """
    import ast
    import pathlib

    source = pathlib.Path(module_name.replace(".", "/") + ".py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != invoke_name:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pass_node = keywords.get("pass_id")
        if not (isinstance(pass_node, ast.Constant) and pass_node.value == "P0"):
            continue
        must_fit = keywords.get("prompt_must_fit")
        assert isinstance(must_fit, ast.Constant) and must_fit.value is True, (
            f"{module_name} P0 is source-grounded and must pass prompt_must_fit=True"
        )
        return
    pytest.fail(f"no P0 call found in {module_name}")


def _outline_payload(*, drop_visual_prompt=False):
    """A Gemini OutlineV4 shaped exactly like the failing live P3 artifact:
    nested shots/beats with NO parent scene_id and NO beat order."""
    shot = {"shot_id": "shot_001", "description": "The lab at night."}
    if not drop_visual_prompt:
        shot["visual_prompt"] = "A dim robotics lab, one monitor glowing."
    return {
        "title": "Signal Lost",
        "premise": "A team weighs a machine's judgment.",
        "setting": "University robotics lab",
        "time_of_day": "night",
        "cast": [
            {"char_id": "announcer", "name": "ANNOUNCER",
             "character_description": "The voice of the show.", "gender": "male"},
            {"char_id": "c01", "name": "Dr. Hart",
             "character_description": "Lead roboticist.", "gender": "female"},
        ],
        "scenes": [{
            "scene_id": "s001",
            "env": "lab",
            "description": "The team reviews the trial.",
            "shots": [shot],
            "beats": [
                {"beat_id": "b001", "line_id": "l001", "shot_id": "shot_001",
                 "speaker": "Dr. Hart", "char_id": "c01", "speaker_role": "character",
                 "intent": "report", "mood": "tense"},
                {"beat_id": "b002", "line_id": "l002", "shot_id": "shot_001",
                 "speaker": "Dr. Hart", "char_id": "c01", "speaker_role": "character",
                 "intent": "press", "mood": "urgent"},
            ],
        }],
        "music_cues": [{"cue_id": "music_open", "placement": "open",
                        "description": "cold hum", "generation_prompt": "low drone",
                        "anchor_beat_id": "b001"}],
        "advisory_word_bands": [{"beat_id": "b001", "advisory_word_center": 15},
                                {"beat_id": "b002", "advisory_word_center": 15}],
    }


def test_outline_repair_derives_parent_scene_and_beat_order():
    """The MECHANICAL half: a shot nested in s001 IS in s001. Not a creative call."""
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    repaired = lane.repair_outline_metadata(_json.dumps(_outline_payload()))
    assert repaired is not None
    scene = repaired.scenes[0]
    assert scene.shots[0].scene_id == "s001"
    assert [b.scene_id for b in scene.beats] == ["s001", "s001"]
    assert [b.order for b in scene.beats] == [1, 2]
    # Authored content is untouched, byte for byte.
    assert scene.shots[0].visual_prompt == "A dim robotics lab, one monitor glowing."
    assert scene.beats[0].intent == "report"


def test_outline_repair_refuses_to_invent_a_missing_visual_prompt():
    """The CREATIVE half: Python must never author story/image content.

    A missing visual_prompt fails closed so the typed creative repair still has to
    write it -- the deterministic path may normalize metadata, never imagine.
    """
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    assert lane.repair_outline_metadata(
        _json.dumps(_outline_payload(drop_visual_prompt=True))
    ) is None


def test_outline_rejects_and_then_repairs_a_renamed_announcer():
    """The announcer is a fixed ROLE LABEL, not a character the writer invents.

    CastLock's Gate 1 invariants skip the announcer row by the EXACT string
    "ANNOUNCER". A model-chosen "Narrator" makes them judge the announcer's kokoro
    preset (bm_george) as an invalid Bark voice and kill the run in the media tail.
    Catch it at P3, where the metadata repair can normalize the label for free.
    """
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    payload = _outline_payload()
    payload["cast"][0]["name"] = "Narrator"
    outline = lane.OutlineV4.model_validate(
        lane.normalize_outline_graph_metadata(_json.dumps(_outline_payload()))
    )
    assert lane.validate_outline_cast_labels(outline) is None

    renamed = lane.OutlineV4.model_validate(
        {**lane.normalize_outline_graph_metadata(_json.dumps(payload)),
         "cast": [{**payload["cast"][0], "name": "Narrator"}, payload["cast"][1]]}
    )
    assert "ANNOUNCER" in (lane.validate_outline_cast_labels(renamed) or "")

    # ...and the deterministic repair normalizes the label without touching story.
    repaired = lane.repair_outline_metadata(_json.dumps(payload))
    assert repaired is not None
    assert repaired.cast[0].name == "ANNOUNCER"
    assert repaired.premise == payload["premise"]


def test_outline_output_reservation_leaves_room_for_its_own_repair_prompt():
    """P3's output reservation must not eat the input budget its repair needs.

    Live (2026-07-11): P3 reserved a flat 3600 tokens against an 8192 cap, leaving
    4592 for input -- and the typed-repair prompt was 5408, so the guard
    LEFT-TRUNCATED it and cut off the schema/instruction prefix. The model was not
    ignoring the repair rules; it never received them. Every repair was doomed
    before it was sent.
    """
    from nodes import _otr_scifi_gemini as lane

    CONTEXT_CAP = 8192
    OBSERVED_REPAIR_PROMPT_TOKENS = 5408

    budget = lane.outline_output_token_budget(30, 5)
    assert CONTEXT_CAP - budget > OBSERVED_REPAIR_PROMPT_TOKENS, (
        "the 30-word P3 reservation must leave room for the observed repair prompt"
    )
    # It still scales up with the work, and stays inside the old ceiling.
    assert lane.outline_output_token_budget(720, 12) <= 3600
    assert (
        lane.outline_output_token_budget(720, 12)
        > lane.outline_output_token_budget(30, 5)
    )
    for bad in (True, 29, 901, 30.0):
        with pytest.raises(lane.SciFiGeminiTargetRangeError):
            lane.outline_output_token_budget(bad, 5)


def test_outline_repair_drops_forbidden_extra_keys_without_touching_story():
    """Strict models forbid extras, and the model garnishes its output.

    An unrequested key is not authored content -- the contract never had a slot for
    it -- so dropping it invents nothing and discards no story. (Live: once the
    truncation fix let the model finally READ the contract, its next failure was
    5 x extra_forbidden.)
    """
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    payload = _outline_payload()
    payload["mood_board"] = "not in the schema"                    # root extra
    payload["scenes"][0]["camera_notes"] = "nope"                  # scene extra
    payload["scenes"][0]["shots"][0]["lens"] = "35mm"              # shot extra
    payload["scenes"][0]["beats"][0]["subtext"] = "nope"           # beat extra
    payload["cast"][0]["voice_preset"] = "bm_george"               # cast extra

    repaired = lane.repair_outline_metadata(_json.dumps(payload))
    assert repaired is not None
    # Authored content survives byte for byte.
    assert repaired.premise == payload["premise"]
    assert repaired.scenes[0].shots[0].visual_prompt == (
        "A dim robotics lab, one monitor glowing."
    )
    assert repaired.scenes[0].beats[0].intent == "report"


def test_generic_extra_key_repair_saves_a_scene_draft_without_touching_dialogue():
    """P4 threw away a whole scene -- dialogue and all -- over an extra key.

    The model hands each drafted line a `fact_ids` list (the outline's beats carry
    one) but DraftLineV4 has no such field, so extra="forbid" rejected the scene.
    Pruning the unrequested key loses no authored work; the dialogue is untouched.
    """
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    raw = _json.dumps({
        "scene_id": "s001",
        "lines": [
            {"beat_id": "b001", "text": "The lab is quiet tonight.",
             "fact_uses": [], "non_fact": True, "fact_ids": ["F01"]},
            {"beat_id": "b002", "text": "Too quiet.",
             "fact_uses": [], "non_fact": True, "fact_ids": []},
        ],
    })
    repaired = lane.repair_forbidden_extra_keys(raw, lane.SceneDraftV4)
    assert repaired is not None
    assert [ln.text for ln in repaired.lines] == [
        "The lab is quiet tonight.", "Too quiet.",
    ]
    assert not hasattr(repaired.lines[0], "fact_ids")

    # A genuinely missing required field is the model's job, not ours.
    assert lane.repair_forbidden_extra_keys(
        _json.dumps({"scene_id": "s001"}), lane.SceneDraftV4
    ) is None


def test_outline_repair_fails_closed_without_an_owning_scene_id():
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    payload = _outline_payload()
    del payload["scenes"][0]["scene_id"]
    assert lane.repair_outline_metadata(_json.dumps(payload)) is None
    assert lane.repair_outline_metadata("{not json") is None


def test_scene_draft_must_cover_every_beat_in_its_scene():
    """A skipped beat is AUTHORED work missing -- catch it where it can be rewritten.

    Live: the model drafted no line for b004, the scene passed its critique anyway,
    and the run died far downstream in `_assemble` with a bare "missing draft for
    b004" -- after every generation had already been paid for. Python cannot invent
    the dialogue, so the pass that owns it must demand it.
    """
    from nodes import _otr_scifi_gemini as lane

    scene = lane.SceneV4.model_validate({
        "scene_id": "s001", "env": "lab", "description": "d",
        "shots": [{"shot_id": "sh1", "scene_id": "s001",
                   "description": "d", "visual_prompt": "v"}],
        "beats": [
            {"beat_id": "b003", "line_id": "l003", "scene_id": "s001",
             "shot_id": "sh1", "speaker": "Dr. Hart", "char_id": "c01",
             "speaker_role": "character", "intent": "i", "mood": "m", "order": 1},
            {"beat_id": "b004", "line_id": "l004", "scene_id": "s001",
             "shot_id": "sh1", "speaker": "Dr. Hart", "char_id": "c01",
             "speaker_role": "character", "intent": "i", "mood": "m", "order": 2},
        ],
    })

    def draft(beat_ids):
        return lane.SceneDraftV4.model_validate({
            "lines": [
                {"beat_id": b, "text": f"line for {b}", "fact_uses": [],
                 "non_fact": True}
                for b in beat_ids
            ],
        })

    assert lane.validate_scene_draft_covers_beats(draft(["b003", "b004"]), scene) is None
    assert "b004" in lane.validate_scene_draft_covers_beats(draft(["b003"]), scene)
    assert "more than once" in lane.validate_scene_draft_covers_beats(
        draft(["b003", "b003", "b004"]), scene
    )
    assert "outside this scene" in lane.validate_scene_draft_covers_beats(
        draft(["b003", "b004", "b999"]), scene
    )


def test_no_lane_seam_treats_the_word_target_as_a_quota():
    """The requested word count is a SCALE REQUEST, not a quota.

    Live kill (2026-07-11, Gemini): the scene-critique seam ordered the critic to
    "Ensure the total word count of the lines equals the scene's target word limit."
    At 30 words over 6 beats that is ~5 words a beat -- exact equality is
    unreachable, so the critic failed the scene, the bounded rewrite missed too, and
    the run died with "scene scene_01 failed its bounded rewrite". The model was
    obeying us.

    Operator law: word count never causes trimming, padding, culling, or a rewrite.
    It is a statistic recorded after the fact. Guard the seams so it cannot creep
    back in as a gate.
    """
    import glob
    import json as _json
    import re

    repo = pathlib.Path(__file__).resolve().parent.parent
    # A sentence that talks about the word target AND commands an exact match is a
    # quota. A sentence that explicitly frames it as advisory is the cure, not the
    # disease -- so it is not an offender even though it says the word "quota".
    mentions = re.compile(r"word count|word target|target word", re.IGNORECASE)
    commands = re.compile(
        r"\b(must equal|equals|meet it exactly|match(es)? exactly|must match)\b",
        re.IGNORECASE,
    )
    cures = re.compile(r"advisory|not a quota|never pad|never fail", re.IGNORECASE)

    offenders = []
    for path in glob.glob(str(repo / "nodes" / "story_packs" / "scifi_*" / "*.json")):
        # The Codex pack is owned by a concurrent workstream; guard the lanes we own.
        if "scifi_codex" in path:
            continue
        pack = _json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for seam, text in (pack.get("prompt_stages") or {}).items():
            if not isinstance(text, str):
                continue
            for sentence in re.split(r"(?<=\.)\s+|\n", text):
                if (
                    mentions.search(sentence)
                    and commands.search(sentence)
                    and not cures.search(sentence)
                ):
                    offenders.append(
                        f"{pathlib.Path(path).name}::{seam}: {sentence.strip()[:120]}"
                    )
    assert not offenders, (
        "a lane seam is treating the advisory word target as a hard quota:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
