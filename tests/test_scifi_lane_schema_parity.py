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
import json
import pathlib
import typing
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

LANE_MODULES = (
    "nodes._otr_scifi_codex",
    "nodes._otr_scifi_gemini",
    "nodes._otr_scifi_sonnet",
    "nodes._otr_original_codex56sol",
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


# A model-authored collection whose ITEM has no declared shape is a coin flip: the
# model must guess whether to return a list of things or an object grouping those
# things. Both readings satisfy `list[dict[str, str]]`-shaped prose, so which one you
# get depends on the model, and the seam's wording, and the day.
#
# Live case (2026-07-11, Codex P6, gemma-4-E4B on the creative slot):
#     issues: Input should be a valid list
#       [type=list_type, input_value={'blur_of_causality': [{...}]}, input_type=dict]
# `ListenerReviewV4.issues` was `list[dict[str, str]]` and the seam named six
# diagnostic lenses. The model grouped its issues under those lenses -- a completely
# reasonable reading of what we wrote -- and P6 exhausted its ladder and killed the
# roll. Mistral had been guessing the other way and we mistook that for a contract.
#
# Each entry below is a field allowed to stay shapeless, with the reason it is NOT a
# coin flip. Anything else must nest a real model. Add to this list only with a reason
# that a NEW model, reading only the schema, could not misread.
ALLOWED_SHAPELESS = {
    # justification: Python BUILDS this plan and the model copies back the literal it
    # was handed. The shape is demonstrated by value, not described in prose, so there
    # is nothing for the model to guess.
    "nodes._otr_scifi_codex.AdvisoryWordPlanV4.per_beat",
    # justification: guarded. _has_forbidden_script_scene_keys pins the container to a
    # list of dicts and rejects score-shaped echoes, and a deterministic repair covers
    # the rest. The free keys inside are deliberate room for the script's scene prose.
    "nodes._otr_scifi_codex.ScriptArtifactV4.scenes",
    # justification: the source payload Python hands IN (headline/summary/full_text ->
    # text), echoed back. A real mapping over known keys, never guessed.
    "nodes._otr_scifi_gemini.GeminiPayloadV4.payload",
    # justification: same input payload echo as the Gemini lane above.
    "nodes._otr_scifi_sonnet.PayloadV4.payload",
    # justification: a true mapping, not a collection -- line_id -> the fact IDs that
    # line speaks. The key IS the identifier, so "should this have been a list?" has
    # no second reading, which is exactly the ambiguity this guard exists to catch.
    "nodes._otr_scifi_gemini.SceneCritiqueV4.line_fact_ids",
}


def test_no_lane_asks_a_model_to_count_words():
    """No strict lane model may carry a model-reported word count.

    Word count is advisory and never a gate (operator law). A field that asks the model
    to report counts invites a seam that audits them -- and an LLM cannot count words,
    so the gate fails on a measurement the model cannot make and the writer cannot fix.
    Python measures the real count objectively at assembly (`word_receipt`).

    Live case (2026-07-11): `FinalAuditV4.observed_word_counts` was REQUIRED, and
    `codex_final_audit_system` told the auditor to check the "exact word count" and
    "pass only if all checks are true" -- a word-count gate that could demand a full
    P9 rewrite. Nothing in Python ever read the field.
    """
    offenders = []
    for module_name in LANE_MODULES:
        module = importlib.import_module(module_name)
        for model_name, model in _strict_models(module):
            for field_name in model.model_fields:
                lowered = field_name.lower()
                if "word" in lowered and ("count" in lowered or "len" in lowered):
                    offenders.append(f"{module_name}.{model_name}.{field_name}")
    assert not offenders, (
        "these fields ask a model to report a word count -- Python measures words, "
        "models write them:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("pack_path", sorted(
    pathlib.Path(__file__).resolve().parents[1].joinpath("nodes", "story_packs").glob("*/*_v1.json")
))
def test_no_seam_audits_word_count(pack_path):
    """A seam may steer toward a length. It may never make length a pass/fail check."""
    import json

    text = json.dumps(json.loads(pack_path.read_text(encoding="utf-8"))).lower()
    forbidden = ("exact word count", "verify the word count", "correct word count")
    hits = [phrase for phrase in forbidden if phrase in text]
    assert not hits, (
        f"{pack_path.name} makes word count auditable: {hits}. Word count is advisory; "
        "gating on it asks a model to enforce something it cannot measure."
    )


def _is_shapeless(annotation) -> bool:
    """True when a container's items have no declared shape (dict/Any, not a model)."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is typing.Any:
        return True
    if origin is dict or annotation is dict:
        return True
    if origin in (list, set, frozenset, tuple):
        return any(_is_shapeless(arg) for arg in args)
    if origin is typing.Union or str(origin) == "types.UnionType":
        return any(_is_shapeless(a) for a in args if a is not type(None))
    return False


@pytest.mark.parametrize("module_name", LANE_MODULES)
def test_no_model_authored_collection_is_shapeless(module_name):
    """A collection the model FILLS must declare what one item looks like.

    Otherwise the model has to guess between `[{...}, {...}]` and
    `{"category": [{...}]}` -- and a schema that admits two readings will eventually
    be read the other way. Nest a real model (see ListenerIssueV4) instead.
    """
    module = importlib.import_module(module_name)
    violations = []
    for model_name, model in _strict_models(module):
        for field_name, field in model.model_fields.items():
            path = f"{module_name}.{model_name}.{field_name}"
            if path in ALLOWED_SHAPELESS:
                continue
            if _is_shapeless(field.annotation):
                violations.append(f"{path}: {field.annotation!r}")
    assert not violations, (
        "these model-authored fields have no declared item shape, so the model must "
        "guess the container -- nest a real model, or justify it in ALLOWED_SHAPELESS:"
        "\n  " + "\n  ".join(violations)
    )


def test_listener_issue_is_typed_and_keeps_category_open():
    """The P6 fix pins the SHAPE without pinning the listener's vocabulary."""
    from nodes import _otr_scifi_codex as lane

    issues = lane.ListenerReviewV4.model_fields["issues"]
    assert typing.get_args(issues.annotation)[0] is lane.ListenerIssueV4, (
        "issues must be a list of typed ListenerIssueV4, not a shapeless container"
    )
    # A listener that coins a better word for the flaw than our six is doing its job.
    # Pinning `category` to an enum would reject it for that.
    assert lane.ListenerIssueV4.model_fields["category"].annotation is str


def test_listener_review_repair_flattens_grouped_issues_verbatim():
    """The exact payload that killed the 2026-07-11 roll must now normalize.

    Python re-homes the model's sentences into the right container. It must never
    write one: every word in the repaired review has to appear in the raw output.
    """
    from nodes import _otr_scifi_codex as lane

    raw = """{
      "strengths": ["The cold open earns its silence."],
      "issues": {
        "blurred_causality": [
          {"line_id": "l004", "direction": "Show the relay failing before Vesh names it."}
        ],
        "stalled_pacing": [
          {"direction": "Cut the second corridor beat; it repeats the first."}
        ]
      },
      "require_full_retake": true
    }"""

    review = lane.repair_listener_review_shape(raw)
    assert review is not None
    assert [i.category for i in review.issues] == ["blurred_causality", "stalled_pacing"]
    assert review.issues[0].line_id == "l004"
    assert review.issues[1].line_id is None
    assert review.strengths == ["The cold open earns its silence."]
    assert review.require_full_retake is True
    for issue in review.issues:
        assert issue.direction in raw, "Python may re-home the model's words, never write them"


def test_listener_review_repair_fails_closed_on_an_unmappable_issue():
    """Fail closed rather than silently drop a diagnosis.

    If we cannot tell which of several strings is the direction, guessing would either
    invent a critique or quietly lose one. Return None and let the typed repair ask.
    """
    from nodes import _otr_scifi_codex as lane

    raw = '{"issues": {"pacing": [{"where": "l002", "why": "slow", "how": "trim"}]}}'
    assert lane.repair_listener_review_shape(raw) is None


def test_no_lane_demands_frozen_clean():
    """`frozen_with_warns` is a CLEAN freeze and must never block a finished episode.

    It means the reviewer read the ledger and made NO edits; the warns are soft gaps --
    notes. The structural verdicts (frozen_with_doctor_edits, too_many_edits,
    needs_full_rerun) are the ones that block, because those are defects.

    Live case (2026-07-11, Codex, gemma-4-E4B): the pre-tail audit demanded
    `frozen_clean` outright while the saved-ledger audit ten lines below accepted
    `frozen_with_warns` -- so the IDENTICAL ledger was illegal before the save and legal
    after it. "frozen_with_warns -- 2 soft gap(s)" killed a roll that had passed P0-P8.
    Gemini and Sonnet had already been fixed; Codex was the last lane holding the stale
    gate, and nothing was watching for the divergence.
    """
    import pathlib
    import re

    nodes = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    offenders = []
    for lane in sorted(nodes.glob("_otr_scifi_*.py")):
        for lineno, line in enumerate(lane.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            # An equality test against frozen_clean alone -- the membership form
            # `not in ("frozen_clean", "frozen_with_warns")` is the correct gate.
            if re.search(r'(?:!=|==)\s*["\']frozen_clean["\']', line):
                offenders.append(f"{lane.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "a freeze gate compares against frozen_clean alone, so frozen_with_warns -- a "
        "clean freeze with no reviewer edits -- would kill a finished episode:\n  "
        + "\n  ".join(offenders)
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


@pytest.mark.parametrize(
    "module_name,root_name,facts_key,entities_key,numbers_key,fact_def",
    (
        ("nodes._otr_scifi_codex", "FactIndexV4", "facts", "entities", "numbers", "FactV4"),
        ("nodes._otr_scifi_gemini", "FactIndexV4", "facts", "entities", "numbers", "FactV4"),
        ("nodes._otr_scifi_sonnet", "FragmentDossierV4", "verified_facts", "named_entities", "key_numbers", "EvidenceFactV4"),
    ),
)
def test_source_grounded_p0_has_a_finite_shared_output_envelope(
    module_name, root_name, facts_key, entities_key, numbers_key, fact_def,
):
    """BUG-11.50: a capped P0 needs a schema-bounded artifact surface."""
    module = importlib.import_module(module_name)
    root_schema = getattr(module, root_name).model_json_schema()
    properties = root_schema["properties"]
    assert properties[facts_key]["maxItems"] == 6
    assert properties[entities_key]["maxItems"] == 4
    assert properties[numbers_key]["maxItems"] == 4
    assert properties["tone"]["minLength"] == 1
    assert properties["tone"]["maxLength"] == 80
    assert module.SourceSpanV4.model_json_schema()["properties"]["quote"]["maxLength"] == 240
    fact_schema = root_schema["$defs"][fact_def]
    assert fact_schema["properties"]["source_spans"]["maxItems"] == 1
    assert fact_schema["properties"]["claim"]["maxLength"] == 240
    assert module.p0_output_token_budget() == 2800


@pytest.mark.parametrize(
    "module_name,invoke_name,envelope_name,seam,root_name,kind",
    (
        ("nodes._otr_scifi_gemini", "invoke_gemini_structured", "GeminiPayloadV4", "gemini_fact_extraction", "FactIndexV4", "fact_index"),
        ("nodes._otr_scifi_sonnet", "invoke_sonnet_structured", "PayloadV4", "sonnet_intake_system", "FragmentDossierV4", "dossier"),
    ),
)
def test_sibling_p0_typed_repairs_are_compact_and_require_scalar_tone(
    module_name, invoke_name, envelope_name, seam, root_name, kind,
):
    """Gemini and Sonnet must not revive the generic copyable repair envelope."""
    module = importlib.import_module(module_name)
    source_payload = {
        "headline": "Signal report",
        "summary": "A careful observatory report.",
        "full_text": "A careful observatory report records a signal.",
        "source": "Test Wire",
        "date": "2026-07-12",
        "link": "https://example.invalid/report",
        "seed_text": "A careful observatory report records a signal.",
    }
    envelope = getattr(module, envelope_name)(
        payload=source_payload, source_mode="rss", payload_sha256="0" * 64,
    )
    span = {"field": "headline", "start": 0, "end": 6, "quote": "Signal"}
    if kind == "fact_index":
        artifact = {
            "facts": [{
                "fact_id": "F01", "claim": "A signal is reported.",
                "source_spans": [span], "numeric_tokens": [],
            }],
            "entities": [], "numbers": [], "tone": [], "payload_sha256": "0" * 64,
        }
    else:
        artifact = {
            "verified_facts": [{
                "fact_id": "fact_1", "claim": "A signal is reported.",
                "source_spans": [span],
            }],
            "key_numbers": [], "named_entities": [], "tone": [],
            "headline_clean": "Signal report", "provenance_note": "Test Wire, 2026-07-12.",
            "payload_sha256": "0" * 64,
        }
    repaired = {**artifact, "tone": "cautious"}
    replies = iter((json.dumps(artifact), json.dumps(repaired)))
    calls: list[dict[str, object]] = []
    journal: dict[str, object] = {}

    def slot_fn(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return next(replies)

    result = getattr(module, invoke_name)(
        pass_id="P0", slot="technical", slot_fn=slot_fn,
        seam_ref=seam, pack=SimpleNamespace(prompt_stages={seam: "Read source."}),
        typed_inputs={"payload": envelope.model_dump(mode="json")},
        result_type=getattr(module, root_name), post_validator=lambda _: None,
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=module.p0_output_token_budget(), journal=journal,
        prompt_must_fit=True,
    )

    assert result.tone == "cautious"
    assert len(calls) == 2
    assert all(
        getattr(call["messages"], "_otr_prompt_must_fit", False)
        for call in calls
    )
    assert "P0 COMPACT EXTRACTION CONTRACT" in calls[0]["messages"][0]["content"]
    repair_system = calls[-1]["messages"][0]["content"]
    repair_user = calls[-1]["messages"][1]["content"]
    assert "tone is one nonempty scalar" in repair_system
    assert "<failed_fact_index>" in repair_user
    assert "<source_evidence>" in repair_user
    assert "original_request" not in repair_user
    assert "artifact_inputs" not in repair_user
    attempts = journal["calls"][0]["attempts"]
    assert all(
        {"temperature", "max_new_tokens", "raw_chars", "raw_sha256"}
        <= set(attempt)
        for attempt in attempts
    )


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


def test_gemini_draft_seams_ask_for_the_schema_the_lane_actually_enforces():
    """A seam may not ask the model for a field its strict schema forbids.

    The draft and rewrite seams told the model to "return its fact_ids" and showed
    `{"fact_ids": ["F01"]}` -- but DraftLineV4 has `fact_uses` and forbids extras. So
    the model obeyed the seam, strict mode rejected the artifact, and the
    extra-key repair then DELETED the model's fact attribution to make it validate.
    The critic (correctly) reported "F01 is missing from line_fact_ids" and the run
    died. A contract that contradicts itself is not a model failure.
    """
    import json as _json

    from nodes import _otr_scifi_gemini as lane

    repo = pathlib.Path(__file__).resolve().parent.parent
    pack = _json.loads(
        (repo / "nodes" / "story_packs" / "scifi_gemini" / "scifi_gemini_v1.json")
        .read_text(encoding="utf-8")
    )
    stages = pack["prompt_stages"]
    draft_fields = set(lane.DraftLineV4.model_fields)

    for seam in ("gemini_scene_draft", "gemini_scene_rewrite"):
        text = stages[seam]
        # The worked example must not SHOW a key the strict line model forbids.
        assert '"fact_ids": ["F01"]' not in text, (
            f"{seam} shows a fact_ids field that DraftLineV4 forbids"
        )
        # ...and must show the one it actually requires.
        assert "fact_uses" in text, f"{seam} never mentions fact_uses"
        assert "fact_uses" in draft_fields and "fact_ids" not in draft_fields


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


def _sonnet_line(lane, text, cites=(), non_fact=False, speaker="ORUM", char_id="c02"):
    return lane.DraftLineV4(
        text=text, cites=list(cites), non_fact=non_fact,
        speaker=speaker, char_id=char_id, source_pass="P2a",
    )


def test_sonnet_ceremonial_line_may_cite_nothing_instead_of_a_fact_that_cannot_exist():
    """`cites` required >= 1, so lines that state no fact cited a sentinel `fact_0`.

    No such fact can ever exist: the P0 dossier contract is one-based (fact_1,
    fact_2, ...). Every ceremonial line in the episode -- the cold open, the
    Warden's rulings, the sign-off -- was carrying a FALSE citation, and the seal
    and sign-off borrowed the attestation's real fact id for a claim they never
    make. An honest empty is the truthful record.
    """
    from nodes import _otr_scifi_sonnet as lane

    ceremonial = _sonnet_line(
        lane, "The Archive is convened.", non_fact=True,
        speaker="ANNOUNCER", char_id="announcer",
    )
    assert ceremonial.cites == []

    # A line that states a fact must still cite one...
    assert _sonnet_line(lane, "The wording says sixty kelvin.", cites=["fact_1"]).cites

    # ...and the two may never disagree.
    with pytest.raises(Exception):
        _sonnet_line(lane, "x", cites=["fact_1"], non_fact=True)   # cites a fact it disclaims
    with pytest.raises(Exception):
        _sonnet_line(lane, "x", cites=[], non_fact=False)          # claims a fact, cites none

    # The attestation contract is 1-3, matching every line schema it becomes.
    assert lane.AttestationV4.model_fields["attestation_cites"].metadata


def test_sonnet_rewrite_corrections_are_written_back_into_the_record():
    """They never were -- which is why Sonnet has never completed a run.

    The loop validated the doctor's corrections, threw them away, and re-audited
    the UNCHANGED draft. The recheck re-read the very text it had just condemned,
    so the audit could only exhaust. Python integrates the model's replacement
    text; it authors none of it.
    """
    from nodes import _otr_scifi_sonnet as lane

    events = [
        _sonnet_line(lane, "cold open", non_fact=True,
                     speaker="ANNOUNCER", char_id="announcer"),
        _sonnet_line(lane, "ORUM says a wrong thing", cites=["fact_1"]),
        _sonnet_line(lane, "THESSALY speculates", cites=["fact_2"],
                     speaker="THESSALY", char_id="c03"),
    ]
    audited = lane._audited_line_indices(events)
    assert audited == [1, 2], "the cold open is never numbered for the audit"

    audit = lane.AuditVerdictV4.model_validate({
        "status": "defect", "defects": ["line 0 misstates the wording"],
        "flagged_line_refs": [0], "invented_fact_flags": [], "severity": "critical",
        "sfw_pass": True,
    })
    rewrite = lane.RewriteResultV4.model_validate({
        "corrected_lines": [{
            "line_ref": 0, "speaker": "ORUM",
            "text": "ORUM says the accurate thing", "cites": ["fact_1"],
        }],
        "vesh_resolution": "The record stands corrected.",
    })

    lane._apply_rewrite_corrections(events, audited, rewrite, audit, 0)

    assert events[1].text == "ORUM says the accurate thing"
    assert events[1].char_id == "c02"          # locked cast identity survives
    assert events[1].source_pass == "P5:0"
    assert events[2].text == "THESSALY speculates"   # untouched line is byte-identical
    assert events[0].text == "cold open"

    # The same line returned twice is incoherent -- two texts for one line, nothing
    # to choose between them. Fail closed.
    with pytest.raises(lane.SonnetCompletenessError):
        lane._apply_rewrite_corrections(
            events, audited,
            lane.RewriteResultV4.model_validate({
                "corrected_lines": [
                    {"line_ref": 0, "speaker": "ORUM", "text": "a", "cites": ["fact_1"]},
                    {"line_ref": 0, "speaker": "ORUM", "text": "b", "cites": ["fact_1"]},
                ],
                "vesh_resolution": "r",
            }),
            audit, 0,
        )

    # An index that does not exist points at no line we can correct. Refuse the edit,
    # but do not kill the episode over the doctor miscounting -- the recheck is still
    # the judge (live: it returned line_ref 4 for a 0..3 draft).
    kept = [line.text for line in events]
    lane._apply_rewrite_corrections(
        events, audited,
        lane.RewriteResultV4.model_validate({
            "corrected_lines": [
                {"line_ref": 9, "speaker": "ORUM", "text": "t", "cites": ["fact_1"]},
            ],
            "vesh_resolution": "r",
        }),
        audit, 0,
    )
    assert [line.text for line in events] == kept

    # An eager doctor correcting a line the AUDIT never flagged is coherent but out
    # of scope. The auditor decides what is defective -- so the edit is ignored and
    # the original line stands, rather than the whole episode dying over it (live:
    # "rewrite corrected line 3, which the audit never flagged").
    before = events[2].text
    lane._apply_rewrite_corrections(
        events, audited,
        lane.RewriteResultV4.model_validate({
            "corrected_lines": [
                {"line_ref": 1, "speaker": "THESSALY", "text": "unasked-for edit",
                 "cites": ["fact_2"]},
            ],
            "vesh_resolution": "r",
        }),
        audit, 0,
    )
    assert events[2].text == before


# --------------------------------------------------------------------------- #
# The two guards below exist because the lessons were NOT automatically carried
# from one lane to the next. Codex, Gemini and Sonnet were each written
# independently, so each one independently re-made the same mistakes: a seam that
# asks for a field its schema forbids, and Python putting words in a character's
# mouth. A lesson only survives as an executable guard that runs across ALL lanes.
# --------------------------------------------------------------------------- #

LANE_SOURCES = (
    "nodes/_otr_scifi_codex.py",
    "nodes/_otr_scifi_gemini.py",
    "nodes/_otr_scifi_sonnet.py",
)


@pytest.mark.parametrize("source", LANE_SOURCES)
def test_python_never_puts_words_in_a_characters_mouth(source):
    """No lane may hand a spoken-text field a string Python wrote.

    Live (Sonnet): the Warden's closing ruling was hardcoded as
    `DraftLineV4(text="The record holds now.", ...)` -- Python speaking for a
    character. It did not even need to: RewriteResultV4 already carried
    `vesh_resolution`, the model's own line, and the lane was throwing it away.

    A literal assigned to `text=` is dialogue no model authored. If a line must be
    spoken, a model field must supply it. Python judges; the LLM writes.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / source).read_text(encoding="utf-8"))

    spoken = {"text", "premise", "title", "claim", "spoken_claim"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in spoken:
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.strip():
                offenders.append(
                    f"{source}:{value.lineno}: {kw.arg}={value.value!r} -- Python authored this"
                )
            if isinstance(value, ast.JoinedStr):  # an f-string built into spoken text
                offenders.append(
                    f"{source}:{value.lineno}: {kw.arg}=<f-string> -- Python composed this"
                )
    assert not offenders, (
        "Python is authoring story text; a model field must supply it:\n  "
        + "\n  ".join(offenders)
    )


def test_every_seam_example_validates_against_the_schema_it_feeds():
    """A seam's worked example must be a legal instance of its own artifact.

    This is the guard that would have caught most of today's kills on its own:
      - gemini_scene_draft showed {"fact_ids": ["F01"]} while DraftLineV4 declares
        `fact_uses` and forbids extras -- so the model obeyed the seam, strict mode
        rejected it, and the repair then DELETED the model's fact attribution.
      - gemini_scene_outline asked the cast for tts_model / voice_preset, which
        CastV4 forbids.
      - the rewrite seam reintroduced fact_ids after the draft seam was fixed.

    When a seam and its schema disagree, the model is not wrong -- we are. So the
    example the model is shown must itself validate.
    """
    import json as _json
    import re

    from nodes import _otr_scifi_gemini as gemini

    repo = pathlib.Path(__file__).resolve().parent.parent
    pack = _json.loads(
        (repo / "nodes" / "story_packs" / "scifi_gemini" / "scifi_gemini_v1.json")
        .read_text(encoding="utf-8")
    )
    stages = pack["prompt_stages"]

    # The seams whose worked example IS the pass's whole artifact.
    checked = {
        "gemini_scene_draft": gemini.SceneDraftV4,
        "gemini_scene_rewrite": gemini.SceneDraftV4,
    }

    for seam, model in checked.items():
        text = stages[seam]
        # The examples are written with doubled braces for str.format.
        example = text[text.index('{{"lines"'):].replace("{{", "{").replace("}}", "}")
        # Take the first complete JSON object.
        depth, end = 0, None
        for i, ch in enumerate(example):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        assert end, f"{seam}: could not read its worked example"
        payload = _json.loads(example[:end])

        # It must validate -- extras forbidden, required fields present.
        model.model_validate(payload)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
